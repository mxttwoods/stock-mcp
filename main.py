from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import Any, Optional
from functools import wraps

import yfinance as yf
import pandas as pd

from fastmcp import FastMCP, Context
from starlette.requests import Request
from starlette.exceptions import HTTPException
from starlette import status
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

mcp = FastMCP("yfinance-mcp", stateless_http=True)

# ---- Configuration ----
CLIENT_ID = os.getenv("MCP_CLIENT_ID", "stock-mcp-client")
CLIENT_SECRET = os.getenv("MCP_CLIENT_SECRET", "super-secret-key")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))  # seconds
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))  # per minute

# ---- Metrics ----
metrics = {"requests": 0, "errors": 0, "cache_hits": 0, "cache_misses": 0}


async def verify_auth(request: Request):
    """
    Verify client credentials via headers.
    Raises HTTPException(401) if credentials are missing or invalid.
    """
    client_id = request.headers.get("X-Client-Id")
    client_secret = request.headers.get("X-Client-Secret")

    if client_id != CLIENT_ID or client_secret != CLIENT_SECRET:
        logger.warning(f"Unauthorized access attempt from {request.client}")
        metrics["errors"] += 1
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication credentials",
        )

    metrics["requests"] += 1


# Add authentication dependency
mcp.dependencies.append(verify_auth)


# ---- Error handling decorator ----
def handle_errors(func):
    """Decorator to handle errors and return proper responses."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except HTTPException:
            raise  # Re-raise HTTP exceptions
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            metrics["errors"] += 1
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Internal error: {str(e)}",
            )

    return wrapper


# ---- TTL cache to reduce rate-limit pain ----
@dataclass
class CacheItem:
    ts: float
    value: Any


_CACHE: dict[str, CacheItem] = {}
TTL_SECONDS = 20


def _cache_get(key: str) -> Optional[Any]:
    item = _CACHE.get(key)
    if not item:
        metrics["cache_misses"] += 1
        return None
    if time.time() - item.ts > TTL_SECONDS:
        _CACHE.pop(key, None)
        metrics["cache_misses"] += 1
        return None
    metrics["cache_hits"] += 1
    logger.debug(f"Cache hit for key: {key}")
    return item.value


def _cache_set(key: str, value: Any) -> Any:
    _CACHE[key] = CacheItem(time.time(), value)
    return value


def _df_to_rows(df: pd.DataFrame, max_rows: int = 5000) -> list[dict]:
    # Make JSON-friendly rows; cap size to avoid huge tool outputs
    if df is None or df.empty:
        return []
    if len(df) > max_rows:
        df = df.tail(max_rows)

    df = df.copy()
    df.reset_index(inplace=True)

    # ensure timestamps become strings
    for col in df.columns:
        if (
            "date" in str(col).lower()
            or "time" in str(col).lower()
            or str(df[col].dtype).startswith("datetime")
        ):
            df[col] = df[col].astype(str)

    # Replace NaN with None
    df = df.where(pd.notnull(df), None)
    return df.to_dict(orient="records")


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    """
    Health check endpoint for monitoring (HTTP route, not MCP tool).
    Returns server status and basic metrics.
    """
    return JSONResponse(
        {
            "status": "healthy",
            "service": "yfinance-mcp",
            "timestamp": time.time(),
            "metrics": metrics.copy(),
            "cache_size": len(_CACHE),
            "config": {
                "timeout": REQUEST_TIMEOUT,
                "rate_limit": RATE_LIMIT_REQUESTS,
                "cache_ttl": TTL_SECONDS,
            },
        }
    )


@mcp.tool()
@handle_errors
def quote(symbol: str, context: Context) -> dict:
    """
    Best-effort quote snapshot for a single symbol (not execution-grade).
    Uses fast_info when available; falls back to .info pieces.
    """
    symbol = symbol.upper().strip()
    key = f"quote:{symbol}"
    hit = _cache_get(key)
    if hit:
        return hit

    t = yf.Ticker(symbol)

    fast = {}
    try:
        fast = dict(getattr(t, "fast_info", {}) or {})
    except Exception:
        fast = {}

    info = {}
    try:
        info = t.get_info()  # can be slower / occasionally blocked
    except Exception:
        info = {}

    price = fast.get("last_price") or info.get("regularMarketPrice")
    prev = fast.get("previous_close") or info.get("regularMarketPreviousClose")

    change = None
    change_pct = None
    if price is not None and prev not in (None, 0):
        try:
            change = float(price) - float(prev)
            change_pct = (change / float(prev)) * 100.0
        except Exception:
            pass

    out = {
        "symbol": symbol,
        "price": float(price) if price is not None else None,
        "previousClose": float(prev) if prev is not None else None,
        "change": change,
        "changePercent": change_pct,
        "currency": info.get("currency") or fast.get("currency"),
        "marketTime": info.get("regularMarketTime"),
        "raw": {
            "fast_info": fast,
            "info_subset": {
                "marketState": info.get("marketState"),
                "exchange": info.get("exchange"),
                "quoteType": info.get("quoteType"),
            },
        },
    }
    return _cache_set(key, out)


@mcp.tool()
@handle_errors
def history(
    symbol: str,
    period: str = "1mo",
    interval: str = "1d",
    auto_adjust: bool = True,
) -> dict:
    """
    OHLCV history. period examples: 5d,1mo,3mo,6mo,1y,5y,max
    interval examples: 1m,5m,15m,60m,1d,1wk,1mo
    """
    symbol = symbol.upper().strip()
    key = f"hist:{symbol}:{period}:{interval}:{auto_adjust}"
    hit = _cache_get(key)
    if hit:
        return hit

    t = yf.Ticker(symbol)
    df = t.history(period=period, interval=interval, auto_adjust=auto_adjust)

    if df is None or df.empty:
        return {"symbol": symbol, "period": period, "interval": interval, "rows": []}

    out = {
        "symbol": symbol,
        "period": period,
        "interval": interval,
        "auto_adjust": auto_adjust,
        "rows": _df_to_rows(df, max_rows=5000),
    }
    return _cache_set(key, out)


@mcp.tool()
@handle_errors
def options_expirations(symbol: str) -> dict:
    """List available option expiration dates for a symbol."""
    symbol = symbol.upper().strip()
    key = f"opts_exp:{symbol}"
    hit = _cache_get(key)
    if hit:
        return hit

    t = yf.Ticker(symbol)
    exps = list(getattr(t, "options", []) or [])
    out = {"symbol": symbol, "expirations": exps}
    return _cache_set(key, out)


@mcp.tool()
@handle_errors
def option_chain(symbol: str, expiration: str) -> dict:
    """
    Get option chain (calls & puts) for a specific expiration (YYYY-MM-DD).
    """
    symbol = symbol.upper().strip()
    key = f"opts_chain:{symbol}:{expiration}"
    hit = _cache_get(key)
    if hit:
        return hit

    t = yf.Ticker(symbol)
    chain = t.option_chain(expiration)

    out = {
        "symbol": symbol,
        "expiration": expiration,
        "calls": _df_to_rows(chain.calls, max_rows=15000),
        "puts": _df_to_rows(chain.puts, max_rows=15000),
    }
    return _cache_set(key, out)


@mcp.tool()
@handle_errors
def fundamentals(symbol: str) -> dict:
    """
    Basic fundamentals snapshot (best-effort).
    """
    symbol = symbol.upper().strip()
    key = f"fund:{symbol}"
    hit = _cache_get(key)
    if hit:
        return hit

    t = yf.Ticker(symbol)

    info = {}
    try:
        info = t.get_info()
    except Exception:
        info = {}

    # Keep it small & useful
    keep = [
        "shortName",
        "sector",
        "industry",
        "exchange",
        "marketCap",
        "trailingPE",
        "forwardPE",
        "profitMargins",
        "grossMargins",
        "operatingMargins",
        "revenueGrowth",
        "earningsGrowth",
        "beta",
        "dividendYield",
    ]
    out = {
        "symbol": symbol,
        "info": {k: info.get(k) for k in keep},
        "raw_size": len(info),
    }
    return _cache_set(key, out)


@mcp.tool()
@handle_errors
def calendar(symbol: str) -> dict:
    """
    Get upcoming events (earnings, etc.) for a symbol.
    """
    symbol = symbol.upper().strip()
    key = f"cal:{symbol}"
    hit = _cache_get(key)
    if hit:
        return hit

    t = yf.Ticker(symbol)
    cal = t.calendar

    # Handle different return types (DataFrame or dict)
    if isinstance(cal, pd.DataFrame):
        rows = _df_to_rows(cal)
    elif isinstance(cal, dict):
        rows = [{"key": k, "value": str(v)} for k, v in cal.items()]
    else:
        rows = []

    out = {"symbol": symbol, "calendar": rows}
    return _cache_set(key, out)


@mcp.tool()
@handle_errors
def analyst_recommendations(symbol: str) -> dict:
    """
    Get analyst recommendations and price targets.
    """
    symbol = symbol.upper().strip()
    key = f"rec:{symbol}"
    hit = _cache_get(key)
    if hit:
        return hit

    t = yf.Ticker(symbol)

    recs = pd.DataFrame()
    try:
        recs = t.recommendations
    except Exception:
        pass

    targets = {}
    try:
        targets = t.analyst_price_targets
    except Exception:
        pass

    out = {
        "symbol": symbol,
        "recommendations": _df_to_rows(recs, max_rows=100) if recs is not None else [],
        "targets": targets,
    }
    return _cache_set(key, out)


@mcp.tool()
@handle_errors
def news(symbol: str) -> dict:
    """
    Get latest news for a symbol.
    """
    symbol = symbol.upper().strip()
    key = f"news:{symbol}"
    hit = _cache_get(key)
    if hit:
        return hit

    t = yf.Ticker(symbol)
    news_list = []
    try:
        news_list = t.news
    except Exception:
        pass

    out = {"symbol": symbol, "news": news_list}
    return _cache_set(key, out)


@mcp.tool()
@handle_errors
def financials(symbol: str) -> dict:
    """
    Get quarterly financials: Income Statement, Balance Sheet, Cash Flow.
    """
    symbol = symbol.upper().strip()
    key = f"fin:{symbol}"
    hit = _cache_get(key)
    if hit:
        return hit

    t = yf.Ticker(symbol)

    def get_df_safe(attr):
        try:
            df = getattr(t, attr)
            return _df_to_rows(df) if df is not None else []
        except Exception:
            return []

    out = {
        "symbol": symbol,
        "income_stmt": get_df_safe("quarterly_income_stmt"),
        "balance_sheet": get_df_safe("quarterly_balance_sheet"),
        "cash_flow": get_df_safe("quarterly_cashflow"),
    }
    return _cache_set(key, out)


@mcp.tool()
@handle_errors
def holders(symbol: str) -> dict:
    """
    Get major and institutional holders.
    """
    symbol = symbol.upper().strip()
    key = f"hold:{symbol}"
    hit = _cache_get(key)
    if hit:
        return hit

    t = yf.Ticker(symbol)

    def get_df_safe(attr):
        try:
            df = getattr(t, attr)
            return _df_to_rows(df) if df is not None else []
        except Exception:
            return []

    out = {
        "symbol": symbol,
        "major_holders": get_df_safe("major_holders"),
        "institutional_holders": get_df_safe("institutional_holders"),
    }
    return _cache_set(key, out)


@mcp.tool
def greet(name: str) -> str:
    return f"Hello, {name}!"


if __name__ == "__main__":
    import argparse
    import os

    # Environment validation
    required_env_vars = {"MCP_CLIENT_ID": CLIENT_ID, "MCP_CLIENT_SECRET": CLIENT_SECRET}

    logger.info("Starting yfinance MCP server")
    logger.info(
        f"Configuration: timeout={REQUEST_TIMEOUT}s, rate_limit={RATE_LIMIT_REQUESTS}/min, cache_ttl={TTL_SECONDS}s"
    )

    # Warn about default credentials
    if CLIENT_ID == "stock-mcp-client" or CLIENT_SECRET == "super-secret-key":
        logger.warning(
            "⚠️  Using default credentials! Set MCP_CLIENT_ID and MCP_CLIENT_SECRET environment variables for production."
        )

    parser = argparse.ArgumentParser(description="yfinance MCP Server")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run in local stdio mode instead of HTTP mode",
    )
    args = parser.parse_args()

    if args.local:
        # Local development mode using stdio
        logger.info("Starting MCP server in stdio mode")
        mcp.run()
    else:
        # HTTP mode - for production use uvicorn
        port = int(os.getenv("PORT", "8080"))

        logger.info(f"Starting MCP server in HTTP mode on port {port}")
        logger.info("Health check available at /health")
        logger.info(
            "For production, run with: uvicorn main:app --host 0.0.0.0 --port 8080"
        )

        # Use http transport - mcp.run() doesn't accept host/port
        # PORT env var is used by FastMCP internally
        mcp.run(transport="http", host="0.0.0.0", port=port)

# middleware = [
#     Middleware(
#         CORSMiddleware,
#         allow_origins=["*"],  # Allow all origins; use specific origins for security
#         allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
#         allow_headers=[
#             "mcp-protocol-version",
#             "mcp-session-id",
#             "Authorization",
#             "Content-Type",
#         ],
#         expose_headers=["mcp-session-id"],
#     )
# ]
# Export ASGI app for production deployment with uvicorn
# Note: CORS middleware would need to be added via a reverse proxy (nginx, caddy)
# or by wrapping the app if needed for browser-based clients
# app = mcp.http_app(
#     transport="http",
#     path="/api/mcp",
# )
