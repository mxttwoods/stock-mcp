from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import yfinance as yf
import pandas as pd

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.exceptions import HTTPException
from starlette import status

mcp = FastMCP("yfinance-mcp")

# ---- Authentication ----
# Use environment variables for credentials (with defaults for local dev)
CLIENT_ID = os.getenv("MCP_CLIENT_ID", "stock-mcp-client")
CLIENT_SECRET = os.getenv("MCP_CLIENT_SECRET", "super-secret-key")


async def verify_auth(request: Request):
    """
    Verify client credentials via headers.
    Raises HTTPException(401) if credentials are missing or invalid.
    """
    client_id = request.headers.get("X-Client-Id")
    client_secret = request.headers.get("X-Client-Secret")

    if client_id != CLIENT_ID or client_secret != CLIENT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication credentials",
        )


# Add authentication dependency
mcp.dependencies.append(verify_auth)


# ---- tiny TTL cache to reduce rate-limit pain ----
@dataclass
class CacheItem:
    ts: float
    value: Any


_CACHE: dict[str, CacheItem] = {}
TTL_SECONDS = 20


def _cache_get(key: str) -> Optional[Any]:
    item = _CACHE.get(key)
    if not item:
        return None
    if time.time() - item.ts > TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
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


@mcp.tool()
def quote(symbol: str) -> dict:
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


if __name__ == "__main__":
    # Check if running in web mode (Render/cloud) or stdio mode (local)
    import sys

    # If PORT env var is set, run in SSE mode for web deployment
    if os.getenv("PORT"):
        port = int(os.getenv("PORT", "8000"))
        print(f"Starting MCP server in SSE mode on port {port}", file=sys.stderr)
        mcp.run(transport="sse", port=port)
    else:
        # Default to stdio for local development
        print("Starting MCP server in stdio mode", file=sys.stderr)
        mcp.run()
