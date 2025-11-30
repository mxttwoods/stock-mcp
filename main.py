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
from starlette.exceptions import HTTPException
from starlette import status
from starlette.responses import JSONResponse

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    "yfinance-mcp",
    stateless_http=True,
)

# ---- Configuration ----
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))  # seconds
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))  # per minute


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
        return None
    if time.time() - item.ts > TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
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
    Returns server status and basic info.
    """
    return JSONResponse(
        {
            "status": "healthy",
            "service": "yfinance-mcp",
            "timestamp": time.time(),
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
    Get real-time quote snapshot for a stock symbol.

    Use this for quick price checks and current market state before deeper analysis.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "TSLA", "SPY"). Will be auto-converted to uppercase.

    Returns:
        dict with:
        - symbol: The ticker symbol
        - price: Current price (float or None)
        - previousClose: Previous day's closing price
        - change: Dollar change from previous close
        - changePercent: Percentage change from previous close
        - currency: Currency code (e.g., "USD")
        - marketTime: Unix timestamp of last market time
        - raw: Additional metadata (market state, exchange, quote type)

    Example:
        Input: quote("AAPL")
        Output: {"symbol": "AAPL", "price": 189.50, "previousClose": 188.00,
                 "change": 1.50, "changePercent": 0.80, "currency": "USD", ...}

    Note: Data is cached for 20 seconds. This is NOT real-time execution-grade data.
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
    Get OHLCV (Open, High, Low, Close, Volume) historical data for technical analysis.

    Use this for: trend analysis, charting, backtesting strategies, and technical indicators.

    Args:
        symbol: Stock ticker symbol (e.g., "TSLA", "NVDA")
        period: Time range. Examples:
            - Intraday: "1d", "5d"
            - Short-term: "1mo", "3mo"
            - Long-term: "1y", "5y", "max"
        interval: Data frequency. Examples:
            - Intraday: "1m", "5m", "15m", "60m"
            - Daily+: "1d", "1wk", "1mo"
            (Note: Intraday intervals limited to <60 days of data)
        auto_adjust: If True, adjusts prices for splits/dividends (recommended: True)

    Returns:
        dict with:
        - symbol: The ticker symbol
        - period, interval, auto_adjust: Your input parameters
        - rows: Array of dicts, each with Date, Open, High, Low, Close, Volume
                (max 5000 rows, newest data if larger)

    Examples:
        1. Analyze yearly trend:
           history("AAPL", period="1y", interval="1d")

        2. Intraday pattern for last 5 days:
           history("SPY", period="5d", interval="5m")

        3. Weekly data for 5 years:
           history("MSFT", period="5y", interval="1wk")

    Note: Cached for 20 seconds. Use larger intervals for longer periods to reduce data size.
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
    """
    List all available option expiration dates for a symbol.

    Use this FIRST before calling option_chain to see what dates are available.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "SPY")

    Returns:
        dict with:
        - symbol: The ticker symbol
        - expirations: Array of date strings in YYYY-MM-DD format

    Example:
        Input: options_expirations("AAPL")
        Output: {"symbol": "AAPL", "expirations": ["2025-12-20", "2026-01-17", ...]}

    Workflow:
        Step 1: Use this tool to get available dates
        Step 2: Pick a date from the expirations list
        Step 3: Call option_chain(symbol, chosen_date) for detailed strikes

    Note: Not all stocks have options. Returns empty array if no options available.
    """
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
    Get complete option chain (calls and puts) for a specific expiration date.

    Use for: covered calls, protective puts, spreads, and options strategy analysis.
    IMPORTANT: Call options_expirations(symbol) FIRST to get valid expiration dates.

    Args:
        symbol: Stock ticker symbol
        expiration: Expiration date in YYYY-MM-DD format (must be from options_expirations list)

    Returns:
        dict with:
        - symbol: The ticker symbol
        - expiration: The expiration date
        - calls: Array of call option contracts, each with:
            * strike: Strike price
            * lastPrice: Last traded price (premium)
            * bid, ask: Current bid/ask prices
            * volume: Trading volume
            * openInterest: Open interest
            * impliedVolatility: IV percentage
            * Greeks: delta, gamma, theta, vega, rho (if available)
        - puts: Array of put option contracts (same structure as calls)
        (max 15000 rows per calls/puts)

    Example:
        Input: option_chain("SPY", "2025-12-20")
        Output: {"symbol": "SPY", "expiration": "2025-12-20",
                 "calls": [{"strike": 500, "lastPrice": 12.50, "impliedVolatility": 0.18, ...}, ...],
                 "puts": [{"strike": 500, "lastPrice": 8.20, ...}, ...]}

    Strategy Tips:
        - Covered calls: Look for strikes 5-10% above current price, ~30-45 days out
        - Protective puts: Strikes 5-10% below current, check earnings first (use calendar tool)
        - High volume + open interest often means better fills

    Note: Check calendar(symbol) for earnings dates to avoid IV crush!
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
    Get key fundamental metrics snapshot for company analysis.

    Use for: value investing, company screening, comparing financial health, and valuation.

    Args:
        symbol: Stock ticker symbol

    Returns:
        dict with:
        - symbol: The ticker symbol
        - info: Dict containing:
            * shortName: Company name
            * sector, industry: Classification
            * exchange: Which exchange it trades on
            * marketCap: Market capitalization
            * trailingPE, forwardPE: Price-to-earnings ratios
            * profitMargins, grossMargins, operatingMargins: Profitability metrics (%)
            * revenueGrowth, earningsGrowth: Growth rates (%)
            * beta: Volatility vs market
            * dividendYield: Dividend yield (%)
        - raw_size: Number of total fields in yfinance info (for reference)

    Example:
        Input: fundamentals("MSFT")
        Output: {"symbol": "MSFT",
                 "info": {"shortName": "Microsoft", "sector": "Technology",
                          "trailingPE": 35.2, "marketCap": 3000000000000, ...},
                 "raw_size": 142}

    Analysis Tips:
        - Compare P/E ratios within same sector
        - Look for positive earnings growth + healthy margins
        - High beta (>1.5) = more volatile than market
        - Dividend yield good for income strategies

    Note: Some fields may be None if not available for that symbol.
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
    Get upcoming corporate events, especially earnings dates.

    Use this to: time trades around earnings, avoid IV crush on options, plan entry/exit points.
    CRITICAL for options traders: check this before selling options to avoid earnings surprises!

    Args:
        symbol: Stock ticker symbol

    Returns:
        dict with:
        - symbol: The ticker symbol
        - calendar: Array of event dicts with keys like:
            * Earnings Date: Upcoming earnings announcement
            * Ex-Dividend Date: When stock goes ex-dividend
            * Dividend Date: When dividends are paid
            (Format may vary, check actual output)

    Example:
        Input: calendar("NFLX")
        Output: {"symbol": "NFLX",
                 "calendar": [{"Earnings Date": "2025-01-20", ...}]}

    Options Trading Warning:
        - Implied Volatility (IV) typically spikes before earnings
        - IV "crush" happens after earnings when uncertainty resolves
        - Selling options right before earnings = high risk of assignment
        - Check this tool before ANY options strategy!

    Note: Events are forward-looking. May be empty if no upcoming events scheduled.
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
    Get analyst ratings and price targets from Wall Street.

    Use this to: understand Wall Street consensus, gauge sentiment, and compare targets to current price.

    Args:
        symbol: Stock ticker symbol

    Returns:
        dict with:
        - symbol: The ticker symbol
        - recommendations: Array of historical rating changes (max 100 rows)
            * Each entry has: date, firm, toGrade (Buy/Hold/Sell), fromGrade, action
        - targets: Dict with analyst price targets:
            * current: Latest consensus target
            * mean, median: Average targets
            * low, high: Range of targets

    Example:
        Input: analyst_recommendations("AMD")
        Output: {"symbol": "AMD",
                 "recommendations": [{"date": "2025-11-15", "firm": "Goldman",
                                      "toGrade": "Buy", "fromGrade": "Neutral"}, ...],
                 "targets": {"current": 185, "mean": 180, "low": 150, "high": 210}}

    Interpretation:
        - More "Buy" upgrades = bullish sentiment
        - Price below targets = potential upside (but verify fundamentals!)
        - Wide target range (high - low) = high uncertainty
        - Track changes over time to see sentiment shifts

    Note: This is historical data. Analysts can be wrong. Use as one signal among many.
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
    Get latest news articles for a stock symbol.

    Use this to: understand market-moving events, identify catalysts, perform sentiment analysis.

    Args:
        symbol: Stock ticker symbol

    Returns:
        dict with:
        - symbol: The ticker symbol
        - news: Array of news article dicts, each with:
            * title: Article headline
            * publisher: News source
            * link: URL to full article
            * providerPublishTime: Unix timestamp of publication
            * type: Article type (e.g., "STORY")
            * thumbnail: Image URL (if available)

    Example:
        Input: news("TSLA")
        Output: {"symbol": "TSLA",
                 "news": [{"title": "Tesla reports record deliveries",
                          "publisher": "Reuters",
                          "link": "https://...",
                          "providerPublishTime": 1701234567}, ...]}

    Use Cases:
        - Explain sudden price movements (check news when unusual volatility)
        - Research company before investing
        - Track ongoing stories (product launches, regulatory issues)
        - Sentiment analysis for trading decisions

    Note: News can drive short-term volatility. Combine with fundamentals for full picture.
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
    Get quarterly financial statements: Income Statement, Balance Sheet, Cash Flow.

    Use this for: deep financial analysis, understanding revenue trends, evaluating cash position and debt.

    Args:
        symbol: Stock ticker symbol

    Returns:
        dict with:
        - symbol: The ticker symbol
        - income_stmt: Quarterly income statement rows (revenue, expenses, net income, EPS, etc.)
        - balance_sheet: Quarterly balance sheet rows (assets, liabilities, equity, cash, debt, etc.)
        - cash_flow: Quarterly cash flow rows (operating CF, investing CF, financing CF, etc.)

        Each is an array of dicts with quarterly data going back several periods.

    Example:
        Input: financials("NVDA")
        Output: {"symbol": "NVDA",
                 "income_stmt": [{"Total Revenue": 18120000000, "Net Income": 9243000000, ...}, ...],
                 "balance_sheet": [{"Total Assets": 65728000000, "Total Debt": 9703000000, ...}, ...],
                 "cash_flow": [{"Operating Cash Flow": 10438000000, ...}, ...]}

    Analysis Tips:
        - Revenue trend: Growing or declining?
        - Profitability: Net income margins improving?
        - Cash position: Enough cash vs debt?
        - Cash flow: Is operating CF positive and growing?
        - Use this with fundamentals() for complete picture

    Note: Field names vary by company. Not all companies report all fields. Returns empty arrays if data unavailable.
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
    Get institutional and major holder ownership information.

    Use this to: understand ownership structure, gauge institutional confidence, identify potential squeeze setups.

    Args:
        symbol: Stock ticker symbol

    Returns:
        dict with:
        - symbol: The ticker symbol
        - major_holders: Summary stats array showing:
            * % held by insiders
            * % held by institutions
            * % float held by institutions
            * Number of institutions holding
        - institutional_holders: Top institutional holders array, each with:
            * Holder name (e.g., "Vanguard Group")
            * Shares held
            * Date reported
            * % Out (percentage of total shares)
            * Value of holdings

    Example:
        Input: holders("GME")
        Output: {"symbol": "GME",
                 "major_holders": [{"0": "8.72%", "1": "insidersPercentHeld"}, ...],
                 "institutional_holders": [{"Holder": "Vanguard Group",
                                            "Shares": 23500000, "% Out": 7.8}, ...]}

    Analysis Tips:
        - High institutional ownership (>70%) = confidence from big players
        - Low float + high short interest = potential short squeeze
        - Increasing institutional positions = bullish signal
        - Track insider ownership for alignment with shareholders

    Note: Data is typically updated quarterly. May not reflect very recent changes.
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


@mcp.tool()
@handle_errors
def dividends(symbol: str) -> dict:
    """
    Get historical dividends for a stock.

    Use this for: dividend stock analysis, yield calculations, dividend growth tracking.

    Args:
        symbol: Stock ticker symbol

    Returns:
        dict with:
        - symbol: The ticker symbol
        - dividends: Array of dividend payment records with:
            * Date: Ex-dividend date
            * Dividends: Dividend amount per share

    Example:
        Input: dividends("AAPL")
        Output: {"symbol": "AAPL",
                 "dividends": [{"Date": "2025-11-08", "Dividends": 0.25}, ...]}

    Analysis Tips:
        - Calculate dividend yield: (annual dividends / current price) * 100
        - Look for consistent dividend growth (aristocrats have 25+ years)
        - Check dividend payout ratio in fundamentals (sustainable if <60%)
        - Compare with dividendYield in fundamentals() for current rate

    Note: Returns empty if stock doesn't pay dividends. History goes back many years.
    """
    symbol = symbol.upper().strip()
    key = f"div:{symbol}"
    hit = _cache_get(key)
    if hit:
        return hit

    t = yf.Ticker(symbol)
    div = t.dividends

    rows = _df_to_rows(div, max_rows=1000) if div is not None and not div.empty else []

    out = {"symbol": symbol, "dividends": rows}
    return _cache_set(key, out)


@mcp.tool()
@handle_errors
def splits(symbol: str) -> dict:
    """
    Get historical stock splits.

    Use this for: understanding historical price adjustments, reverse split warnings (bearish).

    Args:
        symbol: Stock ticker symbol

    Returns:
        dict with:
        - symbol: The ticker symbol
        - splits: Array of split records with:
            * Date: Split date
            * Stock Splits: Split ratio (e.g., 2.0 = 2:1 split, 0.5 = 1:2 reverse split)

    Example:
        Input: splits("AAPL")
        Output: {"symbol": "AAPL",
                 "splits": [{"Date": "2020-08-31", "Stock Splits": 4.0}, ...]}

    Interpretation:
        - Forward splits (ratio > 1): Often bullish, makes shares more accessible
        - Reverse splits (ratio < 1): Often bearish, can indicate struggling company
        - Recent splits may affect options strike prices

    Note: Returns empty if no splits in history. Rare events, so usually small dataset.
    """
    symbol = symbol.upper().strip()
    key = f"splits:{symbol}"
    hit = _cache_get(key)
    if hit:
        return hit

    t = yf.Ticker(symbol)
    sp = t.splits

    rows = _df_to_rows(sp, max_rows=100) if sp is not None and not sp.empty else []

    out = {"symbol": symbol, "splits": rows}
    return _cache_set(key, out)


@mcp.tool()
@handle_errors
def actions(symbol: str) -> dict:
    """
    Get all corporate actions (dividends + splits combined).

    Use this for: complete corporate action history in one call.

    Args:
        symbol: Stock ticker symbol

    Returns:
        dict with:
        - symbol: The ticker symbol
        - actions: Array of action records with:
            * Date: Action date
            * Dividends: Dividend amount (if dividend event)
            * Stock Splits: Split ratio (if split event)

    Example:
        Input: actions("MSFT")
        Output: {"symbol": "MSFT",
                 "actions": [{"Date": "2024-11-20", "Dividends": 0.75, "Stock Splits": null},
                             {"Date": "2003-02-18", "Dividends": null, "Stock Splits": 2.0}, ...]}

    Tip: This is more efficient than calling dividends() and splits() separately.

    Note: Combines both event types. Check which fields are null to determine event type.
    """
    symbol = symbol.upper().strip()
    key = f"actions:{symbol}"
    hit = _cache_get(key)
    if hit:
        return hit

    t = yf.Ticker(symbol)
    act = t.actions

    rows = _df_to_rows(act, max_rows=1500) if act is not None and not act.empty else []

    out = {"symbol": symbol, "actions": rows}
    return _cache_set(key, out)


@mcp.tool()
@handle_errors
def earnings_history(symbol: str) -> dict:
    """
    Get historical earnings data (actual vs estimates).

    Use this for: analyzing earnings beat/miss patterns, EPS trends over time.

    Args:
        symbol: Stock ticker symbol

    Returns:
        dict with:
        - symbol: The ticker symbol
        - earnings: Array of historical earnings with quarterly data

    Example:
        Input: earnings_history("NVDA")
        Output: {"symbol": "NVDA",
                 "earnings": [{"quarter": "2024Q3", "actual": 5.20, "estimate": 4.98}, ...]}

    Analysis Tips:
        - Consistent beats = strong execution
        - Pattern of beats often leads to price momentum
        - Large misses can trigger sell-offs
        - Use with financials() for complete picture

    Note: Structure may vary. Contains quarterly EPS data going back several years.
    """
    symbol = symbol.upper().strip()
    key = f"earnings_hist:{symbol}"
    hit = _cache_get(key)
    if hit:
        return hit

    t = yf.Ticker(symbol)

    # Get earnings data - handle different possible formats
    earnings_data = []
    try:
        earnings = t.earnings_dates
        if earnings is not None and not earnings.empty:
            earnings_data = _df_to_rows(earnings, max_rows=200)
    except Exception:
        pass

    out = {"symbol": symbol, "earnings": earnings_data}
    return _cache_set(key, out)


@mcp.tool()
@handle_errors
def earnings_estimates(symbol: str) -> dict:
    """
    Get future earnings estimates from analysts.

    Use this for: understanding growth expectations, forward valuations.

    Args:
        symbol: Stock ticker symbol

    Returns:
        dict with:
        - symbol: The ticker symbol
        - estimates: Forward earnings estimates data

    Example:
        Input: earnings_estimates("TSLA")
        Output: {"symbol": "TSLA",
                 "estimates": {"current_quarter": {...}, "next_quarter": {...}, ...}}

    Analysis Tips:
        - Compare estimates to current earnings for expected growth
        - Forward P/E = price / forward EPS estimate
        - Rising estimates = bullish, falling = bearish
        - Use with analyst_recommendations() for complete picture

    Note: Format varies by source. May include quarterly and annual estimates.
    """
    symbol = symbol.upper().strip()
    key = f"earnings_est:{symbol}"
    hit = _cache_get(key)
    if hit:
        return hit

    t = yf.Ticker(symbol)

    # Try to get analyst estimates from earnings forecast
    estimates_data = {}
    try:
        # Get earnings forecast/trend data if available
        if hasattr(t, "earnings_forecasts"):
            forecasts = t.earnings_forecasts
            if forecasts is not None:
                if isinstance(forecasts, pd.DataFrame):
                    estimates_data = {"forecasts": _df_to_rows(forecasts)}
                else:
                    estimates_data = forecasts
    except Exception:
        pass

    out = {"symbol": symbol, "estimates": estimates_data}
    return _cache_set(key, out)


if __name__ == "__main__":
    import argparse
    import os

    logger.info("Starting yfinance MCP server")
    logger.info(
        f"Configuration: timeout={REQUEST_TIMEOUT}s, rate_limit={RATE_LIMIT_REQUESTS}/min, cache_ttl={TTL_SECONDS}s"
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
        port = int(os.getenv("PORT", "10000"))

        logger.info(f"Starting MCP server in HTTP mode on port {port}")
        logger.info("Health check available at /health")
        logger.info(
            "For production, run with: uvicorn main:app --host 0.0.0.0 --port 10000"
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
app = mcp.http_app(
    transport="http",
    path="/api/mcp",
)
