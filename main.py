from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fastmcp import Context, FastMCP
from starlette import status
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from yfinance import EquityQuery, Industry, Sector, screen

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "DEBUG"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    "yfinance-mcp",
    stateless_http=True,
)

# ---- Configuration ----
# Cache TTL for reducing API rate limits


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
        "debtToEquity",
        "currentRatio",
        "returnOnEquity",
        "totalCash",
        "totalDebt",
        "quickRatio",
        # Additional useful metrics
        "pegRatio",
        "priceToBook",
        "enterpriseValue",
        "freeCashflow",
        "operatingCashflow",
        "fiftyTwoWeekHigh",
        "fiftyTwoWeekLow",
        "revenue",
        "revenuePerShare",
        "shortRatio",
        "sharesShort",
        "sharesOutstanding",
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

    if div is not None and not div.empty:
        # Convert Series to DataFrame properly
        div_df = div.reset_index()
        rows = _df_to_rows(div_df, max_rows=1000)
    else:
        rows = []

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

    if sp is not None and not sp.empty:
        sp_df = sp.reset_index()
        rows = _df_to_rows(sp_df, max_rows=100)
    else:
        rows = []

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

    if act is not None and not act.empty:
        act_df = act.reset_index()
        rows = _df_to_rows(act_df, max_rows=1500)
    else:
        rows = []

    out = {"symbol": symbol, "actions": rows}
    return _cache_set(key, out)


@mcp.tool()
@handle_errors
def sharpe_ratio(
    symbol: str,
    period: str = "1y",
    risk_free_rate: float = 0.04,
) -> dict:
    """
    Calculate the Sharpe ratio for a stock symbol over a specified period.

    The Sharpe ratio measures risk-adjusted return: higher is better.
    It shows how much excess return you receive for the extra volatility endured.

    Use this for: comparing risk-adjusted performance across stocks, evaluating if returns justify volatility.

    Args:
        symbol: Stock ticker symbol
        period: Time period for calculation. Examples:
            - "1mo", "3mo", "6mo": Short-term analysis
            - "1y", "2y", "3y", "5y": Long-term analysis
            - "max": All available history
            Default: "1y"
        risk_free_rate: Annual risk-free rate (e.g., 10-year Treasury yield).
            Default: 0.04 (4%)
            Common values: 0.03-0.05 depending on current rates

    Returns:
        dict with:
        - symbol: The ticker symbol
        - period: Time period used
        - risk_free_rate: Risk-free rate used (annual)
        - sharpe_ratio: The calculated Sharpe ratio
        - annualized_return: Annual return (%)
        - annualized_volatility: Annual volatility/standard deviation (%)
        - total_return: Total return over period (%)
        - days_analyzed: Number of trading days in analysis

    Example:
        Input: sharpe_ratio("AAPL", period="1y", risk_free_rate=0.04)
        Output: {"symbol": "AAPL",
                 "period": "1y",
                 "sharpe_ratio": 1.85,
                 "annualized_return": 28.5,
                 "annualized_volatility": 13.2,
                 "total_return": 28.5,
                 "days_analyzed": 252}

    Interpretation:
        - Sharpe > 1.0: Good risk-adjusted returns
        - Sharpe > 2.0: Very good risk-adjusted returns
        - Sharpe > 3.0: Excellent risk-adjusted returns
        - Sharpe < 1.0: Returns may not justify the risk
        - Negative Sharpe: You're losing money vs risk-free rate

    Analysis Tips:
        - Compare Sharpe ratios across stocks in same sector
        - Higher Sharpe = better risk-adjusted performance
        - Use consistent time periods and risk-free rates when comparing
        - Consider multiple time periods (1y, 3y, 5y) for full picture
        - Combine with fundamentals() and analyst_recommendations()

    Note: Data is cached for 20 seconds. Based on daily close prices.
    """
    symbol = symbol.upper().strip()
    key = f"sharpe:{symbol}:{period}:{risk_free_rate}"
    hit = _cache_get(key)
    if hit:
        return hit

    # Fetch historical data
    t = yf.Ticker(symbol)
    df = t.history(period=period, interval="1d", auto_adjust=True)

    if df is None or df.empty or len(df) < 2:
        return {
            "symbol": symbol,
            "period": period,
            "risk_free_rate": risk_free_rate,
            "sharpe_ratio": None,
            "annualized_return": None,
            "annualized_volatility": None,
            "total_return": None,
            "days_analyzed": 0,
            "error": "Insufficient data to calculate Sharpe ratio",
        }

    # Calculate daily returns
    daily_returns = df["Close"].pct_change().dropna()

    if len(daily_returns) < 2:
        return {
            "symbol": symbol,
            "period": period,
            "risk_free_rate": risk_free_rate,
            "sharpe_ratio": None,
            "annualized_return": None,
            "annualized_volatility": None,
            "total_return": None,
            "days_analyzed": len(daily_returns),
            "error": "Insufficient returns data",
        }

    # Calculate metrics
    mean_daily_return = float(np.mean(daily_returns))
    std_daily_return = float(np.std(daily_returns, ddof=1))

    # Annualize (assuming 252 trading days per year)
    trading_days_per_year = 252
    annualized_return = mean_daily_return * trading_days_per_year
    annualized_volatility = std_daily_return * np.sqrt(trading_days_per_year)

    # Calculate Sharpe ratio
    if annualized_volatility == 0:
        sharpe = None
    else:
        excess_return = annualized_return - risk_free_rate
        sharpe = float(excess_return / annualized_volatility)

    # Calculate total return over period
    total_return = float((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1))

    out = {
        "symbol": symbol,
        "period": period,
        "risk_free_rate": risk_free_rate,
        "sharpe_ratio": round(sharpe, 3) if sharpe is not None else None,
        "annualized_return": round(annualized_return * 100, 2),  # Convert to %
        "annualized_volatility": round(annualized_volatility * 100, 2),  # Convert to %
        "total_return": round(total_return * 100, 2),  # Convert to %
        "days_analyzed": len(daily_returns),
    }
    return _cache_set(key, out)


@mcp.tool()
@handle_errors
def beta(
    symbol: str,
    benchmark: str = "SPY",
    period: str = "3y",
) -> dict:
    """
    Calculate beta (systematic risk) for a stock vs a benchmark.

    Beta measures how much a stock moves relative to the market.
    Use this to understand a stock's sensitivity to market movements.

    Args:
        symbol: Stock ticker symbol
        benchmark: Benchmark ticker (default: "SPY" for S&P 500)
        period: Time period for calculation
            - "1y", "2y", "3y", "5y": Long-term analysis
            Default: "3y" (industry standard)

    Returns:
        dict with:
        - symbol: The ticker symbol
        - benchmark: Benchmark ticker used
        - period: Time period
        - beta: Beta coefficient
        - correlation: Correlation with benchmark
        - r_squared: R² (coefficient of determination)
        - alpha: Annualized alpha (excess return)
        - symbol_volatility: Stock's annualized volatility (%)
        - benchmark_volatility: Benchmark's annualized volatility (%)
        - days_analyzed: Number of trading days

    Example:
        Input: beta("AAPL", benchmark="SPY", period="3y")
        Output: {"symbol": "AAPL", "benchmark": "SPY",
                 "beta": 1.15, "correlation": 0.78,
                 "alpha": 5.2, ...}

    Interpretation:
        - Beta = 1.0: Moves with market
        - Beta > 1.0: More volatile than market (amplifies moves)
        - Beta < 1.0: Less volatile than market (dampens moves)
        - Beta < 0: Moves opposite to market (rare)
        - High R²: Beta is reliable predictor
        - Positive alpha: Outperforming benchmark

    Note: 3-year period is standard for beta calculation. Data cached for 20 seconds.
    """
    symbol = symbol.upper().strip()
    benchmark = benchmark.upper().strip()
    key = f"beta:{symbol}:{benchmark}:{period}"
    hit = _cache_get(key)
    if hit:
        return hit

    # Download data for both
    ticker_data = yf.download([symbol, benchmark], period=period, progress=False)

    if ticker_data.empty or "Close" not in ticker_data:
        return {
            "symbol": symbol,
            "benchmark": benchmark,
            "period": period,
            "error": "Insufficient data to calculate beta",
        }

    # Extract close prices
    try:
        if isinstance(ticker_data.columns, pd.MultiIndex):
            stock_prices = ticker_data["Close"][symbol]
            bench_prices = ticker_data["Close"][benchmark]
        else:
            stock_prices = ticker_data["Close"]
            bench_prices = ticker_data["Close"]
    except Exception:
        return {
            "symbol": symbol,
            "benchmark": benchmark,
            "period": period,
            "error": "Error extracting price data",
        }

    # Calculate returns
    stock_returns = stock_prices.pct_change().dropna()
    bench_returns = bench_prices.pct_change().dropna()

    # Align data
    combined = pd.DataFrame({"stock": stock_returns, "bench": bench_returns}).dropna()

    if len(combined) < 20:
        return {
            "symbol": symbol,
            "benchmark": benchmark,
            "period": period,
            "error": "Insufficient aligned data points",
        }

    # Calculate beta using covariance
    covariance = float(np.cov(combined["stock"], combined["bench"])[0][1])
    bench_variance = float(np.var(combined["bench"], ddof=1))

    if bench_variance == 0:
        beta_val = None
    else:
        beta_val = covariance / bench_variance

    # Calculate correlation and R²
    correlation = float(np.corrcoef(combined["stock"], combined["bench"])[0][1])
    r_squared = correlation**2

    # Calculate alpha (annualized)
    stock_annual_return = float(np.mean(combined["stock"]) * 252)
    bench_annual_return = float(np.mean(combined["bench"]) * 252)
    alpha = stock_annual_return - (beta_val * bench_annual_return if beta_val else 0)

    # Calculate volatilities
    stock_vol = float(np.std(combined["stock"], ddof=1) * np.sqrt(252))
    bench_vol = float(np.std(combined["bench"], ddof=1) * np.sqrt(252))

    out = {
        "symbol": symbol,
        "benchmark": benchmark,
        "period": period,
        "beta": round(beta_val, 3) if beta_val is not None else None,
        "correlation": round(correlation, 3),
        "r_squared": round(r_squared, 3),
        "alpha": round(alpha * 100, 2),  # Convert to %
        "symbol_volatility": round(stock_vol * 100, 2),  # Convert to %
        "benchmark_volatility": round(bench_vol * 100, 2),  # Convert to %
        "days_analyzed": len(combined),
    }
    return _cache_set(key, out)


@mcp.tool()
@handle_errors
def max_drawdown(
    symbol: str,
    period: str = "1y",
) -> dict:
    """
    Calculate maximum drawdown - the largest peak-to-trough decline.

    Max drawdown measures the worst loss from a peak. Key risk metric for understanding
    downside risk and portfolio resilience.

    Use this for: risk assessment, comparing downside protection, stress testing.

    Args:
        symbol: Stock ticker symbol
        period: Time period. Examples:
            - "6mo", "1y": Recent performance
            - "3y", "5y", "max": Long-term analysis
            Default: "1y"

    Returns:
        dict with:
        - symbol: The ticker symbol
        - period: Time period
        - max_drawdown: Maximum drawdown percentage (negative)
        - max_drawdown_duration: Days from peak to trough
        - recovery_duration: Days from trough to recovery (None if not recovered)
        - peak_date: Date of peak before max drawdown
        - trough_date: Date of trough
        - recovery_date: Date back to peak (None if not recovered)
        - current_drawdown: Current drawdown from all-time high (negative)
        - is_recovered: Boolean - has it recovered from max drawdown

    Example:
        Input: max_drawdown("AAPL", period="1y")
        Output: {"symbol": "AAPL", "max_drawdown": -15.3,
                 "current_drawdown": -5.2, "is_recovered": true, ...}

    Interpretation:
        - Lower drawdown = better downside protection
        - Longer recovery = worse risk characteristics
        - Compare across stocks for risk comparison
        - Max drawdown < -20%: High volatility stock
        - Max drawdown > -10%: Lower volatility, defensive

    Analysis Tips:
        - Compare with volatility - low volatility + low drawdown = defensive
        - Check recovery time - faster recovery = resilient
        - Use with beta() and sharpe_ratio() for complete risk picture

    Note: Data cached for 20 seconds. Based on daily close prices.
    """
    symbol = symbol.upper().strip()
    key = f"maxdd:{symbol}:{period}"
    hit = _cache_get(key)
    if hit:
        return hit

    t = yf.Ticker(symbol)
    df = t.history(period=period, interval="1d", auto_adjust=True)

    if df is None or df.empty or len(df) < 2:
        return {
            "symbol": symbol,
            "period": period,
            "error": "Insufficient data to calculate drawdown",
        }

    # Calculate cumulative returns and running maximum
    prices = df["Close"]
    running_max = prices.expanding().max()
    drawdown = (prices - running_max) / running_max

    # Find maximum drawdown
    max_dd = float(drawdown.min())
    max_dd_date = drawdown.idxmin()

    # Find the peak before max drawdown
    peak_date = running_max[:max_dd_date].idxmax()
    peak_price = float(running_max[peak_date])

    # Calculate drawdown duration
    dd_duration = (max_dd_date - peak_date).days

    # Find recovery date (if recovered)
    recovery_date = None
    recovery_duration = None
    is_recovered = False

    future_prices = prices[max_dd_date:]
    recovered_mask = future_prices >= peak_price

    if recovered_mask.any():
        recovery_date = future_prices[recovered_mask].index[0]
        recovery_duration = (recovery_date - max_dd_date).days
        is_recovered = True

    # Current drawdown
    current_price = float(prices.iloc[-1])
    all_time_high = float(prices.max())
    current_dd = (
        ((current_price - all_time_high) / all_time_high) if all_time_high > 0 else 0
    )

    out = {
        "symbol": symbol,
        "period": period,
        "max_drawdown": round(max_dd * 100, 2),  # Convert to %
        "max_drawdown_duration": dd_duration,
        "recovery_duration": recovery_duration,
        "peak_date": str(peak_date.date()) if peak_date is not pd.NaT else None,
        "trough_date": str(max_dd_date.date()) if max_dd_date is not pd.NaT else None,
        "recovery_date": str(recovery_date.date())
        if recovery_date is not None and recovery_date is not pd.NaT
        else None,
        "current_drawdown": round(current_dd * 100, 2),  # Convert to %
        "is_recovered": is_recovered,
    }
    return _cache_set(key, out)


# Portfolio Analysis Tools


@mcp.tool()
@handle_errors
def correlation(
    symbols: list[str],
    period: str = "1y",
) -> dict:
    """
    Calculate correlation matrix between multiple stocks.

    Correlation shows how stocks move together. Essential for portfolio diversification.

    Use this for: portfolio construction, diversification analysis, finding uncorrelated assets.

    Args:
        symbols: List of stock ticker symbols (2-20 symbols recommended)
        period: Time period for calculation
            - "6mo", "1y": Recent correlations
            - "3y", "5y": Long-term relationships
            Default: "1y"

    Returns:
        dict with:
        - symbols: List of ticker symbols
        - period: Time period used
        - correlation_matrix: Dict of dicts with pairwise correlations
        - mean_correlation: Average correlation across all pairs
        - min_correlation: Minimum pairwise correlation
        - max_correlation: Maximum pairwise correlation (excluding self)
        - days_analyzed: Number of trading days

    Example:
        Input: correlation(["AAPL", "MSFT", "GOOGL"], period="1y")
        Output: {"symbols": ["AAPL", "MSFT", "GOOGL"],
                 "correlation_matrix": {
                     "AAPL": {"AAPL": 1.0, "MSFT": 0.75, "GOOGL": 0.68},
                     "MSFT": {"AAPL": 0.75, "MSFT": 1.0, "GOOGL": 0.82},
                     ...
                 },
                 "mean_correlation": 0.75, ...}

    Interpretation:
        - Correlation 1.0: Perfect positive correlation
        - Correlation 0.0: No correlation (ideal for diversification)
        - Correlation -1.0: Perfect negative correlation (rare)
        - Mean correlation > 0.7: Highly correlated (poor diversification)
        - Mean correlation < 0.3: Well diversified

    Analysis Tips:
        - Look for low/negative correlations for diversification
        - High correlation = portfolio won't be protected in downturn
        - Correlations change over time - check multiple periods
        - Use with portfolio_optimization() to build efficient portfolios

    Note: Data cached for 20 seconds. Requires at least 2 symbols.
    """
    if len(symbols) < 2:
        return {"error": "Need at least 2 symbols for correlation analysis"}

    symbols = [s.upper().strip() for s in symbols]
    key = f"corr:{':'.join(sorted(symbols))}:{period}"
    hit = _cache_get(key)
    if hit:
        return hit

    # Download data
    data = yf.download(symbols, period=period, progress=False)

    if data.empty or "Close" not in data:
        return {
            "symbols": symbols,
            "period": period,
            "error": "Insufficient data for correlation analysis",
        }

    # Extract close prices
    try:
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            # Single ticker returns non-MultiIndex
            prices = pd.DataFrame({symbols[0]: data["Close"]})
    except Exception:
        return {
            "symbols": symbols,
            "period": period,
            "error": "Error extracting price data",
        }

    # Calculate returns
    returns = prices.pct_change().dropna()

    if len(returns) < 20:
        return {
            "symbols": symbols,
            "period": period,
            "error": "Insufficient data points for correlation",
        }

    # Calculate correlation matrix
    corr_df = returns.corr()

    # Convert to nested dict format
    corr_matrix = {}
    for sym1 in corr_df.index:
        corr_matrix[sym1] = {}
        for sym2 in corr_df.columns:
            corr_matrix[sym1][sym2] = round(float(corr_df.loc[sym1, sym2]), 3)

    # Calculate statistics (excluding diagonal)
    mask = np.ones(corr_df.shape, dtype=bool)
    np.fill_diagonal(mask, False)
    off_diagonal = corr_df.values[mask]

    out = {
        "symbols": symbols,
        "period": period,
        "correlation_matrix": corr_matrix,
        "mean_correlation": round(float(np.mean(off_diagonal)), 3),
        "min_correlation": round(float(np.min(off_diagonal)), 3),
        "max_correlation": round(float(np.max(off_diagonal)), 3),
        "days_analyzed": len(returns),
    }
    return _cache_set(key, out)


@mcp.tool()
@handle_errors
def portfolio_metrics(
    symbols: list[str],
    weights: list[float],
    period: str = "1y",
    risk_free_rate: float = 0.04,
) -> dict:
    """
    Calculate comprehensive portfolio statistics for given holdings and weights.

    Analyze a specific portfolio allocation with detailed risk/return metrics.

    Use this for: evaluating portfolio allocations, comparing strategies, risk analysis.

    Args:
        symbols: List of stock ticker symbols
        weights: Portfolio weights (must sum to ~1.0)
            Example: [0.4, 0.3, 0.3] for 40%, 30%, 30% allocation
        period: Time period for calculation
            - "6mo", "1y": Recent performance
            - "3y", "5y": Long-term analysis
            Default: "1y"
        risk_free_rate: Annual risk-free rate (default: 0.04 = 4%)

    Returns:
        dict with:
        - symbols: Stock tickers
        - weights: Portfolio weights
        - period: Time period
        - expected_return: Portfolio expected annual return (%)
        - volatility: Portfolio annualized volatility (%)
        - sharpe_ratio: Portfolio Sharpe ratio
        - beta_vs_spy: Portfolio beta vs S&P 500
        - var_95: Value at Risk at 95% confidence (%)
        - max_drawdown: Portfolio maximum drawdown (%)
        - diversification_ratio: Weighted avg vol / portfolio vol
        - total_return: Total return over period (%)

    Example:
        Input: portfolio_metrics(
            symbols=["AAPL", "MSFT", "GOOGL"],
            weights=[0.4, 0.3, 0.3],
            period="1y"
        )
        Output: {"expected_return": 18.5, "volatility": 22.3,
                 "sharpe_ratio": 0.82, "diversification_ratio": 1.15, ...}

    Interpretation:
        - Higher Sharpe = better risk-adjusted returns
        - VaR_95 = expected max loss in 95% of scenarios
        - Diversification ratio > 1 = diversification benefit
        - Compare metrics across different weight allocations

    Analysis Tips:
        - Use with efficient_frontier() to see if allocation is optimal
        - Compare Sharpe to individual assets
        - Check beta to understand market sensitivity
        - Monitor max_drawdown for downside risk

    Note: Weights must sum to approximately 1.0. Data cached for 20 seconds.
    """
    if len(symbols) != len(weights):
        return {"error": "Number of symbols must match number of weights"}

    if not (0.95 <= sum(weights) <= 1.05):
        return {"error": f"Weights must sum to ~1.0, got {sum(weights)}"}

    symbols = [s.upper().strip() for s in symbols]
    weights = np.array(weights)

    key = f"portmetrics:{':'.join(symbols)}:{':'.join(map(str, weights))}:{period}:{risk_free_rate}"
    hit = _cache_get(key)
    if hit:
        return hit

    # Download data
    data = yf.download(symbols, period=period, progress=False)

    if data.empty or "Close" not in data:
        return {
            "symbols": symbols,
            "weights": weights.tolist(),
            "period": period,
            "error": "Insufficient data",
        }

    # Extract prices and calculate returns
    try:
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            prices = pd.DataFrame({symbols[0]: data["Close"]})

        returns = prices.pct_change().dropna()

        if len(returns) < 20:
            return {
                "symbols": symbols,
                "weights": weights.tolist(),
                "period": period,
                "error": "Insufficient return data",
            }

        # Portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)

        # Expected return (annualized)
        expected_return = float(np.mean(portfolio_returns) * 252)

        # Volatility (annualized)
        volatility = float(np.std(portfolio_returns, ddof=1) * np.sqrt(252))

        # Sharpe ratio
        sharpe = (
            (expected_return - risk_free_rate) / volatility if volatility > 0 else None
        )

        # Beta vs SPY
        spy_data = yf.download("SPY", period=period, progress=False)
        beta_vs_spy = None

        if not spy_data.empty and "Close" in spy_data:
            spy_returns = spy_data["Close"].pct_change().dropna()
            aligned = pd.DataFrame(
                {"portfolio": portfolio_returns, "spy": spy_returns}
            ).dropna()

            if len(aligned) > 20:
                cov = float(np.cov(aligned["portfolio"], aligned["spy"])[0][1])
                var_spy = float(np.var(aligned["spy"], ddof=1))
                if var_spy > 0:
                    beta_vs_spy = cov / var_spy

        # VaR at 95% confidence
        var_95 = float(np.percentile(portfolio_returns, 5) * np.sqrt(252))

        # Max drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = float(drawdown.min())

        # Diversification ratio
        individual_vols = returns.std(ddof=1) * np.sqrt(252)
        weighted_avg_vol = float((individual_vols * weights).sum())
        div_ratio = weighted_avg_vol / volatility if volatility > 0 else None

        # Total return
        total_return = float((cumulative.iloc[-1] - 1))

        out = {
            "symbols": symbols,
            "weights": [round(w, 4) for w in weights.tolist()],
            "period": period,
            "expected_return": round(expected_return * 100, 2),
            "volatility": round(volatility * 100, 2),
            "sharpe_ratio": round(sharpe, 3) if sharpe is not None else None,
            "beta_vs_spy": round(beta_vs_spy, 3) if beta_vs_spy is not None else None,
            "var_95": round(var_95 * 100, 2),
            "max_drawdown": round(max_dd * 100, 2),
            "diversification_ratio": round(div_ratio, 3)
            if div_ratio is not None
            else None,
            "total_return": round(total_return * 100, 2),
        }
        return _cache_set(key, out)

    except Exception as e:
        return {
            "symbols": symbols,
            "weights": weights.tolist(),
            "period": period,
            "error": f"Calculation error: {str(e)}",
        }


@mcp.tool()
@handle_errors
def efficient_frontier(
    symbols: list[str],
    period: str = "1y",
    risk_free_rate: float = 0.04,
    num_points: int = 100,
) -> dict:
    """
    Calculate the efficient frontier - the set of optimal portfolios.

    The efficient frontier shows the best possible return for each level of risk.
    Essential for Modern Portfolio Theory (MPT) based portfolio construction.

    Use this for: finding optimal allocations, visualizing risk/return tradeoffs.

    Args:
        symbols: List of stock ticker symbols (2-10 recommended)
        period: Time period for historical data
            - "1y", "3y", "5y": Different lookback periods
            Default: "1y"
        risk_free_rate: Annual risk-free rate (default: 0.04 = 4%)
        num_points: Number of frontier points to calculate (default: 100)

    Returns:
        dict with:
        - symbols: Stock tickers
        - period: Time period
        - frontier_points: Array of points, each with:
            * expected_return: Expected return (%)
            * volatility: Portfolio volatility (%)
            * sharpe_ratio: Sharpe ratio
            * weights: Allocation weights
        - min_volatility_portfolio: Lowest risk portfolio
        - max_sharpe_portfolio: Best risk-adjusted portfolio
        - individual_assets: Return/volatility for each asset

    Example:
        Input: efficient_frontier(["AAPL", "MSFT", "GOOGL", "BND"])
        Output: {
            "frontier_points": [
                {"expected_return": 8.5, "volatility": 12.3, "weights": [...]},
                ...
            ],
            "max_sharpe_portfolio": {"expected_return": 15.2, ...}
        }

    Interpretation:
        - Points on frontier = optimal portfolios (max return for given risk)
        - Points below frontier = suboptimal allocations
        - Max Sharpe = best risk-adjusted portfolio
        - Min volatility = lowest risk portfolio

    Analysis Tips:
        - Include bonds/defensive assets for better frontier
        - Max Sharpe often better than min volatility for growth
        - Check if current portfolio is on the frontier
        - Use portfolio_optimization() to find specific optimal allocation

    Note: Computationally intensive. Data cached for 20 seconds. Assumes normal distribution.
    """
    if len(symbols) < 2:
        return {"error": "Need at least 2 symbols for efficient frontier"}

    symbols = [s.upper().strip() for s in symbols]
    key = f"ef:{':'.join(sorted(symbols))}:{period}:{risk_free_rate}:{num_points}"
    hit = _cache_get(key)
    if hit:
        return hit

    # Download data
    data = yf.download(symbols, period=period, progress=False)

    if data.empty or "Close" not in data:
        return {
            "symbols": symbols,
            "period": period,
            "error": "Insufficient data",
        }

    try:
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            prices = pd.DataFrame({symbols[0]: data["Close"]})

        returns = prices.pct_change().dropna()

        if len(returns) < 20:
            return {
                "symbols": symbols,
                "period": period,
                "error": "Insufficient return data",
            }

        # Calculate expected returns and covariance
        mean_returns = returns.mean() * 252  # Annualized
        cov_matrix = returns.cov() * 252  # Annualized

        # Import scipy for optimization
        from scipy.optimize import minimize

        n_assets = len(symbols)

        # Helper functions
        def portfolio_stats(weights):
            returns = np.dot(weights, mean_returns)
            vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return returns, vol

        def neg_sharpe(weights):
            ret, vol = portfolio_stats(weights)
            return -(ret - risk_free_rate) / vol if vol > 0 else 0

        def portfolio_volatility(weights):
            return portfolio_stats(weights)[1]

        # Constraints and bounds
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))
        init_guess = np.array([1.0 / n_assets] * n_assets)

        # Find minimum volatility portfolio
        opt_min_vol = minimize(
            portfolio_volatility,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        min_vol_return, min_vol = portfolio_stats(opt_min_vol.x)

        # Find maximum Sharpe ratio portfolio
        opt_max_sharpe = minimize(
            neg_sharpe,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        max_sharpe_return, max_sharpe_vol = portfolio_stats(opt_max_sharpe.x)

        # Generate frontier points
        target_returns = np.linspace(
            min_vol_return, max_sharpe_return * 1.5, num_points
        )
        frontier_points = []

        for target in target_returns:
            cons = (
                {"type": "eq", "fun": lambda x: np.sum(x) - 1},
                {"type": "eq", "fun": lambda x: np.dot(x, mean_returns) - target},
            )

            result = minimize(
                portfolio_volatility,
                init_guess,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
            )

            if result.success:
                ret, vol = portfolio_stats(result.x)
                sharpe = (ret - risk_free_rate) / vol if vol > 0 else None

                frontier_points.append(
                    {
                        "expected_return": round(float(ret * 100), 2),
                        "volatility": round(float(vol * 100), 2),
                        "sharpe_ratio": round(float(sharpe), 3) if sharpe else None,
                        "weights": [round(float(w), 4) for w in result.x],
                    }
                )

        # Individual asset stats
        individual_assets = []
        for i, symbol in enumerate(symbols):
            ret = float(mean_returns.iloc[i])
            vol = float(np.sqrt(cov_matrix.iloc[i, i]))
            sharpe = (ret - risk_free_rate) / vol if vol > 0 else None

            individual_assets.append(
                {
                    "symbol": symbol,
                    "expected_return": round(ret * 100, 2),
                    "volatility": round(vol * 100, 2),
                    "sharpe_ratio": round(sharpe, 3) if sharpe else None,
                }
            )

        out = {
            "symbols": symbols,
            "period": period,
            "num_points": len(frontier_points),
            "frontier_points": frontier_points,
            "min_volatility_portfolio": {
                "expected_return": round(float(min_vol_return * 100), 2),
                "volatility": round(float(min_vol * 100), 2),
                "sharpe_ratio": round(
                    float((min_vol_return - risk_free_rate) / min_vol), 3
                ),
                "weights": [round(float(w), 4) for w in opt_min_vol.x],
            },
            "max_sharpe_portfolio": {
                "expected_return": round(float(max_sharpe_return * 100), 2),
                "volatility": round(float(max_sharpe_vol * 100), 2),
                "sharpe_ratio": round(float(-neg_sharpe(opt_max_sharpe.x)), 3),
                "weights": [round(float(w), 4) for w in opt_max_sharpe.x],
            },
            "individual_assets": individual_assets,
        }
        return _cache_set(key, out)

    except Exception as e:
        return {
            "symbols": symbols,
            "period": period,
            "error": f"Optimization error: {str(e)}",
        }


@mcp.tool()
@handle_errors
def portfolio_optimization(
    symbols: list[str],
    period: str = "1y",
    risk_free_rate: float = 0.04,
    optimization_method: str = "max_sharpe",
    constraints: dict = None,
) -> dict:
    """
    Find optimal portfolio allocation using Modern Portfolio Theory (MPT).

    Optimize portfolio weights based on different objectives and constraints.

    Use this for: automated portfolio allocation, rebalancing decisions, strategy testing.

    Args:
        symbols: List of stock ticker symbols
        period: Historical data period (default: "1y")
        risk_free_rate: Annual risk-free rate (default: 0.04 = 4%)
        optimization_method: Optimization objective
            - "max_sharpe": Maximum Sharpe ratio (best risk-adjusted return)
            - "min_volatility": Minimum variance (lowest risk)
            - "max_return": Maximum return (ignores risk - not recommended)
            - "target_return": Target a specific return level
            - "target_risk": Target a specific volatility level
            Default: "max_sharpe"
        constraints: Optional dict with:
            - max_weight: Maximum weight per asset (default: 1.0)
            - min_weight: Minimum weight per asset (default: 0.0)
            - target_return: Required for "target_return" method (e.g., 0.15 for 15%)
            - target_risk: Required for "target_risk" method (e.g., 0.20 for 20%)

    Returns:
        dict with:
        - symbols: Stock tickers
        - optimal_weights: Optimized allocation weights
        - expected_return: Expected annual return (%)
        - volatility: Portfolio volatility (%)
        - sharpe_ratio: Portfolio Sharpe ratio
        - optimization_method: Method used
        - constraints_applied: Constraints that were applied

    Example:
        Input: portfolio_optimization(
            symbols=["AAPL", "MSFT", "BND", "GLD"],
            optimization_method="max_sharpe",
            constraints={"max_weight": 0.4, "min_weight": 0.05}
        )
        Output: {
            "optimal_weights": [0.35, 0.30, 0.25, 0.10],
            "expected_return": 12.5,
            "sharpe_ratio": 1.45,
            ...
        }

    Interpretation:
        - max_sharpe: Best for growth with reasonable risk
        - min_volatility: Best for conservative/defensive portfolios
        - target_return/risk: For specific allocation goals
        - Use constraints to enforce diversification or limits

    Analysis Tips:
        - Include diverse assets (stocks, bonds, commodities)
        - Use max_weight to enforce diversification
        - Compare results across different methods
        - Backtest optimized weights before implementing
        - Reoptimize periodically (quarterly recommended)

    Note: Based on historical data - past performance doesn't guarantee future results.
    """
    if len(symbols) < 2:
        return {"error": "Need at least 2 symbols for optimization"}

    symbols = [s.upper().strip() for s in symbols]

    # Parse constraints
    if constraints is None:
        constraints = {}

    max_weight = constraints.get("max_weight", 1.0)
    min_weight = constraints.get("min_weight", 0.0)
    target_return = constraints.get("target_return")
    target_risk = constraints.get("target_risk")

    # Validation
    if optimization_method == "target_return" and target_return is None:
        return {"error": "target_return method requires 'target_return' in constraints"}

    if optimization_method == "target_risk" and target_risk is None:
        return {"error": "target_risk method requires 'target_risk' in constraints"}

    key = f"portopt:{':'.join(sorted(symbols))}:{period}:{optimization_method}:{str(constraints)}"
    hit = _cache_get(key)
    if hit:
        return hit

    # Download data
    data = yf.download(symbols, period=period, progress=False)

    if data.empty or "Close" not in data:
        return {
            "symbols": symbols,
            "optimization_method": optimization_method,
            "error": "Insufficient data",
        }

    try:
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            prices = pd.DataFrame({symbols[0]: data["Close"]})

        returns = prices.pct_change().dropna()

        if len(returns) < 20:
            return {
                "symbols": symbols,
                "optimization_method": optimization_method,
                "error": "Insufficient return data",
            }

        # Calculate expected returns and covariance
        mean_returns = returns.mean() * 252  # Annualized
        cov_matrix = returns.cov() * 252  # Annualized

        from scipy.optimize import minimize

        n_assets = len(symbols)

        def portfolio_stats(weights):
            ret = np.dot(weights, mean_returns)
            vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return ret, vol

        # Set up optimization based on method
        if optimization_method == "max_sharpe":

            def objective(weights):
                ret, vol = portfolio_stats(weights)
                return -(ret - risk_free_rate) / vol if vol > 0 else 0

        elif optimization_method == "min_volatility":

            def objective(weights):
                return portfolio_stats(weights)[1]

        elif optimization_method == "max_return":

            def objective(weights):
                return -portfolio_stats(weights)[0]

        elif optimization_method == "target_return":

            def objective(weights):
                return portfolio_stats(weights)[1]

        elif optimization_method == "target_risk":

            def objective(weights):
                return -portfolio_stats(weights)[0]

        else:
            return {"error": f"Unknown optimization method: {optimization_method}"}

        # Constraints
        cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

        if optimization_method == "target_return":
            cons.append(
                {"type": "eq", "fun": lambda x: np.dot(x, mean_returns) - target_return}
            )

        if optimization_method == "target_risk":
            cons.append(
                {
                    "type": "eq",
                    "fun": lambda x: np.sqrt(np.dot(x.T, np.dot(cov_matrix, x)))
                    - target_risk,
                }
            )

        # Bounds
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        init_guess = np.array([1.0 / n_assets] * n_assets)

        # Optimize
        result = minimize(
            objective,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )

        if not result.success:
            return {
                "symbols": symbols,
                "optimization_method": optimization_method,
                "error": f"Optimization failed: {result.message}",
            }

        optimal_weights = result.x
        expected_ret, vol = portfolio_stats(optimal_weights)
        sharpe = (expected_ret - risk_free_rate) / vol if vol > 0 else None

        out = {
            "symbols": symbols,
            "optimal_weights": [round(float(w), 4) for w in optimal_weights],
            "expected_return": round(float(expected_ret * 100), 2),
            "volatility": round(float(vol * 100), 2),
            "sharpe_ratio": round(float(sharpe), 3) if sharpe else None,
            "optimization_method": optimization_method,
            "constraints_applied": {
                "max_weight": max_weight,
                "min_weight": min_weight,
                "target_return": target_return,
                "target_risk": target_risk,
            },
        }
        return _cache_set(key, out)

    except Exception as e:
        return {
            "symbols": symbols,
            "optimization_method": optimization_method,
            "error": f"Optimization error: {str(e)}",
        }


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


# Sector & Industry Tools
@mcp.tool()
@handle_errors
def get_sector_overview(sector_key: str) -> dict:
    """
    Get comprehensive sector information including top companies, ETFs, and industries.

    Use this for: sector rotation strategies, finding sector leaders, exploring investment themes.

    Args:
        sector_key: Sector identifier. Common sectors:
            - "technology" - Tech companies
            - "healthcare" - Healthcare & pharma
            - "financial-services" - Banks, insurance, investment firms
            - "consumer-cyclical" - Retail, automotive, leisure
            - "industrials" - Manufacturing, aerospace, defense
            - "energy" - Oil, gas, renewable energy
            - "utilities" - Electric, water, gas utilities
            - "real-estate" - REITs and real estate
            - "basic-materials" - Mining, chemicals, metals
            - "consumer-defensive" - Food, beverages, household products
            - "communication-services" - Telecom, media, entertainment

    Returns:
        dict with:
        - sector_key: The sector identifier
        - name: Sector display name
        - symbol: Sector index symbol (if available)
        - overview: Sector description
        - top_companies: Array of leading companies in sector (symbol, name, market cap, weight)
        - top_etfs: Top sector ETFs
        - top_mutual_funds: Top sector mutual funds
        - industries: List of industries within this sector

    Example:
        Input: get_sector_overview("technology")
        Output: {"sector_key": "technology", "name": "Technology",
                 "top_companies": [{"symbol": "AAPL", "name": "Apple Inc.", ...}, ...],
                 "industries": ["software-infrastructure", "semiconductors", ...]}

    Strategy Tips:
        - Compare sector performance to identify rotation opportunities
        - Use top_companies for sector-based stock picking
        - Check industries list for sub-sector specialization
        - Top ETFs provide easy sector exposure

    Note: Some sectors may have limited data. Check if fields are None.
    """
    sector_key = sector_key.lower().strip()
    cache_key = f"sector:{sector_key}"
    hit = _cache_get(cache_key)
    if hit:
        return hit

    try:
        sector = Sector(sector_key)

        # Get all available sector data
        out = {
            "sector_key": sector_key,
            "name": getattr(sector, "name", None),
            "symbol": getattr(sector, "symbol", None),
            "overview": getattr(sector, "overview", None),
        }

        # Top companies
        try:
            top_cos = sector.top_companies
            if top_cos is not None and not top_cos.empty:
                out["top_companies"] = _df_to_rows(top_cos, max_rows=50)
            else:
                out["top_companies"] = []
        except Exception:
            out["top_companies"] = []

        # Top ETFs
        try:
            top_etfs = sector.top_etfs
            if top_etfs is not None and not top_etfs.empty:
                out["top_etfs"] = _df_to_rows(top_etfs, max_rows=20)
            else:
                out["top_etfs"] = []
        except Exception:
            out["top_etfs"] = []

        # Top mutual funds
        try:
            top_mf = sector.top_mutual_funds
            if top_mf is not None and not top_mf.empty:
                out["top_mutual_funds"] = _df_to_rows(top_mf, max_rows=20)
            else:
                out["top_mutual_funds"] = []
        except Exception:
            out["top_mutual_funds"] = []

        # Industries list
        try:
            industries = sector.industries
            if isinstance(industries, dict):
                out["industries"] = list(industries.keys())
            elif isinstance(industries, list):
                out["industries"] = industries
            else:
                out["industries"] = []
        except Exception:
            out["industries"] = []

        # Cache for 1 hour (sector data doesn't change often)
        return _cache_set(cache_key, out)

    except Exception as e:
        logger.error(f"Error fetching sector {sector_key}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid sector key or data unavailable: {str(e)}",
        )


@mcp.tool()
@handle_errors
def get_industry_overview(industry_key: str) -> dict:
    """
    Get industry-specific data including top performing and growth companies.

    Use this for: finding industry leaders, growth stock discovery, competitive analysis.

    Args:
        industry_key: Industry identifier. Examples:
            - "software-infrastructure" - Cloud, enterprise software
            - "semiconductors" - Chip manufacturers
            - "biotechnology" - Biotech & life sciences
            - "drug-manufacturers-general" - Big pharma
            - "banks-regional" - Regional banks
            - "oil-gas-exploration" - E&P companies
            - "solar" - Solar energy companies
            (Use get_sector_overview to see all industries in a sector)

    Returns:
        dict with:
        - industry_key: The industry identifier
        - name: Industry display name
        - sector_key: Parent sector
        - sector_name: Parent sector name
        - overview: Industry description
        - top_companies: Leading companies by market cap
        - top_performing_companies: Best recent performers
        - top_growth_companies: Highest growth companies

    Example:
        Input: get_industry_overview("software-infrastructure")
        Output: {"industry_key": "software-infrastructure",
                 "name": "Software - Infrastructure",
                 "sector_key": "technology",
                 "top_companies": [{"symbol": "MSFT", ...}, ...]}

    Strategy Tips:
        - Compare top_performing vs top_growth to find momentum
        - Use for picking best-of-breed in an industry
        - Cross-reference with sector trends
        - Check overview for industry-specific dynamics

    Note: Some industries may have sparse data. Check for None/empty arrays.
    """
    industry_key = industry_key.lower().strip()
    cache_key = f"industry:{industry_key}"
    hit = _cache_get(cache_key)
    if hit:
        return hit

    try:
        industry = Industry(industry_key)

        out = {
            "industry_key": industry_key,
            "name": getattr(industry, "name", None),
            "sector_key": getattr(industry, "sector_key", None),
            "sector_name": getattr(industry, "sector_name", None),
            "overview": getattr(industry, "overview", None),
        }

        # Top companies
        try:
            top_cos = industry.top_companies
            if top_cos is not None and not top_cos.empty:
                out["top_companies"] = _df_to_rows(top_cos, max_rows=50)
            else:
                out["top_companies"] = []
        except Exception:
            out["top_companies"] = []

        # Top performing
        try:
            top_perf = industry.top_performing_companies
            if top_perf is not None and not top_perf.empty:
                out["top_performing_companies"] = _df_to_rows(top_perf, max_rows=30)
            else:
                out["top_performing_companies"] = []
        except Exception:
            out["top_performing_companies"] = []

        # Top growth
        try:
            top_growth = industry.top_growth_companies
            if top_growth is not None and not top_growth.empty:
                out["top_growth_companies"] = _df_to_rows(top_growth, max_rows=30)
            else:
                out["top_growth_companies"] = []
        except Exception:
            out["top_growth_companies"] = []

        return _cache_set(cache_key, out)

    except Exception as e:
        logger.error(f"Error fetching industry {industry_key}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid industry key or data unavailable: {str(e)}",
        )


# Stock Screening Tools
@mcp.tool()
@handle_errors
def screen_stocks(
    region: str = "us",
    sector: str = None,
    industry: str = None,
    exchange: str = None,
    limit: int = 25,
) -> dict:
    """
    Screen stocks by criteria to discover investment opportunities.

    Use this for: stock discovery, building watchlists, finding stocks matching criteria.

    Args:
        region: Geographic region. Options: "us", "br", "au",  "ca", "fr", "de", "hk", "in", etc.
        sector: Sector filter (use sector keys from get_sector_overview)
        industry: Industry filter (use industry keys from get_industry_overview)
        exchange: Exchange filter. Examples: "nas" (NASDAQ), "nyse", "amex"
        limit: Max results to return (1-250, default: 25)

    Returns:
        dict with:
        - criteria: The search criteria used
        - count: Number of results found
        - results: Array of matching stocks with:
            * symbol: Ticker symbol
            * name: Company name
            * exchange: Stock exchange
            * Additional fields vary by result

    Example:
        Input: screen_stocks(region="us", sector="technology", limit=10)
        Output: {"criteria": {"region": "us", "sector": "technology"},
                 "count": 10,
                 "results": [{"symbol": "AAPL", "name": "Apple Inc.", ...}, ...]}

    Common Screening Strategies:
        1. Sector plays: screen_stocks(region="us", sector="energy")
        2. Industry focus: screen_stocks(region="us", industry="semiconductors")
        3. Exchange-specific: screen_stocks(region="us", exchange="nyse")
        4. Combination: screen_stocks(region="us", sector="healthcare", exchange="nas")

    Note: Results limited to 250. Refine criteria if hitting limit. Some combinations may return few/no results.
    """
    # Build query
    filters = []
    criteria_dict = {"region": region}

    if region:
        filters.append(EquityQuery("eq", ["region", region]))

    if sector:
        filters.append(EquityQuery("eq", ["sector", sector.lower()]))
        criteria_dict["sector"] = sector

    if industry:
        filters.append(EquityQuery("eq", ["industry", industry.lower()]))
        criteria_dict["industry"] = industry

    if exchange:
        filters.append(EquityQuery("eq", ["exchange", exchange.lower()]))
        criteria_dict["exchange"] = exchange

    # Combine filters with AND
    if len(filters) > 1:
        query = filters[0]
        for f in filters[1:]:
            query = query & f
    else:
        query = filters[0] if filters else EquityQuery("eq", ["region", region])

    cache_key = f"screen:{region}:{sector}:{industry}:{exchange}:{limit}"
    hit = _cache_get(cache_key)
    if hit:
        return hit

    try:
        # Run screen
        results = screen(query, size=min(limit, 250))

        # Convert to list of dicts
        results_list = []
        if results is not None and not results.empty:
            results_list = _df_to_rows(results, max_rows=limit)

        out = {
            "criteria": criteria_dict,
            "count": len(results_list),
            "results": results_list,
        }

        return _cache_set(cache_key, out)

    except Exception as e:
        logger.error(f"Screening error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Screening failed: {str(e)}",
        )


@mcp.tool()
@handle_errors
def screen_predefined(screener_name: str) -> dict:
    """
    Use Yahoo Finance predefined stock screeners for quick market insights.

    Use this for: finding hot stocks, market movers, identifying opportunities/risks.

    Args:
        screener_name: Predefined screener. Options:
            - "most_actives" - Highest volume stocks
            - "day_gainers" - Biggest percentage gainers today
            - "day_losers" - Biggest percentage losers today
            - "growth_technology_stocks" - Tech growth stocks
            - "aggressive_small_caps" - High risk small cap stocks
            - "small_cap_gainers" - Small cap stocks gaining today
            - "undervalued_large_caps" - Value large caps
            - "undervalued_growth_stocks" - Growth at reasonable price
            - "conservative_foreign_funds" - Conservative international funds
            - "high_yield_bond" - High yield bond funds

    Returns:
        dict with:
        - screener: The screener name used
        - count: Number of results
        - results: Array of stocks with symbol, name, price, change%, volume, etc.

    Example:
        Input: screen_predefined("day_gainers")
        Output: {"screener": "day_gainers",
                 "count": 25,
                 "results": [{"symbol": "XYZ", "name": "XYZ Corp",
                             "regularMarketChangePercent": 15.2, ...}, ...]}

    Trading Strategies:
        - day_gainers: Momentum trading, identify breakouts
        - day_losers: Contrarian plays, dip buying opportunities
        - most_actives: High liquidity for day trading
        - growth_technology_stocks: Long-term tech exposure
        - undervalued_large_caps: Value investing ideas

    Note: Predefined screeners return ~25-100 results. Data updates throughout trading day.
    """
    screener_name = screener_name.lower().strip()
    cache_key = f"screen_pre:{screener_name}"
    hit = _cache_get(cache_key)
    if hit:
        return hit

    try:
        # Use predefined screener
        results = screen(screener_name)

        results_list = []
        if results is not None and not results.empty:
            results_list = _df_to_rows(results, max_rows=250)

        out = {
            "screener": screener_name,
            "count": len(results_list),
            "results": results_list,
        }

        return _cache_set(cache_key, out)

    except Exception as e:
        logger.error(f"Predefined screener error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid screener name or data unavailable: {str(e)}",
        )


# Enhanced Ticker Analysis
@mcp.tool()
@handle_errors
def get_upgrades_downgrades(symbol: str) -> dict:
    """
    Get recent analyst rating changes (upgrades and downgrades).

    Use this for: tracking sentiment changes, identifying catalysts, confirming trends.

    Args:
        symbol: Stock ticker symbol

    Returns:
        dict with:
        - symbol: The ticker symbol
        - upgrades_downgrades: Array of rating changes with:
            * GradeDate: Date of rating change
            * Firm: Analyst firm name
            * Action: "up", "down", "init", "main", "reit"
            * FromGrade: Previous rating
            * ToGrade: New rating

    Example:
        Input: get_upgrades_downgrades("NVDA")
        Output: {"symbol": "NVDA",
                 "upgrades_downgrades": [
                     {"GradeDate": "2025-11-28", "Firm": "Goldman Sachs",
                      "Action": "up", "FromGrade": "Neutral", "ToGrade": "Buy"},
                     ...
                 ]}

    Interpretation:
        - Multiple upgrades = Building bullish consensus
        - Downgrades after rally = Potential reversal warning
        - Upgrade from major firms (Goldman, Morgan Stanley) = Strong signal
        - "init" = Initial coverage (new analyst following)
        - "main" = Maintained rating
        - "reit" = Reiterated rating

    Trading Signals:
        - Upgrade cluster often precedes price moves
        - Downgrade after earnings = Reassessment of outlook
        - Track which firms are consistently right

    Note: Returns recent history (typically last ~2 years). May be empty if no recent changes.
    """
    symbol = symbol.upper().strip()
    cache_key = f"upgrades:{symbol}"
    hit = _cache_get(cache_key)
    if hit:
        return hit

    t = yf.Ticker(symbol)

    try:
        upgrades = t.upgrades_downgrades
        if upgrades is not None and not upgrades.empty:
            rows = _df_to_rows(upgrades, max_rows=500)
        else:
            rows = []

        out = {"symbol": symbol, "upgrades_downgrades": rows}

        return _cache_set(cache_key, out)

    except Exception:
        # Return empty if not available
        return {"symbol": symbol, "upgrades_downgrades": []}


@mcp.tool()
@handle_errors
def get_insider_transactions(symbol: str) -> dict:
    """
    Get insider buying and selling activity.

    Use this for: gauging insider confidence, identifying potential catalysts, risk assessment.

    Args:
        symbol: Stock ticker symbol

    Returns:
        dict with:
        - symbol: The ticker symbol
        - insider_transactions: Array of transactions with:
            * Date: Transaction date (recent first)
            * Insider: Name and title of insider
            * Transaction: "Buy", "Sale", "Option Exercise", etc.
            * Shares: Number of shares
            * Value: Dollar value of transaction
            * Shares Owned: Total shares owned after transaction

    Example:
        Input: get_insider_transactions("TSLA")
        Output: {"symbol": "TSLA",
                 "insider_transactions": [
                     {"Date": "2025-11-15", "Insider": "Elon Musk - CEO",
                      "Transaction": "Sale", "Shares": 500000, "Value": 125000000,
                      "Shares Owned": 238000000},
                     ...
                 ]}

    Interpretation:
        - Insider buying = Strong bullish signal (putting own money in)
        - Insider selling = Often routine (diversification, taxes, estate planning)
        - Multiple insiders buying = Very bullish
        - C-level executives buying = Strongest signal
        - Director buying after price drop = Potential bottom
        - Heavy selling before earnings = Potential warning

    Red Flags:
        - Clustered selling by multiple insiders
        - CEO selling large stakes
        - Selling by directors/board members

    Green Flags:
        - Any insider buying (especially CFO/CEO)
        - Buying during blackout window exceptions
        - Form 4 filings with open market purchases

    Note: Regulated insider trades only (officers, directors, 10%+ holders). Delayed reporting (usually 2 days).
    """
    symbol = symbol.upper().strip()
    cache_key = f"insider:{symbol}"
    hit = _cache_get(cache_key)
    if hit:
        return hit

    t = yf.Ticker(symbol)

    try:
        insider_trans = t.insider_transactions
        if insider_trans is not None and not insider_trans.empty:
            rows = _df_to_rows(insider_trans, max_rows=500)
        else:
            rows = []

        out = {"symbol": symbol, "insider_transactions": rows}

        return _cache_set(cache_key, out)

    except Exception:
        return {"symbol": symbol, "insider_transactions": []}


@mcp.tool()
@handle_errors
def get_insider_roster(symbol: str) -> dict:
    """
    Get current insider ownership roster.

    Use this for: understanding ownership structure, tracking key stakeholders.

    Args:
        symbol: Stock ticker symbol

    Returns:
        dict with:
        - symbol: The ticker symbol
        - insider_roster: Array of insiders with:
            * Name: Insider name
            * Position: Title/role
            * Shares Owned: Current share ownership
            * Date: As of date

    Example:
        Input: get_insider_roster("AAPL")
        Output: {"symbol": "AAPL",
                 "insider_roster": [
                     {"Name": "Tim Cook", "Position": "CEO",
                      "Shares Owned": 3200000, "Date": "2025-10-31"},
                     ...
                 ]}

    Use Cases:
        - Verify management has skin in the game
        - Track ownership changes over time
        - Compare with total shares outstanding
        - Identify key decision makers

    Note: Filed quarterly. May not reflect very recent changes.
    """
    symbol = symbol.upper().strip()
    cache_key = f"insider_roster:{symbol}"
    hit = _cache_get(cache_key)
    if hit:
        return hit

    t = yf.Ticker(symbol)

    try:
        roster = t.insider_roster_holders
        if roster is not None and not roster.empty:
            rows = _df_to_rows(roster, max_rows=100)
        else:
            rows = []

        out = {"symbol": symbol, "insider_roster": rows}

        return _cache_set(cache_key, out)

    except Exception:
        return {"symbol": symbol, "insider_roster": []}


# Multi-Ticker Operations
@mcp.tool()
@handle_errors
def download_batch(
    symbols: list[str], period: str = "1mo", interval: str = "1d"
) -> dict:
    """
    Download historical data for multiple tickers at once (more efficient than individual calls).

    Use this for: portfolio analysis, comparing multiple stocks, backtesting strategies.

    Args:
        symbols: List of ticker symbols (e.g., ["AAPL", "MSFT", "GOOGL"])
        period: Time range (same as history tool): "5d", "1mo", "3mo", "6mo", "1y", "5y", "max"
        interval: Data frequency: "1d", "1wk", "1mo" (intraday: "1m", "5m", "15m", "60m")

    Returns:
        dict with:
        - symbols: List of requested symbols
        - period, interval: Parameters used
        - data: Dict mapping each symbol to its history data:
            * Each symbol key contains array of OHLCV rows
            * Missing/invalid symbols will have empty arrays

    Example:
        Input: download_batch(["AAPL", "MSFT"], period="5d", interval="1d")
        Output: {"symbols": ["AAPL", "MSFT"],
                 "period": "5d",
                 "interval": "1d",
                 "data": {
                     "AAPL": [{"Date": "2025-11-25", "Open": 189.5, ...}, ...],
                     "MSFT": [{"Date": "2025-11-25", "Open": 378.2, ...}, ...]
                 }}

    Efficiency:
        - 10x faster than calling history() 10 times
        - Single API call to Yahoo Finance
        - Automatic alignment of dates

    Use Cases:
        - Portfolio tracking across multiple holdings
        - Sector comparison (get all tech stocks)
        - Correlation analysis
        - Pairs trading research

    Note: Max ~10 symbols recommended for performance. Results may vary by symbol (some may fail).
    """
    symbols_upper = [s.upper().strip() for s in symbols]
    cache_key = f"batch:{','.join(sorted(symbols_upper))}:{period}:{interval}"
    hit = _cache_get(cache_key)
    if hit:
        return hit

    try:
        # Use yf.download for batch download
        data = yf.download(
            symbols_upper,
            period=period,
            interval=interval,
            group_by="ticker",
            progress=False,
        )

        # Parse results per symbol
        results = {}
        for symbol in symbols_upper:
            try:
                if len(symbols_upper) == 1:
                    # Single ticker - data is not grouped
                    symbol_data = data
                else:
                    # Multiple tickers - data is grouped by ticker
                    symbol_data = data[symbol]

                if symbol_data is not None and not symbol_data.empty:
                    results[symbol] = _df_to_rows(symbol_data, max_rows=5000)
                else:
                    results[symbol] = []
            except Exception:
                results[symbol] = []

        out = {
            "symbols": symbols_upper,
            "period": period,
            "interval": interval,
            "data": results,
        }

        return _cache_set(cache_key, out)

    except Exception as e:
        logger.error(f"Batch download error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch download failed: {str(e)}",
        )


if __name__ == "__main__":
    import argparse
    import os

    logger.info("Starting yfinance MCP server")
    logger.info(f"Configuration: cache_ttl={TTL_SECONDS}s")

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
