"""
V5: Simplified Sequential Stock Rating (Hedge Fund Edition)

KEY CHANGES FROM V4:
- Removed strict Pydantic output types (causing max turns)
- Agents return simple text summaries instead
- Final synthesis uses GPT-4o to create structured output
- Much more token-efficient

Install: pip install openai-agents
Run: python stock_rating_workflow_v5.py
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel, Field
from typing import Literal
from agents import Agent, Runner, trace, AgentHooks
from agents.mcp import MCPServerStreamableHttp, MCPServerStreamableHttpParams

MCP_SERVER_URL = "http://localhost:10000/api/mcp"
MCP_TIMEOUT = 30


class StockHooks(AgentHooks):
    def on_run_start(self, context):
        print(f"  🚀 {context.agent.name} starting...")

    def on_run_end(self, context, result):
        print(f"  ✅ {context.agent.name} completed")


# MCP Connection
yfinance_server = MCPServerStreamableHttp(
    params=MCPServerStreamableHttpParams(url=MCP_SERVER_URL, timeout=MCP_TIMEOUT),
    name="yfinance",
)

# SIMPLE TEXT-BASED COLLECTORS (no structured output = faster)
screening_agent = Agent(
    name="screener",
    instructions="""Call fundamentals() ONCE. Extract and check these STRICT criteria:

HEDGE FUND SCREENING CRITERIA (Tiered Analysis):
1. Market Cap: $50B - $500B USD (Large Cap quality companies)
2. Market: United States (check exchange field)
3. Revenue Growth (YoY): > 6% (use revenueGrowth field)
4. EPS Growth (YoY): > 6% (use earningsGrowth field)
5. P/E Ratio: < 50 (use trailingPE or forwardPE)
6. PEG Ratio: < 1.1 (if available - growth at reasonable price)

SCORING LOGIC (Be nuanced, not binary):

**STRONG PASS**: All 6 criteria met
**MARGINAL PASS**: 5 of 6 criteria met + strong performance in others
  Examples:
  - P/E slightly over 50 (50-55) BUT PEG < 1.0 (growth justifies premium)
  - P/E over 50 BUT Revenue growth >20% + EPS growth >30% (high growth justifies premium)
  - Missing PEG BUT all other metrics strong

**FAIL**:
  - Market cap outside range (hard fail)
  - Growth metrics <6% (hard fail)
  - P/E >55 without exceptional growth (>30% revenue + eps)

OUTPUT FORMAT:
Result: STRONG PASS / MARGINAL PASS / FAIL

Detailed Breakdown:
- Market Cap: [value] - [PASS/FAIL]
- Revenue Growth: [value] - [PASS/FAIL]
- EPS Growth: [value] - [PASS/FAIL]
- P/E Ratio: [value] - [PASS/FAIL/MARGINAL]
- PEG Ratio: [value if available] - [PASS/FAIL/NA]

Screening Score: [X/6 criteria met]

Justification: [If MARGINAL PASS, explain why it's worth analyzing despite failing one criterion]

Other Context:
- Exchange: [value]
- Sector: [value]
- Industry: [value]

PHILOSOPHY: We want quality large caps with growth at reasonable prices.
A stock with 40% EPS growth at P/E of 52 is more attractive than 8% growth at P/E of 35.
Use PEG ratio and growth momentum to contextualize valuation multiples.

Extract from ONE fundamentals() call. Do not call repeatedly.""",
    mcp_servers=[yfinance_server],
    hooks=StockHooks(),
    model="gpt-4o-mini",
)

price_agent = Agent(
    name="price",
    instructions="""Call quote() and history(period='6mo', interval='1d') ONCE EACH.

TECHNICAL ANALYSIS (calculate from history data):
1. **Moving Averages**: Compare current price to 50-day & 200-day MA
   - Golden Cross (50 > 200) = Bullish, Death Cross (50 < 200) = Bearish
   - Price above/below MA = trend strength

2. **Momentum**:
   - Recent 30-day performance vs 6-month average
   - Acceleration or deceleration in trend?

3. **Volume Analysis**:
   - Recent volume vs 30-day average (rising volume confirms trends)
   - Accumulation (price up + volume up) or Distribution (price down + volume up)?

4. **Support/Resistance**:
   - Key levels from 52-week high/low
   - Recent consolidation zones

5. **Volatility**:
   - 30-day volatility vs historical average
   - Recent range expansion/contraction

Summarize in 3-4 sentences: trend, momentum, volume character, key levels.""",
    mcp_servers=[yfinance_server],
    hooks=StockHooks(),
    model="gpt-4o-mini",
)

fundamentals_agent = Agent(
    name="fundamentals",
    instructions="""Call fundamentals() and financials() ONCE EACH.

FINANCIAL HEALTH & QUALITY:
1. **Profitability**:
   - Gross, Operating, Net Margins (trending up or down?)
   - Return on Equity (ROE) - aim for >15%
   - Return on Assets (ROA)

2. **Cash Flow Quality**:
   - Operating Cash Flow vs Net Income (OCF/NI > 0.8 is healthy)
   - Free Cash Flow yield (FCF / Market Cap)
   - Cash flow trend (growing or declining?)

3. **Balance Sheet Strength**:
   - Debt-to-Equity ratio (<1.0 is safe, <0.5 is excellent)
   - Current Ratio (>1.5 is healthy)
   - Total Cash vs Total Debt

4. **Capital Efficiency**:
   - Asset turnover
   - Working capital management

5. **Valuation Context**:
   - P/E vs 5-year historical average (if available)
   - PEG ratio (<1 = growth at discount)
   - Price-to-Free-Cash-Flow

Summarize in 3-4 sentences: quality rating, financial health, valuation assessment.""",
    mcp_servers=[yfinance_server],
    hooks=StockHooks(),
    model="gpt-4o-mini",
)

sentiment_agent = Agent(
    name="sentiment",
    instructions="""Call news(), analyst_recommendations(), and get_insider_transactions() ONCE EACH.

SENTIMENT & CONVICTION ANALYSIS:
1. **News Sentiment**:
   - Recent headlines tone (bullish/neutral/bearish)
   - Any major catalysts or concerns?

2. **Analyst Consensus**:
   - Rating distribution (upgrades vs downgrades recently?)
   - Target price vs current price (upside implied?)
   - Changes in consensus (improving or deteriorating?)

3. **Insider Activity** (CRITICAL signal):
   - Recent insider buying = Strong bullish signal
   - Cluster buying by multiple insiders = Very bullish
   - Insider selling = Often routine, but heavy selling = caution
   - Net insider sentiment (buying - selling)

4. **Contrarian Signals**:
   - Is sentiment extremely positive/negative? (often marks tops/bottoms)
   - Market positioning vs fundamentals

Summarize in 3-4 sentences: consensus view, insider conviction, contrarian opportunities.""",
    mcp_servers=[yfinance_server],
    hooks=StockHooks(),
    model="gpt-4o-mini",
)

events_agent = Agent(
    name="events",
    instructions="""Call calendar() ONCE.
Summarize in 1-2 sentences:
- Next earnings date
- Any upcoming catalysts""",
    mcp_servers=[yfinance_server],
    hooks=StockHooks(),
    model="gpt-4o-mini",
)

risk_agent = Agent(
    name="risk_analyst",
    instructions="""Call quote() and history(period='1y', interval='1d') ONCE EACH.

RISK ASSESSMENT:
1. **Volatility Analysis**:
   - 30-day volatility vs 1-year average
   - Recent volatility spikes or compression
   - Volatility trend (rising or falling?)

2. **Beta Analysis** (if available in fundamentals):
   - Beta < 0.8 = Low risk (defensive)
   - Beta 0.8-1.2 = Market risk
   - Beta > 1.5 = High risk (aggressive)

3. **Drawdown Analysis**:
   - Maximum drawdown in last year (worst peak-to-trough decline)
   - Current drawdown from 52-week high
   - Recovery pattern from drawdowns

4. **Risk-Adjusted Performance**:
   - 1-year return vs volatility (higher return per unit of risk is better)
   - Consistency of returns (smooth vs choppy)

5. **Downside Risk**:
   - Frequency of large down days (>3% drops)
   - Support level strength

Summarize in 3-4 sentences: volatility profile, beta/systematic risk, max drawdown, risk-adjusted returns.""",
    mcp_servers=[yfinance_server],
    hooks=StockHooks(),
    model="gpt-4o-mini",
)


# FINAL STRUCTURED OUTPUT
class FinalRating(BaseModel):
    rating: Literal["strong_buy", "buy", "hold", "sell", "strong_sell"]
    confidence: float = Field(ge=1, le=10)
    target_price: float
    reasoning: str
    bull_case: str
    bear_case: str
    position_size: Literal["full", "half", "quarter", "none"]


synthesis_agent = Agent(
    name="portfolio_manager",
    instructions="""You are a hedge fund Portfolio Manager focused on quality large cap growth stocks.

CONTEXT: This stock has ALREADY passed strict fundamental screening:
- Market Cap: $50B-$500B (Large Cap quality)
- Revenue Growth: >6% YoY
- EPS Growth: >6% YoY
- P/E: <50
- PEG: <1.1 (growth at reasonable price)

INPUT: Screening results + 4 analyst summaries (price, fundamentals, sentiment, events)

CREATE INVESTMENT DECISION:
1. Rating (strong_buy/buy/hold/sell/strong_sell)
2. Confidence (1-10)
3. Target price (realistic 12-month estimate)
4. Reasoning (2-3 sentences on key thesis)
5. Bull case (1-2 sentences - what must go right)
6. Bear case (1-2 sentences - what could go wrong)
7. Position size (full/half/quarter/none)

DECISION FRAMEWORK:
- **Strong Buy**: 3:1 reward/risk minimum, all factors positive, imminent catalyst, conf 9-10 → Full position
- **Buy**: 2:1 reward/risk, solid fundamentals, no immediate catalyst, conf 7-8 → Half position
- **Hold**: Fairly valued, wait for better entry, conf 5-6 → Quarter or none
- **Sell**: Thesis broken, valuation stretched, conf 6-8 → None
- **Strong Sell**: Fundamental deterioration + technical breakdown, conf 9-10 → None (short if applicable)

PHILOSOPHY:
"Capital Preservation" then "Alpha Generation"
We invest in quality large caps with proven growth, trading at reasonable valuations.
We are not in the business of losing money.""",
    output_type=FinalRating,
    model="gpt-4o",
)


async def main():
    print("🤖 V5: Simplified Sequential Hedge Fund Workflow")
    print("=" * 70)

    symbol = input("\nEnter stock symbol: ").upper().strip()
    if not symbol or len(symbol) > 5:
        print("❌ Invalid symbol")
        return

    print(f"\n📊 Analyzing {symbol}...")
    print("=" * 70)

    try:
        async with yfinance_server:
            with trace(f"Stock Rating: {symbol}"):
                # PHASE 1: SCREENING
                print("\n🔍 Phase 1: Screening")
                print("-" * 70)

                screening_result = await Runner.run(
                    screening_agent, f"Screen {symbol}", max_turns=30
                )
                screening_text = str(screening_result.final_output)

                print(f"\n{screening_text}")

                # Three-tier screening result
                if "STRONG PASS" in screening_text.upper():
                    print(
                        "\n✅ Stock STRONGLY passed screening. Proceeding with comprehensive analysis..."
                    )
                elif "MARGINAL PASS" in screening_text.upper():
                    print(
                        "\n⚠️  Stock MARGINALLY passed screening. Proceeding with analysis (use caution)..."
                    )
                elif "FAIL" in screening_text.upper():
                    print(
                        "\n❌ Stock failed screening. Recommendation: Pass on this opportunity."
                    )
                    # return
                else:
                    # Fallback logic
                    print(
                        "\n✅ Stock passed screening. Proceeding with deep analysis..."
                    )

                # PHASE 2: DATA COLLECTION (sequential)
                print("\n\n📈 Phase 2: Data Collection")
                print("-" * 70)

                summaries = []
                agents = [
                    (price_agent, "price"),
                    (fundamentals_agent, "fundamentals"),
                    (sentiment_agent, "sentiment"),
                    (events_agent, "events"),
                    (risk_agent, "risk"),  # NEW: Risk assessment
                ]

                for agent, name in agents:
                    try:
                        result = await Runner.run(
                            agent, f"Analyze {symbol}", max_turns=30
                        )
                        summary = str(result.final_output)
                        summaries.append(f"{name.upper()}: {summary}")
                        await asyncio.sleep(1)  # Rate limit buffer
                    except Exception as e:
                        print(f"  ⚠️  {name} failed: {e}")
                        summaries.append(f"{name.upper()}: Unable to collect data")

                # PHASE 3: SYNTHESIS
                print("\n\n⭐ Phase 3: Investment Decision")
                print("-" * 70)

                combined_data = f"""SCREENING:
{screening_text}

DATA:
{chr(10).join(summaries)}"""

                rating_result = await Runner.run(
                    synthesis_agent,
                    f"Create investment recommendation for {symbol}:\n\n{combined_data}",
                    max_turns=15,
                )
                rating = rating_result.final_output

                # DISPLAY
                print("\n" + "=" * 70)
                print(f"📋 RECOMMENDATION: {symbol}")
                print("=" * 70)
                print(f"🎯 {rating.rating.upper().replace('_', ' ')}")
                print(f"📊 Confidence: {rating.confidence}/10")
                print(f"💰 Target: ${rating.target_price:.2f}")
                print(f"⚖️  Size: {rating.position_size.upper()}")

                print(f"\n💭 Reasoning:\n  {rating.reasoning}")
                print(f"\n🐂 Bull Case:\n  {rating.bull_case}")
                print(f"\n🐻 Bear Case:\n  {rating.bear_case}")
                print("\n" + "=" * 70)

                # Export
                if input("\nExport? (y/n): ").lower() == "y":
                    import json
                    from datetime import datetime

                    report = {
                        "symbol": symbol,
                        "timestamp": datetime.now().isoformat(),
                        "screening": screening_text,
                        "data_summaries": summaries,
                        "rating": rating.model_dump(),
                    }

                    filename = (
                        f"{symbol}_v5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    )
                    with open(filename, "w") as f:
                        json.dump(report, f, indent=2)
                    print(f"✅ Saved to {filename}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Cancelled")
