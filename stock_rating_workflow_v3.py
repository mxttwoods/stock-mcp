"""
V3: Parallelized Stock Rating Workflow

Uses parallelization for 3x faster data collection.
Combines best practices: HostedMCPTool, parallel execution, hooks, structured outputs.

Install: pip install openai-agents
Run: export OPENAI_API_KEY=your_key && python stock_rating_workflow_v3.py
"""

import asyncio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from pydantic import BaseModel, Field
from typing import Literal
from agents import (
    Agent,
    Runner,
    trace,
    HostedMCPTool,
    WebSearchTool,
    AgentHooks,
)

MCP_SERVER_URL = "http://localhost:10000/api/mcp"
MCP_SERVER_URL = "https://stock-mcp-po2g.onrender.com/api/mcp"


# Specific data models for strict JSON schema compliance
class PriceData(BaseModel):
    """Price and technical data"""

    current_price: float
    change_percent: float
    trend: Literal["uptrend", "downtrend", "sideways"]
    support_level: float
    resistance_level: float
    volume_trend: Literal["increasing", "decreasing", "stable"]
    recent_high: float
    recent_low: float


class FundamentalData(BaseModel):
    """Fundamental financial data"""

    market_cap: float
    pe_ratio: float
    profit_margin: float
    revenue_growth: float
    debt_to_equity: float
    sector: str
    industry: str
    valuation_assessment: Literal["undervalued", "fairly valued", "overvalued"]

class SentimentData(BaseModel):
    """Sentiment and market opinion data"""

    news_sentiment: Literal["bullish", "neutral", "bearish"]
    top_headlines: list[str]
    analyst_rating_avg: Literal["strong buy", "buy", "hold", "sell", "strong sell"]
    target_price_avg: float
    insider_activity: Literal["buying", "selling", "neutral", "none"]
    overall_sentiment_score: float


class EventData(BaseModel):
    """Events and catalysts"""

    next_earnings_date: str
    days_to_earnings: int
    recent_rating_change: str
    upcoming_catalysts: list[str]
    catalyst_risk_level: Literal["low", "medium", "high"]


class KeyMetric(BaseModel):
    """Single key metric entry"""

    name: str
    value: str | float | int


# Hooks for monitoring
class StockHooks(AgentHooks):
    def on_run_start(self, context):
        print(f"  🚀 {context.agent.name} starting...")

    def on_run_end(self, context, result):
        print(f"  ✅ {context.agent.name} completed")

    def on_tool_call(self, context, tool_name, arguments):
        print(f"    🔧 {tool_name}")


# MCP and Web tools
yfinance_mcp = HostedMCPTool(
    tool_config={
        "type": "mcp",
        "server_label": "yfinance",
        "server_url": MCP_SERVER_URL,
        "require_approval": "never",
    }
)
# web_search = WebSearchTool()


# Specialized data collector agents (run in parallel)
price_agent = Agent(
    name="price_collector",
    instructions="""Get current price and technicals.
Use quote() and history().
Be precise with numbers.""",
    tools=[yfinance_mcp],
    output_type=PriceData,
    hooks=StockHooks(),
    model="gpt-5",
)

fundamentals_agent = Agent(
    name="fundamentals_collector",
    instructions="""Get fundamentals.
Use fundamentals() and financials().
If data missing, use 0.0 or "Unknown".""",
    tools=[yfinance_mcp],
    output_type=FundamentalData,
    hooks=StockHooks(),
    model="gpt-5",
)

sentiment_agent = Agent(
    name="sentiment_collector",
    instructions="""Get sentiment.
Use news(), analyst_recommendations(), get_insider_transactions().
Assess overall sentiment 0-10.""",
    tools=[yfinance_mcp],
    output_type=SentimentData,
    hooks=StockHooks(),
    model="gpt-5",
)

events_agent = Agent(
    name="events_collector",
    instructions="""Get events.
Use calendar(), get_upgrades_downgrades().
Identify next earnings and catalysts.""",
    tools=[yfinance_mcp],
    output_type=EventData,
    hooks=StockHooks(),
    model="gpt-5",
)


# Synthesis models
class StockAnalysis(BaseModel):
    """Comprehensive analysis"""

    symbol: str
    current_price: float = Field(gt=0)
    fundamental_score: float = Field(ge=0, le=10)
    technical_score: float = Field(ge=0, le=10)
    sentiment_score: float = Field(ge=0, le=10)
    risk_level: Literal["low", "medium", "high", "extreme"]
    catalysts: list[str]
    risks: list[str]
    summary: str


class FinalRating(BaseModel):
    """Investment rating"""

    rating: Literal["strong_buy", "buy", "hold", "sell", "strong_sell"]
    confidence: float = Field(ge=1, le=10)
    target_price: float = Field(gt=0)
    time_horizon: Literal["short_term", "medium_term", "long_term"]
    reasoning: str
    key_metrics: list[KeyMetric]


# Synthesis agent
analyst_agent = Agent(
    name="analyst",
    instructions="""You're an expert stock analyst synthesizing multiple data sources.

INPUT: You'll receive 2-4 data sources (Price, Fundamentals, Sentiment, Events)

ANALYSIS REQUIREMENTS:

1. FUNDAMENTAL SCORE (0-10):
   - Valuation: P/E vs industry avg, PEG ratio
   - Profitability: Profit margins, ROE
   - Growth: Revenue & earnings growth rates
   - Balance sheet: Debt levels, current ratio
   Score 8-10: Excellent metrics across board
   Score 5-7: Good metrics, some concerns
   Score 0-4: Weak fundamentals or high risk

2. TECHNICAL SCORE (0-10):
   - Trend: Strong uptrend vs downtrend
   - Price levels: Near highs (+) or lows (-)
   - Volume: Increasing on up days (+)
   - Momentum: Recent performance
   Score 8-10: Strong uptrend, good momentum
   Score 5-7: Neutral or mixed signals
   Score 0-4: Downtrend or weak technicals

3. SENTIMENT SCORE (0-10):
   - News tone: Bullish vs bearish headlines
   - Analyst ratings: Buy > Hold > Sell
   - Insider activity: Buying (+) selling (-)
   - Rating changes: Recent upgrades (+)
   Score 8-10: Very bullish across indicators
   Score 5-7: Mixed or neutral sentiment
   Score 0-4: Negative sentiment indicators

4. RISK LEVEL:
   - LOW: Profitable, stable, investment grade
   - MEDIUM: Some volatility or debt concerns
   - HIGH: Unprofitable, high debt, or volatile
   - EXTREME: Penny stock, distressed, or speculative

5. CATALYSTS: List 3-5 specific upcoming events that could move price
6. RISKS: List 3-5 specific risk factors
7. SUMMARY: 2-3 sentences with key takeaways

Be objective. Use actual numbers from data. Don't make up information.""",
    output_type=StockAnalysis,
    hooks=StockHooks(),
    model="gpt-5",
)

# Rating agent
rating_agent = Agent(
    name="portfolio_manager",
    instructions="""You're a portfolio manager making final investment decisions.

INPUT: StockAnalysis with scores, risk level, catalysts, risks

RATING DECISION RULES:

1. STRONG BUY:
   - Total score > 24/30
   - All individual scores ≥ 7/10
   - Risk level: LOW
   - Clear catalysts present
   - Confidence: 8-10

2. BUY:
   - Total score 20-24/30
   - Fundamental score ≥ 6/10
   - Risk level: LOW or MEDIUM
   - Positive outlook
   - Confidence: 6-8

3. HOLD:
   - Total score 15-20/30
   - Mixed signals or fairly valued
   - Any risk level
   - Confidence: 4-6

4. SELL:
   - Total score 10-15/30 OR
   - Risk level: HIGH without clear catalysts
   - Fundamental concerns
   - Confidence: 6-8 (confident in sell)

5. STRONG SELL:
   - Total score < 10/30 OR
   - Risk level: EXTREME OR
   - Fundamental deterioration
   - Confidence: 8-10 (confident in sell)

CONFIDENCE CALCULATION (1-10 Scale):
- 9-10: Extremely confident. All data aligns perfectly.
- 7-8: Very confident. Most data aligns, minor discrepancies.
- 5-6: Moderately confident. Mixed signals or missing some data.
- 3-4: Low confidence. Conflicting data or major uncertainties.
- 1-2: No confidence. Insufficient or contradictory data.

TARGET PRICE:
- Use current price + reasonable % based on rating
- Consider analyst consensus if available
- Apply P/E multiples or growth projections
- Be realistic, not overly optimistic

TIME HORIZON:
- short_term: Technical setup, momentum play (1-3 months)
- medium_term: Growth story, catalyst event (3-12 months)
- long_term: Value investment, fundamental play (1-3 years)

KEY METRICS:
Include: P/E, Market Cap, Growth Rate, Debt/Equity, Profit Margin, Target Price vs Current

REASONING:
Explain rating in 2-3 sentences citing specific scores, risks, and outlook.

Be decisive but honest about uncertainty.""",
    output_type=FinalRating,
    hooks=StockHooks(),
)


async def main():
    print("🤖 V3: Parallelized Stock Rating Workflow")
    print("=" * 60)

    symbol = input("\nEnter stock symbol: ").upper().strip()

    if not symbol or len(symbol) > 5:
        print("❌ Invalid symbol")
        return

    print(f"\n📊 Analyzing {symbol} with parallel data collection...")
    print("=" * 60)

    with trace(f"Parallel Stock Rating: {symbol}"):
        # PHASE 1: Parallel data collection (3x faster!)
        print("\n📈 Phase 1: Parallel Data Collection")

        # Run 4 data collectors in parallel
        price_task = Runner.run(price_agent, f"Collect price data for {symbol}")
        fundamentals_task = Runner.run(
            fundamentals_agent, f"Collect fundamentals for {symbol}"
        )
        sentiment_task = Runner.run(sentiment_agent, f"Collect sentiment for {symbol}")
        events_task = Runner.run(events_agent, f"Collect events for {symbol}")

        # Wait for all to complete
        results = await asyncio.gather(
            price_task,
            fundamentals_task,
            sentiment_task,
            events_task,
            return_exceptions=True,
        )

        # Check for errors
        collected_data = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"  ⚠️  Data collector {i + 1} failed: {result}")
            else:
                collected_data.append(result.final_output)

        if len(collected_data) < 2:
            print("\n❌ Insufficient data collected. Aborting.")
            return

        print(f"\n✅ Collected {len(collected_data)}/4 data sources")

        # PHASE 2: Synthesize analysis
        print("\n📊 Phase 2: Analysis Synthesis")

        # Combine all data for analyst
        all_data = "\n\n".join(
            [
                f"{ix + 1}. {data.__class__.__name__}: {data.model_dump_json()}"
                for ix, data in enumerate(collected_data)
            ]
        )

        analysis_prompt = f"""Analyze {symbol} using this collected data:

{all_data}

Synthesize comprehensive analysis with scores and insights."""

        analysis_result = await Runner.run(analyst_agent, analysis_prompt)
        analysis = analysis_result.final_output

        print(f"\n  Current Price: ${analysis.current_price:.2f}")
        print(
            f"  Scores: F:{analysis.fundamental_score}/10 T:{analysis.technical_score}/10 S:{analysis.sentiment_score}/10"
        )
        print(f"  Risk: {analysis.risk_level.upper()}")

        # Quality gate
        total = (
            analysis.fundamental_score
            + analysis.technical_score
            + analysis.sentiment_score
        )
        if total < 5:
            print(f"\n⚠️  Low total score ({total}/30). Proceed?")
            if input("(y/n): ").lower() != "y":
                return

        # PHASE 3: Generate rating
        print("\n⭐ Phase 3: Rating Generation")

        rating_result = await Runner.run(rating_agent, str(analysis))
        rating = rating_result.final_output

        # Display results
        print("\n" + "=" * 60)
        print(f"📋 RATING: {symbol}")
        print("=" * 60)
        print(f"🎯 {rating.rating.upper().replace('_', ' ')}")
        print(f"📊 Confidence: {rating.confidence:.0%}")
        print(f"💰 Target: ${rating.target_price:.2f}")
        print(f"⏰ Horizon: {rating.time_horizon.replace('_', ' ').title()}")
        print(f"\n💭 {rating.reasoning}")

        print(f"\n📈 Key Metrics:")
        for metric in rating.key_metrics:
            print(f"  {metric.name}: {metric.value}")

        print(f"\n🎯 Catalysts:")
        for c in analysis.catalysts[:3]:
            print(f"  • {c}")

        print(f"\n⚠️  Risks:")
        for r in analysis.risks[:3]:
            print(f"  • {r}")

        print("\n" + "=" * 60)

        # Export
        if input("\nExport? (y/n): ").lower() == "y":
            import json
            from datetime import datetime

            report = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "data_sources": len(collected_data),
                "analysis": analysis.model_dump(),
                "rating": rating.model_dump(),
            }

            filename = (
                f"{symbol}_parallel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(filename, "w") as f:
                json.dump(report, f, indent=2)
            print(f"✅ Saved to {filename}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Cancelled")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
