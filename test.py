"""
Comprehensive test suite for yfinance MCP server.
Uses FastMCP client library for proper MCP testing.
"""

import asyncio
import argparse
import json
from fastmcp import Client

# Parse arguments
parser = argparse.ArgumentParser(description="Test yfinance MCP server")
parser.add_argument(
    "--local", action="store_true", help="Test local server (default: production)"
)
parser.add_argument(
    "--symbol", default="AAPL", help="Stock symbol to test with (default: AAPL)"
)
args = parser.parse_args()

# Configure server URL
baseurl = (
    "http://localhost:10000/api/mcp"
    if args.local
    else "https://stock-mcp-po2g.onrender.com/api/mcp"
)

print(f"🧪 Testing yfinance MCP server at: {baseurl}")
print(f"📊 Test symbol: {args.symbol}\n")

client = Client(baseurl)


def parse_result(result):
    """Parse FastMCP result into dict."""
    if hasattr(result, "content") and result.content:
        return json.loads(result.content[0].text)
    return result


async def test_server_connectivity():
    """Test basic server connectivity."""
    print("=" * 60)
    print("TEST: Server Connectivity")
    print("=" * 60)

    async with client:
        try:
            await client.ping()
            print("✅ Server is reachable")
            return True
        except Exception as e:
            print(f"❌ Server ping failed: {e}")
            return False


async def test_list_tools():
    """List all available tools."""
    print("\n" + "=" * 60)
    print("TEST: List Available Tools")
    print("=" * 60)

    async with client:
        try:
            tools = await client.list_tools()
            print(f"✅ Found {len(tools)} tools:")
            for tool in tools:
                desc = tool.description or "No description"
                print(f"  📌 {tool.name}: {desc[:80]}...")
            return True
        except Exception as e:
            print(f"❌ Failed to list tools: {e}")
            return False


async def test_quote(symbol: str):
    """Test quote endpoint."""
    print("\n" + "=" * 60)
    print(f"TEST: Quote for {symbol}")
    print("=" * 60)

    async with client:
        try:
            result = await client.call_tool("quote", {"symbol": symbol})
            data = parse_result(result)
            print("✅ Quote retrieved successfully")
            print(f"  💰 Price: ${data.get('price', 'N/A')}")
            print(f"  📈 Change: {data.get('changePercent', 0):.2f}%")
            print(f"  💵 Currency: {data.get('currency', 'N/A')}")
            return True
        except Exception as e:
            print(f"❌ Quote test failed: {e}")
            return False


async def test_history(symbol: str):
    """Test history endpoint."""
    print("\n" + "=" * 60)
    print(f"TEST: History for {symbol}")
    print("=" * 60)

    async with client:
        try:
            result = await client.call_tool(
                "history", {"symbol": symbol, "period": "5d", "interval": "1d"}
            )
            data = parse_result(result)
            rows = data.get("rows", [])
            print("✅ History retrieved successfully")
            print(f"  📊 Data points: {len(rows)}")
            if rows:
                latest = rows[-1]
                print(f"  📅 Latest: {latest.get('Date', 'N/A')}")
                print(f"  💵 Close: ${latest.get('Close', 'N/A')}")
            return True
        except Exception as e:
            print(f"❌ History test failed: {e}")
            return False


async def test_fundamentals(symbol: str):
    """Test fundamentals endpoint."""
    print("\n" + "=" * 60)
    print(f"TEST: Fundamentals for {symbol}")
    print("=" * 60)

    async with client:
        try:
            result = await client.call_tool("fundamentals", {"symbol": symbol})
            data = parse_result(result)
            info = data.get("info", {})
            print("✅ Fundamentals retrieved successfully")
            print(f"  🏢 Name: {info.get('shortName', 'N/A')}")
            print(f"  🏭 Sector: {info.get('sector', 'N/A')}")
            print(f"  📈 P/E Ratio: {info.get('trailingPE', 'N/A')}")
            print(
                f"  💼 Market Cap: ${info.get('marketCap', 0):,.0f}"
                if info.get("marketCap")
                else "  💼 Market Cap: N/A"
            )
            return True
        except Exception as e:
            print(f"❌ Fundamentals test failed: {e}")
            return False


async def test_options(symbol: str):
    """Test options endpoints."""
    print("\n" + "=" * 60)
    print(f"TEST: Options for {symbol}")
    print("=" * 60)

    async with client:
        try:
            # First get available expirations
            result = await client.call_tool("options_expirations", {"symbol": symbol})
            data = parse_result(result)
            expirations = data.get("expirations", [])
            print("✅ Options expirations retrieved")
            print(f"  📅 Available dates: {len(expirations)}")

            if expirations:
                # Test option chain for first expiration
                expiry = expirations[0]
                print(f"  🔍 Testing chain for: {expiry}")
                chain_result = await client.call_tool(
                    "option_chain", {"symbol": symbol, "expiration": expiry}
                )
                chain_data = parse_result(chain_result)
                calls = chain_data.get("calls", [])
                puts = chain_data.get("puts", [])
                print(f"  📞 Calls: {len(calls)}")
                print(f"  📉 Puts: {len(puts)}")
            return True
        except Exception as e:
            print(f"❌ Options test failed: {e}")
            return False


async def test_dividends_and_splits(symbol: str):
    """Test dividends, splits, and actions endpoints."""
    print("\n" + "=" * 60)
    print(f"TEST: Dividends & Splits for {symbol}")
    print("=" * 60)

    async with client:
        try:
            # Test dividends
            div_result = await client.call_tool("dividends", {"symbol": symbol})
            div_data = parse_result(div_result)
            dividends = div_data.get("dividends", [])
            print(f"✅ Dividends: {len(dividends)} payments found")

            # Test splits
            split_result = await client.call_tool("splits", {"symbol": symbol})
            split_data = parse_result(split_result)
            splits = split_data.get("splits", [])
            print(f"✅ Splits: {len(splits)} events found")

            # Test actions (combined)
            actions_result = await client.call_tool("actions", {"symbol": symbol})
            actions_data = parse_result(actions_result)
            actions = actions_data.get("actions", [])
            print(f"✅ Actions: {len(actions)} total events")

            return True
        except Exception as e:
            print(f"❌ Dividends & Splits test failed: {e}")
            return False


async def test_earnings(symbol: str):
    """Test earnings endpoints."""
    print("\n" + "=" * 60)
    print(f"TEST: Earnings for {symbol}")
    print("=" * 60)

    async with client:
        try:
            # Test earnings history
            hist_result = await client.call_tool("earnings_history", {"symbol": symbol})
            hist_data = parse_result(hist_result)
            earnings = hist_data.get("earnings", [])
            print(f"✅ Earnings History: {len(earnings)} periods")

            # Test earnings estimates
            est_result = await client.call_tool(
                "earnings_estimates", {"symbol": symbol}
            )
            est_data = parse_result(est_result)
            estimates = est_data.get("estimates", {})
            print(
                f"✅ Earnings Estimates: {'Available' if estimates else 'Not available'}"
            )

            return True
        except Exception as e:
            print(f"❌ Earnings test failed: {e}")
            return False


async def test_news_and_calendar(symbol: str):
    """Test news and calendar endpoints."""
    print("\n" + "=" * 60)
    print(f"TEST: News & Calendar for {symbol}")
    print("=" * 60)

    async with client:
        try:
            # Test news
            news_result = await client.call_tool("news", {"symbol": symbol})
            news_data = parse_result(news_result)
            news = news_data.get("news", [])
            print(f"✅ News: {len(news)} articles found")
            if news:
                print(f"  📰 Latest: {news[0].get('title', 'N/A')[:60]}...")

            # Test calendar
            cal_result = await client.call_tool("calendar", {"symbol": symbol})
            cal_data = parse_result(cal_result)
            calendar = cal_data.get("calendar", [])
            print(f"✅ Calendar: {len(calendar)} events")

            return True
        except Exception as e:
            print(f"❌ News & Calendar test failed: {e}")
            return False


async def run_all_tests():
    """Run comprehensive test suite."""
    print("\n🚀 Starting yfinance MCP Test Suite")
    print("=" * 60 + "\n")

    results = {}

    # Run all tests
    results["connectivity"] = await test_server_connectivity()
    results["list_tools"] = await test_list_tools()
    results["quote"] = await test_quote(args.symbol)
    results["history"] = await test_history(args.symbol)
    results["fundamentals"] = await test_fundamentals(args.symbol)
    results["options"] = await test_options(args.symbol)
    results["dividends"] = await test_dividends_and_splits(args.symbol)
    results["earnings"] = await test_earnings(args.symbol)
    results["news"] = await test_news_and_calendar(args.symbol)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check output above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    exit(exit_code)
