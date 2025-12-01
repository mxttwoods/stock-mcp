"""
Quick test script for the sharpe_ratio endpoint.
"""

import sys

sys.path.insert(0, "/Users/matthewwoods/Development/stock-mcp")

from main import sharpe_ratio


def test_sharpe_ratio():
    """Test the Sharpe ratio calculation for a few stocks."""

    test_symbols = ["AAPL", "MSFT", "NVDA"]

    print("Testing Sharpe Ratio Endpoint")
    print("=" * 60)

    for symbol in test_symbols:
        print(f"\n{symbol}:")
        result = sharpe_ratio(symbol, period="1y", risk_free_rate=0.04)

        print(f"  Sharpe Ratio: {result.get('sharpe_ratio')}")
        print(f"  Annualized Return: {result.get('annualized_return')}%")
        print(f"  Annualized Volatility: {result.get('annualized_volatility')}%")
        print(f"  Total Return: {result.get('total_return')}%")
        print(f"  Days Analyzed: {result.get('days_analyzed')}")

        if result.get("error"):
            print(f"  ERROR: {result.get('error')}")


if __name__ == "__main__":
    test_sharpe_ratio()
