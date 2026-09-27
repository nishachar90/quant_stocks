import pandas as pd

from src.analysis.stock_analysis import analyze_stock


def analyze_stock_universe(
    symbols,
    market_symbol,
    start_date,
    end_date,
):
    results = []
    failed_stocks = []

    for symbol in symbols:
        try:
            result = analyze_stock(
                symbol,
                market_symbol,
                start_date,
                end_date,
            )

            results.append(result)

        except Exception as error:
            failed_stocks.append(
                {
                    "symbol": symbol,
                    "error": str(error),
                }
            )

    return results, failed_stocks


def results_to_dataframe(results):
    rows = []

    for result in results:
        row = {
            "symbol": result["symbol"],
            "start_date": result["start_date"],
            "end_date": result["end_date"],

            "total_return": result["performance"]["total_return"],
            "cagr": result["performance"]["cagr"],

            "annualized_volatility": (
                result["risk"]["annualized_volatility"]
            ),
            "maximum_drawdown": (
                result["risk"]["maximum_drawdown"]
            ),
            "sharpe_ratio": (
                result["risk"]["sharpe_ratio"]
            ),
            "sortino_ratio": (
                result["risk"]["sortino_ratio"]
            ),

            "beta": (
                result["market_relationship"]["beta"]
            ),
            "correlation": (
                result["market_relationship"]["correlation"]
            ),

            "net_profit_margin": (
                result["fundamentals"]["net_profit_margin"]
            ),
            "operating_margin": (
                result["fundamentals"]["operating_margin"]
            ),
            "roe": (
                result["fundamentals"]["roe"]
            ),
            "debt_to_equity": (
                result["fundamentals"]["debt_to_equity"]
            ),
            "free_cash_flow": (
                result["fundamentals"]["free_cash_flow"]
            ),
            "revenue_growth": (
                result["fundamentals"]["revenue_growth"]
            ),
        }

        rows.append(row)

    return pd.DataFrame(rows)
