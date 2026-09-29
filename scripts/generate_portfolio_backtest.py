# scripts/generate_portfolio_backtest.py

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.analysis.portfolio_backtest import (
    PortfolioBacktestConfig,
    calculate_backtest_metrics,
    calculate_buy_and_hold_backtest,
    calculate_rebalanced_backtest,
    generate_backtest_summary,
    load_price_data,
    print_backtest_audit,
)


PORTFOLIO_FILE = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "nifty50_portfolio.csv"
)

BACKTEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "nifty50_portfolio_backtest.csv"
)

REBALANCED_FILE = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "nifty50_portfolio_rebalanced_backtest.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "nifty50_portfolio_backtest_summary.csv"
)


def main() -> None:
    print("=" * 70)
    print("NIFTY 50 PORTFOLIO BACKTEST")
    print("=" * 70)

    portfolio = pd.read_csv(
        PORTFOLIO_FILE
    )

    selected = portfolio[
        portfolio["selected"] == True
    ].copy()

    if selected.empty:
        raise ValueError(
            "No selected holdings found."
        )

    start_date = str(
        selected["start_date"].iloc[0]
    )

    end_date = str(
        selected["end_date"].iloc[0]
    )

    symbols = selected[
        "symbol"
    ].tolist()

    print()
    print("INPUT")
    print(
        f"Holdings: {len(selected)}"
    )
    print(
        f"Start date: {start_date}"
    )
    print(
        f"End date: {end_date}"
    )

    prices = load_price_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        cache_dir=PROJECT_ROOT / "data" / "cache",
    )

    config = PortfolioBacktestConfig(
        initial_capital=1_000_000.0,
    )

    buy_and_hold = calculate_buy_and_hold_backtest(
        prices=prices,
        portfolio=selected,
        config=config,
    )

    rebalanced = calculate_rebalanced_backtest(
        prices=prices,
        portfolio=selected,
        config=config,
    )

    buy_and_hold_metrics = calculate_backtest_metrics(
        buy_and_hold,
        config,
    )

    rebalanced_metrics = calculate_backtest_metrics(
        rebalanced,
        config,
    )

    buy_and_hold.to_csv(
        BACKTEST_FILE,
        index=True,
        index_label="Date",
    )

    rebalanced.to_csv(
        REBALANCED_FILE,
        index=True,
        index_label="Date",
    )

    summary = pd.DataFrame(
        [
            {
                "strategy": "buy_and_hold",
                **buy_and_hold_metrics,
            },
            {
                "strategy": "rebalanced",
                **rebalanced_metrics,
            },
        ]
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print()
    print("BUY AND HOLD")
    print_backtest_audit(
        buy_and_hold,
        buy_and_hold_metrics,
    )

    print()
    print("REBALANCED")
    print_backtest_audit(
        rebalanced,
        rebalanced_metrics,
    )

    print()
    print(
        f"Backtest file: {BACKTEST_FILE}"
    )
    print(
        f"Rebalanced file: {REBALANCED_FILE}"
    )
    print(
        f"Summary file: {SUMMARY_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
