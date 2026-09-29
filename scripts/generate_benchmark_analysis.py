# scripts/generate_benchmark_analysis.py

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.analysis.benchmark_analysis import (
    BenchmarkAnalysisConfig,
    calculate_benchmark_metrics,
    calculate_comparison_metrics,
    generate_comparison_summary,
    load_benchmark_data,
    print_benchmark_comparison,
)
from src.analysis.portfolio_backtest import (
    PortfolioBacktestConfig,
    calculate_backtest_metrics,
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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "nifty50_portfolio_benchmark_comparison.csv"
)


def load_backtest(
    path: Path,
) -> pd.DataFrame:
    return pd.read_csv(
        path,
        index_col="Date",
        parse_dates=True,
    )


def main() -> None:
    print("=" * 70)
    print("NIFTY 50 PORTFOLIO BENCHMARK ANALYSIS")
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

    backtest_config = PortfolioBacktestConfig(
        initial_capital=1_000_000.0,
        trading_days=252,
        risk_free_rate=0.06,
    )

    benchmark_config = BenchmarkAnalysisConfig(
        benchmark_symbol="^NSEI",
        risk_free_rate=0.06,
        trading_days=252,
    )

    benchmark = load_benchmark_data(
        start_date=start_date,
        end_date=end_date,
        cache_dir=PROJECT_ROOT
        / "data"
        / "cache",
        benchmark_symbol=benchmark_config.benchmark_symbol,
    )

    benchmark_metrics = (
        calculate_benchmark_metrics(
            benchmark,
            benchmark_config,
        )
    )

    rows = []

    strategies = [
        (
            "buy_and_hold",
            BACKTEST_FILE,
        ),
        (
            "rebalanced",
            REBALANCED_FILE,
        ),
    ]

    for strategy_name, backtest_file in strategies:

        backtest = load_backtest(
            backtest_file
        )

        portfolio_metrics = (
            calculate_backtest_metrics(
                backtest,
                backtest_config,
            )
        )

        comparison_metrics = (
            calculate_comparison_metrics(
                backtest,
                benchmark,
                benchmark_config,
            )
        )

        summary = generate_comparison_summary(
            strategy_name,
            portfolio_metrics,
            benchmark_metrics,
            comparison_metrics,
        )

        rows.append(summary)

    result = pd.concat(
        rows,
        ignore_index=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()

    print_benchmark_comparison(
        result
    )

    print()

    print(
        f"Output file: {OUTPUT_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
