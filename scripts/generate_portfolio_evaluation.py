"""
Generate complete portfolio evaluation outputs.
"""

from pathlib import Path

import pandas as pd

from src.analysis.benchmark_analysis import analyze_benchmark
from src.analysis.performance_evaluation import evaluate_portfolio_performance
from src.analysis.portfolio_attribution import calculate_portfolio_attribution


BASE_DIR = Path(__file__).resolve().parents[1]

BACKTEST_FILE = (
    BASE_DIR
    / "data"
    / "analysis"
    / "nifty50_portfolio_backtest.csv"
)

BENCHMARK_FILE = (
    BASE_DIR
    / "data"
    / "analysis"
    / "nifty50_portfolio_benchmark_comparison.csv"
)

PORTFOLIO_FILE = (
    BASE_DIR
    / "data"
    / "analysis"
    / "nifty50_portfolio.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "analysis"


def main() -> None:
    backtest = pd.read_csv(BACKTEST_FILE)
    benchmark = pd.read_csv(BENCHMARK_FILE)
    portfolio = pd.read_csv(PORTFOLIO_FILE)

    performance = evaluate_portfolio_performance(
        backtest=backtest,
        benchmark=benchmark,
    )

    attribution = calculate_portfolio_attribution(
        portfolio=portfolio,
        backtest=backtest,
    )

    performance_path = (
        OUTPUT_DIR
        / "nifty50_portfolio_performance_evaluation.csv"
    )

    attribution_path = (
        OUTPUT_DIR
        / "nifty50_portfolio_attribution.csv"
    )

    pd.DataFrame([performance]).to_csv(
        performance_path,
        index=False,
    )

    attribution.to_csv(
        attribution_path,
        index=False,
    )

    print("=" * 70)
    print("PORTFOLIO EVALUATION")
    print("=" * 70)

    print()
    print("PERFORMANCE EVALUATION")
    print(pd.DataFrame([performance]).to_string(index=False))

    print()
    print("ATTRIBUTION")
    print(attribution.to_string(index=False))

    print()
    print("OUTPUTS")
    print(f"- {performance_path}")
    print(f"- {attribution_path}")

    print()
    print("PORTFOLIO EVALUATION: COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
