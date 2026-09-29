# scripts/generate_portfolio_visualizations.py

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.portfolio_charts import (
    generate_all_charts,
)


PORTFOLIO_FILE = (
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

BENCHMARK_FILE = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "^NSEI__2016-09-29__2026-09-27__adjusted.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "charts"
)


def main() -> None:
    result = generate_all_charts(
        portfolio_backtest_path=PORTFOLIO_FILE,
        rebalanced_backtest_path=REBALANCED_FILE,
        benchmark_path=BENCHMARK_FILE,
        output_dir=OUTPUT_DIR,
    )

    print("=" * 70)
    print("NIFTY 50 PORTFOLIO VISUALIZATION")
    print("=" * 70)

    print()

    for name, path in result.items():
        print(
            f"- {name}: {path}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
