# scripts/generate_portfolio.py

from pathlib import Path

import pandas as pd

from src.analysis.portfolio_construction import (
    PortfolioConstructionConfig,
    add_portfolio_columns,
    construct_portfolio,
    print_portfolio_audit,
)


INPUT_FILE = Path(
    "data/analysis/nifty50_eligible.csv"
)

PORTFOLIO_FILE = Path(
    "data/analysis/nifty50_portfolio.csv"
)

HOLDINGS_FILE = Path(
    "data/analysis/nifty50_holdings.csv"
)


def main() -> None:
    print("=" * 70)
    print("NIFTY 50 PORTFOLIO CONSTRUCTION")
    print("=" * 70)

    dataset = pd.read_csv(INPUT_FILE)

    if "eligible" not in dataset.columns:
        raise ValueError(
            "Input dataset must contain the 'eligible' column."
        )

    dataset = dataset.rename(
        columns={"eligible": "eligibility"}
    )

    config = PortfolioConstructionConfig()

    portfolio = construct_portfolio(
        dataset,
        config,
    )

    full_dataset = add_portfolio_columns(
        dataset,
        portfolio,
    )

    PORTFOLIO_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_dataset.to_csv(
        PORTFOLIO_FILE,
        index=False,
    )

    portfolio.to_csv(
        HOLDINGS_FILE,
        index=False,
    )

    print()
    print_portfolio_audit(
        portfolio,
        config,
    )

    print()
    print(f"Portfolio file: {PORTFOLIO_FILE}")
    print(f"Holdings file: {HOLDINGS_FILE}")

    print("=" * 70)


if __name__ == "__main__":
    main()
