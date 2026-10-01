"""
Generate complete NIFTY 50 portfolio evaluation outputs.
"""

from pathlib import Path

import pandas as pd

from src.analysis.performance_evaluation import (
    evaluate_performance,
)
from src.analysis.portfolio_attribution import (
    calculate_peer_group_attribution,
    calculate_security_attribution,
    calculate_sector_attribution,
)


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

CLASSIFICATION_FILE = (
    BASE_DIR
    / "data"
    / "reference"
    / "nifty50_classification.csv"
)

PEER_GROUP_FILE = (
    BASE_DIR
    / "data"
    / "reference"
    / "peer_groups.csv"
)


OUTPUT_DIR = BASE_DIR / "data" / "analysis"


PERFORMANCE_OUTPUT_FILE = (
    OUTPUT_DIR
    / "nifty50_portfolio_performance_evaluation.csv"
)

SECURITY_ATTRIBUTION_OUTPUT_FILE = (
    OUTPUT_DIR
    / "nifty50_portfolio_security_attribution.csv"
)

SECTOR_ATTRIBUTION_OUTPUT_FILE = (
    OUTPUT_DIR
    / "nifty50_portfolio_sector_attribution.csv"
)

PEER_GROUP_ATTRIBUTION_OUTPUT_FILE = (
    OUTPUT_DIR
    / "nifty50_portfolio_peer_group_attribution.csv"
)


def load_backtest() -> pd.DataFrame:
    """Load portfolio backtest data."""

    if not BACKTEST_FILE.exists():
        raise FileNotFoundError(
            f"Backtest file not found: {BACKTEST_FILE}"
        )

    backtest = pd.read_csv(BACKTEST_FILE)

    if "Date" in backtest.columns and "date" not in backtest.columns:
        backtest = backtest.rename(
            columns={"Date": "date"}
        )

    if "date" not in backtest.columns:
        raise ValueError(
            "Backtest must contain a date column."
        )

    backtest["date"] = pd.to_datetime(
        backtest["date"]
    )

    return backtest


def load_benchmark() -> pd.DataFrame:
    """Load benchmark comparison data."""

    if not BENCHMARK_FILE.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {BENCHMARK_FILE}"
        )

    return pd.read_csv(BENCHMARK_FILE)


def load_portfolio() -> pd.DataFrame:
    """Load portfolio holdings."""

    if not PORTFOLIO_FILE.exists():
        raise FileNotFoundError(
            f"Portfolio file not found: {PORTFOLIO_FILE}"
        )

    return pd.read_csv(PORTFOLIO_FILE)


def load_classification() -> pd.DataFrame:
    """Load classification data and attach peer groups."""

    if not CLASSIFICATION_FILE.exists():
        raise FileNotFoundError(
            f"Classification file not found: {CLASSIFICATION_FILE}"
        )

    if not PEER_GROUP_FILE.exists():
        raise FileNotFoundError(
            f"Peer-group file not found: {PEER_GROUP_FILE}"
        )

    classification = pd.read_csv(
        CLASSIFICATION_FILE
    )

    peer_groups = pd.read_csv(
        PEER_GROUP_FILE
    )

    required_classification_columns = {
        "symbol",
        "sector",
        "business_type",
    }

    missing_classification_columns = (
        required_classification_columns
        - set(classification.columns)
    )

    if missing_classification_columns:
        raise ValueError(
            "classification is missing required columns: "
            f"{sorted(missing_classification_columns)}"
        )

    required_peer_group_columns = {
        "business_type",
        "peer_group",
    }

    missing_peer_group_columns = (
        required_peer_group_columns
        - set(peer_groups.columns)
    )

    if missing_peer_group_columns:
        raise ValueError(
            "peer_groups is missing required columns: "
            f"{sorted(missing_peer_group_columns)}"
        )

    classification = classification.merge(
        peer_groups,
        on="business_type",
        how="left",
        validate="many_to_one",
    )

    if classification["peer_group"].isna().any():
        missing = (
            classification.loc[
                classification["peer_group"].isna(),
                "business_type",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Missing peer-group mappings for business types: "
            f"{missing}"
        )

    return classification


def build_security_returns(
    portfolio: pd.DataFrame,
    backtest: pd.DataFrame,
) -> pd.Series:
    """Calculate total-period return for every portfolio security."""

    required_columns = {
        "symbol",
        "weight",
    }

    missing = (
        required_columns
        - set(portfolio.columns)
    )

    if missing:
        raise ValueError(
            "Portfolio is missing required columns: "
            f"{sorted(missing)}"
        )

    if "date" not in backtest.columns:
        raise ValueError(
            "Backtest must contain a date column."
        )

    security_returns = {}

    for symbol in portfolio["symbol"]:
        value_column = f"value_{symbol}.NS"

        if value_column not in backtest.columns:
            raise ValueError(
                f"Backtest is missing value column for "
                f"{symbol}: {value_column}"
            )

        values = pd.to_numeric(
            backtest[value_column],
            errors="coerce",
        ).dropna()

        if len(values) < 2:
            raise ValueError(
                f"Insufficient backtest data for {symbol}."
            )

        first_value = float(
            values.iloc[0]
        )

        last_value = float(
            values.iloc[-1]
        )

        if first_value <= 0:
            raise ValueError(
                f"Invalid starting value for {symbol}."
            )

        security_returns[str(symbol)] = (
            last_value / first_value
        ) - 1.0

    return pd.Series(
        security_returns,
        dtype=float,
    )


def evaluate_strategy(
    backtest: pd.DataFrame,
    benchmark: pd.DataFrame,
    portfolio: pd.DataFrame,
    classification: pd.DataFrame,
    strategy: str,
):
    """Evaluate one portfolio strategy."""

    strategy_backtest = backtest.copy()

    if "strategy" in strategy_backtest.columns:
        strategy_backtest = strategy_backtest[
            strategy_backtest["strategy"] == strategy
        ].copy()

    if strategy_backtest.empty:
        raise ValueError(
            f"No backtest data found for strategy: {strategy}"
        )

    performance = evaluate_performance(
        portfolio=strategy_backtest,
        benchmark=benchmark,
    )

    security_returns = build_security_returns(
        portfolio=portfolio,
        backtest=strategy_backtest,
    )

    security_attribution = calculate_security_attribution(
        portfolio=portfolio,
        classification=classification,
        security_returns=security_returns,
    )

    sector_attribution = calculate_sector_attribution(
        portfolio=portfolio,
        classification=classification,
        security_returns=security_returns,
    )

    peer_group_attribution = calculate_peer_group_attribution(
        portfolio=portfolio,
        classification=classification,
        security_returns=security_returns,
    )

    return (
        pd.DataFrame([performance]),
        security_attribution,
        sector_attribution,
        peer_group_attribution,
    )


def main() -> None:
    """Run the complete portfolio evaluation pipeline."""

    print("=" * 70)
    print("NIFTY 50 PORTFOLIO EVALUATION")
    print("=" * 70)

    backtest = load_backtest()
    benchmark = load_benchmark()
    portfolio = load_portfolio()
    classification = load_classification()

    print()
    print("INPUT")
    print(f"Holdings: {len(portfolio)}")
    print(f"Start date: {backtest['date'].min()}")
    print(f"End date: {backtest['date'].max()}")

    print()

    strategies = [
        "buy_and_hold",
        "rebalanced",
    ]

    performance_outputs = []
    security_outputs = []
    sector_outputs = []
    peer_group_outputs = []

    for strategy in strategies:
        print(f"Evaluating: {strategy}")

        (
            performance,
            security_attribution,
            sector_attribution,
            peer_group_attribution,
        ) = evaluate_strategy(
            backtest=backtest,
            benchmark=benchmark,
            portfolio=portfolio,
            classification=classification,
            strategy=strategy,
        )

        performance["strategy"] = strategy

        security_attribution["strategy"] = strategy

        sector_attribution["strategy"] = strategy

        peer_group_attribution["strategy"] = strategy

        performance_outputs.append(
            performance
        )

        security_outputs.append(
            security_attribution
        )

        sector_outputs.append(
            sector_attribution
        )

        peer_group_outputs.append(
            peer_group_attribution
        )

    performance_output = pd.concat(
        performance_outputs,
        ignore_index=True,
    )

    security_output = pd.concat(
        security_outputs,
        ignore_index=True,
    )

    sector_output = pd.concat(
        sector_outputs,
        ignore_index=True,
    )

    peer_group_output = pd.concat(
        peer_group_outputs,
        ignore_index=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    performance_output.to_csv(
        PERFORMANCE_OUTPUT_FILE,
        index=False,
    )

    security_output.to_csv(
        SECURITY_ATTRIBUTION_OUTPUT_FILE,
        index=False,
    )

    sector_output.to_csv(
        SECTOR_ATTRIBUTION_OUTPUT_FILE,
        index=False,
    )

    peer_group_output.to_csv(
        PEER_GROUP_ATTRIBUTION_OUTPUT_FILE,
        index=False,
    )

    print()
    print("OUTPUTS")
    print(
        f"- {PERFORMANCE_OUTPUT_FILE}"
    )
    print(
        f"- {SECURITY_ATTRIBUTION_OUTPUT_FILE}"
    )
    print(
        f"- {SECTOR_ATTRIBUTION_OUTPUT_FILE}"
    )
    print(
        f"- {PEER_GROUP_ATTRIBUTION_OUTPUT_FILE}"
    )

    print()
    print("PERFORMANCE")
    print(
        performance_output.to_string(
            index=False
        )
    )

    print()
    print("SECTOR ATTRIBUTION")
    print(
        sector_output.to_string(
            index=False
        )
    )

    print()
    print("PEER GROUP ATTRIBUTION")
    print(
        peer_group_output.to_string(
            index=False
        )
    )

    print()
    print("PORTFOLIO EVALUATION: COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
