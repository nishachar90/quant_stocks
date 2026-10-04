"""
Generate complete NIFTY 50 portfolio evaluation outputs.
"""

from pathlib import Path

import pandas as pd

from src.analysis.performance_evaluation import evaluate_performance
from src.analysis.portfolio_attribution import (
    calculate_peer_group_attribution,
    calculate_security_attribution,
    calculate_sector_attribution,
)


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
ANALYSIS_DIR = DATA_DIR / "analysis"
REFERENCE_DIR = DATA_DIR / "reference"
CACHE_DIR = DATA_DIR / "cache"


BACKTEST_FILES = {
    "buy_and_hold": (
        ANALYSIS_DIR
        / "nifty50_portfolio_backtest.csv"
    ),
    "rebalanced": (
        ANALYSIS_DIR
        / "nifty50_portfolio_rebalanced_backtest.csv"
    ),
}


PORTFOLIO_FILE = (
    ANALYSIS_DIR
    / "nifty50_portfolio.csv"
)

CLASSIFICATION_FILE = (
    REFERENCE_DIR
    / "nifty50_classification.csv"
)

PEER_GROUP_FILE = (
    REFERENCE_DIR
    / "peer_groups.csv"
)


BENCHMARK_SYMBOL = "^NSEI"

BENCHMARK_CACHE_FILE = (
    CACHE_DIR
    / "^NSEI__2016-09-29__2026-09-27__adjusted.csv"
)


PERFORMANCE_OUTPUT_FILE = (
    ANALYSIS_DIR
    / "nifty50_portfolio_performance_evaluation.csv"
)

SECURITY_ATTRIBUTION_OUTPUT_FILE = (
    ANALYSIS_DIR
    / "nifty50_portfolio_security_attribution.csv"
)

SECTOR_ATTRIBUTION_OUTPUT_FILE = (
    ANALYSIS_DIR
    / "nifty50_portfolio_sector_attribution.csv"
)

PEER_GROUP_ATTRIBUTION_OUTPUT_FILE = (
    ANALYSIS_DIR
    / "nifty50_portfolio_peer_group_attribution.csv"
)


def load_backtest(
    strategy: str = "buy_and_hold",
) -> pd.DataFrame:
    """Load portfolio backtest data for one strategy."""

    if strategy not in BACKTEST_FILES:
        raise ValueError(
            f"Unknown strategy: {strategy}"
        )

    backtest_file = BACKTEST_FILES[strategy]

    if not backtest_file.exists():
        raise FileNotFoundError(
            f"Backtest file not found: {backtest_file}"
        )

    backtest = pd.read_csv(
        backtest_file,
        index_col=0,
        parse_dates=True,
    )

    if backtest.empty:
        raise ValueError(
            f"Backtest file is empty: {backtest_file}"
        )

    required_columns = {
        "portfolio_value",
        "daily_return",
        "cumulative_return",
        "drawdown",
    }

    missing_columns = (
        required_columns
        - set(backtest.columns)
    )

    if missing_columns:
        raise ValueError(
            "Backtest is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if not isinstance(
        backtest.index,
        pd.DatetimeIndex,
    ):
        raise ValueError(
            "Backtest index must be a DatetimeIndex."
        )

    backtest.index = pd.to_datetime(
        backtest.index
    )

    if backtest.index.tz is not None:
        backtest.index = (
            backtest.index.tz_localize(None)
        )

    return backtest


def load_benchmark(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Load and slice the existing full-range NIFTY 50 benchmark cache."""

    if not BENCHMARK_CACHE_FILE.exists():
        raise FileNotFoundError(
            f"Benchmark cache not found: "
            f"{BENCHMARK_CACHE_FILE}"
        )

    benchmark = pd.read_csv(
        BENCHMARK_CACHE_FILE,
        index_col=0,
        parse_dates=True,
    )

    if benchmark.empty:
        raise ValueError(
            "Benchmark data cannot be empty."
        )

    if "Close" not in benchmark.columns:
        raise ValueError(
            "Benchmark must contain a Close column."
        )

    benchmark.index = pd.to_datetime(
        benchmark.index
    )

    if benchmark.index.tz is not None:
        benchmark.index = (
            benchmark.index.tz_localize(None)
        )

    start_date = pd.Timestamp(
        start_date
    ).normalize()

    end_date = pd.Timestamp(
        end_date
    ).normalize()

    benchmark = benchmark.loc[
        (benchmark.index >= start_date)
        & (benchmark.index <= end_date)
    ].copy()

    if benchmark.empty:
        raise ValueError(
            "No benchmark data available for the "
            f"requested date range: "
            f"{start_date.date()} to {end_date.date()}"
        )

    return benchmark


def load_portfolio() -> pd.DataFrame:
    """Load the selected portfolio holdings."""

    if not PORTFOLIO_FILE.exists():
        raise FileNotFoundError(
            f"Portfolio file not found: {PORTFOLIO_FILE}"
        )

    portfolio = pd.read_csv(
        PORTFOLIO_FILE
    )

    if portfolio.empty:
        raise ValueError(
            "Portfolio cannot be empty."
        )

    required_columns = {
        "symbol",
        "weight",
        "selected",
    }

    missing_columns = (
        required_columns
        - set(portfolio.columns)
    )

    if missing_columns:
        raise ValueError(
            "Portfolio is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    portfolio["symbol"] = (
        portfolio["symbol"]
        .astype(str)
        .str.strip()
    )

    selected_values = (
        portfolio["selected"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    portfolio = portfolio.loc[
        selected_values.isin(
            {"true", "1", "yes"}
        )
    ].copy()

    if portfolio.empty:
        raise ValueError(
            "Portfolio contains no selected holdings."
        )

    portfolio["weight"] = pd.to_numeric(
        portfolio["weight"],
        errors="coerce",
    )

    if portfolio["weight"].isna().any():
        raise ValueError(
            "Selected portfolio contains invalid weights."
        )

    if (portfolio["weight"] <= 0).any():
        raise ValueError(
            "Selected portfolio weights must be positive."
        )

    duplicate_symbols = (
        portfolio["symbol"]
        .duplicated()
    )

    if duplicate_symbols.any():
        duplicates = (
            portfolio.loc[
                duplicate_symbols,
                "symbol",
            ]
            .tolist()
        )

        raise ValueError(
            "Selected portfolio contains duplicate symbols: "
            f"{duplicates}"
        )

    return portfolio


def load_classification() -> pd.DataFrame:
    """Load classification data and attach peer groups."""

    if not CLASSIFICATION_FILE.exists():
        raise FileNotFoundError(
            f"Classification file not found: "
            f"{CLASSIFICATION_FILE}"
        )

    if not PEER_GROUP_FILE.exists():
        raise FileNotFoundError(
            f"Peer-group file not found: "
            f"{PEER_GROUP_FILE}"
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
            "Classification is missing required columns: "
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
            "Peer groups are missing required columns: "
            f"{sorted(missing_peer_group_columns)}"
        )

    classification["symbol"] = (
        classification["symbol"]
        .astype(str)
        .str.strip()
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
    """Calculate total-period return for each portfolio security."""

    security_returns = {}

    for symbol in portfolio["symbol"]:
        symbol = str(symbol).strip()

        value_column = f"value_{symbol}"

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

        security_returns[symbol] = (
            last_value / first_value
        ) - 1.0

    return pd.Series(
        security_returns,
        dtype=float,
    )


def evaluate_strategy(
    backtest: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate portfolio performance for one strategy."""

    performance = evaluate_performance(
        portfolio=backtest,
        benchmark=benchmark,
    )

    return pd.DataFrame(
        [performance]
    )


def calculate_attribution_outputs(
    portfolio: pd.DataFrame,
    classification: pd.DataFrame,
    backtest: pd.DataFrame,
):
    """Calculate attribution outputs for the buy-and-hold strategy."""

    security_returns = build_security_returns(
        portfolio=portfolio,
        backtest=backtest,
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
        security_attribution,
        sector_attribution,
        peer_group_attribution,
    )


def main() -> None:
    """Run the complete portfolio evaluation pipeline."""

    print("=" * 70)
    print("NIFTY 50 PORTFOLIO EVALUATION")
    print("=" * 70)

    portfolio = load_portfolio()

    classification = load_classification()

    buy_and_hold_backtest = load_backtest(
        strategy="buy_and_hold"
    )

    start_date = buy_and_hold_backtest.index.min()
    end_date = buy_and_hold_backtest.index.max()

    benchmark = load_benchmark(
        start_date=start_date,
        end_date=end_date,
    )

    print()
    print("INPUT")
    print(f"Holdings: {len(portfolio)}")
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")

    performance_outputs = []

    for strategy in BACKTEST_FILES:
        backtest = load_backtest(
            strategy=strategy
        )

        print()
        print(f"Evaluating: {strategy}")

        performance = evaluate_strategy(
            backtest=backtest,
            benchmark=benchmark,
        )

        performance["strategy"] = strategy

        performance_outputs.append(
            performance
        )

    (
        security_attribution,
        sector_attribution,
        peer_group_attribution,
    ) = calculate_attribution_outputs(
        portfolio=portfolio,
        classification=classification,
        backtest=buy_and_hold_backtest,
    )

    security_attribution["strategy"] = "buy_and_hold"
    sector_attribution["strategy"] = "buy_and_hold"
    peer_group_attribution["strategy"] = "buy_and_hold"

    performance_output = pd.concat(
        performance_outputs,
        ignore_index=True,
    )

    OUTPUT_DIR = ANALYSIS_DIR

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    performance_output.to_csv(
        PERFORMANCE_OUTPUT_FILE,
        index=False,
    )

    security_attribution.to_csv(
        SECURITY_ATTRIBUTION_OUTPUT_FILE,
        index=False,
    )

    sector_attribution.to_csv(
        SECTOR_ATTRIBUTION_OUTPUT_FILE,
        index=False,
    )

    peer_group_attribution.to_csv(
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
        sector_attribution.to_string(
            index=False
        )
    )

    print()
    print("PEER GROUP ATTRIBUTION")
    print(
        peer_group_attribution.to_string(
            index=False
        )
    )

    print()
    print("PORTFOLIO EVALUATION: COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
