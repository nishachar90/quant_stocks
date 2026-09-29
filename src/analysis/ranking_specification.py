from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FACTOR_SCORE_FILE = (
    PROJECT_ROOT / "data" / "analysis" / "nifty50_factor_scores.csv"
)

MIN_RANK = 1


RANKING_COLUMNS = [
    "overall_score",
    "performance_score",
    "risk-adjusted_performance_score",
    "risk_score",
    "quality_score",
    "growth_score",
]


RANKING_SPECIFICATION = {
    "ranking_method": "descending_overall_score",
    "primary_ranking_metric": "overall_score",
    "rank_direction": "higher_score_better",
    "tie_method": "average",
    "rank_starts_at": MIN_RANK,
    "overall_percentile_method": "rank_percentile",
    "factor_ranks": True,
    "factor_percentiles": True,
    "reference_context_preserved": True,
}


def load_dataset():
    """Load the factor-score dataset."""
    if not FACTOR_SCORE_FILE.exists():
        raise FileNotFoundError(
            f"Factor score file not found: {FACTOR_SCORE_FILE}"
        )

    return pd.read_csv(FACTOR_SCORE_FILE)


def audit_required_columns(df):
    """Verify that all ranking inputs are present."""
    required_columns = ["symbol"] + RANKING_COLUMNS

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required ranking columns: {missing_columns}"
        )


def audit_unique_symbols(df):
    """Verify that every stock appears exactly once."""
    duplicate_count = df["symbol"].duplicated().sum()

    if duplicate_count != 0:
        raise ValueError(
            f"Duplicate symbols found: {duplicate_count}"
        )


def audit_score_ranges(df):
    """Verify all ranking inputs are on the expected 0-100 scale."""
    for column in RANKING_COLUMNS:
        invalid = df[
            (df[column] < 0) |
            (df[column] > 100)
        ]

        if not invalid.empty:
            raise ValueError(
                f"{column} contains scores outside 0-100."
            )


def audit_missing_scores(df):
    """Verify that ranking inputs contain no missing scores."""
    missing = df[RANKING_COLUMNS].isna().sum()

    missing = missing[missing > 0]

    if not missing.empty:
        raise ValueError(
            f"Missing ranking scores detected:\n{missing}"
        )


def calculate_ranks(df):
    """Calculate overall and factor ranks."""
    result = df.copy()

    for column in RANKING_COLUMNS:
        rank_column = f"{column}_rank"

        result[rank_column] = (
            result[column]
            .rank(
                ascending=False,
                method=RANKING_SPECIFICATION["tie_method"],
            )
        )

    return result


def calculate_percentiles(df):
    """Calculate percentile position for overall and factor scores."""
    result = df.copy()

    for column in RANKING_COLUMNS:
        percentile_column = f"{column}_percentile"

        result[percentile_column] = (
            result[column]
            .rank(
                ascending=True,
                method="average",
                pct=True,
            )
            * 100
        )

    return result


def validate_ranking_specification():
    """Validate the ranking design."""
    df = load_dataset()

    audit_required_columns(df)
    audit_unique_symbols(df)
    audit_score_ranges(df)
    audit_missing_scores(df)

    ranked = calculate_ranks(df)
    ranked = calculate_percentiles(ranked)

    rank_columns = [
        f"{column}_rank"
        for column in RANKING_COLUMNS
    ]

    percentile_columns = [
        f"{column}_percentile"
        for column in RANKING_COLUMNS
    ]

    if ranked[rank_columns].isna().any().any():
        raise ValueError("Missing ranks detected.")

    if ranked[percentile_columns].isna().any().any():
        raise ValueError("Missing percentiles detected.")

    if (ranked[rank_columns] < MIN_RANK).any().any():
        raise ValueError("Invalid rank below minimum rank detected.")

    if (
        (ranked[percentile_columns] < 0)
        | (ranked[percentile_columns] > 100)
    ).any().any():
        raise ValueError(
            "Percentile values outside 0-100 detected."
        )

    return True


def main():
    print("=" * 70)
    print("NIFTY 50 RANKING SPECIFICATION AUDIT")
    print("=" * 70)

    df = load_dataset()

    print("\nDATASET")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    audit_required_columns(df)
    print("\nREQUIRED COLUMNS")
    print("PASS")

    audit_unique_symbols(df)
    print("\nUNIQUE SYMBOLS")
    print("PASS")

    audit_score_ranges(df)
    print("\nSCORE RANGES")
    print("PASS")

    audit_missing_scores(df)
    print("\nMISSING SCORES")
    print("PASS")

    ranked = calculate_ranks(df)
    ranked = calculate_percentiles(ranked)

    print("\nRANKING SPECIFICATION")
    for key, value in RANKING_SPECIFICATION.items():
        print(f"- {key}: {value}")

    print("\nRANK COLUMNS")
    for column in RANKING_COLUMNS:
        print(f"- {column}_rank")

    print("\nPERCENTILE COLUMNS")
    for column in RANKING_COLUMNS:
        print(f"- {column}_percentile")

    print("\nRANKING VALIDATION")
    validate_ranking_specification()
    print("PASS")

    print("\n" + "=" * 70)
    print("RANKING SPECIFICATION VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
