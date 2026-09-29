"""
NIFTY 50 Ranking Engine

Generates the final ranked dataset from the factor-score dataset.

Input:
    data/analysis/nifty50_factor_scores.csv

Output:
    data/analysis/nifty50_ranked.csv

Ranking method:
    descending_overall

Higher overall scores receive better ranks.
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "analysis" / "nifty50_factor_scores.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "analysis" / "nifty50_ranked.csv"


# ---------------------------------------------------------------------------
# Ranking specification
# ---------------------------------------------------------------------------

RANKING_METHOD = "descending_overall"

SYMBOL_COLUMN = "symbol"
OVERALL_SCORE_COLUMN = "overall_score"
RANK_COLUMN = "overall_rank"
PERCENTILE_COLUMN = "overall_percentile"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_input(df: pd.DataFrame) -> None:
    """
    Validate the input factor-score dataset.
    """

    required_columns = {
        SYMBOL_COLUMN,
        OVERALL_SCORE_COLUMN,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if df.empty:
        raise ValueError("Input dataset is empty.")

    if df[SYMBOL_COLUMN].isna().any():
        raise ValueError("Input dataset contains missing symbols.")

    if df[SYMBOL_COLUMN].duplicated().any():
        raise ValueError("Input dataset contains duplicate symbols.")

    if not pd.api.types.is_numeric_dtype(df[OVERALL_SCORE_COLUMN]):
        raise ValueError(
            f"Column '{OVERALL_SCORE_COLUMN}' must be numeric."
        )

    if df[OVERALL_SCORE_COLUMN].isna().any():
        raise ValueError(
            f"Column '{OVERALL_SCORE_COLUMN}' contains missing values."
        )

    if not df[OVERALL_SCORE_COLUMN].between(0, 100).all():
        raise ValueError(
            f"Column '{OVERALL_SCORE_COLUMN}' contains values outside 0-100."
        )


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def calculate_ranked_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate the ranked dataset from factor scores.

    Ranking is based exclusively on overall_score.

    Higher overall_score -> better rank.

    Ties receive the same rank using competition ranking:
        1, 2, 2, 4

    The output is sorted by:
        1. overall_rank ascending
        2. symbol ascending
    """

    validate_input(df)

    ranked = df.copy()

    ranked[RANK_COLUMN] = (
        ranked[OVERALL_SCORE_COLUMN]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    # Convert the score into a 0-100 percentile-style ranking.
    #
    # Highest score receives 100.
    # Lowest score receives 0.
    #
    # For a single-stock dataset, the percentile is defined as 100.
    n = len(ranked)

    if n == 1:
        ranked[PERCENTILE_COLUMN] = 100.0
    else:
        ranked[PERCENTILE_COLUMN] = (
            (n - ranked[RANK_COLUMN]) / (n - 1) * 100
        ).round(2)

    ranked = ranked.sort_values(
        by=[RANK_COLUMN, SYMBOL_COLUMN],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return ranked


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------


def load_factor_scores(
    input_file: Path = INPUT_FILE,
) -> pd.DataFrame:
    """
    Load the factor-score dataset.
    """

    if not input_file.exists():
        raise FileNotFoundError(
            f"Factor-score dataset not found: {input_file}"
        )

    return pd.read_csv(input_file)


def save_ranked_dataset(
    df: pd.DataFrame,
    output_file: Path = OUTPUT_FILE,
) -> None:
    """
    Save the ranked dataset to CSV.
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_file,
        index=False,
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_ranking(
    input_file: Path = INPUT_FILE,
    output_file: Path = OUTPUT_FILE,
) -> pd.DataFrame:
    """
    Complete ranking pipeline.

    Returns the ranked DataFrame and writes it to disk.
    """

    factor_scores = load_factor_scores(input_file)

    ranked = calculate_ranked_dataset(factor_scores)

    save_ranked_dataset(
        ranked,
        output_file,
    )

    return ranked


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def print_ranking_audit(
    ranked: pd.DataFrame,
) -> None:
    """
    Print a concise ranking audit.
    """

    print("=" * 70)
    print("NIFTY 50 RANKING ENGINE")
    print("=" * 70)

    print()
    print("DATASET")
    print(f"Rows: {len(ranked)}")
    print(f"Columns: {len(ranked.columns)}")

    print()
    print("RANKING METHOD")
    print(f"- ranking_method: {RANKING_METHOD}")

    print()
    print("RANK RANGE")
    print(f"Minimum rank: {ranked[RANK_COLUMN].min()}")
    print(f"Maximum rank: {ranked[RANK_COLUMN].max()}")

    print()
    print("TOP 10")
    print(
        ranked[
            [
                SYMBOL_COLUMN,
                OVERALL_SCORE_COLUMN,
                RANK_COLUMN,
                PERCENTILE_COLUMN,
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print()
    print(f"OUTPUT: {OUTPUT_FILE}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Command-line execution
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    ranked_dataset = run_ranking()
    print_ranking_audit(ranked_dataset)
