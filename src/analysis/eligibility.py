"""
NIFTY 50 Eligibility Engine

Determines whether ranked stocks are eligible for the next stage of
portfolio construction.

This layer does NOT rank stocks.

Input:
    data/analysis/nifty50_ranked.csv

Output:
    data/analysis/nifty50_eligible.csv

The eligibility rules are deliberately configurable. No investment
threshold is hidden inside the engine.
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "analysis" / "nifty50_ranked.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "analysis" / "nifty50_eligible.csv"


# ---------------------------------------------------------------------------
# Required columns
# ---------------------------------------------------------------------------

SYMBOL_COLUMN = "symbol"
OVERALL_SCORE_COLUMN = "overall_score"
RANK_COLUMN = "overall_rank"
PERCENTILE_COLUMN = "overall_percentile"

ELIGIBLE_COLUMN = "eligible"
ELIGIBILITY_REASON_COLUMN = "eligibility_reason"


# ---------------------------------------------------------------------------
# Eligibility specification
# ---------------------------------------------------------------------------

# These limits are intentionally configurable.
#
# None means that the corresponding criterion is not currently enforced.

MAX_RANK = None
MIN_SCORE = None
MIN_PERCENTILE = None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_input(df: pd.DataFrame) -> None:
    """
    Validate the ranked dataset.
    """

    required_columns = {
        SYMBOL_COLUMN,
        OVERALL_SCORE_COLUMN,
        RANK_COLUMN,
        PERCENTILE_COLUMN,
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

    numeric_columns = [
        OVERALL_SCORE_COLUMN,
        RANK_COLUMN,
        PERCENTILE_COLUMN,
    ]

    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"Column '{column}' must be numeric."
            )

        if df[column].isna().any():
            raise ValueError(
                f"Column '{column}' contains missing values."
            )

    if not df[OVERALL_SCORE_COLUMN].between(0, 100).all():
        raise ValueError(
            f"Column '{OVERALL_SCORE_COLUMN}' contains values outside 0-100."
        )

    if (df[RANK_COLUMN] < 1).any():
        raise ValueError(
            f"Column '{RANK_COLUMN}' contains invalid ranks."
        )

    if not df[PERCENTILE_COLUMN].between(0, 100).all():
        raise ValueError(
            f"Column '{PERCENTILE_COLUMN}' contains values outside 0-100."
        )


# ---------------------------------------------------------------------------
# Eligibility specification
# ---------------------------------------------------------------------------


def get_eligibility_specification() -> dict:
    """
    Return the currently active eligibility rules.
    """

    return {
        "max_rank": MAX_RANK,
        "min_score": MIN_SCORE,
        "min_percentile": MIN_PERCENTILE,
    }


# ---------------------------------------------------------------------------
# Individual rule checks
# ---------------------------------------------------------------------------


def check_rank_eligibility(row: pd.Series) -> bool:
    """
    Check maximum-rank criterion.

    If MAX_RANK is None, the criterion is disabled.
    """

    if MAX_RANK is None:
        return True

    return row[RANK_COLUMN] <= MAX_RANK


def check_score_eligibility(row: pd.Series) -> bool:
    """
    Check minimum-score criterion.

    If MIN_SCORE is None, the criterion is disabled.
    """

    if MIN_SCORE is None:
        return True

    return row[OVERALL_SCORE_COLUMN] >= MIN_SCORE


def check_percentile_eligibility(row: pd.Series) -> bool:
    """
    Check minimum-percentile criterion.

    If MIN_PERCENTILE is None, the criterion is disabled.
    """

    if MIN_PERCENTILE is None:
        return True

    return row[PERCENTILE_COLUMN] >= MIN_PERCENTILE


# ---------------------------------------------------------------------------
# Reason generation
# ---------------------------------------------------------------------------


def get_ineligibility_reasons(row: pd.Series) -> list[str]:
    """
    Return all eligibility criteria failed by a stock.
    """

    reasons = []

    if MAX_RANK is not None and not check_rank_eligibility(row):
        reasons.append(
            f"rank_above_{MAX_RANK}"
        )

    if MIN_SCORE is not None and not check_score_eligibility(row):
        reasons.append(
            f"score_below_{MIN_SCORE}"
        )

    if MIN_PERCENTILE is not None and not check_percentile_eligibility(row):
        reasons.append(
            f"percentile_below_{MIN_PERCENTILE}"
        )

    return reasons


def get_eligibility_reason(row: pd.Series) -> str:
    """
    Return a human-readable eligibility reason.
    """

    reasons = get_ineligibility_reasons(row)

    if not reasons:
        return "eligible"

    return ";".join(reasons)


# ---------------------------------------------------------------------------
# Eligibility calculation
# ---------------------------------------------------------------------------


def calculate_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the configured eligibility rules.

    Existing columns are preserved.
    """

    validate_input(df)

    eligible = df.copy()

    eligible[ELIGIBLE_COLUMN] = eligible.apply(
        lambda row: (
            check_rank_eligibility(row)
            and check_score_eligibility(row)
            and check_percentile_eligibility(row)
        ),
        axis=1,
    )

    eligible[ELIGIBILITY_REASON_COLUMN] = eligible.apply(
        get_eligibility_reason,
        axis=1,
    )

    return eligible


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------


def load_ranked_dataset(
    input_file: Path = INPUT_FILE,
) -> pd.DataFrame:
    """
    Load the ranked dataset.
    """

    if not input_file.exists():
        raise FileNotFoundError(
            f"Ranked dataset not found: {input_file}"
        )

    return pd.read_csv(input_file)


def save_eligible_dataset(
    df: pd.DataFrame,
    output_file: Path = OUTPUT_FILE,
) -> None:
    """
    Save the eligibility dataset.
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


def run_eligibility(
    input_file: Path = INPUT_FILE,
    output_file: Path = OUTPUT_FILE,
) -> pd.DataFrame:
    """
    Complete eligibility pipeline.
    """

    ranked = load_ranked_dataset(input_file)

    eligible = calculate_eligibility(ranked)

    save_eligible_dataset(
        eligible,
        output_file,
    )

    return eligible


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def print_eligibility_audit(
    df: pd.DataFrame,
) -> None:
    """
    Print a concise eligibility audit.
    """

    specification = get_eligibility_specification()

    print("=" * 70)
    print("NIFTY 50 ELIGIBILITY ENGINE")
    print("=" * 70)

    print()
    print("DATASET")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print()
    print("ELIGIBILITY SPECIFICATION")
    print(f"- max_rank: {specification['max_rank']}")
    print(f"- min_score: {specification['min_score']}")
    print(f"- min_percentile: {specification['min_percentile']}")

    print()
    print("RESULTS")
    print(f"Eligible: {df[ELIGIBLE_COLUMN].sum()}")
    print(f"Ineligible: {(~df[ELIGIBLE_COLUMN]).sum()}")

    print()
    print("OUTPUT")
    print(OUTPUT_FILE)

    print("=" * 70)


# ---------------------------------------------------------------------------
# Command-line execution
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    eligible_dataset = run_eligibility()
    print_eligibility_audit(eligible_dataset)
