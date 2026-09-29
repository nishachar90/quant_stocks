"""
NIFTY 50 Portfolio Construction Specification

Defines and audits the rules that the portfolio construction engine
will use to convert eligible ranked stocks into a portfolio.

This module defines the specification only.
It does NOT construct the portfolio.
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "analysis" / "nifty50_eligible.csv"


# ---------------------------------------------------------------------------
# Portfolio specification
# ---------------------------------------------------------------------------

PORTFOLIO_SPECIFICATION = {
    # Target number of holdings.
    "target_holdings": 10,

    # Hard limits on number of holdings.
    "minimum_holdings": 8,
    "maximum_holdings": 12,

    # Position-weight constraints.
    "minimum_position_weight": 0.05,
    "maximum_position_weight": 0.15,

    # Concentration constraints.
    "maximum_sector_weight": 0.30,
    "maximum_business_type_weight": 0.20,

    # Portfolio construction method.
    "weighting_method": "score_proportional",

    # Selection method.
    "selection_method": "rank_order",

    # Portfolio weights must sum to 100%.
    "target_total_weight": 1.00,
}


# ---------------------------------------------------------------------------
# Required input columns
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "symbol",
    "overall_score",
    "overall_rank",
    "overall_percentile",
    "eligible",
    "sector",
    "business_type",
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_specification() -> None:
    """
    Validate the portfolio specification itself.
    """

    target = PORTFOLIO_SPECIFICATION["target_holdings"]
    minimum = PORTFOLIO_SPECIFICATION["minimum_holdings"]
    maximum = PORTFOLIO_SPECIFICATION["maximum_holdings"]

    if minimum < 1:
        raise ValueError(
            "minimum_holdings must be at least 1."
        )

    if minimum > target:
        raise ValueError(
            "minimum_holdings cannot exceed target_holdings."
        )

    if target > maximum:
        raise ValueError(
            "target_holdings cannot exceed maximum_holdings."
        )

    min_weight = PORTFOLIO_SPECIFICATION[
        "minimum_position_weight"
    ]
    max_weight = PORTFOLIO_SPECIFICATION[
        "maximum_position_weight"
    ]

    if not 0 < min_weight <= 1:
        raise ValueError(
            "minimum_position_weight must be between 0 and 1."
        )

    if not 0 < max_weight <= 1:
        raise ValueError(
            "maximum_position_weight must be between 0 and 1."
        )

    if min_weight > max_weight:
        raise ValueError(
            "minimum_position_weight cannot exceed "
            "maximum_position_weight."
        )

    max_sector = PORTFOLIO_SPECIFICATION[
        "maximum_sector_weight"
    ]

    max_business_type = PORTFOLIO_SPECIFICATION[
        "maximum_business_type_weight"
    ]

    if not 0 < max_sector <= 1:
        raise ValueError(
            "maximum_sector_weight must be between 0 and 1."
        )

    if not 0 < max_business_type <= 1:
        raise ValueError(
            "maximum_business_type_weight must be between 0 and 1."
        )

    target_weight = PORTFOLIO_SPECIFICATION[
        "target_total_weight"
    ]

    if target_weight != 1.00:
        raise ValueError(
            "target_total_weight must equal 1.00."
        )

    valid_weighting_methods = {
        "equal_weight",
        "score_proportional",
    }

    if PORTFOLIO_SPECIFICATION["weighting_method"] not in (
        valid_weighting_methods
    ):
        raise ValueError(
            "Unsupported weighting method."
        )

    valid_selection_methods = {
        "rank_order",
    }

    if PORTFOLIO_SPECIFICATION["selection_method"] not in (
        valid_selection_methods
    ):
        raise ValueError(
            "Unsupported selection method."
        )


def validate_input(df: pd.DataFrame) -> None:
    """
    Validate the eligible-stock dataset.
    """

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if df.empty:
        raise ValueError("Input dataset is empty.")

    if df["symbol"].isna().any():
        raise ValueError("Input dataset contains missing symbols.")

    if df["symbol"].duplicated().any():
        raise ValueError(
            "Input dataset contains duplicate symbols."
        )

    numeric_columns = {
        "overall_score",
        "overall_rank",
        "overall_percentile",
    }

    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"Column '{column}' must be numeric."
            )

        if df[column].isna().any():
            raise ValueError(
                f"Column '{column}' contains missing values."
            )

    if df["eligible"].isna().any():
        raise ValueError(
            "Column 'eligible' contains missing values."
        )

    if df["sector"].isna().any():
        raise ValueError(
            "Column 'sector' contains missing values."
        )

    if df["business_type"].isna().any():
        raise ValueError(
            "Column 'business_type' contains missing values."
        )


# ---------------------------------------------------------------------------
# Specification access
# ---------------------------------------------------------------------------


def get_portfolio_specification() -> dict:
    """
    Return a copy of the portfolio specification.
    """

    return PORTFOLIO_SPECIFICATION.copy()


# ---------------------------------------------------------------------------
# Feasibility checks
# ---------------------------------------------------------------------------


def minimum_weight_feasibility() -> bool:
    """
    Check whether the target number of holdings can satisfy
    the minimum position weight constraint.
    """

    target = PORTFOLIO_SPECIFICATION["target_holdings"]
    minimum_weight = PORTFOLIO_SPECIFICATION[
        "minimum_position_weight"
    ]

    return target * minimum_weight <= 1.00


def maximum_weight_feasibility() -> bool:
    """
    Check whether the target number of holdings can satisfy
    the maximum position weight constraint.
    """

    target = PORTFOLIO_SPECIFICATION["target_holdings"]
    maximum_weight = PORTFOLIO_SPECIFICATION[
        "maximum_position_weight"
    ]

    return target * maximum_weight >= 1.00


def holding_count_is_feasible() -> bool:
    """
    Check whether the target holding count lies inside
    the configured minimum/maximum range.
    """

    target = PORTFOLIO_SPECIFICATION["target_holdings"]
    minimum = PORTFOLIO_SPECIFICATION["minimum_holdings"]
    maximum = PORTFOLIO_SPECIFICATION["maximum_holdings"]

    return minimum <= target <= maximum


def specification_is_feasible() -> bool:
    """
    Check all internal portfolio specification constraints.
    """

    validate_specification()

    return (
        holding_count_is_feasible()
        and minimum_weight_feasibility()
        and maximum_weight_feasibility()
    )


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def load_eligible_dataset(
    input_file: Path = INPUT_FILE,
) -> pd.DataFrame:
    """
    Load the eligibility dataset.
    """

    if not input_file.exists():
        raise FileNotFoundError(
            f"Eligible dataset not found: {input_file}"
        )

    return pd.read_csv(input_file)


# ---------------------------------------------------------------------------
# Dataset audit
# ---------------------------------------------------------------------------


def audit_input_dataset(
    df: pd.DataFrame,
) -> dict:
    """
    Return basic information required before portfolio construction.
    """

    validate_input(df)

    eligible_count = int(
        df["eligible"].sum()
    )

    return {
        "rows": len(df),
        "eligible_count": eligible_count,
        "ineligible_count": len(df) - eligible_count,
        "unique_sectors": int(
            df["sector"].nunique()
        ),
        "unique_business_types": int(
            df["business_type"].nunique()
        ),
        "required_holdings": PORTFOLIO_SPECIFICATION[
            "target_holdings"
        ],
    }


# ---------------------------------------------------------------------------
# Full specification audit
# ---------------------------------------------------------------------------


def run_specification_audit(
    input_file: Path = INPUT_FILE,
) -> dict:
    """
    Run the complete portfolio specification audit.
    """

    validate_specification()

    if not specification_is_feasible():
        raise ValueError(
            "Portfolio specification is not feasible."
        )

    df = load_eligible_dataset(input_file)

    audit = audit_input_dataset(df)

    if audit["eligible_count"] < PORTFOLIO_SPECIFICATION[
        "minimum_holdings"
    ]:
        raise ValueError(
            "Number of eligible stocks is below "
            "minimum_holdings."
        )

    return audit


# ---------------------------------------------------------------------------
# Audit output
# ---------------------------------------------------------------------------


def print_specification_audit(
    audit: dict,
) -> None:
    """
    Print the portfolio specification audit.
    """

    specification = get_portfolio_specification()

    print("=" * 70)
    print("NIFTY 50 PORTFOLIO SPECIFICATION AUDIT")
    print("=" * 70)

    print()
    print("INPUT DATASET")
    print(f"Rows: {audit['rows']}")
    print(f"Eligible stocks: {audit['eligible_count']}")
    print(f"Ineligible stocks: {audit['ineligible_count']}")
    print(f"Unique sectors: {audit['unique_sectors']}")
    print(
        f"Unique business types: "
        f"{audit['unique_business_types']}"
    )

    print()
    print("PORTFOLIO SPECIFICATION")
    print(
        f"- target_holdings: "
        f"{specification['target_holdings']}"
    )
    print(
        f"- minimum_holdings: "
        f"{specification['minimum_holdings']}"
    )
    print(
        f"- maximum_holdings: "
        f"{specification['maximum_holdings']}"
    )
    print(
        f"- minimum_position_weight: "
        f"{specification['minimum_position_weight']}"
    )
    print(
        f"- maximum_position_weight: "
        f"{specification['maximum_position_weight']}"
    )
    print(
        f"- maximum_sector_weight: "
        f"{specification['maximum_sector_weight']}"
    )
    print(
        f"- maximum_business_type_weight: "
        f"{specification['maximum_business_type_weight']}"
    )
    print(
        f"- weighting_method: "
        f"{specification['weighting_method']}"
    )
    print(
        f"- selection_method: "
        f"{specification['selection_method']}"
    )
    print(
        f"- target_total_weight: "
        f"{specification['target_total_weight']}"
    )

    print()
    print("FEASIBILITY")
    print(
        f"- holding_count: "
        f"{holding_count_is_feasible()}"
    )
    print(
        f"- minimum_weight: "
        f"{minimum_weight_feasibility()}"
    )
    print(
        f"- maximum_weight: "
        f"{maximum_weight_feasibility()}"
    )
    print(
        f"- specification: "
        f"{specification_is_feasible()}"
    )

    print()
    print("PORTFOLIO SPECIFICATION AUDIT: PASS")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Command-line execution
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    audit_result = run_specification_audit()
    print_specification_audit(audit_result)
