"""
Factor Scoring Specification
============================

Defines how the metrics selected by factor_specification.py
will eventually be converted into comparable 0-100 scores.

This module does NOT:
- calculate scores
- rank stocks
- modify datasets
- select stocks

It only defines and audits the scoring methodology.
"""

from pathlib import Path

import pandas as pd


# ======================================================================
# PATHS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PEER_ANALYSIS_FILE = (
    PROJECT_ROOT / "data" / "analysis" / "nifty50_peer_analysis.csv"
)


# ======================================================================
# SCORING CONFIGURATION
# ======================================================================

# Minimum number of companies required for a peer-relative score.
#
# A peer group containing only 1 or 2 companies does not provide enough
# information for a meaningful peer-relative percentile.
MIN_PEER_GROUP_SIZE = 3


# Metric role weights.
#
# Primary metrics receive twice the importance of supporting metrics.
#
# Example:
#
#   1 primary + 2 supporting
#
#   primary weight = 2
#   supporting weights = 1 + 1
#
#   normalized:
#       primary      = 50%
#       supporting   = 25% each
#
PRIMARY_ROLE_WEIGHT = 2.0
SUPPORTING_ROLE_WEIGHT = 1.0


# Factor weights.
#
# We intentionally begin with equal weights rather than introducing
# arbitrary investment preferences at this stage.
#
# The five currently scoreable factors therefore each receive 20%.
FACTOR_WEIGHTS = {
    "Performance": 0.20,
    "Risk-Adjusted Performance": 0.20,
    "Risk": 0.20,
    "Quality": 0.20,
    "Growth": 0.20,
}


# ======================================================================
# METRIC SCORING SPECIFICATION
# ======================================================================

METRIC_SPECIFICATION = {
    "Performance": {
        "cagr": {
            "role": "primary",
            "direction": "higher_is_better",
        },
    },

    "Risk-Adjusted Performance": {
        "sharpe_ratio": {
            "role": "primary",
            "direction": "higher_is_better",
        },
    },

    "Risk": {
        "annualized_volatility": {
            "role": "primary",
            "direction": "lower_is_better",
        },
        "maximum_drawdown": {
            "role": "primary",
            "direction": "higher_is_better",
        },
        "beta": {
            "role": "supporting",
            "direction": "lower_is_better",
        },
    },

    "Quality": {
        "roe": {
            "role": "primary",
            "direction": "higher_is_better",
        },
        "net_profit_margin": {
            "role": "supporting",
            "direction": "higher_is_better",
        },
        "operating_margin": {
            "role": "supporting",
            "direction": "higher_is_better",
        },
    },

    "Growth": {
        "revenue_growth": {
            "role": "primary",
            "direction": "higher_is_better",
        },
    },
}


# ======================================================================
# EXCLUDED / DEFERRED METRICS
# ======================================================================

DEFERRED_METRICS = {
    "total_return": (
        "Highly correlated with CAGR and therefore should not be "
        "independently scored initially."
    ),
    "sortino_ratio": (
        "Highly correlated with Sharpe ratio and therefore should not "
        "receive an independent score initially."
    ),
    "correlation": (
        "Strongly related to beta in the current dataset and provides "
        "limited additional information for the initial model."
    ),
    "debt_to_equity": (
        "Interpretation varies substantially by business type. "
        "Banks, NBFCs and insurers require specialized treatment."
    ),
    "free_cash_flow": (
        "Absolute FCF is not directly comparable across companies "
        "of different sizes and is not yet normalized."
    ),
}


# Factors that are currently defined conceptually but cannot participate
# in the initial model.
INACTIVE_FACTORS = {
    "Financial Strength": (
        "Debt-to-equity requires business-type-specific treatment and "
        "absolute free cash flow requires normalization."
    ),
    "Value": (
        "The current dataset does not contain suitable valuation "
        "metrics such as earnings yield, P/E, P/B, EV/EBITDA or FCF yield."
    ),
}


# ======================================================================
# SCORING METHOD
# ======================================================================

SCORING_METHOD = {
    "metric_scale": "0-100",
    "base_method": "percentile_rank",
    "preferred_reference_group": "peer_group",
    "minimum_peer_group_size": MIN_PEER_GROUP_SIZE,
    "fallback_1": "sector",
    "fallback_2": "universe",
    "missing_value_handling": "renormalize_available_weights",
    "winsorization": False,
}


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def metric_score_direction(direction: str) -> str:
    """
    Return the scoring orientation for a metric.

    Higher-is-better metrics retain their percentile rank.

    Lower-is-better metrics reverse their percentile rank so that
    a better observation receives a higher score.
    """

    if direction == "higher_is_better":
        return "normal"

    if direction == "lower_is_better":
        return "reverse"

    raise ValueError(f"Unsupported direction: {direction}")


def calculate_role_weights(metrics: dict) -> dict:
    """
    Calculate normalized metric weights within a factor.

    Primary metrics receive twice the weight of supporting metrics.

    The resulting weights always sum to 1.0.
    """

    raw_weights = {}

    for metric, specification in metrics.items():
        role = specification["role"]

        if role == "primary":
            raw_weights[metric] = PRIMARY_ROLE_WEIGHT

        elif role == "supporting":
            raw_weights[metric] = SUPPORTING_ROLE_WEIGHT

        else:
            raise ValueError(
                f"Unsupported metric role '{role}' for metric '{metric}'."
            )

    total_weight = sum(raw_weights.values())

    return {
        metric: weight / total_weight
        for metric, weight in raw_weights.items()
    }


# ======================================================================
# AUDIT
# ======================================================================

def load_dataset() -> pd.DataFrame:
    """Load the peer-analysis dataset."""

    if not PEER_ANALYSIS_FILE.exists():
        raise FileNotFoundError(
            f"Peer analysis dataset not found: {PEER_ANALYSIS_FILE}"
        )

    return pd.read_csv(PEER_ANALYSIS_FILE)


def audit_metric_availability(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Audit availability of metrics used by the scoring specification."""

    rows = []

    for factor, metrics in METRIC_SPECIFICATION.items():
        for metric, specification in metrics.items():

            available = int(df[metric].notna().sum())
            missing = int(df[metric].isna().sum())

            rows.append(
                {
                    "factor": factor,
                    "metric": metric,
                    "role": specification["role"],
                    "direction": specification["direction"],
                    "available": available,
                    "missing": missing,
                    "usable_now": available > 0,
                }
            )

    return pd.DataFrame(rows)


def audit_factor_weights() -> pd.DataFrame:
    """Audit factor-level weights."""

    rows = []

    for factor, weight in FACTOR_WEIGHTS.items():
        rows.append(
            {
                "factor": factor,
                "weight": weight,
                "weight_percent": weight * 100,
            }
        )

    return pd.DataFrame(rows)


def audit_metric_weights() -> pd.DataFrame:
    """Audit metric-level weights within each factor."""

    rows = []

    for factor, metrics in METRIC_SPECIFICATION.items():

        normalized_weights = calculate_role_weights(metrics)

        for metric, specification in metrics.items():

            rows.append(
                {
                    "factor": factor,
                    "metric": metric,
                    "role": specification["role"],
                    "raw_role_weight": (
                        PRIMARY_ROLE_WEIGHT
                        if specification["role"] == "primary"
                        else SUPPORTING_ROLE_WEIGHT
                    ),
                    "normalized_metric_weight": normalized_weights[metric],
                    "weight_percent": (
                        normalized_weights[metric] * 100
                    ),
                }
            )

    return pd.DataFrame(rows)


def validate_specification(df: pd.DataFrame) -> None:
    """Validate the complete scoring specification."""

    errors = []

    # --------------------------------------------------------------
    # Dataset columns
    # --------------------------------------------------------------

    for factor, metrics in METRIC_SPECIFICATION.items():

        for metric in metrics:

            if metric not in df.columns:
                errors.append(
                    f"Missing metric column in dataset: {metric}"
                )

    # --------------------------------------------------------------
    # Factor weights
    # --------------------------------------------------------------

    factor_weight_total = sum(FACTOR_WEIGHTS.values())

    if abs(factor_weight_total - 1.0) > 1e-9:
        errors.append(
            "Active factor weights must sum to 1.0."
        )

    # --------------------------------------------------------------
    # Metric specification
    # --------------------------------------------------------------

    valid_roles = {
        "primary",
        "supporting",
    }

    valid_directions = {
        "higher_is_better",
        "lower_is_better",
    }

    for factor, metrics in METRIC_SPECIFICATION.items():

        if factor not in FACTOR_WEIGHTS:
            errors.append(
                f"Scored factor '{factor}' has no factor weight."
            )

        for metric, specification in metrics.items():

            role = specification["role"]
            direction = specification["direction"]

            if role not in valid_roles:
                errors.append(
                    f"Invalid role '{role}' for metric '{metric}'."
                )

            if direction not in valid_directions:
                errors.append(
                    f"Invalid direction '{direction}' for metric '{metric}'."
                )

    # --------------------------------------------------------------
    # Normalized metric weights
    # --------------------------------------------------------------

    for factor, metrics in METRIC_SPECIFICATION.items():

        weights = calculate_role_weights(metrics)

        total = sum(weights.values())

        if abs(total - 1.0) > 1e-9:
            errors.append(
                f"Metric weights for factor '{factor}' do not sum to 1.0."
            )

    # --------------------------------------------------------------
    # Dataset sanity
    # --------------------------------------------------------------

    required_columns = {
        "symbol",
        "sector",
        "business_type",
        "peer_group",
        "peer_group_size",
    }

    for column in required_columns:

        if column not in df.columns:
            errors.append(
                f"Required grouping column missing: {column}"
            )

    # --------------------------------------------------------------
    # Final result
    # --------------------------------------------------------------

    if errors:

        print()
        print("VALIDATION FAILED")

        for error in errors:
            print(f"- {error}")

        raise ValueError(
            "Factor scoring specification validation failed."
        )

    print()
    print("VALIDATION PASSED")
    print("All scoring metrics exist in the dataset.")
    print("All factor weights sum to 100%.")
    print("All metric weights normalize correctly.")
    print("All directions and roles are valid.")
    print("Required peer, sector and universe grouping columns exist.")


# ======================================================================
# MAIN AUDIT
# ======================================================================

def main() -> None:

    print("=" * 70)
    print("FACTOR SCORING SPECIFICATION AUDIT")
    print("=" * 70)

    print()
    print("LOADING DATASET")
    print(f"File: {PEER_ANALYSIS_FILE}")

    df = load_dataset()

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # --------------------------------------------------------------
    # 1. SCORING METHOD
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("1. SCORING METHOD")
    print("=" * 70)

    for key, value in SCORING_METHOD.items():
        print(f"{key}: {value}")

    # --------------------------------------------------------------
    # 2. METRIC AVAILABILITY
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("2. METRIC AVAILABILITY")
    print("=" * 70)

    availability = audit_metric_availability(df)

    print(
        availability.to_string(index=False)
    )

    # --------------------------------------------------------------
    # 3. FACTOR WEIGHTS
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("3. FACTOR WEIGHTS")
    print("=" * 70)

    factor_weights = audit_factor_weights()

    print(
        factor_weights.to_string(index=False)
    )

    # --------------------------------------------------------------
    # 4. METRIC WEIGHTS
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("4. METRIC WEIGHTS WITHIN FACTORS")
    print("=" * 70)

    metric_weights = audit_metric_weights()

    print(
        metric_weights.to_string(index=False)
    )

    # --------------------------------------------------------------
    # 5. SCORE DIRECTIONS
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("5. SCORE DIRECTIONS")
    print("=" * 70)

    for factor, metrics in METRIC_SPECIFICATION.items():

        print()
        print(factor)

        for metric, specification in metrics.items():

            scoring_orientation = metric_score_direction(
                specification["direction"]
            )

            print(
                f"  - {metric}"
                f" | role={specification['role']}"
                f" | direction={specification['direction']}"
                f" | percentile_orientation={scoring_orientation}"
            )

    # --------------------------------------------------------------
    # 6. FALLBACK HIERARCHY
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("6. REFERENCE GROUP HIERARCHY")
    print("=" * 70)

    print(
        f"1. Peer group if peer_group_size >= "
        f"{MIN_PEER_GROUP_SIZE}"
    )

    print("2. Sector if peer group is too small")

    print("3. Full NIFTY 50 universe if sector is insufficient")

    # --------------------------------------------------------------
    # 7. MISSING VALUE HANDLING
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("7. MISSING VALUE HANDLING")
    print("=" * 70)

    print(
        "Missing metric values will not be imputed."
    )

    print(
        "Available metric weights will be renormalized "
        "within the factor."
    )

    # --------------------------------------------------------------
    # 8. DEFERRED METRICS
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("8. DEFERRED METRICS")
    print("=" * 70)

    for metric, reason in DEFERRED_METRICS.items():
        print(f"- {metric}: {reason}")

    # --------------------------------------------------------------
    # 9. INACTIVE FACTORS
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("9. INACTIVE FACTORS")
    print("=" * 70)

    for factor, reason in INACTIVE_FACTORS.items():
        print(f"- {factor}: {reason}")

    # --------------------------------------------------------------
    # 10. VALIDATION
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("10. SPECIFICATION VALIDATION")
    print("=" * 70)

    validate_specification(df)

    # --------------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("FACTOR SCORING SPECIFICATION AUDIT COMPLETE")
    print("=" * 70)

    print()
    print("No data was modified.")
    print("No scores were calculated.")
    print("No stocks were ranked.")


if __name__ == "__main__":
    main()
