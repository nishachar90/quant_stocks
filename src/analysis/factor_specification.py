import pandas as pd


INPUT_FILE = "data/analysis/nifty50_peer_analysis.csv"


FACTOR_SPECIFICATION = {
    "Performance": {
        "metrics": {
            "cagr": {
                "role": "primary",
                "direction": "higher_is_better",
                "usable_now": True,
                "reason": "Represents long-term annualized historical performance.",
            },
            "total_return": {
                "role": "supporting",
                "direction": "higher_is_better",
                "usable_now": False,
                "reason": (
                    "Highly correlated with CAGR and therefore should not "
                    "be independently scored."
                ),
            },
        },
    },

    "Risk-Adjusted Performance": {
        "metrics": {
            "sharpe_ratio": {
                "role": "primary",
                "direction": "higher_is_better",
                "usable_now": True,
                "reason": (
                    "Measures historical return relative to total volatility."
                ),
            },
            "sortino_ratio": {
                "role": "supporting",
                "direction": "higher_is_better",
                "usable_now": False,
                "reason": (
                    "Highly correlated with Sharpe ratio and should not "
                    "receive an independent score initially."
                ),
            },
        },
    },

    "Risk": {
        "metrics": {
            "annualized_volatility": {
                "role": "primary",
                "direction": "lower_is_better",
                "usable_now": True,
                "reason": "Measures historical variability of returns.",
            },
            "maximum_drawdown": {
                "role": "primary",
                "direction": "higher_is_better",
                "usable_now": True,
                "reason": (
                    "Less-negative maximum drawdown represents lower "
                    "historical peak-to-trough loss."
                ),
            },
            "beta": {
                "role": "supporting",
                "direction": "lower_is_better",
                "usable_now": True,
                "reason": (
                    "Measures sensitivity to broad market movements; "
                    "interpretation is risk-oriented rather than purely "
                    "quality-oriented."
                ),
            },
            "correlation": {
                "role": "supporting",
                "direction": "lower_is_better",
                "usable_now": False,
                "reason": (
                    "Strongly related to beta in this dataset and provides "
                    "limited additional information for the initial model."
                ),
            },
        },
    },

    "Quality": {
        "metrics": {
            "roe": {
                "role": "primary",
                "direction": "higher_is_better",
                "usable_now": True,
                "reason": (
                    "Measures return generated on shareholders' equity."
                ),
            },
            "net_profit_margin": {
                "role": "supporting",
                "direction": "higher_is_better",
                "usable_now": True,
                "reason": (
                    "Measures profitability after expenses; correlated with "
                    "operating margin, so it should not be blindly double-counted."
                ),
            },
            "operating_margin": {
                "role": "supporting",
                "direction": "higher_is_better",
                "usable_now": True,
                "reason": (
                    "Measures operating profitability; only 40 of 50 stocks "
                    "currently have data."
                ),
            },
        },
    },

    "Growth": {
        "metrics": {
            "revenue_growth": {
                "role": "primary",
                "direction": "higher_is_better",
                "usable_now": True,
                "reason": "Measures historical revenue growth.",
            },
        },
    },

    "Financial Strength": {
        "metrics": {
            "debt_to_equity": {
                "role": "conditional",
                "direction": "lower_is_better",
                "usable_now": False,
                "reason": (
                    "Interpretation varies substantially by business type. "
                    "Banks, NBFCs and insurers require specialized treatment."
                ),
            },
            "free_cash_flow": {
                "role": "supporting",
                "direction": "higher_is_better",
                "usable_now": False,
                "reason": (
                    "Absolute FCF is not directly comparable across companies "
                    "of different sizes and is not yet normalized."
                ),
            },
        },
    },

    "Value": {
        "metrics": {},
        "status": "not_available",
        "reason": (
            "The current dataset does not contain suitable valuation metrics "
            "such as earnings yield, P/E, P/B, EV/EBITDA or FCF yield."
        ),
    },
}


def load_dataset():
    print("=" * 70)
    print("FACTOR SPECIFICATION AUDIT")
    print("=" * 70)

    print("\nLOADING DATASET")
    print(f"File: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return df


def flatten_specification():
    rows = []

    for factor_name, factor_data in FACTOR_SPECIFICATION.items():
        metrics = factor_data.get("metrics", {})

        for metric_name, specification in metrics.items():
            rows.append(
                {
                    "factor": factor_name,
                    "metric": metric_name,
                    "role": specification["role"],
                    "direction": specification["direction"],
                    "usable_now": specification["usable_now"],
                    "reason": specification["reason"],
                }
            )

    return pd.DataFrame(rows)


def audit_metric_availability(df, specification_df):
    print("\n")
    print("=" * 70)
    print("1. METRIC AVAILABILITY")
    print("=" * 70)

    results = []

    for _, row in specification_df.iterrows():
        metric = row["metric"]

        if metric not in df.columns:
            available = 0
            missing = len(df)
        else:
            available = int(df[metric].notna().sum())
            missing = int(df[metric].isna().sum())

        results.append(
            {
                "factor": row["factor"],
                "metric": metric,
                "available": available,
                "missing": missing,
                "usable_now": row["usable_now"],
            }
        )

    result = pd.DataFrame(results)

    print(result.to_string(index=False))

    return result


def audit_roles(specification_df):
    print("\n")
    print("=" * 70)
    print("2. FACTOR ROLES")
    print("=" * 70)

    role_summary = (
        specification_df.groupby(
            ["factor", "role"],
            dropna=False,
        )
        .size()
        .reset_index(name="metric_count")
    )

    print(role_summary.to_string(index=False))

    return role_summary


def audit_directions(specification_df):
    print("\n")
    print("=" * 70)
    print("3. FACTOR DIRECTIONS")
    print("=" * 70)

    direction_summary = specification_df[
        [
            "factor",
            "metric",
            "role",
            "direction",
        ]
    ].copy()

    print(direction_summary.to_string(index=False))

    return direction_summary


def display_factor_structure():
    print("\n")
    print("=" * 70)
    print("4. FACTOR STRUCTURE")
    print("=" * 70)

    for factor_name, factor_data in FACTOR_SPECIFICATION.items():
        print(f"\n{factor_name}")

        status = factor_data.get("status")

        if status:
            print(f"  Status: {status}")

        metrics = factor_data.get("metrics", {})

        if not metrics:
            print("  Metrics: none currently available")
            print(f"  Reason: {factor_data.get('reason')}")
            continue

        for metric_name, specification in metrics.items():
            print(
                f"  - {metric_name}"
                f" | role={specification['role']}"
                f" | direction={specification['direction']}"
                f" | usable_now={specification['usable_now']}"
            )


def display_current_scoring_candidates(specification_df):
    print("\n")
    print("=" * 70)
    print("5. CURRENT SCORING CANDIDATES")
    print("=" * 70)

    candidates = specification_df[
        specification_df["usable_now"] == True
    ].copy()

    candidates = candidates[
        candidates["role"].isin(
            ["primary", "supporting"]
        )
    ]

    print(
        candidates[
            [
                "factor",
                "metric",
                "role",
                "direction",
            ]
        ].to_string(index=False)
    )


def display_excluded_metrics(specification_df):
    print("\n")
    print("=" * 70)
    print("6. NOT CURRENTLY SUITABLE FOR DIRECT SCORING")
    print("=" * 70)

    excluded = specification_df[
        specification_df["usable_now"] == False
    ].copy()

    print(
        excluded[
            [
                "factor",
                "metric",
                "role",
                "reason",
            ]
        ].to_string(index=False)
    )


def validate_specification(df, specification_df):
    print("\n")
    print("=" * 70)
    print("7. SPECIFICATION VALIDATION")
    print("=" * 70)

    errors = []

    for _, row in specification_df.iterrows():
        metric = row["metric"]

        if metric not in df.columns:
            errors.append(
                f"Metric missing from dataset: {metric}"
            )

        if row["direction"] not in [
            "higher_is_better",
            "lower_is_better",
        ]:
            errors.append(
                f"Invalid direction for metric: {metric}"
            )

        if row["role"] not in [
            "primary",
            "supporting",
            "conditional",
        ]:
            errors.append(
                f"Invalid role for metric: {metric}"
            )

    if errors:
        print("VALIDATION FAILED")

        for error in errors:
            print(f"- {error}")

        raise ValueError(
            "Factor specification validation failed."
        )

    print("VALIDATION PASSED")
    print("All specified metrics exist in the dataset.")
    print("All directions are valid.")
    print("All factor roles are valid.")


def main():
    df = load_dataset()

    specification_df = flatten_specification()

    audit_metric_availability(
        df,
        specification_df,
    )

    audit_roles(
        specification_df,
    )

    audit_directions(
        specification_df,
    )

    display_factor_structure()

    display_current_scoring_candidates(
        specification_df,
    )

    display_excluded_metrics(
        specification_df,
    )

    validate_specification(
        df,
        specification_df,
    )

    print("\n")
    print("=" * 70)
    print("FACTOR SPECIFICATION AUDIT COMPLETE")
    print("=" * 70)

    print("\nNo data was modified.")
    print("No scores were calculated.")
    print("No stocks were ranked.")


if __name__ == "__main__":
    main()
