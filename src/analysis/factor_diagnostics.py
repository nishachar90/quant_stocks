import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/analysis/nifty50_peer_analysis.csv"


# These are the underlying quantitative metrics.
#
# We deliberately DO NOT include:
# - peer_mean_*
# - peer_median_*
# - vs_peer_median_*
# - peer_ratio_*
# - peer_percentile_*
#
# Those are transformations of the underlying metrics and
# should not be treated as independent factors.
#
CORE_METRICS = [
    "total_return",
    "cagr",
    "annualized_volatility",
    "maximum_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
    "beta",
    "correlation",
    "net_profit_margin",
    "operating_margin",
    "roe",
    "debt_to_equity",
    "free_cash_flow",
    "revenue_growth",
]


# Correlation threshold used to flag potentially redundant metrics.
#
# This is NOT a rule saying that one metric must be removed.
# It is simply a diagnostic warning.
CORRELATION_WARNING_THRESHOLD = 0.70


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():
    """
    Load the peer-analysis dataset.
    """

    print("=" * 70)
    print("FACTOR DIAGNOSTICS")
    print("=" * 70)

    print("\nLOADING DATASET")
    print(f"File: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return df


# ============================================================
# VALIDATE METRICS
# ============================================================

def validate_metrics(df):
    """
    Confirm that all expected core metrics exist.
    """

    missing_columns = [
        metric
        for metric in CORE_METRICS
        if metric not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing expected core metrics: "
            f"{missing_columns}"
        )


# ============================================================
# DATA COMPLETENESS
# ============================================================

def analyze_missing_values(df):
    """
    Display completeness of each candidate metric.
    """

    print("\n")
    print("=" * 70)
    print("1. DATA COMPLETENESS")
    print("=" * 70)

    rows = []

    for metric in CORE_METRICS:
        total = len(df)
        available = df[metric].notna().sum()
        missing = df[metric].isna().sum()

        available_pct = (
            available / total * 100
            if total > 0
            else 0
        )

        rows.append(
            {
                "metric": metric,
                "available": available,
                "missing": missing,
                "available_pct": available_pct,
            }
        )

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False,
            formatters={
                "available_pct": "{:.1f}%".format,
            },
        )
    )

    return result


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

def analyze_distributions(df):
    """
    Display descriptive statistics for each metric.
    """

    print("\n")
    print("=" * 70)
    print("2. METRIC DISTRIBUTIONS")
    print("=" * 70)

    statistics = []

    for metric in CORE_METRICS:

        series = pd.to_numeric(
            df[metric],
            errors="coerce",
        ).dropna()

        if series.empty:
            continue

        statistics.append(
            {
                "metric": metric,
                "count": int(series.count()),
                "mean": series.mean(),
                "median": series.median(),
                "std": series.std(),
                "min": series.min(),
                "max": series.max(),
            }
        )

    result = pd.DataFrame(statistics)

    print(
        result.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    return result


# ============================================================
# PEARSON CORRELATION
# ============================================================

def calculate_pearson_correlation(df):
    """
    Calculate Pearson correlation between core metrics.

    Pearson correlation measures linear relationships.
    """

    numeric_df = df[CORE_METRICS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    correlation_matrix = numeric_df.corr(
        method="pearson"
    )

    return correlation_matrix


# ============================================================
# SPEARMAN CORRELATION
# ============================================================

def calculate_spearman_correlation(df):
    """
    Calculate Spearman rank correlation.

    Spearman is useful here because our eventual factor
    engine will primarily care about relative ordering
    rather than raw numerical distance.
    """

    numeric_df = df[CORE_METRICS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    correlation_matrix = numeric_df.corr(
        method="spearman"
    )

    return correlation_matrix


# ============================================================
# DISPLAY CORRELATION MATRIX
# ============================================================

def display_correlation_matrix(
    correlation_matrix,
    title,
):
    """
    Display a correlation matrix.
    """

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(
        correlation_matrix.to_string(
            float_format=lambda value: f"{value:7.2f}"
        )
    )


# ============================================================
# FIND HIGH-CORRELATION PAIRS
# ============================================================

def find_high_correlation_pairs(
    correlation_matrix,
    threshold=CORRELATION_WARNING_THRESHOLD,
):
    """
    Find unique metric pairs whose absolute correlation
    exceeds the warning threshold.
    """

    pairs = []

    metrics = correlation_matrix.columns

    for i in range(len(metrics)):
        for j in range(i + 1, len(metrics)):

            metric_a = metrics[i]
            metric_b = metrics[j]

            correlation = correlation_matrix.loc[
                metric_a,
                metric_b,
            ]

            if pd.isna(correlation):
                continue

            if abs(correlation) >= threshold:
                pairs.append(
                    {
                        "metric_a": metric_a,
                        "metric_b": metric_b,
                        "correlation": correlation,
                        "absolute_correlation": abs(correlation),
                    }
                )

    result = pd.DataFrame(pairs)

    if not result.empty:
        result = result.sort_values(
            "absolute_correlation",
            ascending=False,
        )

    return result


# ============================================================
# DISPLAY HIGH-CORRELATION PAIRS
# ============================================================

def display_high_correlation_pairs(
    result,
    title,
):
    """
    Display potentially redundant metric pairs.
    """

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)

    if result.empty:
        print(
            "No metric pairs exceeded the "
            f"{CORRELATION_WARNING_THRESHOLD:.2f} threshold."
        )
        return

    display_result = result[
        [
            "metric_a",
            "metric_b",
            "correlation",
        ]
    ].copy()

    print(
        display_result.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


# ============================================================
# FACTOR GROUP PREVIEW
# ============================================================

def display_candidate_factor_groups():
    """
    Display the initial conceptual grouping.

    This is only a diagnostic classification.
    It is NOT the final factor model.
    """

    print("\n")
    print("=" * 70)
    print("5. CANDIDATE FACTOR GROUPS")
    print("=" * 70)

    groups = {
        "Performance / Momentum": [
            "total_return",
            "cagr",
        ],
        "Risk-Adjusted Performance": [
            "sharpe_ratio",
            "sortino_ratio",
        ],
        "Risk": [
            "annualized_volatility",
            "maximum_drawdown",
            "beta",
            "correlation",
        ],
        "Quality": [
            "net_profit_margin",
            "operating_margin",
            "roe",
        ],
        "Growth": [
            "revenue_growth",
        ],
        "Financial Strength": [
            "debt_to_equity",
            "free_cash_flow",
        ],
    }

    for group_name, metrics in groups.items():

        print(f"\n{group_name}")

        for metric in metrics:
            print(f"  - {metric}")


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_dataset()

    validate_metrics(df)

    analyze_missing_values(df)

    analyze_distributions(df)

    pearson = calculate_pearson_correlation(df)

    display_correlation_matrix(
        pearson,
        "3. PEARSON CORRELATION MATRIX",
    )

    high_pearson = find_high_correlation_pairs(
        pearson
    )

    display_high_correlation_pairs(
        high_pearson,
        "4. HIGH PEARSON CORRELATIONS",
    )

    spearman = calculate_spearman_correlation(df)

    display_correlation_matrix(
        spearman,
        "5. SPEARMAN CORRELATION MATRIX",
    )

    high_spearman = find_high_correlation_pairs(
        spearman
    )

    display_high_correlation_pairs(
        high_spearman,
        "6. HIGH SPEARMAN CORRELATIONS",
    )

    display_candidate_factor_groups()

    print("\n")
    print("=" * 70)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 70)

    print(
        "\nNo data was modified."
        "\nNo scores were calculated."
        "\nNo stocks were ranked."
    )


if __name__ == "__main__":
    main()
