import pandas as pd


# ============================================================
# PEER ANALYSIS CONFIGURATION
# ============================================================

PEER_METRICS = [
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
    "revenue_growth",
]


# ============================================================
# VALIDATE INPUT DATA
# ============================================================

def validate_peer_analysis_data(
    analysis_df,
    classification_df,
    peer_groups_df,
):
    """
    Validate that the three datasets can be joined
    correctly for peer analysis.
    """

    required_analysis_columns = {
        "symbol",
        *PEER_METRICS,
    }

    required_classification_columns = {
        "symbol",
        "business_type",
    }

    required_peer_group_columns = {
        "business_type",
        "peer_group",
    }

    missing_analysis = (
        required_analysis_columns
        - set(analysis_df.columns)
    )

    missing_classification = (
        required_classification_columns
        - set(classification_df.columns)
    )

    missing_peer_groups = (
        required_peer_group_columns
        - set(peer_groups_df.columns)
    )

    if missing_analysis:
        raise ValueError(
            "Missing columns in analysis dataset: "
            f"{sorted(missing_analysis)}"
        )

    if missing_classification:
        raise ValueError(
            "Missing columns in classification dataset: "
            f"{sorted(missing_classification)}"
        )

    if missing_peer_groups:
        raise ValueError(
            "Missing columns in peer-group dataset: "
            f"{sorted(missing_peer_groups)}"
        )

    # --------------------------------------------------------
    # Duplicate checks
    # --------------------------------------------------------

    if analysis_df["symbol"].duplicated().any():
        raise ValueError(
            "Duplicate symbols found in analysis dataset."
        )

    if classification_df["symbol"].duplicated().any():
        raise ValueError(
            "Duplicate symbols found in classification dataset."
        )

    if peer_groups_df["business_type"].duplicated().any():
        raise ValueError(
            "Duplicate business types found in peer-group dataset."
        )


# ============================================================
# BUILD ENRICHED DATASET
# ============================================================

def build_peer_dataset(
    analysis_df,
    classification_df,
    peer_groups_df,
):
    """
    Combine quantitative analysis, business classification,
    and peer-group mapping into one dataframe.
    """

    validate_peer_analysis_data(
        analysis_df,
        classification_df,
        peer_groups_df,
    )

    # --------------------------------------------------------
    # Join stock classification
    # --------------------------------------------------------

    enriched_df = analysis_df.merge(
        classification_df[
            [
                "symbol",
                "sector",
                "business_type",
            ]
        ],
        on="symbol",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Join peer-group mapping
    # --------------------------------------------------------

    enriched_df = enriched_df.merge(
        peer_groups_df[
            [
                "business_type",
                "peer_group",
            ]
        ],
        on="business_type",
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # Validate joins
    # --------------------------------------------------------

    if enriched_df["business_type"].isna().any():
        missing_symbols = enriched_df.loc[
            enriched_df["business_type"].isna(),
            "symbol",
        ].tolist()

        raise ValueError(
            "Stocks without business classification: "
            f"{missing_symbols}"
        )

    if enriched_df["peer_group"].isna().any():
        missing_types = enriched_df.loc[
            enriched_df["peer_group"].isna(),
            "business_type",
        ].unique().tolist()

        raise ValueError(
            "Business types without peer groups: "
            f"{missing_types}"
        )

    return enriched_df


# ============================================================
# ADD PEER GROUP SIZE
# ============================================================

def add_peer_group_size(df):
    """
    Add the number of stocks belonging to each peer group.
    """

    df = df.copy()

    df["peer_group_size"] = (
        df.groupby("peer_group")["symbol"]
        .transform("count")
    )

    return df


# ============================================================
# ADD PEER MEAN
# ============================================================

def add_peer_mean(df, metric):
    """
    Add the mean value of a metric within each peer group.
    """

    column_name = f"peer_mean_{metric}"

    df[column_name] = (
        df.groupby("peer_group")[metric]
        .transform("mean")
    )

    return df


# ============================================================
# ADD PEER MEDIAN
# ============================================================

def add_peer_median(df, metric):
    """
    Add the median value of a metric within each peer group.
    """

    column_name = f"peer_median_{metric}"

    df[column_name] = (
        df.groupby("peer_group")[metric]
        .transform("median")
    )

    return df


# ============================================================
# ADD DIFFERENCE FROM PEER MEDIAN
# ============================================================

def add_difference_from_peer_median(df, metric):
    """
    Calculate the absolute difference between the stock
    metric and the peer-group median.

    For example:

        stock CAGR = 12%
        peer median = 10%

        difference = +2 percentage points
    """

    median_column = f"peer_median_{metric}"
    difference_column = f"vs_peer_median_{metric}"

    df[difference_column] = (
        df[metric]
        - df[median_column]
    )

    return df


# ============================================================
# ADD RATIO TO PEER MEDIAN
# ============================================================

def add_ratio_to_peer_median(df, metric):
    """
    Calculate the ratio of the stock metric to the
    peer-group median.

    Example:

        stock CAGR = 12%
        peer median = 10%

        ratio = 1.20
    """

    median_column = f"peer_median_{metric}"
    ratio_column = f"peer_ratio_{metric}"

    df[ratio_column] = (
        df[metric]
        / df[median_column].replace(0, pd.NA)
    )

    return df


# ============================================================
# ADD PEER PERCENTILE
# ============================================================

def add_peer_percentile(df, metric):
    """
    Calculate the percentile rank of each stock within
    its peer group.

    Peer groups with only one stock receive NaN because
    there is no meaningful peer comparison.
    """

    percentile_column = f"peer_percentile_{metric}"

    def calculate_percentile(group):
        if len(group) < 2:
            return pd.Series(
                [pd.NA] * len(group),
                index=group.index,
                dtype="Float64",
            )

        return group.rank(
            method="average",
            pct=True,
        )

    df[percentile_column] = (
        df.groupby("peer_group")[metric]
        .transform(calculate_percentile)
    )

    return df


# ============================================================
# CALCULATE ALL PEER METRICS
# ============================================================

def calculate_peer_metrics(
    df,
    metrics=None,
):
    """
    Calculate peer-relative statistics for the requested
    metrics.

    For each metric we calculate:

        peer mean
        peer median
        difference from peer median
        ratio to peer median
        percentile within peer group
    """

    if metrics is None:
        metrics = PEER_METRICS

    result = df.copy()

    result = add_peer_group_size(result)

    for metric in metrics:

        if metric not in result.columns:
            raise ValueError(
                f"Metric '{metric}' not found in dataframe."
            )

        result = add_peer_mean(
            result,
            metric,
        )

        result = add_peer_median(
            result,
            metric,
        )

        result = add_difference_from_peer_median(
            result,
            metric,
        )

        result = add_ratio_to_peer_median(
            result,
            metric,
        )

        result = add_peer_percentile(
            result,
            metric,
        )

    # --------------------------------------------------------
    # Remove peer-relative calculations for singleton groups
    # --------------------------------------------------------

    singleton_mask = result["peer_group_size"] < 2

    relative_columns = []

    for metric in metrics:
        relative_columns.extend(
            [
                f"peer_mean_{metric}",
                f"peer_median_{metric}",
                f"vs_peer_median_{metric}",
                f"peer_ratio_{metric}",
                f"peer_percentile_{metric}",
            ]
        )

    result.loc[
        singleton_mask,
        relative_columns,
    ] = pd.NA

    return result


# ============================================================
# COMPLETE PEER ANALYSIS PIPELINE
# ============================================================

def analyze_peers(
    analysis_df,
    classification_df,
    peer_groups_df,
    metrics=None,
):
    """
    Complete peer-analysis pipeline.

    Steps:

        1. Validate input datasets
        2. Join stock classification
        3. Join peer groups
        4. Calculate peer-group sizes
        5. Calculate peer-relative statistics
    """

    enriched_df = build_peer_dataset(
        analysis_df,
        classification_df,
        peer_groups_df,
    )

    result = calculate_peer_metrics(
        enriched_df,
        metrics=metrics,
    )

    return result
