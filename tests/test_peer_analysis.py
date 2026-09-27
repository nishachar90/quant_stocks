import pandas as pd

from src.analysis.peer_analysis import (
    analyze_peers,
    PEER_METRICS,
)


# ============================================================
# FILE PATHS
# ============================================================

ANALYSIS_FILE = "data/analysis/nifty50_analysis.csv"

CLASSIFICATION_FILE = (
    "data/reference/nifty50_classification.csv"
)

PEER_GROUP_FILE = (
    "data/reference/peer_groups.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("=" * 60)
    print("PEER ANALYSIS AUDIT")
    print("=" * 60)

    print("\nLOADING DATASETS")

    analysis_df = pd.read_csv(
        ANALYSIS_FILE
    )

    classification_df = pd.read_csv(
        CLASSIFICATION_FILE
    )

    peer_groups_df = pd.read_csv(
        PEER_GROUP_FILE
    )

    print(
        f"\nAnalysis rows: {len(analysis_df)}"
    )

    print(
        f"Classification rows: "
        f"{len(classification_df)}"
    )

    print(
        f"Peer-group rows: "
        f"{len(peer_groups_df)}"
    )

    return (
        analysis_df,
        classification_df,
        peer_groups_df,
    )


# ============================================================
# BASIC INPUT CHECKS
# ============================================================

def test_input_data():

    (
        analysis_df,
        classification_df,
        peer_groups_df,
    ) = load_data()

    print("\nINPUT DATA CHECK")

    assert len(analysis_df) == 50

    assert len(classification_df) == 50

    assert len(peer_groups_df) == 29

    assert analysis_df["symbol"].nunique() == 50

    assert classification_df["symbol"].nunique() == 50

    assert (
        classification_df["business_type"]
        .nunique()
        == 29
    )

    assert (
        peer_groups_df["business_type"]
        .nunique()
        == 29
    )

    print("✓ Analysis dataset: 50 stocks")

    print(
        "✓ Classification dataset: 50 stocks"
    )

    print(
        "✓ Classification: 29 business types"
    )

    print(
        "✓ Peer mapping: 29 business types"
    )


# ============================================================
# RUN PEER ANALYSIS
# ============================================================

def run_peer_analysis():

    (
        analysis_df,
        classification_df,
        peer_groups_df,
    ) = load_data()

    result = analyze_peers(
        analysis_df,
        classification_df,
        peer_groups_df,
    )

    return result


# ============================================================
# OUTPUT STRUCTURE CHECK
# ============================================================

def test_peer_analysis_structure():

    result = run_peer_analysis()

    print("\nOUTPUT STRUCTURE")

    assert len(result) == 50

    required_columns = [
        "symbol",
        "sector",
        "business_type",
        "peer_group",
        "peer_group_size",
    ]

    for column in required_columns:

        assert column in result.columns

        print(
            f"✓ {column}"
        )

    for metric in PEER_METRICS:

        expected_columns = [
            f"peer_mean_{metric}",
            f"peer_median_{metric}",
            f"vs_peer_median_{metric}",
            f"peer_ratio_{metric}",
            f"peer_percentile_{metric}",
        ]

        for column in expected_columns:

            assert column in result.columns

    print(
        "\n✓ All peer-relative columns created"
    )


# ============================================================
# JOIN VALIDATION
# ============================================================

def test_peer_group_mapping():

    result = run_peer_analysis()

    print("\nPEER GROUP MAPPING")

    missing_business_types = (
        result["business_type"]
        .isna()
        .sum()
    )

    missing_peer_groups = (
        result["peer_group"]
        .isna()
        .sum()
    )

    print(
        "Missing business types:",
        missing_business_types,
    )

    print(
        "Missing peer groups:",
        missing_peer_groups,
    )

    assert missing_business_types == 0

    assert missing_peer_groups == 0

    print(
        "✓ All 50 stocks have business types"
    )

    print(
        "✓ All 50 stocks have peer groups"
    )


# ============================================================
# PEER GROUP SIZE CHECK
# ============================================================

def test_peer_group_sizes():

    result = run_peer_analysis()

    print("\nPEER GROUP SIZES")

    distribution = (
        result[
            [
                "peer_group",
                "peer_group_size",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "peer_group_size",
            ascending=False,
        )
    )

    print(
        distribution.to_string(
            index=False
        )
    )

    calculated_sizes = (
        result
        .groupby("peer_group")["symbol"]
        .count()
    )

    reported_sizes = (
        result
        .groupby("peer_group")[
            "peer_group_size"
        ]
        .first()
    )

    pd.testing.assert_series_equal(
        calculated_sizes.sort_index(),
        reported_sizes.sort_index(),
        check_names=False,
    )

    print(
        "\n✓ Peer-group sizes are correct"
    )


# ============================================================
# PEER MEDIAN CHECK
# ============================================================

def test_peer_median_calculation():

    result = run_peer_analysis()

    print("\nPEER MEDIAN CHECK")

    # --------------------------------------------------------
    # Pick one peer group with multiple stocks
    # --------------------------------------------------------

    peer_group = "Banks"

    group = result[
        result["peer_group"] == peer_group
    ]

    assert len(group) >= 2

    expected_median = (
        group["cagr"].median()
    )

    calculated_median = (
        group["peer_median_cagr"]
        .iloc[0]
    )

    print(
        f"Peer group: {peer_group}"
    )

    print(
        f"Expected CAGR median: "
        f"{expected_median:.6f}"
    )

    print(
        f"Calculated CAGR median: "
        f"{calculated_median:.6f}"
    )

    assert (
        abs(
            expected_median
            - calculated_median
        )
        < 1e-12
    )

    print(
        "✓ Peer median calculation is correct"
    )


# ============================================================
# DIFFERENCE CHECK
# ============================================================

def test_difference_calculation():

    result = run_peer_analysis()

    print("\nDIFFERENCE CHECK")

    row = result[
        result["peer_group"] == "Banks"
    ].iloc[0]

    expected_difference = (
        row["cagr"]
        - row["peer_median_cagr"]
    )

    calculated_difference = (
        row["vs_peer_median_cagr"]
    )

    print(
        f"Stock CAGR: "
        f"{row['cagr']:.6f}"
    )

    print(
        f"Peer median CAGR: "
        f"{row['peer_median_cagr']:.6f}"
    )

    print(
        f"Expected difference: "
        f"{expected_difference:.6f}"
    )

    print(
        f"Calculated difference: "
        f"{calculated_difference:.6f}"
    )

    assert (
        abs(
            expected_difference
            - calculated_difference
        )
        < 1e-12
    )

    print(
        "✓ Difference calculation is correct"
    )


# ============================================================
# PERCENTILE CHECK
# ============================================================

def test_peer_percentile():

    result = run_peer_analysis()

    print("\nPEER PERCENTILE CHECK")

    banks = result[
        result["peer_group"] == "Banks"
    ].copy()

    assert len(banks) >= 2

    expected = (
        banks["cagr"]
        .rank(
            method="average",
            pct=True,
        )
    )

    calculated = (
        banks["peer_percentile_cagr"]
    )

    # --------------------------------------------------------
    # Compare numerical values only.
    #
    # Pandas may represent these as:
    #
    #   float64
    #
    # or:
    #
    #   Float64
    #
    # The dtype difference is not a calculation error.
    # --------------------------------------------------------

    expected_values = (
        expected
        .astype(float)
        .to_numpy()
    )

    calculated_values = (
        calculated
        .astype(float)
        .to_numpy()
    )

    assert (
        len(expected_values)
        == len(calculated_values)
    )

    assert (
        abs(
            expected_values
            - calculated_values
        )
        < 1e-12
    ).all()

    print(
        "✓ Peer percentile calculation is correct"
    )


# ============================================================
# SINGLETON GROUP CHECK
# ============================================================

def test_singleton_peer_groups():

    result = run_peer_analysis()

    print("\nSINGLETON PEER GROUP CHECK")

    singleton_groups = (
        result.loc[
            result["peer_group_size"] == 1,
            "peer_group",
        ]
        .unique()
        .tolist()
    )

    print(
        "Singleton peer groups:"
    )

    for group in singleton_groups:
        print(
            f"- {group}"
        )

    relative_columns = []

    for metric in PEER_METRICS:

        relative_columns.extend(
            [
                f"peer_mean_{metric}",
                f"peer_median_{metric}",
                f"vs_peer_median_{metric}",
                f"peer_ratio_{metric}",
                f"peer_percentile_{metric}",
            ]
        )

    singleton_rows = result[
        result["peer_group_size"] == 1
    ]

    if len(singleton_rows) > 0:

        assert (
            singleton_rows[
                relative_columns
            ]
            .isna()
            .all()
            .all()
        )

    print(
        "✓ Singleton groups do not receive "
        "misleading peer-relative values"
    )


# ============================================================
# FINAL AUDIT
# ============================================================

def main():

    print("=" * 60)
    print("PEER ANALYSIS AUDIT")
    print("=" * 60)

    test_input_data()

    test_peer_analysis_structure()

    test_peer_group_mapping()

    test_peer_group_sizes()

    test_peer_median_calculation()

    test_difference_calculation()

    test_peer_percentile()

    test_singleton_peer_groups()

    print("\n" + "=" * 60)
    print("PEER ANALYSIS AUDIT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
