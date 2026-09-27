import pandas as pd

from src.analysis.peer_analysis import analyze_peers


# ============================================================
# FILE PATHS
# ============================================================

ANALYSIS_FILE = (
    "data/analysis/nifty50_analysis.csv"
)

CLASSIFICATION_FILE = (
    "data/reference/nifty50_classification.csv"
)

PEER_GROUP_FILE = (
    "data/reference/peer_groups.csv"
)

OUTPUT_FILE = (
    "data/analysis/nifty50_peer_analysis.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 60)
    print("NIFTY 50 PEER ANALYSIS")
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
        f"Analysis rows: "
        f"{len(analysis_df)}"
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
# RUN PEER ANALYSIS
# ============================================================

def generate_peer_analysis():

    (
        analysis_df,
        classification_df,
        peer_groups_df,
    ) = load_data()

    print("\nRUNNING PEER ANALYSIS")

    peer_analysis_df = analyze_peers(
        analysis_df,
        classification_df,
        peer_groups_df,
    )

    print(
        f"✓ Peer analysis completed"
    )

    print(
        f"✓ Output rows: "
        f"{len(peer_analysis_df)}"
    )

    return peer_analysis_df


# ============================================================
# SAVE DATASET
# ============================================================

def save_peer_analysis(
    peer_analysis_df
):

    peer_analysis_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\n✓ Saved peer-analysis dataset:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )


# ============================================================
# DISPLAY SUMMARY
# ============================================================

def display_summary(
    peer_analysis_df
):

    print("\n" + "=" * 60)
    print("PEER ANALYSIS SUMMARY")
    print("=" * 60)

    print(
        f"\nRows: "
        f"{len(peer_analysis_df)}"
    )

    print(
        f"Columns: "
        f"{len(peer_analysis_df.columns)}"
    )

    print("\nPEER GROUP DISTRIBUTION")

    distribution = (
        peer_analysis_df[
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

    print("\nOUTPUT COLUMNS")

    for column in peer_analysis_df.columns:

        print(
            f"- {column}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    peer_analysis_df = (
        generate_peer_analysis()
    )

    save_peer_analysis(
        peer_analysis_df
    )

    display_summary(
        peer_analysis_df
    )

    print("\n" + "=" * 60)
    print("PEER ANALYSIS DATASET CREATED")
    print("=" * 60)


if __name__ == "__main__":
    main()
