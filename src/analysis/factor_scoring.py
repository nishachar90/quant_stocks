"""
Multi-Factor Scoring Engine
===========================

Converts the metrics defined by factor_scoring_specification.py
into comparable 0-100 scores.

This module:

1. Loads the peer-analysis dataset.
2. Selects an appropriate reference group for each stock.
3. Calculates percentile-based metric scores.
4. Handles higher-is-better and lower-is-better metrics.
5. Calculates weighted factor scores.
6. Calculates the overall multi-factor score.
7. Preserves audit information about reference groups.
8. Saves the resulting dataset.

Reference hierarchy:

    1. Peer group
    2. Sector
    3. Full NIFTY 50 universe

A reference group must contain at least
MIN_PEER_GROUP_SIZE observations.

Missing metric values are not imputed.
Available metric weights are renormalized.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.factor_scoring_specification import (
    FACTOR_WEIGHTS,
    METRIC_SPECIFICATION,
    MIN_PEER_GROUP_SIZE,
    PEER_ANALYSIS_FILE,
    calculate_role_weights,
    load_dataset,
    metric_score_direction,
    validate_specification,
)


# ======================================================================
# PATHS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "nifty50_factor_scores.csv"
)


# ======================================================================
# PERCENTILE SCORING
# ======================================================================

def percentile_score(
    value: float,
    reference_values: pd.Series,
    direction: str,
) -> float:
    """
    Convert one observation into a 0-100 percentile score.

    Parameters
    ----------
    value:
        The stock's metric value.

    reference_values:
        All valid observations in the selected reference group.

    direction:
        Either:
            higher_is_better
            lower_is_better

    Returns
    -------
    float
        Score between 0 and 100.

    Notes
    -----
    Percentile ranking uses average ranks for ties.

    For example:

        [10, 20, 20, 30]

    The two observations equal to 20 receive the average rank
    between positions 2 and 3.

    Lower-is-better metrics are reversed so that a better
    observation always receives a higher score.
    """

    if pd.isna(value):
        return np.nan

    clean_values = pd.Series(reference_values).dropna()

    if clean_values.empty:
        return np.nan

    ranks = clean_values.rank(
        method="average",
        ascending=True,
    )

    value_rank = ranks[
        np.isclose(
            clean_values.astype(float),
            float(value),
            equal_nan=False,
        )
    ]

    if value_rank.empty:
        return np.nan

    rank = float(value_rank.iloc[0])
    n = len(clean_values)

    if n == 1:
        score = 100.0
    else:
        score = ((rank - 1) / (n - 1)) * 100.0

    if direction == "higher_is_better":
        return float(score)

    if direction == "lower_is_better":
        return float(100.0 - score)

    raise ValueError(
        f"Unsupported direction: {direction}"
    )


# ======================================================================
# REFERENCE GROUP SELECTION
# ======================================================================

def select_reference_group(
    df: pd.DataFrame,
    row_index: int,
) -> tuple[str, str]:
    """
    Select the reference hierarchy for one stock.

    Priority:

        1. Peer group
        2. Sector
        3. Universe

    Returns
    -------
    tuple[str, str]
        (reference_level, reference_name)
    """

    row = df.loc[row_index]

    # --------------------------------------------------------------
    # 1. Peer group
    # --------------------------------------------------------------

    peer_group = row["peer_group"]

    peer_mask = (
        df["peer_group"] == peer_group
    )

    peer_count = int(peer_mask.sum())

    if peer_count >= MIN_PEER_GROUP_SIZE:
        return "peer_group", str(peer_group)

    # --------------------------------------------------------------
    # 2. Sector
    # --------------------------------------------------------------

    sector = row["sector"]

    sector_mask = (
        df["sector"] == sector
    )

    sector_count = int(sector_mask.sum())

    if sector_count >= MIN_PEER_GROUP_SIZE:
        return "sector", str(sector)

    # --------------------------------------------------------------
    # 3. Full universe
    # --------------------------------------------------------------

    return "universe", "NIFTY50"


def build_reference_mask(
    df: pd.DataFrame,
    row_index: int,
    reference_level: str,
    reference_name: str,
) -> pd.Series:
    """
    Build a boolean mask for the selected reference group.
    """

    if reference_level == "peer_group":

        return (
            df["peer_group"] == reference_name
        )

    if reference_level == "sector":

        return (
            df["sector"] == reference_name
        )

    if reference_level == "universe":

        return pd.Series(
            True,
            index=df.index,
        )

    raise ValueError(
        f"Unsupported reference level: {reference_level}"
    )


# ======================================================================
# METRIC SCORING
# ======================================================================

def score_metric(
    df: pd.DataFrame,
    metric: str,
    direction: str,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Score one metric for the complete dataset.

    Returns
    -------
    scores:
        0-100 metric scores.

    reference_levels:
        Reference hierarchy used for each stock.

    reference_names:
        Actual reference group used for each stock.
    """

    scores = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    reference_levels = pd.Series(
        None,
        index=df.index,
        dtype="object",
    )

    reference_names = pd.Series(
        None,
        index=df.index,
        dtype="object",
    )

    for row_index in df.index:

        value = df.loc[row_index, metric]

        if pd.isna(value):
            continue

        reference_level, reference_name = (
            select_reference_group(
                df,
                row_index,
            )
        )

        reference_mask = build_reference_mask(
            df,
            row_index,
            reference_level,
            reference_name,
        )

        reference_values = df.loc[
            reference_mask,
            metric,
        ]

        score = percentile_score(
            value=value,
            reference_values=reference_values,
            direction=direction,
        )

        scores.loc[row_index] = score

        reference_levels.loc[row_index] = (
            reference_level
        )

        reference_names.loc[row_index] = (
            reference_name
        )

    return (
        scores,
        reference_levels,
        reference_names,
    )


# ======================================================================
# METRIC SCORE DATASET
# ======================================================================

def calculate_metric_scores(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate scores for every active metric.
    """

    result = df.copy()

    for factor, metrics in METRIC_SPECIFICATION.items():

        for metric, specification in metrics.items():

            direction = specification["direction"]

            (
                scores,
                reference_levels,
                reference_names,
            ) = score_metric(
                result,
                metric,
                direction,
            )

            result[f"{metric}_score"] = scores

            result[
                f"{metric}_score_reference_level"
            ] = reference_levels

            result[
                f"{metric}_score_reference"
            ] = reference_names

    return result


# ======================================================================
# FACTOR SCORING
# ======================================================================

def calculate_factor_score(
    df: pd.DataFrame,
    factor: str,
    metrics: dict,
) -> pd.Series:
    """
    Calculate the weighted score for one factor.

    Missing metrics are excluded and the remaining metric weights
    are renormalized for each stock.
    """

    normalized_weights = calculate_role_weights(
        metrics
    )

    factor_scores = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    for row_index in df.index:

        weighted_sum = 0.0
        available_weight = 0.0

        for metric, weight in normalized_weights.items():

            score_column = f"{metric}_score"

            score = df.loc[
                row_index,
                score_column,
            ]

            if pd.isna(score):
                continue

            weighted_sum += (
                float(score) * weight
            )

            available_weight += weight

        if available_weight > 0:

            factor_scores.loc[row_index] = (
                weighted_sum / available_weight
            )

    return factor_scores


def calculate_factor_scores(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate all active factor scores.
    """

    result = df.copy()

    for factor, metrics in METRIC_SPECIFICATION.items():

        factor_score = calculate_factor_score(
            result,
            factor,
            metrics,
        )

        factor_column = (
            f"{factor.lower().replace(' ', '_')}_score"
        )

        result[factor_column] = factor_score

    return result


# ======================================================================
# OVERALL MULTI-FACTOR SCORE
# ======================================================================

def calculate_overall_score(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Calculate the overall multi-factor score.

    Factor weights come directly from FACTOR_WEIGHTS.

    Missing factor scores are handled by renormalizing the
    available factor weights.
    """

    overall_scores = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    factor_columns = {}

    for factor in FACTOR_WEIGHTS:

        factor_column = (
            f"{factor.lower().replace(' ', '_')}_score"
        )

        factor_columns[factor] = factor_column

    for row_index in df.index:

        weighted_sum = 0.0
        available_weight = 0.0

        for factor, weight in FACTOR_WEIGHTS.items():

            factor_column = factor_columns[factor]

            score = df.loc[
                row_index,
                factor_column,
            ]

            if pd.isna(score):
                continue

            weighted_sum += (
                float(score) * weight
            )

            available_weight += weight

        if available_weight > 0:

            overall_scores.loc[row_index] = (
                weighted_sum / available_weight
            )

    return overall_scores


# ======================================================================
# AUDIT HELPERS
# ======================================================================

def audit_metric_scores(
    df: pd.DataFrame,
) -> None:
    """
    Print an audit of all metric scores.
    """

    print()
    print("=" * 70)
    print("METRIC SCORE AUDIT")
    print("=" * 70)

    for factor, metrics in METRIC_SPECIFICATION.items():

        for metric in metrics:

            score_column = f"{metric}_score"

            scores = df[score_column]

            available = int(
                scores.notna().sum()
            )

            missing = int(
                scores.isna().sum()
            )

            if available > 0:

                minimum = scores.min()
                maximum = scores.max()
                mean = scores.mean()

                print(
                    f"{metric}: "
                    f"scored={available}, "
                    f"missing={missing}, "
                    f"min={minimum:.2f}, "
                    f"max={maximum:.2f}, "
                    f"mean={mean:.2f}"
                )

            else:

                print(
                    f"{metric}: "
                    f"scored=0, "
                    f"missing={missing}"
                )


def audit_reference_groups(
    df: pd.DataFrame,
) -> None:
    """
    Print an audit of reference-level usage.
    """

    print()
    print("=" * 70)
    print("REFERENCE GROUP AUDIT")
    print("=" * 70)

    for factor, metrics in METRIC_SPECIFICATION.items():

        for metric in metrics:

            column = (
                f"{metric}_score_reference_level"
            )

            counts = (
                df[column]
                .value_counts(dropna=False)
                .to_dict()
            )

            print(
                f"{metric}: {counts}"
            )


def audit_factor_scores(
    df: pd.DataFrame,
) -> None:
    """
    Print an audit of factor-level scores.
    """

    print()
    print("=" * 70)
    print("FACTOR SCORE AUDIT")
    print("=" * 70)

    for factor in METRIC_SPECIFICATION:

        factor_column = (
            f"{factor.lower().replace(' ', '_')}_score"
        )

        scores = df[factor_column]

        available = int(
            scores.notna().sum()
        )

        missing = int(
            scores.isna().sum()
        )

        if available > 0:

            print(
                f"{factor}: "
                f"scored={available}, "
                f"missing={missing}, "
                f"min={scores.min():.2f}, "
                f"max={scores.max():.2f}, "
                f"mean={scores.mean():.2f}"
            )

        else:

            print(
                f"{factor}: "
                f"scored=0, "
                f"missing={missing}"
            )


def audit_overall_score(
    df: pd.DataFrame,
) -> None:
    """
    Print an audit of the overall multi-factor score.
    """

    print()
    print("=" * 70)
    print("OVERALL MULTI-FACTOR SCORE AUDIT")
    print("=" * 70)

    scores = df["overall_score"]

    available = int(
        scores.notna().sum()
    )

    missing = int(
        scores.isna().sum()
    )

    print(
        f"Scored: {available}"
    )

    print(
        f"Missing: {missing}"
    )

    if available > 0:

        print(
            f"Minimum: {scores.min():.2f}"
        )

        print(
            f"Maximum: {scores.max():.2f}"
        )

        print(
            f"Mean: {scores.mean():.2f}"
        )

        print(
            f"Median: {scores.median():.2f}"
        )


# ======================================================================
# VALIDATION
# ======================================================================

def validate_scores(
    df: pd.DataFrame,
) -> None:
    """
    Validate the generated scoring dataset.
    """

    errors = []

    # --------------------------------------------------------------
    # Row count
    # --------------------------------------------------------------

    if len(df) == 0:
        errors.append(
            "Scoring dataset contains zero rows."
        )

    # --------------------------------------------------------------
    # Metric score columns
    # --------------------------------------------------------------

    for factor, metrics in METRIC_SPECIFICATION.items():

        for metric in metrics:

            column = f"{metric}_score"

            if column not in df.columns:

                errors.append(
                    f"Missing metric score column: {column}"
                )

                continue

            valid_scores = df[column].dropna()

            if not valid_scores.empty:

                if (
                    (valid_scores < 0).any()
                    or (valid_scores > 100).any()
                ):

                    errors.append(
                        f"Scores outside 0-100 range: {column}"
                    )

    # --------------------------------------------------------------
    # Factor score columns
    # --------------------------------------------------------------

    for factor in METRIC_SPECIFICATION:

        column = (
            f"{factor.lower().replace(' ', '_')}_score"
        )

        if column not in df.columns:

            errors.append(
                f"Missing factor score column: {column}"
            )

            continue

        valid_scores = df[column].dropna()

        if not valid_scores.empty:

            if (
                (valid_scores < 0).any()
                or (valid_scores > 100).any()
            ):

                errors.append(
                    f"Factor scores outside 0-100 range: {column}"
                )

    # --------------------------------------------------------------
    # Overall score
    # --------------------------------------------------------------

    if "overall_score" not in df.columns:

        errors.append(
            "Missing overall_score column."
        )

    else:

        valid_scores = df["overall_score"].dropna()

        if not valid_scores.empty:

            if (
                (valid_scores < 0).any()
                or (valid_scores > 100).any()
            ):

                errors.append(
                    "Overall scores outside 0-100 range."
                )

    # --------------------------------------------------------------
    # Final result
    # --------------------------------------------------------------

    if errors:

        print()
        print("=" * 70)
        print("SCORING VALIDATION FAILED")
        print("=" * 70)

        for error in errors:
            print(f"- {error}")

        raise ValueError(
            "Factor scoring validation failed."
        )

    print()
    print("=" * 70)
    print("SCORING VALIDATION PASSED")
    print("=" * 70)

    print(
        "All metric scores are within 0-100."
    )

    print(
        "All factor scores are within 0-100."
    )

    print(
        "Overall score is within 0-100."
    )


# ======================================================================
# MAIN PIPELINE
# ======================================================================

def main() -> None:

    print("=" * 70)
    print("NIFTY 50 MULTI-FACTOR SCORING ENGINE")
    print("=" * 70)

    # --------------------------------------------------------------
    # 1. LOAD DATASET
    # --------------------------------------------------------------

    print()
    print("LOADING PEER ANALYSIS DATASET")
    print(f"File: {PEER_ANALYSIS_FILE}")

    df = load_dataset()

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # --------------------------------------------------------------
    # 2. VALIDATE SPECIFICATION
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("1. VALIDATING SCORING SPECIFICATION")
    print("=" * 70)

    validate_specification(df)

    # --------------------------------------------------------------
    # 3. CALCULATE METRIC SCORES
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("2. CALCULATING METRIC SCORES")
    print("=" * 70)

    scored_df = calculate_metric_scores(df)

    print(
        "Metric scoring complete."
    )

    # --------------------------------------------------------------
    # 4. CALCULATE FACTOR SCORES
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("3. CALCULATING FACTOR SCORES")
    print("=" * 70)

    scored_df = calculate_factor_scores(
        scored_df
    )

    print(
        "Factor scoring complete."
    )

    # --------------------------------------------------------------
    # 5. CALCULATE OVERALL SCORE
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("4. CALCULATING OVERALL MULTI-FACTOR SCORE")
    print("=" * 70)

    scored_df["overall_score"] = (
        calculate_overall_score(
            scored_df
        )
    )

    print(
        "Overall scoring complete."
    )

    # --------------------------------------------------------------
    # 6. AUDITS
    # --------------------------------------------------------------

    audit_metric_scores(
        scored_df
    )

    audit_reference_groups(
        scored_df
    )

    audit_factor_scores(
        scored_df
    )

    audit_overall_score(
        scored_df
    )

    # --------------------------------------------------------------
    # 7. VALIDATE RESULTS
    # --------------------------------------------------------------

    validate_scores(
        scored_df
    )

    # --------------------------------------------------------------
    # 8. SAVE OUTPUT
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("5. SAVING OUTPUT")
    print("=" * 70)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scored_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Output file: {OUTPUT_FILE}"
    )

    print(
        f"Rows: {len(scored_df)}"
    )

    print(
        f"Columns: {len(scored_df.columns)}"
    )

    # --------------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("MULTI-FACTOR SCORING COMPLETE")
    print("=" * 70)

    print()
    print(
        "The scoring engine calculated:"
    )

    print(
        "- Metric-level percentile scores"
    )

    print(
        "- Peer/sector/universe reference levels"
    )

    print(
        "- Factor-level weighted scores"
    )

    print(
        "- Overall multi-factor score"
    )

    print()
    print(
        "No stocks were selected or ranked."
    )


if __name__ == "__main__":
    main()
