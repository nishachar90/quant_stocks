"""
Tests for the NIFTY 50 ranking engine.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.analysis.ranking import (
    INPUT_FILE,
    OUTPUT_FILE,
    OVERALL_SCORE_COLUMN,
    PERCENTILE_COLUMN,
    RANK_COLUMN,
    SYMBOL_COLUMN,
    calculate_ranked_dataset,
    load_factor_scores,
    run_ranking,
    save_ranked_dataset,
    validate_input,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_scores() -> pd.DataFrame:
    """
    Small deterministic factor-score dataset.
    """

    return pd.DataFrame(
        {
            "symbol": [
                "AAA",
                "BBB",
                "CCC",
                "DDD",
                "EEE",
            ],
            "overall_score": [
                80.0,
                60.0,
                95.0,
                60.0,
                40.0,
            ],
        }
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_validate_input_passes(sample_scores):
    validate_input(sample_scores)


def test_validate_input_rejects_missing_symbol(sample_scores):
    df = sample_scores.drop(columns=[SYMBOL_COLUMN])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_input(df)


def test_validate_input_rejects_missing_overall_score(sample_scores):
    df = sample_scores.drop(columns=[OVERALL_SCORE_COLUMN])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_input(df)


def test_validate_input_rejects_empty_dataset():
    df = pd.DataFrame(
        columns=[
            SYMBOL_COLUMN,
            OVERALL_SCORE_COLUMN,
        ]
    )

    with pytest.raises(ValueError, match="empty"):
        validate_input(df)


def test_validate_input_rejects_duplicate_symbols(sample_scores):
    df = sample_scores.copy()
    df.loc[1, SYMBOL_COLUMN] = "AAA"

    with pytest.raises(ValueError, match="duplicate"):
        validate_input(df)


def test_validate_input_rejects_missing_scores(sample_scores):
    df = sample_scores.copy()
    df.loc[0, OVERALL_SCORE_COLUMN] = None

    with pytest.raises(ValueError, match="missing values"):
        validate_input(df)


def test_validate_input_rejects_scores_below_zero(sample_scores):
    df = sample_scores.copy()
    df.loc[0, OVERALL_SCORE_COLUMN] = -1

    with pytest.raises(ValueError, match="outside 0-100"):
        validate_input(df)


def test_validate_input_rejects_scores_above_100(sample_scores):
    df = sample_scores.copy()
    df.loc[0, OVERALL_SCORE_COLUMN] = 101

    with pytest.raises(ValueError, match="outside 0-100"):
        validate_input(df)


def test_validate_input_rejects_non_numeric_scores(sample_scores):
    df = sample_scores.copy()
    df[OVERALL_SCORE_COLUMN] = "invalid"

    with pytest.raises(ValueError, match="must be numeric"):
        validate_input(df)


# ---------------------------------------------------------------------------
# Ranking behavior
# ---------------------------------------------------------------------------


def test_ranking_columns_are_added(sample_scores):
    ranked = calculate_ranked_dataset(sample_scores)

    assert RANK_COLUMN in ranked.columns
    assert PERCENTILE_COLUMN in ranked.columns


def test_ranking_is_descending(sample_scores):
    ranked = calculate_ranked_dataset(sample_scores)

    assert ranked[SYMBOL_COLUMN].tolist() == [
        "CCC",
        "AAA",
        "BBB",
        "DDD",
        "EEE",
    ]


def test_highest_score_gets_rank_one(sample_scores):
    ranked = calculate_ranked_dataset(sample_scores)

    top_row = ranked.iloc[0]

    assert top_row[OVERALL_SCORE_COLUMN] == 95.0
    assert top_row[RANK_COLUMN] == 1


def test_lowest_score_gets_worst_rank(sample_scores):
    ranked = calculate_ranked_dataset(sample_scores)

    bottom_row = ranked.iloc[-1]

    assert bottom_row[OVERALL_SCORE_COLUMN] == 40.0
    assert bottom_row[RANK_COLUMN] == 5


def test_tied_scores_receive_same_rank(sample_scores):
    ranked = calculate_ranked_dataset(sample_scores)

    bbb_rank = ranked.loc[
        ranked[SYMBOL_COLUMN] == "BBB",
        RANK_COLUMN,
    ].iloc[0]

    ddd_rank = ranked.loc[
        ranked[SYMBOL_COLUMN] == "DDD",
        RANK_COLUMN,
    ].iloc[0]

    assert bbb_rank == 3
    assert ddd_rank == 3


def test_competition_ranking_leaves_rank_gap(sample_scores):
    ranked = calculate_ranked_dataset(sample_scores)

    assert ranked[RANK_COLUMN].tolist() == [
        1,
        2,
        3,
        3,
        5,
    ]


def test_ranking_is_sorted_by_rank(sample_scores):
    ranked = calculate_ranked_dataset(sample_scores)

    ranks = ranked[RANK_COLUMN].tolist()

    assert ranks == sorted(ranks)


def test_tied_scores_are_sorted_by_symbol(sample_scores):
    ranked = calculate_ranked_dataset(sample_scores)

    tied = ranked[
        ranked[OVERALL_SCORE_COLUMN] == 60.0
    ]

    assert tied[SYMBOL_COLUMN].tolist() == [
        "BBB",
        "DDD",
    ]


# ---------------------------------------------------------------------------
# Percentile behavior
# ---------------------------------------------------------------------------


def test_highest_score_gets_100_percentile(sample_scores):
    ranked = calculate_ranked_dataset(sample_scores)

    top_percentile = ranked.iloc[0][PERCENTILE_COLUMN]

    assert top_percentile == 100.0


def test_lowest_score_gets_zero_percentile(sample_scores):
    ranked = calculate_ranked_dataset(sample_scores)

    bottom_percentile = ranked.iloc[-1][PERCENTILE_COLUMN]

    assert bottom_percentile == 0.0


def test_percentiles_are_between_zero_and_hundred(sample_scores):
    ranked = calculate_ranked_dataset(sample_scores)

    assert ranked[PERCENTILE_COLUMN].between(0, 100).all()


def test_single_stock_gets_100_percentile():
    df = pd.DataFrame(
        {
            SYMBOL_COLUMN: ["AAA"],
            OVERALL_SCORE_COLUMN: [75.0],
        }
    )

    ranked = calculate_ranked_dataset(df)

    assert ranked.iloc[0][RANK_COLUMN] == 1
    assert ranked.iloc[0][PERCENTILE_COLUMN] == 100.0


# ---------------------------------------------------------------------------
# Data preservation
# ---------------------------------------------------------------------------


def test_original_columns_are_preserved(sample_scores):
    ranked = calculate_ranked_dataset(sample_scores)

    for column in sample_scores.columns:
        assert column in ranked.columns


def test_original_row_count_is_preserved(sample_scores):
    ranked = calculate_ranked_dataset(sample_scores)

    assert len(ranked) == len(sample_scores)


def test_symbols_are_preserved(sample_scores):
    ranked = calculate_ranked_dataset(sample_scores)

    assert set(ranked[SYMBOL_COLUMN]) == set(
        sample_scores[SYMBOL_COLUMN]
    )


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------


def test_save_ranked_dataset(sample_scores, tmp_path):
    output_file = tmp_path / "ranked.csv"

    save_ranked_dataset(
        sample_scores,
        output_file,
    )

    assert output_file.exists()

    loaded = pd.read_csv(output_file)

    assert len(loaded) == len(sample_scores)
    assert list(loaded.columns) == list(sample_scores.columns)


def test_load_factor_scores(sample_scores, tmp_path):
    input_file = tmp_path / "factor_scores.csv"

    sample_scores.to_csv(
        input_file,
        index=False,
    )

    loaded = load_factor_scores(input_file)

    pd.testing.assert_frame_equal(
        loaded,
        sample_scores,
    )


def test_load_factor_scores_missing_file(tmp_path):
    input_file = tmp_path / "does_not_exist.csv"

    with pytest.raises(
        FileNotFoundError,
        match="Factor-score dataset not found",
    ):
        load_factor_scores(input_file)


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def test_run_ranking(sample_scores, tmp_path):
    input_file = tmp_path / "factor_scores.csv"
    output_file = tmp_path / "ranked.csv"

    sample_scores.to_csv(
        input_file,
        index=False,
    )

    ranked = run_ranking(
        input_file=input_file,
        output_file=output_file,
    )

    assert output_file.exists()
    assert len(ranked) == 5

    assert ranked.iloc[0][SYMBOL_COLUMN] == "CCC"
    assert ranked.iloc[0][RANK_COLUMN] == 1

    saved = pd.read_csv(output_file)

    assert len(saved) == 5
    assert RANK_COLUMN in saved.columns
    assert PERCENTILE_COLUMN in saved.columns


# ---------------------------------------------------------------------------
# Real factor-score dataset
# ---------------------------------------------------------------------------


def test_real_factor_score_dataset_exists():
    assert INPUT_FILE.exists(), (
        f"Expected factor-score dataset not found: {INPUT_FILE}"
    )


def test_real_factor_score_dataset_can_be_ranked():
    if not INPUT_FILE.exists():
        pytest.skip("Factor-score dataset not available.")

    df = load_factor_scores(INPUT_FILE)
    ranked = calculate_ranked_dataset(df)

    assert len(ranked) == 50

    assert ranked[SYMBOL_COLUMN].is_unique

    assert ranked[RANK_COLUMN].notna().all()

    assert ranked[PERCENTILE_COLUMN].notna().all()

    assert ranked[OVERALL_SCORE_COLUMN].between(
        0,
        100,
    ).all()

    assert ranked[RANK_COLUMN].min() == 1

    assert ranked[RANK_COLUMN].max() <= 50

    assert ranked[PERCENTILE_COLUMN].between(
        0,
        100,
    ).all()


def test_real_factor_score_dataset_has_single_top_rank():
    if not INPUT_FILE.exists():
        pytest.skip("Factor-score dataset not available.")

    df = load_factor_scores(INPUT_FILE)
    ranked = calculate_ranked_dataset(df)

    assert (ranked[RANK_COLUMN] == 1).sum() == 1


def test_real_factor_score_dataset_is_descending():
    if not INPUT_FILE.exists():
        pytest.skip("Factor-score dataset not available.")

    df = load_factor_scores(INPUT_FILE)
    ranked = calculate_ranked_dataset(df)

    scores = ranked[OVERALL_SCORE_COLUMN].tolist()

    assert scores == sorted(
        scores,
        reverse=True,
    )
