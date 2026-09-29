"""
Tests for the NIFTY 50 eligibility engine.
"""

import pandas as pd
import pytest

from src.analysis.eligibility import (
    ELIGIBILITY_REASON_COLUMN,
    ELIGIBLE_COLUMN,
    OVERALL_SCORE_COLUMN,
    PERCENTILE_COLUMN,
    RANK_COLUMN,
    SYMBOL_COLUMN,
    calculate_eligibility,
    check_percentile_eligibility,
    check_rank_eligibility,
    check_score_eligibility,
    get_eligibility_reason,
    get_eligibility_specification,
    load_ranked_dataset,
    run_eligibility,
    save_eligible_dataset,
    validate_input,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_ranked() -> pd.DataFrame:
    """
    Small deterministic ranked dataset.
    """

    return pd.DataFrame(
        {
            SYMBOL_COLUMN: [
                "AAA",
                "BBB",
                "CCC",
                "DDD",
                "EEE",
            ],
            OVERALL_SCORE_COLUMN: [
                95.0,
                80.0,
                65.0,
                50.0,
                35.0,
            ],
            RANK_COLUMN: [
                1,
                2,
                3,
                4,
                5,
            ],
            PERCENTILE_COLUMN: [
                100.0,
                75.0,
                50.0,
                25.0,
                0.0,
            ],
        }
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_input_passes(sample_ranked):
    validate_input(sample_ranked)


def test_validate_input_rejects_missing_symbol(sample_ranked):
    df = sample_ranked.drop(columns=[SYMBOL_COLUMN])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_input(df)


def test_validate_input_rejects_missing_score(sample_ranked):
    df = sample_ranked.drop(columns=[OVERALL_SCORE_COLUMN])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_input(df)


def test_validate_input_rejects_missing_rank(sample_ranked):
    df = sample_ranked.drop(columns=[RANK_COLUMN])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_input(df)


def test_validate_input_rejects_missing_percentile(sample_ranked):
    df = sample_ranked.drop(columns=[PERCENTILE_COLUMN])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_input(df)


def test_validate_input_rejects_empty_dataset():
    df = pd.DataFrame(
        columns=[
            SYMBOL_COLUMN,
            OVERALL_SCORE_COLUMN,
            RANK_COLUMN,
            PERCENTILE_COLUMN,
        ]
    )

    with pytest.raises(ValueError, match="empty"):
        validate_input(df)


def test_validate_input_rejects_duplicate_symbols(sample_ranked):
    df = sample_ranked.copy()
    df.loc[1, SYMBOL_COLUMN] = "AAA"

    with pytest.raises(ValueError, match="duplicate"):
        validate_input(df)


def test_validate_input_rejects_missing_values(sample_ranked):
    df = sample_ranked.copy()
    df.loc[0, OVERALL_SCORE_COLUMN] = None

    with pytest.raises(ValueError, match="missing values"):
        validate_input(df)


def test_validate_input_rejects_invalid_scores(sample_ranked):
    df = sample_ranked.copy()
    df.loc[0, OVERALL_SCORE_COLUMN] = 101

    with pytest.raises(ValueError, match="outside 0-100"):
        validate_input(df)


def test_validate_input_rejects_invalid_percentile(sample_ranked):
    df = sample_ranked.copy()
    df.loc[0, PERCENTILE_COLUMN] = -1

    with pytest.raises(ValueError, match="outside 0-100"):
        validate_input(df)


def test_validate_input_rejects_invalid_rank(sample_ranked):
    df = sample_ranked.copy()
    df.loc[0, RANK_COLUMN] = 0

    with pytest.raises(ValueError, match="invalid ranks"):
        validate_input(df)


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------


def test_eligibility_specification_contains_expected_keys():
    specification = get_eligibility_specification()

    assert set(specification.keys()) == {
        "max_rank",
        "min_score",
        "min_percentile",
    }


def test_default_specification_has_no_hidden_thresholds():
    specification = get_eligibility_specification()

    assert specification["max_rank"] is None
    assert specification["min_score"] is None
    assert specification["min_percentile"] is None


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def test_rank_check_passes_when_disabled(sample_ranked):
    row = sample_ranked.iloc[0]

    assert check_rank_eligibility(row) is True


def test_score_check_passes_when_disabled(sample_ranked):
    row = sample_ranked.iloc[0]

    assert check_score_eligibility(row) is True


def test_percentile_check_passes_when_disabled(sample_ranked):
    row = sample_ranked.iloc[0]

    assert check_percentile_eligibility(row) is True


# ---------------------------------------------------------------------------
# Eligibility calculation
# ---------------------------------------------------------------------------


def test_eligibility_columns_are_added(sample_ranked):
    result = calculate_eligibility(sample_ranked)

    assert ELIGIBLE_COLUMN in result.columns
    assert ELIGIBILITY_REASON_COLUMN in result.columns


def test_all_stocks_are_eligible_when_rules_are_disabled(sample_ranked):
    result = calculate_eligibility(sample_ranked)

    assert result[ELIGIBLE_COLUMN].all()


def test_all_stocks_have_eligible_reason_when_rules_are_disabled(
    sample_ranked,
):
    result = calculate_eligibility(sample_ranked)

    assert (
        result[ELIGIBILITY_REASON_COLUMN] == "eligible"
    ).all()


def test_original_columns_are_preserved(sample_ranked):
    result = calculate_eligibility(sample_ranked)

    for column in sample_ranked.columns:
        assert column in result.columns


def test_row_count_is_preserved(sample_ranked):
    result = calculate_eligibility(sample_ranked)

    assert len(result) == len(sample_ranked)


def test_symbols_are_preserved(sample_ranked):
    result = calculate_eligibility(sample_ranked)

    assert set(result[SYMBOL_COLUMN]) == set(
        sample_ranked[SYMBOL_COLUMN]
    )


# ---------------------------------------------------------------------------
# Reason generation
# ---------------------------------------------------------------------------


def test_eligible_row_has_eligible_reason(sample_ranked):
    row = sample_ranked.iloc[0]

    assert get_eligibility_reason(row) == "eligible"


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------


def test_save_eligible_dataset(sample_ranked, tmp_path):
    output_file = tmp_path / "eligible.csv"

    result = calculate_eligibility(sample_ranked)

    save_eligible_dataset(
        result,
        output_file,
    )

    assert output_file.exists()

    loaded = pd.read_csv(output_file)

    assert len(loaded) == len(result)

    assert ELIGIBLE_COLUMN in loaded.columns
    assert ELIGIBILITY_REASON_COLUMN in loaded.columns


def test_load_ranked_dataset(sample_ranked, tmp_path):
    input_file = tmp_path / "ranked.csv"

    sample_ranked.to_csv(
        input_file,
        index=False,
    )

    loaded = load_ranked_dataset(input_file)

    pd.testing.assert_frame_equal(
        loaded,
        sample_ranked,
    )


def test_load_ranked_dataset_missing_file(tmp_path):
    input_file = tmp_path / "missing.csv"

    with pytest.raises(
        FileNotFoundError,
        match="Ranked dataset not found",
    ):
        load_ranked_dataset(input_file)


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def test_run_eligibility(sample_ranked, tmp_path):
    input_file = tmp_path / "ranked.csv"
    output_file = tmp_path / "eligible.csv"

    sample_ranked.to_csv(
        input_file,
        index=False,
    )

    result = run_eligibility(
        input_file=input_file,
        output_file=output_file,
    )

    assert output_file.exists()

    assert len(result) == 5

    assert ELIGIBLE_COLUMN in result.columns
    assert ELIGIBILITY_REASON_COLUMN in result.columns

    assert result[ELIGIBLE_COLUMN].all()


# ---------------------------------------------------------------------------
# Real ranked dataset
# ---------------------------------------------------------------------------


def test_real_ranked_dataset_exists():
    from src.analysis.eligibility import INPUT_FILE

    assert INPUT_FILE.exists(), (
        f"Expected ranked dataset not found: {INPUT_FILE}"
    )


def test_real_ranked_dataset_can_be_processed():
    from src.analysis.eligibility import INPUT_FILE

    if not INPUT_FILE.exists():
        pytest.skip("Ranked dataset not available.")

    df = load_ranked_dataset(INPUT_FILE)

    result = calculate_eligibility(df)

    assert len(result) == 50

    assert result[SYMBOL_COLUMN].is_unique

    assert result[ELIGIBLE_COLUMN].notna().all()

    assert result[ELIGIBILITY_REASON_COLUMN].notna().all()

    assert result[ELIGIBLE_COLUMN].dtype == bool
