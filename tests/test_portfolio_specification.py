"""
Tests for the NIFTY 50 portfolio construction specification.
"""

import pandas as pd
import pytest

from src.analysis.portfolio_specification import (
    PORTFOLIO_SPECIFICATION,
    REQUIRED_COLUMNS,
    audit_input_dataset,
    get_portfolio_specification,
    holding_count_is_feasible,
    load_eligible_dataset,
    maximum_weight_feasibility,
    minimum_weight_feasibility,
    run_specification_audit,
    specification_is_feasible,
    validate_input,
    validate_specification,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_eligible() -> pd.DataFrame:
    """
    Small deterministic eligible-stock dataset.
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
                90.0,
                85.0,
                80.0,
                75.0,
                70.0,
            ],
            "overall_rank": [
                1,
                2,
                3,
                4,
                5,
            ],
            "overall_percentile": [
                100.0,
                75.0,
                50.0,
                25.0,
                0.0,
            ],
            "eligible": [
                True,
                True,
                True,
                True,
                True,
            ],
            "sector": [
                "Technology",
                "Financial Services",
                "Healthcare",
                "Technology",
                "Consumer",
            ],
            "business_type": [
                "IT Services",
                "Private Bank",
                "Pharmaceuticals",
                "IT Services",
                "Consumer Goods",
            ],
        }
    )


# ---------------------------------------------------------------------------
# Specification validation
# ---------------------------------------------------------------------------


def test_specification_contains_expected_keys():
    expected_keys = {
        "target_holdings",
        "minimum_holdings",
        "maximum_holdings",
        "minimum_position_weight",
        "maximum_position_weight",
        "maximum_sector_weight",
        "maximum_business_type_weight",
        "weighting_method",
        "selection_method",
        "target_total_weight",
    }

    assert set(PORTFOLIO_SPECIFICATION.keys()) == expected_keys


def test_specification_is_valid():
    validate_specification()


def test_specification_is_feasible():
    assert specification_is_feasible() is True


def test_holding_count_is_feasible():
    assert holding_count_is_feasible() is True


def test_minimum_weight_is_feasible():
    assert minimum_weight_feasibility() is True


def test_maximum_weight_is_feasible():
    assert maximum_weight_feasibility() is True


def test_target_holdings_are_within_range():
    specification = get_portfolio_specification()

    assert (
        specification["minimum_holdings"]
        <= specification["target_holdings"]
        <= specification["maximum_holdings"]
    )


def test_total_weight_is_one():
    specification = get_portfolio_specification()

    assert specification["target_total_weight"] == 1.00


def test_weighting_method_is_supported():
    specification = get_portfolio_specification()

    assert specification["weighting_method"] in {
        "equal_weight",
        "score_proportional",
    }


def test_selection_method_is_supported():
    specification = get_portfolio_specification()

    assert specification["selection_method"] == "rank_order"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_input_contains_all_required_columns(sample_eligible):
    validate_input(sample_eligible)

    assert REQUIRED_COLUMNS.issubset(
        sample_eligible.columns
    )


def test_input_rejects_missing_columns(sample_eligible):
    df = sample_eligible.drop(
        columns=["sector"]
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_input(df)


def test_input_rejects_empty_dataset():
    df = pd.DataFrame(
        columns=list(REQUIRED_COLUMNS)
    )

    with pytest.raises(ValueError, match="empty"):
        validate_input(df)


def test_input_rejects_missing_symbol(sample_eligible):
    df = sample_eligible.copy()
    df.loc[0, "symbol"] = None

    with pytest.raises(ValueError, match="missing symbols"):
        validate_input(df)


def test_input_rejects_duplicate_symbols(sample_eligible):
    df = sample_eligible.copy()
    df.loc[1, "symbol"] = "AAA"

    with pytest.raises(ValueError, match="duplicate symbols"):
        validate_input(df)


def test_input_rejects_missing_score(sample_eligible):
    df = sample_eligible.copy()
    df.loc[0, "overall_score"] = None

    with pytest.raises(ValueError, match="missing values"):
        validate_input(df)


def test_input_rejects_missing_sector(sample_eligible):
    df = sample_eligible.copy()
    df.loc[0, "sector"] = None

    with pytest.raises(ValueError, match="missing values"):
        validate_input(df)


def test_input_rejects_missing_business_type(sample_eligible):
    df = sample_eligible.copy()
    df.loc[0, "business_type"] = None

    with pytest.raises(ValueError, match="missing values"):
        validate_input(df)


# ---------------------------------------------------------------------------
# Dataset audit
# ---------------------------------------------------------------------------


def test_audit_input_dataset(sample_eligible):
    audit = audit_input_dataset(sample_eligible)

    assert audit["rows"] == 5
    assert audit["eligible_count"] == 5
    assert audit["ineligible_count"] == 0
    assert audit["unique_sectors"] == 4
    assert audit["unique_business_types"] == 4
    assert audit["required_holdings"] == 10


def test_audit_counts_ineligible_stocks(sample_eligible):
    df = sample_eligible.copy()

    df.loc[3, "eligible"] = False
    df.loc[4, "eligible"] = False

    audit = audit_input_dataset(df)

    assert audit["eligible_count"] == 3
    assert audit["ineligible_count"] == 2


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------


def test_load_eligible_dataset(sample_eligible, tmp_path):
    input_file = tmp_path / "eligible.csv"

    sample_eligible.to_csv(
        input_file,
        index=False,
    )

    loaded = load_eligible_dataset(input_file)

    pd.testing.assert_frame_equal(
        loaded,
        sample_eligible,
    )


def test_load_eligible_dataset_missing_file(tmp_path):
    input_file = tmp_path / "missing.csv"

    with pytest.raises(
        FileNotFoundError,
        match="Eligible dataset not found",
    ):
        load_eligible_dataset(input_file)


# ---------------------------------------------------------------------------
# Real dataset
# ---------------------------------------------------------------------------


def test_real_eligible_dataset_exists():
    from src.analysis.portfolio_specification import INPUT_FILE

    assert INPUT_FILE.exists(), (
        f"Expected eligible dataset not found: {INPUT_FILE}"
    )


def test_real_eligible_dataset_passes_validation():
    from src.analysis.portfolio_specification import INPUT_FILE

    if not INPUT_FILE.exists():
        pytest.skip("Eligible dataset not available.")

    df = load_eligible_dataset(INPUT_FILE)

    validate_input(df)


def test_real_eligible_dataset_has_50_rows():
    from src.analysis.portfolio_specification import INPUT_FILE

    if not INPUT_FILE.exists():
        pytest.skip("Eligible dataset not available.")

    df = load_eligible_dataset(INPUT_FILE)

    assert len(df) == 50


def test_real_eligible_dataset_has_unique_symbols():
    from src.analysis.portfolio_specification import INPUT_FILE

    if not INPUT_FILE.exists():
        pytest.skip("Eligible dataset not available.")

    df = load_eligible_dataset(INPUT_FILE)

    assert df["symbol"].is_unique


def test_real_eligible_dataset_has_enough_eligible_stocks():
    from src.analysis.portfolio_specification import INPUT_FILE

    if not INPUT_FILE.exists():
        pytest.skip("Eligible dataset not available.")

    df = load_eligible_dataset(INPUT_FILE)

    eligible_count = int(
        df["eligible"].sum()
    )

    assert eligible_count >= (
        PORTFOLIO_SPECIFICATION["minimum_holdings"]
    )


def test_real_specification_audit_passes():
    from src.analysis.portfolio_specification import INPUT_FILE

    if not INPUT_FILE.exists():
        pytest.skip("Eligible dataset not available.")

    audit = run_specification_audit(INPUT_FILE)

    assert audit["rows"] == 50
    assert audit["eligible_count"] >= (
        PORTFOLIO_SPECIFICATION["minimum_holdings"]
    )
