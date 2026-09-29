import pandas as pd

from src.analysis.ranking_specification import (
    RANKING_COLUMNS,
    RANKING_SPECIFICATION,
    calculate_percentiles,
    calculate_ranks,
    load_dataset,
    validate_ranking_specification,
)


def test_dataset_loads():
    df = load_dataset()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50


def test_required_ranking_columns_exist():
    df = load_dataset()

    required = ["symbol"] + RANKING_COLUMNS

    for column in required:
        assert column in df.columns


def test_symbols_are_unique():
    df = load_dataset()

    assert df["symbol"].is_unique


def test_ranking_scores_have_no_missing_values():
    df = load_dataset()

    assert df[RANKING_COLUMNS].isna().sum().sum() == 0


def test_ranking_scores_are_0_to_100():
    df = load_dataset()

    assert (df[RANKING_COLUMNS] >= 0).all().all()
    assert (df[RANKING_COLUMNS] <= 100).all().all()


def test_ranks_are_generated():
    df = load_dataset()
    ranked = calculate_ranks(df)

    for column in RANKING_COLUMNS:
        rank_column = f"{column}_rank"

        assert rank_column in ranked.columns
        assert ranked[rank_column].notna().all()
        assert (ranked[rank_column] >= 1).all()
        assert (ranked[rank_column] <= len(df)).all()


def test_higher_score_gets_better_rank():
    df = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "overall_score": [90, 50, 10],
            "performance_score": [90, 50, 10],
            "risk-adjusted_performance_score": [90, 50, 10],
            "risk_score": [90, 50, 10],
            "quality_score": [90, 50, 10],
            "growth_score": [90, 50, 10],
        }
    )

    ranked = calculate_ranks(df)

    assert ranked.loc[0, "overall_score_rank"] == 1
    assert ranked.loc[1, "overall_score_rank"] == 2
    assert ranked.loc[2, "overall_score_rank"] == 3


def test_percentiles_are_0_to_100():
    df = load_dataset()
    result = calculate_percentiles(df)

    for column in RANKING_COLUMNS:
        percentile_column = f"{column}_percentile"

        assert percentile_column in result.columns
        assert result[percentile_column].notna().all()
        assert (result[percentile_column] >= 0).all()
        assert (result[percentile_column] <= 100).all()


def test_specification_uses_overall_score():
    assert (
        RANKING_SPECIFICATION["primary_ranking_metric"]
        == "overall_score"
    )


def test_specification_ranks_higher_scores_first():
    assert (
        RANKING_SPECIFICATION["rank_direction"]
        == "higher_score_better"
    )


def test_full_ranking_specification_validation():
    assert validate_ranking_specification() is True


if __name__ == "__main__":
    print("=" * 70)
    print("RANKING SPECIFICATION TEST")
    print("=" * 70)

    test_dataset_loads()
    print("PASS: dataset loads")

    test_required_ranking_columns_exist()
    print("PASS: required columns")

    test_symbols_are_unique()
    print("PASS: unique symbols")

    test_ranking_scores_have_no_missing_values()
    print("PASS: no missing scores")

    test_ranking_scores_are_0_to_100()
    print("PASS: score ranges")

    test_ranks_are_generated()
    print("PASS: ranks generated")

    test_higher_score_gets_better_rank()
    print("PASS: ranking direction")

    test_percentiles_are_0_to_100()
    print("PASS: percentiles")

    test_specification_uses_overall_score()
    print("PASS: overall score is primary")

    test_specification_ranks_higher_scores_first()
    print("PASS: higher score ranks first")

    test_full_ranking_specification_validation()
    print("PASS: full specification validation")

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
