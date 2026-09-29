"""
Tests for the NIFTY 50 Portfolio Construction Engine.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.portfolio_construction import (
    PortfolioConstructionConfig,
    add_portfolio_columns,
    audit_portfolio,
    construct_portfolio,
)


def make_dataset() -> pd.DataFrame:
    """Create a deterministic synthetic ranked dataset."""

    rows = []

    sectors = [
        "Banking",
        "IT",
        "Energy",
        "Consumer",
        "Healthcare",
    ]

    business_types = [
        "Private Bank",
        "IT Services",
        "Energy Producer",
        "Consumer Goods",
        "Pharmaceuticals",
    ]

    for i in range(20):
        rows.append(
            {
                "symbol": f"STOCK{i + 1:02d}",
                "overall_score": 100.0 - i,
                "overall_rank": i + 1,
                "eligibility": True,
                "sector": sectors[i % len(sectors)],
                "business_type": business_types[
                    i % len(business_types)
                ],
            }
        )

    return pd.DataFrame(rows)


def test_default_configuration():
    config = PortfolioConstructionConfig()

    assert config.target_holdings == 10
    assert config.minimum_holdings == 8
    assert config.maximum_holdings == 12

    assert config.minimum_position_weight == 0.05
    assert config.maximum_position_weight == 0.15

    assert config.maximum_sector_weight == 0.30
    assert config.maximum_business_type_weight == 0.20

    assert config.weighting_method == "score_proportional"
    assert config.selection_method == "rank_order"
    assert config.target_total_weight == 1.0


def test_constructs_target_number_of_holdings():
    dataset = make_dataset()

    portfolio = construct_portfolio(dataset)

    assert len(portfolio) == 10


def test_selected_symbols_are_unique():
    dataset = make_dataset()

    portfolio = construct_portfolio(dataset)

    assert portfolio["symbol"].nunique() == len(portfolio)


def test_only_eligible_stocks_are_selected():
    dataset = make_dataset()

    dataset.loc[
        dataset["overall_rank"].isin([1, 2, 3]),
        "eligibility",
    ] = False

    portfolio = construct_portfolio(dataset)

    assert portfolio["eligibility"].apply(
        lambda value: value is True
        or str(value).lower() == "true"
    ).all()

    assert not portfolio["symbol"].isin(
        ["STOCK01", "STOCK02", "STOCK03"]
    ).any()


def test_selection_follows_rank_order():
    dataset = make_dataset()

    portfolio = construct_portfolio(dataset)

    ranks = portfolio["overall_rank"].tolist()

    assert ranks == sorted(ranks)


def test_weights_are_positive():
    dataset = make_dataset()

    portfolio = construct_portfolio(dataset)

    assert (portfolio["weight"] > 0).all()


def test_weights_sum_to_one():
    dataset = make_dataset()

    portfolio = construct_portfolio(dataset)

    assert np.isclose(
        portfolio["weight"].sum(),
        1.0,
    )


def test_minimum_position_weight():
    dataset = make_dataset()

    portfolio = construct_portfolio(dataset)

    assert (
        portfolio["weight"] >= 0.05 - 1e-9
    ).all()


def test_maximum_position_weight():
    dataset = make_dataset()

    portfolio = construct_portfolio(dataset)

    assert (
        portfolio["weight"] <= 0.15 + 1e-9
    ).all()


def test_sector_constraint():
    dataset = make_dataset()

    portfolio = construct_portfolio(dataset)

    sector_weights = (
        portfolio.groupby("sector")["weight"].sum()
    )

    assert (
        sector_weights <= 0.30 + 1e-9
    ).all()


def test_business_type_constraint():
    dataset = make_dataset()

    portfolio = construct_portfolio(dataset)

    business_weights = (
        portfolio.groupby("business_type")["weight"].sum()
    )

    assert (
        business_weights <= 0.20 + 1e-9
    ).all()


def test_score_proportionality():
    dataset = make_dataset()

    portfolio = construct_portfolio(dataset)

    # With these scores and 10 holdings, the highest-scoring
    # stocks should not receive less weight than lower-scoring
    # stocks unless a weight cap has been reached.
    ordered = portfolio.sort_values(
        "overall_score",
        ascending=False,
    )

    weights = ordered["weight"].to_numpy()

    assert all(
        weights[i] >= weights[i + 1] - 1e-9
        for i in range(len(weights) - 1)
    )


def test_unselected_stocks_receive_zero_weight():
    dataset = make_dataset()

    portfolio = construct_portfolio(dataset)

    full = add_portfolio_columns(
        dataset,
        portfolio,
    )

    unselected = full[
        ~full["selected"]
    ]

    assert (unselected["weight"] == 0.0).all()


def test_selected_stocks_receive_positive_weight():
    dataset = make_dataset()

    portfolio = construct_portfolio(dataset)

    full = add_portfolio_columns(
        dataset,
        portfolio,
    )

    selected = full[
        full["selected"]
    ]

    assert (selected["weight"] > 0).all()


def test_audit_passes_for_valid_portfolio():
    dataset = make_dataset()

    portfolio = construct_portfolio(dataset)

    audit = audit_portfolio(portfolio)

    assert all(audit.values())


def test_missing_required_column_fails():
    dataset = make_dataset().drop(
        columns=["overall_score"]
    )

    with pytest.raises(ValueError):
        construct_portfolio(dataset)


def test_duplicate_symbols_fail():
    dataset = make_dataset()

    dataset.loc[
        1,
        "symbol",
    ] = dataset.loc[0, "symbol"]

    with pytest.raises(ValueError):
        construct_portfolio(dataset)


def test_empty_dataset_fails():
    dataset = make_dataset().iloc[0:0]

    with pytest.raises(ValueError):
        construct_portfolio(dataset)


def test_insufficient_eligible_stocks_fail():
    dataset = make_dataset()

    dataset["eligibility"] = False

    with pytest.raises(ValueError):
        construct_portfolio(dataset)


def test_zero_score_fails():
    dataset = make_dataset()

    dataset.loc[
        0,
        "overall_score",
    ] = 0.0

    with pytest.raises(ValueError):
        construct_portfolio(dataset)


def test_negative_score_fails():
    dataset = make_dataset()

    dataset.loc[
        0,
        "overall_score",
    ] = -1.0

    with pytest.raises(ValueError):
        construct_portfolio(dataset)


def test_unsupported_weighting_method_fails():
    dataset = make_dataset()

    config = PortfolioConstructionConfig(
        weighting_method="equal_weight",
    )

    with pytest.raises(ValueError):
        construct_portfolio(
            dataset,
            config,
        )


def test_unsupported_selection_method_fails():
    dataset = make_dataset()

    config = PortfolioConstructionConfig(
        selection_method="top_score",
    )

    with pytest.raises(ValueError):
        construct_portfolio(
            dataset,
            config,
        )


def test_holdings_respect_maximum_holdings():
    dataset = make_dataset()

    config = PortfolioConstructionConfig(
        target_holdings=10,
        maximum_holdings=12,
    )

    portfolio = construct_portfolio(
        dataset,
        config,
    )

    assert len(portfolio) <= 12


def test_audit_contains_expected_checks():
    dataset = make_dataset()

    portfolio = construct_portfolio(dataset)

    audit = audit_portfolio(portfolio)

    expected = {
        "holding_count",
        "target_holdings",
        "unique_symbols",
        "minimum_weight",
        "maximum_weight",
        "sector_constraint",
        "business_type_constraint",
        "total_weight",
    }

    assert set(audit.keys()) == expected


def test_full_dataset_preserves_original_rows():
    dataset = make_dataset()

    portfolio = construct_portfolio(dataset)

    full = add_portfolio_columns(
        dataset,
        portfolio,
    )

    assert len(full) == len(dataset)
    assert set(full["symbol"]) == set(dataset["symbol"])


def test_full_dataset_has_exactly_target_selected():
    dataset = make_dataset()

    portfolio = construct_portfolio(dataset)

    full = add_portfolio_columns(
        dataset,
        portfolio,
    )

    assert full["selected"].sum() == 10


def test_full_dataset_weights_sum_to_one():
    dataset = make_dataset()

    portfolio = construct_portfolio(dataset)

    full = add_portfolio_columns(
        dataset,
        portfolio,
    )

    assert np.isclose(
        full["weight"].sum(),
        1.0,
    )


def test_weight_order_is_consistent():
    dataset = make_dataset()

    portfolio = construct_portfolio(dataset)

    portfolio = portfolio.sort_values(
        "overall_score",
        ascending=False,
    )

    for first, second in zip(
        portfolio["weight"].iloc[:-1],
        portfolio["weight"].iloc[1:],
    ):
        assert first >= second - 1e-9
