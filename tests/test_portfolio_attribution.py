from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.portfolio_attribution import (
    calculate_peer_group_attribution,
    calculate_sector_attribution,
    calculate_security_attribution,
)


@pytest.fixture
def portfolio() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "weight": [0.50, 0.30, 0.20],
        }
    )


@pytest.fixture
def classification() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "sector": [
                "Technology",
                "Technology",
                "Finance",
            ],
            "peer_group": [
                "IT",
                "IT",
                "Banks",
            ],
        }
    )


@pytest.fixture
def security_returns() -> pd.Series:
    return pd.Series(
        {
            "AAA": 0.10,
            "BBB": 0.05,
            "CCC": -0.02,
        }
    )


def test_security_attribution_columns(
    portfolio,
    classification,
    security_returns,
):
    result = calculate_security_attribution(
        portfolio,
        classification,
        security_returns,
    )

    expected_columns = [
        "symbol",
        "sector",
        "peer_group",
        "beginning_weight",
        "ending_weight",
        "return",
        "return_contribution",
        "return_contribution_pct",
    ]

    assert list(result.columns) == expected_columns


def test_security_attribution_row_count(
    portfolio,
    classification,
    security_returns,
):
    result = calculate_security_attribution(
        portfolio,
        classification,
        security_returns,
    )

    assert len(result) == 3


def test_security_attribution_symbols(
    portfolio,
    classification,
    security_returns,
):
    result = calculate_security_attribution(
        portfolio,
        classification,
        security_returns,
    )

    assert result["symbol"].tolist() == [
        "AAA",
        "BBB",
        "CCC",
    ]


def test_security_attribution_weights(
    portfolio,
    classification,
    security_returns,
):
    result = calculate_security_attribution(
        portfolio,
        classification,
        security_returns,
    )

    expected = np.array([0.50, 0.30, 0.20])

    np.testing.assert_allclose(
        result["beginning_weight"].to_numpy(),
        expected,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        result["ending_weight"].to_numpy(),
        expected,
        atol=1e-12,
    )


def test_security_attribution_returns(
    portfolio,
    classification,
    security_returns,
):
    result = calculate_security_attribution(
        portfolio,
        classification,
        security_returns,
    )

    expected = np.array([
        0.10,
        0.05,
        -0.02,
    ])

    np.testing.assert_allclose(
        result["return"].to_numpy(),
        expected,
        atol=1e-12,
    )


def test_security_return_contributions(
    portfolio,
    classification,
    security_returns,
):
    result = calculate_security_attribution(
        portfolio,
        classification,
        security_returns,
    )

    expected = np.array([
        0.050,
        0.015,
        -0.004,
    ])

    np.testing.assert_allclose(
        result["return_contribution"].to_numpy(),
        expected,
        atol=1e-12,
    )


def test_security_contribution_reconciles_to_portfolio_return(
    portfolio,
    classification,
    security_returns,
):
    result = calculate_security_attribution(
        portfolio,
        classification,
        security_returns,
    )

    expected_portfolio_return = 0.061

    assert np.isclose(
        result["return_contribution"].sum(),
        expected_portfolio_return,
        atol=1e-12,
    )


def test_security_contribution_percentages(
    portfolio,
    classification,
    security_returns,
):
    result = calculate_security_attribution(
        portfolio,
        classification,
        security_returns,
    )

    expected = np.array([
        0.050 / 0.061,
        0.015 / 0.061,
        -0.004 / 0.061,
    ])

    np.testing.assert_allclose(
        result["return_contribution_pct"].to_numpy(),
        expected,
        atol=1e-12,
    )


def test_security_contribution_percentages_reconcile(
    portfolio,
    classification,
    security_returns,
):
    result = calculate_security_attribution(
        portfolio,
        classification,
        security_returns,
    )

    assert np.isclose(
        result["return_contribution_pct"].sum(),
        1.0,
        atol=1e-12,
    )


def test_sector_attribution_columns(
    portfolio,
    classification,
    security_returns,
):
    result = calculate_sector_attribution(
        portfolio,
        classification,
        security_returns,
    )

    expected_columns = [
        "sector",
        "beginning_weight",
        "ending_weight",
        "return",
        "return_contribution",
        "return_contribution_pct",
    ]

    assert list(result.columns) == expected_columns


def test_sector_attribution_aggregation(
    portfolio,
    classification,
    security_returns,
):
    result = calculate_sector_attribution(
        portfolio,
        classification,
        security_returns,
    )

    technology = result[
        result["sector"] == "Technology"
    ].iloc[0]

    finance = result[
        result["sector"] == "Finance"
    ].iloc[0]

    assert np.isclose(
        technology["beginning_weight"],
        0.80,
        atol=1e-12,
    )

    assert np.isclose(
        technology["ending_weight"],
        0.80,
        atol=1e-12,
    )

    assert np.isclose(
        technology["return"],
        0.08125,
        atol=1e-12,
    )

    assert np.isclose(
        technology["return_contribution"],
        0.065,
        atol=1e-12,
    )

    assert np.isclose(
        finance["beginning_weight"],
        0.20,
        atol=1e-12,
    )

    assert np.isclose(
        finance["ending_weight"],
        0.20,
        atol=1e-12,
    )

    assert np.isclose(
        finance["return"],
        -0.02,
        atol=1e-12,
    )

    assert np.isclose(
        finance["return_contribution"],
        -0.004,
        atol=1e-12,
    )


def test_sector_attribution_reconciles(
    portfolio,
    classification,
    security_returns,
):
    result = calculate_sector_attribution(
        portfolio,
        classification,
        security_returns,
    )

    assert np.isclose(
        result["beginning_weight"].sum(),
        1.0,
        atol=1e-12,
    )

    assert np.isclose(
        result["ending_weight"].sum(),
        1.0,
        atol=1e-12,
    )

    assert np.isclose(
        result["return_contribution"].sum(),
        0.061,
        atol=1e-12,
    )

    assert np.isclose(
        result["return_contribution_pct"].sum(),
        1.0,
        atol=1e-12,
    )


def test_peer_group_attribution_columns(
    portfolio,
    classification,
    security_returns,
):
    result = calculate_peer_group_attribution(
        portfolio,
        classification,
        security_returns,
    )

    expected_columns = [
        "peer_group",
        "beginning_weight",
        "ending_weight",
        "return",
        "return_contribution",
        "return_contribution_pct",
    ]

    assert list(result.columns) == expected_columns


def test_peer_group_attribution_aggregation(
    portfolio,
    classification,
    security_returns,
):
    result = calculate_peer_group_attribution(
        portfolio,
        classification,
        security_returns,
    )

    it = result[
        result["peer_group"] == "IT"
    ].iloc[0]

    banks = result[
        result["peer_group"] == "Banks"
    ].iloc[0]

    assert np.isclose(
        it["beginning_weight"],
        0.80,
        atol=1e-12,
    )

    assert np.isclose(
        it["ending_weight"],
        0.80,
        atol=1e-12,
    )

    assert np.isclose(
        it["return"],
        0.08125,
        atol=1e-12,
    )

    assert np.isclose(
        it["return_contribution"],
        0.065,
        atol=1e-12,
    )

    assert np.isclose(
        banks["beginning_weight"],
        0.20,
        atol=1e-12,
    )

    assert np.isclose(
        banks["ending_weight"],
        0.20,
        atol=1e-12,
    )

    assert np.isclose(
        banks["return"],
        -0.02,
        atol=1e-12,
    )

    assert np.isclose(
        banks["return_contribution"],
        -0.004,
        atol=1e-12,
    )
