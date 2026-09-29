# tests/test_portfolio_transaction_costs.py

import numpy as np
import pandas as pd
import pytest

from src.analysis.portfolio_transaction_costs import (
    calculate_one_way_turnover,
    calculate_transaction_cost,
    calculate_transaction_cost_series,
    calculate_turnover_series,
    validate_transaction_cost_bps,
    validate_weights,
)


def test_zero_transaction_cost():
    assert calculate_transaction_cost(
        1_000_000,
        0,
    ) == 0


def test_transaction_cost():
    result = calculate_transaction_cost(
        1_000_000,
        10,
    )

    assert np.isclose(
        result,
        1_000,
    )


def test_negative_transaction_cost_fails():
    with pytest.raises(ValueError):
        validate_transaction_cost_bps(
            -1
        )


def test_valid_weights():
    validate_weights(
        pd.Series(
            [
                0.6,
                0.4,
            ]
        )
    )


def test_invalid_weights_sum():
    with pytest.raises(ValueError):
        validate_weights(
            pd.Series(
                [
                    0.7,
                    0.7,
                ]
            )
        )


def test_negative_weights_fail():
    with pytest.raises(ValueError):
        validate_weights(
            pd.Series(
                [
                    1.1,
                    -0.1,
                ]
            )
        )


def test_one_way_turnover():
    old = pd.Series(
        {
            "AAA": 0.60,
            "BBB": 0.40,
        }
    )

    new = pd.Series(
        {
            "AAA": 0.50,
            "BBB": 0.50,
        }
    )

    result = calculate_one_way_turnover(
        old,
        new,
    )

    assert np.isclose(
        result,
        0.10,
    )


def test_new_position_turnover():
    old = pd.Series(
        {
            "AAA": 1.0,
        }
    )

    new = pd.Series(
        {
            "AAA": 0.5,
            "BBB": 0.5,
        }
    )

    result = calculate_one_way_turnover(
        old,
        new,
    )

    assert np.isclose(
        result,
        0.50,
    )


def test_turnover_series_first_period():
    weights = pd.DataFrame(
        {
            "AAA": [
                0.60,
                0.50,
                0.70,
            ],
            "BBB": [
                0.40,
                0.50,
                0.30,
            ],
        },
        index=pd.date_range(
            "2020-01-01",
            periods=3,
        ),
    )

    result = calculate_turnover_series(
        weights
    )

    assert np.isclose(
        result.iloc[0],
        0.50,
    )

    assert np.isclose(
        result.iloc[1],
        0.10,
    )

    assert np.isclose(
        result.iloc[2],
        0.20,
    )


def test_transaction_cost_series():
    portfolio_values = pd.Series(
        [
            1_000_000,
            1_100_000,
        ]
    )

    turnover = pd.Series(
        [
            0.50,
            0.10,
        ]
    )

    result = calculate_transaction_cost_series(
        portfolio_values,
        turnover,
        10,
    )

    assert np.isclose(
        result.iloc[0],
        500,
    )

    assert np.isclose(
        result.iloc[1],
        110,
    )
