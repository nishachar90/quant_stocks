# tests/test_portfolio_backtest.py

import numpy as np
import pandas as pd
import pytest

from src.analysis.portfolio_backtest import (
    PortfolioBacktestConfig,
    calculate_backtest_metrics,
    calculate_buy_and_hold_backtest,
    calculate_rebalanced_backtest,
    generate_backtest_summary,
    prepare_price_data,
)


def sample_portfolio() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [
                "AAA",
                "BBB",
            ],
            "weight": [
                0.60,
                0.40,
            ],
            "selected": [
                True,
                True,
            ],
        }
    )


def sample_prices() -> pd.DataFrame:
    dates = pd.date_range(
        "2020-01-01",
        periods=5,
        freq="D",
    )

    return pd.DataFrame(
        {
            "AAA": [
                100,
                105,
                110,
                108,
                120,
            ],
            "BBB": [
                100,
                102,
                104,
                106,
                108,
            ],
        },
        index=dates,
    )


def test_prepare_price_data():
    prices = sample_prices()

    result = prepare_price_data(
        prices
    )

    assert not result.empty
    assert list(result.columns) == [
        "AAA",
        "BBB",
    ]
    assert not result.isna().any().any()


def test_empty_prices_fail():
    with pytest.raises(ValueError):
        prepare_price_data(
            pd.DataFrame()
        )


def test_buy_and_hold_preserves_observations():
    result = calculate_buy_and_hold_backtest(
        sample_prices(),
        sample_portfolio(),
    )

    assert len(result) == 5


def test_buy_and_hold_starts_at_initial_capital():
    config = PortfolioBacktestConfig(
        initial_capital=1_000_000
    )

    result = calculate_buy_and_hold_backtest(
        sample_prices(),
        sample_portfolio(),
        config,
    )

    assert np.isclose(
        result["portfolio_value"].iloc[0],
        1_000_000,
    )


def test_buy_and_hold_values_increase():
    result = calculate_buy_and_hold_backtest(
        sample_prices(),
        sample_portfolio(),
    )

    assert (
        result["portfolio_value"].iloc[-1]
        > result["portfolio_value"].iloc[0]
    )


def test_cumulative_return_matches_portfolio_value():
    result = calculate_buy_and_hold_backtest(
        sample_prices(),
        sample_portfolio(),
    )

    expected = (
        result["portfolio_value"]
        / result["portfolio_value"].iloc[0]
        - 1.0
    )

    assert np.allclose(
        result["cumulative_return"],
        expected,
    )


def test_drawdown_never_positive():
    result = calculate_buy_and_hold_backtest(
        sample_prices(),
        sample_portfolio(),
    )

    assert (
        result["drawdown"] <= 1e-12
    ).all()


def test_rebalanced_backtest():
    result = calculate_rebalanced_backtest(
        sample_prices(),
        sample_portfolio(),
    )

    assert len(result) == 5
    assert np.isclose(
        result["portfolio_value"].iloc[0],
        1_000_000,
    )


def test_missing_symbol_fails():
    portfolio = pd.DataFrame(
        {
            "symbol": ["AAA", "CCC"],
            "weight": [0.5, 0.5],
            "selected": [True, True],
        }
    )

    with pytest.raises(ValueError):
        calculate_buy_and_hold_backtest(
            sample_prices(),
            portfolio,
        )


def test_weights_must_sum_to_one():
    portfolio = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "weight": [0.7, 0.7],
            "selected": [True, True],
        }
    )

    with pytest.raises(ValueError):
        calculate_buy_and_hold_backtest(
            sample_prices(),
            portfolio,
        )


def test_no_selected_holdings_fail():
    portfolio = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "weight": [0.6, 0.4],
            "selected": [False, False],
        }
    )

    with pytest.raises(ValueError):
        calculate_buy_and_hold_backtest(
            sample_prices(),
            portfolio,
        )


def test_metrics_are_generated():
    backtest = calculate_buy_and_hold_backtest(
        sample_prices(),
        sample_portfolio(),
    )

    metrics = calculate_backtest_metrics(
        backtest
    )

    expected = {
        "start_value",
        "end_value",
        "total_return",
        "cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "maximum_drawdown",
        "trading_observations",
    }

    assert expected.issubset(
        metrics.keys()
    )


def test_total_return_is_positive():
    backtest = calculate_buy_and_hold_backtest(
        sample_prices(),
        sample_portfolio(),
    )

    metrics = calculate_backtest_metrics(
        backtest
    )

    assert metrics["total_return"] > 0


def test_summary_dataframe():
    backtest = calculate_buy_and_hold_backtest(
        sample_prices(),
        sample_portfolio(),
    )

    metrics = calculate_backtest_metrics(
        backtest
    )

    summary = generate_backtest_summary(
        metrics
    )

    assert summary.shape == (1, 9)


def test_invalid_initial_capital():
    config = PortfolioBacktestConfig(
        initial_capital=0
    )

    with pytest.raises(ValueError):
        calculate_buy_and_hold_backtest(
            sample_prices(),
            sample_portfolio(),
            config,
        )


def test_negative_transaction_cost_fails():
    config = PortfolioBacktestConfig(
        transaction_cost_bps=-1
    )

    with pytest.raises(ValueError):
        calculate_buy_and_hold_backtest(
            sample_prices(),
            sample_portfolio(),
            config,
        )
