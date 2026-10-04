import pandas as pd
import pytest

from scripts.generate_portfolio_evaluation import (
    build_security_returns,
    evaluate_strategy,
    load_backtest,
)


def test_load_buy_and_hold_backtest():
    """Verify that the buy-and-hold backtest loads successfully."""

    backtest = load_backtest(
        strategy="buy_and_hold"
    )

    assert isinstance(backtest, pd.DataFrame)
    assert not backtest.empty
    assert isinstance(
        backtest.index,
        pd.DatetimeIndex,
    )

    assert "portfolio_value" in backtest.columns
    assert "daily_return" in backtest.columns
    assert "cumulative_return" in backtest.columns
    assert "drawdown" in backtest.columns


def test_load_rebalanced_backtest():
    """Verify that the rebalanced backtest loads successfully."""

    backtest = load_backtest(
        strategy="rebalanced"
    )

    assert isinstance(backtest, pd.DataFrame)
    assert not backtest.empty
    assert isinstance(
        backtest.index,
        pd.DatetimeIndex,
    )

    assert "portfolio_value" in backtest.columns
    assert "daily_return" in backtest.columns
    assert "cumulative_return" in backtest.columns
    assert "drawdown" in backtest.columns


def test_build_security_returns():
    """Verify security-level returns are calculated correctly."""

    backtest = pd.DataFrame(
        {
            "portfolio_value": [
                1_000_000.0,
                1_100_000.0,
            ],
            "daily_return": [
                0.0,
                0.10,
            ],
            "cumulative_return": [
                0.0,
                0.10,
            ],
            "drawdown": [
                0.0,
                0.0,
            ],
            "value_RELIANCE": [
                600_000.0,
                660_000.0,
            ],
            "value_TITAN": [
                400_000.0,
                440_000.0,
            ],
        },
        index=pd.to_datetime(
            [
                "2020-01-01",
                "2020-01-02",
            ]
        ),
    )

    portfolio = pd.DataFrame(
        {
            "symbol": [
                "RELIANCE",
                "TITAN",
            ],
            "weight": [
                0.60,
                0.40,
            ],
        }
    )

    security_returns = build_security_returns(
        portfolio=portfolio,
        backtest=backtest,
    )

    assert isinstance(
        security_returns,
        pd.Series,
    )

    assert list(
        security_returns.index
    ) == [
        "RELIANCE",
        "TITAN",
    ]

    assert security_returns["RELIANCE"] == pytest.approx(
        0.10
    )

    assert security_returns["TITAN"] == pytest.approx(
        0.10
    )


def test_build_security_returns_uses_correct_value_columns():
    """Verify evaluation uses value_SYMBOL rather than value_SYMBOL.NS."""

    backtest = pd.DataFrame(
        {
            "portfolio_value": [
                1_000_000.0,
                1_050_000.0,
            ],
            "daily_return": [
                0.0,
                0.05,
            ],
            "cumulative_return": [
                0.0,
                0.05,
            ],
            "drawdown": [
                0.0,
                0.0,
            ],
            "value_RELIANCE": [
                600_000.0,
                630_000.0,
            ],
            "value_TITAN": [
                400_000.0,
                420_000.0,
            ],
        },
        index=pd.to_datetime(
            [
                "2020-01-01",
                "2020-01-02",
            ]
        ),
    )

    portfolio = pd.DataFrame(
        {
            "symbol": [
                "RELIANCE",
                "TITAN",
            ],
            "weight": [
                0.60,
                0.40,
            ],
        }
    )

    security_returns = build_security_returns(
        portfolio=portfolio,
        backtest=backtest,
    )

    assert security_returns["RELIANCE"] == pytest.approx(
        0.05
    )

    assert security_returns["TITAN"] == pytest.approx(
        0.05
    )


def test_build_security_returns_missing_security_column():
    """Verify a missing security value column raises an error."""

    backtest = pd.DataFrame(
        {
            "portfolio_value": [
                1_000_000.0,
                1_050_000.0,
            ],
            "daily_return": [
                0.0,
                0.05,
            ],
            "cumulative_return": [
                0.0,
                0.05,
            ],
            "drawdown": [
                0.0,
                0.0,
            ],
        },
        index=pd.to_datetime(
            [
                "2020-01-01",
                "2020-01-02",
            ]
        ),
    )

    portfolio = pd.DataFrame(
        {
            "symbol": [
                "RELIANCE",
            ],
            "weight": [
                1.0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="value_RELIANCE",
    ):
        build_security_returns(
            portfolio=portfolio,
            backtest=backtest,
        )
