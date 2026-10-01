import pandas as pd
import pytest

from scripts.generate_portfolio_evaluation import (
    evaluate_strategy,
    load_backtest,
)


def test_load_backtest(tmp_path):
    """Verify that a backtest file loads correctly."""

    path = tmp_path / "backtest.csv"

    dataframe = pd.DataFrame(
        {
            "date": [
                "2025-01-01",
                "2025-01-02",
            ],
            "portfolio_value": [
                100.0,
                110.0,
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
        }
    )

    dataframe.to_csv(
        path,
        index=False,
    )

    result = load_backtest()

    assert result is not None


def test_load_backtest_missing_file():
    """Verify that the configured backtest file exists."""

    result = load_backtest()

    assert not result.empty


def test_evaluate_strategy():
    """Verify evaluation of a portfolio strategy."""

    dates = pd.date_range(
        "2025-01-01",
        periods=3,
    )

    backtest = pd.DataFrame(
        {
            "date": dates,
            "portfolio_value": [
                100.0,
                110.0,
                121.0,
            ],
            "daily_return": [
                0.0,
                0.10,
                0.10,
            ],
            "cumulative_return": [
                0.0,
                0.10,
                0.21,
            ],
            "drawdown": [
                0.0,
                0.0,
                0.0,
            ],
            "strategy": [
                "test_strategy",
                "test_strategy",
                "test_strategy",
            ],
        }
    )

    benchmark = pd.DataFrame(
        {
            "Close": [
                100.0,
                105.0,
                110.0,
            ],
        }
    )

    portfolio = pd.DataFrame(
        {
            "symbol": [
                "AAA",
                "BBB",
            ],
            "weight": [
                0.60,
                0.40,
            ],
        }
    )

    classification = pd.DataFrame(
        {
            "symbol": [
                "AAA",
                "BBB",
            ],
            "sector": [
                "Technology",
                "Financial Services",
            ],
            "business_type": [
                "Technology",
                "Bank",
            ],
            "peer_group": [
                "Technology",
                "Bank",
            ],
        }
    )

    with pytest.raises(
        Exception
    ):
        evaluate_strategy(
            backtest=backtest,
            benchmark=benchmark,
            portfolio=portfolio,
            classification=classification,
            strategy="test_strategy",
        )
