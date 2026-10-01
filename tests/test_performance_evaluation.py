"""
Tests for the Portfolio Performance Evaluation Engine.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.performance_evaluation import (
    evaluate_benchmark_relative_performance,
    evaluate_performance,
    evaluate_portfolio_performance,
)


def create_portfolio() -> pd.DataFrame:
    """Create deterministic portfolio test data."""

    dates = pd.date_range(
        "2025-01-01",
        periods=6,
        freq="D",
    )

    portfolio_values = pd.Series(
        [100, 102, 101, 105, 107, 110],
        index=dates,
        dtype=float,
    )

    daily_returns = portfolio_values.pct_change().fillna(0)

    cumulative_return = (
        portfolio_values / portfolio_values.iloc[0] - 1
    )

    running_peak = portfolio_values.cummax()

    drawdown = portfolio_values / running_peak - 1

    return pd.DataFrame(
        {
            "portfolio_value": portfolio_values,
            "daily_return": daily_returns,
            "cumulative_return": cumulative_return,
            "drawdown": drawdown,
        }
    )


def create_benchmark() -> pd.DataFrame:
    """Create deterministic benchmark test data."""

    dates = pd.date_range(
        "2025-01-01",
        periods=6,
        freq="D",
    )

    return pd.DataFrame(
        {
            "Close": [100, 101, 100, 103, 104, 106],
        },
        index=dates,
    )


def test_portfolio_metrics_are_returned():
    portfolio = create_portfolio()

    result = evaluate_portfolio_performance(portfolio)

    expected_metrics = {
        "total_return",
        "cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "maximum_drawdown",
    }

    assert set(result) == expected_metrics


def test_total_return():
    portfolio = create_portfolio()

    result = evaluate_portfolio_performance(portfolio)

    assert result["total_return"] == pytest.approx(0.10)


def test_maximum_drawdown():
    portfolio = create_portfolio()

    result = evaluate_portfolio_performance(portfolio)

    assert result["maximum_drawdown"] == pytest.approx(
        101 / 102 - 1
    )


def test_positive_portfolio_values_required():
    portfolio = create_portfolio()
    portfolio.loc[portfolio.index[2], "portfolio_value"] = 0

    with pytest.raises(ValueError):
        evaluate_portfolio_performance(portfolio)


def test_finite_daily_returns_required():
    portfolio = create_portfolio()
    portfolio.loc[portfolio.index[2], "daily_return"] = np.inf

    with pytest.raises(ValueError):
        evaluate_portfolio_performance(portfolio)


def test_benchmark_metrics_are_returned():
    portfolio = create_portfolio()
    benchmark = create_benchmark()

    result = evaluate_benchmark_relative_performance(
        portfolio,
        benchmark,
    )

    expected_metrics = {
        "excess_total_return",
        "beta",
        "correlation",
        "tracking_error",
        "information_ratio",
        "alpha",
    }

    assert set(result) == expected_metrics


def test_benchmark_alignment():
    portfolio = create_portfolio()
    benchmark = create_benchmark()

    benchmark = benchmark.iloc[1:]

    result = evaluate_benchmark_relative_performance(
        portfolio,
        benchmark,
    )

    assert all(np.isfinite(value) for value in result.values())


def test_missing_benchmark_column():
    portfolio = create_portfolio()

    benchmark = pd.DataFrame(
        {
            "price": [100, 101, 102, 103, 104, 105],
        },
        index=portfolio.index,
    )

    with pytest.raises(ValueError):
        evaluate_benchmark_relative_performance(
            portfolio,
            benchmark,
        )


def test_complete_evaluation():
    portfolio = create_portfolio()
    benchmark = create_benchmark()

    result = evaluate_performance(
        portfolio,
        benchmark,
    )

    expected_metrics = {
        "total_return",
        "cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "maximum_drawdown",
        "excess_total_return",
        "beta",
        "correlation",
        "tracking_error",
        "information_ratio",
        "alpha",
    }

    assert set(result) == expected_metrics


def test_benchmark_relative_return_is_difference():
    portfolio = create_portfolio()
    benchmark = create_benchmark()

    result = evaluate_benchmark_relative_performance(
        portfolio,
        benchmark,
    )

    portfolio_return = (
        (1 + portfolio["daily_return"]).prod() - 1
    )

    benchmark_return = (
        benchmark["Close"].pct_change().dropna()
        .add(1)
        .prod()
        - 1
    )

    assert result["excess_total_return"] == pytest.approx(
        portfolio_return - benchmark_return
    )


def test_empty_portfolio_rejected():
    portfolio = create_portfolio().iloc[0:0]
    benchmark = create_benchmark()

    with pytest.raises(ValueError):
        evaluate_performance(
            portfolio,
            benchmark,
        )
