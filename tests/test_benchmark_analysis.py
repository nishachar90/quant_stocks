# tests/test_benchmark_analysis.py

import numpy as np
import pandas as pd
import pytest

from src.analysis.benchmark_analysis import (
    BenchmarkAnalysisConfig,
    align_portfolio_and_benchmark,
    calculate_benchmark_metrics,
    calculate_comparison_metrics,
    generate_comparison_summary,
    prepare_benchmark_returns,
)


def sample_benchmark() -> pd.DataFrame:
    dates = pd.date_range(
        "2020-01-01",
        periods=5,
        freq="D",
    )

    return pd.DataFrame(
        {
            "Close": [
                100,
                102,
                104,
                106,
                108,
            ]
        },
        index=dates,
    )


def sample_backtest() -> pd.DataFrame:
    dates = pd.date_range(
        "2020-01-01",
        periods=5,
        freq="D",
    )

    value = np.array(
        [
            1_000_000,
            1_020_000,
            1_040_000,
            1_060_000,
            1_080_000,
        ],
        dtype=float,
    )

    returns = (
        pd.Series(value, index=dates)
        .pct_change()
        .fillna(0.0)
    )

    return pd.DataFrame(
        {
            "portfolio_value": value,
            "daily_return": returns.values,
        },
        index=dates,
    )


def test_prepare_benchmark_returns():
    result = prepare_benchmark_returns(
        sample_benchmark()
    )

    assert len(result) == 5
    assert (
        "benchmark_return"
        in result.columns
    )

    assert np.isclose(
        result["benchmark_return"].iloc[0],
        0.0,
    )


def test_benchmark_metrics():
    metrics = calculate_benchmark_metrics(
        sample_benchmark()
    )

    assert metrics["total_return"] > 0
    assert metrics["end_value"] > metrics["start_value"]
    assert metrics["trading_observations"] == 5


def test_alignment():
    result = align_portfolio_and_benchmark(
        sample_backtest(),
        sample_benchmark(),
    )

    assert len(result) == 5
    assert {
        "portfolio_value",
        "daily_return",
        "benchmark_price",
        "benchmark_return",
    }.issubset(result.columns)


def test_comparison_metrics():
    comparison = calculate_comparison_metrics(
        sample_backtest(),
        sample_benchmark(),
    )

    expected = {
        "portfolio_total_return",
        "benchmark_total_return",
        "excess_total_return",
        "beta",
        "correlation",
        "tracking_error",
        "information_ratio",
        "alpha",
        "overlapping_observations",
    }

    assert expected.issubset(
        comparison.keys()
    )

    assert comparison[
        "overlapping_observations"
    ] == 5


def test_comparison_has_positive_correlation():
    comparison = calculate_comparison_metrics(
        sample_backtest(),
        sample_benchmark(),
    )

    assert comparison[
        "correlation"
    ] > 0


def test_summary_generation():
    benchmark = sample_benchmark()
    backtest = sample_backtest()

    config = BenchmarkAnalysisConfig()

    portfolio_metrics = {
        "cagr": 0.10,
        "annualized_volatility": 0.15,
        "sharpe_ratio": 0.50,
        "sortino_ratio": 0.70,
        "maximum_drawdown": -0.20,
    }

    benchmark_metrics = calculate_benchmark_metrics(
        benchmark,
        config,
    )

    comparison_metrics = calculate_comparison_metrics(
        backtest,
        benchmark,
        config,
    )

    summary = generate_comparison_summary(
        "test_strategy",
        portfolio_metrics,
        benchmark_metrics,
        comparison_metrics,
    )

    assert summary.shape == (1, 20)
    assert (
        summary["strategy"].iloc[0]
        == "test_strategy"
    )


def test_missing_benchmark_close_fails():
    benchmark = pd.DataFrame(
        {
            "Open": [100, 101],
        }
    )

    with pytest.raises(ValueError):
        prepare_benchmark_returns(
            benchmark
        )


def test_empty_benchmark_fails():
    with pytest.raises(ValueError):
        prepare_benchmark_returns(
            pd.DataFrame()
        )


def test_empty_backtest_fails():
    with pytest.raises(ValueError):
        align_portfolio_and_benchmark(
            pd.DataFrame(),
            sample_benchmark(),
        )


def test_no_overlap_fails():
    benchmark = sample_benchmark()

    backtest = sample_backtest()
    backtest.index = pd.date_range(
        "2030-01-01",
        periods=5,
        freq="D",
    )

    with pytest.raises(ValueError):
        align_portfolio_and_benchmark(
            backtest,
            benchmark,
        )
