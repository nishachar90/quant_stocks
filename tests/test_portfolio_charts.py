# tests/test_portfolio_charts.py

from pathlib import Path

import numpy as np
import pandas as pd

from src.visualization.portfolio_charts import (
    generate_all_charts,
    plot_cumulative_growth,
    plot_drawdowns,
    plot_rolling_returns,
    prepare_cumulative_growth,
    prepare_drawdowns,
    prepare_rolling_returns,
)


def sample_series() -> pd.DataFrame:
    dates = pd.date_range(
        "2020-01-01",
        periods=10,
        freq="D",
    )

    return pd.DataFrame(
        {
            "portfolio_value": np.linspace(
                100,
                120,
                10,
            )
        },
        index=dates,
    )


def sample_rebalanced() -> pd.DataFrame:
    dates = pd.date_range(
        "2020-01-01",
        periods=10,
        freq="D",
    )

    return pd.DataFrame(
        {
            "portfolio_value": np.linspace(
                100,
                122,
                10,
            )
        },
        index=dates,
    )


def sample_benchmark() -> pd.DataFrame:
    dates = pd.date_range(
        "2020-01-01",
        periods=10,
        freq="D",
    )

    return pd.DataFrame(
        {
            "Close": np.linspace(
                100,
                115,
                10,
            )
        },
        index=dates,
    )


def test_cumulative_growth_columns():
    result = prepare_cumulative_growth(
        sample_series(),
        sample_rebalanced(),
        sample_benchmark(),
    )

    assert list(result.columns) == [
        "buy_and_hold",
        "rebalanced",
        "benchmark",
    ]


def test_cumulative_growth_starts_at_100():
    result = prepare_cumulative_growth(
        sample_series(),
        sample_rebalanced(),
        sample_benchmark(),
    )

    assert np.allclose(
        result.iloc[0].values,
        100.0,
    )


def test_drawdown_starts_at_zero():
    result = prepare_drawdowns(
        sample_series(),
        sample_rebalanced(),
        sample_benchmark(),
    )

    assert np.allclose(
        result.iloc[0].values,
        0.0,
    )


def test_drawdowns_never_positive():
    result = prepare_drawdowns(
        sample_series(),
        sample_rebalanced(),
        sample_benchmark(),
    )

    assert (
        result <= 1e-12
    ).all().all()


def test_rolling_returns_window():
    result = prepare_rolling_returns(
        sample_series(),
        sample_rebalanced(),
        sample_benchmark(),
        window=5,
    )

    assert result.shape == (
        10,
        3,
    )


def test_invalid_rolling_window():
    try:
        prepare_rolling_returns(
            sample_series(),
            sample_rebalanced(),
            sample_benchmark(),
            window=0,
        )
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError"
    )


def test_cumulative_plot(tmp_path):
    data = prepare_cumulative_growth(
        sample_series(),
        sample_rebalanced(),
        sample_benchmark(),
    )

    output = (
        tmp_path
        / "cumulative.png"
    )

    plot_cumulative_growth(
        data,
        output,
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_drawdown_plot(tmp_path):
    data = prepare_drawdowns(
        sample_series(),
        sample_rebalanced(),
        sample_benchmark(),
    )

    output = (
        tmp_path
        / "drawdown.png"
    )

    plot_drawdowns(
        data,
        output,
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_rolling_return_plot(tmp_path):
    data = prepare_rolling_returns(
        sample_series(),
        sample_rebalanced(),
        sample_benchmark(),
        window=5,
    )

    output = (
        tmp_path
        / "rolling.png"
    )

    plot_rolling_returns(
        data,
        output,
        window=5,
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_generate_all_charts(tmp_path):
    portfolio_path = (
        tmp_path
        / "portfolio.csv"
    )

    rebalanced_path = (
        tmp_path
        / "rebalanced.csv"
    )

    benchmark_path = (
        tmp_path
        / "benchmark.csv"
    )

    sample_series().to_csv(
        portfolio_path,
        index_label="Date",
    )

    sample_rebalanced().to_csv(
        rebalanced_path,
        index_label="Date",
    )

    sample_benchmark().to_csv(
        benchmark_path,
        index_label="Date",
    )

    result = generate_all_charts(
        portfolio_path,
        rebalanced_path,
        benchmark_path,
        tmp_path / "charts",
    )

    assert set(result.keys()) == {
        "cumulative_growth",
        "drawdowns",
        "rolling_returns",
    }

    for path in result.values():
        assert isinstance(
            path,
            Path,
        )
        assert path.exists()
        assert path.stat().st_size > 0
