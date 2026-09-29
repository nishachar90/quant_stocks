# src/visualization/portfolio_charts.py

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_time_series(
    path: str | Path,
) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    data = pd.read_csv(
        path,
        index_col="Date",
        parse_dates=True,
    )

    if data.empty:
        raise ValueError(
            f"Dataset is empty: {path}"
        )

    data = data.sort_index()

    return data


def prepare_cumulative_growth(
    portfolio_backtest: pd.DataFrame,
    rebalanced_backtest: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    portfolio = portfolio_backtest[
        ["portfolio_value"]
    ].rename(
        columns={
            "portfolio_value": "buy_and_hold"
        }
    )

    rebalanced = rebalanced_backtest[
        ["portfolio_value"]
    ].rename(
        columns={
            "portfolio_value": "rebalanced"
        }
    )

    benchmark = benchmark[
        ["Close"]
    ].rename(
        columns={
            "Close": "benchmark"
        }
    )

    result = pd.concat(
        [
            portfolio,
            rebalanced,
            benchmark,
        ],
        axis=1,
        join="inner",
    ).dropna()

    if result.empty:
        raise ValueError(
            "No common dates available for cumulative growth."
        )

    normalized = (
        result
        / result.iloc[0]
        * 100.0
    )

    return normalized


def prepare_drawdowns(
    portfolio_backtest: pd.DataFrame,
    rebalanced_backtest: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    growth = prepare_cumulative_growth(
        portfolio_backtest,
        rebalanced_backtest,
        benchmark,
    )

    running_max = growth.cummax()

    return (
        growth / running_max
        - 1.0
    )


def prepare_rolling_returns(
    portfolio_backtest: pd.DataFrame,
    rebalanced_backtest: pd.DataFrame,
    benchmark: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    growth = prepare_cumulative_growth(
        portfolio_backtest,
        rebalanced_backtest,
        benchmark,
    )

    if window <= 0:
        raise ValueError(
            "window must be greater than zero."
        )

    return (
        growth
        / growth.shift(window)
        - 1.0
    )


def plot_cumulative_growth(
    data: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    for column in data.columns:
        ax.plot(
            data.index,
            data[column],
            label=column.replace(
                "_",
                " ",
            ).title(),
        )

    ax.set_title(
        "Portfolio Growth vs NIFTY 50"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel(
        "Growth of ₹100"
    )
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_drawdowns(
    data: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    for column in data.columns:
        ax.plot(
            data.index,
            data[column] * 100.0,
            label=column.replace(
                "_",
                " ",
            ).title(),
        )

    ax.set_title(
        "Portfolio and Benchmark Drawdowns"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel(
        "Drawdown (%)"
    )
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_rolling_returns(
    data: pd.DataFrame,
    output_path: str | Path,
    window: int = 252,
) -> None:
    if window <= 0:
        raise ValueError(
            "window must be greater than zero."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    for column in data.columns:
        ax.plot(
            data.index,
            data[column] * 100.0,
            label=column.replace(
                "_",
                " ",
            ).title(),
        )

    ax.set_title(
        f"Rolling {window}-Trading-Day Returns"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel(
        "Return (%)"
    )
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def generate_all_charts(
    portfolio_backtest_path: str | Path,
    rebalanced_backtest_path: str | Path,
    benchmark_path: str | Path,
    output_dir: str | Path = "data/analysis/charts",
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    portfolio = load_time_series(
        portfolio_backtest_path
    )

    rebalanced = load_time_series(
        rebalanced_backtest_path
    )

    benchmark = load_time_series(
        benchmark_path
    )

    cumulative = prepare_cumulative_growth(
        portfolio,
        rebalanced,
        benchmark,
    )

    drawdowns = prepare_drawdowns(
        portfolio,
        rebalanced,
        benchmark,
    )

    rolling_returns = prepare_rolling_returns(
        portfolio,
        rebalanced,
        benchmark,
    )

    cumulative_path = (
        output_dir
        / "portfolio_cumulative_growth.png"
    )

    drawdown_path = (
        output_dir
        / "portfolio_drawdowns.png"
    )

    rolling_returns_path = (
        output_dir
        / "portfolio_rolling_returns.png"
    )

    plot_cumulative_growth(
        cumulative,
        cumulative_path,
    )

    plot_drawdowns(
        drawdowns,
        drawdown_path,
    )

    plot_rolling_returns(
        rolling_returns,
        rolling_returns_path,
    )

    return {
        "cumulative_growth": cumulative_path,
        "drawdowns": drawdown_path,
        "rolling_returns": rolling_returns_path,
    }
