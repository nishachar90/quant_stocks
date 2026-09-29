# src/analysis/benchmark_analysis.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BenchmarkAnalysisConfig:
    benchmark_symbol: str = "^NSEI"
    risk_free_rate: float = 0.06
    trading_days: int = 252


REQUIRED_BACKTEST_COLUMNS = {
    "portfolio_value",
    "daily_return",
}

REQUIRED_BENCHMARK_COLUMNS = {
    "Close",
}


def _validate_config(
    config: BenchmarkAnalysisConfig,
) -> None:
    if not config.benchmark_symbol:
        raise ValueError(
            "benchmark_symbol cannot be empty."
        )

    if config.trading_days <= 0:
        raise ValueError(
            "trading_days must be greater than zero."
        )

    if config.risk_free_rate < 0:
        raise ValueError(
            "risk_free_rate cannot be negative."
        )


def _validate_backtest(
    backtest: pd.DataFrame,
) -> pd.DataFrame:
    missing = REQUIRED_BACKTEST_COLUMNS.difference(
        backtest.columns
    )

    if missing:
        raise ValueError(
            "Missing backtest columns: "
            + ", ".join(sorted(missing))
        )

    if backtest.empty:
        raise ValueError(
            "Backtest dataset is empty."
        )

    result = backtest.copy()

    result["portfolio_value"] = pd.to_numeric(
        result["portfolio_value"],
        errors="coerce",
    )

    result["daily_return"] = pd.to_numeric(
        result["daily_return"],
        errors="coerce",
    )

    if result[
        ["portfolio_value", "daily_return"]
    ].isna().any().any():
        raise ValueError(
            "Backtest contains invalid numeric values."
        )

    result.index = pd.to_datetime(
        result.index
    )

    result = result.sort_index()

    return result


def _validate_benchmark(
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    missing = REQUIRED_BENCHMARK_COLUMNS.difference(
        benchmark.columns
    )

    if missing:
        raise ValueError(
            "Missing benchmark columns: "
            + ", ".join(sorted(missing))
        )

    if benchmark.empty:
        raise ValueError(
            "Benchmark dataset is empty."
        )

    result = benchmark.copy()

    result["Close"] = pd.to_numeric(
        result["Close"],
        errors="coerce",
    )

    result = result[
        ["Close"]
    ].dropna()

    if result.empty:
        raise ValueError(
            "Benchmark contains no valid Close prices."
        )

    result.index = pd.to_datetime(
        result.index
    )

    result = result.sort_index()

    return result


def load_benchmark_data(
    start_date: str,
    end_date: str,
    cache_dir: str | Path = "data/cache",
    benchmark_symbol: str = "^NSEI",
) -> pd.DataFrame:
    cache_dir = Path(cache_dir)

    cache_path = (
        cache_dir
        / (
            f"{benchmark_symbol}"
            f"__{start_date}"
            f"__{end_date}"
            f"__adjusted.csv"
        )
    )

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Benchmark cache not found: {cache_path}"
        )

    benchmark = pd.read_csv(
        cache_path,
        index_col=0,
        parse_dates=True,
    )

    return _validate_benchmark(
        benchmark
    )


def prepare_benchmark_returns(
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    benchmark = _validate_benchmark(
        benchmark
    )

    returns = benchmark["Close"].pct_change()

    result = pd.DataFrame(
        {
            "benchmark_price": benchmark["Close"],
            "benchmark_return": returns,
        }
    )

    result["benchmark_return"] = (
        result["benchmark_return"]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    return result


def align_portfolio_and_benchmark(
    backtest: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    portfolio = _validate_backtest(
        backtest
    )

    benchmark_returns = (
        prepare_benchmark_returns(
            benchmark
        )
    )

    result = pd.concat(
        [
            portfolio[
                [
                    "portfolio_value",
                    "daily_return",
                ]
            ],
            benchmark_returns[
                [
                    "benchmark_price",
                    "benchmark_return",
                ]
            ],
        ],
        axis=1,
        join="inner",
    ).dropna()

    if len(result) < 2:
        raise ValueError(
            "Portfolio and benchmark do not have enough "
            "overlapping observations."
        )

    return result


def calculate_benchmark_metrics(
    benchmark: pd.DataFrame,
    config: BenchmarkAnalysisConfig | None = None,
) -> dict[str, float]:
    if config is None:
        config = BenchmarkAnalysisConfig()

    _validate_config(config)

    data = prepare_benchmark_returns(
        benchmark
    )

    returns = data[
        "benchmark_return"
    ]

    start_price = float(
        data["benchmark_price"].iloc[0]
    )

    end_price = float(
        data["benchmark_price"].iloc[-1]
    )

    total_return = (
        end_price / start_price
        - 1.0
    )

    elapsed_days = (
        data.index[-1]
        - data.index[0]
    ).days

    years = (
        elapsed_days / 365.25
        if elapsed_days > 0
        else np.nan
    )

    cagr = (
        (end_price / start_price) ** (1.0 / years)
        - 1.0
        if np.isfinite(years) and years > 0
        else np.nan
    )

    volatility = (
        returns.std(ddof=1)
        * np.sqrt(config.trading_days)
    )

    annualized_mean_return = (
        returns.mean()
        * config.trading_days
    )

    excess_return = (
        annualized_mean_return
        - config.risk_free_rate
    )

    sharpe_ratio = (
        excess_return / volatility
        if volatility > 0
        else np.nan
    )

    downside = returns[
        returns < 0
    ]

    downside_deviation = (
        downside.std(ddof=1)
        * np.sqrt(config.trading_days)
        if len(downside) > 1
        else np.nan
    )

    sortino_ratio = (
        excess_return / downside_deviation
        if (
            np.isfinite(downside_deviation)
            and downside_deviation > 0
        )
        else np.nan
    )

    cumulative = (
        data["benchmark_price"]
        / data["benchmark_price"].iloc[0]
    )

    running_max = cumulative.cummax()

    drawdown = (
        cumulative
        / running_max
        - 1.0
    )

    return {
        "start_value": start_price,
        "end_value": end_price,
        "total_return": float(total_return),
        "cagr": float(cagr),
        "annualized_volatility": float(
            volatility
        ),
        "sharpe_ratio": float(
            sharpe_ratio
        ),
        "sortino_ratio": float(
            sortino_ratio
        ),
        "maximum_drawdown": float(
            drawdown.min()
        ),
        "trading_observations": float(
            len(data)
        ),
    }


def calculate_comparison_metrics(
    backtest: pd.DataFrame,
    benchmark: pd.DataFrame,
    config: BenchmarkAnalysisConfig | None = None,
) -> dict[str, float]:
    if config is None:
        config = BenchmarkAnalysisConfig()

    _validate_config(config)

    aligned = align_portfolio_and_benchmark(
        backtest,
        benchmark,
    )

    portfolio_returns = aligned[
        "daily_return"
    ]

    benchmark_returns = aligned[
        "benchmark_return"
    ]

    covariance = np.cov(
        portfolio_returns,
        benchmark_returns,
        ddof=1,
    )[0, 1]

    benchmark_variance = np.var(
        benchmark_returns,
        ddof=1,
    )

    beta = (
        covariance / benchmark_variance
        if benchmark_variance > 0
        else np.nan
    )

    correlation = (
        portfolio_returns.corr(
            benchmark_returns
        )
    )

    active_return = (
        portfolio_returns
        - benchmark_returns
    )

    tracking_error = (
        active_return.std(ddof=1)
        * np.sqrt(config.trading_days)
    )

    information_ratio = (
        (
            active_return.mean()
            * config.trading_days
        )
        / tracking_error
        if tracking_error > 0
        else np.nan
    )

    portfolio_annualized_return = (
        portfolio_returns.mean()
        * config.trading_days
    )

    benchmark_annualized_return = (
        benchmark_returns.mean()
        * config.trading_days
    )

    alpha = (
        portfolio_annualized_return
        - (
            config.risk_free_rate
            + beta
            * (
                benchmark_annualized_return
                - config.risk_free_rate
            )
        )
    )

    portfolio_total_return = (
        aligned["portfolio_value"].iloc[-1]
        / aligned["portfolio_value"].iloc[0]
        - 1.0
    )

    benchmark_total_return = (
        aligned["benchmark_price"].iloc[-1]
        / aligned["benchmark_price"].iloc[0]
        - 1.0
    )

    return {
        "portfolio_total_return": float(
            portfolio_total_return
        ),
        "benchmark_total_return": float(
            benchmark_total_return
        ),
        "excess_total_return": float(
            portfolio_total_return
            - benchmark_total_return
        ),
        "beta": float(beta),
        "correlation": float(correlation),
        "tracking_error": float(
            tracking_error
        ),
        "information_ratio": float(
            information_ratio
        ),
        "alpha": float(alpha),
        "overlapping_observations": float(
            len(aligned)
        ),
    }


def generate_comparison_summary(
    strategy_name: str,
    portfolio_metrics: dict[str, float],
    benchmark_metrics: dict[str, float],
    comparison_metrics: dict[str, float],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "strategy": strategy_name,
                "portfolio_total_return": comparison_metrics[
                    "portfolio_total_return"
                ],
                "benchmark_total_return": comparison_metrics[
                    "benchmark_total_return"
                ],
                "excess_total_return": comparison_metrics[
                    "excess_total_return"
                ],
                "portfolio_cagr": portfolio_metrics[
                    "cagr"
                ],
                "benchmark_cagr": benchmark_metrics[
                    "cagr"
                ],
                "portfolio_annualized_volatility": portfolio_metrics[
                    "annualized_volatility"
                ],
                "benchmark_annualized_volatility": benchmark_metrics[
                    "annualized_volatility"
                ],
                "portfolio_sharpe_ratio": portfolio_metrics[
                    "sharpe_ratio"
                ],
                "benchmark_sharpe_ratio": benchmark_metrics[
                    "sharpe_ratio"
                ],
                "portfolio_sortino_ratio": portfolio_metrics[
                    "sortino_ratio"
                ],
                "benchmark_sortino_ratio": benchmark_metrics[
                    "sortino_ratio"
                ],
                "portfolio_maximum_drawdown": portfolio_metrics[
                    "maximum_drawdown"
                ],
                "benchmark_maximum_drawdown": benchmark_metrics[
                    "maximum_drawdown"
                ],
                "beta": comparison_metrics[
                    "beta"
                ],
                "correlation": comparison_metrics[
                    "correlation"
                ],
                "tracking_error": comparison_metrics[
                    "tracking_error"
                ],
                "information_ratio": comparison_metrics[
                    "information_ratio"
                ],
                "alpha": comparison_metrics[
                    "alpha"
                ],
                "overlapping_observations": comparison_metrics[
                    "overlapping_observations"
                ],
            }
        ]
    )


def print_benchmark_comparison(
    summary: pd.DataFrame,
) -> None:
    print("=" * 70)
    print("NIFTY 50 PORTFOLIO vs NIFTY 50 BENCHMARK")
    print("=" * 70)

    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print("=" * 70)
