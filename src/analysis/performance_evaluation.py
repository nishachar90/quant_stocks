"""
Portfolio Performance Evaluation Engine

Calculates portfolio performance metrics and benchmark-relative metrics
according to the Portfolio Performance Evaluation Specification.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


TRADING_DAYS = 252
DEFAULT_RISK_FREE_RATE = 0.0


def _validate_columns(
    dataframe: pd.DataFrame,
    required_columns: tuple[str, ...],
    dataset_name: str,
) -> None:
    """Validate that all required columns are present."""

    missing = [
        column for column in required_columns if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}"
        )


def _validate_portfolio_values(portfolio: pd.DataFrame) -> None:
    """Validate portfolio values and daily returns."""

    if not np.isfinite(portfolio["portfolio_value"]).all():
        raise ValueError("Portfolio values must be finite.")

    if (portfolio["portfolio_value"] <= 0).any():
        raise ValueError("Portfolio values must be positive.")

    if not np.isfinite(portfolio["daily_return"]).all():
        raise ValueError("Portfolio daily returns must be finite.")


def _calculate_cagr(
    beginning_value: float,
    ending_value: float,
    number_of_observations: int,
    trading_days: int,
) -> float:
    """Calculate annualized compounded growth rate."""

    if beginning_value <= 0 or ending_value <= 0:
        raise ValueError("Portfolio values must be positive.")

    if number_of_observations <= 1:
        return np.nan

    years = (number_of_observations - 1) / trading_days

    if years <= 0:
        return np.nan

    return (ending_value / beginning_value) ** (1 / years) - 1


def _calculate_sharpe_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float,
    trading_days: int,
) -> float:
    """Calculate Sharpe ratio using arithmetic annualized return."""

    daily_risk_free_rate = (1 + risk_free_rate) ** (1 / trading_days) - 1

    excess_returns = daily_returns - daily_risk_free_rate

    volatility = daily_returns.std(ddof=1)

    if volatility == 0 or not np.isfinite(volatility):
        return np.nan

    annualized_excess_return = excess_returns.mean() * trading_days
    annualized_volatility = volatility * np.sqrt(trading_days)

    return annualized_excess_return / annualized_volatility


def _calculate_sortino_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float,
    trading_days: int,
) -> float:
    """Calculate Sortino ratio using downside deviation."""

    daily_risk_free_rate = (1 + risk_free_rate) ** (1 / trading_days) - 1

    excess_returns = daily_returns - daily_risk_free_rate

    downside_returns = np.minimum(excess_returns, 0)

    downside_deviation = np.sqrt(
        np.mean(downside_returns**2)
    ) * np.sqrt(trading_days)

    if downside_deviation == 0 or not np.isfinite(downside_deviation):
        return np.nan

    annualized_excess_return = excess_returns.mean() * trading_days

    return annualized_excess_return / downside_deviation


def _calculate_maximum_drawdown(
    portfolio_values: pd.Series,
) -> float:
    """Calculate the largest peak-to-trough decline."""

    running_peak = portfolio_values.cummax()

    drawdown = portfolio_values / running_peak - 1

    return float(drawdown.min())


def evaluate_portfolio_performance(
    portfolio: pd.DataFrame,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    trading_days: int = TRADING_DAYS,
) -> Dict[str, float]:
    """
    Calculate portfolio-level performance metrics.

    Required columns:
        portfolio_value
        daily_return
        cumulative_return
        drawdown
    """

    required_columns = (
        "portfolio_value",
        "daily_return",
        "cumulative_return",
        "drawdown",
    )

    _validate_columns(
        portfolio,
        required_columns,
        "Portfolio",
    )

    if portfolio.empty:
        raise ValueError("Portfolio dataframe cannot be empty.")

    _validate_portfolio_values(portfolio)

    values = portfolio["portfolio_value"]
    daily_returns = portfolio["daily_return"]

    total_return = values.iloc[-1] / values.iloc[0] - 1

    cagr = _calculate_cagr(
        beginning_value=float(values.iloc[0]),
        ending_value=float(values.iloc[-1]),
        number_of_observations=len(values),
        trading_days=trading_days,
    )

    annualized_volatility = (
        daily_returns.std(ddof=1) * np.sqrt(trading_days)
    )

    sharpe_ratio = _calculate_sharpe_ratio(
        daily_returns,
        risk_free_rate,
        trading_days,
    )

    sortino_ratio = _calculate_sortino_ratio(
        daily_returns,
        risk_free_rate,
        trading_days,
    )

    maximum_drawdown = _calculate_maximum_drawdown(values)

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "sortino_ratio": float(sortino_ratio),
        "maximum_drawdown": float(maximum_drawdown),
    }


def evaluate_benchmark_relative_performance(
    portfolio: pd.DataFrame,
    benchmark: pd.DataFrame,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    trading_days: int = TRADING_DAYS,
) -> Dict[str, float]:
    """
    Calculate benchmark-relative performance metrics.

    Required portfolio column:
        daily_return

    Required benchmark column:
        Close
    """

    _validate_columns(
        portfolio,
        ("daily_return",),
        "Portfolio",
    )

    _validate_columns(
        benchmark,
        ("Close",),
        "Benchmark",
    )

    if portfolio.empty or benchmark.empty:
        raise ValueError("Portfolio and benchmark data cannot be empty.")

    portfolio_returns = portfolio["daily_return"].copy()

    benchmark_prices = benchmark["Close"].copy()

    benchmark_returns = benchmark_prices.pct_change()

    aligned = pd.concat(
        [
            portfolio_returns.rename("portfolio_return"),
            benchmark_returns.rename("benchmark_return"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    if aligned.empty:
        raise ValueError(
            "Portfolio and benchmark have no date-aligned observations."
        )

    if not np.isfinite(aligned["portfolio_return"]).all():
        raise ValueError(
            "Portfolio returns must be finite after alignment."
        )

    if not np.isfinite(aligned["benchmark_return"]).all():
        raise ValueError(
            "Benchmark returns must be finite after alignment."
        )

    portfolio_total_return = (
        (1 + aligned["portfolio_return"]).prod() - 1
    )

    benchmark_total_return = (
        (1 + aligned["benchmark_return"]).prod() - 1
    )

    active_returns = (
        aligned["portfolio_return"]
        - aligned["benchmark_return"]
    )

    benchmark_variance = aligned["benchmark_return"].var(ddof=1)

    if benchmark_variance == 0:
        beta = np.nan
    else:
        beta = (
            aligned["portfolio_return"].cov(
                aligned["benchmark_return"]
            )
            / benchmark_variance
        )

    correlation = aligned["portfolio_return"].corr(
        aligned["benchmark_return"]
    )

    tracking_error = (
        active_returns.std(ddof=1)
        * np.sqrt(trading_days)
    )

    if tracking_error == 0 or not np.isfinite(tracking_error):
        information_ratio = np.nan
    else:
        information_ratio = (
            active_returns.mean() * trading_days
        ) / tracking_error

    benchmark_daily_risk_free = (
        (1 + risk_free_rate) ** (1 / trading_days) - 1
    )

    benchmark_excess_return = (
        aligned["benchmark_return"]
        - benchmark_daily_risk_free
    )

    portfolio_excess_return = (
        aligned["portfolio_return"]
        - benchmark_daily_risk_free
    )

    annualized_portfolio_excess_return = (
        portfolio_excess_return.mean() * trading_days
    )

    annualized_benchmark_excess_return = (
        benchmark_excess_return.mean() * trading_days
    )

    if pd.isna(beta):
        alpha = np.nan
    else:
        alpha = (
            annualized_portfolio_excess_return
            - beta * annualized_benchmark_excess_return
        )

    return {
        "excess_total_return": float(
            portfolio_total_return - benchmark_total_return
        ),
        "beta": float(beta),
        "correlation": float(correlation),
        "tracking_error": float(tracking_error),
        "information_ratio": float(information_ratio),
        "alpha": float(alpha),
    }


def evaluate_performance(
    portfolio: pd.DataFrame,
    benchmark: pd.DataFrame,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    trading_days: int = TRADING_DAYS,
) -> Dict[str, float]:
    """
    Calculate complete portfolio performance evaluation.
    """

    portfolio_metrics = evaluate_portfolio_performance(
        portfolio=portfolio,
        risk_free_rate=risk_free_rate,
        trading_days=trading_days,
    )

    benchmark_metrics = evaluate_benchmark_relative_performance(
        portfolio=portfolio,
        benchmark=benchmark,
        risk_free_rate=risk_free_rate,
        trading_days=trading_days,
    )

    return {
        **portfolio_metrics,
        **benchmark_metrics,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("PORTFOLIO PERFORMANCE EVALUATION ENGINE")
    print("=" * 70)
    print("Status: implementation loaded successfully")
