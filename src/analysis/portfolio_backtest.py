# src/analysis/portfolio_backtest.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioBacktestConfig:
    initial_capital: float = 1_000_000.0
    trading_days: int = 252
    risk_free_rate: float = 0.06
    transaction_cost_bps: float = 0.0


REQUIRED_PORTFOLIO_COLUMNS = {
    "symbol",
    "weight",
    "selected",
}

PRICE_COLUMN = "Close"


def _validate_config(
    config: PortfolioBacktestConfig,
) -> None:
    if config.initial_capital <= 0:
        raise ValueError(
            "initial_capital must be greater than zero."
        )

    if config.trading_days <= 0:
        raise ValueError(
            "trading_days must be greater than zero."
        )

    if config.transaction_cost_bps < 0:
        raise ValueError(
            "transaction_cost_bps cannot be negative."
        )


def _validate_portfolio(
    portfolio: pd.DataFrame,
) -> pd.DataFrame:
    missing = REQUIRED_PORTFOLIO_COLUMNS.difference(
        portfolio.columns
    )

    if missing:
        raise ValueError(
            "Missing required portfolio columns: "
            + ", ".join(sorted(missing))
        )

    selected = portfolio[
        portfolio["selected"] == True
    ].copy()

    if selected.empty:
        raise ValueError(
            "Portfolio contains no selected holdings."
        )

    if selected["symbol"].duplicated().any():
        raise ValueError(
            "Duplicate selected symbols found."
        )

    selected["weight"] = pd.to_numeric(
        selected["weight"],
        errors="coerce",
    )

    if selected["weight"].isna().any():
        raise ValueError(
            "Selected holdings contain missing weights."
        )

    if (selected["weight"] <= 0).any():
        raise ValueError(
            "Selected holdings must have positive weights."
        )

    if not np.isclose(
        selected["weight"].sum(),
        1.0,
        atol=1e-8,
    ):
        raise ValueError(
            "Selected portfolio weights must sum to one."
        )

    return selected.reset_index(drop=True)


def load_price_data(
    symbols: list[str],
    start_date: str,
    end_date: str,
    cache_dir: str | Path = "data/cache",
) -> pd.DataFrame:
    cache_dir = Path(cache_dir)

    series = []

    for symbol in symbols:

        cache_path = (
            cache_dir
            / (
                f"{symbol}"
                f"__{start_date}"
                f"__{end_date}"
                f"__adjusted.csv"
            )
        )

        if not cache_path.exists():
            raise FileNotFoundError(
                f"Price cache not found: {cache_path}"
            )

        data = pd.read_csv(
            cache_path,
            index_col=0,
            parse_dates=True,
        )

        if PRICE_COLUMN not in data.columns:
            raise ValueError(
                f"{PRICE_COLUMN} column missing for {symbol}."
            )

        close = pd.to_numeric(
            data[PRICE_COLUMN],
            errors="coerce",
        ).rename(symbol)

        series.append(close)

    prices = pd.concat(
        series,
        axis=1,
    ).sort_index()

    prices = prices.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if prices.empty:
        raise ValueError(
            "No price data available."
        )

    return prices


def prepare_price_data(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    if prices.empty:
        raise ValueError(
            "Price dataset is empty."
        )

    result = prices.copy()

    result = result.apply(
        pd.to_numeric,
        errors="coerce",
    )

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    result = result.sort_index()

    result = result.dropna(
        how="all"
    )

    if result.isna().all().any():
        result = result.dropna(
            axis=1,
            how="all",
        )

    if result.empty:
        raise ValueError(
            "Price dataset contains no usable values."
        )

    result = result.ffill()

    result = result.dropna(
        how="any"
    )

    if result.empty:
        raise ValueError(
            "No common valid price history exists "
            "across the selected holdings."
        )

    return result


def calculate_buy_and_hold_backtest(
    prices: pd.DataFrame,
    portfolio: pd.DataFrame,
    config: PortfolioBacktestConfig | None = None,
) -> pd.DataFrame:
    if config is None:
        config = PortfolioBacktestConfig()

    _validate_config(config)

    holdings = _validate_portfolio(
        portfolio
    )

    symbols = holdings["symbol"].tolist()

    missing_symbols = [
        symbol
        for symbol in symbols
        if symbol not in prices.columns
    ]

    if missing_symbols:
        raise ValueError(
            "Missing price columns: "
            + ", ".join(missing_symbols)
        )

    prices = prepare_price_data(
        prices[symbols]
    )

    weights = (
        holdings
        .set_index("symbol")["weight"]
        .reindex(symbols)
    )

    first_prices = prices.iloc[0]

    shares = (
        config.initial_capital
        * weights
        / first_prices
    )

    values = prices.mul(
        shares,
        axis=1,
    )

    portfolio_value = values.sum(
        axis=1
    )

    daily_return = (
        portfolio_value
        .pct_change()
        .fillna(0.0)
    )

    cumulative_return = (
        portfolio_value
        / portfolio_value.iloc[0]
        - 1.0
    )

    running_max = (
        portfolio_value
        .cummax()
    )

    drawdown = (
        portfolio_value
        / running_max
        - 1.0
    )

    result = pd.DataFrame(
        {
            "portfolio_value": portfolio_value,
            "daily_return": daily_return,
            "cumulative_return": cumulative_return,
            "drawdown": drawdown,
        }
    )

    for symbol in symbols:
        result[
            f"value_{symbol}"
        ] = values[symbol]

    return result


def calculate_rebalanced_backtest(
    prices: pd.DataFrame,
    portfolio: pd.DataFrame,
    config: PortfolioBacktestConfig | None = None,
) -> pd.DataFrame:
    if config is None:
        config = PortfolioBacktestConfig()

    _validate_config(config)

    holdings = _validate_portfolio(
        portfolio
    )

    symbols = holdings["symbol"].tolist()

    missing_symbols = [
        symbol
        for symbol in symbols
        if symbol not in prices.columns
    ]

    if missing_symbols:
        raise ValueError(
            "Missing price columns: "
            + ", ".join(missing_symbols)
        )

    prices = prepare_price_data(
        prices[symbols]
    )

    returns = prices.pct_change()

    returns = returns.fillna(0.0)

    weights = (
        holdings
        .set_index("symbol")["weight"]
        .reindex(symbols)
    )

    portfolio_return = returns.mul(
        weights,
        axis=1,
    ).sum(axis=1)

    portfolio_value = (
        config.initial_capital
        * (1.0 + portfolio_return).cumprod()
    )

    cumulative_return = (
        portfolio_value
        / config.initial_capital
        - 1.0
    )

    running_max = (
        portfolio_value
        .cummax()
    )

    drawdown = (
        portfolio_value
        / running_max
        - 1.0
    )

    result = pd.DataFrame(
        {
            "portfolio_value": portfolio_value,
            "daily_return": portfolio_return,
            "cumulative_return": cumulative_return,
            "drawdown": drawdown,
        }
    )

    return result


def calculate_backtest_metrics(
    backtest: pd.DataFrame,
    config: PortfolioBacktestConfig | None = None,
) -> dict[str, float]:
    if config is None:
        config = PortfolioBacktestConfig()

    _validate_config(config)

    if backtest.empty:
        raise ValueError(
            "Backtest dataset is empty."
        )

    portfolio_value = pd.to_numeric(
        backtest["portfolio_value"],
        errors="coerce",
    )

    daily_return = pd.to_numeric(
        backtest["daily_return"],
        errors="coerce",
    )

    if portfolio_value.isna().any():
        raise ValueError(
            "Backtest contains invalid portfolio values."
        )

    if daily_return.isna().any():
        raise ValueError(
            "Backtest contains invalid daily returns."
        )

    start_value = float(
        portfolio_value.iloc[0]
    )

    end_value = float(
        portfolio_value.iloc[-1]
    )

    total_return = (
        end_value / start_value
        - 1.0
    )

    elapsed_days = (
        portfolio_value.index[-1]
        - portfolio_value.index[0]
    ).days

    years = (
        elapsed_days / 365.25
        if elapsed_days > 0
        else np.nan
    )

    cagr = (
        (end_value / start_value) ** (1.0 / years)
        - 1.0
        if np.isfinite(years) and years > 0
        else np.nan
    )

    volatility = (
        daily_return.std(ddof=1)
        * np.sqrt(config.trading_days)
    )

    excess_return = (
        daily_return.mean()
        * config.trading_days
        - config.risk_free_rate
    )

    sharpe = (
        excess_return / volatility
        if volatility > 0
        else np.nan
    )

    downside = daily_return[
        daily_return < 0
    ]

    downside_deviation = (
        downside.std(ddof=1)
        * np.sqrt(config.trading_days)
        if len(downside) > 1
        else np.nan
    )

    sortino = (
        excess_return / downside_deviation
        if (
            np.isfinite(downside_deviation)
            and downside_deviation > 0
        )
        else np.nan
    )

    maximum_drawdown = float(
        backtest["drawdown"].min()
    )

    return {
        "start_value": start_value,
        "end_value": end_value,
        "total_return": float(total_return),
        "cagr": float(cagr),
        "annualized_volatility": float(volatility),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "maximum_drawdown": maximum_drawdown,
        "trading_observations": float(
            len(backtest)
        ),
    }


def generate_backtest_summary(
    metrics: dict[str, float],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "start_value": metrics["start_value"],
                "end_value": metrics["end_value"],
                "total_return": metrics["total_return"],
                "cagr": metrics["cagr"],
                "annualized_volatility": metrics[
                    "annualized_volatility"
                ],
                "sharpe_ratio": metrics[
                    "sharpe_ratio"
                ],
                "sortino_ratio": metrics[
                    "sortino_ratio"
                ],
                "maximum_drawdown": metrics[
                    "maximum_drawdown"
                ],
                "trading_observations": metrics[
                    "trading_observations"
                ],
            }
        ]
    )


def print_backtest_audit(
    backtest: pd.DataFrame,
    metrics: dict[str, float],
) -> None:
    print("=" * 70)
    print("NIFTY 50 PORTFOLIO BACKTEST")
    print("=" * 70)

    print()
    print("BACKTEST PERIOD")
    print(
        f"Start: {backtest.index[0].date()}"
    )
    print(
        f"End: {backtest.index[-1].date()}"
    )

    print()
    print("PERFORMANCE")

    print(
        f"Start value: "
        f"{metrics['start_value']:,.2f}"
    )

    print(
        f"End value: "
        f"{metrics['end_value']:,.2f}"
    )

    print(
        f"Total return: "
        f"{metrics['total_return']:.2%}"
    )

    print(
        f"CAGR: "
        f"{metrics['cagr']:.2%}"
    )

    print(
        f"Annualized volatility: "
        f"{metrics['annualized_volatility']:.2%}"
    )

    print(
        f"Sharpe ratio: "
        f"{metrics['sharpe_ratio']:.4f}"
    )

    print(
        f"Sortino ratio: "
        f"{metrics['sortino_ratio']:.4f}"
    )

    print(
        f"Maximum drawdown: "
        f"{metrics['maximum_drawdown']:.2%}"
    )

    print()
    print("=" * 70)
