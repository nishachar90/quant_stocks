import numpy as np
import pandas as pd


def calculate_annualized_volatility(returns, trading_days=252):
    daily_volatility = returns.std()

    return daily_volatility * np.sqrt(trading_days)


def calculate_max_drawdown(returns):
    growth = (1 + returns).cumprod()
    running_peak = growth.cummax()

    drawdown = growth / running_peak - 1

    return drawdown.min()


def calculate_drawdown_series(returns):
    growth = (1 + returns).cumprod()
    running_peak = growth.cummax()

    return growth / running_peak - 1


def calculate_sharpe_ratio(
    returns,
    risk_free_rate=0.06,
    trading_days=252
):
    daily_risk_free_rate = risk_free_rate / trading_days
    excess_returns = returns - daily_risk_free_rate

    return (
        excess_returns.mean()
        / excess_returns.std()
    ) * np.sqrt(trading_days)


def calculate_sortino_ratio(
    returns,
    risk_free_rate=0.06,
    trading_days=252
):
    daily_risk_free_rate = risk_free_rate / trading_days
    excess_returns = returns - daily_risk_free_rate

    downside_returns = excess_returns[excess_returns < 0]

    downside_deviation = np.sqrt(
        (downside_returns ** 2).mean()
    )

    return (
        excess_returns.mean()
        / downside_deviation
    ) * np.sqrt(trading_days)


def calculate_beta(stock_returns, market_returns):
    returns = pd.concat(
        [stock_returns, market_returns],
        axis=1,
        join="inner"
    ).dropna()

    stock = returns.iloc[:, 0]
    market = returns.iloc[:, 1]

    covariance = stock.cov(market)
    market_variance = market.var()

    return covariance / market_variance


def calculate_correlation(stock_returns, market_returns):
    returns = pd.concat(
        [stock_returns, market_returns],
        axis=1,
        join="inner"
    ).dropna()

    return returns.iloc[:, 0].corr(returns.iloc[:, 1])
