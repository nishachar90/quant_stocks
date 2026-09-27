import numpy as np


def calculate_daily_returns(data):
    return data["Close"].pct_change()


def calculate_log_returns(data):
    return np.log(data["Close"] / data["Close"].shift(1))


def calculate_cumulative_returns(returns):
    return (1 + returns).cumprod() - 1


def calculate_total_return(data):
    start_price = data["Close"].iloc[0]
    end_price = data["Close"].iloc[-1]

    return (end_price / start_price) - 1


def calculate_cagr(data, years=None):
    start_price = data["Close"].iloc[0]
    end_price = data["Close"].iloc[-1]

    if years is None:
        days = (data.index[-1] - data.index[0]).days
        years = days / 365.25

    return (end_price / start_price) ** (1 / years) - 1
