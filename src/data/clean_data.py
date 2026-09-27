import pandas as pd


def check_missing_data(data):
    return data.isna().sum()


def check_duplicate_dates(data):
    return data.index.duplicated().sum()


def sort_by_date(data):
    return data.sort_index()


def remove_missing_prices(data):
    price_columns = ["Open", "High", "Low", "Close"]
    return data.dropna(subset=price_columns)


def clean_data(data):
    data = sort_by_date(data)
    data = remove_missing_prices(data)

    return data
