import pandas as pd


def find_row(statement, possible_names):
    """
    Find the first matching row from a list of possible
    financial-statement row names.
    """

    for name in possible_names:
        if name in statement.index:
            return statement.loc[name]

    return None


def get_revenue(statement):
    return find_row(
        statement,
        [
            "Total Revenue",
            "Operating Revenue",
            "TotalRevenue",
        ],
    )


def get_net_income(statement):
    return find_row(
        statement,
        [
            "Net Income",
            "NetIncome",
            "Net Income Common Stockholders",
        ],
    )


def get_operating_income(statement):
    return find_row(
        statement,
        [
            "Operating Income",
            "OperatingIncome",
        ],
    )


def get_total_debt(statement):
    return find_row(
        statement,
        [
            "Total Debt",
            "TotalDebt",
        ],
    )


def get_shareholders_equity(statement):
    return find_row(
        statement,
        [
            "Stockholders Equity",
            "StockholdersEquity",
            "Total Equity Gross Minority Interest",
        ],
    )


def get_operating_cash_flow(statement):
    return find_row(
        statement,
        [
            "Operating Cash Flow",
            "OperatingCashFlow",
            "Total Cash From Operating Activities",
        ],
    )


def get_capital_expenditure(statement):
    return find_row(
        statement,
        [
            "Capital Expenditure",
            "CapitalExpenditure",
        ],
    )


def get_total_assets(statement):
    return find_row(
        statement,
        [
            "Total Assets",
            "TotalAssets",
        ],
    )


def get_current_assets(statement):
    return find_row(
        statement,
        [
            "Current Assets",
            "CurrentAssets",
        ],
    )


def get_current_liabilities(statement):
    return find_row(
        statement,
        [
            "Current Liabilities",
            "CurrentLiabilities",
        ],
    )


def get_cash(statement):
    return find_row(
        statement,
        [
            "Cash Cash Equivalents And Short Term Investments",
            "Cash And Cash Equivalents",
            "CashCashEquivalentsAndShortTermInvestments",
        ],
    )


def get_financial_series(statement, possible_names):
    """
    Return a financial row as a pandas Series.

    Returns None if none of the supplied row names
    exist in the statement.
    """

    series = find_row(statement, possible_names)

    if series is None:
        return None

    return pd.to_numeric(series, errors="coerce").dropna()
