from src.data.normalize_financials import (
    get_revenue,
    get_net_income,
    get_operating_income,
    get_total_debt,
    get_shareholders_equity,
    get_operating_cash_flow,
    get_capital_expenditure,
)


def get_latest_value(series):
    if series is None:
        return None

    values = series.dropna()

    if values.empty:
        return None

    return values.iloc[0]


def calculate_net_profit_margin(income_statement):
    revenue = get_latest_value(
        get_revenue(income_statement)
    )

    net_income = get_latest_value(
        get_net_income(income_statement)
    )

    if revenue is None or net_income is None:
        return None

    return net_income / revenue


def calculate_operating_margin(income_statement):
    revenue = get_latest_value(
        get_revenue(income_statement)
    )

    operating_income = get_latest_value(
        get_operating_income(income_statement)
    )

    if revenue is None or operating_income is None:
        return None

    return operating_income / revenue


def calculate_roe(income_statement, balance_sheet):
    net_income = get_latest_value(
        get_net_income(income_statement)
    )

    equity = get_latest_value(
        get_shareholders_equity(balance_sheet)
    )

    if net_income is None or equity is None:
        return None

    return net_income / equity


def calculate_debt_to_equity(balance_sheet):
    debt = get_latest_value(
        get_total_debt(balance_sheet)
    )

    equity = get_latest_value(
        get_shareholders_equity(balance_sheet)
    )

    if debt is None or equity is None:
        return None

    return debt / equity


def calculate_free_cash_flow(cash_flow_statement):
    operating_cash_flow = get_latest_value(
        get_operating_cash_flow(cash_flow_statement)
    )

    capital_expenditure = get_latest_value(
        get_capital_expenditure(cash_flow_statement)
    )

    if operating_cash_flow is None or capital_expenditure is None:
        return None

    return operating_cash_flow + capital_expenditure


def calculate_revenue_growth(income_statement):
    revenue = get_revenue(income_statement)

    if revenue is None:
        return None

    revenue = revenue.dropna()

    if len(revenue) < 2:
        return None

    latest_revenue = revenue.iloc[0]
    previous_revenue = revenue.iloc[1]

    return (latest_revenue / previous_revenue) - 1
