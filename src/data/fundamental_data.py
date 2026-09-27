import yfinance as yf


def get_income_statement(symbol):
    ticker = yf.Ticker(symbol)

    return ticker.income_stmt


def get_balance_sheet(symbol):
    ticker = yf.Ticker(symbol)

    return ticker.balance_sheet


def get_cash_flow_statement(symbol):
    ticker = yf.Ticker(symbol)

    return ticker.cashflow
