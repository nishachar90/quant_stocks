from src.config import (
    RISK_FREE_RATE,
    TRADING_DAYS,
)

from src.data.market_data import download_stock_data

from src.data.clean_data import clean_data

from src.data.fundamental_data import (
    get_income_statement,
    get_balance_sheet,
    get_cash_flow_statement,
)

from src.analysis.returns import (
    calculate_daily_returns,
    calculate_total_return,
    calculate_cagr,
)

from src.analysis.risk_metrics import (
    calculate_annualized_volatility,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_beta,
    calculate_correlation,
)

from src.analysis.fundamentals import (
    calculate_net_profit_margin,
    calculate_operating_margin,
    calculate_roe,
    calculate_debt_to_equity,
    calculate_free_cash_flow,
    calculate_revenue_growth,
)


def analyze_stock(
    symbol,
    market_symbol,
    start_date,
    end_date,
):
    # --------------------------------------------------
    # Market data
    # --------------------------------------------------

    stock = download_stock_data(
        symbol,
        start_date,
        end_date,
    )

    stock = clean_data(stock)

    # --------------------------------------------------
    # Returns
    # --------------------------------------------------

    daily_returns = calculate_daily_returns(stock)

    total_return = calculate_total_return(stock)

    cagr = calculate_cagr(stock)

    # --------------------------------------------------
    # Risk metrics
    # --------------------------------------------------

    volatility = calculate_annualized_volatility(
        daily_returns,
        TRADING_DAYS,
    )

    max_drawdown = calculate_max_drawdown(
        daily_returns
    )

    sharpe = calculate_sharpe_ratio(
        daily_returns,
        RISK_FREE_RATE,
        TRADING_DAYS,
    )

    sortino = calculate_sortino_ratio(
        daily_returns,
        RISK_FREE_RATE,
        TRADING_DAYS,
    )

    # --------------------------------------------------
    # Market comparison
    # --------------------------------------------------

    market = download_stock_data(
        market_symbol,
        start_date,
        end_date,
    )

    market_returns = calculate_daily_returns(
        market
    )

    beta = calculate_beta(
        daily_returns,
        market_returns,
    )

    correlation = calculate_correlation(
        daily_returns,
        market_returns,
    )

    # --------------------------------------------------
    # Fundamental data
    # --------------------------------------------------

    income_statement = get_income_statement(symbol)

    balance_sheet = get_balance_sheet(symbol)

    cash_flow = get_cash_flow_statement(symbol)

    # --------------------------------------------------
    # Fundamental metrics
    # --------------------------------------------------

    net_profit_margin = calculate_net_profit_margin(
        income_statement
    )

    operating_margin = calculate_operating_margin(
        income_statement
    )

    roe = calculate_roe(
        income_statement,
        balance_sheet,
    )

    debt_to_equity = calculate_debt_to_equity(
        balance_sheet
    )

    free_cash_flow = calculate_free_cash_flow(
        cash_flow
    )

    revenue_growth = calculate_revenue_growth(
        income_statement
    )

    # --------------------------------------------------
    # Final result
    # --------------------------------------------------

    return {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,

        "performance": {
            "total_return": total_return,
            "cagr": cagr,
        },

        "risk": {
            "annualized_volatility": volatility,
            "maximum_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
        },

        "market_relationship": {
            "beta": beta,
            "correlation": correlation,
        },

        "fundamentals": {
            "net_profit_margin": net_profit_margin,
            "operating_margin": operating_margin,
            "roe": roe,
            "debt_to_equity": debt_to_equity,
            "free_cash_flow": free_cash_flow,
            "revenue_growth": revenue_growth,
        },
    }

