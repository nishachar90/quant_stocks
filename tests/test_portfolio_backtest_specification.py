# tests/test_portfolio_backtest_specification.py

from src.analysis.portfolio_backtest_specification import (
    PortfolioBacktestSpecification,
    audit_specification,
    validate_specification,
)


def test_default_specification_passes():
    specification = (
        PortfolioBacktestSpecification()
    )

    assert audit_specification(
        specification
    )


def test_initial_capital_positive():
    specification = (
        PortfolioBacktestSpecification(
            initial_capital=1_000_000
        )
    )

    checks = validate_specification(
        specification
    )

    assert checks["initial_capital"]


def test_zero_initial_capital_fails():
    specification = (
        PortfolioBacktestSpecification(
            initial_capital=0
        )
    )

    checks = validate_specification(
        specification
    )

    assert not checks["initial_capital"]


def test_negative_risk_free_rate_fails():
    specification = (
        PortfolioBacktestSpecification(
            risk_free_rate=-0.01
        )
    )

    checks = validate_specification(
        specification
    )

    assert not checks["risk_free_rate"]


def test_trading_days_positive():
    specification = (
        PortfolioBacktestSpecification(
            trading_days=252
        )
    )

    checks = validate_specification(
        specification
    )

    assert checks["trading_days"]


def test_zero_trading_days_fails():
    specification = (
        PortfolioBacktestSpecification(
            trading_days=0
        )
    )

    checks = validate_specification(
        specification
    )

    assert not checks["trading_days"]


def test_common_history_policy():
    specification = (
        PortfolioBacktestSpecification(
            start_date_policy="common_available_history"
        )
    )

    checks = validate_specification(
        specification
    )

    assert checks["start_date_policy"]


def test_invalid_history_policy_fails():
    specification = (
        PortfolioBacktestSpecification(
            start_date_policy="fixed_2016"
        )
    )

    checks = validate_specification(
        specification
    )

    assert not checks["start_date_policy"]


def test_buy_and_hold_enabled():
    specification = (
        PortfolioBacktestSpecification(
            include_buy_and_hold=True
        )
    )

    checks = validate_specification(
        specification
    )

    assert checks["buy_and_hold"]


def test_rebalanced_enabled():
    specification = (
        PortfolioBacktestSpecification(
            include_rebalanced=True
        )
    )

    checks = validate_specification(
        specification
    )

    assert checks["rebalanced"]


def test_negative_transaction_cost_fails():
    specification = (
        PortfolioBacktestSpecification(
            transaction_cost_bps=-1
        )
    )

    checks = validate_specification(
        specification
    )

    assert not checks["transaction_costs"]
