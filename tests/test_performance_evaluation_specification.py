from src.analysis.performance_evaluation_specification import (
    SPECIFICATION,
    validate_specification,
)


def test_required_portfolio_columns():
    assert "portfolio_value" in SPECIFICATION.required_portfolio_columns
    assert "daily_return" in SPECIFICATION.required_portfolio_columns
    assert "cumulative_return" in SPECIFICATION.required_portfolio_columns
    assert "drawdown" in SPECIFICATION.required_portfolio_columns


def test_required_benchmark_columns():
    assert "Close" in SPECIFICATION.required_benchmark_columns


def test_portfolio_performance_metrics():
    expected = {
        "total_return",
        "cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "maximum_drawdown",
    }

    assert set(SPECIFICATION.portfolio_performance_metrics) == expected


def test_benchmark_relative_metrics():
    expected = {
        "excess_total_return",
        "beta",
        "correlation",
        "tracking_error",
        "information_ratio",
        "alpha",
    }

    assert set(SPECIFICATION.benchmark_relative_metrics) == expected


def test_attribution_levels():
    expected = {
        "security",
        "sector",
        "peer_group",
    }

    assert set(SPECIFICATION.attribution_levels) == expected


def test_attribution_metrics():
    expected = {
        "beginning_weight",
        "ending_weight",
        "return",
        "return_contribution",
        "return_contribution_pct",
    }

    assert set(SPECIFICATION.attribution_metrics) == expected


def test_transaction_cost_metrics():
    expected = {
        "gross_return",
        "net_return",
        "transaction_cost",
        "transaction_cost_drag",
    }

    assert set(SPECIFICATION.transaction_cost_metrics) == expected


def test_supported_strategies():
    assert "buy_and_hold" in SPECIFICATION.supported_strategies
    assert "rebalanced" in SPECIFICATION.supported_strategies


def test_attribution_principles():
    assert len(SPECIFICATION.attribution_principles) >= 1

    assert any(
        "reconcile" in principle.lower()
        for principle in SPECIFICATION.attribution_principles
    )


def test_validation_rules():
    assert len(SPECIFICATION.validation_rules) >= 1

    assert any(
        "transaction costs" in rule.lower()
        for rule in SPECIFICATION.validation_rules
    )


def test_specification_validation():
    validate_specification()
