"""
Portfolio Performance Evaluation Specification

Defines the contractual specification for evaluating portfolio
performance and benchmark-relative performance.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PerformanceEvaluationSpecification:
    """
    Contract for portfolio performance evaluation.
    """

    required_portfolio_columns: Tuple[str, ...] = (
        "portfolio_value",
        "daily_return",
        "cumulative_return",
        "drawdown",
    )

    required_benchmark_columns: Tuple[str, ...] = (
        "Close",
    )

    portfolio_performance_metrics: Tuple[str, ...] = (
        "total_return",
        "cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "maximum_drawdown",
    )

    benchmark_relative_metrics: Tuple[str, ...] = (
        "excess_total_return",
        "beta",
        "correlation",
        "tracking_error",
        "information_ratio",
        "alpha",
    )

    attribution_levels: Tuple[str, ...] = (
        "security",
        "sector",
        "peer_group",
    )

    attribution_metrics: Tuple[str, ...] = (
        "beginning_weight",
        "ending_weight",
        "return",
        "return_contribution",
        "return_contribution_pct",
    )

    transaction_cost_metrics: Tuple[str, ...] = (
        "gross_return",
        "net_return",
        "transaction_cost",
        "transaction_cost_drag",
    )

    supported_strategies: Tuple[str, ...] = (
        "buy_and_hold",
        "rebalanced",
    )

    performance_metric_definitions: Tuple[str, ...] = (
        "total_return is the compounded change in portfolio value.",
        "cagr is the annualized compounded growth rate.",
        "annualized_volatility is daily return volatility annualized using trading_days.",
        "sharpe_ratio uses arithmetic annualized return relative to the configured risk-free rate.",
        "sortino_ratio uses downside deviation relative to the configured risk-free rate.",
        "maximum_drawdown is the largest peak-to-trough decline.",
    )

    benchmark_metric_definitions: Tuple[str, ...] = (
        "excess_total_return equals portfolio total return minus benchmark total return.",
        "beta measures portfolio sensitivity to benchmark daily returns.",
        "correlation measures linear association between portfolio and benchmark daily returns.",
        "tracking_error is annualized volatility of active daily returns.",
        "information_ratio is annualized mean active return divided by tracking error.",
        "alpha is the CAPM-style annualized excess return after accounting for beta.",
    )

    attribution_principles: Tuple[str, ...] = (
        "Security contributions must reconcile to portfolio return.",
        "Sector contributions must reconcile to portfolio return.",
        "Peer-group contributions must reconcile to portfolio return.",
        "Attribution must use information available at the relevant portfolio date.",
        "Attribution must distinguish gross and net performance when transaction costs are present.",
        "Contribution percentages must be based on portfolio-level return contribution.",
    )

    validation_rules: Tuple[str, ...] = (
        "Portfolio values must be positive.",
        "Daily returns must be finite.",
        "Benchmark returns must be finite after alignment.",
        "Portfolio and benchmark observations must be date-aligned.",
        "Attribution contribution totals must reconcile within numerical tolerance.",
        "Transaction costs must be non-negative.",
        "Net performance must not exceed gross performance solely because of transaction costs.",
    )


SPECIFICATION = PerformanceEvaluationSpecification()


def validate_specification() -> None:
    """Validate the performance evaluation specification itself."""

    assert SPECIFICATION.required_portfolio_columns
    assert SPECIFICATION.required_benchmark_columns

    assert SPECIFICATION.portfolio_performance_metrics
    assert SPECIFICATION.benchmark_relative_metrics

    assert SPECIFICATION.attribution_levels
    assert SPECIFICATION.attribution_metrics

    assert SPECIFICATION.transaction_cost_metrics

    assert "buy_and_hold" in SPECIFICATION.supported_strategies
    assert "rebalanced" in SPECIFICATION.supported_strategies

    assert len(SPECIFICATION.performance_metric_definitions) == len(
        SPECIFICATION.portfolio_performance_metrics
    )

    assert len(SPECIFICATION.benchmark_metric_definitions) == len(
        SPECIFICATION.benchmark_relative_metrics
    )

    assert SPECIFICATION.attribution_principles
    assert SPECIFICATION.validation_rules


def print_specification() -> None:
    """Print the performance evaluation specification."""

    print("=" * 70)
    print("PORTFOLIO PERFORMANCE EVALUATION SPECIFICATION")
    print("=" * 70)

    print("\nREQUIRED PORTFOLIO COLUMNS")
    for column in SPECIFICATION.required_portfolio_columns:
        print(f"- {column}")

    print("\nREQUIRED BENCHMARK COLUMNS")
    for column in SPECIFICATION.required_benchmark_columns:
        print(f"- {column}")

    print("\nPORTFOLIO PERFORMANCE METRICS")
    for metric in SPECIFICATION.portfolio_performance_metrics:
        print(f"- {metric}")

    print("\nBENCHMARK-RELATIVE METRICS")
    for metric in SPECIFICATION.benchmark_relative_metrics:
        print(f"- {metric}")

    print("\nATTRIBUTION LEVELS")
    for level in SPECIFICATION.attribution_levels:
        print(f"- {level}")

    print("\nATTRIBUTION METRICS")
    for metric in SPECIFICATION.attribution_metrics:
        print(f"- {metric}")

    print("\nTRANSACTION COST METRICS")
    for metric in SPECIFICATION.transaction_cost_metrics:
        print(f"- {metric}")

    print("\nSUPPORTED STRATEGIES")
    for strategy in SPECIFICATION.supported_strategies:
        print(f"- {strategy}")

    print("\nATTRIBUTION PRINCIPLES")
    for principle in SPECIFICATION.attribution_principles:
        print(f"- {principle}")

    print("\nVALIDATION RULES")
    for rule in SPECIFICATION.validation_rules:
        print(f"- {rule}")

    print("\nSPECIFICATION STATUS")
    print("PASS")


if __name__ == "__main__":
    validate_specification()
    print_specification()
