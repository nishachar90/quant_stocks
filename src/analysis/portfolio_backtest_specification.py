# src/analysis/portfolio_backtest_specification.py

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioBacktestSpecification:
    initial_capital: float = 1_000_000.0
    risk_free_rate: float = 0.06
    trading_days: int = 252

    start_date_policy: str = "common_available_history"

    include_buy_and_hold: bool = True
    include_rebalanced: bool = True

    transaction_cost_bps: float = 0.0


def validate_specification(
    specification: PortfolioBacktestSpecification,
) -> dict[str, bool]:
    return {
        "initial_capital": (
            specification.initial_capital > 0
        ),
        "risk_free_rate": (
            specification.risk_free_rate >= 0
        ),
        "trading_days": (
            specification.trading_days > 0
        ),
        "start_date_policy": (
            specification.start_date_policy
            == "common_available_history"
        ),
        "buy_and_hold": (
            specification.include_buy_and_hold
        ),
        "rebalanced": (
            specification.include_rebalanced
        ),
        "transaction_costs": (
            specification.transaction_cost_bps >= 0
        ),
    }


def audit_specification(
    specification: PortfolioBacktestSpecification,
) -> bool:
    checks = validate_specification(
        specification
    )

    return all(checks.values())


def print_specification_audit(
    specification: PortfolioBacktestSpecification,
) -> None:
    checks = validate_specification(
        specification
    )

    print("=" * 70)
    print("NIFTY 50 PORTFOLIO BACKTEST SPECIFICATION AUDIT")
    print("=" * 70)

    print()
    print("BACKTEST SPECIFICATION")

    print(
        f"- initial_capital: "
        f"{specification.initial_capital}"
    )

    print(
        f"- risk_free_rate: "
        f"{specification.risk_free_rate}"
    )

    print(
        f"- trading_days: "
        f"{specification.trading_days}"
    )

    print(
        f"- start_date_policy: "
        f"{specification.start_date_policy}"
    )

    print(
        f"- include_buy_and_hold: "
        f"{specification.include_buy_and_hold}"
    )

    print(
        f"- include_rebalanced: "
        f"{specification.include_rebalanced}"
    )

    print(
        f"- transaction_cost_bps: "
        f"{specification.transaction_cost_bps}"
    )

    print()
    print("VALIDATION")

    for name, passed in checks.items():
        print(
            f"- {name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    print()
    print(
        "BACKTEST SPECIFICATION AUDIT: "
        + (
            "PASS"
            if all(checks.values())
            else "FAIL"
        )
    )

    print("=" * 70)


if __name__ == "__main__":
    specification = (
        PortfolioBacktestSpecification()
    )

    print_specification_audit(
        specification
    )
