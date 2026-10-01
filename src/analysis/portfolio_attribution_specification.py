"""
Portfolio Attribution Specification

Defines the contractual specification for portfolio return
attribution at security, sector, and peer-group levels.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PortfolioAttributionSpecification:
    """
    Contract for portfolio return attribution.
    """

    attribution_levels: Tuple[str, ...] = (
        "security",
        "sector",
        "peer_group",
    )

    required_portfolio_columns: Tuple[str, ...] = (
        "symbol",
        "weight",
    )

    required_classification_columns: Tuple[str, ...] = (
        "symbol",
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

    reconciliation_tolerance: float = 1e-10

    validation_rules: Tuple[str, ...] = (
        "Portfolio weights must be positive.",
        "Portfolio weights must sum to one.",
        "Security symbols must be unique.",
        "Every selected security must have classification data.",
        "Security returns must be finite.",
        "Beginning and ending weights must be finite.",
        "Security contributions must reconcile to portfolio return.",
        "Sector contributions must reconcile to portfolio return.",
        "Peer-group contributions must reconcile to portfolio return.",
        "Contribution percentages must be based on portfolio-level return contribution.",
    )


SPECIFICATION = PortfolioAttributionSpecification()


def validate_specification() -> None:
    """Validate the portfolio attribution specification."""

    assert SPECIFICATION.attribution_levels

    assert "security" in SPECIFICATION.attribution_levels
    assert "sector" in SPECIFICATION.attribution_levels
    assert "peer_group" in SPECIFICATION.attribution_levels

    assert SPECIFICATION.required_portfolio_columns
    assert SPECIFICATION.required_classification_columns
    assert SPECIFICATION.attribution_metrics

    assert SPECIFICATION.reconciliation_tolerance > 0
    assert SPECIFICATION.validation_rules


def print_specification() -> None:
    """Print the portfolio attribution specification."""

    print("=" * 70)
    print("PORTFOLIO ATTRIBUTION SPECIFICATION")
    print("=" * 70)

    print("\nATTRIBUTION LEVELS")
    for level in SPECIFICATION.attribution_levels:
        print(f"- {level}")

    print("\nREQUIRED PORTFOLIO COLUMNS")
    for column in SPECIFICATION.required_portfolio_columns:
        print(f"- {column}")

    print("\nREQUIRED CLASSIFICATION COLUMNS")
    for column in SPECIFICATION.required_classification_columns:
        print(f"- {column}")

    print("\nATTRIBUTION METRICS")
    for metric in SPECIFICATION.attribution_metrics:
        print(f"- {metric}")

    print("\nRECONCILIATION TOLERANCE")
    print(SPECIFICATION.reconciliation_tolerance)

    print("\nVALIDATION RULES")
    for rule in SPECIFICATION.validation_rules:
        print(f"- {rule}")

    print("\nSPECIFICATION STATUS")
    print("PASS")


if __name__ == "__main__":
    validate_specification()
    print_specification()
