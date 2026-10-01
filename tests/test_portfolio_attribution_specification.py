"""
Tests for the Portfolio Attribution Specification.
"""

from src.analysis.portfolio_attribution_specification import (
    SPECIFICATION,
    validate_specification,
)


def test_specification_validates():
    validate_specification()


def test_attribution_levels():
    assert SPECIFICATION.attribution_levels == (
        "security",
        "sector",
        "peer_group",
    )


def test_required_portfolio_columns():
    assert SPECIFICATION.required_portfolio_columns == (
        "symbol",
        "weight",
    )


def test_required_classification_columns():
    assert SPECIFICATION.required_classification_columns == (
        "symbol",
        "sector",
        "peer_group",
    )


def test_attribution_metrics():
    assert SPECIFICATION.attribution_metrics == (
        "beginning_weight",
        "ending_weight",
        "return",
        "return_contribution",
        "return_contribution_pct",
    )


def test_reconciliation_tolerance():
    assert SPECIFICATION.reconciliation_tolerance > 0


def test_security_level_supported():
    assert "security" in SPECIFICATION.attribution_levels


def test_sector_level_supported():
    assert "sector" in SPECIFICATION.attribution_levels


def test_peer_group_level_supported():
    assert "peer_group" in SPECIFICATION.attribution_levels


def test_validation_rules_exist():
    assert SPECIFICATION.validation_rules


def test_all_levels_are_unique():
    assert len(SPECIFICATION.attribution_levels) == len(
        set(SPECIFICATION.attribution_levels)
    )
