"""
Tests for the Multi-Factor Scoring Engine
=========================================

These tests verify:

1. Percentile scoring.
2. Score direction.
3. Reference-group selection.
4. Missing-value handling.
5. Factor scoring.
6. Overall scoring.
7. Output score ranges.
"""

import numpy as np
import pandas as pd

from src.analysis.factor_scoring import (
    calculate_factor_score,
    calculate_overall_score,
    percentile_score,
    select_reference_group,
)


# ======================================================================
# PERCENTILE SCORE TESTS
# ======================================================================

def test_percentile_score_higher_is_better():
    values = pd.Series(
        [10, 20, 30, 40]
    )

    score = percentile_score(
        value=30,
        reference_values=values,
        direction="higher_is_better",
    )

    assert np.isclose(
        score,
        66.6666666667,
    )


def test_percentile_score_lower_is_better():
    values = pd.Series(
        [10, 20, 30, 40]
    )

    score = percentile_score(
        value=20,
        reference_values=values,
        direction="lower_is_better",
    )

    assert np.isclose(
        score,
        66.6666666667,
    )


def test_percentile_score_tied_values():
    values = pd.Series(
        [10, 20, 20, 30]
    )

    score = percentile_score(
        value=20,
        reference_values=values,
        direction="higher_is_better",
    )

    assert np.isclose(
        score,
        50.0,
    )


def test_percentile_score_missing_value():
    values = pd.Series(
        [10, 20, 30, 40]
    )

    score = percentile_score(
        value=np.nan,
        reference_values=values,
        direction="higher_is_better",
    )

    assert np.isnan(score)


def test_percentile_score_empty_reference():
    values = pd.Series(
        dtype=float
    )

    score = percentile_score(
        value=20,
        reference_values=values,
        direction="higher_is_better",
    )

    assert np.isnan(score)


# ======================================================================
# REFERENCE GROUP TESTS
# ======================================================================

def test_reference_group_prefers_peer_group():
    df = pd.DataFrame(
        {
            "symbol": [
                "A",
                "B",
                "C",
            ],
            "peer_group": [
                "Banks",
                "Banks",
                "Banks",
            ],
            "sector": [
                "Financial Services",
                "Financial Services",
                "Financial Services",
            ],
        }
    )

    level, name = select_reference_group(
        df,
        0,
    )

    assert level == "peer_group"
    assert name == "Banks"


def test_reference_group_falls_back_to_sector():
    df = pd.DataFrame(
        {
            "symbol": [
                "A",
                "B",
                "C",
            ],
            "peer_group": [
                "Specialist A",
                "Specialist B",
                "Specialist C",
            ],
            "sector": [
                "Financial Services",
                "Financial Services",
                "Financial Services",
            ],
        }
    )

    level, name = select_reference_group(
        df,
        0,
    )

    assert level == "sector"
    assert name == "Financial Services"


def test_reference_group_falls_back_to_universe():
    df = pd.DataFrame(
        {
            "symbol": [
                "A",
                "B",
                "C",
            ],
            "peer_group": [
                "Specialist A",
                "Specialist B",
                "Specialist C",
            ],
            "sector": [
                "Sector A",
                "Sector B",
                "Sector C",
            ],
        }
    )

    level, name = select_reference_group(
        df,
        0,
    )

    assert level == "universe"
    assert name == "NIFTY50"


# ======================================================================
# FACTOR SCORE TESTS
# ======================================================================

def test_factor_score_single_metric():
    df = pd.DataFrame(
        {
            "cagr_score": [
                20.0,
                40.0,
                60.0,
                80.0,
            ]
        }
    )

    metrics = {
        "cagr": {
            "role": "primary",
            "direction": "higher_is_better",
        }
    }

    scores = calculate_factor_score(
        df,
        "Performance",
        metrics,
    )

    expected = pd.Series(
        [
            20.0,
            40.0,
            60.0,
            80.0,
        ]
    )

    pd.testing.assert_series_equal(
        scores.reset_index(drop=True),
        expected,
        check_names=False,
    )


def test_factor_score_renormalizes_missing_values():
    df = pd.DataFrame(
        {
            "roe_score": [
                80.0,
                60.0,
                40.0,
            ],
            "net_profit_margin_score": [
                20.0,
                np.nan,
                60.0,
            ],
        }
    )

    metrics = {
        "roe": {
            "role": "primary",
            "direction": "higher_is_better",
        },
        "net_profit_margin": {
            "role": "supporting",
            "direction": "higher_is_better",
        },
    }

    scores = calculate_factor_score(
        df,
        "Quality",
        metrics,
    )

    # Row 0:
    # ROE = 80 with weight 2
    # NPM = 20 with weight 1
    #
    # (80*2 + 20*1) / 3 = 60
    assert np.isclose(
        scores.iloc[0],
        60.0,
    )

    # Row 1:
    # NPM is missing.
    # Available ROE weight is renormalized to 100%.
    assert np.isclose(
        scores.iloc[1],
        60.0,
    )

    # Row 2:
    # (40*2 + 60*1) / 3 = 46.6667
    assert np.isclose(
        scores.iloc[2],
        46.6666666667,
    )


# ======================================================================
# OVERALL SCORE TESTS
# ======================================================================

def test_overall_score_equal_factor_weights():
    df = pd.DataFrame(
        {
            "performance_score": [
                20.0,
                40.0,
            ],
            "risk-adjusted_performance_score": [
                40.0,
                60.0,
            ],
            "risk_score": [
                60.0,
                80.0,
            ],
            "quality_score": [
                80.0,
                20.0,
            ],
            "growth_score": [
                100.0,
                40.0,
            ],
        }
    )

    scores = calculate_overall_score(
        df
    )

    # All five factors have 20% weight.
    #
    # Row 0:
    # (20 + 40 + 60 + 80 + 100) / 5 = 60
    #
    # Row 1:
    # (40 + 60 + 80 + 20 + 40) / 5 = 48

    assert np.isclose(
        scores.iloc[0],
        60.0,
    )

    assert np.isclose(
        scores.iloc[1],
        48.0,
    )


def test_overall_score_renormalizes_missing_factors():
    df = pd.DataFrame(
        {
            "performance_score": [
                80.0,
            ],
            "risk-adjusted_performance_score": [
                np.nan,
            ],
            "risk_score": [
                60.0,
            ],
            "quality_score": [
                40.0,
            ],
            "growth_score": [
                20.0,
            ],
        }
    )

    scores = calculate_overall_score(
        df
    )

    # Four available factors:
    #
    # (80 + 60 + 40 + 20) / 4 = 50

    assert np.isclose(
        scores.iloc[0],
        50.0,
    )


# ======================================================================
# TEST RUNNER
# ======================================================================

if __name__ == "__main__":

    test_percentile_score_higher_is_better()
    test_percentile_score_lower_is_better()
    test_percentile_score_tied_values()
    test_percentile_score_missing_value()
    test_percentile_score_empty_reference()

    test_reference_group_prefers_peer_group()
    test_reference_group_falls_back_to_sector()
    test_reference_group_falls_back_to_universe()

    test_factor_score_single_metric()
    test_factor_score_renormalizes_missing_values()

    test_overall_score_equal_factor_weights()
    test_overall_score_renormalizes_missing_factors()

    print()
    print("=" * 70)
    print("ALL FACTOR SCORING TESTS PASSED")
    print("=" * 70)
