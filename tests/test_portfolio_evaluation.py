import pandas as pd

from scripts.generate_portfolio_evaluation import (
    evaluate_strategy,
    load_backtest,
)


def test_load_backtest():
    """Verify that the portfolio backtest loads successfully."""

    backtest = load_backtest()

    assert isinstance(backtest, pd.DataFrame)
    assert not backtest.empty
    assert "date" in backtest.columns
    assert "portfolio_value" in backtest.columns
    assert "daily_return" in backtest.columns


def test_evaluate_strategy():
    """Verify successful portfolio evaluation and attribution outputs."""

    backtest = load_backtest()

    portfolio = pd.DataFrame(
        {
            "symbol": ["RELIANCE", "TITAN"],
            "weight": [0.60, 0.40],
        }
    )

    classification = pd.DataFrame(
        {
            "symbol": ["RELIANCE", "TITAN"],
            "sector": [
                "Oil Gas & Consumable Fuels",
                "Consumer Durables",
            ],
            "business_type": [
                "Integrated Energy",
                "Jewellery",
            ],
            "peer_group": [
                "Integrated Energy",
                "Jewellery",
            ],
        }
    )

    benchmark = pd.DataFrame(
        {
            "Close": [100.0, 101.0, 102.0],
        }
    )

    result = evaluate_strategy(
        backtest=backtest,
        benchmark=benchmark,
        portfolio=portfolio,
        classification=classification,
        strategy="test_strategy",
    )

    assert isinstance(result, tuple)
    assert len(result) == 4

    performance = result[0]
    security_attribution = result[1]
    sector_attribution = result[2]
    peer_group_attribution = result[3]

    assert isinstance(
        performance,
        pd.DataFrame,
    )

    assert isinstance(
        security_attribution,
        pd.DataFrame,
    )

    assert isinstance(
        sector_attribution,
        pd.DataFrame,
    )

    assert isinstance(
        peer_group_attribution,
        pd.DataFrame,
    )

    assert not performance.empty
    assert not security_attribution.empty
    assert not sector_attribution.empty
    assert not peer_group_attribution.empty

    assert "total_return" in performance.columns
    assert "cagr" in performance.columns
    assert "sharpe_ratio" in performance.columns
    assert "sortino_ratio" in performance.columns
    assert "maximum_drawdown" in performance.columns
    assert "excess_total_return" in performance.columns
    assert "beta" in performance.columns
    assert "correlation" in performance.columns

    assert "symbol" in security_attribution.columns
    assert "return" in security_attribution.columns
    assert "return_contribution" in security_attribution.columns

    assert "sector" in sector_attribution.columns
    assert "return" in sector_attribution.columns
    assert "return_contribution" in sector_attribution.columns

    assert "peer_group" in peer_group_attribution.columns
    assert "return" in peer_group_attribution.columns
    assert "return_contribution" in peer_group_attribution.columns
