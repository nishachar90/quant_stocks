"""Portfolio Attribution"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


AttributionLevel = Literal["security", "sector", "peer_group"]


@dataclass(frozen=True)
class PortfolioAttributionConfig:
    """Configuration for portfolio return attribution."""

    reconciliation_tolerance: float = 1e-10


REQUIRED_PORTFOLIO_COLUMNS = [
    "symbol",
    "weight",
]

REQUIRED_CLASSIFICATION_COLUMNS = [
    "symbol",
    "sector",
    "peer_group",
]

ATTRIBUTION_COLUMNS = [
    "beginning_weight",
    "ending_weight",
    "return",
    "return_contribution",
    "return_contribution_pct",
]


def _validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    dataframe_name: str,
) -> None:
    """Validate that a dataframe contains all required columns."""

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{dataframe_name} is missing required columns: {missing}"
        )


def _validate_portfolio(
    portfolio: pd.DataFrame,
    config: PortfolioAttributionConfig,
) -> None:
    """Validate portfolio weights and symbols."""

    _validate_required_columns(
        portfolio,
        REQUIRED_PORTFOLIO_COLUMNS,
        "portfolio",
    )

    if portfolio.empty:
        raise ValueError("Portfolio must not be empty.")

    if portfolio["symbol"].duplicated().any():
        raise ValueError("Security symbols must be unique.")

    weights = pd.to_numeric(
        portfolio["weight"],
        errors="coerce",
    )

    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError("Portfolio weights must be finite.")

    if (weights <= 0).any():
        raise ValueError("Portfolio weights must be positive.")

    weight_sum = float(weights.sum())

    if not np.isclose(
        weight_sum,
        1.0,
        atol=config.reconciliation_tolerance,
        rtol=0.0,
    ):
        raise ValueError(
            f"Portfolio weights must sum to one. "
            f"Observed sum: {weight_sum}"
        )


def _validate_classification(
    portfolio: pd.DataFrame,
    classification: pd.DataFrame,
) -> None:
    """Validate classification data for every portfolio security."""

    _validate_required_columns(
        classification,
        REQUIRED_CLASSIFICATION_COLUMNS,
        "classification",
    )

    if classification["symbol"].duplicated().any():
        raise ValueError("Classification symbols must be unique.")

    portfolio_symbols = set(
        portfolio["symbol"].astype(str)
    )

    classification_symbols = set(
        classification["symbol"].astype(str)
    )

    missing = sorted(
        portfolio_symbols - classification_symbols
    )

    if missing:
        raise ValueError(
            "Every selected security must have classification data. "
            f"Missing: {missing}"
        )


def _validate_returns(
    security_returns: pd.Series,
    portfolio_symbols: set[str],
) -> pd.Series:
    """Validate and normalize security returns."""

    returns = pd.to_numeric(
        security_returns,
        errors="coerce",
    )

    returns.index = returns.index.astype(str)

    missing = sorted(
        portfolio_symbols - set(returns.index)
    )

    if missing:
        raise ValueError(
            "Security returns are missing for: "
            f"{missing}"
        )

    returns = returns.loc[
        sorted(portfolio_symbols)
    ]

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError(
            "Security returns must be finite."
        )

    return returns


def _prepare_base_attribution(
    portfolio: pd.DataFrame,
    classification: pd.DataFrame,
    security_returns: pd.Series,
) -> pd.DataFrame:
    """Create internal security-level attribution data."""

    # Keep only the portfolio fields required for attribution.
    # The portfolio file may already contain columns such as
    # sector, business_type, peer_group, etc. Keeping only
    # symbol and weight prevents merge suffixes such as
    # sector_x / sector_y from being created.
    portfolio_data = portfolio[
        [
            "symbol",
            "weight",
        ]
    ].copy()

    portfolio_data["symbol"] = (
        portfolio_data["symbol"]
        .astype(str)
    )

    portfolio_data["weight"] = pd.to_numeric(
        portfolio_data["weight"],
        errors="coerce",
    )

    classification_data = classification[
        REQUIRED_CLASSIFICATION_COLUMNS
    ].copy()

    classification_data["symbol"] = (
        classification_data["symbol"]
        .astype(str)
    )

    returns = security_returns.copy()
    returns.index = returns.index.astype(str)

    portfolio_data = portfolio_data.merge(
        classification_data,
        on="symbol",
        how="left",
        validate="one_to_one",
    )

    portfolio_data["return"] = (
        portfolio_data["symbol"]
        .map(returns)
        .astype(float)
    )

    portfolio_data["beginning_weight"] = (
        portfolio_data["weight"]
    )

    portfolio_data["ending_weight"] = (
        portfolio_data["weight"]
    )

    portfolio_data["return_contribution"] = (
        portfolio_data["beginning_weight"]
        * portfolio_data["return"]
    )

    portfolio_return = float(
        portfolio_data["return_contribution"].sum()
    )

    if np.isclose(
        portfolio_return,
        0.0,
        atol=1e-15,
        rtol=0.0,
    ):
        portfolio_data["return_contribution_pct"] = np.nan
    else:
        portfolio_data["return_contribution_pct"] = (
            portfolio_data["return_contribution"]
            / portfolio_return
        )

    return portfolio_data


def _aggregate_attribution(
    security_attribution: pd.DataFrame,
    level: AttributionLevel,
) -> pd.DataFrame:
    """Aggregate attribution to the requested level."""

    if level == "security":
        columns = [
            "symbol",
            "sector",
            "peer_group",
            *ATTRIBUTION_COLUMNS,
        ]

        return security_attribution[
            columns
        ].copy()

    if level == "sector":
        group_column = "sector"
    elif level == "peer_group":
        group_column = "peer_group"
    else:
        raise ValueError(
            "level must be one of: "
            "'security', 'sector', 'peer_group'"
        )

    grouped = (
        security_attribution
        .groupby(
            group_column,
            dropna=False,
        )
        .agg(
            beginning_weight=(
                "beginning_weight",
                "sum",
            ),
            ending_weight=(
                "ending_weight",
                "sum",
            ),
            return_contribution=(
                "return_contribution",
                "sum",
            ),
        )
        .reset_index()
    )

    grouped["return"] = np.where(
        grouped["beginning_weight"] != 0,
        grouped["return_contribution"]
        / grouped["beginning_weight"],
        np.nan,
    )

    total_contribution = float(
        grouped["return_contribution"].sum()
    )

    if np.isclose(
        total_contribution,
        0.0,
        atol=1e-15,
        rtol=0.0,
    ):
        grouped["return_contribution_pct"] = np.nan
    else:
        grouped["return_contribution_pct"] = (
            grouped["return_contribution"]
            / total_contribution
        )

    return grouped[
        [
            group_column,
            "beginning_weight",
            "ending_weight",
            "return",
            "return_contribution",
            "return_contribution_pct",
        ]
    ].copy()


def calculate_attribution(
    portfolio: pd.DataFrame,
    classification: pd.DataFrame,
    security_returns: pd.Series,
    level: AttributionLevel = "security",
    config: PortfolioAttributionConfig | None = None,
) -> pd.DataFrame:
    """Calculate portfolio return attribution."""

    if config is None:
        config = PortfolioAttributionConfig()

    _validate_portfolio(
        portfolio,
        config,
    )

    _validate_classification(
        portfolio,
        classification,
    )

    portfolio_symbols = set(
        portfolio["symbol"]
        .astype(str)
    )

    returns = _validate_returns(
        security_returns,
        portfolio_symbols,
    )

    security_attribution = _prepare_base_attribution(
        portfolio,
        classification,
        returns,
    )

    result = _aggregate_attribution(
        security_attribution,
        level,
    )

    return result.reset_index(drop=True)


def calculate_security_attribution(
    portfolio: pd.DataFrame,
    classification: pd.DataFrame,
    security_returns: pd.Series,
    config: PortfolioAttributionConfig | None = None,
) -> pd.DataFrame:
    """Calculate security-level portfolio return attribution."""

    return calculate_attribution(
        portfolio=portfolio,
        classification=classification,
        security_returns=security_returns,
        level="security",
        config=config,
    )


def calculate_sector_attribution(
    portfolio: pd.DataFrame,
    classification: pd.DataFrame,
    security_returns: pd.Series,
    config: PortfolioAttributionConfig | None = None,
) -> pd.DataFrame:
    """Calculate sector-level portfolio return attribution."""

    return calculate_attribution(
        portfolio=portfolio,
        classification=classification,
        security_returns=security_returns,
        level="sector",
        config=config,
    )


def calculate_peer_group_attribution(
    portfolio: pd.DataFrame,
    classification: pd.DataFrame,
    security_returns: pd.Series,
    config: PortfolioAttributionConfig | None = None,
) -> pd.DataFrame:
    """Calculate peer-group-level portfolio return attribution."""

    return calculate_attribution(
        portfolio=portfolio,
        classification=classification,
        security_returns=security_returns,
        level="peer_group",
        config=config,
    )
