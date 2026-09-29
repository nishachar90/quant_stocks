from __future__ import annotations

import numpy as np
import pandas as pd


def validate_transaction_cost_bps(
    transaction_cost_bps: float,
) -> None:
    if transaction_cost_bps < 0:
        raise ValueError(
            "transaction_cost_bps cannot be negative."
        )


def calculate_one_way_turnover(
    old_weights: pd.Series,
    new_weights: pd.Series,
) -> float:
    validate_weights(
        old_weights,
        name="old_weights",
    )

    validate_weights(
        new_weights,
        name="new_weights",
    )

    combined = pd.concat(
        [
            old_weights.rename("old"),
            new_weights.rename("new"),
        ],
        axis=1,
    ).fillna(0.0)

    turnover = (
        (combined["new"] - combined["old"])
        .abs()
        .sum()
        / 2.0
    )

    return float(turnover)


def calculate_transaction_cost(
    traded_value: float,
    transaction_cost_bps: float,
) -> float:
    validate_transaction_cost_bps(
        transaction_cost_bps
    )

    if traded_value < 0:
        raise ValueError(
            "traded_value cannot be negative."
        )

    return float(
        traded_value
        * transaction_cost_bps
        / 10_000.0
    )


def validate_weights(
    weights: pd.Series,
    name: str = "weights",
) -> None:
    numeric = pd.to_numeric(
        weights,
        errors="coerce",
    )

    if numeric.isna().any():
        raise ValueError(
            f"{name} contains invalid weights."
        )

    if (numeric < 0).any():
        raise ValueError(
            f"{name} contains negative weights."
        )

    if not np.isclose(
        numeric.sum(),
        1.0,
        atol=1e-8,
    ):
        raise ValueError(
            f"{name} must sum to one."
        )


def calculate_turnover_series(
    target_weights: pd.DataFrame,
    actual_weights: pd.DataFrame | None = None,
) -> pd.Series:
    if target_weights.empty:
        raise ValueError(
            "target_weights is empty."
        )

    target_weights = target_weights.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if target_weights.isna().any().any():
        raise ValueError(
            "target_weights contains invalid values."
        )

    if actual_weights is None:
        actual_weights = (
            target_weights.shift(1)
            .fillna(0.0)
        )
    else:
        actual_weights = actual_weights.apply(
            pd.to_numeric,
            errors="coerce",
        )

        actual_weights = actual_weights.reindex(
            index=target_weights.index,
            columns=target_weights.columns,
        ).fillna(0.0)

    turnover = (
        target_weights
        .subtract(actual_weights)
        .abs()
        .sum(axis=1)
        / 2.0
    )

    return turnover


def calculate_transaction_cost_series(
    portfolio_values: pd.Series,
    turnover: pd.Series,
    transaction_cost_bps: float,
) -> pd.Series:
    validate_transaction_cost_bps(
        transaction_cost_bps
    )

    portfolio_values = pd.to_numeric(
        portfolio_values,
        errors="coerce",
    )

    turnover = pd.to_numeric(
        turnover,
        errors="coerce",
    )

    if portfolio_values.isna().any():
        raise ValueError(
            "portfolio_values contains invalid values."
        )

    if turnover.isna().any():
        raise ValueError(
            "turnover contains invalid values."
        )

    if (turnover < 0).any():
        raise ValueError(
            "turnover cannot be negative."
        )

    traded_value = (
        portfolio_values
        * turnover
    )

    return traded_value * (
        transaction_cost_bps
        / 10_000.0
    )
