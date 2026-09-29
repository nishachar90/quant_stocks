"""
NIFTY 50 Portfolio Construction Engine
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioConstructionConfig:
    target_holdings: int = 10
    minimum_holdings: int = 8
    maximum_holdings: int = 12

    minimum_position_weight: float = 0.05
    maximum_position_weight: float = 0.15

    maximum_sector_weight: float = 0.30
    maximum_business_type_weight: float = 0.20

    weighting_method: str = "score_proportional"
    selection_method: str = "rank_order"
    target_total_weight: float = 1.0


REQUIRED_COLUMNS = {
    "symbol",
    "overall_score",
    "overall_rank",
    "eligibility",
    "sector",
    "business_type",
}


def _is_eligible(value: object) -> bool:
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    return str(value).strip().lower() in {
        "true",
        "yes",
        "eligible",
        "1",
    }


def _validate_input(dataset: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(dataset.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    if dataset.empty:
        raise ValueError("Input dataset is empty.")

    if dataset["symbol"].duplicated().any():
        raise ValueError("Duplicate symbols found in input dataset.")

    if dataset["overall_rank"].isna().any():
        raise ValueError("Missing overall_rank values found.")

    if dataset["overall_score"].isna().any():
        raise ValueError("Missing overall_score values found.")

    if dataset["sector"].isna().any():
        raise ValueError("Missing sector values found.")

    if dataset["business_type"].isna().any():
        raise ValueError("Missing business_type values found.")


def _selection_group_counts(
    portfolio: pd.DataFrame,
) -> tuple[dict[str, int], dict[str, int]]:
    sectors = (
        portfolio["sector"]
        .astype(str)
        .value_counts()
        .to_dict()
    )

    business_types = (
        portfolio["business_type"]
        .astype(str)
        .value_counts()
        .to_dict()
    )

    return sectors, business_types


def _select_holdings(
    dataset: pd.DataFrame,
    config: PortfolioConstructionConfig,
) -> pd.DataFrame:
    eligible = dataset[
        dataset["eligibility"].apply(_is_eligible)
    ].copy()

    eligible = eligible.sort_values(
        by=["overall_rank", "symbol"],
        ascending=[True, True],
        kind="stable",
    )

    if len(eligible) < config.minimum_holdings:
        raise ValueError(
            "Not enough eligible stocks to construct the portfolio. "
            f"Required at least {config.minimum_holdings}, "
            f"found {len(eligible)}."
        )

    if config.target_holdings <= 0:
        raise ValueError(
            "target_holdings must be greater than zero."
        )

    if (
        config.target_holdings < config.minimum_holdings
        or config.target_holdings > config.maximum_holdings
    ):
        raise ValueError(
            "target_holdings must lie between minimum_holdings "
            "and maximum_holdings."
        )

    target_selection_weight = (
        config.target_total_weight
        / config.target_holdings
    )

    selected_rows: list[pd.Series] = []

    sector_counts: dict[str, int] = {}
    business_type_counts: dict[str, int] = {}

    for _, row in eligible.iterrows():
        if len(selected_rows) >= config.target_holdings:
            break

        sector = str(row["sector"])
        business_type = str(row["business_type"])

        prospective_sector_weight = (
            sector_counts.get(sector, 0) + 1
        ) * target_selection_weight

        prospective_business_weight = (
            business_type_counts.get(business_type, 0) + 1
        ) * target_selection_weight

        if (
            prospective_sector_weight
            > config.maximum_sector_weight + 1e-12
        ):
            continue

        if (
            prospective_business_weight
            > config.maximum_business_type_weight + 1e-12
        ):
            continue

        selected_rows.append(row)

        sector_counts[sector] = (
            sector_counts.get(sector, 0) + 1
        )

        business_type_counts[business_type] = (
            business_type_counts.get(business_type, 0) + 1
        )

    if len(selected_rows) < config.minimum_holdings:
        raise ValueError(
            "Portfolio construction could not satisfy the minimum "
            f"holding count of {config.minimum_holdings}. "
            f"Only {len(selected_rows)} holdings could be selected "
            "under the concentration constraints."
        )

    return pd.DataFrame(selected_rows).reset_index(drop=True)


def _raw_score_weights(
    scores: pd.Series,
    target_total: float,
) -> np.ndarray:
    values = pd.to_numeric(
        scores,
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Scores must contain only finite values."
        )

    if (values <= 0).any():
        raise ValueError(
            "Score-proportional weighting requires strictly "
            "positive overall scores."
        )

    total_score = values.sum()

    if total_score <= 0:
        raise ValueError(
            "Total score must be positive."
        )

    return (
        values / total_score
    ) * target_total


def _group_total(
    weights: np.ndarray,
    groups: np.ndarray,
    group: str,
) -> float:
    return float(
        weights[groups == group].sum()
    )


def _can_receive(
    index: int,
    amount: float,
    weights: np.ndarray,
    portfolio: pd.DataFrame,
    config: PortfolioConstructionConfig,
) -> bool:
    if amount <= 0:
        return False

    new_weight = weights[index] + amount

    if (
        new_weight
        > config.maximum_position_weight + 1e-12
    ):
        return False

    sectors = (
        portfolio["sector"]
        .astype(str)
        .to_numpy()
    )

    business_types = (
        portfolio["business_type"]
        .astype(str)
        .to_numpy()
    )

    sector = sectors[index]
    business_type = business_types[index]

    new_sector_weight = (
        _group_total(weights, sectors, sector)
        + amount
    )

    if (
        new_sector_weight
        > config.maximum_sector_weight + 1e-12
    ):
        return False

    new_business_weight = (
        _group_total(
            weights,
            business_types,
            business_type,
        )
        + amount
    )

    if (
        new_business_weight
        > config.maximum_business_type_weight + 1e-12
    ):
        return False

    return True


def _can_give(
    index: int,
    amount: float,
    weights: np.ndarray,
    config: PortfolioConstructionConfig,
) -> bool:
    return (
        weights[index] - amount
        >= config.minimum_position_weight - 1e-12
    )


def _allocate_score_proportional_weights(
    portfolio: pd.DataFrame,
    config: PortfolioConstructionConfig,
) -> np.ndarray:
    """
    Start from an exactly feasible equal-weight portfolio and move
    weights toward the score-proportional target without ever
    violating portfolio constraints.
    """

    count = len(portfolio)

    minimum_total = (
        count * config.minimum_position_weight
    )

    maximum_total = (
        count * config.maximum_position_weight
    )

    if (
        minimum_total
        > config.target_total_weight + 1e-12
    ):
        raise ValueError(
            "Minimum position weight makes the portfolio infeasible."
        )

    if (
        maximum_total
        < config.target_total_weight - 1e-12
    ):
        raise ValueError(
            "Maximum position weight makes the portfolio infeasible."
        )

    sectors = (
        portfolio["sector"]
        .astype(str)
        .to_numpy()
    )

    business_types = (
        portfolio["business_type"]
        .astype(str)
        .to_numpy()
    )

    equal_weight = (
        config.target_total_weight
        / count
    )

    weights = np.full(
        count,
        equal_weight,
        dtype=float,
    )

    raw_weights = _raw_score_weights(
        portfolio["overall_score"],
        config.target_total_weight,
    )

    order = np.argsort(
        -pd.to_numeric(
            portfolio["overall_score"],
            errors="coerce",
        ).to_numpy(dtype=float)
    )

    tolerance = 1e-11

    for _ in range(500):

        moved = False

        errors = (
            raw_weights - weights
        )

        positive_indices = [
            int(i)
            for i in order
            if errors[i] > tolerance
        ]

        negative_indices = [
            int(i)
            for i in reversed(order)
            if errors[i] < -tolerance
        ]

        if not positive_indices or not negative_indices:
            break

        for receiver in positive_indices:

            receiver_need = (
                raw_weights[receiver]
                - weights[receiver]
            )

            if receiver_need <= tolerance:
                continue

            for donor in negative_indices:

                if donor == receiver:
                    continue

                donor_available = (
                    weights[donor]
                    - raw_weights[donor]
                )

                if donor_available <= tolerance:
                    continue

                if not _can_give(
                    donor,
                    donor_available,
                    weights,
                    config,
                ):
                    donor_available = max(
                        0.0,
                        weights[donor]
                        - config.minimum_position_weight,
                    )

                if donor_available <= tolerance:
                    continue

                receiver_capacity = (
                    config.maximum_position_weight
                    - weights[receiver]
                )

                if receiver_capacity <= tolerance:
                    continue

                receiver_sector = sectors[receiver]
                receiver_business = (
                    business_types[receiver]
                )

                sector_capacity = (
                    config.maximum_sector_weight
                    - _group_total(
                        weights,
                        sectors,
                        receiver_sector,
                    )
                )

                business_capacity = (
                    config.maximum_business_type_weight
                    - _group_total(
                        weights,
                        business_types,
                        receiver_business,
                    )
                )

                receiver_capacity = min(
                    receiver_capacity,
                    sector_capacity,
                    business_capacity,
                )

                if receiver_capacity <= tolerance:
                    continue

                transfer = min(
                    receiver_need,
                    donor_available,
                    receiver_capacity,
                )

                if transfer <= tolerance:
                    continue

                if not _can_receive(
                    receiver,
                    transfer,
                    weights,
                    portfolio,
                    config,
                ):
                    continue

                if not _can_give(
                    donor,
                    transfer,
                    weights,
                    config,
                ):
                    continue

                weights[receiver] += transfer
                weights[donor] -= transfer

                moved = True

                if (
                    weights[receiver]
                    >= raw_weights[receiver] - tolerance
                ):
                    break

        if not moved:
            break

    # --------------------------------------------------------------
    # Final redistribution of tiny floating-point differences.
    # --------------------------------------------------------------

    difference = (
        config.target_total_weight
        - weights.sum()
    )

    if abs(difference) > 1e-10:

        if difference > 0:

            for receiver in order:

                if difference <= 1e-10:
                    break

                receiver_capacity = (
                    config.maximum_position_weight
                    - weights[receiver]
                )

                receiver_sector = sectors[receiver]
                receiver_business = (
                    business_types[receiver]
                )

                receiver_capacity = min(
                    receiver_capacity,
                    config.maximum_sector_weight
                    - _group_total(
                        weights,
                        sectors,
                        receiver_sector,
                    ),
                    config.maximum_business_type_weight
                    - _group_total(
                        weights,
                        business_types,
                        receiver_business,
                    ),
                )

                addition = min(
                    difference,
                    max(0.0, receiver_capacity),
                )

                if addition > 0 and _can_receive(
                    receiver,
                    addition,
                    weights,
                    portfolio,
                    config,
                ):
                    weights[receiver] += addition
                    difference -= addition

        else:

            reduction_needed = -difference

            for donor in reversed(order):

                if reduction_needed <= 1e-10:
                    break

                available = (
                    weights[donor]
                    - config.minimum_position_weight
                )

                reduction = min(
                    reduction_needed,
                    max(0.0, available),
                )

                if reduction > 0:
                    weights[donor] -= reduction
                    reduction_needed -= reduction

    # --------------------------------------------------------------
    # Clamp floating-point noise.
    # --------------------------------------------------------------

    weights[np.abs(weights) < 1e-14] = 0.0

    total = weights.sum()

    if not np.isclose(
        total,
        config.target_total_weight,
        atol=1e-9,
    ):
        raise ValueError(
            "Unable to construct a portfolio with the requested "
            "weight constraints."
        )

    weights *= (
        config.target_total_weight
        / total
    )

    return weights


def _validate_weights(
    portfolio: pd.DataFrame,
    config: PortfolioConstructionConfig,
) -> None:
    weights = portfolio["weight"]

    if weights.isna().any():
        raise ValueError(
            "Portfolio contains missing weights."
        )

    if (weights <= 0).any():
        raise ValueError(
            "Portfolio contains non-positive weights."
        )

    if (
        weights
        < config.minimum_position_weight - 1e-9
    ).any():
        raise ValueError(
            "Minimum position-weight constraint violated."
        )

    if (
        weights
        > config.maximum_position_weight + 1e-9
    ).any():
        raise ValueError(
            "Maximum position-weight constraint violated."
        )

    if not np.isclose(
        weights.sum(),
        config.target_total_weight,
        atol=1e-8,
    ):
        raise ValueError(
            "Portfolio weights do not sum to target total."
        )

    sector_weights = (
        portfolio
        .groupby("sector")["weight"]
        .sum()
    )

    if (
        sector_weights
        > config.maximum_sector_weight + 1e-9
    ).any():
        raise ValueError(
            "Maximum sector-weight constraint violated."
        )

    business_weights = (
        portfolio
        .groupby("business_type")["weight"]
        .sum()
    )

    if (
        business_weights
        > config.maximum_business_type_weight + 1e-9
    ).any():
        raise ValueError(
            "Maximum business-type-weight constraint violated."
        )


def construct_portfolio(
    dataset: pd.DataFrame,
    config: PortfolioConstructionConfig | None = None,
) -> pd.DataFrame:
    if config is None:
        config = PortfolioConstructionConfig()

    _validate_input(dataset)

    if config.selection_method != "rank_order":
        raise ValueError(
            "Unsupported selection method: "
            f"{config.selection_method}"
        )

    if config.weighting_method != "score_proportional":
        raise ValueError(
            "Unsupported weighting method: "
            f"{config.weighting_method}"
        )

    selected = _select_holdings(
        dataset,
        config,
    )

    selected["selected"] = True

    selected["weight"] = (
        _allocate_score_proportional_weights(
            selected,
            config,
        )
    )

    selected = selected.sort_values(
        by=["overall_rank", "symbol"],
        ascending=[True, True],
        kind="stable",
    ).reset_index(drop=True)

    _validate_weights(
        selected,
        config,
    )

    return selected


def add_portfolio_columns(
    dataset: pd.DataFrame,
    portfolio: pd.DataFrame,
) -> pd.DataFrame:
    result = dataset.copy()

    result["selected"] = False
    result["weight"] = 0.0

    portfolio_weights = (
        portfolio
        .set_index("symbol")["weight"]
    )

    selected_symbols = set(
        portfolio["symbol"]
    )

    mask = result["symbol"].isin(
        selected_symbols
    )

    result.loc[mask, "selected"] = True

    result.loc[mask, "weight"] = (
        result.loc[mask, "symbol"]
        .map(portfolio_weights)
    )

    return result


def audit_portfolio(
    portfolio: pd.DataFrame,
    config: PortfolioConstructionConfig | None = None,
) -> dict[str, bool]:
    if config is None:
        config = PortfolioConstructionConfig()

    required = {
        "symbol",
        "overall_score",
        "overall_rank",
        "sector",
        "business_type",
        "weight",
    }

    missing = required.difference(
        portfolio.columns
    )

    if missing:
        raise ValueError(
            "Missing portfolio columns: "
            + ", ".join(sorted(missing))
        )

    holding_count = len(portfolio)

    sector_weights = (
        portfolio
        .groupby("sector")["weight"]
        .sum()
    )

    business_weights = (
        portfolio
        .groupby("business_type")["weight"]
        .sum()
    )

    return {
        "holding_count": (
            config.minimum_holdings
            <= holding_count
            <= config.maximum_holdings
        ),
        "target_holdings": (
            holding_count
            == config.target_holdings
        ),
        "unique_symbols": (
            portfolio["symbol"].nunique()
            == holding_count
        ),
        "minimum_weight": (
            portfolio["weight"].min()
            >= config.minimum_position_weight - 1e-9
        ),
        "maximum_weight": (
            portfolio["weight"].max()
            <= config.maximum_position_weight + 1e-9
        ),
        "sector_constraint": (
            sector_weights.max()
            <= config.maximum_sector_weight + 1e-9
        ),
        "business_type_constraint": (
            business_weights.max()
            <= config.maximum_business_type_weight + 1e-9
        ),
        "total_weight": (
            np.isclose(
                portfolio["weight"].sum(),
                config.target_total_weight,
                atol=1e-8,
            )
        ),
    }


def print_portfolio_audit(
    portfolio: pd.DataFrame,
    config: PortfolioConstructionConfig | None = None,
) -> None:
    if config is None:
        config = PortfolioConstructionConfig()

    audit = audit_portfolio(
        portfolio,
        config,
    )

    print("=" * 70)
    print("NIFTY 50 PORTFOLIO CONSTRUCTION AUDIT")
    print("=" * 70)

    print()
    print("PORTFOLIO")
    print(f"Holdings: {len(portfolio)}")
    print(
        f"Total weight: "
        f"{portfolio['weight'].sum():.6f}"
    )

    print()
    print("HOLDINGS")

    display_columns = [
        "symbol",
        "overall_rank",
        "overall_score",
        "sector",
        "business_type",
        "weight",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in portfolio.columns
    ]

    print(
        portfolio[available_columns].to_string(
            index=False
        )
    )

    print()
    print("CONSTRAINT AUDIT")

    for name, passed in audit.items():
        print(
            f"- {name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    print()
    print(
        "PORTFOLIO CONSTRUCTION AUDIT: "
        + (
            "PASS"
            if all(audit.values())
            else "FAIL"
        )
    )

    print("=" * 70)


if __name__ == "__main__":
    print(
        "Portfolio construction module loaded successfully."
    )
