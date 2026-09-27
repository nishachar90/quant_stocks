from pathlib import Path

import pandas as pd


DATA_FILE = Path("data/analysis/nifty50_analysis.csv")


def main():
    print("=" * 60)
    print("NIFTY 50 DATA QUALITY AUDIT")
    print("=" * 60)

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    print("\nLOADING DATASET")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    print("File:", DATA_FILE)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    # --------------------------------------------------
    # Columns
    # --------------------------------------------------

    print("\nCOLUMNS")

    for column in df.columns:
        print("-", column)

    # --------------------------------------------------
    # Duplicate stocks
    # --------------------------------------------------

    print("\nDUPLICATE STOCK CHECK")

    duplicate_symbols = df[
        df["symbol"].duplicated(keep=False)
    ]

    if duplicate_symbols.empty:
        print("No duplicate stocks found.")
    else:
        print("Duplicate stocks found:")
        print(duplicate_symbols["symbol"].tolist())

    # --------------------------------------------------
    # Missing values
    # --------------------------------------------------

    print("\nMISSING VALUES")

    missing = df.isna().sum()

    for column, count in missing.items():
        if count > 0:
            print(
                f"{column}: {count} missing"
            )

    if missing.sum() == 0:
        print("No missing values found.")

    # --------------------------------------------------
    # Data types
    # --------------------------------------------------

    print("\nDATA TYPES")

    print(df.dtypes)

    # --------------------------------------------------
    # Numeric summary
    # --------------------------------------------------

    print("\nNUMERIC SUMMARY")

    numeric_df = df.select_dtypes(
        include="number"
    )

    print(
        numeric_df.describe().T
    )

    # --------------------------------------------------
    # Infinite values
    # --------------------------------------------------

    print("\nINFINITE VALUE CHECK")

    infinite_counts = (
        numeric_df
        .isin([float("inf"), float("-inf")])
        .sum()
    )

    infinite_values = infinite_counts[
        infinite_counts > 0
    ]

    if infinite_values.empty:
        print("No infinite values found.")
    else:
        print(infinite_values)

    # --------------------------------------------------
    # Return metrics
    # --------------------------------------------------

    print("\nRETURN METRICS")

    print(
        "CAGR minimum:",
        df["cagr"].min()
    )

    print(
        "CAGR maximum:",
        df["cagr"].max()
    )

    print(
        "Total return minimum:",
        df["total_return"].min()
    )

    print(
        "Total return maximum:",
        df["total_return"].max()
    )

    # --------------------------------------------------
    # Risk metrics
    # --------------------------------------------------

    print("\nRISK METRICS")

    print(
        "Volatility minimum:",
        df["annualized_volatility"].min()
    )

    print(
        "Volatility maximum:",
        df["annualized_volatility"].max()
    )

    print(
        "Maximum drawdown minimum:",
        df["maximum_drawdown"].min()
    )

    print(
        "Maximum drawdown maximum:",
        df["maximum_drawdown"].max()
    )

    # --------------------------------------------------
    # Fundamental metrics
    # --------------------------------------------------

    print("\nFUNDAMENTAL METRICS")

    fundamental_columns = [
        "net_profit_margin",
        "operating_margin",
        "roe",
        "debt_to_equity",
        "free_cash_flow",
        "revenue_growth",
    ]

    print(
        df[fundamental_columns].describe().T
    )

    # --------------------------------------------------
    # Final result
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("DATA QUALITY AUDIT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
