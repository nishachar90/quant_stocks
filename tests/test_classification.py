from pathlib import Path

import pandas as pd


CLASSIFICATION_FILE = Path("data/reference/nifty50_classification.csv")


def main():
    print("=" * 60)
    print("NIFTY 50 CLASSIFICATION AUDIT")
    print("=" * 60)

    df = pd.read_csv(CLASSIFICATION_FILE)

    print("\nDATASET")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nCOLUMNS")
    for column in df.columns:
        print(f"- {column}")

    print("\nDUPLICATE SYMBOLS")
    duplicates = df["symbol"].duplicated().sum()
    print(f"Duplicate symbols: {duplicates}")

    print("\nMISSING VALUES")
    print(df.isna().sum())

    print("\nSECTOR DISTRIBUTION")
    print(df["sector"].value_counts().to_string())

    print("\nBUSINESS TYPE DISTRIBUTION")
    print(df["business_type"].value_counts().to_string())

    print("\nCLASSIFICATION AUDIT COMPLETE")


if __name__ == "__main__":
    main()
