import pandas as pd

CLASSIFICATION_FILE = "data/reference/nifty50_classification.csv"
APPLICABILITY_FILE = "data/reference/metric_applicability.csv"

EXPECTED_METRICS = {
    "roe",
    "revenue_growth",
    "net_profit_margin",
    "operating_margin",
    "debt_to_equity",
    "free_cash_flow",
}

VALID_TREATMENTS = {
    "primary",
    "secondary",
    "special",
    "exclude",
}


print("=" * 60)
print("METRIC APPLICABILITY AUDIT")
print("=" * 60)


# ---------------------------------------------------------
# LOAD FILES
# ---------------------------------------------------------

classification = pd.read_csv(CLASSIFICATION_FILE)
applicability = pd.read_csv(APPLICABILITY_FILE)


# ---------------------------------------------------------
# BASIC INFORMATION
# ---------------------------------------------------------

print("\nCLASSIFICATION")
print(f"Rows: {len(classification)}")
print(f"Columns: {len(classification.columns)}")

print("\nAPPLICABILITY")
print(f"Rows: {len(applicability)}")
print(f"Columns: {len(applicability.columns)}")


# ---------------------------------------------------------
# CHECK COLUMNS
# ---------------------------------------------------------

expected_columns = {
    "business_type",
    "metric",
    "treatment",
}

print("\nCOLUMNS")
print(list(applicability.columns))

assert set(applicability.columns) == expected_columns


# ---------------------------------------------------------
# CHECK DUPLICATES
# ---------------------------------------------------------

duplicates = applicability.duplicated(
    subset=["business_type", "metric"]
).sum()

print(f"\nDuplicate business_type + metric combinations: {duplicates}")

assert duplicates == 0


# ---------------------------------------------------------
# CHECK EXPECTED METRICS
# ---------------------------------------------------------

actual_metrics = set(applicability["metric"])

print("\nMETRICS FOUND")
print(sorted(actual_metrics))

missing_metrics = EXPECTED_METRICS - actual_metrics
extra_metrics = actual_metrics - EXPECTED_METRICS

print(f"Missing metrics: {missing_metrics}")
print(f"Unexpected metrics: {extra_metrics}")

assert not missing_metrics
assert not extra_metrics


# ---------------------------------------------------------
# CHECK TREATMENTS
# ---------------------------------------------------------

actual_treatments = set(applicability["treatment"])

print("\nTREATMENTS FOUND")
print(sorted(actual_treatments))

invalid_treatments = actual_treatments - VALID_TREATMENTS

print(f"Invalid treatments: {invalid_treatments}")

assert not invalid_treatments


# ---------------------------------------------------------
# CHECK BUSINESS TYPES
# ---------------------------------------------------------

classification_types = set(classification["business_type"])
applicability_types = set(applicability["business_type"])

missing_business_types = classification_types - applicability_types
extra_business_types = applicability_types - classification_types

print("\nBUSINESS TYPE CHECK")
print(f"Business types in classification: {len(classification_types)}")
print(f"Business types in applicability: {len(applicability_types)}")

print(f"Missing business types: {missing_business_types}")
print(f"Unexpected business types: {extra_business_types}")

assert not missing_business_types
assert not extra_business_types


# ---------------------------------------------------------
# CHECK SIX METRICS FOR EVERY BUSINESS TYPE
# ---------------------------------------------------------

print("\nMETRIC COMPLETENESS")

for business_type in sorted(classification_types):

    rows = applicability[
        applicability["business_type"] == business_type
    ]

    metrics = set(rows["metric"])

    missing = EXPECTED_METRICS - metrics
    extra = metrics - EXPECTED_METRICS

    print(
        f"{business_type}: "
        f"{len(metrics)} metrics"
    )

    assert not missing
    assert not extra


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("METRIC APPLICABILITY AUDIT COMPLETE")
print("=" * 60)
