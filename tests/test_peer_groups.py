import pandas as pd

CLASSIFICATION_FILE = "data/reference/nifty50_classification.csv"
PEER_GROUP_FILE = "data/reference/peer_groups.csv"


print("=" * 60)
print("PEER GROUP AUDIT")
print("=" * 60)


# ---------------------------------------------------------
# LOAD FILES
# ---------------------------------------------------------

classification = pd.read_csv(CLASSIFICATION_FILE)
peer_groups = pd.read_csv(PEER_GROUP_FILE)


print("\nCLASSIFICATION")
print(f"Rows: {len(classification)}")

print("\nPEER GROUPS")
print(f"Rows: {len(peer_groups)}")


# ---------------------------------------------------------
# CHECK COLUMNS
# ---------------------------------------------------------

expected_columns = {
    "business_type",
    "peer_group",
}

print("\nCOLUMNS")
print(list(peer_groups.columns))

assert set(peer_groups.columns) == expected_columns


# ---------------------------------------------------------
# CHECK DUPLICATES
# ---------------------------------------------------------

duplicates = peer_groups.duplicated(
    subset=["business_type"]
).sum()

print(f"\nDuplicate business types: {duplicates}")

assert duplicates == 0


# ---------------------------------------------------------
# CHECK MISSING VALUES
# ---------------------------------------------------------

print("\nMISSING VALUES")
print(peer_groups.isna().sum())

assert peer_groups["business_type"].notna().all()
assert peer_groups["peer_group"].notna().all()


# ---------------------------------------------------------
# CHECK BUSINESS TYPES
# ---------------------------------------------------------

classification_types = set(
    classification["business_type"]
)

peer_group_types = set(
    peer_groups["business_type"]
)

missing = classification_types - peer_group_types
extra = peer_group_types - classification_types

print("\nBUSINESS TYPE CHECK")
print(f"Classification business types: {len(classification_types)}")
print(f"Peer-group business types: {len(peer_group_types)}")

print(f"Missing: {missing}")
print(f"Unexpected: {extra}")

assert not missing
assert not extra


# ---------------------------------------------------------
# PEER GROUP DISTRIBUTION
# ---------------------------------------------------------

print("\nPEER GROUP DISTRIBUTION")

distribution = (
    peer_groups["peer_group"]
    .value_counts()
    .sort_index()
)

print(distribution)


# ---------------------------------------------------------
# ACTUAL STOCK COUNT BY PEER GROUP
# ---------------------------------------------------------

merged = classification.merge(
    peer_groups,
    on="business_type",
    how="left",
)

stock_distribution = (
    merged["peer_group"]
    .value_counts()
    .sort_values(ascending=False)
)

print("\nACTUAL NIFTY 50 STOCKS PER PEER GROUP")

print(stock_distribution)


# ---------------------------------------------------------
# CHECK THAT EVERY STOCK HAS A PEER GROUP
# ---------------------------------------------------------

missing_stock_peer_groups = merged["peer_group"].isna().sum()

print(
    f"\nStocks without a peer group: "
    f"{missing_stock_peer_groups}"
)

assert missing_stock_peer_groups == 0


# ---------------------------------------------------------
# CHECK STOCK COUNT
# ---------------------------------------------------------

print(f"\nTotal classified stocks: {len(classification)}")
print(f"Total mapped stocks: {len(merged)}")

assert len(classification) == len(merged)


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("PEER GROUP AUDIT COMPLETE")
print("=" * 60)
