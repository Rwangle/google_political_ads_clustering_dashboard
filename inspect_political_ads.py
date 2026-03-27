import sqlite3
import pandas as pd
import os

# --- CONFIG ---
# Paths are resolved relative to this script's location
# so the script works regardless of where the terminal is run from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "db", "political_ads.db")

con = sqlite3.connect(DB_PATH)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def subsection(title):
    print(f"\n--- {title} ---")

def inspect_column(df, col):
    """Print a standard profile for a single column."""
    if col not in df.columns:
        print(f"  [MISSING] {col} not in dataframe")
        return

    s = df[col]
    print(f"\n  Column: {col}")
    print(f"  dtype:       {s.dtype}")
    print(f"  null count:  {s.isna().sum():,} ({s.isna().mean()*100:.1f}%)")
    print(f"  unique vals: {s.nunique():,}")

    if s.dtype in ["float64", "int64"]:
        print(f"  min:         {s.min():,.2f}")
        print(f"  max:         {s.max():,.2f}")
        print(f"  mean:        {s.mean():,.2f}")
        print(f"  median:      {s.median():,.2f}")
    else:
        print(f"  sample values:")
        for v in s.dropna().unique()[:8]:
            print(f"    - {v}")


# ============================================================
# 1. ADVERTISER STATS
#    Key columns: Spend_USD, Total_Creatives, Advertiser_Name
# ============================================================
section("1. ADVERTISER STATS")
advertiser = pd.read_sql("SELECT * FROM advertiser_stats", con)
print(f"Shape: {advertiser.shape}")

subsection("Dashboard columns")
for col in ["Advertiser_Name", "Spend_USD", "Total_Creatives", "Regions", "Elections"]:
    inspect_column(advertiser, col)

subsection("Clustering columns")
for col in ["Total_Creatives", "Spend_USD"]:
    inspect_column(advertiser, col)


# ============================================================
# 2. CREATIVE STATS
#    Dashboard: Spend_Range, Impressions, Num_of_Days
#    Clustering: Ad_Type, Age/Gender/Geo targeting, Impressions
# ============================================================
section("2. CREATIVE STATS")
creative = pd.read_sql("SELECT * FROM creative_stats", con)
print(f"Shape: {creative.shape}")

subsection("Dashboard columns")
for col in ["Spend_Range_Min_USD", "Spend_Range_Max_USD", "Impressions", "Num_of_Days"]:
    inspect_column(creative, col)

subsection("Clustering columns")
for col in [
    "Ad_Type",
    "Age_Targeting",
    "Gender_Targeting",
    "Geo_Targeting_Included",
    "Geo_Targeting_Excluded",
    "Impressions",
    "Num_of_Days",
    "Spend_Range_Min_USD",
    "Spend_Range_Max_USD",
]:
    inspect_column(creative, col)


# ============================================================
# 3. WEEKLY SPEND
#    Dashboard: Spend_USD over Week_Start_Date
#    Clustering: spending cadence features
# ============================================================
section("3. WEEKLY SPEND")
weekly = pd.read_sql("SELECT * FROM weekly_spend", con)
print(f"Shape: {weekly.shape}")

subsection("Dashboard + Clustering columns")
for col in ["Week_Start_Date", "Spend_USD"]:
    inspect_column(weekly, col)

subsection("Date range check")
print(f"  Earliest week: {weekly['Week_Start_Date'].min()}")
print(f"  Latest week:   {weekly['Week_Start_Date'].max()}")


# ============================================================
# 4. ADVERTISER GEO SPEND
#    Dashboard: spend by state per advertiser
#    Clustering: geographic concentration
# ============================================================
section("4. ADVERTISER GEO SPEND")
geo = pd.read_sql("SELECT * FROM advertiser_geo_spend", con)
print(f"Shape: {geo.shape}")

subsection("Dashboard + Clustering columns")
for col in ["Country", "Country_Subdivision_Primary", "Spend_USD"]:
    inspect_column(geo, col)

subsection("State coverage check")
us = geo[geo["Country"] == "US"]
print(f"  US rows:    {len(us):,}")
print(f"  US states:  {us['Country_Subdivision_Primary'].nunique()}")
print(f"  Sample states: {list(us['Country_Subdivision_Primary'].unique()[:5])}")


# ============================================================
# 5. QUICK CROSS-TABLE CHECKS
# ============================================================
section("5. CROSS-TABLE CHECKS")

subsection("Advertiser ID coverage")
ids_advertiser = set(advertiser["Advertiser_ID"].unique())
ids_creative   = set(creative["Advertiser_ID"].unique())
ids_weekly     = set(weekly["Advertiser_ID"].unique())
ids_geo        = set(geo["Advertiser_ID"].unique())

print(f"  Advertisers in advertiser_stats: {len(ids_advertiser):,}")
print(f"  Advertisers in creative_stats:   {len(ids_creative):,}")
print(f"  Advertisers in weekly_spend:     {len(ids_weekly):,}")
print(f"  Advertisers in geo_spend:        {len(ids_geo):,}")
print(f"  In advertiser_stats but not creative: {len(ids_advertiser - ids_creative):,}")
print(f"  In creative but not advertiser_stats: {len(ids_creative - ids_advertiser):,}")

subsection("Impressions bucket distribution (creative_stats)")
print(creative["Impressions"].value_counts().to_string())

subsection("Ad_Type distribution (creative_stats)")
print(creative["Ad_Type"].value_counts().to_string())

con.close()
print("\n=== Inspection complete ===")
