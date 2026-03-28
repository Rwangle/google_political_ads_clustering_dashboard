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

# ============================================================
# 6. CALIFORNIA FOCUS
#    Base population: advertisers who deliberately targeted CA
# ============================================================
con = sqlite3.connect(DB_PATH)

section("6. CALIFORNIA — DELIBERATELY TARGETED ADVERTISERS")

# Get advertiser IDs who explicitly targeted California
ca_advertiser_ids = pd.read_sql("""
    SELECT DISTINCT Advertiser_ID
    FROM creative_stats
    WHERE Geo_Targeting_Included LIKE '%California%'
""", con)["Advertiser_ID"].tolist()

print(f"  Advertisers who explicitly targeted CA: {len(ca_advertiser_ids):,}")
print(f"  Total ads targeting CA:                 52,071")

# --- Top advertisers by total spend ---
subsection("Top 20 CA-targeting advertisers by total spend (USD)")
top_spenders = pd.read_sql(f"""
    SELECT 
        a.Advertiser_Name,
        a.Spend_USD            AS Total_Spend_USD,
        a.Total_Creatives,
        COUNT(c.Ad_ID)         AS CA_Ads,
        SUM(c.Spend_Range_Min_USD + c.Spend_Range_Max_USD) / 2 AS Est_CA_Spend_USD
    FROM advertiser_stats a
    JOIN creative_stats c ON a.Advertiser_ID = c.Advertiser_ID
    WHERE a.Advertiser_ID IN ({','.join(['?' for _ in ca_advertiser_ids])})
      AND c.Geo_Targeting_Included LIKE '%California%'
    GROUP BY a.Advertiser_ID, a.Advertiser_Name, a.Spend_USD, a.Total_Creatives
    ORDER BY a.Spend_USD DESC
    LIMIT 20
""", con, params=ca_advertiser_ids)

print(top_spenders.to_string(index=False))

# --- Spend distribution among CA advertisers ---
subsection("Spend distribution among CA-targeting advertisers")
spend_dist = pd.read_sql(f"""
    SELECT 
        CASE 
            WHEN Spend_USD = 0          THEN '0'
            WHEN Spend_USD < 10000      THEN '1 - <10k'
            WHEN Spend_USD < 100000     THEN '10k - 100k'
            WHEN Spend_USD < 1000000    THEN '100k - 1M'
            WHEN Spend_USD < 10000000   THEN '1M - 10M'
            ELSE '> 10M'
        END AS Spend_Bucket,
        COUNT(*) AS Num_Advertisers
    FROM advertiser_stats
    WHERE Advertiser_ID IN ({','.join(['?' for _ in ca_advertiser_ids])})
    GROUP BY Spend_Bucket
    ORDER BY MIN(Spend_USD)
""", con, params=ca_advertiser_ids)

print(spend_dist.to_string(index=False))

# --- Ad type breakdown for CA ads ---
subsection("Ad type breakdown for CA-targeted ads")
ad_types = pd.read_sql("""
    SELECT Ad_Type, COUNT(*) as Count
    FROM creative_stats
    WHERE Geo_Targeting_Included LIKE '%California%'
    GROUP BY Ad_Type
    ORDER BY Count DESC
""", con)
print(ad_types.to_string(index=False))

con.close()
print("\n=== California inspection complete ===")
