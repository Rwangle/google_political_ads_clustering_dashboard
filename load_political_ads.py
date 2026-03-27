import sqlite3
import pandas as pd
import os
from datetime import datetime

# --- CONFIG ---
# Paths are resolved relative to this script's location
# so the script works regardless of where the terminal is run from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "raw")
DB_PATH  = os.path.join(SCRIPT_DIR, "db", "political_ads.db")

# CSV filenames
FILES = {
    "advertiser_stats":    "google-political-ads-advertiser-stats.csv",
    "creative_stats":      "google-political-ads-creative-stats.csv",
    "weekly_spend":        "google-political-ads-advertiser-weekly-spend.csv",
    "geo_spend":           "google-political-ads-geo-spend.csv",
    "advertiser_geo_spend":"google-political-ads-advertiser-geo-spend.csv",
    "last_updated":        "google-political-ads-updated.csv",
}

# Columns to keep per table (dropping deprecated + non-USD currency columns)
COLUMNS = {
    "advertiser_stats": [
        "Advertiser_ID",
        "Advertiser_Name",
        "Regions",
        "Elections",
        "Total_Creatives",
        "Spend_USD",
    ],
    "creative_stats": [
        "Ad_ID",
        "Ad_Type",
        "Regions",
        "Advertiser_ID",
        "Advertiser_Name",
        "Date_Range_Start",
        "Date_Range_End",
        "Num_of_Days",
        "Impressions",
        "Age_Targeting",
        "Gender_Targeting",
        "Geo_Targeting_Included",
        "Geo_Targeting_Excluded",
        "Spend_Range_Min_USD",
        "Spend_Range_Max_USD",
    ],
    "weekly_spend": [
        "Advertiser_ID",
        "Advertiser_Name",
        "Week_Start_Date",
        "Spend_USD",
    ],
    "geo_spend": [
        "Country",
        "Country_Subdivision_Primary",
        "Country_Subdivision_Secondary",
        "Spend_USD",
    ],
    "advertiser_geo_spend": [
        "Advertiser_ID",
        "Advertiser_Name",
        "Country",
        "Country_Subdivision_Primary",
        "Spend_USD",
    ],
    "last_updated": [
        "Report_Data_Updated_Time (PT)",
    ],
}


def setup_db(db_path):
    """Create db directory if it doesn't exist and return connection."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = sqlite3.connect(db_path)
    print(f"Connected to database: {db_path}")
    return con


def load_table(con, table_name, file_path, columns):
    """Load a single CSV into SQLite, keeping only relevant columns."""
    if not os.path.exists(file_path):
        print(f"  [SKIP] File not found: {file_path}")
        return

    print(f"  Loading {table_name}...", end=" ")

    df = pd.read_csv(file_path, low_memory=False)

    # Keep only columns that actually exist in this CSV
    cols_to_keep = [c for c in columns if c in df.columns]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        print(f"\n  [WARN] Missing columns in {table_name}: {missing}")

    df = df[cols_to_keep]

    # Sanitize column names for SQLite (remove spaces and special characters)
    df.columns = [c.replace(" ", "_").replace("(", "").replace(")", "") for c in df.columns]

    # Write to SQLite — replace table if it already exists
    df.to_sql(table_name, con, if_exists="replace", index=False)

    print(f"done. ({len(df):,} rows, {len(cols_to_keep)} columns)")


def log_snapshot(con):
    """Record when this snapshot was taken."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS snapshot_log (
            snapshot_taken_at TEXT,
            report_data_updated_date TEXT
        )
    """)

    # Try to read the report's own updated date
    try:
        updated_date = pd.read_sql(
            "SELECT Report_Data_Updated_Time_PT FROM last_updated LIMIT 1", con
        ).iloc[0, 0]
    except Exception:
        updated_date = "unknown"

    con.execute(
        "INSERT INTO snapshot_log VALUES (?, ?)",
        (datetime.now().isoformat(), updated_date)
    )
    con.commit()
    print(f"\nSnapshot logged at {datetime.now().isoformat()}")
    print(f"Report data updated date: {updated_date}")


def main():
    print("=== Google Political Ads — SQLite Loader ===\n")

    con = setup_db(DB_PATH)

    print("Loading tables:")
    for table_name, filename in FILES.items():
        file_path = os.path.join(DATA_DIR, filename)
        load_table(con, table_name, file_path, COLUMNS[table_name])

    log_snapshot(con)

    # Quick sanity check
    print("\n--- Table row counts ---")
    for table in COLUMNS.keys():
        try:
            count = pd.read_sql(f"SELECT COUNT(*) as n FROM {table}", con).iloc[0, 0]
            print(f"  {table}: {count:,} rows")
        except Exception as e:
            print(f"  {table}: error — {e}")

    con.close()
    print("\nDone. Database saved to:", DB_PATH)


if __name__ == "__main__":
    main()
