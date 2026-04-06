import sqlite3
import pandas as pd
import os
from datetime import datetime

# roger wrote most of this, i just added column filtering and cleaned it up - rohit

scrpt_DIR = os.path.dirname(  os.path.abspath(
    __file__))
DATA_DIRR = os.path.join(scrpt_DIR,  "data","raw")
db_pathh  = os.path.join(scrpt_DIR,"db",
                          "political_ads.db")

fileMap = {
    "advertiser_stats":     "google-political-ads-advertiser-stats.csv",
    "creative_stats":       "google-political-ads-creative-stats.csv",
    "weekly_spend":         "google-political-ads-advertiser-weekly-spend.csv",
    "geo_spend":            "google-political-ads-geo-spend.csv",
    "advertiser_geo_spend": "google-political-ads-advertiser-geo-spend.csv",
    "last_updated":         "google-political-ads-updated.csv",
}

# theres a ton of currency cols we dont need (EUR INR etc)
keepCols = {
    "advertiser_stats": [
        "Advertiser_ID","Advertiser_Name","Regions",
        "Elections","Total_Creatives","Spend_USD",
    ],
    "creative_stats": [
        "Ad_ID","Ad_Type","Regions","Advertiser_ID",
        "Advertiser_Name","Date_Range_Start","Date_Range_End",
        "Num_of_Days","Impressions","Age_Targeting",
        "Gender_Targeting","Geo_Targeting_Included",
        "Geo_Targeting_Excluded","Spend_Range_Min_USD",
        "Spend_Range_Max_USD",
    ],
    "weekly_spend": [
        "Advertiser_ID","Advertiser_Name",
        "Week_Start_Date","Spend_USD",],
    "geo_spend": [
        "Country","Country_Subdivision_Primary",
        "Country_Subdivision_Secondary","Spend_USD",],
    "advertiser_geo_spend": [
        "Advertiser_ID","Advertiser_Name",
        "Country","Country_Subdivision_Primary",  "Spend_USD",],
    "last_updated": [
        "Report_Data_Updated_Time (PT)",],
}


def setup_db(    pth):
    os.makedirs(os.path.dirname(pth),exist_ok=True )
    conn = sqlite3.connect(pth  )
    print("Connected to database: " + pth)
    return(conn)


def load_table(conn,  tbl,filepath, cols):
    if not(os.path.exists(filepath)):
        print(f"  [SKIP] File not found: {filepath}")
        return

    print("  Loading %s..." % tbl,end=" ")
    df = pd.read_csv(filepath,low_memory =False)

    actual = [c for c in cols if (c in df.columns)  ]
    dropped= [c for c in cols if (c not in df.columns)]
    if(  dropped):
        print(f"\n  [WARN] Missing columns in {tbl}: {dropped}")
    df = df[  actual]

    # sqlite doesnt like spaces and parens
    df.columns = [c.replace(" ","_").replace("(","").replace(")","")
                  for c in df.columns]

    df.to_sql(tbl,conn,if_exists= "replace",index=False)
    print("done. ({:,} rows, {} columns)".format(len(df),len(actual)))


def log_snapshot(conn  ):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS snapshot_log (
            snapshot_taken_at TEXT,
            report_data_updated_date TEXT
        )
    """)
    try:
        dt = pd.read_sql(
            "SELECT Report_Data_Updated_Time_PT FROM last_updated LIMIT 1",conn
        ).iloc[0,0]
    except Exception:
        dt = "unknown"

    conn.execute("INSERT INTO snapshot_log VALUES (?, ?)",
                 (datetime.now(  ).isoformat(),dt))
    conn.commit()
    print(f"\nSnapshot logged at {datetime.now().isoformat()}")
    print(f"Report data updated date: {dt}")


def main():
    print("=== Google Political Ads — SQLite Loader ===\n")
    conn = setup_db(db_pathh)

    print("Loading tables:")
    for tblName,fname in fileMap.items():
        fpath = os.path.join(DATA_DIRR,fname)
        load_table(conn,tblName,fpath,keepCols[  tblName])

    log_snapshot(conn)

    print("\n--- Table row counts ---")
    for t in keepCols.keys():
        try:
            n = pd.read_sql(f"SELECT COUNT(*) as n FROM {t}",conn).iloc[0,0]
            print(f"  {t}: {n:,} rows")
        except Exception as e:
            print(f"  {t}: error — {e}")

    conn.close()
    print("\nDone. Database saved to:",db_pathh)


if __name__=="__main__":
    main()
