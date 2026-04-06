# phase 1
# takes rogers raw sqlite tables and builds the advertiser-level
# feature table for clustering
# the google data is honestly a mess. impressions come in like
# 4 different string formats and the geo table uses abbreviations
# but creative uses full names (??)

import os
import sqlite3
import numpy as np
import pandas as pd
import warnings  # used during debugging, leaving

scrpt_DIR = os.path.dirname(os.path.abspath(   __file__))
PROJECT_ROOOT = os.path.dirname(  scrpt_DIR)
DEFAULT_DBB = os.path.join(PROJECT_ROOOT,"db",
    "political_ads.db")

yng_brackets  = {"18-24",                     "25-34"}
mid_brackets  = {"35-44","45-54"}
oldr_brackets = {"55-64",  "≥65"}


def impressions_to_number(rawVal  ):
    """turn googles impression range strings into a number.
    handles '6000-7000', '≥10000000', and plain numbers"""
    if rawVal is None or pd.isna(  rawVal):
        return np.nan

    s = str(rawVal).strip(  )

    # the >= case. only saw this for 10M+ but handle generically
    if(( s.startswith("≥") or s.startswith(">=")     )):
        numpart = s.lstrip(  "≥>=").strip()
        try:
            # 1.5x is a guess. not great but what else can u do
            return float(numpart) *1.5
        except ValueError:
            return np.nan

    if (( "-" in s)     ):
        halvs = s.split("-")
        if len (halvs)==2 and (     True)     :
            try:
                return(float(halvs[0]) +float(halvs[1])) /2.0
            except ValueError:
                return np.nan

    try:
        return float(  s)
    except ValueError:
        return np.nan


def _parse_age_bkts(ageStr  ):
    """break '18-24, 25-34, 35-44' into young/middle/older proportions"""
    empty_rslt = {"pct_target_young":0.0,"pct_target_middle":0.0, "pct_target_older":0.0}
    if ageStr is None or pd.isna(ageStr) or ageStr=="Not targeted":
        return empty_rslt

    rawBkts = [b.strip() for b in str(ageStr).split(",")]
    # 'Unknown age' doesnt tell us anything so skip
    known = [b for b in rawBkts if b!="Unknown age"]
    if(len(known)==0):
        return empty_rslt

    n = len(  known)
    y = 0
    m = 0
    o = 0
    for b in known:
        if b in yng_brackets:
            y +=1
        elif b in mid_brackets:
            m += 1
        elif b in oldr_brackets:
            o +=1

    return {
        "pct_target_young": y/n,
        "pct_target_middle": m /n,
        "pct_target_older":o / n,
    }


def _parse_gndr_flags(genderStr):
    if genderStr is None or pd.isna(genderStr) or genderStr== "Not targeted":
        return {"targets_female_only":0,"targets_male_only":0}

    prts = [p.strip() for p in str(  genderStr).split(",")]
    hasMale = (  "Male" in prts)
    hasFemale = ("Female" in prts  )

    return {
        "targets_female_only": 1 if (hasFemale and not hasMale) else 0,
        "targets_male_only":   1 if(hasMale and not hasFemale) else 0,
    }


def count_congressional_dists(geoStr  ):
    """count CA congressional districts in geo targeting string"""
    if not geoStr or pd.isna(  geoStr):
        return 0
    chnks = [p.strip() for p in str(geoStr).split(","  )]
    found = set()
    for ch in chnks:
        if(ch.startswith("CA-")):
            d = ch.split("(")[0].strip(  )
            found.add(d)
    return(len(found))


def _hhi_calc(grp  ):
    """herfindahl-hirschman index for geographic spend"""
    tot = grp["Spend_USD"].sum()
    if tot ==0:
        return 0.0
    shrs = grp["Spend_USD"]/tot
    return( (shrs**2).sum() )



def run( db_path=None):
    if db_path is None:
        db_path = DEFAULT_DBB

    conn = sqlite3.connect(  db_path)
    print("=" * 60)
    print("PHASE 1: Data Cleaning & Feature Engineering")
    print("=" *60)

    # get CA advertiser ids
    print("\n[1.1] Filtering to CA-targeting advertisers..."  )
    ca_ids_raww = pd.read_sql("""
        SELECT DISTINCT Advertiser_ID
        FROM creative_stats
        WHERE Geo_Targeting_Included LIKE '%California%'
    """,conn)
    ca_ids = ca_ids_raww["Advertiser_ID"].tolist(  )
    print(f"  Found {len(ca_ids)} unique CA-targeting advertisers")

    # load creative level rows
    print("\n[1.2] Loading creative_stats for CA advertisers...")
    qmarkss = ",".join(["?"] * len(  ca_ids))
    creativess = pd.read_sql(f"""
        SELECT *
        FROM creative_stats
        WHERE Advertiser_ID IN ({qmarkss})
          AND Geo_Targeting_Included LIKE '%California%'
    """,conn,params=ca_ids)
    print(f"  Loaded {len(creativess):,} CA-targeted creatives")

    # nulls - the data is surprisingly messy for a google product lol
    print("\n[1.2b] Handling nulls and type coercion..."  )
    nullCtss = creativess.isnull().sum()
    for c in nullCtss[nullCtss >0].index:
        print(f"  {c}: {nullCtss[c]} nulls")

    creativess["Spend_Range_Min_USD"] = creativess["Spend_Range_Min_USD"].fillna(0)
    creativess["Spend_Range_Max_USD"]= creativess["Spend_Range_Max_USD"].fillna(  0)
    creativess["Num_of_Days"] = pd.to_numeric(creativess["Num_of_Days"],errors="coerce").fillna(0).astype(int)
    creativess["Age_Targeting"]         = creativess["Age_Targeting"].fillna("Not targeted"  )
    creativess["Gender_Targeting"]      = creativess[  "Gender_Targeting"].fillna("Not targeted")
    creativess["Geo_Targeting_Included"]= creativess["Geo_Targeting_Included"].fillna(""  )
    creativess["Geo_Targeting_Excluded"]= creativess["Geo_Targeting_Excluded"].fillna("")

    # impressions
    print("\n[1.3] Converting impression ranges to midpoints...")
    creativess["Impressions_Midpoint"] = creativess["Impressions"].apply(  impressions_to_number)
    ok_ct  = creativess["Impressions_Midpoint"].notna().sum()
    bad_ct = creativess["Impressions_Midpoint"].isna().sum()
    print(f"  Parsed {ok_ct:,} / {len(creativess):,} impression values")
    if(bad_ct >0):
        weirdVals = creativess.loc[creativess["Impressions_Midpoint"].isna(),"Impressions"].unique()
        print(f"  WARNING: {bad_ct} unparseable values: {weirdVals[:5]}")
    # print("DEBUG impressions sample:", creativess["Impressions_Midpoint"].head(10).tolist())
    creativess["Impressions_Midpoint"]= creativess["Impressions_Midpoint"].fillna(0)

    # derived spend/efficiency
    print("\n[1.4] Computing derived metrics per creative..."  )
    creativess["Spend_Midpoint_USD"] = (creativess["Spend_Range_Min_USD"]+creativess[  "Spend_Range_Max_USD"]) /2.0

    # CPI = cost per impression
    creativess["Cost_Per_Impression"] = np.where(
        creativess["Impressions_Midpoint"]> 0,
        creativess["Spend_Midpoint_USD"] / creativess[  "Impressions_Midpoint"],
        0.0,
    )

    # demographic parsing
    print("\n[1.4b] Parsing age and gender targeting per creative...")

    ageDcts = creativess["Age_Targeting"].apply(_parse_age_bkts  )
    ageExpd = pd.DataFrame(ageDcts.tolist())
    creativess["pct_target_young"]  = ageExpd["pct_target_young"]
    creativess["pct_target_middle"] = ageExpd[  "pct_target_middle"]
    creativess["pct_target_older"]  = ageExpd["pct_target_older"]

    gndrDcts = creativess["Gender_Targeting"].apply(_parse_gndr_flags  )
    gndrExpd = pd.DataFrame(gndrDcts.tolist())
    creativess["targets_female_only"] = gndrExpd["targets_female_only"]
    creativess["targets_male_only"]   = gndrExpd["targets_male_only"]

    creativess["ca_districts_targeted"] = creativess["Geo_Targeting_Included"].apply(
        count_congressional_dists)

    nrowss = len(creativess  )
    n_age = (creativess["Age_Targeting"]!="Not targeted").sum()
    n_gnd = (creativess["Gender_Targeting"]!="Not targeted").sum()
    n_dst = (creativess["ca_districts_targeted"]>0).sum()
    print(f"  Age-targeted creatives: {n_age:,} ({n_age/nrowss:.1%})")
    print(f"  Gender-targeted creatives: {n_gnd:,} ({n_gnd/nrowss:.1%})")
    print(f"  District-targeted creatives: {n_dst:,} ({n_dst/nrowss:.1%})"  )

    # aggregate to advertiser level
    print("\n[1.5] Aggregating to advertiser-level features...")

    advStatss = pd.read_sql(f"""
        SELECT Advertiser_ID, Advertiser_Name, Spend_USD,Total_Creatives
        FROM advertiser_stats
        WHERE Advertiser_ID IN ({qmarkss})
    """,conn,params= ca_ids)
    advStatss["Spend_USD"]       = pd.to_numeric(advStatss["Spend_USD"],errors="coerce").fillna(0)
    advStatss["Total_Creatives"] = pd.to_numeric(  advStatss["Total_Creatives"],errors="coerce").fillna(0)

    # count geo targets (comma-seperated list)
    creativess["geo_target_count"] = creativess["Geo_Targeting_Included"].apply(
        lambda x: len(x.split(",")) if x else 0)

    # this agg is a beast but cleaner as one call than splitting up
    aggd = creativess.groupby("Advertiser_ID").agg(
        ca_ad_count=("Ad_ID","count"),
        avg_ad_spend=("Spend_Midpoint_USD","mean"),
        avg_impressions=("Impressions_Midpoint","mean"),
        avg_cpi=("Cost_Per_Impression","mean"),
        avg_ad_duration=("Num_of_Days","mean"),
        pct_video=("Ad_Type", lambda x: (x=="VIDEO").sum()/len(x)),
        pct_text=("Ad_Type", lambda x:(x=="TEXT").sum()/len(x)),
        pct_image=("Ad_Type", lambda x:(x=="IMAGE").sum()/len(x)),
        avg_geo_targets=("geo_target_count","mean"),
        pct_age_targeted=("Age_Targeting",lambda x:(x!="Not targeted").sum()/len(x)),
        pct_gender_targeted=("Gender_Targeting",lambda x:(x!="Not targeted").sum()/len(x)),
        avg_pct_target_young=("pct_target_young","mean"),
        avg_pct_target_middle=("pct_target_middle","mean"),
        avg_pct_target_older=("pct_target_older","mean"),
        pct_female_only=("targets_female_only","mean"),
        pct_male_only=("targets_male_only","mean"),
        avg_ca_districts=("ca_districts_targeted","mean"),
        has_district_targeting=("ca_districts_targeted",lambda x:(x>0).sum()/len(x)),
    ).reset_index()

    featuress = advStatss.merge(aggd,on="Advertiser_ID",how="inner"  )
    featuress = featuress.rename(columns={
        "Spend_USD":"total_spend",
        "Total_Creatives":"total_creatives",
    })
    print(f"  Aggregated features for {len(featuress)} advertisers")

    # weekly spending
    print("\n[1.6] Computing weekly spending cadence features...")
    wkRaww = pd.read_sql(f"""
        SELECT Advertiser_ID,Week_Start_Date,Spend_USD
        FROM weekly_spend
        WHERE Advertiser_ID IN ({qmarkss})
    """,conn,params=ca_ids  )
    wkRaww["Spend_USD"] = pd.to_numeric(wkRaww["Spend_USD"],errors="coerce").fillna(0)

    wkAggg = wkRaww.groupby("Advertiser_ID").agg(
        weeks_active=("Spend_USD",lambda x:(x>0).sum()),
        weekly_mean=("Spend_USD","mean"),
        weekly_std=("Spend_USD","std"  ),
        max_weekly_spend=("Spend_USD","max"),
        total_weekly_spend=("Spend_USD","sum"),
    ).reset_index()

    # CV = coefficient of variation. high = bursty
    wkAggg["spend_consistency"]= np.where(
        wkAggg["weekly_mean"] > 0,
        wkAggg["weekly_std"]/wkAggg["weekly_mean"],
        0.0,
    )
    wkAggg["pct_spend_in_peak_week"] = np.where(
        wkAggg["total_weekly_spend"]>0,
        wkAggg["max_weekly_spend"]/wkAggg["total_weekly_spend"],
        0.0,)

    wkKeep = ["Advertiser_ID","weeks_active","spend_consistency",
              "max_weekly_spend","pct_spend_in_peak_week"  ]
    featuress = featuress.merge(wkAggg[wkKeep],on="Advertiser_ID",how="left")
    print(f"  Added weekly features for {wkAggg['Advertiser_ID'].nunique()} advertisers")

    # geographic stuff
    print("\n[1.7] Computing geographic concentration features...")
    geoRaww = pd.read_sql(f"""
        SELECT Advertiser_ID,Country_Subdivision_Primary,Spend_USD
        FROM advertiser_geo_spend
        WHERE Advertiser_ID IN ({qmarkss})
    """,conn,params=ca_ids)
    geoRaww["Spend_USD"] = pd.to_numeric(geoRaww["Spend_USD"],errors="coerce").fillna(0)

    geoByAdvv = geoRaww.groupby("Advertiser_ID").agg(
        num_states=("Country_Subdivision_Primary","nunique"),
        total_geo_spend=("Spend_USD","sum"  ),
    ).reset_index()

    # THIS WAS THE BUG: geo table uses "CA" not "California"
    # spent like 2 hours on this. the creative table uses full
    # names but geo uses abbreviations because of course it does
    ca_onlyy = geoRaww[geoRaww["Country_Subdivision_Primary"]=="CA"]
    caGeoSpnd = ca_onlyy.groupby("Advertiser_ID").agg(
        ca_spend=("Spend_USD","sum"),
    ).reset_index(  )

    geoMrgd = geoByAdvv.merge(caGeoSpnd,on="Advertiser_ID",how="left"  )
    geoMrgd["ca_spend"] = geoMrgd["ca_spend"].fillna(0)
    geoMrgd["pct_spend_ca"] = np.where(
        geoMrgd["total_geo_spend"]>0,
        geoMrgd["ca_spend"] / geoMrgd["total_geo_spend"],
        0.0,
    )

    hhiValss = geoRaww.groupby("Advertiser_ID").apply(  _hhi_calc).reset_index()
    hhiValss.columns = ["Advertiser_ID","geo_hhi"]

    geoMrgd = geoMrgd.merge(hhiValss,on="Advertiser_ID",how="left")
    geoFinl = geoMrgd[["Advertiser_ID","num_states","pct_spend_ca","geo_hhi"]]

    featuress = featuress.merge(geoFinl,on="Advertiser_ID",how="left")

    nonzero_caa = (featuress["pct_spend_ca"]>0).sum()
    debug_totl = len(featuress)  # keeping this for now
    print(f"  Added geo features for {geoFinl['Advertiser_ID'].nunique()} advertisers")
    print(f"  Advertisers with non-zero pct_spend_ca: {nonzero_caa} / {debug_totl}")
    if(nonzero_caa==0  ):
        print("  WARNING: pct_spend_ca is still all zeros — investigate geo data!")

    # final cleanup
    print("\n[1.8] Final cleanup and export...")

    fillThse = ["weeks_active","spend_consistency","max_weekly_spend",
                 "pct_spend_in_peak_week","num_states","pct_spend_ca","geo_hhi"  ]
    for col in fillThse:
        if(col in featuress.columns  ):
            featuress[col] = featuress[col].fillna(0)

    print(f"  Final feature table: {len(featuress)} advertisers x {len(featuress.columns)} columns")
    print(f"  Columns: {list(featuress.columns)}")
    print(f"  Remaining nulls per column:"  )
    leftovr = featuress.isnull().sum()
    for col in leftovr[leftovr>0].index:
        print(f"    {col}: {leftovr[col]}")
    if(leftovr.sum()==0  ):
        print(f"    None!")

    featuress.to_sql("ca_advertiser_features",conn,if_exists="replace",index=False)
    print(f"  Saved to db table: ca_advertiser_features"  )

    csvPathh = os.path.join(PROJECT_ROOOT,"data","processed","ca_advertiser_features.csv")
    os.makedirs(os.path.dirname(csvPathh),exist_ok=True)
    featuress.to_csv(csvPathh,index=False  )
    print(f"  Saved to CSV: {csvPathh}")

    conn.close()
    print("\nPhase 1 complete.")
    return featuress


if __name__=="__main__":
    run()
