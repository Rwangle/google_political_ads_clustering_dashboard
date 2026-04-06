# validation.py - phase 3
# checks if clustering is any good
# silhouette, CH, DB, stability, sensitivity

import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score,silhouette_samples,
                              calinski_harabasz_score,davies_bouldin_score,
                              adjusted_rand_score)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from sklearn.pipeline import Pipeline

scrpt_DIR    = os.path.dirname(os.path.abspath(  __file__))
PROJECT_ROOOT= os.path.dirname(  scrpt_DIR)
DB_PATHH     = os.path.join(PROJECT_ROOOT,"db","political_ads.db")
FIGSS        = os.path.join(PROJECT_ROOOT,"output","figures")

# TODO: centralize this, copy pasted from clustering.py again
ALL_FEATSS = [
    "total_spend","ca_ad_count","avg_cpi","avg_ad_duration",
    "pct_video","pct_text","pct_image",
    "pct_age_targeted","pct_gender_targeted","avg_geo_targets",
    "spend_consistency","pct_spend_in_peak_week","pct_spend_ca","geo_hhi",
    "avg_pct_target_young","avg_pct_target_middle","avg_pct_target_older",
    "pct_female_only","pct_male_only","has_district_targeting",
]
logColss = ["total_spend","ca_ad_count","avg_cpi"]

# sensitivity subsets
SPEND_COLSS = ["total_spend","avg_cpi","spend_consistency",
               "pct_spend_in_peak_week","avg_ad_spend","max_weekly_spend"]
TARGET_COLSS = [
    "pct_age_targeted","pct_gender_targeted","avg_geo_targets",
    "pct_spend_ca","geo_hhi","num_states",
    "avg_pct_target_young","avg_pct_target_middle","avg_pct_target_older",
    "pct_female_only","pct_male_only","has_district_targeting",
]


def _preprocc(X_raww, whchToLog):
    """same preprocessing as clustering.py.
    duplicated. havent fixed."""
    def _logItt(X):
        out = X.copy()
        for i in whchToLog:
            out[:,i] = np.log1p(np.abs(out[:,i]  ))
        return out

    p = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log",FunctionTransformer(_logItt,validate=False)),
        ("scale",StandardScaler(  )),
    ])
    return p.fit_transform(X_raww)


def run(db_path=None, final_k=None, X_scaled=None  ):
    if db_path is None:
        db_path = DB_PATHH

    os.makedirs(FIGSS, exist_ok=True)
    conn = sqlite3.connect(  db_path)

    print("=" * 60)
    print("PHASE 3: Analysis Validation")
    print("="*60)

    featuress = pd.read_sql("SELECT * FROM ca_advertiser_features", conn)
    avail = [f for f in ALL_FEATSS if f in featuress.columns]

    if X_scaled is None:
        raww = featuress[avail].values.copy()
        idxss = [avail.index(f) for f in logColss if f in avail]
        X_scaled = _preprocc(raww, idxss)

    if final_k is None:
        final_k = featuress["cluster_label"].nunique(  )

    theLabelss = featuress["cluster_label"].values
    reportLiness = []

    def log(msg):
        print(msg)
        reportLiness.append(msg)

    # 3.1 internal metrics
    log("\n[3.1] Internal validation metrics")
    log("-"*40)

    for k in [max(2,final_k-1), final_k, final_k+1]:
        km = KMeans(n_clusters=k,n_init=10,random_state=42)
        tmpp = km.fit_predict(X_scaled)
        s  = silhouette_score(X_scaled,tmpp)
        ch = calinski_harabasz_score(X_scaled,tmpp)
        db = davies_bouldin_score(X_scaled,tmpp)
        tag = " <-- chosen" if k==final_k else ""
        log(f"  k={k}: Silhouette={s:.4f}, Calinski-Harabasz={ch:.1f}, Davies-Bouldin={db:.4f}{tag}")

    silFinall = silhouette_score(X_scaled, theLabelss)
    chFinall  = calinski_harabasz_score(X_scaled, theLabelss)
    dbFinall  = davies_bouldin_score(X_scaled, theLabelss)
    log(f"\n  Final model (k={final_k}):")
    log(f"    Silhouette Score:        {silFinall:.4f}")
    log(f"    Calinski-Harabasz Index:  {chFinall:.1f}")
    log(f"    Davies-Bouldin Index:     {dbFinall:.4f}")

    if(silFinall > 0.5  ):
        log("    Silhouette interpretation: GOOD (>0.5)")
    elif(silFinall > 0.25):
        log("    Silhouette interpretation: ACCEPTABLE (>0.25)")
    else:
        log("    Silhouette interpretation: WEAK (<0.25)")

    metricsOutt = {
        "silhouette": silFinall,
        "calinski_harabasz": chFinall,
        "davies_bouldin": dbFinall,
        "final_k": final_k,
    }

    # 3.2 stability
    log("\n[3.2] Stability analysis (10 random seeds)")
    log("-"*40)

    allRunss = []
    for seed in range(10  ):
        km = KMeans(n_clusters=final_k,n_init=10,random_state=seed)
        allRunss.append(km.fit_predict(X_scaled))

    ariListt = []
    for i in range(10):
        for j in range(i+1,10):
            ariListt.append(adjusted_rand_score(allRunss[i],allRunss[j]))

    meanARII = np.mean(ariListt)
    minARII  = np.min(ariListt)
    maxARII  = np.max(ariListt)
    log(f"  Mean ARI: {meanARII:.4f}")
    log(f"  Min ARI:  {minARII:.4f}")
    log(f"  Max ARI:  {maxARII:.4f}")

    if(meanARII > 0.8  ):
        log("  Stability: GOOD (ARI > 0.8)")
    elif(meanARII >0.6):
        log("  Stability: MODERATE (ARI 0.6-0.8)")
    else:
        log("  Stability: WEAK (ARI < 0.6)")

    metricsOutt["mean_ari"] = meanARII
    metricsOutt["min_ari"]  = minARII

    # 3.3 per-cluster silhouette
    log("\n[3.3] Per-cluster silhouette analysis")
    log("-"*40)

    perSamplee = silhouette_samples(X_scaled, theLabelss)

    fig,ax = plt.subplots(figsize=(10,7))
    yLo = 0
    clrss = plt.cm.tab10(np.linspace(0,1,final_k))

    for c in range(final_k):
        valss = perSamplee[theLabelss==c]
        valss.sort()
        n = len(valss)
        yHi = yLo + n
        ax.fill_betweenx(np.arange(yLo,yHi), 0,valss,alpha=0.7,color=clrss[c])
        ax.text(-0.05,yLo+0.5*n, f"C{c}",fontsize=10,fontweight="bold")

        negCtt = (valss<0).sum()
        negPctt = negCtt/n * 100
        log(f"  Cluster {c}: mean sil={valss.mean():.4f}, negative samples={negCtt} ({negPctt:.1f}%)")

        yLo = yHi

    ax.axvline(x=silFinall,color="red",linestyle="--",label=f"Mean: {silFinall:.3f}")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Advertisers (sorted by cluster)")
    ax.set_title("Per-Cluster Silhouette Analysis")
    ax.legend(loc="best")
    fig.savefig(os.path.join(FIGSS,"silhouette_per_cluster.png"),dpi=150,bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {FIGSS}/silhouette_per_cluster.png")

    # 3.4 sensitivity
    log("\n[3.4] Sensitivity to feature subsets")
    log("-"*40)

    spendAvll  = [c for c in SPEND_COLSS if c in featuress.columns]
    targetAvll = [c for c in TARGET_COLSS if c in featuress.columns]

    subsetsToTryy = {
        "A (Spending only)":  spendAvll,
        "B (Targeting only)": targetAvll,
        "C (Full features)":  avail,
    }

    for name,cols in subsetsToTryy.items():
        Xsubb = featuress[cols].values.copy()
        li = [i for i,c in enumerate(cols) if c in logColss]
        Xsubb_s = _preprocc(Xsubb,li)

        km = KMeans(n_clusters=final_k,n_init=10,random_state=42)
        subLabb = km.fit_predict(Xsubb_s)
        subSill = silhouette_score(Xsubb_s,subLabb)

        if(name.startswith("C")):
            ariVss = 1.0
        else:
            ariVss = adjusted_rand_score(theLabelss,subLabb)

        log(f"  Subset {name} ({len(cols)} features):")
        log(f"    Silhouette: {subSill:.4f}, ARI vs full: {ariVss:.4f}")

    conn.close()
    print("\nPhase 3 complete.")
    return metricsOutt, reportLiness


if __name__=="__main__":
    run()
