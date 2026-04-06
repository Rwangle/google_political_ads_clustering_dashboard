# refine_model.py - phase 4
# try winsorization, mini-batch, GMM, DBSCAN, extra features
# see if anything beats plain kmeans (it doesnt but report needs it)

import json
import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans,MiniBatchKMeans,DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score,calinski_harabasz_score,davies_bouldin_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scrpt_DIR    = os.path.dirname(os.path.abspath(  __file__))
PROJECT_ROOOT= os.path.dirname(  scrpt_DIR)
DB_PATHH     = os.path.join(PROJECT_ROOOT,"db","political_ads.db")
FIGSS        = os.path.join(PROJECT_ROOOT,"output","figures")
DATA_OUTT    = os.path.join(PROJECT_ROOOT,"output","data")

# same list as everywhere. should centralize this someday
CLUSTER_FEATSS = [
    "total_spend","ca_ad_count","avg_cpi","avg_ad_duration",
    "pct_video","pct_text","pct_image",
    "pct_age_targeted","pct_gender_targeted","avg_geo_targets",
    "spend_consistency","pct_spend_in_peak_week","pct_spend_ca","geo_hhi",
    "avg_pct_target_young","avg_pct_target_middle","avg_pct_target_older",
    "pct_female_only","pct_male_only","has_district_targeting",
]
logColss = ["total_spend","ca_ad_count","avg_cpi"]


def _preprocc(X_raww,logIdxss, do_winsorize=False):
    """preprocess: impute, optionally winsorize, log, scale.
    not using Pipeline bc need optional winsorize in middle"""
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X_raww)

    if(do_winsorize):
        from scipy.stats import mstats
        for i in range(X.shape[1]):
            X[:,i] = mstats.winsorize(X[:,i], limits=[0.01,0.01])

    for i in logIdxss:
        X[:,i] = np.log1p(np.abs(  X[:,i]))

    scl = StandardScaler()
    return scl.fit_transform(X)


def _label_archtyp(row, meansDff):
    """same naming logic as clustering.py, copy pasted bc
    phase 4 might relabel after switching models"""
    medSpndd = meansDff["total_spend"].median()
    medCVV   = meansDff["spend_consistency"].median()

    if row["total_spend"] > medSpndd*2:
        return "Big Spender / Statewide Blitz"
    elif row["pct_spend_in_peak_week"]>0.5:
        return "Bursty / Event-Driven"
    elif row["spend_consistency"] < medCVV*0.5:
        return "Steady-State Awareness"
    elif(row["pct_age_targeted"]>0.5 or row["pct_gender_targeted"]>0.3  ):
        return "Targeted Persuader"
    elif row["total_spend"]< medSpndd*0.5:
        return "Hyper-Local Grassroots"
    else:
        return "Mixed Strategy"


def run(db_path=None, final_k=None):
    if db_path is None:
        db_path = DB_PATHH

    os.makedirs(FIGSS,exist_ok=True)
    os.makedirs(DATA_OUTT,exist_ok=True  )

    conn = sqlite3.connect(db_path)
    print("="*60)
    print("PHASE 4: Model Refinement")
    print("="*60)

    featuress = pd.read_sql("SELECT * FROM ca_advertiser_features",conn)
    if final_k is None:
        final_k = featuress["cluster_label"].nunique()

    avail = [f for f in CLUSTER_FEATSS if f in featuress.columns]
    X_raww = featuress[avail].values.copy()
    logIdxss = [avail.index(f) for f in logColss if f in avail]

    allResultss = []
    reportLiness = []

    def log(msg):
        print(msg)
        reportLiness.append(msg)

    # 4.1 outlier handling
    log("\n[4.1] Outlier handling comparison")
    log("-"*40)

    xBasee = _preprocc(X_raww,logIdxss,do_winsorize=False)
    xWinss = _preprocc(X_raww,logIdxss,do_winsorize=True  )

    for tag,Xd in [("Baseline (log only)",xBasee),("Winsorized + log",xWinss)]:
        km = KMeans(n_clusters=final_k,n_init=20,random_state=42)
        lb = km.fit_predict(Xd)
        s  = silhouette_score(Xd,lb)
        ch = calinski_harabasz_score(Xd,lb)
        db = davies_bouldin_score(Xd,lb)
        log(f"  {tag}: Sil={s:.4f}, CH={ch:.1f}, DB={db:.4f}")
        allResultss.append({"method":tag,"sil":s,"ch":ch,"db":db,
                            "labels":lb,"X":Xd})

    # 4.2 alt algorithms
    log("\n[4.2] Alternative clustering algorithms")
    log("-"*40)

    xCmpp = xBasee

    mb = MiniBatchKMeans(n_clusters=final_k,n_init=10,random_state=42,batch_size=256)
    mbLabb = mb.fit_predict(xCmpp)
    mbSS = silhouette_score(xCmpp,mbLabb)
    mbChh= calinski_harabasz_score(xCmpp,mbLabb)
    mbDbb= davies_bouldin_score(xCmpp,mbLabb)
    log(f"  Mini-Batch K-Means: Sil={mbSS:.4f}, CH={mbChh:.1f}, DB={mbDbb:.4f}")
    allResultss.append({"method":"Mini-Batch K-Means","sil":mbSS,"ch":mbChh,
                        "db":mbDbb,"labels":mbLabb,"X":xCmpp})

    gm = GaussianMixture(n_components=final_k,n_init=5,random_state=42)
    gmLabb = gm.fit_predict(xCmpp)
    gmSS = silhouette_score(xCmpp,gmLabb)
    gmChh= calinski_harabasz_score(xCmpp,gmLabb)
    gmDbb= davies_bouldin_score(xCmpp,gmLabb)
    log(f"  GMM:               Sil={gmSS:.4f}, CH={gmChh:.1f}, DB={gmDbb:.4f}")
    allResultss.append({"method":"GMM","sil":gmSS,"ch":gmChh,
                        "db":gmDbb,"labels":gmLabb,"X":xCmpp})

    # DBSCAN - usually puts 99% as noise but gotta try
    bestDbSill = -1
    bestDbRess = None
    for epsVall in [0.5,1.0,1.5,2.0,2.5]:
        dbmm = DBSCAN(eps=epsVall,min_samples=5)
        dbLabb = dbmm.fit_predict(xCmpp)
        nCC = len(set(dbLabb))-(1 if -1 in dbLabb else 0)
        nNoisee = (dbLabb==-1).sum()
        if(nCC>=2):
            okk = (dbLabb!=-1)
            if(okk.sum()> nCC):
                try:
                    dss = silhouette_score(xCmpp[okk],dbLabb[okk])
                except Exception:
                    continue
                if dss>bestDbSill:
                    bestDbSill = dss
                    bestDbRess = {"eps":epsVall,"n_clusters":nCC,
                                    "n_noise":nNoisee,"sil":dss,
                                    "labels":dbLabb,"mask":okk}

    if bestDbRess is not None:
        r = bestDbRess
        log("  DBSCAN (eps={}): {} clusters, {} noise points, Sil={:.4f}".format(
            r["eps"],r["n_clusters"],r["n_noise"],r["sil"]))
    else:
        log("  DBSCAN: Could not find a configuration with >= 2 clusters")

    # 4.3 extra feature
    log("\n[4.3] Feature engineering iteration")
    log("-"*40)

    featuress["spend_per_creative"] = np.where(
        featuress["ca_ad_count"]>0,
        featuress["total_spend"]/featuress["ca_ad_count"],
        0,)
    extColss = avail + ["spend_per_creative"]
    xExtRaww = featuress[extColss].values.copy()
    extLogIdxx = [extColss.index(f) for f in logColss+["spend_per_creative"]
                 if f in extColss]
    xExtt = _preprocc(xExtRaww, extLogIdxx)

    kmExtt = KMeans(n_clusters=final_k,n_init=20,random_state=42)
    extLabb = kmExtt.fit_predict(xExtt)
    extSS = silhouette_score(xExtt,extLabb)
    extChh= calinski_harabasz_score(xExtt,extLabb)
    extDbb= davies_bouldin_score(xExtt,extLabb)
    log(f"  Extended features (+spend_per_creative): Sil={extSS:.4f}, CH={extChh:.1f}, DB={extDbb:.4f}")
    allResultss.append({"method":"K-Means + extended features","sil":extSS,
                        "ch":extChh,"db":extDbb,"labels":extLabb,"X":xExtt})

    # 4.4 pick best
    log("\n[4.4] Final model selection")
    log("-"*40)

    log("\n  Algorithm comparison summary:")
    log(f"  {'Method':<35} {'Silhouette':>12} {'CH Index':>12} {'DB Index':>12}")
    log(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*12}")
    for r in allResultss:
        log(f"  {r['method']:<35} {r['sil']:>12.4f} {r['ch']:>12.1f} {r['db']:>12.4f}")

    winnerr = max(allResultss,key=lambda x: x["sil"])
    log(f"\n  Best model: {winnerr['method']} (Silhouette={winnerr['sil']:.4f})")

    featuress["cluster_label"] = winnerr["labels"]

    pcaa = PCA(n_components=2)
    xyy = pcaa.fit_transform(winnerr["X"])
    featuress["pca_x"] = xyy[:,0]
    featuress["pca_y"] = xyy[:,1]

    # relabel archetypes
    cMeanss = featuress[avail+["cluster_label"]].groupby("cluster_label").mean()
    archNmss = {}
    for c in sorted(featuress["cluster_label"].unique()):
        if(c==-1):
            archNmss[c] = "Noise / Outlier"
            continue
        archNmss[c] = _label_archtyp(cMeanss.loc[c],cMeanss)
    featuress["archetype"] = featuress["cluster_label"].map(archNmss)

    # 4.5 export
    log("\n[4.5] Exporting final artifacts for dashboard")
    log("-"*40)

    featuress.to_sql("ca_advertiser_features",conn,if_exists="replace",index=False)
    log("  Updated db table: ca_advertiser_features")

    csvOutt = os.path.join(DATA_OUTT,"ca_clustered_advertisers.csv")
    featuress.to_csv(csvOutt,index=False)
    log(f"  Saved: {csvOutt}")

    # keep field descs from phase 2
    jsonPathh = os.path.join(DATA_OUTT,"cluster_summary.json")
    oldJsonn = {}
    if(os.path.exists(jsonPathh)):
        try:
            with open(jsonPathh,"r") as fh:
                oldJsonn = json.load(fh)
        except Exception:
            pass

    newJsonn = {
        "_metadata": {
            "description": "Final refined cluster profiles after model comparison (Phase 4). "
                           "This file is the machine-readable export of all clustering results, "
                           "structured for the dashboard UI team (Vaishnavi & Ankit).",
            "best_method": winnerr["method"],
            "best_silhouette": round(winnerr["sil"],4),
            "num_advertisers": len(featuress),
            "num_clusters": int(final_k),
            "features_used": avail,
            "proposal_context": "MacQueen (1967) K-Means clustering to 'cluster advertisers based "
                                "on factors like spending, efficiency, and audience reach to identify "
                                "different ad strategies.'",
        },
        "_field_descriptions": oldJsonn.get("_field_descriptions", {
            "archetype": "Human-readable label for the cluster's dominant campaign strategy pattern",
            "size": "Number of CA-targeting advertisers in this cluster",
            "feature_means": "Average value of each clustering feature for advertisers in this cluster",
            "top_spenders": "Top 5 advertisers by total spend within this cluster",
        }),
    }

    logSpp  = np.log1p(featuress["total_spend"])
    cpiColl = featuress["avg_cpi"]
    medSpp  = logSpp.median()
    medCpii = cpiColl.median()

    for c in sorted(featuress["cluster_label"].unique()):
        if(c==-1):
            continue
        mskk = (featuress["cluster_label"]==c)
        sz = int(mskk.sum())
        meansDd = cMeanss.loc[c].to_dict()
        top5 = (featuress[mskk].nlargest(5,"total_spend")
                [["Advertiser_Name","total_spend"]].to_dict("records"))

        cSS = logSpp[mskk]
        cCC = cpiColl[mskk]
        newJsonn[str(c)] = {
            "archetype": archNmss[c],
            "size": sz,
            "feature_means":{k:round(v,4) for k,v in meansDd.items()},
            "quadrant_distribution": {
                "high_spend_high_efficiency": int(((cSS>=medSpp)&(cCC< medCpii)).sum()),
                "high_spend_low_efficiency":  int(((cSS>=medSpp)&(cCC>=medCpii)).sum()),
                "low_spend_high_efficiency":  int(((cSS< medSpp)&(cCC< medCpii)).sum()),
                "low_spend_low_efficiency":   int(((cSS< medSpp)&(cCC>=medCpii)).sum()),
            },
            "top_spenders": top5,
        }

    with open(jsonPathh,"w") as fh:
        json.dump(newJsonn,fh,indent=2,default=str)
    log(f"  Saved: {jsonPathh}")

    # scatter
    fig,ax = plt.subplots(figsize=(10,8))
    expvarr = pcaa.explained_variance_ratio_
    for c in sorted(featuress["cluster_label"].unique()):
        mskk = (featuress["cluster_label"]==c)
        if(c==-1):
            ax.scatter(featuress.loc[mskk,"pca_x"],featuress.loc[mskk,"pca_y"],
                       c="gray",marker="x",alpha=0.4,s=20,label="Noise")
        else:
            ax.scatter(featuress.loc[mskk,"pca_x"],featuress.loc[mskk,"pca_y"],
                       label=f"C{c}: {archNmss[c]}",alpha=0.6,s=30)
    ax.set_xlabel(f"PC1 ({expvarr[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({expvarr[1]:.1%} variance)")
    ax.set_title("Refined CA Political Advertiser Clusters (PCA)")
    ax.legend(fontsize=8,loc="best")
    ax.grid(True,alpha=0.3)
    fig.savefig(os.path.join(FIGSS,"cluster_scatter_refined.png"),dpi=150,bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {FIGSS}/cluster_scatter_refined.png")

    conn.close()
    log("\nPhase 4 complete.")

    return featuress, reportLiness


if __name__=="__main__":
    run()
