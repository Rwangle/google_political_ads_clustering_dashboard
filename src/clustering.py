# clustering.py - phase 2
# kmeans + all the viz for dashboard
# way too long but dont want to split rn

import json
import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import tab10
import seaborn  # might use for heatmaps eventually

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

scrpt_DIR = os.path.dirname(os.path.abspath(   __file__))
PROJECT_ROOOT = os.path.dirname(  scrpt_DIR)
DB_PATHH  = os.path.join(PROJECT_ROOOT,"db","political_ads.db")
FIGSS     = os.path.join(PROJECT_ROOOT,"output","figures")
DATA_OUTT = os.path.join(PROJECT_ROOOT,  "output","data")

# 20 features we cluster on
FEATSS = [
    "total_spend","ca_ad_count","avg_cpi","avg_ad_duration",
    "pct_video","pct_text","pct_image",
    "pct_age_targeted","pct_gender_targeted","avg_geo_targets",
    "spend_consistency","pct_spend_in_peak_week",
    "pct_spend_ca","geo_hhi",
    # added after 1st round when clusters just split on ad format
    "avg_pct_target_young","avg_pct_target_middle","avg_pct_target_older",
    "pct_female_only","pct_male_only",
    "has_district_targeting",
]

# super skewed, need log
logColss = ["total_spend","ca_ad_count","avg_cpi"]


def _make_pipeline(  ):
    """impute, log skewed stuff, standardize.
    returns (pipe, log_index_list). caller populates log_index_list
    before fit_transform. janky but works"""
    logIdxx = []

    def _doLogg(X):
        out = X.copy()
        for i in logIdxx:
            out[:,i] = np.log1p(np.abs(  out[:,i]))
        return out

    pipe = Pipeline([
        ("impute",SimpleImputer(strategy="median")),
        ("log",FunctionTransformer(_doLogg,validate=False)),
        ("scale", StandardScaler(  )),
    ])
    return pipe,logIdxx


def _name_archtyp(clstrRow, allMns):
    """give cluster a human readable name. thresholds are kinda
    arbitrary but names ended up being descriptive enough"""
    medSpnd = allMns["total_spend"].median()
    medCVV  = allMns["spend_consistency"].median(  )

    if clstrRow["total_spend"] > medSpnd*2:
        return "Big Spender / Statewide Blitz"
    elif clstrRow["pct_spend_in_peak_week"]>0.5:
        return "Bursty / Event-Driven"
    elif clstrRow["spend_consistency"]< medCVV *0.5:
        return "Steady-State Awareness"
    elif(clstrRow["pct_age_targeted"]>0.5 or clstrRow["pct_gender_targeted"]>0.3  ):
        return "Targeted Persuader"
    elif clstrRow["total_spend"] < medSpnd *0.5:
        return "Hyper-Local Grassroots"
    else:
        return "Mixed Strategy"


def run( db_path=None):
    if db_path is None:
        db_path = DB_PATHH

    os.makedirs(FIGSS,exist_ok=True  )
    os.makedirs(DATA_OUTT,exist_ok=True)

    conn = sqlite3.connect(db_path)
    print("=" * 60)
    print("PHASE 2: Clustering & Analysis")
    print("="*60)

    # load features
    print("\n[2.1] Loading feature table and preparing for clustering...")
    featuress = pd.read_sql("SELECT * FROM ca_advertiser_features",conn)
    print(f"  Loaded {len(featuress)} advertisers w/ {len(featuress.columns)} cols")

    availbl = [f for f in FEATSS if f in featuress.columns]
    missingFts = [f for f in FEATSS if f not in featuress.columns  ]
    if(missingFts):
        print(f"  WARNING: Missing features (skipping): {missingFts}")
    print(f"  Clustering on {len(availbl)} features")

    X_raww = featuress[availbl].values.copy()

    pipe,logIdxx = _make_pipeline()
    for fn in logColss:
        if fn in availbl:
            logIdxx.append(availbl.index(fn  ))

    X_scaledd = pipe.fit_transform(X_raww)
    print(f"  Preprocessed shape: {X_scaledd.shape}")

    # find best k
    print("\n[2.2] Evaluating k from 2 to 10..."  )
    allInertiass = []
    allSilss     = []

    for k in range(2,11):
        km_tmpp = KMeans(n_clusters=k,n_init=10,random_state=42)
        tmpLabb = km_tmpp.fit_predict(X_scaledd)
        allInertiass.append(km_tmpp.inertia_)
        s = silhouette_score(X_scaledd, tmpLabb)
        allSilss.append(s)
        print(f"  k={k}: inertia={km_tmpp.inertia_:,.0f}, silhouette={s:.4f}")

    # elbow plot
    fig,ax = plt.subplots(figsize=(8,5))
    ax.plot(list(range(2,11)),allInertiass,"bo-",linewidth=2)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia (WCSS)")
    ax.set_title("Elbow Plot: K-Means Inertia vs. k")
    ax.grid(True,alpha=0.3)
    fig.savefig(os.path.join(FIGSS,"elbow_plot.png"),dpi=150,bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGSS}/elbow_plot.png")

    # sil plot
    fig,ax = plt.subplots(figsize=(8,5))
    ax.plot(list(range(2,11)),allSilss,"rs-",linewidth=2)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Average Silhouette Score")
    ax.set_title("Silhouette Score vs. k")
    ax.grid(True,alpha=0.3)
    fig.savefig(os.path.join(FIGSS,"silhouette_plot.png"),dpi=150,bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGSS}/silhouette_plot.png")

    bestIdxx = int(np.argmax(allSilss))
    final_k = bestIdxx + 2  # range starts at 2
    print(f"\n  Optimal k by silhouette: {final_k} (score={allSilss[bestIdxx]:.4f})")

    # fit final
    print(f"\n[2.3] Fitting final K-Means with k={final_k}...")
    kmeansss = KMeans(n_clusters=final_k,n_init=20,random_state=42)
    labelss = kmeansss.fit_predict(X_scaledd)
    featuress["cluster_label"] = labelss
    uniqq,ctss = np.unique(labelss,return_counts=True)
    print(f"  Cluster sizes: {dict(zip(uniqq.tolist(),ctss.tolist()))}")

    # profile
    print("\n[2.4] Profiling clusters...")
    clstrMeans = featuress[availbl + ["cluster_label"]].groupby("cluster_label").mean()

    archNms = {}
    for c in range(final_k):
        archNms[c] = _name_archtyp(clstrMeans.loc[c], clstrMeans)
    # dedupe names (happened once w/ k=5 during testing)
    _seenn = {}
    for c in range(final_k):
        nm = archNms[c]
        if nm in _seenn:
            archNms[c] = f"{nm} {c}"
        _seenn[nm] = True

    featuress["archetype"] = featuress["cluster_label"].map(archNms)

    print("\n  Cluster Profiles:")
    for c in range(final_k):
        row = clstrMeans.loc[c]
        sz =(featuress["cluster_label"]==c).sum()
        print(f"\n  Cluster {c}: {archNms[c]} (n={sz})")
        print(f"    total_spend:     ${row['total_spend']:>12,.0f}")
        print(f"    ca_ad_count:     {row['ca_ad_count']:>12,.1f}")
        print(f"    avg_cpi:         ${row['avg_cpi']:>12,.4f}")
        print(f"    avg_ad_duration: {row['avg_ad_duration']:>12,.1f} days")
        print(f"    pct_video:       {row['pct_video']:>12.1%}")
        print(f"    pct_text:        {row['pct_text']:>12.1%}")
        print(f"    pct_image:       {row['pct_image']:>12.1%}")
        if "pct_spend_ca" in row:
            print(f"    pct_spend_ca:    {row['pct_spend_ca']:>12.1%}")
        if "avg_pct_target_young" in row:
            print(f"    target_young:    {row['avg_pct_target_young']:>12.1%}")
            print(f"    target_middle:   {row['avg_pct_target_middle']:>12.1%}")
            print(f"    target_older:    {row['avg_pct_target_older']:>12.1%}")

    print("\n  Top 5 spenders per cluster:")
    for c in range(final_k):
        biggst = featuress[featuress["cluster_label"]==c].nlargest(5,"total_spend")
        print(f"\n  Cluster {c} ({archNms[c]}):")
        for _,r in biggst.iterrows():
            print(f"    {r['Advertiser_Name']}: ${r['total_spend']:,.0f}")

    # PCA
    print("\n[2.5] Running PCA for 2D visualization...")
    pcaa = PCA(n_components=2)
    coordss = pcaa.fit_transform(X_scaledd)
    featuress["pca_x"] = coordss[:,0]
    featuress["pca_y"] = coordss[:,1]

    expll = pcaa.explained_variance_ratio_
    print(f"  PCA explained variance: PC1={expll[0]:.3f}, PC2={expll[1]:.3f}, total={sum(expll):.3f}")

    fig,ax = plt.subplots(figsize=(10,8))
    for c in range(final_k):
        msk = featuress["cluster_label"]==c
        ax.scatter(featuress.loc[msk,"pca_x"],
                   featuress.loc[msk,"pca_y"],
            label=f"C{c}: {archNms[c]}",
            alpha=0.6,s=30)
    ax.set_xlabel(f"PC1 ({expll[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({expll[1]:.1%} variance)")
    ax.set_title("CA Political Advertiser Clusters (PCA Projection)")
    ax.legend(fontsize=8,loc="best")
    ax.grid(True,alpha=0.3)
    fig.savefig(os.path.join(FIGSS,"cluster_scatter_pca.png"),dpi=150,bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGSS}/cluster_scatter_pca.png")

    # radar
    print("\n[2.5b] Creating radar chart of cluster profiles...")
    normdd = clstrMeans.copy()
    for col in normdd.columns:
        lo,hi = normdd[col].min(),normdd[col].max()
        if hi>lo:
            normdd[col] = (normdd[col]-lo)/(hi-lo)
        else:
            normdd[col] = 0.5

    rdrColss = ["total_spend","ca_ad_count","avg_cpi","pct_video",
               "pct_age_targeted","avg_geo_targets","spend_consistency",
               "pct_spend_ca","geo_hhi"]
    rdrColss = [f for f in rdrColss if f in normdd.columns]
    lblMapp = {"total_spend":"Spend","ca_ad_count":"Ad Count",
              "avg_cpi":"CPI","pct_video":"% Video",
              "pct_age_targeted":"% Age Tgt","avg_geo_targets":"Geo Targets",
              "spend_consistency":"Spend Var","pct_spend_ca":"% CA Spend",
              "geo_hhi":"Geo HHI"}
    dispLblss = [lblMapp.get(f,f) for f in rdrColss]

    angless = np.linspace(0,2*np.pi,len(rdrColss),endpoint=False).tolist()
    angless += angless[:1]

    fig,ax = plt.subplots(figsize=(9,9),subplot_kw=dict(polar=True  ))
    for c in range(final_k):
        valss = normdd.loc[c,rdrColss].tolist()
        valss += valss[:1]
        ax.plot(angless,valss,"o-",linewidth=2,label=f"C{c}: {archNms[c]}")
        ax.fill(angless,valss,alpha=0.1)
    ax.set_xticks(angless[:-1])
    ax.set_xticklabels(dispLblss,fontsize=9)
    ax.set_title("Cluster Profile Comparison (Normalized)",pad=20)
    ax.legend(fontsize=8,loc="upper right",bbox_to_anchor=(1.3,1.1))
    fig.savefig(os.path.join(FIGSS,"cluster_profiles_radar.png"),dpi=150,bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGSS}/cluster_profiles_radar.png")

    # demographic bars
    print("\n[2.5c] Creating demographic breakdown visualizations...")

    ageColss = [c for c in ["avg_pct_target_young","avg_pct_target_middle",
                            "avg_pct_target_older"]
               if c in featuress.columns]
    if(ageColss):
        ageAvgss = featuress.groupby("cluster_label")[ageColss].mean()
        fig,ax = plt.subplots(figsize=(10,6))
        bw = 0.25
        x = np.arange(final_k)
        niceNmss = {"avg_pct_target_young":"Young (18-34)",
                     "avg_pct_target_middle":"Middle (35-54)",
                     "avg_pct_target_older":"Older (55+)"}
        for i,col in enumerate(ageColss):
            ax.bar(x+i*bw, ageAvgss[col],bw, label=niceNmss.get(col,col))
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Average Proportion of Targeted Age Brackets")
        ax.set_title("Age Group Targeting by Cluster")
        ax.set_xticks(x+bw)
        ax.set_xticklabels([f"C{c}: {archNms[c]}" for c in range(final_k)],
                           rotation=15,ha="right",fontsize=9)
        ax.legend()
        ax.grid(True,alpha=0.3,axis="y")
        fig.savefig(os.path.join(FIGSS,"cluster_age_targeting.png"),dpi=150,bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {FIGSS}/cluster_age_targeting.png")

    gndColss = [c for c in ["pct_gender_targeted","pct_female_only","pct_male_only"]
               if c in featuress.columns]
    if(gndColss):
        gndAvgss = featuress.groupby("cluster_label")[gndColss].mean()
        fig,ax = plt.subplots(figsize=(10,6))
        bw = 0.25
        x = np.arange(final_k)
        gndNmss = {"pct_gender_targeted":"Any Gender Targeting",
                     "pct_female_only":"Female-Only Targeting",
                     "pct_male_only":"Male-Only Targeting"}
        for i, col in enumerate(gndColss):
            ax.bar(x + i*bw,gndAvgss[col],bw,label=gndNmss.get(col,col))
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Proportion of Ads")
        ax.set_title("Gender Targeting by Cluster")
        ax.set_xticks(x+bw)
        ax.set_xticklabels([f"C{c}: {archNms[c]}" for c in range(final_k)],
                           rotation=15,ha="right",fontsize=9)
        ax.legend()
        ax.grid(True,alpha=0.3,axis="y")
        fig.savefig(os.path.join(FIGSS,"cluster_gender_targeting.png"),dpi=150,bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {FIGSS}/cluster_gender_targeting.png")

    # district targeting
    if("has_district_targeting" in featuress.columns):
        print("\n[2.5d] Creating district targeting visualization...")
        distAvgss = featuress.groupby("cluster_label")[
            ["has_district_targeting","avg_ca_districts"]
        ].mean()

        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,6))
        clrr = plt.cm.tab10(np.linspace(0,1,final_k))

        bars1 = ax1.bar(range(final_k),distAvgss["has_district_targeting"],color=clrr)
        ax1.set_xlabel("Cluster")
        ax1.set_ylabel("Proportion with District Targeting")
        ax1.set_title("CA Congressional District Targeting by Cluster")
        ax1.set_xticks(range(final_k))
        ax1.set_xticklabels([f"C{c}" for c in range(final_k)])
        for bar,val in zip(bars1,distAvgss["has_district_targeting"]):
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                     f"{val:.1%}",ha="center",fontsize=10)
        ax1.grid(True,alpha=0.3,axis="y")

        bars2 = ax2.bar(range(final_k),distAvgss["avg_ca_districts"],color=clrr)
        ax2.set_xlabel("Cluster")
        ax2.set_ylabel("Avg CA Districts Targeted")
        ax2.set_title("Average Number of CA Districts Targeted by Cluster")
        ax2.set_xticks(range(final_k))
        ax2.set_xticklabels(["C{}".format(c) for c in range(final_k)])
        for bar,val in zip(bars2,distAvgss["avg_ca_districts"]):
            ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.05,
                     "{:.1f}".format(val),ha="center",fontsize=10)
        ax2.grid(True,alpha=0.3,axis="y")

        fig.suptitle("Location Targeting: CA Congressional Districts",fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGSS,"cluster_district_targeting.png"),
                    dpi=150,bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {FIGSS}/cluster_district_targeting.png")

    # spend vs efficiency quadrant
    # answers the proposal question about high spend/low eff
    print("\n[2.5e] Creating spend vs efficiency quadrant plot...")

    logSpndd = np.log1p(featuress["total_spend"])
    cpi_valss = featuress["avg_cpi"]
    medLogSpndd = logSpndd.median()
    medCpii     = cpi_valss.median()

    fig,ax = plt.subplots(figsize=(12,9))
    for c in range(final_k):
        m = (featuress["cluster_label"]==c)
        ax.scatter(logSpndd[m],cpi_valss[m],
                   label=f"C{c}: {archNms[c]}",alpha=0.5,s=25)

    ax.axvline(x=medLogSpndd,color="gray",linestyle="--",alpha=0.7)
    ax.axhline(y=medCpii,     color="gray",linestyle="--",alpha=0.7)

    xl = ax.get_xlim()
    yl = ax.get_ylim()
    kw = dict(fontsize=11,fontweight="bold",alpha=0.4,ha="center",va="center")
    ax.text((xl[0]+medLogSpndd)/2,(medCpii+yl[1])/2,
            "LOW SPEND\nLOW EFFICIENCY",**kw,color="red")
    ax.text((medLogSpndd+xl[1])/2,(medCpii+yl[1])/2,
            "HIGH SPEND\nLOW EFFICIENCY",**kw,color="darkred")
    ax.text((xl[0]+medLogSpndd)/2,(yl[0]+medCpii)/2,
            "LOW SPEND\nHIGH EFFICIENCY",**kw,color="green")
    ax.text((medLogSpndd+xl[1])/2,(yl[0]+medCpii)/2,
            "HIGH SPEND\nHIGH EFFICIENCY",**kw,color="darkgreen")

    ax.set_xlabel("Total Spend (log scale)",fontsize=11)
    ax.set_ylabel("Avg Cost Per Impression ($)",fontsize=11)
    ax.set_title("Spend vs. Efficiency Quadrant Analysis\n"
                 "(Proposal: 'high spending with low efficiency' vs. 'low spending with high efficiency')",
                 fontsize=12)
    ax.legend(fontsize=9,loc="upper right")
    ax.grid(True,alpha=0.3)

    ticksRaww = [100,1000,10000,100000,1000000,10000000,100000000]
    tickLblss = ["$100","$1K","$10K","$100K","$1M","$10M","$100M"]
    axTopp = ax.twiny()
    axTopp.set_xlim(ax.get_xlim())
    axTopp.set_xticks([np.log1p(s) for s in ticksRaww])
    axTopp.set_xticklabels(tickLblss,fontsize=8)
    axTopp.set_xlabel("Actual Spend ($)",fontsize=9)

    fig.savefig(os.path.join(FIGSS,"spend_vs_efficiency_quadrant.png"),
                dpi=150,bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGSS}/spend_vs_efficiency_quadrant.png")

    # quadrant numbers
    print("\n  Spend vs Efficiency quadrant distribution:")
    for c in range(final_k):
        m = (featuress["cluster_label"]==c)
        cS = logSpndd[m]
        cC = cpi_valss[m]
        n = m.sum()
        hh = ((cS>=medLogSpndd) & (cC< medCpii)).sum()
        hl = ((cS>=medLogSpndd) & (cC>=medCpii)).sum()
        lh = ((cS< medLogSpndd) & (cC< medCpii)).sum()
        ll = ((cS< medLogSpndd) & (cC>=medCpii)).sum()
        print(f"  Cluster {c} ({archNms[c]}, n={n}):")
        print(f"    High Spend + High Efficiency: {hh} ({hh/n:.0%})")
        print(f"    High Spend + Low Efficiency:  {hl} ({hl/n:.0%})")
        print(f"    Low Spend + High Efficiency:  {lh} ({lh/n:.0%})")
        print(f"    Low Spend + Low Efficiency:   {ll} ({ll/n:.0%})")

    # save
    print("\n[2.6] Saving outputs...")
    featuress.to_sql("ca_advertiser_features",conn,if_exists="replace",index=False)
    print(f"  Updated db table: ca_advertiser_features")

    featuress.to_csv(os.path.join(DATA_OUTT,"ca_clustered_advertisers.csv"),index=False)
    print(f"  Saved: {DATA_OUTT}/ca_clustered_advertisers.csv")

    # json for dashboard team
    summaryJsonn = {
        "_metadata": {
            "description": "Cluster profiles for CA political advertisers. Each cluster represents "
                           "a distinct ad strategy archetype identified by K-Means clustering on "
                           "{} features.".format(len(availbl)),
            "num_advertisers": len(featuress),
            "num_clusters": final_k,
            "optimal_k_method": "Silhouette score maximization over k=2..10",
            "silhouette_score": float(allSilss[bestIdxx]),
            "features_used": availbl,
            "proposal_context": "MacQueen (1967) K-Means clustering to 'cluster advertisers based "
                                "on factors like spending, efficiency, and audience reach to identify "
                                "different ad strategies.'",
        },
        "_field_descriptions": {
            "archetype": "Human-readable label for the cluster's dominant campaign strategy pattern",
            "size": "Number of CA-targeting advertisers in this cluster",
            "feature_means": {
                "total_spend": "Average total USD spent by advertisers in this cluster (across all their ads, not just CA)",
                "ca_ad_count": "Average number of CA-targeted ad creatives per advertiser",
                "avg_cpi": "Average cost per impression in USD — lower means more efficient at reaching eyeballs (Coppock et al. 2020: 'tools to measure spending efficiency')",
                "avg_ad_duration": "Average number of days each ad ran",
                "pct_video": "Proportion of ads that are video format (0.0 to 1.0)",
                "pct_text": "Proportion of ads that are text format (0.0 to 1.0)",
                "pct_image": "Proportion of ads that are image format (0.0 to 1.0)",
                "pct_age_targeted": "Proportion of ads with any age demographic targeting applied",
                "pct_gender_targeted": "Proportion of ads with any gender demographic targeting applied",
                "avg_geo_targets": "Average number of geographic entities targeted per ad (states, districts, etc.)",
                "spend_consistency": "Coefficient of variation of weekly spend — higher means more bursty/irregular spending",
                "pct_spend_in_peak_week": "Fraction of total spend concentrated in the single highest-spending week",
                "pct_spend_ca": "Fraction of advertiser's total geographic spend allocated to California",
                "geo_hhi": "Herfindahl-Hirschman Index across states (0-1) — higher means spend concentrated in fewer states",
                "avg_pct_target_young": "Average proportion of age-targeted brackets that are young (18-34)",
                "avg_pct_target_middle": "Average proportion of age-targeted brackets that are middle-aged (35-54)",
                "avg_pct_target_older": "Average proportion of age-targeted brackets that are older (55+)",
                "pct_female_only": "Proportion of ads targeting female audiences exclusively",
                "pct_male_only": "Proportion of ads targeting male audiences exclusively",
                "has_district_targeting": "Proportion of ads targeting specific CA congressional districts (vs. statewide)",
            },
            "quadrant_distribution": {
                "description": "Distribution of advertisers across the Spend vs. Efficiency quadrant plot. "
                               "Proposal: 'high spending with low efficiency, low spending with high efficiency'",
                "high_spend_high_efficiency": "Advertisers above median spend AND below median CPI — large, efficient operations",
                "high_spend_low_efficiency": "Advertisers above median spend AND above median CPI — large but costly per impression",
                "low_spend_high_efficiency": "Advertisers below median spend AND below median CPI — small but efficient",
                "low_spend_low_efficiency": "Advertisers below median spend AND above median CPI — small and costly",
            },
            "top_spenders": "Top 5 advertisers by total spend within this cluster",
        },
    }

    for c in range(final_k):
        cMskk = (featuress["cluster_label"]==c)
        sz = int(cMskk.sum())
        meansDd = clstrMeans.loc[c].to_dict()
        top5 = featuress[cMskk].nlargest(5,"total_spend")[
            ["Advertiser_Name","total_spend"]
        ].to_dict("records")

        cLSS = logSpndd[cMskk]
        cCPP = cpi_valss[cMskk]
        summaryJsonn[str(c)] = {
            "archetype": archNms[c],
            "size": sz,
            "feature_means": {k:round(v,4) for k,v in meansDd.items()},
            "quadrant_distribution": {
                "high_spend_high_efficiency": int(((cLSS>=medLogSpndd)&(cCPP< medCpii)).sum()),
                "high_spend_low_efficiency":  int(((cLSS>=medLogSpndd)&(cCPP>=medCpii)).sum()),
                "low_spend_high_efficiency":  int(((cLSS< medLogSpndd)&(cCPP< medCpii)).sum()),
                "low_spend_low_efficiency":   int(((cLSS< medLogSpndd)&(cCPP>=medCpii)).sum()),
            },
            "top_spenders": top5,
        }

    jsonOutt = os.path.join(DATA_OUTT,"cluster_summary.json")
    with open(jsonOutt,"w") as fh:
        json.dump(summaryJsonn,fh,indent=2,default=str)
    print(f"  Saved: {jsonOutt}")

    featuress.to_csv(os.path.join(PROJECT_ROOOT,"data","processed",
                                  "ca_clustered_advertisers.csv"),index=False)
    print(f"  Saved: data/processed/ca_clustered_advertisers.csv")

    conn.close()
    print("\nPhase 2 complete.")

    return featuress,final_k,X_scaledd,pipe,kmeansss


if __name__=="__main__":
    run()
