# refine_model.py - phase 6
# refines k-means specifically — winsorization, mini-batch, GMM,
# extended features. DBSCAN is handled properly in phase 3
# (dbscan_clustering.py) so we don't repeat it here.
# goal: find the best possible k-means variant to use as the
# final model, then export clean artifacts for the dashboard team.

import json
import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scrpt_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOOT = os.path.dirname(scrpt_DIR)
DB_PATHH      = os.path.join(PROJECT_ROOOT, "db", "political_ads.db")
FIGSS         = os.path.join(PROJECT_ROOOT, "output", "figures")
DATA_OUTT     = os.path.join(PROJECT_ROOOT, "output", "data")

CLUSTER_FEATSS = [
    "total_spend", "ca_ad_count", "avg_cpi", "avg_ad_duration",
    "pct_video", "pct_text", "pct_image",
    "pct_age_targeted", "pct_gender_targeted", "avg_geo_targets",
    "spend_consistency", "pct_spend_in_peak_week", "pct_spend_ca", "geo_hhi",
    "avg_pct_target_young", "avg_pct_target_middle", "avg_pct_target_older",
    "pct_female_only", "pct_male_only", "has_district_targeting",
]
logColss = ["total_spend", "ca_ad_count", "avg_cpi"]


def _preprocc(X_raw, logIdxss, do_winsorize=False):
    """impute → optionally winsorize → log → scale"""
    imp = SimpleImputer(strategy="median")
    X   = imp.fit_transform(X_raw)

    if do_winsorize:
        from scipy.stats import mstats
        for i in range(X.shape[1]):
            X[:, i] = mstats.winsorize(X[:, i], limits=[0.01, 0.01])

    for i in logIdxss:
        X[:, i] = np.log1p(np.abs(X[:, i]))

    scl = StandardScaler()
    return scl.fit_transform(X)


def _label_archtyp(row, means_df):
    med_spnd = means_df["total_spend"].median()
    med_cv   = means_df["spend_consistency"].median()

    if row["total_spend"] > med_spnd * 2:
        return "Big Spender / Statewide Blitz"
    elif row["pct_spend_in_peak_week"] > 0.5:
        return "Bursty / Event-Driven"
    elif row["spend_consistency"] < med_cv * 0.5:
        return "Steady-State Awareness"
    elif (row["pct_age_targeted"] > 0.5 or row["pct_gender_targeted"] > 0.3):
        return "Targeted Persuader"
    elif row["total_spend"] < med_spnd * 0.5:
        return "Hyper-Local Grassroots"
    else:
        return "Mixed Strategy"


def run(db_path=None, final_k=None):
    if db_path is None:
        db_path = DB_PATHH

    os.makedirs(FIGSS, exist_ok=True)
    os.makedirs(DATA_OUTT, exist_ok=True)

    conn = sqlite3.connect(db_path)
    print("=" * 60)
    print("PHASE 6: K-Means Refinement")
    print("=" * 60)

    featuress = pd.read_sql("SELECT * FROM ca_advertiser_features", conn)
    if final_k is None:
        final_k = featuress["cluster_label"].nunique()

    avail    = [f for f in CLUSTER_FEATSS if f in featuress.columns]
    X_raw    = featuress[avail].values.copy()
    logIdxss = [avail.index(f) for f in logColss if f in avail]

    all_results = []
    report_lines = []

    def log(msg):
        print(msg)
        report_lines.append(msg)

    # ------------------------------------------------------------------ #
    # 6.1  Outlier handling: baseline vs winsorized
    # ------------------------------------------------------------------ #
    log("\n[6.1] Outlier handling — baseline vs winsorized")
    log("-" * 40)

    X_base = _preprocc(X_raw, logIdxss, do_winsorize=False)
    X_wins = _preprocc(X_raw, logIdxss, do_winsorize=True)

    for tag, Xd in [("Baseline (log only)", X_base), ("Winsorized + log", X_wins)]:
        km  = KMeans(n_clusters=final_k, n_init=20, random_state=42)
        lb  = km.fit_predict(Xd)
        s   = silhouette_score(Xd, lb)
        ch  = calinski_harabasz_score(Xd, lb)
        db  = davies_bouldin_score(Xd, lb)
        log(f"  {tag}: Sil={s:.4f}, CH={ch:.1f}, DB={db:.4f}")
        all_results.append({"method": tag, "sil": s, "ch": ch, "db": db,
                            "labels": lb, "X": Xd})

    # ------------------------------------------------------------------ #
    # 6.2  K-Means variants at final_k
    # ------------------------------------------------------------------ #
    log("\n[6.2] K-Means variants")
    log("-" * 40)

    X_cmp = X_base  # compare variants on baseline preprocessing

    # mini-batch — faster, slightly different solution
    mb     = MiniBatchKMeans(n_clusters=final_k, n_init=10, random_state=42, batch_size=256)
    mb_lab = mb.fit_predict(X_cmp)
    mb_s   = silhouette_score(X_cmp, mb_lab)
    mb_ch  = calinski_harabasz_score(X_cmp, mb_lab)
    mb_db  = davies_bouldin_score(X_cmp, mb_lab)
    log(f"  Mini-Batch K-Means: Sil={mb_s:.4f}, CH={mb_ch:.1f}, DB={mb_db:.4f}")
    all_results.append({"method": "Mini-Batch K-Means", "sil": mb_s, "ch": mb_ch,
                        "db": mb_db, "labels": mb_lab, "X": X_cmp})

    # GMM — soft assignment, captures elliptical clusters
    gm     = GaussianMixture(n_components=final_k, n_init=5, random_state=42)
    gm_lab = gm.fit_predict(X_cmp)
    gm_s   = silhouette_score(X_cmp, gm_lab)
    gm_ch  = calinski_harabasz_score(X_cmp, gm_lab)
    gm_db  = davies_bouldin_score(X_cmp, gm_lab)
    log(f"  GMM:                Sil={gm_s:.4f}, CH={gm_ch:.1f}, DB={gm_db:.4f}")
    all_results.append({"method": "GMM", "sil": gm_s, "ch": gm_ch,
                        "db": gm_db, "labels": gm_lab, "X": X_cmp})

    # ------------------------------------------------------------------ #
    # 6.3  Extended feature: spend_per_creative
    # ------------------------------------------------------------------ #
    log("\n[6.3] Feature engineering — spend_per_creative")
    log("-" * 40)

    featuress["spend_per_creative"] = np.where(
        featuress["ca_ad_count"] > 0,
        featuress["total_spend"] / featuress["ca_ad_count"],
        0,
    )
    ext_cols    = avail + ["spend_per_creative"]
    X_ext_raw   = featuress[ext_cols].values.copy()
    ext_log_idx = [ext_cols.index(f) for f in logColss + ["spend_per_creative"]
                   if f in ext_cols]
    X_ext = _preprocc(X_ext_raw, ext_log_idx)

    km_ext     = KMeans(n_clusters=final_k, n_init=20, random_state=42)
    ext_lab    = km_ext.fit_predict(X_ext)
    ext_s      = silhouette_score(X_ext, ext_lab)
    ext_ch     = calinski_harabasz_score(X_ext, ext_lab)
    ext_db     = davies_bouldin_score(X_ext, ext_lab)
    log(f"  K-Means + spend_per_creative: Sil={ext_s:.4f}, CH={ext_ch:.1f}, DB={ext_db:.4f}")
    all_results.append({"method": "K-Means + extended features", "sil": ext_s,
                        "ch": ext_ch, "db": ext_db, "labels": ext_lab, "X": X_ext})

    # ------------------------------------------------------------------ #
    # 6.4  Pick best k-means variant
    # ------------------------------------------------------------------ #
    log("\n[6.4] Refinement summary")
    log("-" * 40)
    log(f"\n  {'Method':<35} {'Silhouette':>12} {'CH Index':>12} {'DB Index':>12}")
    log(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*12}")
    for r in all_results:
        log(f"  {r['method']:<35} {r['sil']:>12.4f} {r['ch']:>12.1f} {r['db']:>12.4f}")

    winner = max(all_results, key=lambda x: x["sil"])
    log(f"\n  Best K-Means variant: {winner['method']} (Silhouette={winner['sil']:.4f})")

    featuress["cluster_label"] = winner["labels"]

    # PCA coordinates for scatter
    pca = PCA(n_components=2)
    xy  = pca.fit_transform(winner["X"])
    featuress["pca_x"] = xy[:, 0]
    featuress["pca_y"] = xy[:, 1]

    # re-label archetypes with winning labels
    c_means   = featuress[avail + ["cluster_label"]].groupby("cluster_label").mean()
    arch_nms  = {}
    for c in sorted(featuress["cluster_label"].unique()):
        arch_nms[c] = _label_archtyp(c_means.loc[c], c_means)

    # dedupe
    seen = {}
    for c in sorted(featuress["cluster_label"].unique()):
        nm = arch_nms[c]
        if nm in seen:
            arch_nms[c] = f"{nm} ({c})"
        seen[nm] = True

    featuress["archetype"] = featuress["cluster_label"].map(arch_nms)

    # ------------------------------------------------------------------ #
    # 6.5  Refined scatter plot
    # ------------------------------------------------------------------ #
    fig, ax  = plt.subplots(figsize=(10, 8))
    exp_var  = pca.explained_variance_ratio_
    for c in sorted(featuress["cluster_label"].unique()):
        msk = (featuress["cluster_label"] == c)
        ax.scatter(featuress.loc[msk, "pca_x"], featuress.loc[msk, "pca_y"],
                   label=f"C{c}: {arch_nms[c]}", alpha=0.6, s=30)
    ax.set_xlabel(f"PC1 ({exp_var[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({exp_var[1]:.1%} variance)")
    ax.set_title(f"Refined CA Political Advertiser Clusters (PCA)\n"
                 f"Best variant: {winner['method']}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(FIGSS, "cluster_scatter_refined.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {FIGSS}/cluster_scatter_refined.png")

    # ------------------------------------------------------------------ #
    # 6.6  Export
    # ------------------------------------------------------------------ #
    log("\n[6.6] Exporting final artifacts")
    log("-" * 40)

    featuress.to_sql("ca_advertiser_features", conn, if_exists="replace", index=False)
    log("  Updated db table: ca_advertiser_features")

    csv_out = os.path.join(DATA_OUTT, "ca_clustered_advertisers.csv")
    featuress.to_csv(csv_out, index=False)
    log(f"  Saved: {csv_out}")

    # load existing json to preserve field descriptions written in phase 2
    json_path = os.path.join(DATA_OUTT, "cluster_summary.json")
    old_json  = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as fh:
                old_json = json.load(fh)
        except Exception:
            pass

    log_sp   = np.log1p(featuress["total_spend"])
    cpi_col  = featuress["avg_cpi"]
    med_sp   = log_sp.median()
    med_cpi  = cpi_col.median()

    new_json = {
        "_metadata": {
            "description": "Final refined cluster profiles (Phase 6). "
                           "Best k-means variant selected after comparing baseline, "
                           "winsorized, mini-batch, GMM, and extended-feature configurations.",
            "best_variant":      winner["method"],
            "best_silhouette":   round(winner["sil"], 4),
            "num_advertisers":   len(featuress),
            "num_clusters":      int(final_k),
            "features_used":     avail,
        },
        "_field_descriptions": old_json.get("_field_descriptions", {}),
    }

    for c in sorted(featuress["cluster_label"].unique()):
        msk   = (featuress["cluster_label"] == c)
        sz    = int(msk.sum())
        means = c_means.loc[c].to_dict()
        top5  = (featuress[msk].nlargest(5, "total_spend")
                 [["Advertiser_Name", "total_spend"]].to_dict("records"))
        c_ls  = log_sp[msk]
        c_cp  = cpi_col[msk]
        new_json[str(c)] = {
            "archetype": arch_nms[c],
            "size": sz,
            "feature_means": {k: round(v, 4) for k, v in means.items()},
            "quadrant_distribution": {
                "high_spend_high_efficiency": int(((c_ls >= med_sp) & (c_cp <  med_cpi)).sum()),
                "high_spend_low_efficiency":  int(((c_ls >= med_sp) & (c_cp >= med_cpi)).sum()),
                "low_spend_high_efficiency":  int(((c_ls <  med_sp) & (c_cp <  med_cpi)).sum()),
                "low_spend_low_efficiency":   int(((c_ls <  med_sp) & (c_cp >= med_cpi)).sum()),
            },
            "top_spenders": top5,
        }

    with open(json_path, "w") as fh:
        json.dump(new_json, fh, indent=2, default=str)
    log(f"  Saved: {json_path}")

    conn.close()
    log("\nPhase 6 complete.")
    return featuress, report_lines


if __name__ == "__main__":
    run()
