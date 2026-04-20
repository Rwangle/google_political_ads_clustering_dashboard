# validation.py - phase 5
# Compares K-Means vs Hierarchical on clean data.
# Flags per-advertiser agreement between both methods.
# Exports tableau_dashboard.csv — single file for Tableau with:
#   - is_outlier, outlier_reasons  (from phase 2)
#   - kmeans_cluster, kmeans_archetype, kmeans_nearest_centroid
#   - hier_cluster, hier_archetype, hier_nearest_centroid
#   - methods_agree (1 if both methods agree, 0 if they differ)
#   - display_cluster, display_archetype (dashboard default view)
# Outlier archetypes are prefixed "Outlier: reason" for clear display.

import json
import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (adjusted_rand_score, silhouette_score,
                              calinski_harabasz_score, davies_bouldin_score)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from scipy.optimize import linear_sum_assignment

scrpt_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOOT = os.path.dirname(scrpt_DIR)
DB_PATHH      = os.path.join(PROJECT_ROOOT, "db", "political_ads.db")
FIGSS         = os.path.join(PROJECT_ROOOT, "output", "figures")
DATA_OUTT     = os.path.join(PROJECT_ROOOT, "output", "data")

FEATSS = [
    "total_spend", "ca_ad_count", "avg_cpi", "avg_ad_duration",
    "pct_video", "pct_text", "pct_image",
    "pct_age_targeted", "pct_gender_targeted", "avg_geo_targets",
    "spend_consistency", "pct_spend_in_peak_week", "pct_spend_ca", "geo_hhi",
    "avg_pct_target_young", "avg_pct_target_middle", "avg_pct_target_older",
    "pct_female_only", "pct_male_only", "has_district_targeting",
]
logColss = ["total_spend", "ca_ad_count", "avg_cpi"]


def _make_pipeline(log_idx):
    def _log(X):
        out = X.copy()
        for i in log_idx:
            out[:, i] = np.log1p(np.abs(out[:, i]))
        return out
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log",    FunctionTransformer(_log, validate=False)),
        ("scale",  StandardScaler()),
    ])


def run(db_path=None):
    if db_path is None:
        db_path = DB_PATHH
    os.makedirs(FIGSS, exist_ok=True)
    os.makedirs(DATA_OUTT, exist_ok=True)
    conn = sqlite3.connect(db_path)

    print("=" * 60)
    print("PHASE 5: Validation & Comparison")
    print("=" * 60)

    featuress = pd.read_sql("SELECT * FROM ca_advertiser_features", conn)
    avail   = [f for f in FEATSS if f in featuress.columns]
    log_idx = [avail.index(f) for f in logColss if f in avail]
    pipe    = _make_pipeline(log_idx)

    is_out_col = featuress["is_outlier"] if "is_outlier" in featuress.columns \
                 else pd.Series(0, index=featuress.index)
    df_clean = featuress[is_out_col == 0].copy().reset_index(drop=True)
    X_clean  = pipe.fit_transform(df_clean[avail].values.copy())

    km_labels   = df_clean["kmeans_cluster"].values if "kmeans_cluster" in df_clean.columns else None
    hier_labels = df_clean["hier_cluster"].values   if "hier_cluster"   in df_clean.columns else None

    report = []
    def log(msg):
        print(msg); report.append(msg)

    # 5.1 metrics
    log("\n[5.1] Cluster quality — clean data")
    log("-" * 50)
    log(f"  {'Method':<20} {'Silhouette':>12} {'CH':>10} {'DB':>10} {'k':>5}")
    log(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*10} {'-'*5}")

    metrics = {}
    for name, labels in [("K-Means", km_labels), ("Hierarchical", hier_labels)]:
        if labels is None:
            log(f"  {name:<20} not available")
            continue
        _, cts = np.unique(labels, return_counts=True)
        degen  = int(cts.min()) < max(5, int(len(labels) * 0.01))
        sil = silhouette_score(X_clean, labels)
        ch  = calinski_harabasz_score(X_clean, labels)
        dbi = davies_bouldin_score(X_clean, labels)
        k   = len(set(labels))
        flag = " [DEGENERATE]" if degen else ""
        log(f"  {name:<20} {sil:>12.4f} {ch:>10.1f} {dbi:>10.4f} {k:>5}{flag}")
        metrics[name] = {"silhouette": sil, "ch": ch, "db": dbi,
                         "k": k, "degenerate": degen}

    # 5.2 ARI
    ari = None
    pct_agree = None
    df_clean["methods_agree"] = 1

    if km_labels is not None and hier_labels is not None:
        ari = adjusted_rand_score(km_labels, hier_labels)
        log(f"\n[5.2] Cross-algorithm agreement")
        log(f"  ARI (K-Means vs Hierarchical): {ari:.4f}")
        if   ari > 0.7: log("  Interpretation: STRONG — both methods find same structure")
        elif ari > 0.4: log("  Interpretation: MODERATE — broad structure shared")
        else:           log("  Interpretation: WEAK — methods disagree, show both in dashboard")

        # remap hier to best matching km labels for per-advertiser agreement
        k = len(set(km_labels))
        cost = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                cost[i, j] = -np.sum((km_labels == i) & (hier_labels == j))
        row_ind, col_ind = linear_sum_assignment(cost)
        hier_r = hier_labels.copy()
        for r, c in zip(row_ind, col_ind):
            hier_r[hier_labels == c] = r

        agree = (km_labels == hier_r)
        df_clean["methods_agree"] = agree.astype(int)
        pct_agree = float(agree.mean())
        log(f"  Advertisers where both agree: {agree.sum()} ({pct_agree:.1%})")
        log(f"  Advertisers where they differ: {(~agree).sum()} ({1-pct_agree:.1%})")
        log("  High-agreement = most reliable cluster assignment for dashboard.")

    # 5.3 comparison bar chart
    if len(metrics) == 2:
        names    = list(metrics.keys())
        sil_vals = [metrics[n]["silhouette"] for n in names]
        fig, ax  = plt.subplots(figsize=(8, 5))
        bars = ax.bar(names, sil_vals, color=["steelblue", "darkorange"], edgecolor="white")
        for bar, val in zip(bars, sil_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.set_ylabel("Silhouette Score (higher = better)")
        ax.set_title("K-Means vs Hierarchical — Silhouette\n"
                     "(both shown in dashboard, user selects method)")
        ax.set_ylim(0, max(sil_vals) * 1.25)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGSS, "method_comparison.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"\n  Saved: {FIGSS}/method_comparison.png")

    # 5.4 merge agreement back
    featuress = featuress.drop(columns=["methods_agree"], errors="ignore")
    featuress = featuress.merge(
        df_clean[["Advertiser_ID", "methods_agree"]],
        on="Advertiser_ID", how="left"
    )
    featuress["methods_agree"] = featuress["methods_agree"].fillna(1).astype(int)
    # outliers: methods_agree=1 by convention (not a disagreement)
    if "is_outlier" in featuress.columns:
        featuress.loc[featuress["is_outlier"] == 1, "methods_agree"] = 1

    # 5.5 build display columns for Tableau
    # display_label = what the dashboard shows by default
    # for normal advertisers: kmeans_archetype
    # for outliers: "Outlier" + their top reason
    featuress["display_cluster"]   = featuress.get("kmeans_cluster", pd.Series(dtype=int))
    featuress["display_archetype"] = featuress.get("kmeans_archetype", pd.Series(dtype=str))

    if "is_outlier" in featuress.columns and "outlier_reasons" in featuress.columns:
        out_mask = featuress["is_outlier"] == 1
        featuress.loc[out_mask, "display_cluster"]   = -1
        featuress.loc[out_mask, "display_archetype"] = (
            "Outlier: " +
            featuress.loc[out_mask, "outlier_reasons"]
                     .fillna("unusual metrics")
                     .str.split(";").str[0]
                     .str.strip()
        )
        # also prefix kmeans/hier archetype for outliers
        for col in ["kmeans_archetype", "hier_archetype"]:
            if col in featuress.columns:
                featuress.loc[out_mask, col] = (
                    "Outlier: " + featuress.loc[out_mask, col].fillna(""))

    # 5.6 export Tableau CSV
    log("\n[5.6] Exporting tableau_dashboard.csv...")
    dashboard_cols = [
        "Advertiser_ID", "Advertiser_Name",
        "total_spend", "ca_ad_count", "avg_cpi", "avg_ad_duration",
        "pct_video", "pct_text", "pct_image",
        "pct_age_targeted", "pct_gender_targeted", "avg_geo_targets",
        "spend_consistency", "pct_spend_in_peak_week", "pct_spend_ca", "geo_hhi",
        "avg_pct_target_young", "avg_pct_target_middle", "avg_pct_target_older",
        "pct_female_only", "pct_male_only", "has_district_targeting",
        "is_outlier", "outlier_reasons",
        "kmeans_cluster", "kmeans_archetype", "kmeans_nearest_centroid",
        "hier_cluster", "hier_archetype", "hier_nearest_centroid",
        "methods_agree",
        "display_cluster", "display_archetype",
    ]
    export_cols = [c for c in dashboard_cols if c in featuress.columns]
    tableau_df  = featuress[export_cols].copy()

    if "is_outlier" not in tableau_df.columns:
        tableau_df["is_outlier"] = 0
    if "outlier_reasons" not in tableau_df.columns:
        tableau_df["outlier_reasons"] = ""

    csv_path = os.path.join(DATA_OUTT, "tableau_dashboard.csv")
    tableau_df.to_csv(csv_path, index=False)
    log(f"  Saved: {csv_path}")
    log(f"  Rows: {len(tableau_df)}  Columns: {len(tableau_df.columns)}")
    log(f"\n  Columns in Tableau file:")
    for col in tableau_df.columns:
        log(f"    {col}")

    # save metrics
    featuress.to_sql("ca_advertiser_features", conn, if_exists="replace", index=False)
    metrics_out = {
        "kmeans":      metrics.get("K-Means", {}),
        "hierarchical": metrics.get("Hierarchical", {}),
        "ari":          ari,
        "pct_agreement": pct_agree,
    }
    with open(os.path.join(DATA_OUTT, "validation_metrics.json"), "w") as fh:
        json.dump(metrics_out, fh, indent=2, default=str)

    conn.close()
    print("\nPhase 5 complete.")
    return metrics_out, report


if __name__ == "__main__":
    run()
