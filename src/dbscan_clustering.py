# dbscan_clustering.py - phase 5a
# density-based clustering of CA political advertisers
# DBSCAN finds "noise" advertisers that don't fit any cluster —
# which is actually meaningful here (one-off PACs, test accounts, etc.)
# unlike kmeans, we don't pick k; instead we tune eps + min_samples

import json
import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

scrpt_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOOT = os.path.dirname(scrpt_DIR)
DB_PATHH      = os.path.join(PROJECT_ROOOT, "db", "political_ads.db")
FIGSS         = os.path.join(PROJECT_ROOOT, "output", "figures")
DATA_OUTT     = os.path.join(PROJECT_ROOOT, "output", "data")

# same feature list as clustering.py / refine_model.py
FEATSS = [
    "total_spend", "ca_ad_count", "avg_cpi", "avg_ad_duration",
    "pct_video", "pct_text", "pct_image",
    "pct_age_targeted", "pct_gender_targeted", "avg_geo_targets",
    "spend_consistency", "pct_spend_in_peak_week", "pct_spend_ca", "geo_hhi",
    "avg_pct_target_young", "avg_pct_target_middle", "avg_pct_target_older",
    "pct_female_only", "pct_male_only", "has_district_targeting",
]
logColss = ["total_spend", "ca_ad_count", "avg_cpi"]


def _make_pipeline(logIdxx):
    """same preprocessing as clustering.py: impute → log-transform → scale"""
    def _doLog(X):
        out = X.copy()
        for i in logIdxx:
            out[:, i] = np.log1p(np.abs(out[:, i]))
        return out

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log",    FunctionTransformer(_doLog, validate=False)),
        ("scale",  StandardScaler()),
    ])
    return pipe


def _k_dist_plot(X_scaled, k, save_path):
    """
    k-distance plot: sort distances to the k-th nearest neighbor.
    the 'elbow' in this curve is a good starting point for eps.
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
    dists, _ = nbrs.kneighbors(X_scaled)
    k_dists = np.sort(dists[:, k - 1])[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(k_dists, linewidth=1.5, color="steelblue")
    ax.set_xlabel("Points (sorted by distance)", fontsize=11)
    ax.set_ylabel(f"Distance to {k}th nearest neighbor", fontsize=11)
    ax.set_title(f"k-Distance Plot (k={k}) — Elbow ≈ good eps value", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _label_cluster(row, cluster_means):
    """same naming logic as clustering.py"""
    med_spend = cluster_means["total_spend"].median()
    med_cv    = cluster_means["spend_consistency"].median()

    if row["total_spend"] > med_spend * 2:
        return "Big Spender / Statewide Blitz"
    elif row["pct_spend_in_peak_week"] > 0.5:
        return "Bursty / Event-Driven"
    elif row["spend_consistency"] < med_cv * 0.5:
        return "Steady-State Awareness"
    elif (row["pct_age_targeted"] > 0.5 or row["pct_gender_targeted"] > 0.3):
        return "Targeted Persuader"
    elif row["total_spend"] < med_spend * 0.5:
        return "Hyper-Local Grassroots"
    else:
        return "Mixed Strategy"


def _grid_search(X_scaled, eps_vals, min_samples_vals):
    """
    Brute-force grid search over eps × min_samples.
    Returns a DataFrame of results sorted by silhouette score.
    We skip configs that produce < 2 clusters or > 50% noise
    (too degenerate to be useful).
    """
    rows = []
    for eps in eps_vals:
        for ms in min_samples_vals:
            db = DBSCAN(eps=eps, min_samples=ms)
            labels = db.fit_predict(X_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise    = int((labels == -1).sum())
            pct_noise  = n_noise / len(labels)

            if n_clusters < 2 or pct_noise > 0.50:
                rows.append({
                    "eps": eps, "min_samples": ms,
                    "n_clusters": n_clusters, "n_noise": n_noise,
                    "pct_noise": round(pct_noise, 3),
                    "silhouette": np.nan, "ch_index": np.nan, "db_index": np.nan,
                    "note": "skipped (too few clusters or >50% noise)",
                })
                continue

            mask = (labels != -1)
            try:
                sil = silhouette_score(X_scaled[mask], labels[mask])
                ch  = calinski_harabasz_score(X_scaled[mask], labels[mask])
                dbi = davies_bouldin_score(X_scaled[mask], labels[mask])
            except Exception as e:
                rows.append({
                    "eps": eps, "min_samples": ms,
                    "n_clusters": n_clusters, "n_noise": n_noise,
                    "pct_noise": round(pct_noise, 3),
                    "silhouette": np.nan, "ch_index": np.nan, "db_index": np.nan,
                    "note": f"metric error: {e}",
                })
                continue

            rows.append({
                "eps": eps, "min_samples": ms,
                "n_clusters": n_clusters, "n_noise": n_noise,
                "pct_noise": round(pct_noise, 3),
                "silhouette": round(sil, 4),
                "ch_index": round(ch, 1),
                "db_index": round(dbi, 4),
                "note": "",
            })

    df = pd.DataFrame(rows)
    return df.sort_values("silhouette", ascending=False)


def run(db_path=None):
    if db_path is None:
        db_path = DB_PATHH

    os.makedirs(FIGSS, exist_ok=True)
    os.makedirs(DATA_OUTT, exist_ok=True)

    conn = sqlite3.connect(db_path)
    print("=" * 60)
    print("PHASE 5a: DBSCAN Clustering")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 5a.1  Load + preprocess
    # ------------------------------------------------------------------ #
    print("\n[5a.1] Loading features and preprocessing...")
    featuress = pd.read_sql("SELECT * FROM ca_advertiser_features", conn)
    print(f"  Loaded {len(featuress)} advertisers")

    avail = [f for f in FEATSS if f in featuress.columns]
    missing = [f for f in FEATSS if f not in featuress.columns]
    if missing:
        print(f"  WARNING: Missing features (skipped): {missing}")
    print(f"  Using {len(avail)} features")

    X_raw   = featuress[avail].values.copy()
    logIdxx = [avail.index(f) for f in logColss if f in avail]
    pipe    = _make_pipeline(logIdxx)
    X_scaled = pipe.fit_transform(X_raw)
    print(f"  Preprocessed shape: {X_scaled.shape}")

    # ------------------------------------------------------------------ #
    # 5a.2  k-distance plot to guide eps selection
    # ------------------------------------------------------------------ #
    print("\n[5a.2] Generating k-distance plot (k=5)...")
    kdist_path = os.path.join(FIGSS, "dbscan_kdist_plot.png")
    _k_dist_plot(X_scaled, k=5, save_path=kdist_path)
    print(f"  Saved: {kdist_path}")
    print("  Inspect the elbow to calibrate eps for your data.")

    # ------------------------------------------------------------------ #
    # 5a.3  Grid search over eps × min_samples
    # ------------------------------------------------------------------ #
    print("\n[5a.3] Grid search: eps × min_samples...")
    eps_vals        = [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    min_samples_vals = [3, 5, 7, 10, 15]

    grid_df = _grid_search(X_scaled, eps_vals, min_samples_vals)

    print("\n  Top 10 valid configurations by silhouette:")
    valid = grid_df[grid_df["silhouette"].notna()].head(10)
    if len(valid) == 0:
        print("  WARNING: No valid DBSCAN config found. Data may be too uniformly spread.")
        conn.close()
        return None, []

    print(f"  {'eps':>6} {'min_s':>6} {'k':>5} {'noise':>7} {'%noise':>8} "
          f"{'silhouette':>12} {'CH':>10} {'DB':>10}")
    print(f"  {'-'*6} {'-'*6} {'-'*5} {'-'*7} {'-'*8} {'-'*12} {'-'*10} {'-'*10}")
    for _, r in valid.iterrows():
        print(f"  {r['eps']:>6.2f} {r['min_samples']:>6.0f} {r['n_clusters']:>5.0f} "
              f"{r['n_noise']:>7.0f} {r['pct_noise']:>8.1%} "
              f"{r['silhouette']:>12.4f} {r['ch_index']:>10.1f} {r['db_index']:>10.4f}")

    # save grid to csv for inspection
    grid_csv = os.path.join(DATA_OUTT, "dbscan_grid_search.csv")
    grid_df.to_csv(grid_csv, index=False)
    print(f"\n  Full grid saved: {grid_csv}")

    # ------------------------------------------------------------------ #
    # 5a.4  Fit best config
    # ------------------------------------------------------------------ #
    best = valid.iloc[0]
    best_eps = best["eps"]
    best_ms  = int(best["min_samples"])
    print(f"\n[5a.4] Fitting best config: eps={best_eps}, min_samples={best_ms}")
    print(f"  Expected: {int(best['n_clusters'])} clusters, "
          f"{int(best['n_noise'])} noise points ({best['pct_noise']:.1%})")

    db_final = DBSCAN(eps=best_eps, min_samples=best_ms)
    labels   = db_final.fit_predict(X_scaled)

    featuress["dbscan_label"] = labels

    cluster_ids  = sorted(set(labels) - {-1})
    n_clusters   = len(cluster_ids)
    n_noise      = int((labels == -1).sum())
    pct_noise    = n_noise / len(labels)

    print(f"  Clusters found: {n_clusters}")
    print(f"  Noise points:   {n_noise} ({pct_noise:.1%})")

    for c in cluster_ids:
        sz = int((labels == c).sum())
        print(f"    Cluster {c}: {sz} advertisers")

    # ------------------------------------------------------------------ #
    # 5a.5  Cluster profiling + archetype labels
    # ------------------------------------------------------------------ #
    print("\n[5a.5] Profiling clusters...")

    mask_valid   = (labels != -1)
    sil_final    = silhouette_score(X_scaled[mask_valid], labels[mask_valid])
    ch_final     = calinski_harabasz_score(X_scaled[mask_valid], labels[mask_valid])
    db_final_idx = davies_bouldin_score(X_scaled[mask_valid], labels[mask_valid])
    print(f"  Silhouette:        {sil_final:.4f}")
    print(f"  Calinski-Harabasz: {ch_final:.1f}")
    print(f"  Davies-Bouldin:    {db_final_idx:.4f}")

    cluster_means = (featuress[featuress["dbscan_label"] != -1][avail + ["dbscan_label"]]
                     .groupby("dbscan_label").mean())

    archetype_map = {}
    for c in cluster_ids:
        archetype_map[c] = _label_cluster(cluster_means.loc[c], cluster_means)
    archetype_map[-1] = "Noise / Outlier"

    # dedupe archetype names
    seen = {}
    for c in cluster_ids:
        nm = archetype_map[c]
        if nm in seen:
            archetype_map[c] = f"{nm} ({c})"
        seen[nm] = True

    featuress["dbscan_archetype"] = featuress["dbscan_label"].map(archetype_map)

    # ------------------------------------------------------------------ #
    # 5a.6  Visualizations
    # ------------------------------------------------------------------ #
    print("\n[5a.6] Generating visualizations...")

    # PCA scatter
    pca = PCA(n_components=2)
    xy  = pca.fit_transform(X_scaled)
    featuress["dbscan_pca_x"] = xy[:, 0]
    featuress["dbscan_pca_y"] = xy[:, 1]

    fig, ax = plt.subplots(figsize=(11, 8))
    exp_var = pca.explained_variance_ratio_

    # noise points first (gray, behind)
    noise_mask = (featuress["dbscan_label"] == -1)
    ax.scatter(featuress.loc[noise_mask, "dbscan_pca_x"],
               featuress.loc[noise_mask, "dbscan_pca_y"],
               c="lightgray", marker="x", s=20, alpha=0.5,
               label=f"Noise ({n_noise} pts)")

    colors = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 1)))
    for i, c in enumerate(cluster_ids):
        m = (featuress["dbscan_label"] == c)
        ax.scatter(featuress.loc[m, "dbscan_pca_x"],
                   featuress.loc[m, "dbscan_pca_y"],
                   color=colors[i], alpha=0.7, s=35,
                   label=f"C{c}: {archetype_map[c]} (n={m.sum()})")

    ax.set_xlabel(f"PC1 ({exp_var[0]:.1%} variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({exp_var[1]:.1%} variance)", fontsize=11)
    ax.set_title(f"DBSCAN Clusters — CA Political Advertisers\n"
                 f"(eps={best_eps}, min_samples={best_ms}, "
                 f"Silhouette={sil_final:.3f})", fontsize=12)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    scatter_path = os.path.join(FIGSS, "dbscan_cluster_scatter.png")
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {scatter_path}")

    # cluster size bar chart (including noise)
    all_labels_sorted = cluster_ids + [-1]
    bar_labels = [archetype_map[c] + f"\n(C{c})" if c != -1 else "Noise\n(outliers)"
                  for c in all_labels_sorted]
    bar_counts = [(featuress["dbscan_label"] == c).sum() for c in all_labels_sorted]
    bar_colors = [plt.cm.tab10(i / max(n_clusters, 1)) for i in range(n_clusters)] + ["lightgray"]

    fig, ax = plt.subplots(figsize=(max(8, n_clusters * 2), 5))
    bars = ax.bar(bar_labels, bar_counts, color=bar_colors, edgecolor="white", linewidth=0.8)
    for bar, cnt in zip(bars, bar_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(cnt), ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Number of Advertisers")
    ax.set_title("DBSCAN Cluster Sizes (incl. Noise)", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    bar_path = os.path.join(FIGSS, "dbscan_cluster_sizes.png")
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {bar_path}")

    # spend vs CPI, coloured by DBSCAN cluster
    log_spend = np.log1p(featuress["total_spend"])
    cpi_vals  = featuress["avg_cpi"]
    med_ls    = log_spend.median()
    med_cpi   = cpi_vals.median()

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(log_spend[noise_mask], cpi_vals[noise_mask],
               c="lightgray", marker="x", s=20, alpha=0.4, label="Noise")
    for i, c in enumerate(cluster_ids):
        m = (featuress["dbscan_label"] == c)
        ax.scatter(log_spend[m], cpi_vals[m], color=colors[i], alpha=0.6, s=30,
                   label=f"C{c}: {archetype_map[c]}")
    ax.axvline(x=med_ls,  color="gray", linestyle="--", alpha=0.6)
    ax.axhline(y=med_cpi, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("Total Spend (log scale)", fontsize=11)
    ax.set_ylabel("Avg Cost Per Impression ($)", fontsize=11)
    ax.set_title("DBSCAN — Spend vs. Efficiency Quadrant", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ticks_raw = [100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
    tick_lbls = ["$100", "$1K", "$10K", "$100K", "$1M", "$10M", "$100M"]
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([np.log1p(s) for s in ticks_raw])
    ax2.set_xticklabels(tick_lbls, fontsize=8)
    ax2.set_xlabel("Actual Spend ($)", fontsize=9)
    fig.tight_layout()
    quad_path = os.path.join(FIGSS, "dbscan_spend_efficiency.png")
    fig.savefig(quad_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {quad_path}")

    # ------------------------------------------------------------------ #
    # 5a.7  Export
    # ------------------------------------------------------------------ #
    print("\n[5a.7] Exporting results...")

    featuress.to_sql("ca_advertiser_features", conn, if_exists="replace", index=False)
    print("  Updated db table: ca_advertiser_features")

    csv_out = os.path.join(DATA_OUTT, "dbscan_clustered_advertisers.csv")
    featuress.to_csv(csv_out, index=False)
    print(f"  Saved: {csv_out}")

    summary = {
        "_metadata": {
            "method": "DBSCAN",
            "eps": best_eps,
            "min_samples": best_ms,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "pct_noise": round(pct_noise, 4),
            "silhouette": round(sil_final, 4),
            "calinski_harabasz": round(ch_final, 1),
            "davies_bouldin": round(db_final_idx, 4),
            "features_used": avail,
            "note": (
                "DBSCAN does not require specifying k. eps and min_samples were "
                "selected by grid search maximising silhouette on non-noise points. "
                "Noise points (label=-1) are advertisers too isolated to belong to "
                "any dense cluster — often one-off PACs or test campaigns."
            ),
        },
    }

    for c in cluster_ids:
        m = (featuress["dbscan_label"] == c)
        means = cluster_means.loc[c].to_dict()
        top5  = (featuress[m].nlargest(5, "total_spend")
                 [["Advertiser_Name", "total_spend"]].to_dict("records"))
        summary[str(c)] = {
            "archetype": archetype_map[c],
            "size": int(m.sum()),
            "feature_means": {k: round(v, 4) for k, v in means.items()},
            "top_spenders": top5,
        }

    summary["-1"] = {
        "archetype": "Noise / Outlier",
        "size": n_noise,
        "note": "Points not assigned to any cluster. Low-density / isolated advertisers.",
    }

    json_out = os.path.join(DATA_OUTT, "dbscan_cluster_summary.json")
    with open(json_out, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"  Saved: {json_out}")

    conn.close()
    print("\nPhase 5a complete.")

    metrics = {
        "silhouette": sil_final,
        "calinski_harabasz": ch_final,
        "davies_bouldin": db_final_idx,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "best_eps": best_eps,
        "best_min_samples": best_ms,
    }
    return featuress, metrics


if __name__ == "__main__":
    run()
