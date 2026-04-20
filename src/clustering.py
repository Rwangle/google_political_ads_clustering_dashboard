# clustering.py - phase 3
# K-Means on clean data (outliers from phase 2 excluded).
# k chosen by composite score: silhouette + cluster balance + k-size
# reward — hard floor of K_MIN=4 so we always get meaningful segments.
# Random Forest per cluster finds top distinguishing features.
# RF feature importance used to label clusters (no external API needed).
# Outliers assigned to nearest centroid post-hoc for dashboard toggle.

import json
import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

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

FEAT_LABELS = {
    "total_spend": "total spend", "ca_ad_count": "number of CA ads",
    "avg_cpi": "cost per impression", "avg_ad_duration": "ad run length",
    "pct_video": "video ad share", "pct_text": "text ad share",
    "pct_image": "image ad share", "pct_age_targeted": "age targeting rate",
    "pct_gender_targeted": "gender targeting rate",
    "avg_geo_targets": "geographic breadth",
    "spend_consistency": "spending burstiness",
    "pct_spend_in_peak_week": "peak-week spend share",
    "pct_spend_ca": "CA spend share", "geo_hhi": "geographic concentration",
    "avg_pct_target_young": "youth targeting (18-34)",
    "avg_pct_target_middle": "middle-age targeting (35-54)",
    "avg_pct_target_older": "older-age targeting (55+)",
    "pct_female_only": "female-only targeting",
    "pct_male_only": "male-only targeting",
    "has_district_targeting": "district-level targeting",
}

K_MIN, K_MAX = 4, 8   # hard floor — never fewer than 4 clusters


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


def _composite_k(X, k_range, random_state=42):
    """
    Choose k by composite score:
      50% silhouette  — cluster separation quality
      30% balance     — penalise degenerate splits (<3% per cluster)
      20% k-reward    — log-scaled bonus for larger k
    This prevents k collapsing to 2 while still requiring quality.
    """
    n   = len(X)
    min_sz = max(5, int(n * 0.03))
    rows   = []
    for k in k_range:
        km  = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        lbl = km.fit_predict(X)
        sil = silhouette_score(X, lbl)
        _, cts = np.unique(lbl, return_counts=True)
        balance = float((cts >= min_sz).mean())
        if cts.min() < min_sz:
            balance *= 0.5   # hard penalty for degenerate split
        rows.append({"k": k, "sil": sil, "balance": balance, "labels": lbl})

    df = pd.DataFrame(rows)
    for col in ["sil", "balance"]:
        lo, hi = df[col].min(), df[col].max()
        df[col + "_n"] = (df[col] - lo) / (hi - lo + 1e-9)
    karr = np.array(df["k"].tolist(), dtype=float)
    df["k_rew"] = (np.log(karr) - np.log(karr.min())) / \
                  (np.log(karr.max()) - np.log(karr.min()) + 1e-9)
    df["score"] = 0.50 * df["sil_n"] + 0.30 * df["balance_n"] + 0.20 * df["k_rew"]
    best_k = int(df.loc[df["score"].idxmax(), "k"])
    return best_k, df


def _rf_top_features(X_df, labels, feature_names, top_n=5):
    """Train binary RF for each cluster to find its top features."""
    global_rf = RandomForestClassifier(n_estimators=200, max_depth=6,
                                       random_state=42, n_jobs=-1)
    global_rf.fit(X_df, labels)
    global_imp = dict(zip(feature_names, global_rf.feature_importances_))

    cluster_feats = {}
    for c in sorted(set(labels)):
        binary = (labels == c).astype(int)
        rf_c = RandomForestClassifier(n_estimators=100, max_depth=4,
                                      random_state=42, n_jobs=-1)
        rf_c.fit(X_df, binary)
        imps = sorted(zip(feature_names, rf_c.feature_importances_),
                      key=lambda x: x[1], reverse=True)
        cluster_feats[c] = imps[:top_n]
    return cluster_feats, global_imp


def _build_profile(top_feats, c_means, overall_means, cluster_size, total_size):
    """Structured feature profile dict — used by both API and fallback."""
    profile = []
    for feat, imp in top_feats:
        cv = c_means.get(feat, overall_means.get(feat, 0))
        ov = overall_means.get(feat, 1)
        ratio = cv / ov if ov != 0 else 1.0
        direction = "HIGH" if ratio > 1.15 else ("LOW" if ratio < 0.85 else "AVERAGE")
        profile.append({
            "feature": feat,
            "readable": FEAT_LABELS.get(feat, feat),
            "importance": round(imp, 4),
            "cluster_mean": round(float(cv), 4),
            "overall_mean": round(float(ov), 4),
            "ratio": round(ratio, 2),
            "pct_diff": round((ratio - 1) * 100, 1),
            "direction": direction,
        })
    return {
        "cluster_size": cluster_size,
        "total": total_size,
        "pct_of_total": round(cluster_size / total_size * 100, 1),
        "features": profile,
    }




def _fallback_label(cluster_id, profile):
    """Mechanical label used when API is unavailable."""
    feats = [f for f in profile["features"] if f["direction"] != "AVERAGE"]
    if not feats:
        return (f"Cluster {cluster_id}",
                "No strongly dominant features identified.", [])
    short = " · ".join(
        f"{f['direction'].capitalize()} {f['readable']}" for f in feats[:2]
    )
    desc = "Advertisers characterized by " + ", ".join(
        f"{'high' if f['direction']=='HIGH' else 'low'} {f['readable']} "
        f"({f['ratio']}x avg)" for f in feats[:3]
    ) + "."
    metrics = [
        f"• {f['readable']}: {f['direction'].lower()} "
        f"({f['ratio']}x average, {abs(f['pct_diff']):.0f}% "
        f"{'above' if f['pct_diff']>0 else 'below'} average)"
        for f in feats[:3]
    ]
    return short, desc, metrics


def _label_cluster(cluster_id, top_feats, c_means, overall_means, sz, n_total):
    profile = _build_profile(top_feats, c_means, overall_means, sz, n_total)
    short, desc, metrics = _fallback_label(cluster_id, profile)
    print(f"    [label] {short}")
    return short, desc, metrics, profile


def run(db_path=None):
    if db_path is None:
        db_path = DB_PATHH
    os.makedirs(FIGSS, exist_ok=True)
    os.makedirs(DATA_OUTT, exist_ok=True)
    conn = sqlite3.connect(db_path)

    print("=" * 60)
    print("PHASE 3: K-Means Clustering")
    print("=" * 60)

    featuress = pd.read_sql("SELECT * FROM ca_advertiser_features", conn)
    if "is_outlier" not in featuress.columns:
        print("  WARNING: is_outlier missing — run phase 2 first. Treating all as clean.")
        featuress["is_outlier"] = 0
        featuress["outlier_reasons"] = ""

    df_clean    = featuress[featuress["is_outlier"] == 0].copy().reset_index(drop=True)
    df_outliers = featuress[featuress["is_outlier"] == 1].copy().reset_index(drop=True)
    print(f"  Clean: {len(df_clean)}  Outliers: {len(df_outliers)}")

    avail   = [f for f in FEATSS if f in featuress.columns]
    log_idx = [avail.index(f) for f in logColss if f in avail]
    pipe    = _make_pipeline(log_idx)
    X_clean = pipe.fit_transform(df_clean[avail].values.copy())

    # choose k
    print(f"\n[3.1] Composite k-selection (k={K_MIN}..{K_MAX})...")
    print(f"  Score = 50% silhouette + 30% balance + 20% k-reward")
    print(f"  Hard floor: k >= {K_MIN}. Balance penalty if cluster < 3% of data.")
    k_range = list(range(K_MIN, K_MAX + 1))
    best_k, score_df = _composite_k(X_clean, k_range)

    print(f"\n  {'k':>3} {'Silhouette':>11} {'Balance':>9} {'k-reward':>10} {'Score':>8}")
    print(f"  {'-'*3} {'-'*11} {'-'*9} {'-'*10} {'-'*8}")
    for _, row in score_df.iterrows():
        mark = " <-- chosen" if row["k"] == best_k else ""
        print(f"  {int(row['k']):>3} {row['sil']:>11.4f} {row['balance']:>9.4f} "
              f"{row['k_rew']:>10.4f} {row['score']:>8.4f}{mark}")
    print(f"\n  Chosen k={best_k}")

    # fit final
    print(f"\n[3.2] Fitting K-Means (k={best_k})...")
    km = KMeans(n_clusters=best_k, n_init=30, random_state=42)
    clean_labels = km.fit_predict(X_clean)
    df_clean["kmeans_cluster"] = clean_labels
    _, cts = np.unique(clean_labels, return_counts=True)
    print(f"  Cluster sizes: {dict(enumerate(cts.tolist()))}")
    sil = silhouette_score(X_clean, clean_labels)
    ch  = calinski_harabasz_score(X_clean, clean_labels)
    db  = davies_bouldin_score(X_clean, clean_labels)
    print(f"  Silhouette={sil:.4f}  CH={ch:.1f}  DB={db:.4f}")

    # assign outliers to nearest centroid
    X_out = None
    if len(df_outliers) > 0:
        X_out = pipe.transform(df_outliers[avail].values.copy())
        out_labels = km.predict(X_out)
        df_outliers["kmeans_cluster"]          = out_labels
        df_outliers["kmeans_nearest_centroid"] = out_labels
        print(f"  Assigned {len(df_outliers)} outliers to nearest K-Means centroids")
    df_clean["kmeans_nearest_centroid"] = df_clean["kmeans_cluster"]

    # RF labels
    print("\n[3.3] Random Forest — top features per cluster...")
    X_df = pd.DataFrame(X_clean, columns=avail)
    cluster_rf, global_imp = _rf_top_features(X_df, clean_labels, avail)
    c_means_df   = df_clean.groupby("kmeans_cluster")[avail].mean()
    overall_means = dict(df_clean[avail].mean())

    arch_short, arch_desc, arch_metrics, arch_profiles = {}, {}, {}, {}
    n_total = len(df_clean)

    print("  Labeling clusters using Random Forest feature importance...")
    for c in range(best_k):
        sz   = int((clean_labels == c).sum())
        print(f"\n  Cluster {c} (n={sz}):")
        c_means = dict(c_means_df.loc[c])
        short, desc, metrics, profile = _label_cluster(
            c, cluster_rf[c], c_means, overall_means, sz, n_total)
        arch_short[c]    = short
        arch_desc[c]     = desc
        arch_metrics[c]  = metrics
        arch_profiles[c] = profile
        print(f"  Name: {short}")
        print(f"  Desc: {desc}")
        for m in metrics:
            print(f"    {m}")

    df_clean["kmeans_archetype"] = df_clean["kmeans_cluster"].map(arch_short)
    if len(df_outliers) > 0:
        df_outliers["kmeans_archetype"] = df_outliers["kmeans_cluster"].map(arch_short)

    # k-selection plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(score_df["k"], score_df["sil"], "bo-", linewidth=2)
    ax1.axvline(x=best_k, color="red", linestyle="--", label=f"k={best_k}")
    ax1.set_xlabel("k"); ax1.set_ylabel("Silhouette")
    ax1.set_title("Silhouette Score"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(score_df["k"], score_df["score"], "gs-", linewidth=2)
    ax2.axvline(x=best_k, color="red", linestyle="--", label=f"k={best_k}")
    ax2.set_xlabel("k"); ax2.set_ylabel("Composite Score")
    ax2.set_title("Composite Score (50% sil + 30% balance + 20% k-reward)")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.suptitle(f"K-Means k selection — chosen k={best_k}", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGSS, "kmeans_k_selection.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # PCA scatter
    pca    = PCA(n_components=2)
    xy     = pca.fit_transform(X_clean)
    ev     = pca.explained_variance_ratio_
    colors = plt.cm.tab10(np.linspace(0, 1, best_k))
    fig, ax = plt.subplots(figsize=(12, 9))
    for c in range(best_k):
        m = (clean_labels == c)
        ax.scatter(xy[m, 0], xy[m, 1], color=colors[c], alpha=0.6, s=30,
                   label=f"C{c}: {arch_short[c]} (n={m.sum()})")
    if X_out is not None:
        xy_out = pca.transform(X_out)
        ax.scatter(xy_out[:, 0], xy_out[:, 1], c="lightgray", marker="x",
                   s=50, alpha=0.5, linewidths=1.2,
                   label=f"Outliers (n={len(df_outliers)})")
    ax.set_xlabel(f"PC1 ({ev[0]:.1%})"); ax.set_ylabel(f"PC2 ({ev[1]:.1%})")
    ax.set_title(f"K-Means Clusters — CA Political Advertisers\n(k={best_k}, Sil={sil:.3f})")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGSS, "kmeans_cluster_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # feature heatmap
    top10 = sorted(global_imp.items(), key=lambda x: x[1], reverse=True)[:10]
    top10_names = [f for f, _ in top10]
    hmap = np.zeros((best_k, len(top10_names)))
    for c in range(best_k):
        for fi, feat in enumerate(top10_names):
            cv = c_means_df.loc[c, feat]
            ov = overall_means.get(feat, 1)
            hmap[c, fi] = cv / ov if ov != 0 else 1.0
    fig, ax = plt.subplots(figsize=(14, max(5, best_k * 1.3)))
    im = ax.imshow(hmap, cmap="RdYlGn", aspect="auto", vmin=0, vmax=2)
    ax.set_xticks(range(len(top10_names)))
    ax.set_xticklabels([FEAT_LABELS.get(f, f) for f in top10_names],
                       rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(best_k))
    ax.set_yticklabels([f"C{c}: {arch_short[c]}" for c in range(best_k)], fontsize=9)
    plt.colorbar(im, ax=ax, label="Ratio to overall mean (green=high, red=low)")
    ax.set_title("K-Means Cluster Profiles — Top 10 RF Features\n"
                 "(value / overall mean — green means this cluster is high on that feature)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGSS, "kmeans_cluster_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved figures to {FIGSS}/")

    # merge and save
    df_all = pd.concat([df_clean, df_outliers], ignore_index=True)
    for col in ["kmeans_cluster", "kmeans_nearest_centroid", "kmeans_archetype"]:
        if col in featuress.columns:
            featuress = featuress.drop(columns=[col])
    featuress = featuress.merge(
        df_all[["Advertiser_ID", "kmeans_cluster",
                "kmeans_nearest_centroid", "kmeans_archetype"]],
        on="Advertiser_ID", how="left"
    )
    featuress.to_sql("ca_advertiser_features", conn, if_exists="replace", index=False)

    # JSON summary
    summary = {
        "_metadata": {
            "method": "K-Means",
            "k": best_k,
            "k_selection": "Composite score (50% silhouette + 30% balance + 20% k-reward)",
            "k_min_floor": K_MIN,
            "silhouette": round(sil, 4),
            "calinski_harabasz": round(ch, 1),
            "davies_bouldin": round(db, 4),
            "n_clean": len(df_clean),
            "n_outliers": len(df_outliers),
            "features_used": avail,
        }
    }
    for c in range(best_k):
        m    = df_clean["kmeans_cluster"] == c
        top5 = (df_clean[m].nlargest(5, "total_spend")
                [["Advertiser_Name", "total_spend"]].to_dict("records"))
        summary[str(c)] = {
            "cluster_id": c,
            "archetype_name": arch_short[c],
            "archetype_description": arch_desc[c],
            "key_metrics_display": arch_metrics[c],
            "size_clean": int(m.sum()),
            "size_with_outliers": int((df_all["kmeans_cluster"] == c).sum()),
            "top_rf_features": [
                {"feature": f, "readable": FEAT_LABELS.get(f, f),
                 "importance": round(imp, 4),
                 "cluster_mean": round(float(c_means_df.loc[c, f]), 4),
                 "overall_mean": round(float(overall_means.get(f, 0)), 4),
                 "ratio": round(float(c_means_df.loc[c, f]) /
                                float(overall_means.get(f, 1)), 2)
                           if overall_means.get(f, 0) != 0 else 1.0}
                for f, imp in cluster_rf[c]
            ],
            "top_spenders": top5,
        }
    with open(os.path.join(DATA_OUTT, "kmeans_cluster_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    conn.close()
    print("\nPhase 3 complete.")
    return featuress, best_k, X_clean, pipe, km, arch_short


if __name__ == "__main__":
    run()
