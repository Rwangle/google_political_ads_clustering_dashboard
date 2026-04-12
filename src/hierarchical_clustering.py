# hierarchical_clustering.py - phase 4
# Ward linkage on clean data using same k as K-Means (phase 3).
# Outliers assigned to nearest centroid via euclidean distance.
# Same RF + API labeling as K-Means so names are comparable.
# Average linkage excluded — prone to degenerate 1-vs-all splits.

import json
import os
import sqlite3
import numpy as np
import pandas as pd
import urllib.request
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist

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


def _rf_top_features(X_df, labels, feature_names, top_n=5):
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


def _api_label(cluster_id, profile):
    lines = "\n".join(
        f"  - {f['readable']}: {f['direction']} "
        f"({f['ratio']}x avg, RF importance={f['importance']})"
        for f in profile["features"]
    )
    prompt = (
        f"You are analyzing California political advertising data.\n"
        f"A hierarchical cluster contains {profile['cluster_size']} advertisers "
        f"({profile['pct_of_total']}% of the dataset).\n\n"
        f"Random Forest identified these distinguishing features:\n{lines}\n\n"
        f"Provide:\n"
        f"1. SHORT NAME (3-5 words) describing the advertising STRATEGY.\n"
        f"2. DESCRIPTION (2 sentences max) for a non-technical dashboard user.\n"
        f"3. KEY METRICS — exactly 3 bullet points: '• [metric]: [meaning]'\n\n"
        f"Respond ONLY with valid JSON, no markdown:\n"
        f'{{ "short_name": "...", "description": "...", '
        f'"key_metrics": ["• ...", "• ...", "• ..."] }}'
    )
    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 400,
        "messages": [{"role": "user", "content": prompt}]
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data["content"][0]["text"].strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        return (result.get("short_name", f"Cluster {cluster_id}"),
                result.get("description", ""),
                result.get("key_metrics", []))
    except Exception as e:
        print(f"    [API failed for cluster {cluster_id}: {e}]")
        return None


def _build_profile(top_feats, c_means, overall_means, cluster_size, total_size):
    profile = []
    for feat, imp in top_feats:
        cv = c_means.get(feat, overall_means.get(feat, 0))
        ov = overall_means.get(feat, 1)
        ratio = cv / ov if ov != 0 else 1.0
        direction = "HIGH" if ratio > 1.15 else ("LOW" if ratio < 0.85 else "AVERAGE")
        profile.append({
            "feature": feat, "readable": FEAT_LABELS.get(feat, feat),
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


def _label_cluster(cluster_id, top_feats, c_means, overall_means, sz, n_total):
    profile = _build_profile(top_feats, c_means, overall_means, sz, n_total)
    result  = _api_label(cluster_id, profile)
    if result is not None:
        short, desc, metrics = result
        print(f"    [API] {short}")
        return short, desc, metrics
    feats = [f for f in profile["features"] if f["direction"] != "AVERAGE"]
    if not feats:
        return f"Cluster {cluster_id}", "No dominant features.", []
    short = " · ".join(
        f"{f['direction'].capitalize()} {f['readable']}" for f in feats[:2])
    desc = "Advertisers characterized by " + ", ".join(
        f"{'high' if f['direction']=='HIGH' else 'low'} {f['readable']} "
        f"({f['ratio']}x avg)" for f in feats[:3]) + "."
    metrics = [
        f"• {f['readable']}: {f['direction'].lower()} ({f['ratio']}x average)"
        for f in feats[:3]
    ]
    print(f"    [fallback] {short}")
    return short, desc, metrics


def run(db_path=None, k=None):
    if db_path is None:
        db_path = DB_PATHH
    os.makedirs(FIGSS, exist_ok=True)
    os.makedirs(DATA_OUTT, exist_ok=True)
    conn = sqlite3.connect(db_path)

    print("=" * 60)
    print("PHASE 4: Hierarchical (Ward) Clustering")
    print("=" * 60)

    featuress = pd.read_sql("SELECT * FROM ca_advertiser_features", conn)
    if "is_outlier" not in featuress.columns:
        featuress["is_outlier"] = 0

    df_clean    = featuress[featuress["is_outlier"] == 0].copy().reset_index(drop=True)
    df_outliers = featuress[featuress["is_outlier"] == 1].copy().reset_index(drop=True)
    print(f"  Clean: {len(df_clean)}  Outliers: {len(df_outliers)}")

    if k is None:
        k = int(featuress["kmeans_cluster"].nunique()) if "kmeans_cluster" in featuress.columns else 5
        print(f"  Using k={k} from K-Means phase")

    avail   = [f for f in FEATSS if f in featuress.columns]
    log_idx = [avail.index(f) for f in logColss if f in avail]
    pipe    = _make_pipeline(log_idx)
    X_clean = pipe.fit_transform(df_clean[avail].values.copy())

    # dendrogram
    print(f"\n[4.1] Ward linkage dendrogram (k={k})...")
    Z = linkage(X_clean, method="ward")
    merge_heights = sorted(Z[:, 2], reverse=True)
    cut_h = None
    if k - 1 < len(merge_heights):
        cut_h = (merge_heights[k - 2] + merge_heights[k - 1]) / 2

    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, ax=ax, truncate_mode="lastp", p=40,
               leaf_rotation=90, leaf_font_size=8, show_contracted=True,
               color_threshold=0)
    if cut_h is not None:
        ax.axhline(y=cut_h, color="red", linestyle="--", linewidth=1.5,
                   label=f"Cut for k={k}")
        ax.legend(fontsize=10)
    ax.set_xlabel("Advertisers / Merged Groups")
    ax.set_ylabel("Ward Merge Distance")
    ax.set_title(f"Dendrogram — Ward Linkage (truncated top 40 merges, k={k})")
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGSS, "hierarchical_dendrogram.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGSS}/hierarchical_dendrogram.png")

    # fit
    print(f"\n[4.2] Fitting AgglomerativeClustering (Ward, k={k})...")
    agg          = AgglomerativeClustering(n_clusters=k, linkage="ward")
    clean_labels = agg.fit_predict(X_clean)
    df_clean["hier_cluster"] = clean_labels

    _, cts = np.unique(clean_labels, return_counts=True)
    min_sz = int(cts.min())
    if min_sz < max(5, int(len(clean_labels) * 0.01)):
        print(f"  WARNING: smallest cluster={min_sz} — possible degenerate split")

    sil = silhouette_score(X_clean, clean_labels)
    ch  = calinski_harabasz_score(X_clean, clean_labels)
    db  = davies_bouldin_score(X_clean, clean_labels)
    print(f"  Sizes: {dict(enumerate(cts.tolist()))}")
    print(f"  Silhouette={sil:.4f}  CH={ch:.1f}  DB={db:.4f}")

    # assign outliers to nearest centroid
    X_out = None
    if len(df_outliers) > 0:
        X_out = pipe.transform(df_outliers[avail].values.copy())
        centroids = np.array([X_clean[clean_labels == c].mean(axis=0)
                              for c in range(k)])
        out_labels = cdist(X_out, centroids).argmin(axis=1)
        df_outliers["hier_cluster"]          = out_labels
        df_outliers["hier_nearest_centroid"] = out_labels
        print(f"  Assigned {len(df_outliers)} outliers to nearest Ward centroids")
    df_clean["hier_nearest_centroid"] = df_clean["hier_cluster"]

    # RF labels
    print("\n[4.3] Random Forest — top features per cluster...")
    X_df = pd.DataFrame(X_clean, columns=avail)
    cluster_rf, global_imp = _rf_top_features(X_df, clean_labels, avail)
    c_means_df    = df_clean.groupby("hier_cluster")[avail].mean()
    overall_means = dict(df_clean[avail].mean())
    n_total       = len(df_clean)

    arch_short, arch_desc, arch_metrics = {}, {}, {}
    print("  Calling Anthropic API for cluster labels...")
    for c in range(k):
        sz = int((clean_labels == c).sum())
        print(f"\n  Cluster {c} (n={sz}):")
        c_means = dict(c_means_df.loc[c])
        short, desc, metrics = _label_cluster(
            c, cluster_rf[c], c_means, overall_means, sz, n_total)
        arch_short[c]   = short
        arch_desc[c]    = desc
        arch_metrics[c] = metrics
        print(f"  Name: {short}")
        print(f"  Desc: {desc}")
        for m in metrics:
            print(f"    {m}")

    df_clean["hier_archetype"] = df_clean["hier_cluster"].map(arch_short)
    if len(df_outliers) > 0:
        df_outliers["hier_archetype"] = df_outliers["hier_cluster"].map(arch_short)

    # PCA scatter
    pca    = PCA(n_components=2)
    xy     = pca.fit_transform(X_clean)
    ev     = pca.explained_variance_ratio_
    colors = plt.cm.tab10(np.linspace(0, 1, k))
    fig, ax = plt.subplots(figsize=(12, 9))
    for c in range(k):
        m = (clean_labels == c)
        ax.scatter(xy[m, 0], xy[m, 1], color=colors[c], alpha=0.6, s=30,
                   label=f"C{c}: {arch_short[c]} (n={m.sum()})")
    if X_out is not None:
        xy_out = pca.transform(X_out)
        ax.scatter(xy_out[:, 0], xy_out[:, 1], c="lightgray", marker="x",
                   s=50, alpha=0.5, linewidths=1.2,
                   label=f"Outliers (n={len(df_outliers)})")
    ax.set_xlabel(f"PC1 ({ev[0]:.1%})"); ax.set_ylabel(f"PC2 ({ev[1]:.1%})")
    ax.set_title(f"Hierarchical (Ward) Clusters — CA Political Advertisers\n"
                 f"(k={k}, Sil={sil:.3f})")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGSS, "hierarchical_cluster_scatter.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # feature heatmap
    top10       = sorted(global_imp.items(), key=lambda x: x[1], reverse=True)[:10]
    top10_names = [f for f, _ in top10]
    hmap = np.array([[c_means_df.loc[c, f] / overall_means.get(f, 1)
                      if overall_means.get(f, 0) != 0 else 1.0
                      for f in top10_names] for c in range(k)])
    fig, ax = plt.subplots(figsize=(14, max(5, k * 1.3)))
    im = ax.imshow(hmap, cmap="RdYlGn", aspect="auto", vmin=0, vmax=2)
    ax.set_xticks(range(len(top10_names)))
    ax.set_xticklabels([FEAT_LABELS.get(f, f) for f in top10_names],
                       rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"C{c}: {arch_short[c]}" for c in range(k)], fontsize=9)
    plt.colorbar(im, ax=ax, label="Ratio to overall mean")
    ax.set_title("Hierarchical Cluster Profiles — Top 10 RF Features")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGSS, "hierarchical_cluster_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved figures to {FIGSS}/")

    # save
    df_all = pd.concat([df_clean, df_outliers], ignore_index=True)
    for col in ["hier_cluster", "hier_nearest_centroid", "hier_archetype"]:
        if col in featuress.columns:
            featuress = featuress.drop(columns=[col])
    featuress = featuress.merge(
        df_all[["Advertiser_ID", "hier_cluster",
                "hier_nearest_centroid", "hier_archetype"]],
        on="Advertiser_ID", how="left"
    )
    featuress.to_sql("ca_advertiser_features", conn, if_exists="replace", index=False)

    summary = {
        "_metadata": {
            "method": "Agglomerative (Ward)", "k": k,
            "silhouette": round(sil, 4),
            "calinski_harabasz": round(ch, 1),
            "davies_bouldin": round(db, 4),
            "n_clean": len(df_clean),
            "n_outliers": len(df_outliers),
            "features_used": avail,
        }
    }
    for c in range(k):
        m    = df_clean["hier_cluster"] == c
        top5 = (df_clean[m].nlargest(5, "total_spend")
                [["Advertiser_Name", "total_spend"]].to_dict("records"))
        summary[str(c)] = {
            "cluster_id": c,
            "archetype_name": arch_short[c],
            "archetype_description": arch_desc[c],
            "key_metrics_display": arch_metrics[c],
            "size_clean": int(m.sum()),
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
    with open(os.path.join(DATA_OUTT, "hierarchical_cluster_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    conn.close()
    print("\nPhase 4 complete.")
    return featuress, k, arch_short


if __name__ == "__main__":
    run()
