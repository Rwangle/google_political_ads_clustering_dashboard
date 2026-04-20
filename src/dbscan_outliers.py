# dbscan_outliers.py - phase 2
# PURPOSE: outlier detection only. DBSCAN is used purely to flag
# advertisers whose behavior is so unusual they dont belong to any
# dense group. These 117-ish points get is_outlier=1 and a plain-
# English outlier_reasons string. Phases 3+4 run on the clean subset.

import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

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

# human-readable reason checks — each returns a string if triggered
REASON_CHECKS = [
    ("extreme_spend",
     lambda df: df["total_spend"] > df["total_spend"].quantile(0.995),
     "Extreme total spend (top 0.5% of all advertisers)"),
    ("extreme_cpi",
     lambda df: df["avg_cpi"] > df["avg_cpi"].quantile(0.995),
     "Extreme cost-per-impression (top 0.5%)"),
    ("extreme_burst",
     lambda df: df["pct_spend_in_peak_week"] > 0.95,
     "Nearly 100% of spend in a single week"),
    ("extreme_ad_count",
     lambda df: df["ca_ad_count"] > df["ca_ad_count"].quantile(0.995),
     "Extreme number of CA ads (top 0.5%)"),
    ("extreme_geo",
     lambda df: df["avg_geo_targets"] > df["avg_geo_targets"].quantile(0.995),
     "Extreme geographic targeting breadth (top 0.5%)"),
    ("zero_impressions",
     lambda df: (df["avg_cpi"] == 0) & (df["total_spend"] > df["total_spend"].quantile(0.75)),
     "High spend with zero recorded impressions"),
]


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


def _auto_eps(X_scaled, k=5):
    """
    Find eps using perpendicular distance from diagonal (robust elbow method).

    The k-distance curve is sorted descending. The elbow is where the curve
    bends most — found by computing the perpendicular distance of each point
    from the straight line connecting the first and last points of the curve.
    This is more robust than second-derivative because it ignores the early
    spike caused by genuine extreme outliers.

    After finding the elbow eps, we also verify it flags at least 1% and at
    most 20% of the data. If not, we adjust by scanning nearby percentile
    values of the k-distance distribution.
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
    dists, _ = nbrs.kneighbors(X_scaled)
    kd = np.sort(dists[:, k - 1])[::-1]   # descending k-distances
    n  = len(kd)

    # perpendicular distance from line connecting kd[0] to kd[-1]
    x  = np.arange(n, dtype=float)
    x0, y0 = 0.0, float(kd[0])
    x1, y1 = float(n - 1), float(kd[-1])
    dx, dy = x1 - x0, y1 - y0
    denom  = np.sqrt(dx**2 + dy**2)
    perp   = np.abs(dy * x - dx * (kd - y0) + x1 * y0 - y1 * x0) / (denom + 1e-9)
    elbow  = int(np.argmax(perp))
    eps    = float(kd[elbow])

    # sanity check: eps should flag 1%-20% of data as outliers
    # if outside that range, scan k-distance percentiles to find a better value
    def _pct_noise(e):
        lbl = DBSCAN(eps=e, min_samples=k).fit_predict(X_scaled)
        return (lbl == -1).sum() / len(lbl)

    pct = _pct_noise(eps)
    if pct < 0.01 or pct > 0.20:
        # try percentiles of the k-distance distribution
        for pctile in [5, 4, 3, 2, 1, 6, 7, 8, 10]:
            candidate = float(np.percentile(kd, pctile))
            p = _pct_noise(candidate)
            if 0.01 <= p <= 0.20:
                eps = candidate
                break

    return eps, kd


def _build_reasons(df, is_outlier):
    """For each outlier, collect all triggered reason strings."""
    reasons = [""] * len(df)
    df = df.reset_index(drop=True)
    for _, check_fn, label in REASON_CHECKS:
        try:
            mask = check_fn(df) & is_outlier
            for idx in df[mask].index:
                reasons[idx] = (reasons[idx] + "; " + label).lstrip("; ")
        except Exception:
            continue
    for i in range(len(reasons)):
        if is_outlier.iloc[i] and not reasons[i]:
            reasons[i] = "Statistically isolated across multiple metrics"
    return reasons


def run(db_path=None):
    if db_path is None:
        db_path = DB_PATHH
    os.makedirs(FIGSS, exist_ok=True)
    os.makedirs(DATA_OUTT, exist_ok=True)
    conn = sqlite3.connect(db_path)

    print("=" * 60)
    print("PHASE 2: DBSCAN Outlier Detection")
    print("=" * 60)
    print("  Goal: flag unusual advertisers. They are NOT clustered.")
    print("  Dashboard shows them as 'Outlier' with reason labels.")

    # load
    featuress = pd.read_sql("SELECT * FROM ca_advertiser_features", conn)
    featuress = featuress.reset_index(drop=True)
    print(f"\n  Loaded {len(featuress)} advertisers")

    avail   = [f for f in FEATSS if f in featuress.columns]
    log_idx = [avail.index(f) for f in logColss if f in avail]
    pipe    = _make_pipeline(log_idx)
    X       = pipe.fit_transform(featuress[avail].values.copy())

    # auto eps
    print("\n[2.1] Finding eps from k-distance plot...")
    eps, kd = _auto_eps(X, k=5)
    print(f"  Auto eps: {eps:.3f}")

    # k-distance plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(kd, linewidth=1.5, color="steelblue")
    ax.axhline(y=eps, color="red", linestyle="--", label=f"eps={eps:.2f}")
    ax.set_xlabel("Points (sorted by 5th-neighbor distance)")
    ax.set_ylabel("Distance to 5th nearest neighbor")
    ax.set_title("k-Distance Plot — eps for DBSCAN outlier detection")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGSS, "dbscan_kdist.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # run DBSCAN
    print("\n[2.2] Running DBSCAN...")
    db     = DBSCAN(eps=eps, min_samples=5).fit(X)
    labels = db.labels_
    n_out  = int((labels == -1).sum())
    pct    = n_out / len(labels)
    print(f"  eps={eps:.3f}  Outliers: {n_out} ({pct:.1%})")

    # safety net: target 1%-20% flagged as outliers
    # if outside that band, scan k-distance percentiles to recalibrate eps
    if pct < 0.01 or pct > 0.20:
        print(f"  Outlier rate {pct:.1%} outside 1-20% target — rescanning eps...")
        kd_flat = np.sort(
            NearestNeighbors(n_neighbors=5).fit(X).kneighbors(X)[0][:, 4])[::-1]
        found = False
        scan_order = [5, 4, 3, 2, 1, 6, 7, 8, 10, 15] if pct < 0.01 else [95, 90, 85, 80, 75]
        for pctile in scan_order:
            candidate = float(np.percentile(kd_flat, pctile))
            lbl_try   = DBSCAN(eps=candidate, min_samples=5).fit_predict(X)
            p_try     = (lbl_try == -1).sum() / len(lbl_try)
            if 0.01 <= p_try <= 0.20:
                eps    = candidate
                labels = lbl_try
                n_out  = int((labels == -1).sum())
                pct    = p_try
                print(f"  Recalibrated eps={eps:.3f}  Outliers: {n_out} ({pct:.1%})")
                found = True
                break
        if not found:
            # hard fallback: 5th percentile of k-distances (~5% flagged)
            eps    = float(np.percentile(kd_flat, 5))
            labels = DBSCAN(eps=eps, min_samples=5).fit_predict(X)
            n_out  = int((labels == -1).sum())
            pct    = n_out / len(labels)
            print(f"  Fallback eps={eps:.3f}  Outliers: {n_out} ({pct:.1%})")

    is_outlier = pd.Series(labels == -1, index=featuress.index)

    # build reason strings
    print("\n[2.3] Building outlier reason labels...")
    reasons = _build_reasons(featuress, is_outlier)
    featuress["is_outlier"]      = is_outlier.astype(int)
    featuress["outlier_reasons"] = reasons

    # print sample
    out_df = featuress[featuress["is_outlier"] == 1][
        ["Advertiser_Name", "total_spend", "avg_cpi", "outlier_reasons"]
    ].sort_values("total_spend", ascending=False)
    print(f"\n  Sample outliers:")
    for _, r in out_df.head(10).iterrows():
        print(f"    {str(r['Advertiser_Name']):<40} "
              f"${r['total_spend']:>10,.0f}  {r['outlier_reasons'][:60]}")

    # PCA scatter
    pca = PCA(n_components=2)
    xy  = pca.fit_transform(X)
    ev  = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(xy[~is_outlier, 0], xy[~is_outlier, 1],
               c="steelblue", alpha=0.35, s=18, label=f"Normal ({(~is_outlier).sum()})")
    ax.scatter(xy[is_outlier, 0], xy[is_outlier, 1],
               c="red", marker="x", s=60, linewidths=1.5,
               label=f"Outlier ({n_out})")
    ax.set_xlabel(f"PC1 ({ev[0]:.1%})"); ax.set_ylabel(f"PC2 ({ev[1]:.1%})")
    ax.set_title(f"DBSCAN Outlier Detection\n(eps={eps:.2f}, {n_out} outliers = {pct:.1%})")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGSS, "dbscan_outliers.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved figures to {FIGSS}/")

    # reason bar chart
    rc = {}
    for r in reasons:
        if r:
            for p in r.split(";"):
                p = p.strip()
                if p:
                    rc[p] = rc.get(p, 0) + 1
    if rc:
        rs = sorted(rc.items(), key=lambda x: x[1], reverse=True)
        fig, ax = plt.subplots(figsize=(10, max(4, len(rs) * 0.5)))
        ax.barh([x[0] for x in rs], [x[1] for x in rs], color="salmon")
        ax.set_xlabel("Number of outlier advertisers")
        ax.set_title("Why Advertisers Were Flagged as Outliers")
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGSS, "dbscan_outlier_reasons.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # save
    featuress.to_sql("ca_advertiser_features", conn, if_exists="replace", index=False)
    out_df.to_csv(os.path.join(DATA_OUTT, "outlier_advertisers.csv"), index=False)
    print(f"  Saved outlier list: {DATA_OUTT}/outlier_advertisers.csv")
    print(f"  Updated db: added is_outlier, outlier_reasons")

    conn.close()
    print(f"\n  Clean dataset: {(~is_outlier).sum()} advertisers")
    print(f"  Outliers:      {n_out} advertisers")
    print("\nPhase 2 complete.")
    return featuress, n_out, eps


if __name__ == "__main__":
    run()
