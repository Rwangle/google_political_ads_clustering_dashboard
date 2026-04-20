# run_pipeline.py — main entry point
# usage: python run_pipeline.py
#
# Phase 1 — data_cleaning           : build ca_advertiser_features (unchanged)
# Phase 2 — dbscan_outliers         : flag outliers with is_outlier + outlier_reasons
# Phase 3 — clustering              : K-Means on clean data, RF labels, API names
# Phase 4 — hierarchical_clustering : Ward on clean data, same k, RF labels, API names
# Phase 5 — validation              : compare methods, export tableau_dashboard.csv
#
# Key output: output/data/tableau_dashboard.csv
# Dashboard shows kmeans_archetype AND hier_archetype per advertiser.
# Outliers show display_archetype = "Outlier: <reason>"

import time
import sys
import os

scrpt_dirr = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, scrpt_dirr)

from src import data_cleaning
from src import dbscan_outliers
from src import clustering
from src import hierarchical_clustering
from src import validation

db_filee = os.path.join(scrpt_dirr, "db", "political_ads.db")


def main():
    t0 = time.time()
    print("=" * 60)
    print("  CA Political Ad Clustering Pipeline")
    print("=" * 60)

    # Phase 1 — feature engineering (unchanged from original)
    t = time.time()
    data_cleaning.run(db_path=db_filee)
    t1 = time.time() - t
    print(f"\n  [Phase 1 done in {t1:.1f}s]")

    # Phase 2 — DBSCAN outlier detection
    t = time.time()
    _, n_outliers, eps = dbscan_outliers.run(db_path=db_filee)
    t2 = time.time() - t
    print(f"\n  [Phase 2 done in {t2:.1f}s — {n_outliers} outliers flagged]")

    # Phase 3 — K-Means (RF labels + API names)
    t = time.time()
    _, best_k, X_clean, pipe, km_model, km_archetypes = clustering.run(db_path=db_filee)
    t3 = time.time() - t
    print(f"\n  [Phase 3 done in {t3:.1f}s — k={best_k}]")

    # Phase 4 — Hierarchical Ward (same k, RF labels + API names)
    t = time.time()
    _, hier_k, hier_archetypes = hierarchical_clustering.run(db_path=db_filee, k=best_k)
    t4 = time.time() - t
    print(f"\n  [Phase 4 done in {t4:.1f}s — k={hier_k}]")

    # Phase 5 — Compare + export Tableau CSV
    t = time.time()
    val_metrics, _ = validation.run(db_path=db_filee)
    t5 = time.time() - t
    print(f"\n  [Phase 5 done in {t5:.1f}s]")

    total = time.time() - t0
    print("\n" + "=" * 60)
    print(f"  Pipeline complete in {total:.1f}s")
    print()
    print("  Summary:")
    print(f"    Outliers flagged:        {n_outliers}")
    print(f"    K-Means clusters (k):    {best_k}")
    print(f"    Hierarchical clusters:   {hier_k}")
    km_sil = val_metrics.get("kmeans", {}).get("silhouette", "n/a")
    h_sil  = val_metrics.get("hierarchical", {}).get("silhouette", "n/a")
    ari    = val_metrics.get("ari", "n/a")
    agree  = val_metrics.get("pct_agreement", "n/a")
    print(f"    K-Means silhouette:      {km_sil}")
    print(f"    Hierarchical silhouette: {h_sil}")
    print(f"    ARI (agreement):         {ari}")
    print(f"    % advertisers agree:     {agree}")
    print()
    print("  Key output for Tableau:")
    print("    output/data/tableau_dashboard.csv")
    print()
    print("  Figures saved to output/figures/:")
    print("    dbscan_kdist.png")
    print("    dbscan_outliers.png")
    print("    dbscan_outlier_reasons.png")
    print("    kmeans_k_selection.png")
    print("    kmeans_cluster_scatter.png")
    print("    kmeans_cluster_heatmap.png")
    print("    hierarchical_dendrogram.png")
    print("    hierarchical_cluster_scatter.png")
    print("    hierarchical_cluster_heatmap.png")
    print("    method_comparison.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
