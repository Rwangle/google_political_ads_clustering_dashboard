# main entry point. run after rogers loader populates the db
# usage: python run_pipeline.py

import time
import sys
import os

scrpt_dirr = os.path.dirname(  os.path.abspath(
    __file__      ))
sys.path.insert(0,scrpt_dirr)

from src import data_cleaning
from src import clustering
from src import validation
from src import refine_model

db_filee = os.path.join(scrpt_dirr,"db","political_ads.db"  )

def main():
    t_start = time.time(  )
    print("=" * 60)
    print("  CA Political Ad Clustering Pipeline")
    print("  Starting full pipeline run..."  )
    print("=" * 60)

    # phase 1
    print("\n")
    features_df = data_cleaning.run(db_path =db_filee)
    t1 = time.time()- t_start
    print(f"\n  [Phase 1 completed in {t1:.1f}s]")

    # phase 2
    print(  "\n")
    features_df,final_k,X_scaled,prep,km_model = clustering.run(db_path=db_filee)
    t2 = time.time() -t_start -t1
    print(f"\n  [Phase 2 completed in {t2:.1f}s]")

    # phase 3
    print("\n")
    val_metrics,val_lines = validation.run(db_path=db_filee,final_k =final_k,X_scaled=X_scaled)
    t3 = time.time()-t_start -t1- t2
    print(f"\n  [Phase 3 completed in {t3:.1f}s]")

    # phase 4 - try other models (spoiler: kmeans wins)
    print("\n"  )
    final_feats,refine_lines = refine_model.run(db_path= db_filee,  final_k=final_k)
    t4 = time.time()-t_start-t1-t2-t3
    print(f"\n  [Phase 4 completed in {t4:.1f}s]")

    total = time.time()-t_start
    print("\n" + "=" * 60)
    print(f"  Pipeline complete! Total time: {total:.1f}s")
    print("  Outputs saved to: output/")
    print("    - output/figures/        (all plots)")
    print("    - output/data/           (CSVs, JSON)")
    print("="*60)


if __name__=="__main__":
    main()
