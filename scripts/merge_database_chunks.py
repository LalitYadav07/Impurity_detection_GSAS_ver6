import os
import glob
import pandas as pd
import numpy as np
import argparse
import logging

logger = logging.getLogger(__name__)

def merge_database(db_dir):
    """
    Consolidates fragmented database files (catalog_deduplicated_*.csv, profiles64_*.npz, etc.)
    into single monolithic files required for the production pipeline.
    """
    logger.info(f"Merging database chunks in {db_dir}...")
    
    # 1. Identify chunks based on catalog files
    # We look for catalog_deduplicated_*.csv
    catalog_files = sorted(glob.glob(os.path.join(db_dir, "catalog_deduplicated_*.csv")))
    if not catalog_files:
        # Fallback to check if they don't have the underscore (though script adds it)
        catalog_files = sorted(glob.glob(os.path.join(db_dir, "catalog_deduplicated*.csv")))
        # Filter out the final one if it already exists
        catalog_files = [f for f in catalog_files if "_" in os.path.basename(f)]

    if not catalog_files:
        logger.info("No chunked catalog files found to merge.")
        return

    all_catalogs = []
    all_profiles = []
    all_indices = []
    all_stables = []

    current_profile_offset = 0
    
    # Track metadata from the first valid .npz
    q_params = {}

    for cat_path in catalog_files:
        # Extract suffix to find matching files
        # basename might be catalog_deduplicated_0.csv -> suffix is _0
        base = os.path.basename(cat_path)
        suffix = base.replace("catalog_deduplicated", "").replace(".csv", "")
        
        logger.info(f"  Processing chunk {suffix}...")
        
        prof_path = os.path.join(db_dir, f"profiles64{suffix}.npz")
        idx_path = os.path.join(db_dir, f"index_deduplicated{suffix}.csv")
        stab_path = os.path.join(db_dir, f"mp_experimental_stable{suffix}.csv")

        # Load Catalog
        try:
            cat_df = pd.read_csv(cat_path)
            all_catalogs.append(cat_df)
        except Exception as e:
            logger.warning(f"    Error reading catalog {cat_path}: {e}")
            continue

        # Load Profiles
        if os.path.exists(prof_path):
            try:
                with np.load(prof_path) as data:
                    profs = data["profiles"]
                    # Capture parameters from the first chunk
                    if not q_params and "q_min" in data:
                        q_params = {
                            "q_min": data["q_min"],
                            "q_max": data["q_max"],
                            "n_bins": data["n_bins"],
                            "sigma_bins": data["sigma_bins"]
                        }
                    
                    all_profiles.append(profs)
                    
                    # Offset the index mapping
                    if os.path.exists(idx_path):
                        idx_df = pd.read_csv(idx_path)
                        idx_df["row"] = idx_df["row"] + current_profile_offset
                        all_indices.append(idx_df)
                    
                    current_profile_offset += profs.shape[0]
            except Exception as e:
                logger.warning(f"    Error reading profiles {prof_path}: {e}")
        else:
            logger.warning(f"    Profile file {prof_path} missing for catalog chunk.")

        # Load Stability
        if os.path.exists(stab_path):
            try:
                all_stables.append(pd.read_csv(stab_path))
            except Exception as e:
                logger.warning(f"    Error reading stability data {stab_path}: {e}")

    if not all_catalogs:
        logger.info("No valid data chunks recovered.")
        return

    # 2. Concatenate and deduplicate
    logger.info(f"Consolidating {len(all_catalogs)} chunks...")
    merged_catalog = pd.concat(all_catalogs, ignore_index=True)
    merged_index = pd.concat(all_indices, ignore_index=True) if all_indices else None
    merged_profiles = np.vstack(all_profiles) if all_profiles else None
    
    # Deduplicate stability data (keep unique material_id)
    if all_stables:
        merged_stable = pd.concat(all_stables, ignore_index=True)
        merged_stable.drop_duplicates(subset="material_id", keep="first", inplace=True)
    else:
        merged_stable = None

    # 3. Write final monolithic files
    logger.info("Writing monolithic files...")
    
    # Ensure standard names
    merged_catalog.to_csv(os.path.join(db_dir, "catalog_deduplicated.csv"), index=False)
    
    if merged_index is not None:
        merged_index.to_csv(os.path.join(db_dir, "index_deduplicated.csv"), index=False)
    
    if merged_stable is not None:
        merged_stable.to_csv(os.path.join(db_dir, "mp_experimental_stable.csv"), index=False)
    
    if merged_profiles is not None:
        # Fallback params if none found in .npz files
        if not q_params:
            q_params = {"q_min": 0.5, "q_max": 6.0, "n_bins": 64, "sigma_bins": 0.7}
            
        np.savez_compressed(
            os.path.join(db_dir, "profiles64.npz"),
            profiles=merged_profiles,
            **q_params
        )
    
    logger.info(f"SUCCESS. Consolidated {len(merged_catalog)} total structures.")
    logger.info(f"Final directory: {os.path.abspath(db_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge fragmented DB chunks into monolithic files.")
    parser.add_argument("--db-dir", default="data/database_xray", help="Directory containing the chunked files.")
    args = parser.parse_args()
    merge_database(args.db_dir)
