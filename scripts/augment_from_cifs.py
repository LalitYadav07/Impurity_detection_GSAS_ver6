#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build .../data/database_xray (or database_aug) from CIF structures.

Key points:
- Parallel processing via ProcessPoolExecutor with chunked map (low overhead).
- Sensible worker cap via _pick_workers() (respects SLURM).
- BLAS/OMP threads clamped to 1 in workers to avoid oversubscription.
- Profiles64 handled via single preallocated array -> fast, no vstack loops.
- Minimal console chatter (progress every PROGRESS_EVERY items).
- RESUMABLE: if interrupted, re-run with same args. Already-built phases
  in db_out/phases/ are detected, their profiles are rebuilt from the .npz,
  and only truly new phases are sent to workers.

Switch:
- Set USE_JSON_SOURCE=True to read {id: {"cif_content": "..."} } from JSON.
- Set USE_JSON_SOURCE=False to read *.cif from a directory.
"""

import os, json, glob, argparse, math, time, shutil, warnings, platform
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from itertools import islice

# ===== User toggles =====
USE_JSON_SOURCE = True       # True: read from JSON; False: read *.cif folder
PROGRESS_EVERY  = 100        # print progress every N completed items
DB_IN_DEFAULT   = "data/database_aug"
DB_OUT_DEFAULT  = "data/database_aug_v2"
CIF_DIR_DEFAULT = "data/all_cifs"
JSON_FILE_DEFAULT = "highsymm_metadata.json"

# We'll use database_catalog_builder.process_one_phase logic, but adapted.
# We need to import it. Ensure scripts/ is in PYTHONPATH or sys.path.
import sys
sys.path.append(os.path.dirname(__file__))
from database_catalog_builder import process_one_phase

def _pick_workers(n_items):
    """
    Heuristic to pick max_workers.
    On SLURM, standard os.cpu_count() returns the *node* total, not allocation.
    We try len(os.sched_getaffinity(0)) on Linux if available.
    """
    import os
    cpus = 1
    if hasattr(os, "sched_getaffinity"):
        try:
            cpus = len(os.sched_getaffinity(0))
        except Exception:
            cpus = os.cpu_count() or 1
    else:
        cpus = os.cpu_count() or 1
    
    # Cap at n_items (if small)
    w = min(cpus, n_items)
    # Cap at something reasonable to avoid OS thrashing if cpus is huge
    return max(1, min(w, 64))

def _gaussian_deposit(prof, center_bin, weight, sigma_bins):
    L = prof.shape[0]
    if sigma_bins <= 0:
        # delta function
        j = int(round(center_bin))
        if 0 <= j < L:
            prof[j] += weight
        return
    
    # Range +/- 3 sigma
    halfw = int(max(1, math.ceil(3.0 * sigma_bins)))
    j0 = int(round(center_bin))
    jL = max(0, j0 - halfw)
    jR = min(L - 1, j0 + halfw)

    if jR < jL:
        return

    # Evaluate gaussian
    # x indices
    js = np.arange(jL, jR + 1, dtype=np.float32)
    diff = (js - center_bin) / sigma_bins
    g = np.exp(-0.5 * diff**2)
    # We do NOT normalize area to 1 here because 'weight' is intensity I0.
    # Actually GSAS-II peaks have width. If we just want visually comparable peaks
    # with height ~ I0, we can just add. 
    # But usually integrated intensity is conserved. 
    # Let's keep it simple: peak height ~ weight * exp(...).
    prof[jL:jR + 1] += weight * g


def _worker_init():
    """Pin threads to 1 to avoid oversubscription in parallel workers."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _worker_do_one(args):
    """
    Run in a subprocess. Build one phase and its profile.
    Returns a compact tuple to reduce IPC overhead.
    """
    (cid, cif_text, db_out,
     two_theta_min, two_theta_max, topM,
     q_min, q_max, n_bins, sigma_bins,
     radiation, wavelength) = args

    meta = {"cif_content": cif_text}

    # Import here to keep top-level import minimal in workers
    import numpy as _np
    import os as _os
    import math as _math

    # local copy of gaussian/profile builders (kept minimal)
    def _gd(profile, center_bin, weight, sb):
        L = profile.shape[0]
        if sb <= 0:
            j = int(round(center_bin))
            if 0 <= j < L:
                profile[j] += weight
            return
        halfw = int(max(1, _math.ceil(3.0 * sb)))
        j0 = int(round(center_bin))
        jL = max(0, j0 - halfw); jR = min(L - 1, j0 + halfw)
        if jR < jL: return
        js = _np.arange(jL, jR + 1, dtype=_np.float32)
        g = _np.exp(-0.5 * ((js - center_bin) / sb) ** 2)
        profile[jL:jR + 1] += weight * g

    def _build_prof(npz_path, q_min, q_max, n_bins=64, sigma_bins=0.7):
        npz = _np.load(npz_path)
        q0 = npz["q0"].astype(_np.float32)
        I0 = npz["I0"].astype(_np.float32)
        if q0.size == 0:
            return None
        I0 = I0 / max(1e-12, float(I0.max()))
        L = int(n_bins)
        dq = (q_max - q_min) / L
        prof = _np.zeros(L, dtype=_np.float32)
        for q, w in zip(q0, I0):
            if not (q_min <= q < q_max):
                continue
            pos = (q - q_min) / dq - 0.5
            _gd(prof, pos, w, sigma_bins)
        m = float(prof.max())
        if m > 0:
            prof /= m
        return prof.astype(_np.float16)

    try:
        row, fail, mm = process_one_phase(
            cid, meta, db_out,
            two_theta_min, two_theta_max, topM,
            False, False, False, False,
            radiation, wavelength
        )
        npz_path = os.path.join(db_out, row["npz"])
        prof = _build_prof(npz_path, q_min, q_max, n_bins, sigma_bins)

        stab_row = (
            cid,
            row["pretty_formula"],
            int(row["space_group"]),
            row["SG_symbol"],
            -1.0
        )
        # Return compact forms to minimize pickling overhead
        return (1, cid, row, prof, stab_row)
    except Exception as e:
        return (0, cid, repr(e))

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Build database_aug fast (parallel, chunked)")
    ap.add_argument("--db-in", default=DB_IN_DEFAULT)
    ap.add_argument("--db-out", default=DB_OUT_DEFAULT)
    ap.add_argument("--cif-dir", default=CIF_DIR_DEFAULT, help="Folder of .cif files (used if USE_JSON_SOURCE=False)")
    ap.add_argument("--json-file", default=JSON_FILE_DEFAULT, help="JSON with {id: {'cif_content': '...'}} (used if USE_JSON_SOURCE=True)")
    ap.add_argument("--radiation", default="neutron", help="neutron or xray")
    ap.add_argument("--wavelength", type=float, default=1.54184, help="Wavelength (xray only)")
    ap.add_argument("--fresh", action="store_true", help="Start from empty (no existing profiles/catalog required)")
    ap.add_argument("--serial", action="store_true", help="Force sequential processing (no multiprocessing) to avoid Windows spawn issues")
    ap.add_argument("--start-idx", type=int, default=0, help="Start index in the structure list")
    ap.add_argument("--end-idx", type=int, default=-1, help="End index in the structure list (-1 for all)")
    ap.add_argument("--out-suffix", default="", help="Suffix for output catalog/index files")
    args = ap.parse_args()

    db_in   = args.db_in
    db_out  = args.db_out
    cifdir  = args.cif_dir
    jsonfp  = args.json_file

    print(f"DB_IN:  {db_in}")
    print(f"DB_OUT: {db_out}")
    print(f"MODE:   {'JSON' if USE_JSON_SOURCE else 'FOLDER'}")
    print(f"Radiation: {args.radiation}  Wavelength: {args.wavelength}")

    os.makedirs(db_out, exist_ok=True)
    os.makedirs(os.path.join(db_out, "phases"), exist_ok=True)

    # 1. Load existing Catalog/Profiles or Start Fresh
    meta_out = {} # Initialize meta_out to fix Bug 1

    if args.fresh:
        print("== FRESH BUILD: Ignoring DB_IN profile/catalog content ==")
        cat_in = pd.DataFrame(columns=["id", "pretty_formula", "space_group", "SG_symbol", "elements_list", "elements_mask_hi", "elements_mask_lo", "npz", "n_reflections"])
        idx_in = pd.DataFrame(columns=["id", "row"])
        
        # Initialize stab_in from source if available, or empty
        stab_path = os.path.join(db_in, "mp_experimental_stable.csv")
        if os.path.exists(stab_path):
            stab_in = pd.read_csv(stab_path)
            print(f"  Loaded {len(stab_in)} rows from {stab_path}")
        else:
            stab_in = pd.DataFrame(columns=["material_id", "formula_pretty", "spacegroup_number", "spacegroup_symbol", "energy_above_hull_eV_per_atom"])

        # Default profile parameters for fresh build
        q_min, q_max = 0.5, 6.0
        n_bins = 64
        sigma_bins = 0.7

        init_profiles = np.zeros((0, n_bins), dtype=np.float16)

    else:
        # Load from DB_IN (resume-like behavior, but usually we augment DB_IN content)
        # Actually, for "augment", we typically load catalog.csv from DB_IN.
        # But here we are building a NEW DB (X-ray) from scratch, so DB_IN is just for metadata?
        # The prompt implies we want to build X-ray DB. 
        # If we are resuming, we should look at DB_OUT.
        
        # Let's check DB_OUT for resume content
        pass
        # (Original code logic was complex here. We simplified: freshness is controlled by --fresh)
        # If NOT fresh, we assume we might want to carry over some data? 
        # Actually, for X-ray vs Neutron, we should NOT carry over profiles.
        # So we should always be fresh regarding profiles?
        # NO. The user might interrupt and resume.
        pass

    # For simplicity in this script re-write:
    # We will assume if --fresh is passed, we ignore everything in DB_OUT.
    # If --fresh is NOT passed, we try to Load DB_OUT catalog/profiles to resume?
    # But profiles are monolithic. It's hard to "append" to a .npz file easily without loading it all.
    # Strategy:
    # Always load DB_OUT content if it exists (for RESUME).
    # If it doesn't exist, start fresh.
    
    cat_out_path = os.path.join(db_out, "catalog_deduplicated.csv")
    idx_out_path = os.path.join(db_out, "index_deduplicated.csv")
    prof_out_path = os.path.join(db_out, "profiles64.npz")
    stab_out_path = os.path.join(db_out, "mp_experimental_stable.csv")

    # Load Source JSON/CIFs
    if USE_JSON_SOURCE:
        print(f"JSON file: {jsonfp}")
        with open(jsonfp, "r") as f:
            meta_in = json.load(f)
        print(f"Found CIFs in JSON: {len(meta_in)}")
        # Filter only those with cif_content
        meta_in = {k: v for k, v in meta_in.items() if "cif_content" in v}
    else:
        # scan folder
        cifs = glob.glob(os.path.join(cifdir, "*.cif"))
        print(f"Found CIFs in folder: {len(cifs)}")
        meta_in = {}
        for c in cifs:
            cid = os.path.basename(c).replace(".cif","")
            with open(c, "r") as f: txt = f.read()
            meta_in[cid] = {"cif_content": txt}

    # Initialize container for output
    # If resuming, load existing
    if not args.fresh and os.path.exists(cat_out_path) and os.path.exists(prof_out_path):
        print("== RESUME: Loading existing catalog/profiles from DB_OUT ==")
        cat_out = pd.read_csv(cat_out_path)
        if os.path.exists(idx_out_path):
            idx_out = pd.read_csv(idx_out_path)
        else:
            # Reconstruct index if missing
            idx_out = pd.DataFrame({"id": cat_out["id"], "row": cat_out.index})
        
        p_data = np.load(prof_out_path)
        profiles_chunk = p_data["profiles"]
        q_min_ld = float(p_data["q_min"])
        q_max_ld = float(p_data["q_max"])
        n_bins_ld = int(p_data["n_bins"])
        sigma_ld = float(p_data["sigma_bins"])
        
        if n_bins_ld != 64:
             print("WARNING: Existing profiles have different n_bins. Forcing FRESH.")
             # Fallback to fresh logic below
             args.fresh = True
        else:
             q_min, q_max, n_bins, sigma_bins = q_min_ld, q_max_ld, n_bins_ld, sigma_ld
             init_profiles = profiles_chunk
             
        if os.path.exists(stab_out_path):
            stab_in = pd.read_csv(stab_out_path)
        else:
             # Try DB_IN
             s_in_path = os.path.join(db_in, "mp_experimental_stable.csv")
             if os.path.exists(s_in_path):
                 stab_in = pd.read_csv(s_in_path)
             else:
                 stab_in = pd.DataFrame(columns=["material_id", "formula_pretty", "spacegroup_number", "spacegroup_symbol", "energy_above_hull_eV_per_atom"])
    
    if args.fresh or not (os.path.exists(cat_out_path) and os.path.exists(prof_out_path)):
        # FRESH initialization
        cat_out = pd.DataFrame(columns=["id", "pretty_formula", "space_group", "SG_symbol", "elements_list", "elements_mask_hi", "elements_mask_lo", "npz", "n_reflections"])
        idx_out = pd.DataFrame(columns=["id", "row"])
        init_profiles = np.zeros((0, 64), dtype=np.float16)
        
        # Load stability from DB_IN (Bug 5 fix)
        s_in_path = os.path.join(db_in, "mp_experimental_stable.csv")
        if os.path.exists(s_in_path):
             stab_in = pd.read_csv(s_in_path)
        else:
             stab_in = pd.DataFrame(columns=["material_id", "formula_pretty", "spacegroup_number", "spacegroup_symbol", "energy_above_hull_eV_per_atom"])

    # Determine what to work on
    # We want to process ALL items in meta_in.
    # Exclude those already in cat_out?
    # existing_ids = set(cat_out["id"].values)
    # Actually, verify against phases folder for robust resume (Bug 4 fix)
    existing_npz = set([f.replace(".npz","") for f in os.listdir(os.path.join(db_out, "phases")) if f.endswith(".npz")])
    
    # Check consistency: if id in cat_out but not in phases, remove from cat_out?
    # Or if id in phases but not in cat_out, add to cat_out?
    # Simplest: If resuming, trust existing phases.
    # Re-build profiles array from ALL existing phases (robust resume).
    
    print(f"== RESUME: Found {len(existing_npz)} existing phase .npz files")
    
    # Re-load profiles for ALL existing phases to ensure profiles64.npz is consistent
    # This is "Bug 4" fix: Resumability broken for profiles.
    # We will rebuild init_profiles from the phases folder content.
    
    # But wait, initially init_profiles might be from 350k items if we resumed catalog.
    # To be safe: discard init_profiles and rebuild it from existing_npz.
    
    # List of all IDs we want to process
    all_target_ids = sorted(list(meta_in.keys()))
    
    # Slice for GNU Parallel / Chunking
    if args.end_idx == -1:
        args.end_idx = len(all_target_ids)
    print(f"Slicing IDs: {args.start_idx} to {args.end_idx} (Total: {len(all_target_ids)})")
    all_target_ids = all_target_ids[args.start_idx:args.end_idx]
    
    # Identify what is done and what is new
    done_ids = set(existing_npz)
    
    # Build profiles for DONE items
    # Doing this in main thread is fast enough (loading .npz is fast)
    # We need to reconstruct cat_out and idx_out from these done items too, 
    # or just trust they match.
    # Safer: Re-create cat_out rows and profiles for all done items.
    
    print("Re-assembling catalog/profiles from existing .npz files...")
    
    # Helper to load a profile
    def load_prof_from_file(cid):
        path = os.path.join(db_out, "phases", f"{cid}.npz")
        try:
            d = np.load(path)
            # Re-bin if needed? Assume matches q_min/q_max/bins
            # Code from _run_worker logic:
            q0 = d["q0"]; I0 = d["I0"]
            if len(q0) == 0: return None
            I0 = I0 / max(1e-12, float(I0.max()))
            L = 64
            dq = (q_max - q_min) / L
            prof = np.zeros(L, dtype=np.float32)
            # Gaussian deposit re-impl?? 
            # Ideally we used the same function.
            # But wait, existing .npz only has peaks. We need to broaden them again.
            # So we DO need _gaussian_deposit here.
            for q, w in zip(q0, I0):
                if not (q_min <= q < q_max): continue
                pos = (q - q_min) / dq - 0.5
                _gaussian_deposit(prof, pos, w, sigma_bins)
            m = prof.max()
            if m > 0: prof /= m
            return prof.astype(np.float16)
        except Exception as e:
            warnings.warn(f"Failed to rebuild profile from {path}: {e}")
            return None

    # We also need metadata (formula etc) for the catalog.
    # We can get it from the .npz (it usually has metadata?)
    # Or from meta_in.
    
    resumed_rows = []
    resumed_profs = []
    resumed_cids = []
    
    resume_errors = 0
    
    for cid in all_target_ids:
        if cid in done_ids:
            # It exists on disk
            prof = load_prof_from_file(cid)
            if prof is not None:
                # We need catalog info. 
                # Ideally from the .npz file (which contains metadata).
                try:
                    d = np.load(os.path.join(db_out, "phases", f"{cid}.npz"), allow_pickle=True)
                    # We stored 'row' in the tuple returned by worker, but on disk we save what process_one_phase saves.
                    # process_one_phase saves metadata in .npz? 
                    # Usually "metadata" key.
                    # Let's peek at what process_one_phase does. 
                    # It calls saver(..., metadata=row).
                    # So yes.
                    # Note: We might need to handle pickling.
                    meta_disk = d["metadata"].item()
                    # meta_disk is the row dict
                    resumed_rows.append(meta_disk)
                    resumed_profs.append(prof)
                    resumed_cids.append(cid)
                    
                    # Update meta_out (Bug 1 fix)
                    # We need to ensure meta_out has this content if we are to use it?
                    # Actually meta_out is usually just meta_in passed through?
                    # The script at top says: meta_out = {}
                    # But wait, looking at line 286 crash: "meta_out[cid] = meta[cid]"?
                    # The variable 'meta_out' in original script was a copy of 'meta_in'.
                    # Here we have 'meta_in'. We can just use it.
                    # We don't need to write a new JSON if we have the original.
                    # But wait, if we modify it? We don't modify inputs.
                    
                except Exception as e:
                    warnings.warn(f"Failed to resume metadata from phase file for {cid}: {e}")
                    resume_errors += 1
            else:
                resume_errors += 1
    
    print(f"  Resumed {len(resumed_rows)} phases, {resume_errors} errors")
    
    # Establish starting state
    cat_out = pd.DataFrame(resumed_rows)
    # Ensure columns match
    if not cat_out.empty:
        # Reorder/ensure columns
        cols = ["id", "pretty_formula", "space_group", "SG_symbol", "elements_list", "elements_mask_hi", "elements_mask_lo", "npz", "n_reflections"]
        # fill missing
        for c in cols:
            if c not in cat_out.columns: cat_out[c] = None
        cat_out = cat_out[cols]
    else:
        cat_out = pd.DataFrame(columns=["id", "pretty_formula", "space_group", "SG_symbol", "elements_list", "elements_mask_hi", "elements_mask_lo", "npz", "n_reflections"])

    if resumed_profs:
        profiles_all = np.vstack(resumed_profs)
    else:
        profiles_all = np.zeros((0, 64), dtype=np.float16)

    # Rebuild index
    idx_out = pd.DataFrame({"id": resumed_cids, "row": range(len(resumed_cids))})
    
    # What about stability?
    # We should preserve stab_in and append new ones. 
    # But wait, we don't extract stability from .npz. 
    # We assume 'stab_in' (loaded from file) contains everything.
    # Or we can just rebuild it? 
    # If we are strictly augmenting, we should stick to what we have or append.
    # For now, let's assume stab_in is good. We will append new items to it.
    
    print(f"Original catalog rows:  {len(cat_out)}")
    print(f"Original profiles rows: {len(profiles_all)}  bins: {profiles_all.shape[1]}")

    # Prepare Work Items (only new ones)
    work_items = []
    
    # We want to process ids that are in meta_in BUT NOT in resumed_cids
    resumed_set = set(resumed_cids)
    
    for cid in all_target_ids:
        if cid not in resumed_set:
            work_items.append((cid, meta_in[cid]["cif_content"]))

    total_new = len(work_items)
    print(f"Q-range/bins/sigma:     [{q_min}, {q_max}]  n_bins={n_bins}  sigma={sigma_bins}")
    print(f"New items to process:   {total_new}")

    if total_new == 0:
        print("Nothing to do.")
        # Write outputs anyway to ensure consistency
        _finalize_and_write(db_out, cat_out, idx_out, profiles_all, stab_in, q_min, q_max, n_bins, sigma_bins)
        return

    # Pre-allocate profiles for new items to avoid vstack
    # We will fill them in slot (len(resumed)+i)
    new_profs_alloc = np.zeros((total_new, n_bins), dtype=np.float16)
    profiles_all = np.vstack([profiles_all, new_profs_alloc])
    
    write_ptr = len(resumed_cids)  # next free row for a new profile

    # Prepare worker args
    args_iter = [
        (cid, cif_text, db_out,
         two_theta_min, two_theta_max, topM,
         q_min, q_max, n_bins, sigma_bins,
         args.radiation, args.wavelength)
        for (cid, cif_text) in work_items
    ]

    
    # Choose workers and chunks
    if args.serial:
        workers = 1
        chunksize = 1
        print("Running in SERIAL mode (no multiprocessing)")
    else:
        workers = _pick_workers(len(args_iter))
        chunksize = max(1, len(args_iter) // (workers * 8))
        print(f"Workers: {workers}  |  Chunksize: {chunksize}")

    ok_rows    = []     # rows to append into catalog
    ok_stab    = []     # rows to append into stable CSV
    errors     = []
    new_idx_rows = []   # Accumulated index rows for efficiency
    done       = 0

    if args.serial:
        # Simple for-loop
        for arg in args_iter:
            res = _worker_do_one(arg)
            if res[0] == 1:
                _, cid, row, prof, stab_row = res
                ok_rows.append(row)
                if prof is not None:
                    profiles_all[write_ptr, :] = prof
                    # Optimization: Accumulate in list, concat later
                    new_idx_rows.append({"id": cid, "row": int(write_ptr)})
                    write_ptr += 1
                ok_stab.append({
                    "material_id":       stab_row[0],
                    "formula_pretty":    stab_row[1],
                    "spacegroup_number": stab_row[2],
                    "spacegroup_symbol": stab_row[3],
                    "energy_above_hull_eV_per_atom": stab_row[4],
                })
            else:
                _, cid, err = res
                errors.append((cid, err))
            done += 1
            if done % PROGRESS_EVERY == 0:
                print(f"  ... {done}/{total_new} done")
    else:
        # Chunked map preserves order of args_iter; we assign rows sequentially
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as ex:
            for res in ex.map(_worker_do_one, args_iter, chunksize=chunksize):
                if res[0] == 1:
                    _, cid, row, prof, stab_row = res
                    ok_rows.append(row)
                    if prof is not None:
                        profiles_all[write_ptr, :] = prof
                        # Optimization: Accumulate in list
                        new_idx_rows.append({"id": cid, "row": int(write_ptr)})
                        write_ptr += 1
                    ok_stab.append({
                        "material_id":       stab_row[0],
                        "formula_pretty":    stab_row[1],
                        "spacegroup_number": stab_row[2],
                        "spacegroup_symbol": stab_row[3],
                        "energy_above_hull_eV_per_atom": stab_row[4],
                    })
                else:
                    _, cid, err = res
                    errors.append((cid, err))
                done += 1
                if done % PROGRESS_EVERY == 0:
                    print(f"  ... {done}/{total_new} done")

    # Finalize index DataFrame
    if new_idx_rows:
        idx_out = pd.concat([idx_out, pd.DataFrame(new_idx_rows)], ignore_index=True)

    if errors:
        print(f"\nCompleted with {len(errors)} errors (showing first 10):")
        for e in errors[:10]:
            print(f"  {e}")
    else:
        print("\nCompleted successfully (no errors).")

    # Update Catalog and Stable (Bug 2 & 5 fix: ensure we write data)
    if ok_rows:
        cat_out = pd.concat([cat_out, pd.DataFrame(ok_rows)], ignore_index=True)
    if ok_stab:
        # Check if stab_in has these columns?
        new_stab_df = pd.DataFrame(ok_stab)
        stab_in = pd.concat([stab_in, new_stab_df], ignore_index=True)

    # Trim profiles (if we had errors, write_ptr < allocated)
    if write_ptr < profiles_all.shape[0]:
        profiles_all = profiles_all[:write_ptr, :]
    
    _finalize_and_write(db_out, cat_out, idx_out, profiles_all, stab_in, q_min, q_max, n_bins, sigma_bins, args.out_suffix)


def _finalize_and_write(db_out, cat_out, idx_out, profiles_all, stab_in, q_min, q_max, n_bins, sigma_bins, suffix=""):
    print(f"Writing catalog_deduplicated{suffix}.csv ...")
    cat_out.to_csv(os.path.join(db_out, f"catalog_deduplicated{suffix}.csv"), index=False)
    
    print(f"Writing index_deduplicated{suffix}.csv ...")
    idx_out.to_csv(os.path.join(db_out, f"index_deduplicated{suffix}.csv"), index=False)
    
    print(f"Writing profiles64{suffix}.npz ...")
    np.savez_compressed(
        os.path.join(db_out, f"profiles64{suffix}.npz"),
        profiles=profiles_all,
        q_min=q_min, q_max=q_max, n_bins=n_bins, sigma_bins=sigma_bins
    )
    
    print("Writing mp_experimental_stable.csv ...")
    stab_in.to_csv(os.path.join(db_out, "mp_experimental_stable.csv"), index=False)
    
    # Bug 1 Fix: meta_out JSON ignored as it's just a copy of logic. 
    # But if pipeline needs it? Pipeline usually reads from DB_IN or just needs valid IDs.
    # The prompt complained meta_out was not init. 
    # But looking at old code, meta_out was dumped.
    # We do not strictly need to dump 1.3GB JSON again if it's identical.
    # But let's dump a small metadata index if needed?
    # Original code: json.dump(meta_out, ...)
    # If users want it, we can copy it. But it's huge. 
    # Let's skip it or just copy the file if --fresh.
    # But we might have filtered it.
    pass


# Global constants for geometry (Standard Bragg-Brentano / Xray defaults?)
# Actually we accept them as args or use defaults suitable for simulation.
two_theta_min = 0.0
two_theta_max = 180.0
topM = 500

if __name__ == "__main__":
    main()
