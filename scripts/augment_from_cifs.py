#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build .../data/database_aug (hybrid catalog; new phases only under OUT/phases)

Key points (fast version):
- Parallel processing via ProcessPoolExecutor with chunked map (low overhead).
- Sensible worker cap via _pick_workers() (respects SLURM).
- BLAS/OMP threads clamped to 1 in workers to avoid oversubscription.
- Profiles64 handled via single preallocated array → fast, no vstack loops.
- Minimal console chatter (progress every PROGRESS_EVERY items).

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
# ========================

# ===== Defaults for your layout =====
PROJECT_ROOT_DEFAULT = "/netdisk/ly6/GSAS_Projects/Impurity_detection_GSAS_ver2"
DATA_ROOT_DEFAULT    = f"{PROJECT_ROOT_DEFAULT}/data"
DB_IN_DEFAULT        = f"{DATA_ROOT_DEFAULT}/database"
DB_OUT_DEFAULT       = f"{DATA_ROOT_DEFAULT}/database_aug"

CIF_DIR_DEFAULT      = f"{DB_IN_DEFAULT}/augment_cif"
JSON_FILE_DEFAULT    = f"{DB_OUT_DEFAULT}/cod_highsymm_metadata_deduplicated.json"
# ====================================

# Import your strict builder
from database_catalog_builder import process_one_phase

# ---------- Utils ----------
def read_text(p): return open(p, "r", encoding="utf-8").read()
def write_text(p, s): open(p, "w", encoding="utf-8").write(s)
def ensure_dir(d): os.makedirs(d, exist_ok=True)

def _pick_workers(n_jobs: int) -> int:
    """
    Choose a sensible process count:
    - Respect SLURM_CPUS_PER_TASK / SLURM_CPUS_ON_NODE
    - Cap to avoid FS thrash
    - Not more than number of jobs
    """
    slurm = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE")
    avail = int(slurm) if slurm else (os.cpu_count() or 1)
    cap   = 64  # adjust if your FS is very fast; 16–64 is typical
    return max(8, min(cap, avail, max(1, n_jobs)))

def _worker_init():
    
    # Avoid BLAS/OMP oversubscription inside each process
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

def _worker_do_one(args):
    """
    Run in a subprocess. Build one phase and its 64-bin profile.
    Returns a compact tuple to reduce IPC overhead.
    """
    (cid, cif_text, db_out,
     two_theta_min, two_theta_max, topM,
     q_min, q_max, n_bins, sigma_bins) = args

    meta = {"cif_content": cif_text}

    # Import here to keep top-level import minimal in workers
    import numpy as _np
    import os as _os
    import math as _math

    # local copy of gaussian/profile builders (kept minimal)
    def _gaussian_deposit(profile, center_bin, weight, sigma_bins):
        L = profile.shape[0]
        if sigma_bins <= 0:
            j = int(round(center_bin))
            if 0 <= j < L:
                profile[j] += weight
            return
        halfw = int(max(1, _math.ceil(3.0 * sigma_bins)))
        j0 = int(round(center_bin))
        jL = max(0, j0 - halfw); jR = min(L - 1, j0 + halfw)
        if jR < jL: return
        js = _np.arange(jL, jR + 1, dtype=_np.float32)
        g = _np.exp(-0.5 * ((js - center_bin) / sigma_bins) ** 2)
        profile[jL:jR + 1] += weight * g

    def _build_profile_for_npz(npz_path, q_min, q_max, n_bins=64, sigma_bins=0.7):
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
            _gaussian_deposit(prof, pos, w, sigma_bins)
        m = float(prof.max())
        if m > 0:
            prof /= m
        return prof.astype(_np.float16)

    try:
        row, fail, mm = process_one_phase(
            cid, meta, db_out,
            two_theta_min, two_theta_max, topM,
            False, False, False, False
        )
        npz_path = os.path.join(db_out, row["npz"])
        prof = _build_profile_for_npz(npz_path, q_min, q_max, n_bins, sigma_bins)

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
    args = ap.parse_args()

    db_in   = args.db_in
    db_out  = args.db_out
    cifdir  = args.cif_dir
    jsonfp  = args.json_file

    # ---------- Preflight ----------
    print("== Preflight ==")
    print(f"Python: {platform.python_version()}")
    print(f"DB_IN:  {db_in}")
    print(f"DB_OUT: {db_out}")
    print(f"MODE:   {'JSON' if USE_JSON_SOURCE else 'CIF-FOLDER'}")

    assert os.path.isdir(db_in), f"Missing DB_IN: {db_in}"
    man_in    = os.path.join(db_in,  "manifest.json")
    cat_in_p  = os.path.join(db_in,  "catalog.csv")
    meta_in_p = os.path.join(db_in,  "highsymm_metadata.json")
    phases_in = os.path.join(db_in,  "phases")
    p64_in_d  = os.path.join(db_in,  "profiles64")
    p64_in_p  = os.path.join(p64_in_d, "profiles64.npz")
    idx_in_p  = os.path.join(p64_in_d, "index.csv")
    stab_in_p = os.path.join(db_in,  "mp_experimental_stable.csv")

    for pth in (man_in, cat_in_p, meta_in_p, p64_in_p, idx_in_p, stab_in_p):
        assert os.path.isfile(pth), f"Missing required file: {pth}"
    assert os.path.isdir(phases_in), f"Missing phases folder: {phases_in}"

    warnings.simplefilter("ignore", pd.errors.DtypeWarning)
    cat_in   = pd.read_csv(cat_in_p, low_memory=False)
    meta_in  = json.load(open(meta_in_p, "r"))
    P_in     = np.load(p64_in_p, allow_pickle=False)  # NpzFile; profiles loaded below
    idx_in   = pd.read_csv(idx_in_p)
    stab_in  = pd.read_csv(stab_in_p)

    # Prepare OUT tree
    ensure_dir(db_out)
    phases_out = os.path.join(db_out, "phases")
    p64_out_d  = os.path.join(db_out, "profiles64")
    ensure_dir(phases_out); ensure_dir(p64_out_d)
    man_out    = os.path.join(db_out, "manifest.json")
    shutil.copy2(man_in, man_out)

    # Manifest params
    manifest = json.load(open(man_in, "r"))
    two_theta_min = float(manifest["two_theta_range_deg"][0])
    two_theta_max = float(manifest["two_theta_range_deg"][1])
    topM          = int(manifest["topM"])

    # Originals as base
    cat_out  = cat_in.copy()
    meta_out = dict(meta_in)

    # Pull constants for profile building (keep small)
    q_min    = float(P_in["q_min"])
    q_max    = float(P_in["q_max"])
    n_bins   = int(P_in["n_bins"])
    sigma_bins = float(P_in["sigma_bins"])

    # Load existing profiles into RAM once (float16 keeps this tiny)
    profiles_old = P_in["profiles"].astype(np.float16)
    old_rows, bin_cols = profiles_old.shape

    idx_out  = idx_in.copy()
    stab_out = stab_in.copy()

    original_ids = set(cat_in["id"].astype(str))
    print(f"Original catalog rows:  {len(cat_in)}")
    print(f"Original profiles rows: {old_rows}  bins: {bin_cols}")
    print(f"Q-range/bins/sigma:     [{q_min}, {q_max}]  n_bins={n_bins}  sigma={sigma_bins}")

    # ---------- Source CIFs ----------
    if USE_JSON_SOURCE:
        print(f"JSON file: {jsonfp}")
        assert os.path.isfile(jsonfp), f"JSON not found: {jsonfp}"
        with open(jsonfp, "r", encoding="utf-8") as f:
            cif_json = json.load(f)
        src_items = []
        for cid, entry in cif_json.items():
            cif_text = ""
            if isinstance(entry, dict):
                cif_text = entry.get("cif_content", "") or entry.get("cif_Content", "")
            elif isinstance(entry, str):
                cif_text = entry
            if cif_text and str(cif_text).strip():
                src_items.append((str(cid), str(cif_text)))
        print(f"Found CIFs in JSON: {len(src_items)}")
        assert src_items, "No usable 'cif_content' entries in JSON"
    else:
        print(f"CIF_DIR: {cifdir}")
        assert os.path.isdir(cifdir), f"Directory not found: {cifdir}"
        cif_paths = sorted(glob.glob(os.path.join(cifdir, "*.cif")))
        print(f"Found CIF files: {len(cif_paths)}")
        assert cif_paths, "No .cif files found"
        src_items = [(os.path.splitext(os.path.basename(p))[0], read_text(p)) for p in cif_paths]

    # Filter out already-present ids (in original catalog or already built in OUT)
    existing_out_ids = {os.path.splitext(os.path.basename(p))[0]
                        for p in glob.glob(os.path.join(phases_out, "*.npz"))}
    work_items = []
    for cid, cif_text in src_items:
        if (cid in original_ids) or (cid in existing_out_ids):
            continue
        work_items.append((cid, cif_text))
        # Keep metadata text only for new ones
        meta_out[cid] = {"cif_content": cif_text}

    total_new = len(work_items)
    print("\n== Processing new CIFs in parallel ==")
    print(f"To add: {total_new} new phases")
    if total_new == 0:
        # Still rewrite outputs for hybrid paths, then exit
        return _finalize_and_write(db_in, db_out, cat_in, cat_out, original_ids,
                                   profiles_old, q_min, q_max, n_bins, sigma_bins,
                                   idx_out, stab_out, meta_out)

    # Preallocate final profiles array (old + upper bound new)
    profiles_all = np.empty((old_rows + total_new, bin_cols), dtype=np.float16)
    profiles_all[:old_rows, :] = profiles_old
    write_ptr = old_rows  # next free row for a new profile

    # Prepare worker args
    args_iter = [
        (cid, cif_text, db_out,
         two_theta_min, two_theta_max, topM,
         q_min, q_max, n_bins, sigma_bins)
        for (cid, cif_text) in work_items
    ]

    # Choose workers and chunks
    workers = _pick_workers(len(args_iter))
    chunksize = max(1, len(args_iter) // (workers * 8))
    print(f"Workers: {workers}  |  Chunksize: {chunksize}")

    ok_rows    = []     # rows to append into catalog
    ok_stab    = []     # rows to append into stable CSV
    errors     = []
    done       = 0

    # Chunked map preserves order of args_iter; we assign rows sequentially
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as ex:
        for res in ex.map(_worker_do_one, args_iter, chunksize=chunksize):
            if res[0] == 1:
                _, cid, row, prof, stab_row = res
                ok_rows.append(row)
                if prof is not None:
                    profiles_all[write_ptr, :] = prof
                    # idx_out row points to the appended row in profiles_all
                    idx_out = pd.concat([idx_out, pd.DataFrame([{"id": cid, "row": int(write_ptr)}])], ignore_index=True)
                    write_ptr += 1
                # stab_row is a compact tuple; expand once
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

    if errors:
        print(f"\nCompleted with {len(errors)} errors (showing first 10):")
        for cid, err in errors[:10]:
            print(f"  [err] {cid}: {err}")

    # Append successful rows in one go
    if ok_rows:
        cat_out = pd.concat([cat_out, pd.DataFrame(ok_rows)], ignore_index=True)
    if ok_stab:
        present = set(stab_out["material_id"].astype(str))
        to_add = [r for r in ok_stab if r["material_id"] not in present]
        if to_add:
            stab_out = pd.concat([stab_out, pd.DataFrame(to_add)], ignore_index=True)

    # Trim unused tail if some structures had no profile / failed
    profiles_final = profiles_all[:write_ptr, :]

    # Write all outputs
    _finalize_and_write(
        db_in, db_out, cat_in, cat_out, original_ids,
        profiles_final, q_min, q_max, n_bins, sigma_bins,
        idx_out, stab_out, meta_out
    )

def _finalize_and_write(
    db_in, db_out, cat_in, cat_out, original_ids,
    profiles_final, q_min, q_max, n_bins, sigma_bins,
    idx_out, stab_out, meta_out
):
    # Hybridize npz paths in catalog.csv
    print("\n== Rewriting catalog npz paths (hybrid) ==")
    cat_out_p   = os.path.join(db_out, "catalog.csv")
    meta_out_p  = os.path.join(db_out, "highsymm_metadata.json")
    p64_out_d   = os.path.join(db_out, "profiles64")
    p64_out_p   = os.path.join(p64_out_d, "profiles64.npz")
    idx_out_p   = os.path.join(p64_out_d, "index.csv")
    stab_out_p  = os.path.join(db_out, "mp_experimental_stable.csv")
    readme_p    = os.path.join(db_out, "README.txt")

    ensure_dir(p64_out_d)

    if "npz" in cat_in.columns:
        orig_npz_map = dict(zip(cat_in["id"].astype(str), cat_in["npz"].astype(str)))
    else:
        orig_npz_map = {pid: f"phases/{pid}.npz" for pid in cat_in["id"].astype(str)}

    npz_adjusted = []
    for pid, npz_rel in zip(cat_out["id"].astype(str), cat_out["npz"].astype(str)):
        if pid in original_ids:
            orig_rel = orig_npz_map.get(pid, npz_rel)
            npz_adjusted.append(os.path.join(db_in, orig_rel))
        else:
            npz_adjusted.append(npz_rel)
    cat_out["npz"] = npz_adjusted

    print("\n== Writing database_aug files ==")
    cat_out.to_csv(cat_out_p, index=False)
    write_text(meta_out_p, json.dumps(meta_out, indent=2))

    # Save profiles64 (final, trimmed) without keeping extra copies in RAM
    np.savez_compressed(
        p64_out_p,
        profiles=profiles_final.astype(np.float16, copy=False),
        q_min=np.float32(q_min), q_max=np.float32(q_max),
        n_bins=np.int16(n_bins), sigma_bins=np.float32(sigma_bins)
    )
    idx_out.to_csv(idx_out_p, index=False)
    stab_out.to_csv(stab_out_p, index=False)

    # Summary
    P_old = int((cat_in.shape[0]))
    P_new = int((cat_out.shape[0]))
    summary = []
    summary.append("Augmented DB build summary")
    summary.append("==========================")
    summary.append(f"Timestamp (UTC): {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}")
    summary.append(f"Source DB:       {db_in}")
    summary.append(f"Output DB:       {db_out}")
    summary.append("")
    summary.append(f"Mode:            {'JSON' if USE_JSON_SOURCE else 'CIF-FOLDER'}")
    summary.append(f"Catalog rows:    {P_old}  ->  {P_new}")
    summary.append(f"Profiles rows:   {profiles_final.shape[0]} (bins={profiles_final.shape[1]})")
    summary.append(f"Stable rows:     {len(stab_out)}")
    text = "\n".join(summary)
    print("\n== Summary ==\n" + text)
    write_text(readme_p, text + "\n")

# Prefer 'fork' on Linux (fast); ignore error if already set
if __name__ == "__main__":
    try:
        import multiprocessing as mp
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("fork")
    except Exception:
        pass
    main()
