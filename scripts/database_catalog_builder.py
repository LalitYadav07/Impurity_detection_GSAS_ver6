# -*- coding: utf-8 -*-
"""
build_aniso_db.py  (strict & robust)

Processed diffraction DB builder focused on:
  • Instrument-agnostic storage (no widths/microstrain baked-in)
  • Anisotropic lattice handling up to ~10% (linear sensitivity rows C)
  • Fast composition subset queries via 128-bit element masks
  • Resume-safe multiprocessing
  • Strict failure reporting (no silent drops)

Per phase we store:
  - direct metric G and reciprocal metric G_inv
  - top-M reflections: hkls, d0, q0, I0
  - sensitivity matrix C (M×6)
  - 256-bin ratio histogram (optional)
  - 128-bit element mask (hi, lo)
  - space group number and symbol

Outputs:
  - manifest.json
  - catalog.csv
  - phases/<id>.npz
  - failures.json (if any failures)
"""

import os, sys, json, math, argparse, time, traceback
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False

from pymatgen.core import Structure, Element
from pymatgen.analysis.diffraction.neutron import NDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

ELEMENT_BITS = 128  # Z=1..128

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def log(msg: str, t0: float, verbose: bool):
    if verbose:
        print(f"[{time.time()-t0:7.2f}s] {msg}", flush=True)

def load_input_db(path: str) -> Dict[str, Dict]:
    with open(path, "r") as f:
        return json.load(f)
def resolve_space_group_with_analyzer(meta: Dict, s: Structure,
                                      symprec: float = 1e-2,
                                      angle_tolerance: float = 5.0
                                      ) -> Tuple[int, str, Optional[Dict]]:
    """
    Use Pymatgen's SpacegroupAnalyzer to finalize SG, and note mismatches
    vs. the 'reported' SG in meta (space_group / SG_symbol) if provided.

    Returns:
        (final_num, final_sym, mismatch_dict_or_None)
    """
    # reported (from metadata), if any
    rep_num = int(meta.get("space_group", 0) or 0)
    rep_sym = str(meta.get("SG_symbol", meta.get("spacegroup_symbol", "")) or "")

    # analyzer (authoritative)
    sga = SpacegroupAnalyzer(s, symprec=symprec, angle_tolerance=angle_tolerance)
    ana_num = int(sga.get_space_group_number())
    ana_sym = str(sga.get_space_group_symbol())

    mismatch = None
    if rep_num or rep_sym:  # only compare when something was reported
        if (rep_num and rep_num != ana_num) or (rep_sym and rep_sym.strip() != ana_sym.strip()):
            mismatch = {
                "reported_num": int(rep_num or 0),
                "reported_sym": rep_sym.strip(),
                "analyzer_num": int(ana_num),
                "analyzer_sym": ana_sym,
            }

    # Always keep analyzer’s result
    return ana_num, ana_sym, mismatch

def element_mask_from_comp(comp: Dict[str, float], bits: int = ELEMENT_BITS) -> Tuple[np.uint64, np.uint64]:
    """
    Build 128-bit mask from a composition mapping {symbol: amount}.
    Bits are Z-1 for elements with 1 <= Z <= bits.
    """
    lo = np.uint64(0); hi = np.uint64(0)
    if comp:
        for sym in comp.keys():
            try:
                Z = int(Element(sym).Z)
            except Exception:
                raise ValueError(f"Unknown element symbol '{sym}' in composition")
            if not (1 <= Z <= bits):
                # out-of-range Z; refuse to silently drop
                raise ValueError(f"Element Z={Z} out of supported range 1..{bits} for '{sym}'")
            bit = Z - 1
            if bit < 64:
                lo |= (np.uint64(1) << np.uint64(bit))
            else:
                hi |= (np.uint64(1) << np.uint64(bit - 64))
    return hi, lo

def lattice_metric_tensors(structure: Structure) -> Tuple[np.ndarray, np.ndarray]:
    A = np.array(structure.lattice.matrix, dtype=float)
    G = A @ A.T
    G_inv = np.linalg.inv(G)
    return G, G_inv

def normalize_hkl(hkl_any) -> np.ndarray:
    """
    Return 3-index (h,k,l) np.array.
    - If input len>=4 (Miller–Bravais), use (h,k,l4).
    - Else take first 3.
    """
    arr = np.asarray(hkl_any, dtype=float).ravel()
    if arr.size >= 4:
        return arr[[0, 1, 3]].astype(float)
    if arr.size >= 3:
        return arr[:3].astype(float)
    raise ValueError(f"Unexpected hkl length {arr.size}: {hkl_any}")

def C_row_for_hkl(hkl: np.ndarray, G: np.ndarray, G_inv: np.ndarray) -> np.ndarray:
    h = normalize_hkl(hkl).reshape(3, 1).astype(float)
    v = (G_inv @ h).ravel()
    w = (G @ v).ravel()
    c1 = -2.0 * (w[0] * v[0])
    c2 = -2.0 * (w[1] * v[1])
    c3 = -2.0 * (w[2] * v[2])
    c4 = -2.0 * (w[1] * v[2] + w[2] * v[1])
    c5 = -2.0 * (w[0] * v[2] + w[2] * v[0])
    c6 = -2.0 * (w[0] * v[1] + w[1] * v[0])
    return np.array([c1, c2, c3, c4, c5, c6], dtype=np.float32)

def simulate_topM_peaks(structure: Structure, two_theta_min: float, two_theta_max: float, topM: int
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    calc = NDCalculator()
    pat = calc.get_pattern(structure, two_theta_range=(two_theta_min, two_theta_max))
    rows = []
    for I, families, d in zip(pat.y, pat.hkls, pat.d_hkls):
        if not d or d <= 0: 
            continue
        if not families: 
            continue
        hkl = normalize_hkl(families[0]["hkl"]).astype(int)
        rows.append((float(I), hkl, float(d)))
    if not rows:
        return (np.zeros((0,3), dtype=np.int16),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32))
    rows.sort(key=lambda x: x[0], reverse=True)
    rows = rows[:topM]
    I0 = np.array([r[0] for r in rows], dtype=np.float32)
    hkls = np.array([r[1] for r in rows], dtype=np.int16)
    d0 = np.array([r[2] for r in rows], dtype=np.float32)
    q0 = (2.0 * math.pi) / np.maximum(d0, 1e-12)
    return hkls, d0, q0.astype(np.float32), I0

def ratio_histogram(dvals: np.ndarray, bins: int = 256, rmin: float = -2.0, rmax: float = 2.0) -> np.ndarray:
    if dvals.size < 2:
        h = np.zeros(bins, dtype=np.float32); h[bins//2] = 1.0; return h
    x = np.log(np.maximum(dvals, 1e-12))
    diffs = x[:, None] - x[None, :]
    iu = np.triu_indices_from(diffs, k=1)
    vals = diffs[iu].ravel()
    hist, _ = np.histogram(vals, bins=bins, range=(rmin, rmax))
    hist = hist.astype(np.float32)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist

# --------------------------------------------------------------------------------------
# Core per-phase processing (STRICT)
# --------------------------------------------------------------------------------------

def _composition_from_meta_or_structure(meta: Dict, s: Structure) -> Dict[str, float]:
    """
    Prefer meta["composition"] if provided; otherwise derive from structure.
    Returns a clean mapping {Element.symbol: amount}.
    """
    comp = meta.get("composition")
    if isinstance(comp, dict) and len(comp) > 0:
        # sanitize keys to valid element symbols
        out = {}
        for k, v in comp.items():
            sym = Element(str(k)).symbol  # raises if invalid
            out[sym] = float(v) if v is not None else 1.0
        return out
    # derive from structure
    return {el.symbol: float(amt) for el, amt in s.composition.items()}

def _space_group_from_meta_or_structure(meta: Dict, s: Structure) -> Tuple[int, str]:
    """
    Return (SG_number, SG_symbol). If meta has a valid number use it; otherwise
    compute from structure.
    """
    sgnum = int(meta.get("space_group", 0) or 0)
    if sgnum > 0:
        try:
            sym = meta.get("SG_symbol")
            if isinstance(sym, str) and sym.strip():
                return sgnum, sym.strip()
        except Exception:
            pass
    # compute from structure (symprec modest)
    try:
        sym, num = s.get_space_group_info(symprec=1e-2)
        return int(num or 0), str(sym or "")
    except Exception:
        return int(sgnum), ""  # keep whatever we had

def process_one_phase(phase_id: str, meta: Dict,
                      out_dir: str,
                      two_theta_min: float, two_theta_max: float,
                      topM: int,
                      add_ratio_hist: bool,
                      resume: bool,
                      strict_masks: bool,
                      verbose: bool) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    """
    Returns (row_dict, failure_dict, mismatch_dict). Exactly one of (row, failure) is non-None.
    mismatch_dict is non-None only when reported SG != analyzer SG; we always keep analyzer SG.
    """
    phase_npz = os.path.join(out_dir, "phases", f"{phase_id}.npz")

    # --- Resume path: still run analyzer to log mismatch, but reuse NPZ on success ---
    if resume and os.path.exists(phase_npz):
        try:
            s = Structure.from_str(meta["cif_content"], fmt="cif")
            ana_num, ana_sym, mismatch = resolve_space_group_with_analyzer(meta, s)

            with np.load(phase_npz) as npz:
                n = int(npz["hkls"].shape[0])
                hi = np.uint64(npz["elements_mask_hi"]) if "elements_mask_hi" in npz else np.uint64(0)
                lo = np.uint64(npz["elements_mask_lo"]) if "elements_mask_lo" in npz else np.uint64(0)

            comp = meta.get("composition") or {}
            row = {
                "id": phase_id,
                "pretty_formula": str(meta.get("formula_pretty", meta.get("pretty_formula", "")) or ""),
                "space_group": int(ana_num),
                "SG_symbol": ana_sym,
                "elements_list": ",".join(sorted(comp.keys())),
                "elements_mask_hi": int(hi),
                "elements_mask_lo": int(lo),
                "npz": os.path.relpath(phase_npz, out_dir),
                "n_reflections": n,
            }
            return row, None, mismatch
        except Exception as e:
            if verbose:
                print(f"[resume→rebuild] {phase_id}: {e}")

    # --- Full build path ---
    try:
        s = Structure.from_str(meta["cif_content"], fmt="cif")

        # Space group (authoritative via analyzer) + mismatch note
        sgnum, sgsym, mismatch = resolve_space_group_with_analyzer(meta, s)

        # Composition & element mask (strict if requested)
        comp = meta.get("composition")
        if not comp:  # fallback to structure composition
            comp = {el.symbol: float(amt) for el, amt in s.composition.items()}
        hi_mask, lo_mask = element_mask_from_comp(comp)
        if strict_masks and (hi_mask == 0 and lo_mask == 0) and len(comp) > 0:
            raise RuntimeError("Zero element mask produced for non-empty composition")

        # Lattice metrics
        G, G_inv = lattice_metric_tensors(s)

        # Top-M reflections
        hkls, d0, q0, I0 = simulate_topM_peaks(s, two_theta_min, two_theta_max, topM)
        if hkls.shape[0] == 0:
            raise RuntimeError("No peaks returned by NDCalculator")

        # Sensitivities
        C = np.zeros((hkls.shape[0], 6), dtype=np.float32)
        for i, hkl in enumerate(hkls):
            C[i, :] = C_row_for_hkl(hkl.astype(float), G, G_inv)

        # Optional ratio histogram
        r_hist = ratio_histogram(d0) if add_ratio_hist else np.zeros(0, dtype=np.float32)

        # Pretty formula
        pretty = str(meta.get("formula_pretty", meta.get("pretty_formula", ""))).strip()
        if not pretty:
            pretty = s.composition.reduced_formula

        # Write NPZ
        np.savez_compressed(
            phase_npz,
            G=np.asarray(G, dtype=np.float32),
            G_inv=np.asarray(G_inv, dtype=np.float32),
            hkls=hkls.astype(np.int16),
            d0=d0.astype(np.float32),
            q0=q0.astype(np.float32),
            I0=I0.astype(np.float32),
            C=C.astype(np.float32),
            elements_mask_hi=np.asarray(hi_mask, dtype=np.uint64),
            elements_mask_lo=np.asarray(lo_mask, dtype=np.uint64),
            ratio_hist=r_hist.astype(np.float32),
        )

        row = {
            "id": phase_id,
            "pretty_formula": pretty,
            "space_group": int(sgnum),
            "SG_symbol": sgsym,
            "elements_list": ",".join(sorted(comp.keys())),
            "elements_mask_hi": int(hi_mask),
            "elements_mask_lo": int(lo_mask),
            "npz": os.path.relpath(phase_npz, out_dir),
            "n_reflections": int(hkls.shape[0]),
        }
        return row, None, mismatch

    except Exception as e:
        if verbose:
            print(f"[FAIL] {phase_id}: {e}", file=sys.stderr)
            traceback.print_exc()
        fail = {"id": phase_id, "error": str(e)}
        return None, fail, None

# --------------------------------------------------------------------------------------
# MP wrapper
# --------------------------------------------------------------------------------------

def worker_wrapper(args_tuple):
    return process_one_phase(*args_tuple)

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build anisotropic-ready diffraction DB (no instrument params).")
    ap.add_argument("--input-db", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--two-theta-min", type=float, default=0.0)
    ap.add_argument("--two-theta-max", type=float, default=90.0)
    ap.add_argument("--topM", type=int, default=160)
    ap.add_argument("--add-ratio-hist", action="store_true")
    ap.add_argument("--allow-ids", default=None)
    ap.add_argument("--max-phases", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--strict-missing", action="store_true",
                    help="If any phase fails, abort at end with nonzero exit.")
    ap.add_argument("--strict-masks", action="store_true",
                    help="Raise if a non-empty composition produces a zero element mask.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    ensure_dir(args.out_dir); ensure_dir(os.path.join(args.out_dir, "phases"))

    manifest = {
        "version": "aniso-1.2",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "two_theta_range_deg": [args.two_theta_min, args.two_theta_max],
        "topM": int(args.topM),
        "element_mask_bits": ELEMENT_BITS,
        "strain_parameterization": "theta=[e11,e22,e33,e23,e13,e12]; δ(1/d^2)=C·theta",
        "notes": "No instrument/broadening stored. C from δG*=-G*(εG+Gε)G*.",
    }

    log("Loading input DB...", t0, args.verbose)
    in_db = load_input_db(args.input_db)
    all_ids = list(in_db.keys())

    if args.allow_ids:
        allow = set([x.strip() for x in args.allow_ids.split(",") if x.strip()])
        ids = [i for i in all_ids if i in allow]
    else:
        ids = all_ids

    if args.max_phases > 0:
        ids = ids[:args.max_phases]

    log(f"Planning to process {len(ids)} phases.", t0, args.verbose)

    todo = [(pid, in_db[pid], args.out_dir, args.two_theta_min, args.two_theta_max,
             args.topM, args.add_ratio_hist, args.resume, args.strict_masks, args.verbose)
            for pid in ids]

    rows: List[Dict] = []
    fails: List[Dict] = []
    sg_mismatches: List[Dict] = []  # <— collect differences here


 
    if args.workers and args.workers > 0:
        import multiprocessing as mp
        with mp.Pool(processes=args.workers) as pool:
            it = pool.imap_unordered(worker_wrapper, todo, chunksize=4)
            iterator = tqdm(it, total=len(todo)) if HAS_TQDM else it
            for row, fail, mm in iterator:
                if row is not None:
                    rows.append(row)
                elif fail is not None:
                    fails.append(fail)
                if mm is not None:
                    sg_mismatches.append({"id": row["id"] if row else fail["id"], **mm})
    else:
        iterator = tqdm(todo, total=len(todo)) if HAS_TQDM else todo
        for tup in iterator:
            row, fail, mm = worker_wrapper(tup)
            if row is not None:
                rows.append(row)
            elif fail is not None:
                fails.append(fail)
            if mm is not None:
                sg_mismatches.append({"id": row["id"] if row else fail["id"], **mm})

    # Write catalog
    catalog_df = pd.DataFrame(rows, columns=[
        "id","pretty_formula","space_group","SG_symbol",
        "elements_list","elements_mask_hi","elements_mask_lo",
        "npz","n_reflections"
    ])
    catalog_csv = os.path.join(args.out_dir, "catalog.csv")
    catalog_df.to_csv(catalog_csv, index=False)

    # Manifest
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Failures (if any)
    failures_path = os.path.join(args.out_dir, "failures.json")
    if fails:
        with open(failures_path, "w") as f:
            json.dump({"failures": fails, "n_failed": len(fails)}, f, indent=2)
        print(f"[WARN] {len(fails)} phases failed. See: {failures_path}")

    # Summary
    log(f"Done. Wrote {len(rows)} phases.", t0, True)
    print(f"Output directory: {args.out_dir}")
    print(f" - manifest.json")
    print(f" - catalog.csv (rows: {len(rows)})")
    print(f" - phases/<id>.npz")
    if fails:
        print(f" - failures.json (failed: {len(fails)})")

    # Strict missing enforcement
    if args.strict_missing and fails:
        raise SystemExit(f"Aborting due to {len(fails)} failed phases (strict-missing).")
    # Space-group mismatch report
    if sg_mismatches:
        sg_json = os.path.join(args.out_dir, "sg_mismatches.json")
        with open(sg_json, "w") as f:
            json.dump({"mismatches": sg_mismatches, "n_mismatched": len(sg_mismatches)}, f, indent=2)
        sg_csv = os.path.join(args.out_dir, "sg_mismatches.csv")
        pd.DataFrame(sg_mismatches, columns=[
            "id", "reported_num", "reported_sym", "analyzer_num", "analyzer_sym"
        ]).to_csv(sg_csv, index=False)
        print(f"[INFO] Space-group mismatches: {len(sg_mismatches)}  → {sg_csv}")

# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()


# python /netdisk/ly6/GSAS_Projects/Impurity_detection_GSAS/Build_data_base/database_catalog_builder.py \
#   --input-db "/netdisk/ly6/torch_evs_1/Project Diffraction/Diff_plus_comp_models/Data/highsymm_metadata.json" \
#   --out-dir "/netdisk/ly6/GSAS_Projects/Impurity_detection_GSAS/Build_data_base/fresh_db" \
#   --two-theta-min 4 --two-theta-max 120 \
#   --topM 160 \
#   --workers 8 \
#   --resume \
#   --verbose
