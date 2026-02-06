#!/usr/bin/env python3
"""
Legacy Bridge (clean) — ML-histogram only

Integrates GSAS-II pipeline with impurity detection components:
- Element mask filtering (bit-mask, vectorized)
- Space-group pruning (drop low-symmetry SGs after element filter)
- ML histogram screening (64-bin, unified plotting)
- (Optional) stability filtering against a "stable" catalog

Removed legacy/stale pieces:
- shortlist_by_hist_underfill_abs64  (we use ML-only)
- align_score_candidate / screen_by_alignment
- peak_positions / residual-peak finder (alignment removed)
- BoxCapResult data class
- plot_main_refinement_figure and related GSAS plotting helpers
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

# --- Required pipeline components (raise if missing) ---
from aniso_db_loader import DBLoader, CatalogPaths, build_mask  # fast mask filters

# IMPORTANT: file is named "ratio_filter.py"
from ratio_filter import shortlist_by_hist_ML, _load_profiles64_metadata


# --------------------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------------------

@dataclass
class CandidatePhase:
    phase_id: str
    cif_file: Optional[str]
    cif_text: Optional[str]
    element_mask_score: float
    overlap_score: int
    histogram_score: float
    initial_scale: float = 0.05
    # ML diagnostics are attached dynamically as attributes:
    #   ml_score, ml_alpha, ml_explained, ml_cosine, ml_present_prob, ml_beta (not used)


# --------------------------------------------------------------------------------------
# Knee utilities (self-contained: used for stage-0 within this module)
# --------------------------------------------------------------------------------------

def _fmt_list(ids, values=None, limit=20):
    if not ids:
        return "[]"
    if values is None:
        s = ", ".join(ids[:limit])
    else:
        s = ", ".join(f"{pid}({values[i]:.4g})" for i, pid in enumerate(ids[:limit]))
    return f"[{s}{' …' if len(ids) > limit else ''}]"

def _knee_keep_ids(items, id_fn, val_fn, *, label:str,
                   min_points:int, min_rel_span:float, guard_frac:float,
                   max_keep_if_no_knee:int=0, min_keep_at_least:int=0, max_keep_at_most:int=0):
    rows = []
    for it in items:
        try:
            pid = str(id_fn(it))
            v = float(val_fn(it))
            if np.isfinite(v):
                rows.append((pid, v))
        except Exception:
            continue
    rows.sort(key=lambda r: r[1], reverse=True)
    n = len(rows)
    if n == 0:
        print(f"[KNEE] {label}: no finite values → keep 0/0")
        return []

    v0, vN = rows[0][1], rows[-1][1]
    span = abs(v0 - vN)

    def _fallback(reason):
        k = max_keep_if_no_knee or n
        kept = [pid for pid, _ in rows[:k]]
        print(f"[KNEE] {label}: {reason} (n={n}, span≈{span:.4g}) → keep {len(kept)}/{n} { _fmt_list(kept, [v for _,v in rows]) }")
        return kept

    if n < int(min_points):
        return _fallback(f"no knee (n<{min_points})")
    if not (span > max(1e-12, abs(v0) * float(min_rel_span))):
        return _fallback("no knee (flat)")

    # Normalize to 0..1 then max distance to chord from (0,1)→(n-1,0)
    yn = [(v - vN)/(v0 - vN) if (v0 != vN) else 0.0 for _, v in rows]
    x0, y0 = 0.0, 1.0
    x1, y1 = float(n - 1), 0.0
    dx, dy = (x1 - x0), (y1 - y0)
    denom = float(np.hypot(dx, dy)) or 1.0

    imax, dmax = 0, -1.0
    for i, y in enumerate(yn):
        d = abs(dy*i - dx*y + (x1*y0 - y1*x0)) / denom
        if d > dmax:
            dmax, imax = d, i

    lo = int(np.floor(float(guard_frac) * n))
    hi = n - 1 - lo
    if imax < lo or imax > hi:
        return _fallback("no knee (edge)")

    thr = rows[imax][1]
    k = imax
    while k + 1 < n and rows[k + 1][1] >= thr:
        k += 1
    kept = rows[:k + 1]

    # clamps
    if min_keep_at_least and len(kept) < int(min_keep_at_least):
        kept = rows[:int(min_keep_at_least)]
    if max_keep_at_most and len(kept) > int(max_keep_at_most):
        kept = kept[:int(max_keep_at_most)]

    kept_ids = [pid for pid, _ in kept]
    kept_vals = [v for _, v in kept]
    print(f"[KNEE] {label}: n={n}, span≈{span:.4g}, knee@idx={imax} (rank={imax+1}, cut≈{thr:.4g}) → keep {len(kept_ids)}/{n} {_fmt_list(kept_ids, kept_vals)}")
    return kept_ids

def _apply_knee_union_for_ml(items: List[CandidatePhase], stage_label: str, kcfg: Dict[str, Any]) -> List[str]:
    if not items:
        return []
    def _attr(obj, name, default=np.nan):
        try:
            v = getattr(obj, name, default)
            return float(v) if v is not None else float(default)
        except Exception:
            return float(default)

    # four knees
    ids_score = _knee_keep_ids(
        items, id_fn=lambda c: c.phase_id, val_fn=lambda c: _attr(c, "ml_score"),
        label=f"{stage_label}/score",
        min_points=int(kcfg.get("min_points_hist", 5)),
        min_rel_span=float(kcfg.get("min_rel_span", 0.03)),
        guard_frac=float(kcfg.get("guard_frac", 0.05)),
        max_keep_if_no_knee=int(kcfg.get("max_keep_if_no_knee", 0)),
        min_keep_at_least=int(kcfg.get("min_keep_at_least", 0)),
        max_keep_at_most=int(kcfg.get("max_keep_at_most", 0)),
    )
    ids_cos = _knee_keep_ids(
        items, id_fn=lambda c: c.phase_id, val_fn=lambda c: _attr(c, "ml_cosine"),
        label=f"{stage_label}/cosine",
        min_points=int(kcfg.get("min_points_hist", 5)),
        min_rel_span=float(kcfg.get("min_rel_span", 0.03)),
        guard_frac=float(kcfg.get("guard_frac", 0.05)),
        max_keep_if_no_knee=int(kcfg.get("max_keep_if_no_knee", 0)),
        min_keep_at_least=int(kcfg.get("min_keep_at_least", 0)),
        max_keep_at_most=int(kcfg.get("max_keep_at_most", 0)),
    )
    ids_expl = _knee_keep_ids(
        items, id_fn=lambda c: c.phase_id, val_fn=lambda c: _attr(c, "ml_explained"),
        label=f"{stage_label}/explained",
        min_points=int(kcfg.get("min_points_hist", 5)),
        min_rel_span=float(kcfg.get("min_rel_span", 0.03)),
        guard_frac=float(kcfg.get("guard_frac", 0.05)),
        max_keep_if_no_knee=int(kcfg.get("max_keep_if_no_knee", 0)),
        min_keep_at_least=int(kcfg.get("min_keep_at_least", 0)),
        max_keep_at_most=int(kcfg.get("max_keep_at_most", 0)),
    )
    # NOTE: use min_points_pearson for prob
    ids_prob = _knee_keep_ids(
        items, id_fn=lambda c: c.phase_id, val_fn=lambda c: _attr(c, "ml_present_prob"),
        label=f"{stage_label}/prob",
        min_points=int(kcfg.get("min_points_pearson", kcfg.get("min_points_hist", 5))),
        min_rel_span=float(kcfg.get("min_rel_span", 0.03)),
        guard_frac=float(kcfg.get("guard_frac", 0.05)),
        max_keep_if_no_knee=int(kcfg.get("max_keep_if_no_knee", 0)),
        min_keep_at_least=int(kcfg.get("min_keep_at_least", 0)),
        max_keep_at_most=int(kcfg.get("max_keep_at_most", 0)),
    )

    union_ids = list(dict.fromkeys([*ids_score, *ids_cos, *ids_expl, *ids_prob]))
    if not union_ids:
        return []

    # ---------- RRF ordering (prob included only if selective) ----------
    def _ranks(seq): return {pid: i+1 for i, pid in enumerate(seq)}
    rank_maps = {
        "score": _ranks(ids_score),
        "cos":   _ranks(ids_cos),
        "expl":  _ranks(ids_expl),
    }
    include_prob = ids_prob and len(ids_prob) < max(2, int(0.8*len(union_ids)))
    if include_prob:
        rank_maps["prob"] = _ranks(ids_prob)

    k = float(kcfg.get("rrf_k", 60.0))

    def rrf(pid: str) -> float:
        s = 0.0
        for rm in rank_maps.values():
            r = rm.get(pid)
            if r is not None:
                s += 1.0 / (k + r)
        return s

    score_map = {c.phase_id: float(getattr(c, "ml_score", np.nan)) for c in items}
    hist_map  = {c.phase_id: float(getattr(c, "histogram_score", np.nan)) for c in items}

    def nz(v, default=-1e300):
        try:
            v = float(v)
            if np.isfinite(v): return v
        except Exception:
            pass
        return default

    ordered = sorted(set(union_ids),
                     key=lambda pid: (rrf(pid), nz(score_map.get(pid)), nz(hist_map.get(pid))),
                     reverse=True)
    print(f"[KNEE] {stage_label}/UNION (RRF): selected {len(ordered)} → {_fmt_list(ordered)}")
    return ordered


# --------------------------------------------------------------------------------------
# Bridge
# --------------------------------------------------------------------------------------

class LegacyPipelineBridge:
    """
    Bridges existing impurity detection pipeline with GSAS-II integration.
    Adapts element masks, SG pruning, ML histogram screening, and dedup.
    """

    def __init__(self, db_loader: DBLoader, profiles_dir: str):
        if db_loader is None:
            raise ValueError("db_loader is required")
        if not profiles_dir:
            raise ValueError("profiles_dir is required for histogram screening")

        self.db_loader = db_loader
        self.profiles_dir = profiles_dir
        self.allowed_elements: Optional[List[str]] = None
        self.element_mask: Optional[Tuple[np.uint64, np.uint64]] = None
        self.stable_ids: Optional[set[str]] = None

    # ------------------ utilities ------------------

    def _display_name_sg(self, pid: str) -> Tuple[str, Any]:
        """Resolve (pretty_name, SG) using DBLoader (checks stable catalog first)."""
        return self.db_loader.get_display_name_and_sg(pid)

    # ------------------ configuration ------------------

    def set_element_constraints(self, allowed_elements: List[str], stable_ids: Optional[set] = None):
        """Build and store the element mask; print mask halves."""
        if not allowed_elements:
            raise ValueError("allowed_elements must be a non-empty list")
        self.allowed_elements = list(allowed_elements)
        self.stable_ids = stable_ids
        self.element_mask = build_mask(allowed_elements)
        M_hi, M_lo = self.element_mask
        print(f"Element constraints set: {allowed_elements}")
        print(f"Element mask built successfully: Hi={int(M_hi)}, Lo={int(M_lo)}")

    # ------------------ stages ------------------

    def filter_candidates_by_elements(self, candidate_ids: List[str]) -> List[str]:
        """
        Stage: element mask filter.
        Uses fast mask path by default; switches to advanced aware path if knobs set.
        """
        if self.element_mask is None and not self.allowed_elements:
            raise RuntimeError("Element mask not configured; call set_element_constraints() first")

        efd = getattr(self.db_loader, "element_filter_defaults", {}) or {}

        advanced_active = (
            int(efd.get("max_offlist_elements", 0)) > 0
            or str(efd.get("wildcard_relation", "any")).lower() != "any"
            or bool(efd.get("ignore_elements"))
            or bool(efd.get("disallow_offlist"))
            or (efd.get("require_base", True) is False)
            or bool(efd.get("sample_env"))
            or bool(efd.get("disallow_pure"))
        )

        if advanced_active:
            kept_ids = self.db_loader.filter_by_element_mask(
                self.allowed_elements,
                candidate_ids=candidate_ids,
                ignore_elements=efd.get("ignore_elements"),
                require_base=efd.get("require_base", True),
                max_offlist_elements=int(efd.get("max_offlist_elements", 0)),
                disallow_offlist=efd.get("disallow_offlist"),
                wildcard_relation=str(efd.get("wildcard_relation", "any")),
                sample_env=efd.get("sample_env"),
                disallow_pure=efd.get("disallow_pure"),
            )
        else:
            M_hi, M_lo = self.element_mask
            kept_ids = self.db_loader.filter_by_mask_pair_fast(M_hi, M_lo, candidate_ids=candidate_ids)

        total, kept = len(candidate_ids), len(kept_ids)
        print(f"Element filtering: {total} → {kept} candidates (rejected {total - kept})")

        preview = kept_ids[:5]
        if preview:
            print("Element filter: examples (id, name, SG):")
            for pid in preview:
                name, sg = self._display_name_sg(pid)
                print(f"  - {pid}: {name}, SG={sg}")

        return kept_ids

    def filter_candidates_by_space_group(self, candidate_ids: List[str],
                                         exclude_sg: Tuple[int, ...] = (1, 2)) -> List[str]:
        """
        Stage: prune low-symmetry space groups AFTER elemental filtering.
        Default excludes SG=1 and SG=2.
        """
        kept_ids = self.db_loader.drop_low_symmetry_sg(candidate_ids, exclude_sg=exclude_sg)
        print(f"Space-group pruning (exclude {list(exclude_sg)}): {len(candidate_ids)} → {len(kept_ids)} candidates")

        preview = kept_ids[:5]
        if preview:
            print("Space-group stage: examples (id, name, SG):")
            for pid in preview:
                name, sg = self._display_name_sg(pid)
                print(f"  - {pid}: {name}, SG={sg}")

        return kept_ids
    def filter_candidates_by_stability(self, candidate_ids: List[str]) -> List[str]:
        """
        If a stable catalog is attached, prefer candidates that are in the stable set.
        - If no stable IDs are available, returns the input unchanged.
        - If none of the candidates are stable, logs and returns the input unchanged
          (non-fatal; we don't want to eliminate everything at this stage).
        """
        try:
            # 1) gather stable ids from any available source
            stable_ids: set[str] = set()

            # explicitly provided via set_element_constraints(...)
            if self.stable_ids:
                stable_ids |= {str(s) for s in self.stable_ids}

            # DBLoader helpers/fields (support several possible implementations)
            if hasattr(self.db_loader, "get_stable_ids"):
                try:
                    stable_ids |= {str(s) for s in (self.db_loader.get_stable_ids() or [])}
                except Exception:
                    pass

            if hasattr(self.db_loader, "stable_catalog"):
                try:
                    import pandas as _pd
                    sc = getattr(self.db_loader, "stable_catalog")
                    if sc is not None:
                        df = _pd.DataFrame(sc)
                        col = (
                            "material_id" if "material_id" in df.columns
                            else ("id" if "id" in df.columns else None)
                        )
                        if col:
                            stable_ids |= set(df[col].astype(str))
                except Exception:
                    pass

            if hasattr(self.db_loader, "has_stable_catalog") and callable(self.db_loader.has_stable_catalog):
                # if DB says "no stable catalog", treat as no-op
                if not self.db_loader.has_stable_catalog():
                    print("[stable] no stable catalog attached → skipping stability filter")
                    return candidate_ids

            if not stable_ids:
                print("[stable] stable id set is empty → skipping stability filter")
                return candidate_ids

            # 2) filter against the stable set
            kept = [pid for pid in candidate_ids if str(pid) in stable_ids]

            if not kept:
                print(f"[stable] none of {len(candidate_ids)} candidates are in stable list → leaving list unchanged")
                return candidate_ids

            print(f"[stable] stability pruning: {len(candidate_ids)} → {len(kept)} candidates")
            # preview a few
            for pid in kept[:5]:
                try:
                    name, sg = self._display_name_sg(pid)
                    print(f"  - {pid}: {name}, SG={sg}  (stable)")
                except Exception:
                    pass
            return kept

        except Exception as e:
            print(f"[stable] stability filter encountered an error → skipping. reason: {e}")
            return candidate_ids

    def dedup_by_hist_and_elements(
        self,
        hist_scored: List[Tuple[str, float]],
        *,
        corr_threshold: float = 0.95,
    ) -> List[Tuple[str, float]]:
        """
        De-duplicate Stage-3 candidates using the consolidated profiles64 pack.

        Rules:
        - Group by (space_group, elements_mask_hi, elements_mask_lo).
        - Within each group, compare FULL 64-bin profiles via Pearson r.
        - If r >= corr_threshold, keep only the highest histogram score in that cluster.
        """
        if not hist_scored:
            return hist_scored

        score_map: dict[str, float] = {str(pid): float(sc) for pid, sc in hist_scored}
        ids: list[str] = [str(pid) for pid, _ in hist_scored]

        cat = getattr(self.db_loader, "catalog", None)
        if cat is None:
            raise RuntimeError("DBLoader.catalog is not available")

        required = ["id", "space_group", "elements_mask_hi", "elements_mask_lo"]
        missing_cols = [c for c in required if c not in cat.columns]
        if missing_cols:
            raise KeyError(f"Catalog missing required column(s): {missing_cols}")

        view = cat[cat["id"].astype(str).isin(ids)].copy()
        if view.shape[0] != len(ids):
            present = set(view["id"].astype(str))
            missing = sorted(set(ids) - present)
            raise KeyError(f"Candidate ids not present in catalog: {missing[:8]}{'...' if len(missing)>8 else ''}")

        # Normalize types
        view["id"] = view["id"].astype(str)
        view["space_group"] = pd.to_numeric(view["space_group"], errors="coerce").astype("Int64")
        view["elements_mask_hi"] = view["elements_mask_hi"].astype(np.uint64)
        view["elements_mask_lo"] = view["elements_mask_lo"].astype(np.uint64)

        # Load profiles64 (dict return)
        meta = _load_profiles64_metadata(self.profiles_dir)
        profiles: np.ndarray = meta["profiles"]         # shape (N, 64)
        pid_to_row: dict[str, int] = meta["pid_to_row"] # id -> row index

        npz_path = os.path.join(self.profiles_dir, "profiles64.npz")
        idx_csv  = os.path.join(self.profiles_dir, "index.csv")

        def _hist64(pid: str) -> np.ndarray:
            key = str(pid)
            if key not in pid_to_row:
                raise FileNotFoundError(
                    f"profiles64 missing id '{key}'. Looked in: {npz_path} with index {idx_csv}"
                )
            row = int(pid_to_row[key])
            h = np.asarray(profiles[row], dtype=np.float64).ravel()
            if h.size != 64:
                raise ValueError(f"profiles64 for '{key}' has size={h.size}, expected 64")
            if not np.isfinite(h).all():
                raise ValueError(f"profiles64 for '{key}' contains non-finite values")
            if float(np.std(h)) == 0.0:
                raise ValueError(f"profiles64 for '{key}' is constant (std==0); Pearson undefined")
            return h

        def _pearson_full(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.corrcoef(a, b)[0, 1])

        groups: dict[tuple[int, int, int], list[str]] = {}
        for _, r in view.iterrows():
            sg = int(r["space_group"]) if pd.notna(r["space_group"]) else -1
            key = (sg, int(r["elements_mask_hi"]), int(r["elements_mask_lo"]))
            groups.setdefault(key, []).append(str(r["id"]))

        kept: list[str] = []
        dropped_log: list[tuple[str, str, float]] = []  # (pid_dropped, pid_kept, r)

        for key, pids in groups.items():
            if len(pids) == 1:
                kept.append(pids[0])
                continue

            p_sorted = sorted(pids, key=lambda p: score_map.get(p, float("-inf")), reverse=True)

            representatives: list[str] = []
            rep_profile: dict[str, np.ndarray] = {}

            for pid in p_sorted:
                h = _hist64(pid)
                duplicate_of: tuple[str, float] | None = None
                for rep in representatives:
                    r = _pearson_full(h, rep_profile[rep])
                    if r >= corr_threshold:
                        duplicate_of = (rep, r)
                        break
                if duplicate_of is None:
                    representatives.append(pid)
                    rep_profile[pid] = h
                else:
                    base, r = duplicate_of
                    dropped_log.append((pid, base, r))

            kept.extend(representatives)

        before, after = len(hist_scored), len(kept)
        print(f"Duplicate pruning (SG+elements, r>={corr_threshold:.2f}): {before} → {after} (dropped {before - after})")
        for pid, base, r in dropped_log[:12]:
            n1, sg1 = self.db_loader.get_display_name_and_sg(pid)
            n2, sg2 = self.db_loader.get_display_name_and_sg(base)
            print(f"  [dup] drop {pid} ({n1}, SG={sg1}) ~ {base} ({n2}, SG={sg2}); r={r:.3f}")
        if len(dropped_log) > 12:
            print(f"  ... and {len(dropped_log)-12} more")

        kept_set = set(kept)
        return [(pid, score_map[pid]) for pid, _ in hist_scored if pid in kept_set]

    def screen_by_histogram(self,
                            Q: np.ndarray,
                            R: np.ndarray,
                            Q_main_peaks: np.ndarray,
                            candidate_ids: List[str],
                            *,
                            hist_plot_cfg: Optional[dict] = None,
                            topN: Optional[int] = None,
                            work_dir: Optional[str] = None) -> Tuple[List[Tuple[str, float]], List[dict], dict]:
        """
        Stage-3: 64-bin ML screening using continuous residual R(Q).
        Returns (scored, details, meta), where scored = [(phase_id, score)].

        hist_plot_cfg keys (optional):
        - topN: int               (selection cap)
        - plot: bool
        - plot_out_path_png: str
        - plot_top_k: int         (plotting cap only)
        - plot_title: str
        - plot_label_fn: callable(pid)->str
        """
        # Back-compat: if caller set self.hist_params earlier, use it as fallback
        hp = (hist_plot_cfg or getattr(self, "hist_params", {}) or {})
        def _lbl(pid: str) -> str:
            name, sg = self._display_name_sg(pid)
            return f"{name}  [SG {sg}]  {pid}"

        # ensure plot dir exists (if path provided)
        plot_path = hp.get("plot_out_path_png")
        if plot_path:
            Path(plot_path).parent.mkdir(parents=True, exist_ok=True)

        # IMPORTANT: selection topN defaults to cfg.topN (50), not plot_top_k
        selection_topN = int(topN if topN is not None else hp.get("topN", 50))
        plot_top_k = int(hp.get("plot_top_k", 24))

        print(f"[hist-ml] using topN(selection)={selection_topN}  plot_top_k={plot_top_k}")

        if work_dir:
             traces_dir = Path(work_dir) / "Diagnostics" / "Screening_Traces"
             # traces_dir.mkdir(parents=True, exist_ok=True)  <-- Removed: Lazy creation inside shortlist_by_hist_ML
             default_plot_path = str(traces_dir / "diag_hist_grid.png")
        else:
             default_plot_path = os.path.join(os.getcwd(), "diag_hist_grid.png")

        kept_ids, details, meta = shortlist_by_hist_ML(
            Q, R, Q_main_peaks, candidate_ids,
            profiles_dir=self.profiles_dir,
            topN=selection_topN,
            plot=bool(hp.get("plot", False)),
            plot_out_path_png=plot_path or default_plot_path,
            plot_top_k=plot_top_k,
            plot_label_fn=hp.get("plot_label_fn", _lbl),
            plot_title=str(hp.get("plot_title", "Stage-3 Histogram (ML)")),
        )

        # Filter passers (in ML path, all returned 'details' are passers)
        passed = [d for d in details if d.get("ok") and d.get("pass")]
        print(f"[hist-ml] passed={len(passed)} / details={len(details)}; plot_top_k={plot_top_k}")

        def _metric(d: dict, key: str, default: float = 0.0) -> float:
            v = d.get(key, None)
            return float(v) if v is not None else float(default)

        passed.sort(key=lambda d: (_metric(d, "score", 0.0)), reverse=True)
        scored = [(str(d["phase_id"]), _metric(d, "score", _metric(d, "explained_fraction"))) for d in passed]

        print(f"Histogram screening: {len(candidate_ids)} → {len(scored)} candidates")
        ar = meta.get("active_range")
        print(f"Histogram meta: active_bins={meta.get('active_bins')}, active_range={ar}, sum_residual={meta.get('sum_residual'):.3g}")

        if scored:
            hdr = "Top candidates by ML-hist (id, name, SG, score; cos, α, p, explained):"
            print(hdr)
            for pid, sc in scored:
                name, sg = self._display_name_sg(pid)
                d = next((x for x in passed if str(x["phase_id"]) == pid), {})
                cos   = _metric(d, "cosine")
                alpha = _metric(d, "alpha")
                R_cov = _metric(d, "explained_fraction")
                p     = _metric(d, "present_prob", float("nan"))
                p_str = f", p={p:.3f}" if np.isfinite(p) else ""
                print(f"  - {pid}: {name}, SG={sg}, score={sc:.3f} (cos={cos:.3f}, α={alpha:.3f}{p_str}, explained={R_cov:.3f})")
            import sys
            sys.stdout.flush()

        return scored, details, meta



# --------------------------------------------------------------------------------------
# Screener orchestration
# --------------------------------------------------------------------------------------

class IntegratedCandidateScreener:
    """
    Orchestrates the stages to produce a ranked list of candidate phases.
    Flow (clean): element filter → SG prune → (optional) stability filter → ML histogram → dedup.
    """

    def __init__(self, project_manager, db_loader: DBLoader, profiles_dir: str):
        self.project_manager = project_manager
        self.bridge = LegacyPipelineBridge(db_loader, profiles_dir)

    def screen_candidates_comprehensive(self,
                                        residual_Q: np.ndarray, Q: np.ndarray,
                                        residual_native: np.ndarray, x_native: np.ndarray,
                                        Q_main_peaks: np.ndarray,
                                        allowed_elements: List[str],
                                        all_candidate_ids: List[str],
                                        stable_ids: Optional[set] = None,
                                        *,
                                        hist_plot_cfg: Optional[dict] = None,
                                        work_dir: Optional[str] = None) -> List[CandidatePhase]:
        """
        Comprehensive pipeline (clean):
          1) element filter
          2) space-group prune
          3) (optional) stability filter
          4) histogram screen (ML)
          5) deduplicate by SG+elements with 64-bin Pearson
          6) build CandidatePhase objects with ML diagnostics
        """
        print("\n=== Comprehensive Candidate Screening (ML-only) ===")

        # 1) element mask
        print(f"Starting with {len(all_candidate_ids)} total candidates")
        self.bridge.set_element_constraints(allowed_elements, stable_ids)
        candidates = self.bridge.filter_candidates_by_elements(all_candidate_ids)
        if not candidates:
            raise RuntimeError("No candidates pass element filtering")
        print(f"After element filtering: {len(candidates)} candidates")

        # 2) prune low-symmetry SGs
        candidates = self.bridge.filter_candidates_by_space_group(candidates, exclude_sg=(1, 2))
        if not candidates:
            raise RuntimeError("No candidates remain after space-group pruning")

        # 3) keep only stable ids if available
        if hasattr(self.bridge.db_loader, "has_stable_catalog") and self.bridge.db_loader.has_stable_catalog():
            candidates = self.bridge.filter_candidates_by_stability(candidates)
            if not candidates:
                raise RuntimeError("No candidates remain after stability pruning")

        # 4) histogram (ML)
        print("Running histogram screening (ML)...")
        hist_topN = int((hist_plot_cfg or {}).get("topN", 50))
        hist_scored, hist_details, hist_meta = self.bridge.screen_by_histogram(  Q, residual_Q, Q_main_peaks, candidates,
            hist_plot_cfg=hist_plot_cfg,   topN=hist_topN, work_dir=work_dir   )
                # details map (pid -> dict) for convenience
        if isinstance(hist_details, list):
            hist_details_map = {str(d.get("phase_id")): d for d in hist_details if d and d.get("phase_id") is not None}
        else:
            hist_details_map = {str(k): v for k, v in (hist_details or {}).items()}

        # 5) drop near-duplicates (SG+elements, r>=0.95 default)
        hist_scored = self.bridge.dedup_by_hist_and_elements(hist_scored, corr_threshold=0.95)

        if not hist_scored:
            raise RuntimeError("No candidates pass histogram screening")

        # 6) build CandidatePhase objects
        final_candidates: List[CandidatePhase] = []
        for pid, hist_sc in sorted(hist_scored, key=lambda x: x[1], reverse=True):
            det = hist_details_map.get(str(pid), {})
            c = CandidatePhase(
                phase_id=str(pid),
                cif_file=None,         # defer CIF resolution to Stage-4
                cif_text=None,
                element_mask_score=1.0,
                overlap_score=int(det.get("n_matches", 0)),
                histogram_score=float(hist_sc),
                initial_scale=0.05,
            )
            # ML diagnostics
            try:
                c.ml_score        = float(det.get("score", hist_sc))
                c.ml_alpha        = float(det.get("alpha", np.nan))
                c.ml_beta         = float(det.get("beta", np.nan))
                c.ml_explained    = float(det.get("explained_fraction", np.nan))
                c.ml_cosine       = float(det.get("cosine", np.nan))
                c.ml_present_prob = float(det.get("present_prob", np.nan))
                c.ml_n_matches    = int(det.get("n_matches", 0))
                c.ml_overshoot    = float(det.get("overshoot_fraction_cand", np.nan))
            except Exception:
                pass
            final_candidates.append(c)

        print(f"Final screening: {len(all_candidate_ids)} → {len(final_candidates)} candidates")
        print("Top candidates selected:")
        for i, c in enumerate(final_candidates[:5], 1):
            print(f"  {i}. {c.phase_id}: score={c.histogram_score:.3f}")
        return final_candidates


# ================================
# Stage-0 bootstrap and summaries (clean)
# ================================

def write_dummy_nacl_cif(out_dir: str) -> str:
    """Create a tiny valid NaCl CIF on disk for Stage-0 bootstrapping."""
    cif_content = """data_NaCl
_audit_creation_method              'program-generated example'
_chemical_name_common               'sodium chloride'
_chemical_formula_sum               'Na Cl'

_cell_length_a                      5.6402
_cell_length_b                      5.6402
_cell_length_c                      5.6402
_cell_angle_alpha                   90
_cell_angle_beta                    90
_cell_angle_gamma                   90

_space_group_name_H-M_alt           'F m -3 m'
_space_group_IT_number              225

loop_
_symmetry_equiv_pos_as_xyz
  'x, y, z'

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_adp_type
_atom_site_U_iso_or_equiv
Na1 Na 0.00000 0.00000 0.00000 1.0 Uiso 0.005
Cl1 Cl 0.50000 0.50000 0.50000 1.0 Uiso 0.005
"""
    outp = Path(out_dir) / "Technical" / "bootstrap" / "NaCl_dummy.cif"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(cif_content)
    print(f"[stage0] wrote dummy CIF → {outp}")
    return str(outp)

def lattice_tuple_to_str(lat: Tuple[float, float, float, float, float, float]) -> str:
    a,b,c,al,be,ga = lat
    return f"a={a:.4f}, b={b:.4f}, c={c:.4f}, α={al:.2f}, β={be:.2f}, γ={ga:.2f}"

def get_lattice_from_structure_or_cif(db_loader: DBLoader, pid: str, cif_path: Optional[str]):
    """
    Returns (orig_lattice, nudged_lattice) as tuples; if cif_path isn't nudged, nudged==orig.
    """
    orig_lat = nudged_lat = None
    try:
        st = db_loader.load_structure(pid)
        lat = st.lattice
        orig_lat = (lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma)
    except Exception:
        pass
    try:
        if cif_path and Path(cif_path).exists():
            from pymatgen.core import Structure
            stn = Structure.from_file(cif_path)
            lat = stn.lattice
            nudged_lat = (lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma)
    except Exception:
        pass
    if nudged_lat is None:
        nudged_lat = orig_lat
    return orig_lat, nudged_lat

def stage0_bootstrap_no_cif(
    pm,
    work_dir: str,
    allowed_elements: List[str],
    top_candidates: int,
    s4_cfg: Dict[str, Any],
    profiles_dir: Optional[str],
    db_loader: DBLoader,
    stable_ids: Optional[set] = None,
    *,
    hist_plot_cfg: Optional[dict] = None,
    knee_cfg: Optional[dict] = None,
) -> Tuple[str, str, Optional[List[Any]]]:
    """
    Stage-0: make residual≈Yobs (Ycalc ~ 0), then ML screening + (optional) knee + lattice nudge to pick main CIF.
    Returns (main_cif_path, main_phase_id, stage4_results_list_or_None).
    """
    # 1) Add dummy NaCl
    dummy_cif = write_dummy_nacl_cif(work_dir)
    ok = pm.add_phase_from_cif(dummy_cif, "DUMMY_NaCl")
    if not ok:
        raise RuntimeError("[stage0] failed to add dummy NaCl phase")

    # 2) Force Ycalc ~ 0 (scale=0, hold background, no sample scale refine)
    try:
        hist = pm.main_histogram
        phase = pm.main_phase
        phase.set_HAP_refinements({'Use': True, 'Scale': False}, histograms=[hist])
        phase.HAPvalue('Scale', 0.0, targethistlist=[hist])
    except Exception:
        pass
    try:
        hist.set_refinements({'Background': {'type': 'chebyschev-1', 'refine': False, 'no. coeffs': 12, 'coeffs': [0.0]*12}})
    except Exception:
        pass
    try:
        hist.clear_refinements({'Sample Parameters': ['Scale']})
    except Exception:
        pass

    # 0-cycle compute
    pm.project.data['Controls']['data']['max cyc'] = 0
    try:
        pm.project.do_refinements([{'set': {}}])
        print("[stage0] 0-cycle compute complete: residual ≈ Yobs")
    except Exception as e:
        print(f"[stage0] 0-cycle compute skipped (non-fatal): {e}")

    # 3) Arrays via GSASDataExtractor
    from gsas_main_phase_refiner import GSASDataExtractor
    data = GSASDataExtractor.get_all_arrays(hist)
    Q         = data.get('Q', np.array([]))
    yobs      = data.get('yobs', np.array([]))
    x_native  = data.get('x_native', np.array([]))
    if Q.size == 0 or yobs.size == 0:
        raise RuntimeError("[stage0] empty data arrays; check instrument/data import")

    residual_Q       = yobs.copy()
    residual_native  = yobs.copy()
    # Re-derive paths for Stage-0
    diagnostics_dir = (s4_cfg or {}).get("diagnostics_path") or str(Path(work_dir) / "Diagnostics")
    diag_hist_dir = str(Path(diagnostics_dir) / "Screening_Histograms")
    models_dir = str(Path(work_dir) / "Models")
    models_ref_dir = str(Path(models_dir) / "Reference_CIFs")
    models_refined_dir = str(Path(models_dir) / "Refined_CIFs")

    q_main = np.array([])  # no main reflections at Stage-0

    # 4) Screen (ML) with unified plotting
    screener = IntegratedCandidateScreener(pm, db_loader, profiles_dir)
    all_ids = list(db_loader.catalog['id'].astype(str))
    final_candidates: List[CandidatePhase] = screener.screen_candidates_comprehensive(
        residual_Q=residual_Q,
        Q=Q,
        Q_main_peaks=q_main,
        residual_native=residual_native,
        x_native=x_native,
        allowed_elements=allowed_elements,
        all_candidate_ids=all_ids,
        stable_ids=stable_ids,
        hist_plot_cfg=hist_plot_cfg,
        work_dir=work_dir,
    )
    if not final_candidates:
        raise RuntimeError("[stage0] no candidates found from screening")

    # 5) Optional knee on histogram metrics (stage-0 specific)
    if knee_cfg:
        phase_ids = _apply_knee_union_for_ml(final_candidates, "stage0/hist", knee_cfg)[:int(top_candidates)]
        keep_set = set(phase_ids)
        final_candidates = [c for c in final_candidates if c.phase_id in keep_set]
    else:
        phase_ids = [c.phase_id for c in final_candidates[:int(top_candidates)]]



    # 6) Save a Stage-0 histogram summary CSV (top 20)
    try:
        out_dir = Path(diag_hist_dir) / "stage0"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "stage0_histogram_top.csv"
        rows = []
        for c in final_candidates[:20]:
            name, sg = db_loader.get_display_name_and_sg(c.phase_id)
            rows.append({
                "pid": c.phase_id,
                "name": name,
                "sg": sg,
                "ml_score": float(getattr(c, "ml_score", np.nan)),
                "alpha": float(getattr(c, "ml_alpha", np.nan)),
                "cosine": float(getattr(c, "ml_cosine", np.nan)),
                "present_prob": float(getattr(c, "ml_present_prob", np.nan)),
                "explained_fraction": float(getattr(c, "ml_explained", np.nan)),
                "n_matches": int(getattr(c, "ml_n_matches", 0)),
            })
        if rows:
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"[stage0] histogram summary saved: {csv_path}")
    except Exception as e:
        print(f"[stage0] failed to save histogram summary: {e}")

    print(f"[stage0] sending top-{len(phase_ids)} to nudger: {phase_ids}")

    # 7) Nudger
    from lattice_nudger import LatticeNudger
    nudger = LatticeNudger(
        db_loader,
        wavelength_ang=s4_cfg["wavelength"],
        two_theta_range=s4_cfg["two_theta_range"],
    )
    s4_res = nudger.optimize_many(
        phase_ids, Q, residual_Q,
        reps=s4_cfg.get("reps", 50),
        samples=s4_cfg.get("samples", 5000),
        frac_window=s4_cfg.get("len_tol_pct", s4_cfg.get("frac_window", 1.0)),
        angle_window_deg=s4_cfg.get("ang_tol_deg", s4_cfg.get("angle_window_deg", 3.0)),
        out_cif_dir=models_refined_dir,
    )

    # choose best (by best_score); fall back gracefully
    main_pid, main_cif = None, None
    if s4_res:
        best = max(s4_res, key=lambda r: getattr(r, "best_score", 0.0))
        main_pid = str(best.phase_id)
        main_cif = getattr(best, "nudged_cif_path", None)

    if not main_pid:
        main_pid = str(phase_ids[0])
    if not main_cif:
        main_cif = db_loader.ensure_cif_on_disk(main_pid, out_dir=models_ref_dir)

    print(f"[stage0] selected main phase → id={main_pid}, cif={main_cif}")
    return str(main_cif), main_pid, s4_res

def test_legacy_bridge():
    print("GSAS-II Legacy Bridge (clean, ML-only) ready.")
    print("Key stages:")
    print("- Element mask integration (bit-mask, vectorized)")
    print("- Space-group pruning (drop SG 1/2 by default)")
    print("- 64-bin histogram screening (ML)")
    print("- Dedup by SG+elements with full 64-bin Pearson")

if __name__ == "__main__":
    test_legacy_bridge()
