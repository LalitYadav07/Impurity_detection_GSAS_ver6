#!/usr/bin/env python3
from __future__ import annotations
"""
Unified GSAS-II Impurity Detection Pipeline — Sequential Passes (clean)

Changes vs. previous:
- ML histogram only (no underfill path)
- No alignment or residual peak stages
- No BoxCap structures
- No plot_main_refinement_figure (uses plot_gpx_fit_with_ticks only)
- Unified histogram plotting for all stages including Stage-0
- Pretty-name printing in histogram summaries; β removed
- Overshoot not used for ranking
- Proper, reusable knee filter utility; used for Stage-k and Stage-0

Usage examples:
  python gsas_complete_pipeline_nomain.py --config pipeline_config.yaml --dataset cw_tbssl
  python gsas_complete_pipeline_nomain.py --config pipeline_config.yaml --dry-run
"""

# ---------------------------
# Standard library imports
# ---------------------------
# ---------------------------

import argparse
import json
import os
import sys
import traceback
import math
import re
import io
import datetime
from pathlib import Path

# Force UTF-8 for stdout/stderr to avoid 'charmap' errors on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from typing import Any, Dict, Optional, Tuple, List, Iterable, Set
from time import perf_counter
from contextlib import contextmanager
from collections import defaultdict

# ---------------------------
# Optional third-party imports
# ---------------------------
try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False

# ---------------------------
# Headless / No-GUI Patches
# ---------------------------
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend

try:
    import wx
    # Create a dummy app if needed, or just mask it
    import gsas_legacy_bridge
    import gsas_main_phase_refiner
    import lattice_nudger
    if not wx.GetApp():
        app = wx.App(False)
except ImportError:
    pass

try:
    import GSASII.GSASIIctrlGUI as G2gui
    G2gui.haveGUI = False  # Force GSAS-II into headless mode
except ImportError:
    pass


# ---------------------------
# Local path setup
# ---------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------
# GSAS-II availability check
# ---------------------------
try:
    import GSASII.GSASIIscriptable as G2sc  # noqa: F401
    GSAS_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] GSAS-II not available: {e}")
    GSAS_AVAILABLE = False

# ---------------------------
# Project components
# ---------------------------
try:
    from gsas_core_infrastructure import GSASProjectManager
    from gsas_main_phase_refiner import (
        GSASMainPhaseRefiner,
        GSASPatternAnalyzer,
        read_abs_limits_or_bounds,
        set_limits,
        set_excluded,
        compute_gsas_ycalc_pearson,
        compute_gsas_pearson_for_cif,
        joint_refine_one_cycle,
        extract_residual_from_gpx,
        joint_refine_add_phases,
        joint_refine_polish,
        plot_gpx_fit_with_ticks,
    )
    from gsas_legacy_bridge import (
        IntegratedCandidateScreener,
        stage0_bootstrap_no_cif,

    )
    COMPONENTS_OK = True
except ImportError as e:
    print(f"[ERROR] Failed to import integration components: {e}")
    COMPONENTS_OK = False

# ---------------------------
# Database loader
# ---------------------------
try:
    from aniso_db_loader import DBLoader, CatalogPaths
    LEGACY_DB_AVAILABLE = True
except ImportError as e:
    print(f"[ERROR] aniso_db_loader not available: {e}")
    LEGACY_DB_AVAILABLE = False

# ---------------------------
# Early ML path loading
# ---------------------------
try:
    config_file = next((arg for arg in sys.argv if arg.endswith((".yaml", ".yml"))), None)
    if config_file and os.path.isfile(config_file):
        with open(config_file, encoding='utf-8') as f:
            cfg = yaml.safe_load(f)  # type: ignore[name-defined]
        ml_path = os.path.expandvars(os.path.expanduser(cfg.get("ml_components_dir", "")))
        if os.path.isdir(ml_path) and ml_path not in sys.path:
            sys.path.insert(0, ml_path)
            print(f"[INFO] ML model path added: {ml_path}")
except Exception as e:
    print(f"[WARN] Could not preload ML path from YAML: {e}")

# ---- lightweight timing utility ----
class BenchTimer:
    """
    Simple, nestable wall-clock timer with per-block prints and a final summary.
    Repeated labels are accumulated in the summary.
    """
    def __init__(self, run_name: str = ""):
        self.run_name = run_name
        self._t0 = perf_counter()
        self._totals = defaultdict(float)

    @contextmanager
    def block(self, label: str):
        _start = perf_counter()
        try:
            yield
        finally:
            _dt = perf_counter() - _start
            self._totals[label] += _dt
            elapsed = perf_counter() - self._t0
            print(f"[TIME] {label}: {_dt:.3f}s (elapsed so far: {elapsed:.3f}s)")

    def summary(self):
        total = perf_counter() - self._t0
        if total <= 0:
            total = 1e-9
        print("\n" + "═" * 80)
        print(f"TIMING SUMMARY{(' — ' + self.run_name) if self.run_name else ''}")
        print("═" * 80)
        width = max([len(k) for k in self._totals.keys()] + [22])
        for name, secs in sorted(self._totals.items(), key=lambda kv: kv[1], reverse=True):
            print(f"{name:<{width}}  {secs:9.3f}s  {(secs/total):6.1%}")
        print("-" * 80)
        print(f"{'TOTAL':<{width}}  {total:9.3f}s  100.0%")
        print("═" * 80 + "\n")

# ---- Instrumentation for UI progress tracking ----
class EventEmitter:
    def __init__(self, event_file: Optional[str]):
        self.event_file = event_file
        if self.event_file:
            Path(self.event_file).parent.mkdir(parents=True, exist_ok=True)

    def emit(self, stage: str, message: str, percent: float, level: str = "INFO", artifacts: List[str] = None, metrics: Dict[str, Any] = None):
        event = {
            "time": datetime.datetime.now().isoformat(),
            "level": level,
            "stage": stage,
            "message": message,
            "percent": percent,
            "artifacts": artifacts or [],
            "metrics": metrics or {}
        }
        # We still print to stdout so it appears in logs
        # Removed the manual print(f"[{stage}] ...") here to avoid double prints if calling code already prints.
        if self.event_file:
            try:
                with open(self.event_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event) + "\n")
            except Exception:
                pass

class ManifestManager:
    def __init__(self, manifest_file: Optional[str]):
        self.manifest_file = manifest_file
        self.data = {
            "status": "starting",
            "stages": {},
            "artifacts": [],
            "metrics": {},
            "start_time": datetime.datetime.now().isoformat()
        }

    def update_stage(self, stage: str, status: str, result: Any = None):
        self.data["stages"][stage] = {
            "status": status,
            "updated": datetime.datetime.now().isoformat(),
            "result": result
        }
        self.save()

    def add_artifact(self, path: str):
        if path and path not in self.data["artifacts"]:
            self.data["artifacts"].append(str(path))
            self.save()

    def update_metrics(self, metrics: Dict[str, Any]):
        self.data["metrics"].update(metrics)
        self.save()

    def set_status(self, status: str):
        self.data["status"] = status
        self.save()

    def save(self):
        if self.manifest_file:
            try:
                with open(self.manifest_file, "w", encoding="utf-8") as f:
                    json.dump(self.data, f, indent=2)
            except Exception:
                pass


    def save_json(self, path: str):
        with open(path, "w", encoding='utf-8') as f:
            json.dump(self._totals, f, indent=2)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _expand(p: Optional[str]) -> Optional[str]:
    if not p:
        return p
    s = os.path.expandvars(os.path.expanduser(str(p)))
    q = Path(s)
    if not q.is_absolute():
        base = os.environ.get("CONFIG_DIR") or os.getcwd()
        q = Path(base) / q
    return str(q.resolve())

def _guess_mode_and_tag(data_path: str) -> Tuple[Optional[str], Optional[str]]:
    name = Path(data_path).name.lower()
    if "hb2a" in name:
        return "cw", "hb2a"
    if "pg3" in name:
        return "tof", "pg3"
    return None, None

def _default_fmthint(mode: Optional[str]) -> Optional[str]:
    return "xye" if (mode and mode.lower() == "cw") else None

def _write_xye_from_arrays(out_path: str, x, y, sigma=None, shift_positive: bool = True) -> str:
    import numpy as _np
    x = _np.asarray(x, float).ravel()
    y = _np.asarray(y, float).ravel()
    n = int(min(x.size, y.size))
    if n == 0:
        raise ValueError("Cannot write XYE file: empty x/y arrays")
    yw = y[:n].copy()
    if shift_positive:
        m = _np.nanmin(yw)
        if _np.isfinite(m) and m < 0.0:
            yw = yw - m + 1.0
    if sigma is None:
        sigma = _np.ones(n, float)
    else:
        sigma = _np.asarray(sigma, float).ravel()[:n]
        if sigma.size < n:
            sigma = _np.pad(sigma, (0, n - sigma.size), mode='edge')
    with open(out_path, "w", encoding='utf-8') as f:
        for i in range(n):
            f.write(f"{x[i]:.6f} {yw[i]:.6f} {sigma[i]:.6f}\n")
    print(f"[INFO] Wrote residual XYE file: {out_path} ({n} points)")
    return out_path

# ---------------------------
# CIF metadata parsing helpers (unchanged)
# ---------------------------

_CIF_QUOTE_RE = re.compile(r"^[\s\t]*['\"]?(.*?)['\"]?[\s\t]*$")

def _strip_cif_value(v: str) -> str:
    if v is None:
        return ""
    v = v.strip()
    m = _CIF_QUOTE_RE.match(v)
    return m.group(1) if m else v

def _parse_cif_metadata(cif_path: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not cif_path or not Path(cif_path).exists():
        return None, None
    name: Optional[str] = None
    sg_sym: Optional[str] = None
    sg_num: Optional[int] = None
    data_label: Optional[str] = None
    try:
        with open(cif_path, "r", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                l_lower = line.lower()
                
                # Data label
                if l_lower.startswith("data_") and not data_label:
                    data_label = line[5:].strip() or None
                
                # Name tags
                name_tags = [
                    "_chemical_name_common", "_chemical_name_mineral", 
                    "_chemical_name_systematic", "_chemical_formula_sum",
                    "_pd_phase_id"
                ]
                for tag in name_tags:
                    if line.startswith(tag):
                        parts = line.split(None, 1)
                        if len(parts) > 1:
                            val = _strip_cif_value(parts[1])
                            if val:
                                # Filter unhelpful placeholder names
                                v_clean = val.strip().strip("'").strip('"')
                                if v_clean.lower() not in ("#(c)", "vesta_phase_1", "global", "unknown", "none", "") and len(v_clean) > 1:
                                    name = val
                        break
                
                # SG Symbol tags
                sym_tags = ["_space_group_name_H-M_alt", "_symmetry_space_group_name_H-M"]
                for tag in sym_tags:
                    if line.startswith(tag):
                        parts = line.split(None, 1)
                        if len(parts) > 1:
                            val = _strip_cif_value(parts[1])
                            if val: sg_sym = val
                        break

                # SG Number tags
                num_tags = ["_symmetry_Int_Tables_number", "_space_group_IT_number"]
                for tag in num_tags:
                    if line.startswith(tag):
                        parts = line.split(None, 1)
                        if len(parts) > 1:
                            val = _strip_cif_value(parts[1])
                            try:
                                if val: sg_num = int(val)
                            except: pass
                        break

        # Filter unhelpful data label
        if data_label:
            dl_clean = data_label.strip().strip("'").strip('"')
            if dl_clean.lower() in ("vesta_phase_1", "global", "unknown", "none", ""):
                data_label = None

        if not name:
            name = data_label
            
        sg_final = None
        if sg_sym and sg_num:
            sg_final = f"{sg_sym} ({sg_num})"
        elif sg_sym:
            sg_final = str(sg_sym)
        elif sg_num:
            sg_final = str(sg_num)

    except Exception:
        pass
    return (name if name else None), (sg_final if sg_final else None)

# ====== KNEE HELPERS (shared in this file) ======
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
            if math.isfinite(v):
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

    yn = [(v - vN)/(v0 - vN) if (v0 != vN) else 0.0 for _, v in rows]
    x0, y0 = 0.0, 1.0
    x1, y1 = float(n - 1), 0.0
    dx, dy = (x1 - x0), (y1 - y0)
    denom = math.hypot(dx, dy) or 1.0

    imax, dmax = 0, -1.0
    for i, y in enumerate(yn):
        d = abs(dy*i - dx*y + (x1*y0 - y1*x0)) / denom
        if d > dmax:
            dmax, imax = d, i

    lo = int(math.floor(float(guard_frac) * n))
    hi = n - 1 - lo
    if imax < lo or imax > hi:
        return _fallback("no knee (edge)")

    thr = rows[imax][1]
    k = imax
    while k + 1 < n and rows[k + 1][1] >= thr:
        k += 1
    kept = rows[:k + 1]

    if min_keep_at_least and len(kept) < int(min_keep_at_least):
        kept = rows[:int(min_keep_at_least)]
    if max_keep_at_most and len(kept) > int(max_keep_at_most):
        kept = kept[:int(max_keep_at_most)]

    kept_ids = [pid for pid, _ in kept]
    kept_vals = [v for _, v in kept]
    print(f"[KNEE] {label}: n={n}, span≈{span:.4g}, knee@idx={imax} (rank={imax+1}, cut≈{thr:.4g}) → keep {len(kept_ids)}/{n} {_fmt_list(kept_ids, kept_vals)}")
    return kept_ids

# ============================================================================
# MAIN PIPELINE CLASS (SEQUENTIAL VERSION)
# ============================================================================

class UnifiedPipeline:
    """
    Orchestrates the GSAS-II sequential impurity detection workflow.
    """

    def __init__(self, top_cfg: Dict[str, Any]):
        self.top_cfg = top_cfg or {}
        self.db_loader: Any = None
        self.stable_ids: Optional[set] = None
        self.emitter: Optional[EventEmitter] = None
        self.manifest: Optional[ManifestManager] = None

    # ---------------------------
    # DB initialization (unchanged)
    # ---------------------------
    def initialize_database(self, db_cfg: Dict[str, Any]) -> bool:
        if not LEGACY_DB_AVAILABLE:
            print("[ERROR] DBLoader not available.")
            return False

        try:
            cat_csv = _expand(db_cfg.get("catalog_csv"))
            orig_json = _expand(db_cfg.get("original_json"))
            cif_map = _expand(db_cfg.get("cif_map_json"))

            if not cat_csv or not Path(cat_csv).exists():
                raise FileNotFoundError(f"Catalog CSV not found: {cat_csv}")

            self.db_loader = DBLoader(CatalogPaths(
                catalog_csv=cat_csv,
                cif_map_json=cif_map,
                original_json=orig_json
            ))
            print(f"[INFO] Database initialized: {len(self.db_loader.catalog)} entries")

            stable_csv = _expand(db_cfg.get("stable_csv"))
            if stable_csv and Path(stable_csv).exists():
                self.db_loader.attach_stable_catalog(stable_csv)
                print(f"[INFO] Attached stable catalog from: {stable_csv}")

                try:
                    import pandas as pd
                    df = pd.read_csv(stable_csv)
                    id_col = (
                        "material_id" if "material_id" in df.columns
                        else ("id" if "id" in df.columns else None)
                    )
                    self.stable_ids = set(df[id_col].astype(str)) if id_col else None
                except Exception as e:
                    print(f"[WARN] Could not precompute stable_ids: {e}")
            else:
                print("[INFO] No stable catalog configured")

            return True

        except Exception as e:
            print(f"[ERROR] Database initialization failed: {e}")
            traceback.print_exc()
            self.db_loader = None
            self.stable_ids = None
            return False

    # ---------------------------
    # Small DB helpers (unchanged)
    # ---------------------------
    def _catalog_ids(self) -> set:
        if not self.db_loader:
            return set()
        try:
            return set(self.db_loader.catalog['id'].astype(str))
        except Exception:
            return set()

    def _safe_db_display_and_sg(self, pid: str) -> Tuple[str, str]:
        try:
            if not self.db_loader:
                return pid, "—"
            name_disp, sg = self.db_loader.get_display_name_and_sg(pid)
            return (name_disp if name_disp else pid), (str(sg) if sg else "—")
        except Exception:
            return pid, "—"

    def _main_phase_display_and_sg(self, main_phase_name: str, main_cif: Optional[str]) -> Tuple[str, str]:
        # Priority 1: Parse from CIF if available
        name_from_cif, sg_from_cif = _parse_cif_metadata(main_cif)
        
        name_final = name_from_cif
        sg_final = sg_from_cif

        # Priority 2: Use database if CIF metadata is missing or partial
        if (not name_final or not sg_final) and self.db_loader:
            if main_phase_name in self._catalog_ids():
                try:
                    n2, sg2 = self.db_loader.get_display_name_and_sg(main_phase_name)
                    if not name_final:
                        name_final = n2
                    if not sg_final or "(" not in sg_final: # If CIF provided symbol but not number, prefer DB's combined format
                        sg_final = sg2
                except Exception:
                    pass

        # Priority 3: Final fallbacks
        if not name_final:
            name_final = main_phase_name
        if not sg_final:
            sg_final = "—"
            
        return str(name_final), str(sg_final)

    # ---------------------------
    # Helpers for sequential passes
    # ---------------------------
    @staticmethod
    def _filter_ids(all_ids: Iterable[str], exclude: Set[str]) -> List[str]:
        return [pid for pid in all_ids if pid not in exclude]

    @staticmethod
    def _choose_top_new_by_wf(fractions: Dict[str, Dict[str, float]], candidates: List[str]) -> Optional[str]:
        best_pid, best_wf = None, -1.0
        for pid in candidates:
            wf = float(fractions.get(pid, {}).get('weight_fraction_pct', 0.0))
            if wf > best_wf:
                best_wf, best_pid = wf, pid
        return best_pid

    def _make_hist_plot_cfg(self, stage_tag: str, work_dir: str, ds_cfg: Optional[Dict[str, Any]] = None) -> dict:
        """Unified plot+selection config for histogram screening & plotting.

        Precedence:
        selection cap (topN):
            dataset.hist_plot.topN >
            dataset.hist_filter.topN >
            top_cfg.hist_plot.topN >
            top_cfg.hist_filter.topN >
            50

        plotting cap (plot_top_k):
            dataset.hist_plot.top_k / plot_top_k >
            dataset.hist_filter.plot_top_k >
            top_cfg.hist_plot.top_k / plot_top_k >
            top_cfg.hist_filter.plot_top_k >
            24
        """
        hp_plot_g  = self.top_cfg.get("hist_plot", {}) or {}
        hp_plot_ds = (ds_cfg or {}).get("hist_plot", {}) or {}
        hp_filt_g  = self.top_cfg.get("hist_filter", {}) or {}
        hp_filt_ds = (ds_cfg or {}).get("hist_filter", {}) or {}

        enable = bool(hp_plot_ds.get("enable", hp_plot_g.get("enable", True)))
        plot_top_k = int(hp_plot_ds.get("top_k", hp_plot_ds.get("plot_top_k", hp_filt_ds.get("plot_top_k", hp_plot_g.get("top_k", hp_plot_g.get("plot_top_k", hp_filt_g.get("plot_top_k", 24)))))))
        selection_topN = int(hp_plot_ds.get("topN", hp_filt_ds.get("topN", hp_plot_g.get("topN", hp_filt_g.get("topN", 50)))))
        min_active_bins = int(hp_plot_ds.get("min_active_bins", hp_filt_ds.get("min_active_bins", hp_plot_g.get("min_active_bins", hp_filt_g.get("min_active_bins", 2)))))
        min_sum_residual = float(hp_plot_ds.get("min_sum_residual", hp_filt_ds.get("min_sum_residual", hp_plot_g.get("min_sum_residual", hp_filt_g.get("min_sum_residual", 0.0)))))

        if "Diagnostics" in work_dir:
             out_png = str(Path(work_dir) / stage_tag / "hist_grid.png")
        else:
             out_png = str(Path(work_dir) / "Diagnostics" / "Screening_Histograms" / stage_tag / "hist_grid.png")
        
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)

        return {
            "plot": enable, "plot_top_k": plot_top_k, "plot_out_path_png": out_png,
            "plot_title": f"Histogram (ML) — {stage_tag}", "topN": selection_topN,
            "min_active_bins": min_active_bins, "min_sum_residual": min_sum_residual,
        }



    # Core: screen + nudge + Pearson, using provided residual arrays
    def _screen_and_rank_candidates(
        self,
        name: str,
        pass_ix: int,
        pm_for_tools: GSASProjectManager,
        Q, residual_Q, x_native, residual_native,
        allowed_elements: List[str],
        profiles_dir: Optional[str],
        instprm_path: str,
        data_path: str,
        fmthint: Optional[str],
        limits: Optional[Tuple[float, float]],
        exclude_regions: List[Tuple[float, float]],
        work_dir: str,
        top_candidates: int,
        exclude_ids: Set[str],
        joint_top_k: int,
        s4_cfg: Dict[str, Any],
        ds_cfg: Dict[str, Any],
    ) -> Tuple[List[Any], Dict[str, str], Dict[str, float], Dict[str, Any]]:

        # Re-derive paths for internal use (or accept from ds_cfg)
        diagnostics_dir = ds_cfg.get("diagnostics_path") or str(Path(work_dir) / "Diagnostics")
        diag_resid_dir = ds_cfg.get("diag_resid_path") or str(Path(diagnostics_dir) / "Residual_Scanning")
        models_dir = ds_cfg.get("models_path") or str(Path(work_dir) / "Models")
        models_ref_dir = ds_cfg.get("models_ref_path") or str(Path(models_dir) / "Reference_CIFs")
        models_refined_dir = ds_cfg.get("models_refined_path") or str(Path(models_dir) / "Refined_CIFs")
        
        Path(diag_resid_dir).mkdir(parents=True, exist_ok=True)
        Path(models_ref_dir).mkdir(parents=True, exist_ok=True)
        Path(models_refined_dir).mkdir(parents=True, exist_ok=True)
        """
        Returns:
          final_candidates, pid_to_cif, pearson_best_by_pid, result_by_pid
        """
        # Knee filter config (across stages)
        kcfg = (self.top_cfg.get("knee_filter") or {})
        
        # Derive progress context
        seq_max_passes = int(self.top_cfg.get("max_passes", 3))
        progress_base = 40 + ((pass_ix - 1) / seq_max_passes) * 50
        progress_step = 50 / seq_max_passes

        # Stage-3: candidate screening
        analyzer = GSASPatternAnalyzer(pm_for_tools.main_histogram, pm_for_tools.main_phase)
        q_main_raw = analyzer.get_reflection_positions_q()
        if Q.size:
            qmin, qmax = float(Q.min()), float(Q.max())
            q_main = q_main_raw[(q_main_raw >= qmin) & (q_main_raw <= qmax)]
        else:
            q_main = q_main_raw

        screener = IntegratedCandidateScreener(pm_for_tools, self.db_loader, profiles_dir)  # type: ignore[arg-type]

        hist_plot_cfg = self._make_hist_plot_cfg(f"pass{pass_ix}", work_dir, ds_cfg)


        # Filter candidate pool to exclude already-accepted PIDs and main phase
        all_ids = list(self.db_loader.catalog['id'].astype(str))  # type: ignore[union-attr]
        all_ids = self._filter_ids(all_ids, exclude_ids)

        if self.emitter:
            self.emitter.emit(f"Pass {pass_ix}", "ML Screening", progress_base + 0.1 * progress_step, metrics={"pass": pass_ix, "event": "screening_start"})
        
        final_candidates = screener.screen_candidates_comprehensive(
            residual_Q=residual_Q,
            Q=Q,
            residual_native=residual_native,
            x_native=x_native,
            Q_main_peaks=q_main,
            allowed_elements=allowed_elements,
            all_candidate_ids=all_ids,
            stable_ids=self.stable_ids,
            hist_plot_cfg=hist_plot_cfg,
            work_dir=work_dir,
        )
        print(f"[RESULT] [pass {pass_ix}] ML screening complete: found {len(final_candidates)} phases")
        
        if self.emitter:
             self.emitter.emit(f"Pass {pass_ix}", f"Screened {len(final_candidates)} candidates", progress_base + 0.15 * progress_step, metrics={"pass": pass_ix})

        # ----- KNEE: Histogram (union over ml_score, ml_cosine, ml_explained, ml_present_prob) -----
        if kcfg.get("enable_hist", False) and final_candidates:
            def _attr(c, *names):
                for nm in names:
                    v = getattr(c, nm, None)
                    if v is None:
                        continue
                    try:
                        fv = float(v)
                        if math.isfinite(fv):
                            return fv
                    except Exception:
                        pass
                return float("nan")

            ids_score = _knee_keep_ids(
                final_candidates,
                id_fn=lambda c: c.phase_id,
                val_fn=lambda c: _attr(c, "ml_score","histogram_score"),
                label=f"hist/score (pass {pass_ix})",
                min_points=int(kcfg.get("min_points_hist", 5)),
                min_rel_span=float(kcfg.get("min_rel_span", 0.03)),
                guard_frac=float(kcfg.get("guard_frac", 0.05)),
                max_keep_if_no_knee=int(kcfg.get("max_keep_if_no_knee", 0)),
                min_keep_at_least=int(kcfg.get("min_keep_at_least", 0)),
                max_keep_at_most=int(kcfg.get("max_keep_at_most", 0)),
            )
            ids_cos = _knee_keep_ids(
                final_candidates,
                id_fn=lambda c: c.phase_id,
                val_fn=lambda c: _attr(c, "ml_cosine"),
                label=f"hist/cos (pass {pass_ix})",
                min_points=int(kcfg.get("min_points_hist", 5)),
                min_rel_span=float(kcfg.get("min_rel_span", 0.03)),
                guard_frac=float(kcfg.get("guard_frac", 0.05)),
                max_keep_if_no_knee=int(kcfg.get("max_keep_if_no_knee", 0)),
                min_keep_at_least=int(kcfg.get("min_keep_at_least", 0)),
                max_keep_at_most=int(kcfg.get("max_keep_at_most", 0)),
            )
            ids_expl = _knee_keep_ids(
                final_candidates,
                id_fn=lambda c: c.phase_id,
                val_fn=lambda c: _attr(c, "ml_explained"),
                label=f"hist/explained (pass {pass_ix})",
                min_points=int(kcfg.get("min_points_hist", 5)),
                min_rel_span=float(kcfg.get("min_rel_span", 0.03)),
                guard_frac=float(kcfg.get("guard_frac", 0.05)),
                max_keep_if_no_knee=int(kcfg.get("max_keep_if_no_knee", 0)),
                min_keep_at_least=int(kcfg.get("min_keep_at_least", 0)),
                max_keep_at_most=int(kcfg.get("max_keep_at_most", 0)),
            )
            ids_prob = _knee_keep_ids(
                final_candidates,
                id_fn=lambda c: c.phase_id,
                val_fn=lambda c: _attr(c, "ml_present_prob"),
                label=f"hist/prob (pass {pass_ix})",
                min_points=int(kcfg.get("min_points_pearson", kcfg.get("min_points_hist", 5))),
                min_rel_span=float(kcfg.get("min_rel_span", 0.03)),
                guard_frac=float(kcfg.get("guard_frac", 0.05)),
                max_keep_if_no_knee=int(kcfg.get("max_keep_if_no_knee", 0)),
                min_keep_at_least=int(kcfg.get("min_keep_at_least", 0)),
                max_keep_at_most=int(kcfg.get("max_keep_at_most", 0)),
            )

            union_ids = list(dict.fromkeys([*ids_score, *ids_cos, *ids_expl, *ids_prob]))
            if union_ids:
                def _ranks(seq): return {pid: i+1 for i, pid in enumerate(seq)}
                rank_maps = {"score": _ranks(ids_score), "cos": _ranks(ids_cos), "expl": _ranks(ids_expl)}
                include_prob = ids_prob and len(ids_prob) < max(2, int(0.8*len(union_ids)))
                if include_prob:
                    rank_maps["prob"] = _ranks(ids_prob)
                krrf = float(kcfg.get("rrf_k", 60.0))
                def rrf(pid):
                    s = 0.0
                    for rm in rank_maps.values():
                        r = rm.get(pid)
                        if r is not None:
                            s += 1.0 / (krrf + r)
                    return s
                score_map = {c.phase_id: _attr(c, "ml_score", "histogram_score") for c in final_candidates}
                hist_map  = {c.phase_id: _attr(c, "histogram_score") for c in final_candidates}

                def nz(v, default=-1e300):
                    try:
                        if math.isfinite(float(v)): return float(v)
                    except Exception:
                        pass
                    return default
                phase_ids = sorted(set(union_ids),
                                key=lambda pid: (rrf(pid), nz(score_map.get(pid)), nz(hist_map.get(pid))),
                                reverse=True)
                if isinstance(top_candidates, int) and top_candidates > 0:
                    phase_ids = phase_ids[:int(top_candidates)]
                print(f"[KNEE] hist/UNION (pass {pass_ix}, RRF): selected {len(phase_ids)} → {_fmt_list(phase_ids)}")
            else:
                phase_ids = [str(c.phase_id) for c in final_candidates[:int(top_candidates)]]

        else:
            phase_ids = [str(c.phase_id) for c in final_candidates[:int(top_candidates)]]

        if not final_candidates:
            return [], {}, {}, {}

        if not phase_ids:
            print(f"[INFO] [pass {pass_ix}] No candidates selected for Stage-4 after knee.")
            return final_candidates, {}, {}, {}

        # Stage-4: lattice nudging & scoring
        from lattice_nudger import LatticeNudger
        topN = len(phase_ids)
        print(f"[INFO] [pass {pass_ix}] Processing top {topN} candidates")
        
        if self.emitter:
            self.emitter.emit(f"Pass {pass_ix}", f"Lattice Nudging for top {topN} candidates", progress_base + 0.2 * progress_step, metrics={"pass": pass_ix, "event": "nudging_start"})

        stage4_results = []
        try:
            nudger = LatticeNudger(
                self.db_loader,  # type: ignore[arg-type]
                wavelength_ang=float(s4_cfg["wavelength"]),
                two_theta_range=tuple(s4_cfg["two_theta_range"]),
            )
            stage4_results = nudger.optimize_many(
                phase_ids, Q, residual_Q,
                reps=int(s4_cfg["reps"]),
                samples=int(s4_cfg["samples"]),
                frac_window=float(s4_cfg["frac_window"]),
                angle_window_deg=float(s4_cfg["angle_window_deg"]),
                out_cif_dir=models_refined_dir,
            ) or []
            print(f"[RESULT] [pass {pass_ix}] Nudger→ {len(stage4_results)} optimized structures")
        except Exception as e:
            print(f"[ERROR] [pass {pass_ix}] Lattice nudging failed: {e}")
            traceback.print_exc()
            stage4_results = []

        # ----- KNEE: Nudge best_score -----
        if (self.top_cfg.get("knee_filter") or {}).get("enable_nudge", False) and stage4_results:
            ids_nudge = _knee_keep_ids(
                stage4_results,
                id_fn=lambda r: getattr(r, "phase_id", ""),
                val_fn=lambda r: float(getattr(r, "best_score", float("nan"))),
                label=f"nudge/score (pass {pass_ix})",
                min_points=int(kcfg.get("min_points_nudge", 4)),
                min_rel_span=float(kcfg.get("min_rel_span", 0.03)),
                guard_frac=float(kcfg.get("guard_frac", 0.05)),
                max_keep_if_no_knee=int(kcfg.get("max_keep_if_no_knee", 0)),
                min_keep_at_least=int(kcfg.get("min_keep_at_least", 0)),
                max_keep_at_most=int(kcfg.get("max_keep_at_most", 0)),
            )
            if ids_nudge:
                before = len(phase_ids)
                kept_set = set(ids_nudge)
                phase_ids = [pid for pid in phase_ids if pid in kept_set]
                print(f"[KNEE] nudge/filter (pass {pass_ix}): {before} → {len(phase_ids)} {_fmt_list(phase_ids)}")
            topN = len(phase_ids)

        # Build CIF map (nudged > candidate-provided > database)
        result_by_pid = {str(getattr(r, "phase_id", "")): r for r in (stage4_results or [])}
        pid_to_cif: Dict[str, str] = {}
        for pid, r in result_by_pid.items():
            nudged_cif = getattr(r, "nudged_cif_path", None)
            if nudged_cif and Path(nudged_cif).exists():
                pid_to_cif[pid] = str(Path(nudged_cif).resolve())

        # Ensure CIFs for exactly the knee-selected set
        cif_cache_dir = models_ref_dir
        Path(cif_cache_dir).mkdir(parents=True, exist_ok=True)

        for pid in phase_ids:
            if pid in pid_to_cif:
                continue
            try:
                resolved = self.db_loader.ensure_cif_on_disk(pid, out_dir=cif_cache_dir)  # type: ignore[union-attr]
                pid_to_cif[pid] = resolved
            except Exception as e:
                print(f"[WARN] [pass {pass_ix}] CIF resolve failed for {pid}: {e}")

        # Pearson vs residual for nudged/original; select best per PID
        resid_dir = Path(diag_resid_dir)
        resid_dir.mkdir(parents=True, exist_ok=True)
        resid_xye = resid_dir / f"{name}_residual_pass{pass_ix}.xye"
        _write_xye_from_arrays(str(resid_xye), x_native, residual_native, shift_positive=True)

        if self.emitter:
             self.emitter.emit(f"Pass {pass_ix}", "Pearson Refinement (Lattice Refinement)", progress_base + 0.3 * progress_step, metrics={"pass": pass_ix, "event": "pearson_start"})

        pearson_best_by_pid: Dict[str, float] = {}
        for pid in phase_ids:
            nudged_cif = result_by_pid.get(pid).nudged_cif_path if pid in result_by_pid else None
            label = "nudged-only"
            if nudged_cif:
                r_n, _, out_n = _compute_pearson_with_refinement(
                    pid, nudged_cif, f"{name}_p{pass_ix}", work_dir, x_native, residual_native, instprm_path
                )
                best_p = r_n
                best_path = out_n or nudged_cif
            else:
                # fallback to original only if we truly have no nudged structure
                try:
                    orig_cif = self.db_loader.ensure_cif_on_disk(pid, out_dir=cif_cache_dir)  # type: ignore[union-attr]
                except Exception:
                    orig_cif = None
                if orig_cif:
                    r_o, _, out_o = _compute_pearson_with_refinement(
                        pid, orig_cif, f"{name}_p{pass_ix}", work_dir, x_native, residual_native, instprm_path
                    )
                    best_p = r_o
                    best_path = out_o or orig_cif
                    label = "orig (fallback)"
                else:
                    best_p = float("-inf")
                    best_path = ""
                    label = "no-cif"

            if best_path and Path(best_path).exists():
                pid_to_cif[pid] = str(Path(best_path).resolve())
            pearson_best_by_pid[pid] = best_p
            print(f"[RESULT] [pass {pass_ix}] {pid}: {label} (r={best_p:.4f})")


        # ----- KNEE: Pearson r over current candidate set -----
        if kcfg.get("enable_pearson", False) and pearson_best_by_pid:
            pearson_items = [{"pid": pid, "r": float(pearson_best_by_pid.get(pid, float("nan")))}
                             for pid in phase_ids]
            ids_peer = _knee_keep_ids(
                pearson_items,
                id_fn=lambda x: x["pid"],
                val_fn=lambda x: x["r"],
                label=f"pearson/r (pass {pass_ix})",
                min_points=int(kcfg.get("min_points_pearson", 3)),
                min_rel_span=float(kcfg.get("min_rel_span", 0.03)),
                guard_frac=float(kcfg.get("guard_frac", 0.05)),
                max_keep_if_no_knee=int(kcfg.get("max_keep_if_no_knee", 0)),
                min_keep_at_least=int(kcfg.get("min_keep_at_least", 0)),
                max_keep_at_most=int(kcfg.get("max_keep_at_most", 0)),
            )
            if ids_peer:
                before_keys = set(pid_to_cif.keys())
                pid_to_cif = {pid: path for pid, path in pid_to_cif.items() if pid in set(ids_peer)}
                dropped = sorted(list(before_keys - set(pid_to_cif.keys())))
                print(f"[KNEE] pearson/filter (pass {pass_ix}): kept {len(pid_to_cif)}/{len(before_keys)}; dropped={_fmt_list(dropped)}")

        # Filter by Pearson or legacy score thresholds
        min_pearson = float(s4_cfg.get("min_pearson", "nan"))
        if not math.isnan(min_pearson):
            before = set(pid_to_cif.keys())
            pid_to_cif = {pid: path for pid, path in pid_to_cif.items()
                          if pearson_best_by_pid.get(pid, float("-inf")) >= min_pearson}
            dropped = list(before - set(pid_to_cif.keys()))
            if dropped:
                print(f"[INFO] [pass {pass_ix}] Dropped {len(dropped)} phases below Pearson {min_pearson}")
        else:
            min_sc = float(s4_cfg.get("min_score", 0.0))
            if stage4_results and min_sc > 0:
                keep_ids = {str(getattr(r, "phase_id", "")) for r in stage4_results if float(getattr(r, "best_score", 0.0)) >= min_sc}
                before = set(pid_to_cif.keys())
                pid_to_cif = {pid: path for pid, path in pid_to_cif.items() if pid in keep_ids}
                dropped = list(before - set(pid_to_cif.keys()))
                if dropped:
                    print(f"[INFO] [pass {pass_ix}] Dropped {len(dropped)} phases below score {min_sc}")

        if not pid_to_cif:
            return final_candidates, {}, pearson_best_by_pid, result_by_pid

        # Enforce joint_top_k by Pearson r
        if isinstance(joint_top_k, int) and joint_top_k > 0 and len(pid_to_cif) > joint_top_k:
            sorted_pids = sorted(pid_to_cif.keys(),
                                 key=lambda pid: pearson_best_by_pid.get(pid, float("-inf")),
                                 reverse=True)[:joint_top_k]
            pid_to_cif = {pid: pid_to_cif[pid] for pid in sorted_pids}
            print(f"[INFO] [pass {pass_ix}] keep top {joint_top_k} by Pearson for compare-run")

        return final_candidates, pid_to_cif, pearson_best_by_pid, result_by_pid

    # ---------------------------
    # Dataset runner (SEQUENTIAL)
    # ---------------------------
    def run_dataset(self, ds: Dict[str, Any]) -> bool:
        name = ds.get("name", "dataset")
        print("\n" + "=" * 80)
        print(f"PROCESSING DATASET (SEQUENTIAL): {name}")
        print("=" * 80)

        # --------------------------------------------------------------------
        # CONFIG RESOLUTION
        # --------------------------------------------------------------------
        data_path = _expand(ds.get("data_path"))
        if not data_path or not Path(data_path).exists():
            raise RuntimeError(f"[{name}] Data file not found: {data_path}")

        mode = ds.get("mode", "auto")
        tag = None
        if str(mode).lower() == "auto":
            mode, tag = _guess_mode_and_tag(data_path)
            if not mode:
                raise RuntimeError(f"[{name}] Could not infer instrument mode. Specify CW or TOF.")
        else:
            mode = str(mode).lower()
            tag = "hb2a" if mode == "cw" else ("pg3" if mode == "tof" else None)
        print(f"[INFO] Instrument mode: {mode.upper()}, tag: {tag}")

        fmthint = ds.get("fmthint", "auto")
        if fmthint == "auto":
            fmthint = _default_fmthint(mode)

        instprm_path = ds.get("instprm_path")
        if instprm_path == "auto" or not instprm_path:
            imap = self.top_cfg.get("instrument_map", {})
            if "instrument_map" in ds and isinstance(ds["instrument_map"], dict):
                imap = ds["instrument_map"]
            guess_key = tag or ("hb2a" if mode == "cw" else "pg3")
            instprm_path = _expand(imap.get(guess_key))
            if not instprm_path or not Path(instprm_path).exists():
                raise RuntimeError(
                    f"[{name}] Could not resolve instrument parameter file. "
                    f"Provide instrument_map.{guess_key} or explicit path."
                )
        else:
            instprm_path = _expand(instprm_path)
            if not instprm_path or not Path(instprm_path).exists():
                raise RuntimeError(f"[{name}] Instrument parameter file not found: {instprm_path}")

        print(f"[INFO] Instrument parameters: {instprm_path}")

        # Main phase CIF and name
        main_cif = _expand(ds.get("main_cif"))
        main_phase_name = ds.get("main_phase_name") or "auto"
        if main_phase_name == "auto":
            main_phase_name = Path(main_cif).stem if main_cif else "Main"

        # Working directories
        work_root_cfg = self.top_cfg.get("work_root") or os.environ.get("WORK_ROOT")
        work_root = _expand(work_root_cfg) if work_root_cfg else None

        ds_work_dir = ds.get("work_dir")
        if ds_work_dir:
            work_dir = _expand(ds_work_dir)
        elif work_root:
            # Avoid redundant nesting if work_root already ends with name
            wr_path = Path(work_root)
            if wr_path.name == name:
                work_dir = str(wr_path)
            else:
                work_dir = str(wr_path / name)
        else:
            work_dir = _expand(self.top_cfg.get("work_dir")) or str(Path.cwd() / name)

        # --------------------------------------------------------------------
        # DIRECTORY SETUP (Deep Reorganization)
        # --------------------------------------------------------------------
        results_dir = str(Path(work_dir) / "Results")
        results_plots_dir = str(Path(results_dir) / "Plots")
        
        models_dir = str(Path(work_dir) / "Models")
        models_ref_dir = str(Path(models_dir) / "Reference_CIFs")
        models_refined_dir = str(Path(models_dir) / "Refined_CIFs")
        
        diagnostics_dir = str(Path(work_dir) / "Diagnostics")
        diag_hist_dir = str(Path(diagnostics_dir) / "Screening_Histograms")
        diag_resid_dir = str(Path(diagnostics_dir) / "Residual_Scanning")
        diag_traces_dir = str(Path(diagnostics_dir) / "Screening_Traces")
        
        technical_dir = str(Path(work_dir) / "Technical")
        tech_projects_dir = str(Path(technical_dir) / "GSAS_Projects")
        tech_logs_dir = str(Path(technical_dir) / "Logs")
        
        # Create all directories
        print(f"[DEBUG] results_dir: {results_dir}")
        print(f"[DEBUG] models_dir: {models_dir}")
        print(f"[DEBUG] diagnostics_dir: {diagnostics_dir}")
        print(f"[DEBUG] technical_dir: {technical_dir}")
        
        for d in (work_dir, results_dir, results_plots_dir, 
                  models_dir, models_ref_dir, models_refined_dir,
                  diagnostics_dir, diag_hist_dir, diag_resid_dir, diag_traces_dir,
                  technical_dir, tech_projects_dir, tech_logs_dir):
            Path(d).mkdir(parents=True, exist_ok=True)
            
        print(f"[INFO] Working directory: {work_dir}")

        # Initialize instrumentation (Technical/Logs)
        self.emitter = EventEmitter(str(Path(tech_logs_dir) / "run_events.jsonl"))
        self.manifest = ManifestManager(str(Path(tech_logs_dir) / "run_manifest.json"))
        self.manifest.set_status("running")
        self.emitter.emit("Bootstrap", "Starting pipeline", 0)

        # Mappings for internal use
        # Map legacy/config keys to new paths
        plots_dir = results_plots_dir
        cifs_nudged_dir = models_refined_dir
        joint_dir = tech_projects_dir # Main project files go here now

        # Pipeline parameters
        allowed_elements = ds.get("allowed_elements", self.top_cfg.get("allowed_elements", []))
        top_candidates = int(ds.get("top_candidates", self.top_cfg.get("top_candidates", 10)))
        min_impurity_percent = float(ds.get("min_impurity_percent", self.top_cfg.get("min_impurity_percent", 0.5)))
        hap_init = float(ds.get("hap_init", self.top_cfg.get("hap_init", 0.05)))
        max_joint_cycles = int(ds.get("max_joint_cycles", self.top_cfg.get("max_joint_cycles", 8)))
        joint_top_k = int(ds.get("joint_top_k", self.top_cfg.get("joint_top_k", 7)))

        # Sequential controls
        seq_max_passes = int(ds.get("max_passes", self.top_cfg.get("max_passes", 3)))
        rwp_improve_eps = float(ds.get("rwp_improve_eps", self.top_cfg.get("rwp_improve_eps", 0.00)))  # optional gate

        # Stage 4 configuration
        s4_cfg = {
            "wavelength": float(ds.get("stage4", {}).get("wavelength", self.top_cfg.get("stage4", {}).get("wavelength", 1.50))),
            "two_theta_range": tuple(ds.get("stage4", {}).get("two_theta_range", self.top_cfg.get("stage4", {}).get("two_theta_range", [5.0, 160.0]))),
            "frac_window": float(ds.get("stage4", {}).get("frac_window", self.top_cfg.get("stage4", {}).get("frac_window", 0.025))),
            "angle_window_deg": float(ds.get("stage4", {}).get("angle_window_deg", self.top_cfg.get("stage4", {}).get("angle_window_deg", 1.5))),
            "samples": int(ds.get("stage4", {}).get("samples", self.top_cfg.get("stage4", {}).get("samples", 500))),
            "reps": int(ds.get("stage4", {}).get("reps", self.top_cfg.get("stage4", {}).get("reps", 20))),
            "min_score": float(ds.get("stage4", {}).get("min_score", self.top_cfg.get("stage4", {}).get("min_score", 0.02))),
            "min_pearson": ds.get("stage4", {}).get("min_pearson", self.top_cfg.get("stage4", {}).get("min_pearson", "nan")),
        }

        # Database configuration
        db_cfg = self.top_cfg.get("db", {}).copy()
        for k in ("catalog_csv", "original_json", "profiles_dir", "stable_csv", "cif_map_json"):
            if k in ds:
                db_cfg[k] = ds[k]

        if not self.db_loader:
            if not self.initialize_database(db_cfg):
                print(f"[ERROR] [{name}] Database initialization failed")
                return False

        # Resolve display names and SG for main phase
        disp_main, sg_main = self._main_phase_display_and_sg(main_phase_name, main_cif)
        sg_main_disp = sg_main if sg_main not in (None, "", "—") else "unknown"

        profiles_dir = _expand(db_cfg.get("profiles_dir"))

        # Element filter configuration (unchanged)
        ef_global = self.top_cfg.get("element_filter", {}) or {}
        ef_ds = ds.get("element_filter", {}) or {}
        ef = {**ef_global, **ef_ds}
        self.db_loader.element_filter_defaults = {  # type: ignore[union-attr]
            "max_offlist_elements": int(ef.get("max_offlist_elements", 0)),
            "require_base": bool(ef.get("require_base", True)),
            "ignore_elements": list(ef.get("ignore_elements", [])),
            "disallow_offlist": list(ef.get("disallow_offlist", [])),
            "wildcard_relation": str(ef.get("wildcard_relation", "any")),
            "sample_env": ef.get("sample_env", {}),
            "disallow_pure": list(ef.get("disallow_pure", [])),
        }
        print(f"[INFO] Element filter: +{ef.get('max_offlist_elements', 0)} wildcards, relation={ef.get('wildcard_relation', 'any')}")

        # Data range limits and exclusions
        limits = ds.get("limits")
        exclude_regions = ds.get("exclude_regions", [])

        # =========================
        bench = BenchTimer(run_name=name)

        try:
            # --------------------------------------------------------------------
            # STAGE 0: BOOTSTRAP (if no main CIF provided)
            # --------------------------------------------------------------------
            if not main_cif:
                with bench.block("Stage 0: Bootstrap"):
                    print("\n" + "=" * 80)
                    print("STAGE 0: BOOTSTRAP (discovery from scratch)")
                    print("=" * 80)
                    self.emitter.emit("Stage 0", "Bootstrap starting", 5, metrics={"pass_type": "bootstrap"})
                    self.manifest.update_stage("Stage 0", "running")

                    _prev_ml_is_stage0 = os.environ.get("ML_IS_STAGE0")
                    os.environ["ML_IS_STAGE0"] = "1"
                    try:
                        with bench.block("S0: Create project & add histogram"):
                            pm0 = GSASProjectManager(tech_projects_dir, f"{name}_stage0")
                            if not pm0.create_project():
                                raise RuntimeError(f"[{name}] Failed to create Stage-0 project")
                            if not pm0.add_histogram(data_path, instprm_path, fmthint=fmthint):
                                raise RuntimeError(f"[{name}] Failed to add histogram")

                            # Apply limits/exclusions
                            hist0 = pm0.main_histogram
                            abs_lo, abs_hi = read_abs_limits_or_bounds(hist0)
                            if limits and len(limits) == 2:
                                req_lo, req_hi = float(limits[0]), float(limits[1])
                                lo, hi = (
                                    max(req_lo, abs_lo),
                                    min(req_hi, abs_hi),
                                ) if (abs_lo is not None and abs_hi is not None) else (req_lo, req_hi)
                                set_limits(hist0, lo, hi)
                                print(f"[INFO] Data limits applied: [{lo:.2f}, {hi:.2f}]")

                            if exclude_regions:
                                set_excluded(hist0, exclude_regions)
                                print(f"[INFO] Excluded regions: {exclude_regions}")

                        with bench.block("S0: ML bootstrap → main phase"):
                            # Stage-0 knee overrides (fall back to global if absent)
                            kcfg0 = self.top_cfg.get("knee_filter_stage0", self.top_cfg.get("knee_filter", {})) or {}

                            # unified histogram plotting config for stage-0
                            hist_plot_cfg0 = self._make_hist_plot_cfg("stage0", work_dir, ds)


                            s4_res: List[Any]
                            main_cif, main_phase_name, s4_res = stage0_bootstrap_no_cif(
                                pm=pm0,
                                work_dir=str(Path(technical_dir) / "bootstrap"), # Keep bootstrap temp in Technical
                                allowed_elements=allowed_elements,
                                top_candidates=top_candidates,
                                s4_cfg=s4_cfg,
                                ds_cfg={"diagnostics_path": diagnostics_dir}, # Pass diagnostics root
                                profiles_dir=profiles_dir,
                                db_loader=self.db_loader,
                                stable_ids=self.stable_ids,
                                hist_plot_cfg=hist_plot_cfg0,
                                knee_cfg=kcfg0,
                            )

                            # --- OPTIONAL: Stage-0 knee on nudger best_score (keep using shared helper) ---
                            kcfg = (self.top_cfg.get("knee_filter") or {})
                            if kcfg.get("enable_nudge", False) and (s4_res or []):
                                ids_s0 = _knee_keep_ids(
                                    s4_res,
                                    id_fn=lambda r: getattr(r, "phase_id", ""),
                                    val_fn=lambda r: float(getattr(r, "best_score", float("nan"))),
                                    label="stage0/nudge",
                                    min_points=int(kcfg.get("min_points_nudge", 4)),
                                    min_rel_span=float(kcfg.get("min_rel_span", 0.03)),
                                    guard_frac=float(kcfg.get("guard_frac", 0.05)),
                                    max_keep_if_no_knee=int(kcfg.get("max_keep_if_no_knee", 0)),
                                    min_keep_at_least=int(kcfg.get("min_keep_at_least", 0)),
                                    max_keep_at_most=int(kcfg.get("max_keep_at_most", 0)),
                                )
                                if ids_s0:
                                    before = len(s4_res)
                                    keep = set(ids_s0)
                                    s4_res = [r for r in s4_res if getattr(r, "phase_id", "") in keep]
                                    print(f"[KNEE] stage0/nudge/filter: {before} → {len(s4_res)} kept")

                        with bench.block("S0: Re-rank by GSAS Pearson"):
                            print("[INFO] Re-ranking Stage-0 candidates by Pearson correlation...")
                            cif_cache_dir = str(Path(technical_dir) / "cifs_cache")
                            Path(cif_cache_dir).mkdir(parents=True, exist_ok=True)

                            pear_tmp_dir = str(Path(technical_dir) / "Pearson_Temp")
                            Path(pear_tmp_dir).mkdir(parents=True, exist_ok=True)

                            def _pearson_raw(cif_path: Optional[str], tag: str) -> float:
                                if not cif_path or not Path(cif_path).exists():
                                    return float("nan")
                                try:
                                    return float(
                                        compute_gsas_pearson_for_cif(
                                            data_path=data_path,
                                            instprm_path=instprm_path,
                                            fmthint=fmthint,
                                            cif_path=cif_path,
                                            work_dir=pear_tmp_dir,
                                            limits=limits,
                                            exclude_regions=exclude_regions,
                                            tmp_tag=tag,
                                        )
                                    )
                                except Exception:
                                    return float("nan")

                            pearson_best_by_pid: Dict[str, float] = {}
                            path_choice_by_pid: Dict[str, Optional[str]] = {}

                            for r in (s4_res or []):
                                pid = str(getattr(r, "phase_id", ""))
                                if not pid:
                                    continue

                                nudged_cif = getattr(r, "nudged_cif_path", None)
                                if nudged_cif:
                                    p = _pearson_raw(nudged_cif, f"{name}_stage0_sel_{pid}_nudged")
                                    pearson_best_by_pid[pid] = p
                                    path_choice_by_pid[pid]  = nudged_cif
                                else:
                                    try:
                                        orig_cif = self.db_loader.ensure_cif_on_disk(pid, out_dir=cif_cache_dir)  # type: ignore[union-attr]
                                    except Exception:
                                        orig_cif = None
                                    p = _pearson_raw(orig_cif, f"{name}_stage0_sel_{pid}_orig") if orig_cif else float("nan")
                                    pearson_best_by_pid[pid] = p
                                    path_choice_by_pid[pid]  = orig_cif


                            kcfg = (self.top_cfg.get("knee_filter") or {})
                            if kcfg.get("enable_pearson", False) and pearson_best_by_pid:
                                items = [{"pid": pid, "r": float(pearson_best_by_pid[pid])} for pid in pearson_best_by_pid]
                                ids_peer = _knee_keep_ids(
                                    items,
                                    id_fn=lambda x: x["pid"],
                                    val_fn=lambda x: x["r"],
                                    label="stage0/pearson",
                                    min_points=int(kcfg.get("min_points_pearson", 3)),
                                    min_rel_span=float(kcfg.get("min_rel_span", 0.03)),
                                    guard_frac=float(kcfg.get("guard_frac", 0.05)),
                                    max_keep_if_no_knee=int(kcfg.get("max_keep_if_no_knee", 0)),
                                    min_keep_at_least=int(kcfg.get("min_keep_at_least", 0)),
                                    max_keep_at_most=int(kcfg.get("max_keep_at_most", 0)),
                                )
                                if ids_peer:
                                    before_map = dict(pearson_best_by_pid)
                                    keep = set(ids_peer)
                                    pearson_best_by_pid = {pid: pearson_best_by_pid[pid] for pid in keep}
                                    path_choice_by_pid  = {pid: path_choice_by_pid[pid]  for pid in keep}
                                    dropped = sorted(list(set(before_map.keys()) - keep))
                                    print(f"[KNEE] stage0/pearson/filter: kept {len(pearson_best_by_pid)}/{len(before_map)}; dropped={_fmt_list(dropped)}")

                            best_pid, best_cif, best_p = None, None, float("nan")
                            if pearson_best_by_pid:
                                best_pid = max(pearson_best_by_pid, key=lambda p: pearson_best_by_pid[p])
                                best_cif = path_choice_by_pid[best_pid]
                                best_p   = pearson_best_by_pid[best_pid]
                                print(f"[INFO] Stage-0 Pearson winner: {best_pid} (r={best_p:.4f})")

                                if best_pid and best_cif and (best_pid != main_phase_name or best_cif != main_cif):
                                    print(f"[INFO] Pearson override: {main_phase_name} → {best_pid} (r={best_p:.4f})")
                                    main_phase_name = best_pid
                                    main_cif = best_cif
                                    # Update metadata after override
                                    disp_main, sg_main = self._main_phase_display_and_sg(main_phase_name, main_cif)
                                    sg_main_disp = sg_main if sg_main not in (None, "", "—") else "unknown"

                        with bench.block("S0: Summary build/print"):
                            try:
                                gsas_r = best_p if not math.isnan(best_p) else compute_gsas_ycalc_pearson(pm0)
                            except Exception:
                                gsas_r = float("nan")

                            
                    finally:
                        if _prev_ml_is_stage0 is None:
                            os.environ.pop("ML_IS_STAGE0", None)
                        else:
                            os.environ["ML_IS_STAGE0"] = _prev_ml_is_stage0
                    
                    self.emitter.emit("Stage 0", "Bootstrap complete", 20, metrics={"main_phase_id": main_phase_id})
                    self.manifest.update_stage("Stage 0", "complete", {"main_cif": main_cif, "main_phase_name": main_phase_name})

            # --------------------------------------------------------------------
            # STAGE 1: MAIN PHASE REFINEMENT (single-phase base)
            # --------------------------------------------------------------------
            with bench.block("Stage 1: Main phase refinement"):
                print(f"\n{'─' * 80}")
                print(f"STAGE 1: MAIN PHASE REFINEMENT")
                print(f"{'─' * 80}")
                print("\n" + "=" * 80)
                print("STAGE 1: MAIN PHASE REFINEMENT")
                print("=" * 80)
                self.emitter.emit("Stage 1", "Main phase refinement starting", 25)
                self.manifest.update_stage("Stage 1", "running")

                pm = GSASProjectManager(tech_projects_dir, f"{name}_project")
                if not pm.create_project():
                    raise RuntimeError(f"[{name}] Failed to create GSAS project")
                if not pm.add_histogram(data_path, instprm_path, fmthint=fmthint):
                    raise RuntimeError(f"[{name}] Failed to add histogram")

                hist = pm.main_histogram
                abs_lo, abs_hi = read_abs_limits_or_bounds(hist)
                if limits and len(limits) == 2:
                    req_lo, req_hi = float(limits[0]), float(limits[1])
                    lo, hi = (
                        max(req_lo, abs_lo),
                        min(req_hi, abs_hi),
                    ) if (abs_lo is not None and abs_hi is not None) else (req_lo, req_hi)
                    if lo >= hi:
                        raise ValueError(f"Invalid limits after clipping to available range [{abs_lo}, {abs_hi}]")
                    set_limits(hist, lo, hi)
                    print(f"[INFO] Data limits: [{lo:.2f}, {hi:.2f}]")
                if exclude_regions:
                    set_excluded(hist, exclude_regions)
                    print(f"[INFO] Excluded regions: {exclude_regions}")

                if not main_cif:
                    raise RuntimeError(f"[{name}] Main CIF is required for Stage-1")
                if not pm.add_phase_from_cif(main_cif, main_phase_name):
                    raise RuntimeError(f"[{name}] Failed to add main phase from CIF")

                try:
                    phase = pm.main_phase
                    phase.set_HAP_refinements({"Use": True, "Scale": False}, histograms=[hist])
                    phase.HAPvalue("Scale", 1.0, targethistlist=[hist])
                except Exception as e:
                    print(f"[WARN] Phase initialization: {e}")

                main_ref = GSASMainPhaseRefiner(pm)
                _bg = ds.get("background", self.top_cfg.get("background", {})) or {}

                with bench.block("S1: staged refinement"):
                    main_results = main_ref.run_staged_refinement(
                        enable_cell=True,
                        bg_type=_bg.get("type"),
                        bg_terms=int(_bg["terms"]) if _bg.get("terms") is not None else None,
                        bg_coeffs=_bg.get("coeffs"),
                    )
                if not main_results.success:
                    raise RuntimeError(f"[{name}] Main-phase refinement failed: {main_results.error_message}")
                print(f"[RESULT] Rwp = {main_results.rwp:.3f}%")

                with bench.block("S1: plot main refinement"):
                    main_plot = str(Path(plots_dir) / "main_phase_fit.png")
                    # Prepare metadata labels for ticks: ID -> "Name — SG"
                    labels = {main_phase_name: f"{main_phase_name} ({disp_main} — {sg_main_disp})"}
                    plot_gpx_fit_with_ticks(pm.project.filename, main_plot, phase_labels=labels)
                    print(f"[INFO] Main phase plot saved: {main_plot}")
                    self.manifest.add_artifact(main_plot)

                self.emitter.emit("Stage 1", "Main phase refinement complete", 40)
                self.manifest.update_stage("Stage 1", "complete", {"rwp": main_results.rwp})

            # --------------------------------------------------------------------
            # INITIAL RESIDUAL (pass 1 seed)
            # --------------------------------------------------------------------
            with bench.block("Stage 2: Residual extraction (seed for pass 1)"):
                if self.emitter:
                    self.emitter.emit("Stage 2", "Residual extraction (seed for pass discovery)", 45)
                self.manifest.update_stage("Stage 2", "running")
                
                Q, residual_Q = main_ref.get_residual_q()
                x_native, residual_native = main_ref.get_residual_native()
                print(f"[INFO] Extracted {len(Q)} residual points (Q-space)")
                print(f"[INFO] Extracted {len(x_native)} residual points (native space)")
                self.manifest.update_stage("Stage 2", "complete")

            # ========================= SEQUENTIAL PASSES =========================
            accepted: List[str] = []   # accepted impurities in order
            kept_gpx = str(Path(tech_projects_dir) / f"{name}_project.gpx")  # single-phase base to start
            kept_rwp = main_results.rwp

            for pass_ix in range(1, int(seq_max_passes) + 1):

                print("\n" + "=" * 80)
                print(f"SEQUENTIAL PASS {pass_ix} — candidate discovery")
                print("=" * 80)
                progress_base = 40 + ((pass_ix - 1) / seq_max_passes) * 50
                progress_step = 50 / seq_max_passes
                
                print("\n" + "═" * 80)
                print(f"SEQUENTIAL PASS {pass_ix} — candidate discovery")
                print("═" * 80)
                self.emitter.emit(f"Pass {pass_ix}", f"Discovery pass {pass_ix} starting", progress_base, metrics={"pass": pass_ix, "event": "pass_start"})

                # For pass > 1, recompute residual from the kept GPX
                if pass_ix > 1:
                    with bench.block(f"Pass {pass_ix}: residual from kept GPX"):
                        x_native, residual_native, Q, residual_Q, kept_rwp, hist_name, _ = extract_residual_from_gpx(kept_gpx)
                        print(f"[INFO] [pass {pass_ix}] Kept GPX: {kept_gpx}, Rwp={kept_rwp:.3f}%")

                exclude_ids = {main_phase_name, *accepted}

                with bench.block(f"Pass {pass_ix}: screen + nudge + pearson"):
                    final_candidates, pid_to_cif, pearson_best_by_pid, result_by_pid = self._screen_and_rank_candidates(
                                name=name,
                                pass_ix=pass_ix,
                                pm_for_tools=pm,
                                Q=Q, residual_Q=residual_Q, x_native=x_native, residual_native=residual_native,
                                allowed_elements=allowed_elements,
                                profiles_dir=profiles_dir,
                                instprm_path=instprm_path,
                                data_path=data_path,
                                fmthint=fmthint,
                                limits=limits,
                                exclude_regions=exclude_regions,
                                work_dir=work_dir,
                                top_candidates=top_candidates,
                                exclude_ids=exclude_ids,
                                joint_top_k=joint_top_k,
                                s4_cfg=s4_cfg,
                                ds_cfg=ds,  # <-- NEW
                            )


                if not pid_to_cif:
                    print(f"[INFO] [pass {pass_ix}] No candidates passed Pearson/score filters. Stopping.")
                    self.manifest.update_stage(f"Pass {pass_ix}", "complete", {"status": "stopped", "reason": "no candidates"})
                    break

                # Compare-run: kept GPX + top-K new candidates to decide which ONE to accept
                with bench.block(f"Pass {pass_ix}: compare-run joint refinement (kept + top-K new)"):
                    if self.emitter:
                         self.emitter.emit(f"Pass {pass_ix}", "Joint refinement (compare-run) started", progress_base + 0.4 * progress_step, metrics={"pass": pass_ix, "event": "joint_compare_start"})
                    
                    pass_dir = Path(joint_dir)
                    cmp_gpx = str(pass_dir / f"seq_pass{pass_ix}_compare.gpx")
                    fractions_cmp = joint_refine_add_phases(
                        base_gpx=kept_gpx,
                        out_gpx=cmp_gpx,
                        main_phase_name=main_phase_name,
                        pid_to_cif_new=pid_to_cif,
                        hap_init=hap_init,
                        max_joint_cycles=max_joint_cycles,
                        preserve_existing_scales=True,
                    )
                    # Keep TRIAL BLEND Rwp (kept + top-K)
                    _, _, _, _, rwp_compare, _, _ = extract_residual_from_gpx(cmp_gpx)
                    self.manifest.add_artifact(cmp_gpx)


                # Choose one new impurity by highest Wt%
                new_candidates = list(pid_to_cif.keys())
                best_new = self._choose_top_new_by_wf(fractions_cmp, new_candidates)
                if best_new is None:
                    print(f"[INFO] [pass {pass_ix}] No usable candidate in compare-run. Stopping.")
                    self.manifest.update_stage(f"Pass {pass_ix}", "complete", {"status": "stopped", "reason": "no usable candidate in compare-run"})
                    break

                wf_best = float(fractions_cmp.get(best_new, {}).get("weight_fraction_pct", 0.0))
                pearson_best = float(pearson_best_by_pid.get(best_new, float("-inf")))
                if wf_best < float(min_impurity_percent):
                    print(f"[INFO] [pass {pass_ix}] Top Wt% {wf_best:.3f} < threshold {min_impurity_percent}. Stopping.")
                    self.manifest.update_stage(f"Pass {pass_ix}", "complete", {"status": "stopped", "reason": "impurity below threshold", "wf_best": wf_best})
                    break
                if not math.isnan(float(s4_cfg.get("min_pearson", "nan"))) and pearson_best < float(s4_cfg["min_pearson"]):
                    print(f"[INFO] [pass {pass_ix}] Pearson {pearson_best:.3f} < min_pearson {s4_cfg['min_pearson']}. Stopping.")
                    self.manifest.update_stage(f"Pass {pass_ix}", "complete", {"status": "stopped", "reason": "pearson below threshold", "pearson_best": pearson_best})
                    break

                # Commit-run: kept = main + accepted + best_new; then POLISH
                with bench.block(f"Pass {pass_ix}: commit-run joint refinement (kept = main + accepted + best_new)"):
                    self.emitter.emit(f"Pass {pass_ix}", "Joint refinement (commit-run) started", progress_base + 0.6 * progress_step, metrics={"pass": pass_ix, "event": "joint_refine_start"})
                    accepted.append(best_new)
                    commit_gpx = str(Path(joint_dir) / f"seq_pass{pass_ix}_kept.gpx")
                    pid_to_cif_commit = {pid: pid_to_cif[pid] if pid in pid_to_cif else
                                         self.db_loader.ensure_cif_on_disk(pid, out_dir=models_ref_dir)  # type: ignore[union-attr]
                                         for pid in accepted}
                    fractions_kept_quick = joint_refine_one_cycle(
                        base_gpx=str(Path(tech_projects_dir) / f"{name}_project.gpx"),
                        out_gpx=commit_gpx,
                        main_phase_name=main_phase_name,
                        pid_to_cif=pid_to_cif_commit,
                        hap_init=hap_init,
                        max_joint_cycles=max_joint_cycles,
                    )
                    # Rwp for quick accept
                    _, _, _, _, rwp_kept, _, _ = extract_residual_from_gpx(commit_gpx)
                    self.manifest.add_artifact(commit_gpx)


                    # POLISH
                    if self.emitter:
                         self.emitter.emit(f"Pass {pass_ix}", "Polishing model", progress_base + 0.8 * progress_step, metrics={"pass": pass_ix, "event": "polish_start"})
                    
                    polished_gpx = str(Path(joint_dir) / f"seq_pass{pass_ix}_kept_polished.gpx")
                    fractions_polished, rwp_polished = joint_refine_polish(
                        base_gpx=commit_gpx,
                        out_gpx=polished_gpx,
                        main_phase_name=main_phase_name,
                        max_polish_cycles=int(ds.get("polish_cycles", self.top_cfg.get("polish_cycles", 10))),
                        refine_cell_for_all=bool(ds.get("polish_refine_cell", self.top_cfg.get("polish_refine_cell", True))),
                        refine_background=bool(ds.get("polish_refine_background", self.top_cfg.get("polish_refine_background", True))),
                    )
                    kept_gpx = polished_gpx
                    kept_rwp_new = rwp_polished
                    fractions_kept = fractions_polished
                    self.manifest.add_artifact(kept_gpx)


                # PLOTS: TRIAL BLEND and POLISHED ACCEPTED MODEL
                with bench.block(f"Pass {pass_ix}: plots"):
                    trial_png = str(Path(plots_dir) / f"seq_pass{pass_ix}_trial_blend.png")
                    final_png = str(Path(plots_dir) / f"seq_pass{pass_ix}_accepted_model.png")
                    
                    # Build labels mapping ID -> "Name — SG" for all active phases
                    active_pids = [main_phase_name] + accepted
                    labels = {}
                    for p in active_pids:
                        if p == main_phase_name:
                            labels[p] = f"{p} ({disp_main} — {sg_main_disp})"
                        else:
                            d, s = self._safe_db_display_and_sg(p)
                            s_disp = s if s not in (None, "", "—") else "unknown"
                            labels[p] = f"{p} ({d} — {s_disp})"
                            
                    plot_gpx_fit_with_ticks(cmp_gpx, trial_png, phase_labels=labels)
                    plot_gpx_fit_with_ticks(kept_gpx, final_png, phase_labels=labels)
                    self.manifest.add_artifact(trial_png)
                    self.manifest.add_artifact(final_png)

                # PASS SUMMARY
                def _disp(pid):
                    if pid == main_phase_name:
                        nm, _sg = self._main_phase_display_and_sg(main_phase_name, main_cif)
                        return nm or pid
                    nm, _sg = self._safe_db_display_and_sg(pid)
                    return nm or pid

                kept_wf_map = {k: float(v.get('weight_fraction_pct', 0.0)) for k, v in (fractions_kept or {}).items()}
                accepted_labels = [f"{pid} ({_disp(pid)}) — {kept_wf_map.get(pid, 0.0):.2f}%"
                                   for pid in [main_phase_name] + accepted]

                print("\n" + "-" * 80)
                print(f"PASS {pass_ix} SUMMARY")
                print("-" * 80)
                baseline_label = "Baseline model Rwp (before pass)" if pass_ix > 1 else "Main-phase only Rwp (start)"
                print(f"{baseline_label}:              {kept_rwp:.3f}%")
                print(f"TRIAL BLEND Rwp (kept + top-K):     {rwp_compare:.3f}%")
                print(f"ACCEPTED MODEL Rwp (quick accept):  {rwp_kept:.3f}%")
                print(f"ACCEPTED MODEL Rwp (polished):      {kept_rwp_new:.3f}%")
                print(f"Accepted this pass: {best_new} ({_disp(best_new)}), Wt% in trial: {wf_best:.3f}%")
                print("Accepted set (ID (Name) — Wt%):")
                for s in accepted_labels:
                    print(f"  - {s}")
                print("-" * 80 + "\n")

                self.emitter.emit(f"Pass {pass_ix}", f"Discovery pass {pass_ix} complete", progress + 50/seq_max_passes, metrics={"pass": pass_ix, "event": "pass_end"})
                self.manifest.update_stage(f"Pass {pass_ix}", "complete", {
                    "rwp_before": kept_rwp,
                    "rwp_trial_blend": rwp_compare,
                    "rwp_accepted_quick": rwp_kept,
                    "rwp_accepted_polished": kept_rwp_new,
                    "accepted_this_pass": best_new,
                    "accepted_phases": accepted_labels,
                })

                # Early-stop decision based on polished result
                delta = kept_rwp - kept_rwp_new   # improvement if positive
                if rwp_improve_eps > 0 and delta < rwp_improve_eps:
                    if delta < 0:
                        print(f"[INFO] Early stop: Rwp worsened by {abs(delta):.3f} (threshold {rwp_improve_eps}); stopping.")
                    else:
                        print(f"[INFO] Early stop: ΔRwp={delta:.3f} < eps {rwp_improve_eps}; stopping.")
                    kept_rwp = kept_rwp_new
                    break
                kept_rwp = kept_rwp_new

            # --------------------------------------------------------------------
            # FINAL SUMMARY & CSV
            # --------------------------------------------------------------------
            with bench.block("Stage 6: Final reporting (print + CSV)"):
                print(f"\n{'═' * 80}")
                print(f"FINAL SUMMARY: SEQUENTIAL PHASES")
                print(f"{'═' * 80}")

                # Final kept GPX is in kept_gpx
                try:
                    last_lst = Path(kept_gpx).with_suffix(".lst")
                    if last_lst.exists():
                        _, _, _, _, rwp_final, _, _ = extract_residual_from_gpx(kept_gpx)
                    else:
                        rwp_final = kept_rwp
                except Exception:
                    rwp_final = kept_rwp

                # Build CSV rows from accepted list
                rows: List[Dict[str, Any]] = []
                parsed = {}
                try:
                    from gsas_main_phase_refiner import parse_gsas_lst
                    _, _, _, _, _, hist_name, _proj = extract_residual_from_gpx(kept_gpx)
                    lst_path = Path(kept_gpx).with_suffix(".lst")
                    if lst_path.exists():
                        parsed = parse_gsas_lst(lst_path, hist_name)
                    
                    self.manifest.update_metrics({"final_rwp": rwp_final, "phases_found": len(accepted)})
                    self.manifest.add_artifact(str(lst_path))
                    self.manifest.add_artifact(kept_gpx)
                except Exception as e:
                    print(f"[WARN] Final result parsing failed: {e}")

                final_phase_ids = [main_phase_name] + accepted
                for pid in final_phase_ids:
                    pdata = parsed.get(pid, {})
                    wf = float(pdata.get('weight_fraction_pct', 0.0))
                    pf = float(pdata.get('phase_fraction_pct', 0.0))
                    is_main = (pid == main_phase_name)
                    if is_main:
                        disp, sg = self._main_phase_display_and_sg(main_phase_name, main_cif)
                    else:
                        disp, sg = self._safe_db_display_and_sg(pid)
                    rows.append({
                        "pid": pid,
                        "display_name": disp,
                        "sg": sg if sg else "—",
                        "wf": wf,
                        "pf": pf,
                        "is_main": is_main,
                    })

                rows_sorted = sorted(rows, key=lambda r: (not r["is_main"], -r["wf"]))  # main first, then by wf
                total_imp = sum(r["wf"] for r in rows_sorted if not r["is_main"] and r["wf"] >= min_impurity_percent)

                hdr = f"Sequential Phase Quantification for {name}"
                cols = f"{'#':>2}  {'Phase ID':<18}  {'Compound Name':<30}  {'SG':>6}  {'Wt%':>7}  {'Ph%':>7}  {'Notes':<20}"
                rule = "─" * len(cols)
                print(f"\n{hdr}")
                print(rule); print(cols); print(rule)
                for i, r in enumerate(rows_sorted, 1):
                    sg_str = r["sg"] if r["sg"] not in (None, "", "unknown") else "—"
                    # Include ID in the name column for clarity, but keep width in check
                    full_name = f"{r['pid']} ({r['display_name']})"
                    name_disp = full_name[:30]
                    note = "MAIN PHASE" if r["is_main"] else ("" if r["wf"] >= min_impurity_percent else f"<{min_impurity_percent}% (trace)")
                    print(f"{i:>2}  {r['pid']:<18}  {name_disp:<30}  {sg_str:>6}  {r['wf']:7.2f}  {r['pf']:7.2f}  {note:<20}")
                print(rule)
                print(f"Final kept GPX: {kept_gpx}  (Rwp={rwp_final:.3f}%)")
                print(f"Total impurity (Wt% ≥ {min_impurity_percent}%): {total_imp:.2f}%")
                print(f"{'═' * 80}\n")

                csv_path = str(Path(results_dir) / "Summary_Fractions.csv")
                try:
                    import pandas as pd
                    pd.DataFrame(rows).to_csv(csv_path, index=False)
                    print(f"[INFO] Final fractions CSV: {csv_path}")
                    self.manifest.add_artifact(csv_path)
                except Exception:
                    pass

                self.emitter.emit("Final", "Pipeline completed successfully", 100)
                self.manifest.set_status("complete")
                with open(csv_path, "w") as f:
                    f.write("phase_id,compound_name,space_group,weight_fraction_pct,phase_fraction_pct,is_main\n")
                    for r in rows_sorted:
                        f.write(f"{r['pid']},{r['display_name']},{r['sg']},{r['wf']:.6f},{r['pf']:.6f},{int(r['is_main'])}\n")
                print(f"[INFO] Phase quantification saved: {csv_path}")

            return True

        finally:
            bench.summary()

# ============================================================================
# Stage-4 Pearson helper
# ============================================================================

def _compute_pearson_with_refinement(
    pid: str,
    cand_cif: str,
    name: str,
    work_dir: str,
    x_native,
    residual_native,
    instprm_path: str
) -> Tuple[float, str, str]:
    import numpy as _np
    try:
        resid_dir = Path(work_dir) / "Diagnostics" / "Residual_Scanning"
        resid_dir.mkdir(parents=True, exist_ok=True)
        resid_xye = resid_dir / f"{name}_residual.xye"

        if not resid_xye.exists():
            _write_xye_from_arrays(str(resid_xye), x_native, residual_native, shift_positive=True)

        lims = (float(_np.nanmin(x_native)), float(_np.nanmax(x_native)))
        is_nudged = str(cand_cif).endswith("_nudged.cif")
        label = "nudged" if is_nudged else "orig"

        r_val = compute_gsas_pearson_for_cif(
            data_path=str(resid_xye),
            instprm_path=instprm_path,
            fmthint="xye",
            cif_path=cand_cif,
            work_dir=str(resid_dir),
            limits=lims,
            exclude_regions=None,
            tmp_tag=f"{name}_s4_{pid}_{label}",
            refine_cycles=2,
            refine_hist_scale=True,
            refine_cell=True,
            out_refined_cif=None,
            source_cif_for_export=cand_cif
        )

        print(f"[INFO] Stage-4 Pearson ({label}): {pid} → r={float(r_val):.4f}")
        return float(r_val), label, cand_cif

    except Exception as e:
        print(f"[ERROR] Pearson computation failed for {pid}: {e}")
        traceback.print_exc()
        return 0.0, "orig", cand_cif

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def load_config_file(path: str) -> Dict[str, Any]:
    text = Path(path).read_text()
    if HAVE_YAML and (path.endswith((".yml", ".yaml")) or ":" in text):
        cfg = yaml.safe_load(text)  # type: ignore[name-defined]
    else:
        cfg = json.loads(text)

    cfg_dir = str(Path(path).resolve().parent)
    os.environ.setdefault("CONFIG_DIR", cfg_dir)

    def set_env_from_cfg(env_key: str, *cfg_keys: str):
        for ck in (env_key, env_key.lower(), *cfg_keys):
            v = cfg.get(ck)
            if isinstance(v, str) and v:
                os.environ[env_key] = os.path.expandvars(os.path.expanduser(v))
                return

    set_env_from_cfg("PROJECT_ROOT")
    set_env_from_cfg("WORK_ROOT")
    set_env_from_cfg("DATA_ROOT")

    if "DATA_ROOT" not in os.environ and "PROJECT_ROOT" in os.environ:
        os.environ["DATA_ROOT"] = str(Path(os.environ["PROJECT_ROOT"]) / "scripts" / "data")

    return cfg

def main() -> bool:
    parser = argparse.ArgumentParser(
        description="GSAS-II Sequential Impurity Detection Pipeline (clean)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all datasets in config
  python gsas_complete_pipeline_nomain.py --config pipeline_config.yaml

  # Run specific dataset
  python gsas_complete_pipeline_nomain.py --config pipeline_config.yaml --dataset cw_tbssl

  # Validate configuration without running
  python gsas_complete_pipeline_nomain.py --config pipeline_config.yaml --dry-run
        """
    )
    parser.add_argument("--config", required=True, help="YAML/JSON configuration file")
    parser.add_argument("--dataset", help="Process only specified dataset (by name)")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration and exit")
    args = parser.parse_args()

    if not GSAS_AVAILABLE or not COMPONENTS_OK:
        print("[ERROR] GSAS-II and all integration components must be available")
        return False

    cfg = load_config_file(args.config)

    ml_path = _expand(cfg.get("ml_components_dir"))
    if ml_path and os.path.isdir(ml_path) and ml_path not in sys.path:
        sys.path.insert(0, ml_path)
        print(f"[INFO] ML components path: {ml_path}")

    datasets = cfg.get("datasets", [])
    if not datasets:
        print("[ERROR] No datasets found in configuration")
        return False

    pipe = UnifiedPipeline(cfg)

    if args.dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN: Configuration Validation")
        print("=" * 80)
        for ds in datasets:
            name = ds.get("name", "<unnamed>")
            dp = ds.get("data_path")
            ic = ds.get("instprm_path")
            mc = ds.get("main_cif")
            print(f"  [{name}]")
            print(f"    Data: {dp}")
            print(f"    Instrument params: {ic}")
            print(f"    Main CIF: {mc or '(auto-detect)'}")
        print("=" * 80)
        print("Configuration valid. Use without --dry-run to execute.")
        return True

    success = True
    if args.dataset:
        ds = next((d for d in datasets if d.get("name") == args.dataset), None)
        if not ds:
            print(f"[ERROR] Dataset '{args.dataset}' not found in configuration")
            return False
        try:
            ok = pipe.run_dataset(ds)
            success &= ok
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
            return False
        except Exception as e:
            print(f"[FATAL] Dataset '{args.dataset}' failed: {e}")
            traceback.print_exc()
            return False
    else:
        for ds in datasets:
            name = ds.get("name", "<unnamed>")
            try:
                ok = pipe.run_dataset(ds)
                success &= ok
            except KeyboardInterrupt:
                print("\n[INFO] Interrupted by user")
                return False
            except Exception as e:
                print(f"[ERROR] Dataset '{name}' failed: {e}")
                traceback.print_exc()
                if pipe.manifest:
                    pipe.manifest.set_status("failed")
                    pipe.emitter.emit("Error", str(e), 100, level="ERROR")
                success = False

    out_dir = _expand(cfg.get("work_dir")) or str(Path(args.config).resolve().parent)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    summary_path = str(Path(out_dir) / "pipeline_summary.json")

    with open(summary_path, "w") as f:
        json.dump({
            "success": success,
            "datasets_processed": [d.get("name") for d in datasets],
            "note": "Sequential pipeline complete. See per-pass artifacts in 'Results' and 'Diagnostics' folders."
        }, f, indent=2)

    print(f"\n[INFO] Pipeline summary written to: {summary_path}")
    print("\n✅ Pipeline completed successfully" if success else "\n❌ Pipeline completed with errors")
    return success

if __name__ == "__main__":
    try:
        ok = main()
    except Exception as e:
        print(f"[FATAL] Unexpected error: {e}")
        traceback.print_exc()
        ok = False
    sys.exit(0 if ok else 1)
