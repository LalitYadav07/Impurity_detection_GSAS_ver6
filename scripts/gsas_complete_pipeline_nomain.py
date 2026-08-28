#!/usr/bin/env python3
from __future__ import annotations
"""
GSAS-II Impurity Detection Pipeline (Unified Driver)

This is the main entry point for the sequential impurity detection pipeline.
It orchestrates the entire process, including:
- Data loading and instrument parameter setup.
- Sequential candidate screening and ranking (ML-driven).
- Lattice nudging and refinement (Stage-4).
- Parallel Pearson correlation analysis.
- Final report generation and artifact management.

Process Flow:
1. Stage-0: Bootstrap and initial screening.
2. Stage-1..N: Sequential discovery of impurity phases.
3. Verification and final output generation.
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
import csv
import shutil
from pathlib import Path

import numpy as np

# Force UTF-8 for stdout/stderr to avoid 'charmap' errors on Windows without
# replacing streams owned by pytest, notebooks, NOVA, or another host.
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError, ValueError):
                pass

from typing import Any, Dict, Optional, Tuple, List, Iterable, Set
from time import perf_counter
from contextlib import contextmanager
from collections import defaultdict
import concurrent.futures

# ---------------------------
# Optional third-party imports
# ---------------------------
try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False

try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore[assignment]

try:
    from xray_doublet import describe_doublet, resolve_xray_doublet_spec
except Exception:
    describe_doublet = None
    resolve_xray_doublet_spec = None

try:
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    HAVE_PYMATGEN_MATCHER = True
except Exception:
    Structure = None  # type: ignore[assignment]
    StructureMatcher = None  # type: ignore[assignment]
    SpacegroupAnalyzer = None  # type: ignore[assignment]
    HAVE_PYMATGEN_MATCHER = False

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
        if os.environ.get("DISPLAY"):
            app = wx.App(False)
        else:
            print("[WARN] DISPLAY not set; skipping wx.App init for headless mode")
except BaseException as e:
    if isinstance(e, KeyboardInterrupt):
        raise
    print(f"[WARN] wx initialization skipped: {e}")

try:
    import GSASII.GSASIIctrlGUI as G2gui
    G2gui.haveGUI = False  # Force GSAS-II into headless mode
except ImportError:
    pass


# ---------------------------
# Local path setup
# ---------------------------
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
GSAS_PATH = Path(os.environ.get("RADAR_PD_GSASII_ROOT") or PROJECT_ROOT / "GSAS-II")

for p in reversed([str(GSAS_PATH), str(ROOT), str(PROJECT_ROOT)]):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

# Centralized logging: configure early for CLI runs
try:
    # When running from the scripts/ folder this will find scripts/logging_config.py
    from logging_config import configure_logging
except Exception:
    try:
        from scripts.logging_config import configure_logging
    except Exception:
        configure_logging = None

if configure_logging:
    try:
        configure_logging()
    except Exception:
        import logging as _logging
        _logging.basicConfig(level=_logging.INFO)

# ---------------------------
# GSAS-II availability check
# ---------------------------
try:
    import GSASII.GSASIIscriptable as G2sc  # noqa: F401
    GSAS_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] GSAS-II not available: {e}")
    GSAS_AVAILABLE = False

# Allow tests or CI to bypass heavy GSAS-II/component checks by setting
# the environment variable `SKIP_COMPONENT_CHECK=1`. This enables running
# lightweight CLI validation (dry-run) in environments without GSAS-II.
try:
    if os.environ.get("SKIP_COMPONENT_CHECK") in ("1", "true", "True"):
        GSAS_AVAILABLE = True
except Exception:
    pass

# ---------------------------
# Project components
# ---------------------------
try:
    from gsas_core_infrastructure import GSASProjectManager
    from gsas_main_phase_refiner import (
        GSASMainPhaseRefiner,
        GSASDataExtractor,
        GSASPatternAnalyzer,
        read_abs_limits_or_bounds,
        normalize_limits,
        normalize_excluded_regions,
        ensure_usable_range,
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
    from ml_ranker_support import discover_ml_ranker_assets, load_first_json_record, write_ranker_status
    from reference_phase_masks import build_reference_phase_exclusions, merge_reference_phase_exclusion_config
    from magnetic_precheck import run_magnetic_precheck
    COMPONENTS_OK = True
except ImportError as e:
    print(f"[ERROR] Failed to import integration components: {e}")
    COMPONENTS_OK = False
else:
    # Also allow bypass of component import for CI/dry-run when requested
    try:
        if os.environ.get("SKIP_COMPONENT_CHECK") in ("1", "true", "True"):
            COMPONENTS_OK = True
    except Exception:
        pass

try:
    from auto_background_points import coerce_auto_background_params, estimate_background
except Exception:
    try:
        from scripts.auto_background_points import coerce_auto_background_params, estimate_background
    except Exception:
        coerce_auto_background_params = None
        estimate_background = None

from main_phase_anchor import (
    assess_main_fit_for_prenudge as shared_assess_main_fit_for_prenudge,
    main_shadow_filter_decision,
    main_phase_guard_cfg as shared_main_phase_guard_cfg,
    main_phase_guard_violation as shared_main_phase_guard_violation,
    main_phase_shadow_cfg as shared_main_phase_shadow_cfg,
    main_prenudge_cfg as shared_main_prenudge_cfg,
    main_shadow_peaks_from_arrays,
    run_main_phase_cleanup_if_enabled,
    should_adopt_prenudged_main as shared_should_adopt_prenudged_main,
)

# ---------------------------
# Database loader
# ---------------------------
try:
    from aniso_db_loader import DBLoader, CatalogPaths, build_mask
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
        self._records: List[Dict[str, Any]] = []

    @contextmanager
    def block(self, label: str):
        _start = perf_counter()
        try:
            yield
        finally:
            _dt = perf_counter() - _start
            self._totals[label] += _dt
            elapsed = perf_counter() - self._t0
            self._records.append({
                "label": label,
                "seconds": float(_dt),
                "elapsed_s": float(elapsed),
            })
            print(f"[TIME] {label}: {_dt:.3f}s (elapsed so far: {elapsed:.3f}s)")

    def snapshot(self) -> Dict[str, Any]:
        total = perf_counter() - self._t0
        if total <= 0:
            total = 1e-9
        return {
            "run_name": self.run_name,
            "total_s": float(total),
            "blocks": [
                {
                    "label": name,
                    "seconds": float(secs),
                    "fraction": float(secs / total),
                }
                for name, secs in sorted(self._totals.items(), key=lambda kv: kv[1], reverse=True)
            ],
            "events": list(self._records),
        }

    def write_report(self, json_path: str, extra: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        payload = self.snapshot()
        if extra:
            payload.update(extra)

        out_json = Path(json_path)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

        out_csv = out_json.with_suffix(".csv")
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["label", "seconds", "fraction"])
            writer.writeheader()
            for row in payload.get("blocks", []):
                writer.writerow({
                    "label": row.get("label"),
                    "seconds": row.get("seconds"),
                    "fraction": row.get("fraction"),
                })
        return str(out_json), str(out_csv)

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

def _crop_native_arrays_by_q(
    x_native,
    residual_native,
    q_values,
    *,
    q_max: Optional[float],
    min_points: int = 25,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Return native x/residual arrays masked to Q <= q_max for screening-only jobs."""
    import numpy as _np

    x_arr = _np.asarray(x_native, float).ravel()
    y_arr = _np.asarray(residual_native, float).ravel()
    q_arr = _np.asarray(q_values, float).ravel()
    n = int(min(x_arr.size, y_arr.size, q_arr.size))
    meta: Dict[str, Any] = {
        "enabled": False,
        "q_max": None if q_max is None else float(q_max),
        "input_points": int(min(x_arr.size, y_arr.size)),
        "output_points": int(min(x_arr.size, y_arr.size)),
        "reason": "",
    }

    if q_max is None or not math.isfinite(float(q_max)) or float(q_max) <= 0:
        meta["reason"] = "disabled"
        return x_native, residual_native, meta
    if n < max(1, int(min_points)):
        meta["reason"] = "too_few_aligned_points"
        return x_native, residual_native, meta

    x_arr = x_arr[:n]
    y_arr = y_arr[:n]
    q_arr = q_arr[:n]
    mask = _np.isfinite(x_arr) & _np.isfinite(y_arr) & _np.isfinite(q_arr) & (q_arr <= float(q_max))
    kept = int(mask.sum())
    if kept < max(1, int(min_points)):
        meta.update({
            "output_points": kept,
            "reason": "crop_too_small",
        })
        return x_native, residual_native, meta

    meta.update({
        "enabled": True,
        "output_points": kept,
        "q_min_kept": float(_np.nanmin(q_arr[mask])),
        "q_max_kept": float(_np.nanmax(q_arr[mask])),
    })
    return x_arr[mask], y_arr[mask], meta


# ---- Instrumentation for UI progress tracking ----
class EventEmitter:
    def __init__(self, event_file: Optional[str]):
        self.event_file = event_file
        if self.event_file:
            Path(self.event_file).parent.mkdir(parents=True, exist_ok=True)

    def emit(self, stage: str, message: str, percent: float, level: str = "INFO", artifacts: Optional[List[str]] = None, metrics: Optional[Dict[str, Any]] = None):
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
            except Exception as e:
                print(f"[WARN] Event log write failed ({self.event_file}): {type(e).__name__}: {e}")

class ManifestManager:
    def __init__(self, manifest_file: Optional[str]):
        self.manifest_file = manifest_file
        # Explicitly type as Dict[str, Any] to avoid strict union inference issues
        self.data: Dict[str, Any] = {
            "status": "starting",
            "stages": {},
            "artifacts": [],
            "metrics": {},
            "start_time": datetime.datetime.now().isoformat()
        }

    def update_stage(self, stage: str, status: str, result: Any = None):
        if "stages" not in self.data:
            self.data["stages"] = {}
        self.data["stages"][stage] = {
            "status": status,
            "updated": datetime.datetime.now().isoformat(),
            "result": result
        }
        self.save()

    def add_artifact(self, path: str):
        if "artifacts" not in self.data:
            self.data["artifacts"] = []
        if path and path not in self.data["artifacts"]:
            self.data["artifacts"].append(str(path))
            self.save()

    def update_metrics(self, metrics: Dict[str, Any]):
        if "metrics" not in self.data:
            self.data["metrics"] = {}
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
            except Exception as e:
                print(f"[WARN] Manifest save failed ({self.manifest_file}): {type(e).__name__}: {e}")


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

def _mode_from_instprm(instprm_path: Optional[str]) -> Optional[str]:
    if not instprm_path:
        return None
    path = Path(_expand(instprm_path))
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^\s*Type\s*:\s*([^\s#;]+)", text, flags=re.MULTILINE | re.IGNORECASE)
    if match:
        return "tof" if "T" in match.group(1).upper() else "cw"
    if re.search(r"^\s*(?:dif[ABC]|fltPath|2-theta)\s*:", text, flags=re.MULTILINE | re.IGNORECASE):
        return "tof"
    if re.search(r"^\s*Lam\s*:", text, flags=re.MULTILINE | re.IGNORECASE):
        return "cw"
    return None


def _guess_mode_and_tag(data_path: str, instprm_path: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    name = Path(data_path).name.lower()
    instrument_mode = _mode_from_instprm(instprm_path)
    if "hb2a" in name:
        if instrument_mode and instrument_mode != "cw":
            raise RuntimeError("Diffraction filename suggests CW/HB2A but the instrument profile is TOF")
        return "cw", "hb2a"
    if "pg3" in name:
        if instrument_mode and instrument_mode != "tof":
            raise RuntimeError("Diffraction filename suggests TOF/PG3 but the instrument profile is CW")
        return "tof", "pg3"
    return instrument_mode, None

def _default_fmthint(mode: Optional[str]) -> Optional[str]:
    """Return a GSAS-II format hint for the given instrument mode.

    Returns None to let GSAS-II auto-detect the file format, matching the
    GUI behaviour of 'Guess format from file'.  The add_histogram fallback
    chain will try xye / qye / gsas etc. if auto-detect also fails.
    An explicit fmthint can always be set in the dataset config.
    """
    return None


def _instrument_map_keys(mode: Optional[str], tag: Optional[str]) -> List[str]:
    """Return instrument_map lookup keys from most specific to most generic.

    We keep legacy aliases such as `hb2a` and `pg3` for backward
    compatibility, but prefer generic `cw` / `tof` labels in logs and new
    configs so non-HB2A CW runs do not look mislabeled.
    """
    keys: List[str] = []

    def _add(value: Optional[str]):
        if value and value not in keys:
            keys.append(value)

    _add(tag)
    mode_lower = (mode or "").lower()
    if mode_lower == "cw":
        _add("cw")
        _add("hb2a")
    elif mode_lower == "tof":
        _add("tof")
        _add("pg3")
    return keys

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
# CIF metadata parsing helpers
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
                                if val:
                                    sg_num = int(val)
                            except ValueError:
                                continue
                        break

        # Filter unhelpful data label
        if data_label:
            dl_clean = data_label.strip().strip("'").strip('"')
            if dl_clean.lower() in ("vesta_phase_1", "global", "unknown", "none", ""):
                data_label = None

        if not name:
            name = data_label

        # Some valid CIFs publish only the Hermann-Mauguin symbol.  Complete
        # whichever space-group field is missing so downstream summaries do
        # not lose the International Tables number.
        if (not sg_sym or sg_num is None) and HAVE_PYMATGEN_MATCHER:
            try:
                structure = Structure.from_file(str(cif_path))  # type: ignore[union-attr]
                analyzer = SpacegroupAnalyzer(  # type: ignore[misc]
                    structure,
                    symprec=1e-2,
                    angle_tolerance=5.0,
                )
                if not sg_sym:
                    sg_sym = str(analyzer.get_space_group_symbol())
                if sg_num is None:
                    sg_num = int(analyzer.get_space_group_number())
            except Exception as e:
                print(
                    f"[WARN] CIF symmetry analysis failed for {cif_path}: "
                    f"{type(e).__name__}: {e}"
                )
            
        sg_final = None
        if sg_sym and sg_num:
            sg_final = f"{sg_sym} ({sg_num})"
        elif sg_sym:
            sg_final = str(sg_sym)
        elif sg_num:
            sg_final = str(sg_num)

    except Exception as e:
        print(f"[WARN] CIF metadata parse failed for {cif_path}: {type(e).__name__}: {e}")
    return (name if name else None), (sg_final if sg_final else None)

# ====== KNEE HELPERS (shared in this file) ======
def _fmt_list(ids, values=None, limit=20):
    if ids is None:
        ids = []
    elif isinstance(ids, (str, bytes)):
        ids = [ids]
    else:
        try:
            ids = list(ids)
        except TypeError:
            ids = [ids]
    if not ids:
        return "[]"
    if values is None:
        s = ", ".join(str(pid) for pid in ids[:limit])
    else:
        try:
            values = list(values)
        except TypeError:
            values = []
        parts = []
        for i, pid in enumerate(ids[:limit]):
            if i < len(values):
                parts.append(f"{pid}({values[i]:.4g})")
            else:
                parts.append(str(pid))
        s = ", ".join(parts)
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
        self._main_phase_match_cache: Dict[str, Set[str]] = {}
        self._catalog_id_list_cache: Optional[Tuple[str, ...]] = None
        self._filtered_candidate_pool_cache: Dict[Tuple[str, ...], List[str]] = {}

    # ---------------------------
    # DB initialization
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
            self._catalog_id_list_cache = None
            self._filtered_candidate_pool_cache.clear()
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
    # Small DB helpers
    # ---------------------------
    def _catalog_ids(self) -> set:
        if not self.db_loader:
            return set()
        return set(self._catalog_id_list())

    def _catalog_id_list(self) -> List[str]:
        if not self.db_loader:
            return []
        if self._catalog_id_list_cache is None:
            try:
                self._catalog_id_list_cache = tuple(self.db_loader.catalog['id'].astype(str).tolist())
            except Exception:
                self._catalog_id_list_cache = tuple()
        return list(self._catalog_id_list_cache)

    def _filtered_catalog_ids(self, exclude: Set[str]) -> List[str]:
        if not exclude:
            return self._catalog_id_list()
        key = tuple(sorted(str(pid) for pid in exclude if pid is not None))
        cached = self._filtered_candidate_pool_cache.get(key)
        if cached is not None:
            return list(cached)
        filtered = self._filter_ids(self._catalog_id_list(), set(key))
        self._filtered_candidate_pool_cache[key] = list(filtered)
        return filtered

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

    def _matching_db_ids_for_main_phase(self, main_cif: Optional[str]) -> Set[str]:
        """
        Find DB entries that are structurally identical to the provided main-phase CIF.

        This is used to prevent the search DB from rediscovering the known main phase
        under a different database id, which is especially important for custom packs.
        """
        if not main_cif or not self.db_loader or not HAVE_PYMATGEN_MATCHER:
            return set()

        try:
            cache_key = str(Path(main_cif).resolve())
        except Exception:
            cache_key = str(main_cif)
        if cache_key in self._main_phase_match_cache:
            return set(self._main_phase_match_cache[cache_key])

        matches: Set[str] = set()
        try:
            ref_structure = Structure.from_file(main_cif)  # type: ignore[union-attr]
            ref_elements = sorted({str(el) for el in ref_structure.composition.as_dict().keys()})
            ref_hi, ref_lo = build_mask(ref_elements)

            ref_sg: Optional[int] = None
            try:
                ref_sg = int(SpacegroupAnalyzer(ref_structure, symprec=1e-2, angle_tolerance=5.0).get_space_group_number())  # type: ignore[misc]
            except Exception:
                ref_sg = None

            cat = self.db_loader.catalog.copy()
            cat["id"] = cat["id"].astype(str)
            if "elements_mask_hi" in cat.columns and "elements_mask_lo" in cat.columns:
                cat = cat[
                    (cat["elements_mask_hi"].astype(object).map(int) == int(ref_hi)) &
                    (cat["elements_mask_lo"].astype(object).map(int) == int(ref_lo))
                ]
            if ref_sg is not None and "space_group" in cat.columns and pd is not None:
                cat = cat[pd.to_numeric(cat["space_group"], errors="coerce") == ref_sg]

            candidate_ids = cat["id"].astype(str).tolist()
            if not candidate_ids:
                self._main_phase_match_cache[cache_key] = set()
                return set()

            matcher = StructureMatcher(primitive_cell=True, scale=True, attempt_supercell=False)  # type: ignore[operator]
            for pid in candidate_ids:
                try:
                    cand_structure = self.db_loader.load_structure(pid)
                except Exception:
                    continue
                try:
                    if matcher.fit(ref_structure, cand_structure):
                        matches.add(pid)
                except Exception:
                    continue

            if matches:
                print(f"[INFO] Excluding {len(matches)} DB phase(s) structurally matching the main CIF: {sorted(matches)}")
        except Exception as exc:
            print(f"[WARN] Could not resolve DB matches for main CIF exclusion: {exc}")
            matches = set()

        self._main_phase_match_cache[cache_key] = set(matches)
        return set(matches)

    # ---------------------------
    # Helpers for sequential passes
    # ---------------------------
    @staticmethod
    def _filter_ids(all_ids: Iterable[str], exclude: Set[str]) -> List[str]:
        return [pid for pid in all_ids if pid not in exclude]

    @staticmethod
    def _phase_weight_fraction(fractions: Dict[str, Dict[str, float]], pid: str) -> Optional[float]:
        try:
            wf = float(fractions.get(pid, {}).get("weight_fraction_pct", 0.0))
        except Exception:
            return None
        if not math.isfinite(wf) or wf < -1e-6 or wf > 1000.0:
            return None
        return max(0.0, wf)

    def _merged_named_cfg(
        self,
        ds_cfg: Dict[str, Any],
        key: str,
        defaults: Dict[str, Any],
    ) -> Dict[str, Any]:
        cfg = dict(defaults)
        top_value = self.top_cfg.get(key, {})
        ds_value = ds_cfg.get(key, {})
        if isinstance(top_value, dict):
            cfg.update(top_value)
        if isinstance(ds_value, dict):
            cfg.update(ds_value)
        return cfg

    def _main_prenudge_cfg(self, ds_cfg: Dict[str, Any], s4_cfg: Dict[str, Any]) -> Dict[str, Any]:
        return shared_main_prenudge_cfg(self.top_cfg, ds_cfg, s4_cfg)

    def _main_phase_guard_cfg(self, ds_cfg: Dict[str, Any]) -> Dict[str, Any]:
        return shared_main_phase_guard_cfg(self.top_cfg, ds_cfg)

    def _main_phase_shadow_cfg(self, ds_cfg: Dict[str, Any]) -> Dict[str, Any]:
        return shared_main_phase_shadow_cfg(self.top_cfg, ds_cfg)

    def _candidate_peak_qi_for_shadow(
        self,
        pid: str,
        cif_path: Optional[str],
        s4_cfg: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        prefer_cif = bool(cif_path and Path(cif_path).exists() and "nudg" in Path(cif_path).stem.lower())
        if prefer_cif:
            try:
                from pymatgen.io.cif import CifParser
                from database_catalog_builder import simulate_topM_peaks

                structure = CifParser(str(cif_path), occupancy_tolerance=2.0).parse_structures(primitive=False)[0]
                _hkls, _d, q, inten = simulate_topM_peaks(
                    structure,
                    float((s4_cfg.get("two_theta_range") or [0.0, 160.0])[0]),
                    float((s4_cfg.get("two_theta_range") or [0.0, 160.0])[1]),
                    int(s4_cfg.get("shadow_topM", 700)),
                    radiation=str(s4_cfg.get("radiation", "neutron")),
                    wavelength=float(s4_cfg.get("wavelength", 1.54)),
                )
                if len(q):
                    return np.asarray(q, dtype=float), np.asarray(inten, dtype=float)
            except Exception:
                pass
        try:
            q = np.asarray(self.db_loader.load_q0(pid), dtype=float)  # type: ignore[union-attr]
            i = np.asarray(self.db_loader.load_I0(pid), dtype=float)  # type: ignore[union-attr]
            if q.size and i.size:
                return q, i
        except Exception:
            pass
        if not cif_path or not Path(cif_path).exists():
            try:
                resolved = self.db_loader.ensure_cif_on_disk(pid, out_dir=Path.cwd() / "shadow_cif_cache")  # type: ignore[union-attr]
                if resolved and Path(resolved).exists():
                    cif_path = str(resolved)
            except Exception:
                pass
        if not cif_path or not Path(cif_path).exists():
            return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
        try:
            from pymatgen.io.cif import CifParser
            from database_catalog_builder import simulate_topM_peaks

            structure = CifParser(str(cif_path), occupancy_tolerance=2.0).parse_structures(primitive=False)[0]
            _hkls, _d, q, inten = simulate_topM_peaks(
                structure,
                float((s4_cfg.get("two_theta_range") or [0.0, 160.0])[0]),
                float((s4_cfg.get("two_theta_range") or [0.0, 160.0])[1]),
                int(s4_cfg.get("shadow_topM", 700)),
                radiation=str(s4_cfg.get("radiation", "neutron")),
                wavelength=float(s4_cfg.get("wavelength", 1.54)),
            )
            return np.asarray(q, dtype=float), np.asarray(inten, dtype=float)
        except Exception:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    def _main_shadow_anchor_peaks_for_pass(
        self,
        *,
        pass_ix: int,
        pm_for_tools: GSASProjectManager,
        q_main: np.ndarray,
        Q: np.ndarray,
        shadow_cfg: Dict[str, Any],
        s4_cfg: Dict[str, Any],
        ds_cfg: Dict[str, Any],
    ) -> List[float]:
        """Build robust main-phase anchor peaks for the lookalike filter.

        Prefer the fitted GSAS-II ycalc peaks because they reflect the current
        refined main-phase cell. If too few anchors are found, supplement with
        peaks rendered from the current main CIF and finally with GSAS-II
        reflection positions. This mirrors the rapid route's fallback behavior
        while keeping the full pipeline anchored to the fitted model.
        """
        if not bool(shadow_cfg.get("enabled", True)):
            return []

        tol = float(shadow_cfg.get("peak_match_tolerance_q", 0.040))
        want = max(
            1,
            min(
                int(shadow_cfg.get("top_main_peaks", 8)),
                int(shadow_cfg.get("filter_top_main_peaks", 5)),
            ),
        )
        qmin = float(np.nanmin(Q)) if np.asarray(Q).size else float("-inf")
        qmax = float(np.nanmax(Q)) if np.asarray(Q).size else float("inf")
        anchors: List[float] = []
        source_counts: Dict[str, int] = {"fitted_ycalc": 0, "active_main_cif": 0, "gsas_reflections": 0}

        def _add_anchor(qv: Any, source: str) -> bool:
            try:
                qf = float(qv)
            except Exception:
                return False
            if not math.isfinite(qf) or qf < qmin or qf > qmax:
                return False
            if any(abs(qf - old) <= tol for old in anchors):
                return False
            anchors.append(qf)
            source_counts[source] = source_counts.get(source, 0) + 1
            return True

        try:
            arrays_shadow = GSASDataExtractor.get_all_arrays(pm_for_tools.main_histogram)
            for qv in main_shadow_peaks_from_arrays(
                np.asarray(arrays_shadow.get("Q", []), dtype=float),
                np.asarray(arrays_shadow.get("ycalc", []), dtype=float),
                shadow_cfg,
            ):
                _add_anchor(qv, "fitted_ycalc")
        except Exception as exc:
            print(f"[WARN] [pass {pass_ix}] Could not build main-shadow anchors from fitted pattern: {exc}")

        main_cif = ""
        main_cif_key = ""
        if len(anchors) < want:
            for key in ("active_main_cif", "main_cif_cleanup_path", "main_cif_prenudged_path", "main_cif"):
                candidate = _expand(ds_cfg.get(key))
                if candidate and Path(candidate).exists():
                    main_cif = candidate
                    main_cif_key = key
                    break
            main_id = str(ds_cfg.get("main_phase_name") or "main_phase")
            cand_q, cand_i = self._candidate_peak_qi_for_shadow(main_id, main_cif, s4_cfg)
            if cand_q.size and cand_i.size:
                order = np.argsort(np.maximum(np.asarray(cand_i, dtype=float), 0.0))[::-1]
                for idx in order[: max(want * 4, want)]:
                    _add_anchor(float(cand_q[int(idx)]), "active_main_cif")
                    if len(anchors) >= want:
                        break

        if len(anchors) < want:
            for qv in np.asarray(q_main, dtype=float).ravel():
                _add_anchor(qv, "gsas_reflections")
                if len(anchors) >= want:
                    break

        if anchors:
            ds_cfg["_main_shadow_anchor_meta"] = {
                "source_counts": dict(source_counts),
                "active_main_cif": str(_expand(ds_cfg.get("active_main_cif")) or ""),
                "fallback_cif": str(main_cif or ""),
                "fallback_cif_key": str(main_cif_key or ""),
            }
            print(
                f"[INFO] [pass {pass_ix}] Main-shadow anchor peaks available: "
                f"{len(anchors)} anchor peak(s); sources={source_counts}; "
                f"fallback_cif_key={main_cif_key or 'none'}."
            )
        else:
            ds_cfg["_main_shadow_anchor_meta"] = {
                "source_counts": dict(source_counts),
                "active_main_cif": str(_expand(ds_cfg.get("active_main_cif")) or ""),
                "fallback_cif": "",
                "fallback_cif_key": "",
            }
            print(f"[INFO] [pass {pass_ix}] Main-shadow anchor filter has no anchor peaks for this pass.")
        return anchors[: int(shadow_cfg.get("top_main_peaks", 8))]

    def _select_phase_ids_after_hist_knee(
        self,
        final_candidates: List[Any],
        *,
        kcfg: Dict[str, Any],
        top_candidates: int,
        pass_ix: int,
        attempt_label: str,
    ) -> List[str]:
        """Apply the histogram/ML knee selection used before Stage-4 nudging."""
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

            suffix = "" if attempt_label == "initial" else f" [{attempt_label}]"
            ids_score = _knee_keep_ids(
                final_candidates,
                id_fn=lambda c: c.phase_id,
                val_fn=lambda c: _attr(c, "ml_score", "histogram_score"),
                label=f"hist/score (pass {pass_ix}{suffix})",
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
                label=f"hist/cos (pass {pass_ix}{suffix})",
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
                label=f"hist/explained (pass {pass_ix}{suffix})",
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
                label=f"hist/prob (pass {pass_ix}{suffix})",
                min_points=int(kcfg.get("min_points_pearson", kcfg.get("min_points_hist", 5))),
                min_rel_span=float(kcfg.get("min_rel_span", 0.03)),
                guard_frac=float(kcfg.get("guard_frac", 0.05)),
                max_keep_if_no_knee=int(kcfg.get("max_keep_if_no_knee", 0)),
                min_keep_at_least=int(kcfg.get("min_keep_at_least", 0)),
                max_keep_at_most=int(kcfg.get("max_keep_at_most", 0)),
            )

            union_ids = list(dict.fromkeys([*ids_score, *ids_cos, *ids_expl, *ids_prob]))
            if union_ids:
                def _ranks(seq):
                    return {pid: i + 1 for i, pid in enumerate(seq)}

                rank_maps = {"score": _ranks(ids_score), "cos": _ranks(ids_cos), "expl": _ranks(ids_expl)}
                include_prob = ids_prob and len(ids_prob) < max(2, int(0.8 * len(union_ids)))
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
                hist_map = {c.phase_id: _attr(c, "histogram_score") for c in final_candidates}

                def nz(v, default=-1e300):
                    try:
                        if math.isfinite(float(v)):
                            return float(v)
                    except Exception:
                        pass
                    return default

                phase_ids = sorted(
                    set(union_ids),
                    key=lambda pid: (
                        -rrf(pid),
                        -nz(score_map.get(pid)),
                        -nz(hist_map.get(pid)),
                        str(pid),
                    ),
                )
                if isinstance(top_candidates, int) and top_candidates > 0:
                    phase_ids = phase_ids[: int(top_candidates)]
                print(
                    f"[KNEE] hist/UNION (pass {pass_ix}{suffix}, RRF): "
                    f"selected {len(phase_ids)} -> {_fmt_list(phase_ids)}"
                )
                return list(phase_ids)
            return [str(c.phase_id) for c in final_candidates[: int(top_candidates)]]

        return [str(c.phase_id) for c in final_candidates[: int(top_candidates)]]

    def _run_full_screening_attempt(
        self,
        *,
        screener: IntegratedCandidateScreener,
        name: str,
        pass_ix: int,
        attempt_label: str,
        Q: np.ndarray,
        residual_Q: np.ndarray,
        x_native: np.ndarray,
        residual_native: np.ndarray,
        q_main: np.ndarray,
        allowed_elements: List[str],
        all_ids: List[str],
        hist_plot_cfg: Dict[str, Any],
        work_dir: str,
        anchor_ids: Optional[List[str]],
        progress_base: float,
        progress_step: float,
    ) -> List[Any]:
        if self.emitter:
            msg = "ML Screening" if attempt_label == "initial" else f"ML Screening ({attempt_label})"
            self.emitter.emit(
                f"Pass {pass_ix}",
                msg,
                progress_base + 0.1 * progress_step,
                metrics={"pass": pass_ix, "event": "screening_start", "attempt": attempt_label},
            )

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
            anchor_ids=anchor_ids or [],
        )
        suffix = "" if attempt_label == "initial" else f" ({attempt_label})"
        print(f"[RESULT] [pass {pass_ix}] ML screening{suffix} complete: found {len(final_candidates)} phases")
        if self.emitter:
            self.emitter.emit(
                f"Pass {pass_ix}",
                f"Screened {len(final_candidates)} candidates{suffix}",
                progress_base + 0.15 * progress_step,
                metrics={"pass": pass_ix, "attempt": attempt_label},
            )
        return final_candidates

    def _run_stage4_nudge_with_cache(
        self,
        *,
        phase_ids: List[str],
        nudge_cache: Dict[str, Any],
        Q: np.ndarray,
        residual_Q: np.ndarray,
        s4_cfg: Dict[str, Any],
        models_refined_dir: str,
        pass_ix: int,
        attempt_label: str,
        xray_doublet_cfg: Dict[str, Any],
        progress_base: float,
        progress_step: float,
    ) -> List[Any]:
        from lattice_nudger import LatticeNudger

        requested = [str(pid) for pid in phase_ids]
        missing = [pid for pid in requested if pid not in nudge_cache]
        reused = len(requested) - len(missing)
        suffix = "" if attempt_label == "initial" else f" ({attempt_label})"
        print(
            f"[INFO] [pass {pass_ix}] Processing {len(requested)} candidate(s){suffix}; "
            f"reuse {reused}, nudge {len(missing)}."
        )
        if self.emitter:
            self.emitter.emit(
                f"Pass {pass_ix}",
                f"Lattice nudging for {len(missing)} new / {len(requested)} total candidates",
                progress_base + 0.2 * progress_step,
                metrics={
                    "pass": pass_ix,
                    "event": "nudging_start",
                    "attempt": attempt_label,
                    "new_candidates": len(missing),
                    "reused_candidates": reused,
                },
            )

        nudge_t0 = perf_counter()
        if missing:
            try:
                nudger = LatticeNudger(
                    self.db_loader,  # type: ignore[arg-type]
                    wavelength_ang=float(s4_cfg["wavelength"]),
                    two_theta_range=tuple(s4_cfg["two_theta_range"]),
                    radiation=str(s4_cfg.get("radiation", "neutron")),
                    score_q_max=s4_cfg.get("score_q_max", 8.0),
                    lattice_tiebreak_score_tol=s4_cfg.get("lattice_tiebreak_score_tol", 5e-4),
                    xray_doublet_config=xray_doublet_cfg,
                    random_seed=s4_cfg.get("seed", 0),
                )
                new_results = nudger.optimize_many(
                    missing,
                    Q,
                    residual_Q,
                    reps=int(s4_cfg["reps"]),
                    samples=int(s4_cfg["samples"]),
                    frac_window=float(s4_cfg["frac_window"]),
                    angle_window_deg=float(s4_cfg["angle_window_deg"]),
                    out_cif_dir=models_refined_dir,
                    score_q_max=s4_cfg.get("score_q_max", 8.0),
                ) or []
                for result in new_results:
                    pid = str(getattr(result, "phase_id", "") or "")
                    if pid:
                        nudge_cache[pid] = result
                print(f"[RESULT] [pass {pass_ix}] Nudger{suffix} -> {len(new_results)} optimized structures")
            except Exception as e:
                print(f"[ERROR] [pass {pass_ix}] Lattice nudging{suffix} failed: {e}")
                traceback.print_exc()

        nudge_s = perf_counter() - nudge_t0
        print(f"[TIME] [pass {pass_ix}] Lattice nudging{suffix} wall time: {nudge_s:.3f}s")
        if self.emitter:
            self.emitter.emit(
                f"Pass {pass_ix}",
                "Lattice nudging finished",
                progress_base + 0.28 * progress_step,
                metrics={
                    "pass": pass_ix,
                    "event": "nudging_done",
                    "attempt": attempt_label,
                    "wall_s": round(nudge_s, 3),
                },
            )

        stage4_results = [nudge_cache[pid] for pid in requested if pid in nudge_cache]
        stage4_results.sort(
            key=lambda r: (
                -float(getattr(r, "best_score", 0.0) or 0.0),
                float(getattr(r, "lattice_deviation", 0.0) or 0.0),
                str(getattr(r, "phase_id", "")),
            )
        )
        return stage4_results

    def _apply_main_shadow_to_nudge_results(
        self,
        *,
        stage4_results: List[Any],
        main_shadow_q: List[float],
        target_q: np.ndarray,
        target_signal: np.ndarray,
        shadow_cfg: Dict[str, Any],
        s4_cfg: Dict[str, Any],
        diagnostics_path: Path,
        pass_ix: int,
        attempt_label: str = "initial",
    ) -> List[Dict[str, Any]]:
        """Filter nudged candidates that landed on main-phase anchor peaks."""
        records: List[Dict[str, Any]] = []
        if not bool(shadow_cfg.get("enabled", True)) or not main_shadow_q or not stage4_results:
            return records

        for result in stage4_results:
            pid = str(getattr(result, "phase_id", "") or "")
            if not pid:
                continue
            cand_q, cand_i = self._candidate_peak_qi_for_shadow(pid, getattr(result, "nudged_cif_path", None), s4_cfg)
            filtered, metrics = main_shadow_filter_decision(
                cand_q,
                cand_i,
                main_shadow_q,
                np.asarray(target_q, dtype=float),
                np.asarray(target_signal, dtype=float),
                shadow_cfg,
            )
            before = float(getattr(result, "best_score", 0.0) or 0.0)
            try:
                setattr(result, "raw_best_score", before)
                setattr(result, "main_shadow_filtered", bool(filtered))
                setattr(result, "main_shadow_filter_reason", str(metrics.get("filter_reason", "")))
            except Exception:
                pass
            records.append({
                "phase_id": pid,
                "best_score_before": before,
                **{k: v for k, v in metrics.items() if k != "candidate_peaks"},
            })

        try:
            suffix = "" if attempt_label == "initial" else f"_{attempt_label}"
            out_path = Path(diagnostics_path) / f"main_shadow_nudge_pass{pass_ix}{suffix}.json"
            out_path.write_text(
                json.dumps(
                    {
                        "pass": int(pass_ix),
                        "attempt": attempt_label,
                        "config": shadow_cfg,
                        "main_peak_q": [float(q) for q in main_shadow_q],
                        "records": records,
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            self.manifest.add_artifact(str(out_path))
        except Exception:
            pass

        filtered_records = [
            r for r in records
            if bool(r.get("filtered", False))
        ]
        if filtered_records:
            filtered_records.sort(key=lambda r: float(r.get("filter_overlap_fraction", 0.0) or 0.0), reverse=True)
            print(
                f"[INFO] [pass {pass_ix}] Main-shadow nudge filter removed "
                f"{len(filtered_records)} candidate(s): "
                + ", ".join(
                    f"{r['phase_id']} overlap={int(r.get('filter_overlap_count', 0))}/"
                    f"{int(r.get('filter_candidate_peak_count', 0))}"
                    for r in filtered_records[:8]
                )
            )
        return records

    def _refill_full_main_shadow_shortlist(
        self,
        *,
        name: str,
        pass_ix: int,
        screener: IntegratedCandidateScreener,
        final_candidates: List[Any],
        phase_ids: List[str],
        stage4_results: List[Any],
        nudge_cache: Dict[str, Any],
        initial_shadow_records: List[Dict[str, Any]],
        base_exclude_ids: Set[str],
        Q: np.ndarray,
        residual_Q: np.ndarray,
        x_native: np.ndarray,
        residual_native: np.ndarray,
        q_main: np.ndarray,
        allowed_elements: List[str],
        work_dir: str,
        ds_cfg: Dict[str, Any],
        s4_cfg: Dict[str, Any],
        shadow_cfg: Dict[str, Any],
        main_shadow_q: List[float],
        xray_doublet_cfg: Dict[str, Any],
        diagnostics_path: Path,
        models_refined_dir: str,
        top_candidates: int,
        kcfg: Dict[str, Any],
        anchor_ids: Optional[List[str]],
        progress_base: float,
        progress_step: float,
    ) -> Tuple[List[Any], List[str], List[Any], Set[str]]:
        """Refill a full-pipeline pass after main-phase lookalike filtering."""
        filtered_by_main_shadow = {
            str(r.get("phase_id"))
            for r in (initial_shadow_records or [])
            if bool(r.get("filtered", False)) and r.get("phase_id")
        }
        if not filtered_by_main_shadow:
            return final_candidates, phase_ids, stage4_results, set()

        try:
            max_refills = int(shadow_cfg.get("refill_attempts", 2))
        except Exception:
            max_refills = 2
        max_refills = max(0, max_refills)

        shadow_banned_phase_ids: Set[str] = set()
        attempt_audits: List[Dict[str, Any]] = []
        current_candidates = list(final_candidates or [])
        current_phase_ids = [str(pid) for pid in (phase_ids or [])]
        current_stage4_results = list(stage4_results or [])
        current_records = list(initial_shadow_records or [])
        refill_attempt = 0

        while filtered_by_main_shadow:
            newly_flagged = {pid for pid in filtered_by_main_shadow if pid not in shadow_banned_phase_ids}
            shadow_banned_phase_ids.update(filtered_by_main_shadow)
            kept_phase_ids = [pid for pid in current_phase_ids if str(pid) not in filtered_by_main_shadow]
            kept_stage4_results = [
                r for r in current_stage4_results
                if str(getattr(r, "phase_id", "")) not in filtered_by_main_shadow
            ]
            attempt_audits.append({
                "attempt": "initial" if refill_attempt == 0 else f"refill{refill_attempt}",
                "filtered_phase_ids": sorted(filtered_by_main_shadow),
                "newly_flagged_phase_ids": sorted(newly_flagged),
                "kept_phase_ids": list(kept_phase_ids),
                "records": [
                    {k: v for k, v in rec.items() if k != "candidate_peaks"}
                    for rec in current_records
                ],
            })

            if refill_attempt >= max_refills or not newly_flagged:
                if kept_phase_ids and kept_stage4_results:
                    print(
                        f"[WARN] [pass {pass_ix}] Main-shadow refill stopped after "
                        f"{refill_attempt} refill attempt(s); continuing with "
                        f"{len(kept_phase_ids)} non-lookalike candidate(s)."
                    )
                    current_phase_ids = kept_phase_ids
                    current_stage4_results = kept_stage4_results
                else:
                    print(
                        f"[WARN] [pass {pass_ix}] Main-shadow refill removed all candidates "
                        "and no replacement shortlist was available; stopping this pass."
                    )
                    current_phase_ids = []
                    current_stage4_results = []
                break

            refill_attempt += 1
            attempt_label = f"refill{refill_attempt}"
            print(
                f"[INFO] [pass {pass_ix}] Main-shadow refill {refill_attempt}/{max_refills}: "
                f"banning {len(shadow_banned_phase_ids)} phase(s): "
                + ", ".join(sorted(shadow_banned_phase_ids))
            )
            if self.emitter:
                self.emitter.emit(
                    f"Pass {pass_ix}",
                    "Refilling candidate shortlist without main-phase lookalikes",
                    progress_base + 0.18 * progress_step,
                    metrics={
                        "pass": pass_ix,
                        "event": "main_shadow_refill",
                        "attempt": refill_attempt,
                        "banned_phase_ids": sorted(shadow_banned_phase_ids),
                    },
                )

            active_exclude_ids = set(base_exclude_ids or set()) | shadow_banned_phase_ids
            hist_plot_cfg = self._make_hist_plot_cfg(
                f"pass{pass_ix}_main_shadow_{attempt_label}",
                work_dir,
                ds_cfg,
            )
            hist_plot_cfg["xray_doublet"] = xray_doublet_cfg
            all_ids = self._filtered_catalog_ids(active_exclude_ids)
            current_candidates = self._run_full_screening_attempt(
                screener=screener,
                name=name,
                pass_ix=pass_ix,
                attempt_label=attempt_label,
                Q=Q,
                residual_Q=residual_Q,
                x_native=x_native,
                residual_native=residual_native,
                q_main=q_main,
                allowed_elements=allowed_elements,
                all_ids=all_ids,
                hist_plot_cfg=hist_plot_cfg,
                work_dir=work_dir,
                anchor_ids=anchor_ids,
                progress_base=progress_base,
                progress_step=progress_step,
            )
            if not current_candidates:
                print(f"[WARN] [pass {pass_ix}] Main-shadow refill produced no screened candidates.")
                current_phase_ids = []
                current_stage4_results = []
                break

            current_phase_ids = self._select_phase_ids_after_hist_knee(
                current_candidates,
                kcfg=kcfg,
                top_candidates=top_candidates,
                pass_ix=pass_ix,
                attempt_label=attempt_label,
            )
            if not current_phase_ids:
                print(f"[WARN] [pass {pass_ix}] Main-shadow refill produced no candidates after histogram knee.")
                current_stage4_results = []
                break

            current_stage4_results = self._run_stage4_nudge_with_cache(
                phase_ids=current_phase_ids,
                nudge_cache=nudge_cache,
                Q=Q,
                residual_Q=residual_Q,
                s4_cfg=s4_cfg,
                models_refined_dir=models_refined_dir,
                pass_ix=pass_ix,
                attempt_label=attempt_label,
                xray_doublet_cfg=xray_doublet_cfg,
                progress_base=progress_base,
                progress_step=progress_step,
            )
            current_records = self._apply_main_shadow_to_nudge_results(
                stage4_results=current_stage4_results,
                main_shadow_q=main_shadow_q,
                target_q=np.asarray(Q, dtype=float),
                target_signal=np.asarray(residual_Q, dtype=float),
                shadow_cfg=shadow_cfg,
                s4_cfg=s4_cfg,
                diagnostics_path=diagnostics_path,
                pass_ix=pass_ix,
                attempt_label=attempt_label,
            )
            filtered_by_main_shadow = {
                str(r.get("phase_id"))
                for r in (current_records or [])
                if bool(r.get("filtered", False)) and r.get("phase_id")
            }

        try:
            summary_path = Path(diagnostics_path) / f"main_shadow_refill_pass{pass_ix}.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "pass": int(pass_ix),
                        "max_refills": int(max_refills),
                        "refill_attempts": int(refill_attempt),
                        "banned_phase_ids": sorted(shadow_banned_phase_ids),
                        "final_phase_ids": list(current_phase_ids),
                        "audits": attempt_audits,
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            if self.manifest:
                self.manifest.add_artifact(str(summary_path))
        except Exception:
            pass

        return current_candidates, current_phase_ids, current_stage4_results, shadow_banned_phase_ids

    @staticmethod
    def _smooth_1d(y: np.ndarray, width: int) -> np.ndarray:
        y = np.asarray(y, dtype=float).ravel()
        if y.size < 3 or width <= 1:
            return y.astype(float, copy=True)
        width = max(3, min(int(width), y.size if y.size % 2 else y.size - 1))
        if width % 2 == 0:
            width += 1
        if width >= y.size:
            width = y.size if y.size % 2 else max(3, y.size - 1)
        pad = width // 2
        kernel = np.ones(width, dtype=float) / float(width)
        return np.convolve(np.pad(y, pad, mode="edge"), kernel, mode="valid")

    @staticmethod
    def _select_top_peaks(
        q: np.ndarray,
        signal: np.ndarray,
        *,
        top_n: int,
        min_rel_height: float,
        min_sep_q: float,
    ) -> List[Dict[str, float]]:
        q = np.asarray(q, dtype=float).ravel()
        y = np.asarray(signal, dtype=float).ravel()
        n = min(q.size, y.size)
        if n < 3 or top_n <= 0:
            return []
        q = q[:n]
        y = np.maximum(y[:n], 0.0)
        finite = np.isfinite(q) & np.isfinite(y)
        if int(finite.sum()) < 3:
            return []
        q = q[finite]
        y = y[finite]
        smooth_width = max(3, min(17, (y.size // 150) * 2 + 3))
        ys = UnifiedPipeline._smooth_1d(y, smooth_width)
        ymax = float(np.nanmax(ys)) if ys.size else 0.0
        if not math.isfinite(ymax) or ymax <= 0.0:
            return []
        med = float(np.nanmedian(ys))
        mad = float(np.nanmedian(np.abs(ys - med)))
        threshold = max(ymax * float(min_rel_height), med + 2.5 * max(mad, 1e-12))
        local = np.where((ys[1:-1] > ys[:-2]) & (ys[1:-1] >= ys[2:]) & (ys[1:-1] >= threshold))[0] + 1
        if local.size == 0:
            local = np.argsort(ys)[::-1][: max(top_n * 3, top_n)]

        ordered = sorted((int(i) for i in local), key=lambda i: float(ys[i]), reverse=True)
        peaks: List[Dict[str, float]] = []
        for idx in ordered:
            qi = float(q[idx])
            if any(abs(qi - p["q"]) < float(min_sep_q) for p in peaks):
                continue
            peaks.append({"q": qi, "height": float(y[idx]), "smoothed_height": float(ys[idx])})
            if len(peaks) >= top_n:
                break
        return peaks

    @staticmethod
    def _observed_signal_for_prenudge(
        q: np.ndarray,
        yobs: np.ndarray,
        bg_cfg: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        q = np.asarray(q, dtype=float).ravel()
        yobs = np.asarray(yobs, dtype=float).ravel()
        n = min(q.size, yobs.size)
        q = q[:n]
        y = np.maximum(yobs[:n], 0.0)
        finite = np.isfinite(q) & np.isfinite(y)
        meta: Dict[str, Any] = {"background_method": "raw_positive"}
        if int(finite.sum()) < 50 or estimate_background is None or coerce_auto_background_params is None:
            return y, meta
        try:
            params = coerce_auto_background_params((bg_cfg or {}).get("auto_params") or {})
            background, fixed_points, resolved = estimate_background(q[finite], y[finite], params=params)
            signal = np.zeros_like(y, dtype=float)
            signal[finite] = np.maximum(y[finite] - np.asarray(background, dtype=float), 0.0)
            raw_sum = float(np.nansum(y[finite]))
            signal_sum = float(np.nansum(signal[finite]))
            if signal_sum <= max(1e-8, raw_sum * 1e-5):
                return y, meta
            meta = {
                "background_method": "auto_low_envelope",
                "background_points": int(len(fixed_points)),
                "signal_sum_raw_ratio": float(signal_sum / max(raw_sum, 1e-8)),
                "snip_iterations": int(getattr(resolved, "snip_iterations", 0) or 0),
            }
            return signal, meta
        except Exception as exc:
            meta["background_error"] = str(exc)
            return y, meta

    def _assess_main_fit_for_prenudge(
        self,
        main_ref: GSASMainPhaseRefiner,
        rwp: Optional[float],
        mode: str,
        cfg: Dict[str, Any],
        bg_cfg: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        return shared_assess_main_fit_for_prenudge(main_ref, rwp, mode, cfg, bg_cfg)
        arrays = main_ref.get_all_data()
        q = np.asarray(arrays.get("Q", np.array([])), dtype=float).ravel()
        yobs = np.asarray(arrays.get("yobs", np.array([])), dtype=float).ravel()
        ycalc = np.asarray(arrays.get("ycalc", np.array([])), dtype=float).ravel()
        n = min(q.size, yobs.size, ycalc.size)
        audit: Dict[str, Any] = {
            "enabled": bool(cfg.get("enabled", True)),
            "triggered": False,
            "reason": "not_evaluated",
            "rwp": None if rwp is None else float(rwp),
            "points": int(n),
        }
        if n < 50:
            audit["reason"] = "too_few_points"
            return audit, q[:n], np.maximum(yobs[:n], 0.0)

        q = q[:n]
        yobs = yobs[:n]
        ycalc = ycalc[:n]
        obs_signal, bg_meta = self._observed_signal_for_prenudge(q, yobs, bg_cfg)
        calc_signal = np.maximum(ycalc[: obs_signal.size], 0.0)
        if bg_meta.get("background_method") == "auto_low_envelope" and estimate_background is not None:
            try:
                params = coerce_auto_background_params((bg_cfg or {}).get("auto_params") or {}) if coerce_auto_background_params else None
                calc_bg, _fp, _resolved = estimate_background(q[: calc_signal.size], calc_signal, params=params)
                calc_signal = np.maximum(calc_signal - np.asarray(calc_bg, dtype=float), 0.0)
            except Exception:
                pass

        tol_q = float(cfg.get("peak_match_tolerance_q", 0.035))
        min_rel = float(cfg.get("min_peak_prominence_fraction", 0.04))
        obs_peaks = self._select_top_peaks(
            q,
            obs_signal,
            top_n=int(cfg.get("top_observed_peaks", 8)),
            min_rel_height=min_rel,
            min_sep_q=tol_q,
        )
        calc_peaks = self._select_top_peaks(
            q,
            calc_signal,
            top_n=int(cfg.get("top_calculated_peaks", 30)),
            min_rel_height=max(min_rel * 0.5, 0.01),
            min_sep_q=tol_q * 0.5,
        )

        matches = []
        support_weight = 0.0
        total_weight = sum(max(p.get("smoothed_height", 0.0), 0.0) for p in obs_peaks) or 1.0
        for p in obs_peaks:
            nearest = None
            if calc_peaks:
                nearest = min(calc_peaks, key=lambda c: abs(float(c["q"]) - float(p["q"])))
            dq = abs(float(nearest["q"]) - float(p["q"])) if nearest else float("inf")
            supported = bool(dq <= tol_q)
            if supported:
                support_weight += max(float(p.get("smoothed_height", 0.0)), 0.0)
            matches.append({
                "observed_q": float(p["q"]),
                "nearest_calculated_q": None if nearest is None else float(nearest["q"]),
                "delta_q": None if not math.isfinite(dq) else float(dq),
                "supported": supported,
            })

        peak_support = float(sum(1 for m in matches if m["supported"]) / max(len(matches), 1))
        weighted_support = float(support_weight / max(total_weight, 1e-12))
        strongest_supported = bool(matches[0]["supported"]) if matches else False
        strongest_gap = matches[0].get("delta_q") if matches else None
        rwp_value = float(rwp) if rwp is not None and math.isfinite(float(rwp)) else float("nan")

        low_support = weighted_support < float(cfg.get("min_peak_support", 0.50))
        rwp_bad = math.isfinite(rwp_value) and rwp_value >= float(cfg.get("trigger_rwp", 18.0))
        hard_bad = math.isfinite(rwp_value) and rwp_value >= float(cfg.get("hard_rwp", 35.0))
        strongest_bad = (
            matches
            and not strongest_supported
            and math.isfinite(rwp_value)
            and rwp_value >= float(cfg.get("min_rwp_for_strongest_trigger", 8.0))
        )
        strongest_barely_supported = (
            matches
            and strongest_supported
            and strongest_gap is not None
            and math.isfinite(float(strongest_gap))
            and float(strongest_gap) >= tol_q * float(cfg.get("strongest_barely_supported_fraction", 0.75))
            and math.isfinite(rwp_value)
            and rwp_value >= float(cfg.get("min_rwp_for_strongest_trigger", 8.0))
        )
        support_trigger_rwp = math.isfinite(rwp_value) and rwp_value >= float(
            cfg.get("min_rwp_for_peak_support_trigger", cfg.get("trigger_rwp", 18.0))
        )
        triggered = bool(hard_bad or strongest_bad or strongest_barely_supported or (low_support and support_trigger_rwp))
        if hard_bad:
            reason = "hard_rwp_trigger"
        elif strongest_bad:
            reason = "strongest_peak_not_supported"
        elif strongest_barely_supported:
            reason = "strongest_peak_barely_supported"
        elif low_support and rwp_bad:
            reason = "low_peak_support_with_high_rwp"
        elif low_support and support_trigger_rwp:
            reason = "low_peak_support_with_moderate_rwp"
        else:
            reason = "normal_refinement_supported"

        audit.update({
            "triggered": triggered,
            "reason": reason,
            "background": bg_meta,
            "peak_match_tolerance_q": tol_q,
            "observed_peak_count": len(obs_peaks),
            "calculated_peak_count": len(calc_peaks),
            "peak_support_fraction": peak_support,
            "weighted_peak_support": weighted_support,
            "strongest_peak_supported": strongest_supported,
            "strongest_peak_barely_supported": bool(strongest_barely_supported),
            "strongest_peak_delta_q": strongest_gap,
            "top_observed_q": [float(p["q"]) for p in obs_peaks[:5]],
            "top_calculated_q": [float(p["q"]) for p in calc_peaks[:10]],
            "matches": matches[: int(cfg.get("top_observed_peaks", 8))],
        })
        return audit, q, obs_signal

    @staticmethod
    def _should_adopt_prenudged_main(
        before: Dict[str, Any],
        after: Dict[str, Any],
        before_rwp: Optional[float],
        after_rwp: Optional[float],
        nudge_score: Optional[float],
        cfg: Dict[str, Any],
    ) -> Tuple[bool, str]:
        return shared_should_adopt_prenudged_main(before, after, before_rwp, after_rwp, nudge_score, cfg)
        if after_rwp is None or not math.isfinite(float(after_rwp)):
            return False, "nudged_refinement_has_invalid_rwp"
        if before_rwp is None or not math.isfinite(float(before_rwp)):
            return True, "original_rwp_invalid"
        score = float(nudge_score or 0.0)
        if score < float(cfg.get("accept_min_nudge_score", 0.02)):
            return False, "nudge_score_too_low"
        rwp_gain = float(before_rwp) - float(after_rwp)
        if rwp_gain >= 0.25:
            return True, "rwp_improved"
        accept_worsen = float(cfg.get("accept_rwp_worsen", 0.50))
        if float(after_rwp) > float(before_rwp) + accept_worsen:
            return False, "rwp_worsened_too_much"
        before_support = float(before.get("weighted_peak_support", 0.0) or 0.0)
        after_support = float(after.get("weighted_peak_support", 0.0) or 0.0)
        if (
            bool(after.get("strongest_peak_supported"))
            and not bool(before.get("strongest_peak_supported"))
        ):
            return True, "strongest_peak_support_fixed"
        if after_support >= before_support + float(cfg.get("accept_min_support_gain", 0.10)):
            return True, "peak_support_improved"
        if after_support >= before_support and bool(before.get("triggered")):
            return True, "triggered_fit_not_worse"
        return False, "no_fit_evidence_gain"

    def _main_phase_guard_violation(
        self,
        fractions: Dict[str, Dict[str, float]],
        main_phase_name: str,
        ds_cfg: Dict[str, Any],
        *,
        user_supplied_main: bool,
    ) -> Tuple[bool, Optional[float], Dict[str, Any]]:
        return shared_main_phase_guard_violation(
            fractions,
            main_phase_name,
            self.top_cfg,
            ds_cfg,
            user_supplied_main=user_supplied_main,
        )
        cfg = self._main_phase_guard_cfg(ds_cfg)
        if not bool(cfg.get("enabled", True)):
            return False, None, cfg
        if bool(cfg.get("apply_only_user_main", True)) and not user_supplied_main:
            return False, None, cfg
        main_wf = self._phase_weight_fraction(fractions, main_phase_name)
        if main_wf is None:
            return False, None, cfg
        min_wf = float(cfg.get("min_weight_pct", 5.0))
        return bool(main_wf < min_wf), main_wf, cfg

    @staticmethod
    def _copy_gpx_with_lst(src_gpx: str, dst_gpx: str) -> None:
        src = Path(src_gpx)
        dst = Path(dst_gpx)
        if not src.exists():
            raise FileNotFoundError(f"GPX source not found: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        src_lst = src.with_suffix(".lst")
        if src_lst.exists():
            shutil.copy2(src_lst, dst.with_suffix(".lst"))

    @staticmethod
    def _choose_top_new_by_wf(
        fractions: Dict[str, Dict[str, float]],
        candidates: List[str],
        pearson_best_by_pid: Optional[Dict[str, float]] = None,
    ) -> Optional[str]:
        best_pid, best_wf, best_pearson = None, -1.0, float("-inf")
        for pid in candidates:
            wf = UnifiedPipeline._phase_weight_fraction(fractions, pid)
            if wf is None:
                continue
            try:
                pearson = float((pearson_best_by_pid or {}).get(pid, float("-inf")))
            except Exception:
                pearson = float("-inf")
            if not math.isfinite(pearson):
                pearson = float("-inf")
            if (
                best_pid is None
                or wf > best_wf
                or (wf == best_wf and pearson > best_pearson)
                or (wf == best_wf and pearson == best_pearson and str(pid) < str(best_pid))
            ):
                best_pid, best_wf, best_pearson = pid, wf, pearson
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

        diag_hist_dir = (ds_cfg or {}).get("diag_hist_path") or str(Path(work_dir) / "Diagnostics" / "Screening_Histograms")
        out_png = str(Path(diag_hist_dir) / stage_tag / "hist_grid.png")
        
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
        anchor_ids: Optional[List[str]] = None,
    ) -> Tuple[List[Any], Dict[str, str], Dict[str, float], Dict[str, Any]]:

        # Re-derive paths for internal use (or accept from ds_cfg)
        diagnostics_dir = ds_cfg.get("diagnostics_path") or str(Path(work_dir) / "Diagnostics")
        diag_resid_dir = ds_cfg.get("diag_resid_path") or str(Path(diagnostics_dir) / "Residual_Scanning")
        models_dir = ds_cfg.get("models_path") or str(Path(work_dir) / "Models")
        models_ref_dir = ds_cfg.get("models_ref_path") or str(Path(models_dir) / "Reference_CIFs")
        models_refined_dir = ds_cfg.get("models_refined_path") or str(Path(models_dir) / "Refined_CIFs")
        candidate_work_dir = ds_cfg.get("tech_cand_path") or work_dir
        
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
        shadow_cfg = self._main_phase_shadow_cfg(ds_cfg)
        main_shadow_q = self._main_shadow_anchor_peaks_for_pass(
            pass_ix=pass_ix,
            pm_for_tools=pm_for_tools,
            q_main=q_main,
            Q=np.asarray(Q, dtype=float),
            shadow_cfg=shadow_cfg,
            s4_cfg=s4_cfg,
            ds_cfg=ds_cfg,
        )

        screener = IntegratedCandidateScreener(pm_for_tools, self.db_loader, profiles_dir)  # type: ignore[arg-type]

        hist_plot_cfg = self._make_hist_plot_cfg(f"pass{pass_ix}", work_dir, ds_cfg)
        xray_doublet_cfg = {"enabled": False}
        if resolve_xray_doublet_spec is not None:
            try:
                xray_doublet = resolve_xray_doublet_spec(
                    self.top_cfg,
                    dataset=ds_cfg,
                    instprm_path=instprm_path,
                    stage4=s4_cfg,
                )
                xray_doublet_cfg = xray_doublet.to_dict()
                hist_plot_cfg["xray_doublet"] = xray_doublet_cfg
                if xray_doublet.enabled:
                    desc = describe_doublet(xray_doublet) if describe_doublet else "active"
                    print(f"[INFO] PXRD doublet correction active for histogram screening: {desc}")
            except Exception as exc:
                print(f"[WARN] PXRD doublet correction could not be resolved: {exc}")
        else:
            hist_plot_cfg["xray_doublet"] = xray_doublet_cfg


        # Filter candidate pool to exclude already-accepted PIDs and main phase.
        # The full catalog id vector is stable during a run, so cache it instead
        # of rebuilding a pandas Series/list on every discovery pass.
        all_ids = self._filtered_catalog_ids(exclude_ids)

        final_candidates = self._run_full_screening_attempt(
            screener=screener,
            name=name,
            pass_ix=pass_ix,
            attempt_label="initial",
            Q=np.asarray(Q, dtype=float),
            residual_Q=np.asarray(residual_Q, dtype=float),
            x_native=np.asarray(x_native, dtype=float),
            residual_native=np.asarray(residual_native, dtype=float),
            q_main=np.asarray(q_main, dtype=float),
            allowed_elements=allowed_elements,
            all_ids=all_ids,
            hist_plot_cfg=hist_plot_cfg,
            work_dir=work_dir,
            anchor_ids=anchor_ids,
            progress_base=progress_base,
            progress_step=progress_step,
        )

        # ----- ML Surrogate Ranker (Async) -----
        diagnostics_path = Path(ds_cfg.get("diagnostics_path") or Path(work_dir) / "Diagnostics")
        status_path = diagnostics_path / f"ml_rank_status_pass{pass_ix}.json"
        ml_json_path = str(diagnostics_path / f"ml_rank_input_pass{pass_ix}.json")
        output_jsonl = str(diagnostics_path / f"ml_rank_result_pass{pass_ix}.jsonl")
        try:
            ranker_input = []
            for c in final_candidates:
                def _get_val(obj, name):
                    v = getattr(obj, name, None)
                    if v is None: return None
                    try:
                        f = float(v)
                        return None if math.isnan(f) else f
                    except Exception:
                        return None

                ranker_input.append({
                    "mp_id": c.phase_id,
                    "score": _get_val(c, "ml_score"),
                    "cos": _get_val(c, "ml_cosine"),
                    "beta": _get_val(c, "ml_beta"),
                    "alpha": _get_val(c, "ml_alpha"),
                    "p": _get_val(c, "ml_present_prob"),
                    "explained": _get_val(c, "ml_explained"),
                })
            
            # Dump to JSON
            with open(ml_json_path, "w") as f:
                json.dump({"candidates": ranker_input, "run_name": f"{name}_pass{pass_ix}"}, f, indent=2)

            assets = discover_ml_ranker_assets(PROJECT_ROOT)

            write_ranker_status(
                status_path,
                status="input_ready",
                pass_ix=pass_ix,
                input_json=ml_json_path,
                n_candidates=len(ranker_input),
                asset_source=assets.source,
                script_path=str(assets.script_path) if assets.script_path else None,
                model_path=str(assets.model_path) if assets.model_path else None,
            )

            if assets.is_ready:
                cmd = [
                    sys.executable,
                    str(assets.script_path),
                    "--model", str(assets.model_path),
                    "--input", ml_json_path,
                    "--output", output_jsonl,
                    "--format", "json",
                    "--topk", "5"
                ]
                
                print(f"[INFO] Spawning ML Ranker: {assets.script_path}")
                import subprocess
                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                )

                if completed.stdout.strip():
                    print(completed.stdout.strip())
                if completed.stderr.strip():
                    print(f"[WARN] ML Ranker stderr:\n{completed.stderr.strip()}")

                if completed.returncode != 0:
                    write_ranker_status(
                        status_path,
                        status="failed",
                        pass_ix=pass_ix,
                        input_json=ml_json_path,
                        output_jsonl=output_jsonl,
                        asset_source=assets.source,
                        script_path=str(assets.script_path),
                        model_path=str(assets.model_path),
                        returncode=completed.returncode,
                        stdout=completed.stdout[-4000:],
                        stderr=completed.stderr[-4000:],
                    )
                    print(f"[WARN] ML Ranker failed with exit code {completed.returncode}")

                # Read results and log
                if os.path.exists(output_jsonl):
                    try:
                        res_data = load_first_json_record(output_jsonl)
                        
                        ranked = res_data.get("ranked", [])
                        if ranked:
                            write_ranker_status(
                                status_path,
                                status="complete",
                                pass_ix=pass_ix,
                                input_json=ml_json_path,
                                output_jsonl=output_jsonl,
                                asset_source=assets.source,
                                script_path=str(assets.script_path),
                                model_path=str(assets.model_path),
                                n_ranked=len(ranked),
                                top_ids=res_data.get("top_ids", []),
                            )
                            print(f"\n[RESULT] [pass {pass_ix}] Top candidates by ML Ranker (Final):")
                            for r in ranked[:5]:
                                pid = r.get("mp_id")
                                score = r.get("score")
                                name_disp, sg_disp = "Unknown", "—"
                                if self.db_loader:
                                    name_disp, sg_disp = self.db_loader.get_display_name_and_sg(pid)
                                print(f"  - {pid}: {name_disp}, SG={sg_disp}, rank_score={score:.3f}")
                            print("")
                        else:
                            write_ranker_status(
                                status_path,
                                status="complete_empty",
                                pass_ix=pass_ix,
                                input_json=ml_json_path,
                                output_jsonl=output_jsonl,
                                asset_source=assets.source,
                                script_path=str(assets.script_path),
                                model_path=str(assets.model_path),
                                note="Ranker completed but returned no ranked candidates.",
                            )
                    except Exception as re:
                        write_ranker_status(
                            status_path,
                            status="failed_readback",
                            pass_ix=pass_ix,
                            input_json=ml_json_path,
                            output_jsonl=output_jsonl,
                            asset_source=assets.source,
                            script_path=str(assets.script_path),
                            model_path=str(assets.model_path),
                            error=str(re),
                        )
                        print(f"[WARN] Failed to read ML Ranker output: {re}")
                elif assets.is_ready:
                    write_ranker_status(
                        status_path,
                        status="failed_no_output",
                        pass_ix=pass_ix,
                        input_json=ml_json_path,
                        output_jsonl=output_jsonl,
                        asset_source=assets.source,
                        script_path=str(assets.script_path),
                        model_path=str(assets.model_path),
                        note="Ranker process finished without creating an output file.",
                    )

            else:
                write_ranker_status(
                    status_path,
                    status="missing_assets",
                    pass_ix=pass_ix,
                    input_json=ml_json_path,
                    asset_source=assets.source,
                    error=assets.error,
                )
                print(f"[WARN] ML Ranker assets missing: {assets.error}")

        except Exception as e:
            write_ranker_status(
                status_path,
                status="failed_exception",
                pass_ix=pass_ix,
                input_json=ml_json_path,
                output_jsonl=output_jsonl,
                error=str(e),
            )
            print(f"[WARN] Failed to spawn ML Ranker: {e}")

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
                phase_ids = sorted(
                    set(union_ids),
                    key=lambda pid: (
                        -rrf(pid),
                        -nz(score_map.get(pid)),
                        -nz(hist_map.get(pid)),
                        str(pid),
                    ),
                )
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
        nudge_t0 = perf_counter()
        nudge_s = 0.0
        try:
            nudger = LatticeNudger(
                self.db_loader,  # type: ignore[arg-type]
                wavelength_ang=float(s4_cfg["wavelength"]),
                two_theta_range=tuple(s4_cfg["two_theta_range"]),
                radiation=str(s4_cfg.get("radiation", "neutron")),
                score_q_max=s4_cfg.get("score_q_max", 8.0),
                lattice_tiebreak_score_tol=s4_cfg.get("lattice_tiebreak_score_tol", 5e-4),
                xray_doublet_config=xray_doublet_cfg,
                random_seed=s4_cfg.get("seed", 0),
            )
            stage4_results = nudger.optimize_many(
                phase_ids, Q, residual_Q,
                reps=int(s4_cfg["reps"]),
                samples=int(s4_cfg["samples"]),
                frac_window=float(s4_cfg["frac_window"]),
                angle_window_deg=float(s4_cfg["angle_window_deg"]),
                out_cif_dir=models_refined_dir,
                score_q_max=s4_cfg.get("score_q_max", 8.0),
            ) or []
            print(f"[RESULT] [pass {pass_ix}] Nudger→ {len(stage4_results)} optimized structures")
        except Exception as e:
            print(f"[ERROR] [pass {pass_ix}] Lattice nudging failed: {e}")
            traceback.print_exc()
            stage4_results = []
        finally:
            nudge_s = perf_counter() - nudge_t0
            print(f"[TIME] [pass {pass_ix}] Lattice nudging wall time: {nudge_s:.3f}s")
            if self.emitter:
                self.emitter.emit(
                    f"Pass {pass_ix}",
                    "Lattice nudging finished",
                    progress_base + 0.28 * progress_step,
                    metrics={"pass": pass_ix, "event": "nudging_done", "wall_s": round(nudge_s, 3)},
                )

        nudge_cache: Dict[str, Any] = {
            str(getattr(r, "phase_id", "")): r
            for r in (stage4_results or [])
            if str(getattr(r, "phase_id", ""))
        }

        nudge_shadow_records = self._apply_main_shadow_to_nudge_results(
            stage4_results=stage4_results,
            main_shadow_q=main_shadow_q,
            target_q=np.asarray(Q, dtype=float),
            target_signal=np.asarray(residual_Q, dtype=float),
            shadow_cfg=shadow_cfg,
            s4_cfg=s4_cfg,
            diagnostics_path=diagnostics_path,
            pass_ix=pass_ix,
            attempt_label="initial",
        )
        final_candidates, phase_ids, stage4_results, shadow_banned_phase_ids = self._refill_full_main_shadow_shortlist(
            name=name,
            pass_ix=pass_ix,
            screener=screener,
            final_candidates=final_candidates,
            phase_ids=phase_ids,
            stage4_results=stage4_results,
            nudge_cache=nudge_cache,
            initial_shadow_records=nudge_shadow_records,
            base_exclude_ids=set(exclude_ids or set()),
            Q=np.asarray(Q, dtype=float),
            residual_Q=np.asarray(residual_Q, dtype=float),
            x_native=np.asarray(x_native, dtype=float),
            residual_native=np.asarray(residual_native, dtype=float),
            q_main=np.asarray(q_main, dtype=float),
            allowed_elements=allowed_elements,
            work_dir=work_dir,
            ds_cfg=ds_cfg,
            s4_cfg=s4_cfg,
            shadow_cfg=shadow_cfg,
            main_shadow_q=main_shadow_q,
            xray_doublet_cfg=xray_doublet_cfg,
            diagnostics_path=diagnostics_path,
            models_refined_dir=models_refined_dir,
            top_candidates=top_candidates,
            kcfg=kcfg,
            anchor_ids=anchor_ids,
            progress_base=progress_base,
            progress_step=progress_step,
        )
        if shadow_banned_phase_ids:
            print(
                f"[INFO] [pass {pass_ix}] Main-shadow refill banned "
                f"{len(shadow_banned_phase_ids)} phase(s): "
                + ", ".join(sorted(shadow_banned_phase_ids))
            )
        if not phase_ids or not stage4_results:
            return final_candidates, {}, {}, {}

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
        pearson_q_max_raw = s4_cfg.get("pearson_q_max", s4_cfg.get("score_q_max", 8.0))
        try:
            pearson_q_max = float(pearson_q_max_raw)
        except Exception:
            pearson_q_max = float("nan")
        try:
            pearson_min_points = int(s4_cfg.get("pearson_min_points", 25))
        except Exception:
            pearson_min_points = 25
        pearson_x_native, pearson_residual_native, pearson_crop_meta = _crop_native_arrays_by_q(
            x_native,
            residual_native,
            Q,
            q_max=pearson_q_max,
            min_points=pearson_min_points,
        )
        if pearson_crop_meta.get("enabled"):
            print(
                f"[INFO] [pass {pass_ix}] Pearson screening cropped to Q <= "
                f"{pearson_crop_meta.get('q_max'):.3g} A^-1: "
                f"{pearson_crop_meta.get('output_points')}/{pearson_crop_meta.get('input_points')} points."
            )
        else:
            print(
                f"[INFO] [pass {pass_ix}] Pearson screening using full range "
                f"({pearson_crop_meta.get('input_points')} points; "
                f"crop reason={pearson_crop_meta.get('reason')})."
            )
        _write_xye_from_arrays(str(resid_xye), pearson_x_native, pearson_residual_native, shift_positive=True)

        if self.emitter:
             self.emitter.emit(f"Pass {pass_ix}", "Pearson Refinement (Lattice Refinement)", progress_base + 0.3 * progress_step, metrics={"pass": pass_ix, "event": "pearson_start"})

        # --- Parallel Pearson Refinement ---
        pearson_best_by_pid: Dict[str, float] = {}
        pearson_details_by_pid: Dict[str, Dict[str, Any]] = {}
        try:
            pearson_cell_min_r = float(s4_cfg.get("pearson_cell_refine_min_r", 0.5))
        except Exception:
            pearson_cell_min_r = 0.5
        pearson_defer_export = bool(s4_cfg.get("pearson_defer_export", True))

        # Create template project for this pass to avoid redundant histogram parsing
        template_gpx = str(Path(diag_resid_dir) / f"template_pass{pass_ix}.gpx")
        try:
            from gsas_core_infrastructure import GSASProjectManager
            tpm = GSASProjectManager(diag_resid_dir, f"template_pass{pass_ix}")
            if tpm.create_project():
                if tpm.add_histogram(str(resid_xye), instprm_path, fmthint="xye"):
                    tpm.save_project()
                else:
                    template_gpx = None
            else:
                template_gpx = None
        except Exception as te:
            print(f"[WARN] Failed to create template project: {te}")
            template_gpx = None

        # Decide worker count
        is_hf = "SPACE_ID" in os.environ
        cpu_count = os.cpu_count() or 1
        max_workers_cfg = int(s4_cfg.get("max_workers", 0))
        
        if max_workers_cfg > 0:
            workers = max_workers_cfg
        elif is_hf:
            workers = min(2, cpu_count) # Cap at 2 for HF Spaces OOM safety
            print(f"[INFO] Hugging Face Space detected. Capping workers to {workers} for RAM safety.")
        else:
            workers = max(1, cpu_count // 2) # Conservative default
            
        print(f"[INFO] Pearson refinement worker budget: {workers} process(es) before task-count cap.")
        
        # Ensure template file is physically on disk and flushed
        if template_gpx and 'tpm' in locals():
            try:
                tpm.save_project()
                # Force cleanup of the project object to ensure file locks are released
                del tpm
                import time
                time.sleep(0.5) 
            except Exception as e:
                print(f"[WARN] Error flushing template project: {e}")

        sys.stdout.flush()
        
        # We need to collect results. Note: we only refine the NUDGED structures if they exist,
        # otherwise we fallback to original as per the original logic.
        tasks = []
        pearson_source_cif_by_pid: Dict[str, str] = {}
        for pid in phase_ids:
            nudged_cif = result_by_pid.get(pid).nudged_cif_path if pid in result_by_pid else None
            if nudged_cif:
                cand_cif = nudged_cif
            else:
                try:
                    cand_cif = self.db_loader.ensure_cif_on_disk(pid, out_dir=cif_cache_dir)
                except Exception:
                    cand_cif = None
            
            if cand_cif:
                pearson_source_cif_by_pid[pid] = str(cand_cif)
                tasks.append((pid, cand_cif))
            else:
                pearson_best_by_pid[pid] = float("-inf")
                print(f"[RESULT] [pass {pass_ix}] {pid}: no-cif (r=-inf)")

        workers = max(1, min(int(workers), len(tasks))) if tasks else 1
        print(f"[INFO] Pearson refinement work queue: {len(tasks)} task(s), {workers} worker(s).")

        if tasks:
            candidate_work_dir = ds_cfg.get("tech_cand_path") or work_dir
            candidate_resid_dir = Path(candidate_work_dir) / "Diagnostics" / "Residual_Scanning"
            candidate_resid_dir.mkdir(parents=True, exist_ok=True)
            _write_xye_from_arrays(
                str(candidate_resid_dir / f"{name}_p{pass_ix}_residual.xye"),
                pearson_x_native,
                pearson_residual_native,
                shift_positive=True,
            )

            pearson_wall_t0 = perf_counter()
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                # Preparing arguments for _compute_pearson_with_refinement
                # def _compute_pearson_with_refinement(pid, cand_cif, name, work_dir, x_native, residual_native, instprm_path)
                future_to_pid = {
                    executor.submit(
                        _compute_pearson_with_refinement,
                        pid, cif, f"{name}_p{pass_ix}", candidate_work_dir, pearson_x_native, pearson_residual_native, instprm_path,
                        template_gpx=template_gpx,
                        engine=s4_cfg.get("pearson_engine", "surrogate"),
                        cell_refine_min_r=pearson_cell_min_r,
                        export_refined_cif=(not pearson_defer_export),
                    ): pid for pid, cif in tasks
                }

                for future in concurrent.futures.as_completed(future_to_pid):
                    pid = future_to_pid[future]
                    try:
                        result = future.result()
                        if len(result) == 4:
                            best_p, label, best_path, details = result
                        else:
                            best_p, label, best_path = result
                            details = {"pearson": float(best_p), "timings": {}}
                        pearson_best_by_pid[pid] = best_p
                        pearson_details_by_pid[pid] = details
                        if best_path and Path(best_path).exists():
                            pid_to_cif[pid] = str(Path(best_path).resolve())
                        print(f"[RESULT] [pass {pass_ix}] {pid}: {label} (r={best_p:.4f})")
                    except Exception as exc:
                        print(f"[ERROR] [pass {pass_ix}] Pearson worker failed for {pid}: {type(exc).__name__}: {exc}")
                        pearson_best_by_pid[pid] = 0.0
                        pearson_details_by_pid[pid] = {"pearson": 0.0, "error": str(exc), "timings": {}}
            pearson_wall_s = perf_counter() - pearson_wall_t0
            self._print_pearson_timing_summary(
                pass_ix=pass_ix,
                wall_s=pearson_wall_s,
                details_by_pid=pearson_details_by_pid,
                out_dir=candidate_resid_dir,
            )
        else:
            print(f"[INFO] [pass {pass_ix}] Pearson refinement skipped: no valid CIF tasks.")

        # ----- KNEE: Pearson r over current candidate set -----
        if kcfg.get("enable_pearson", False) and pearson_best_by_pid:
            pearson_items = [{"pid": pid, "r": float(pearson_best_by_pid.get(pid, float("nan")))}
                             for pid in phase_ids]
            ids_peer = _knee_keep_ids(
                pearson_items,
                id_fn=lambda x: x["pid"],
                val_fn=lambda x: x["r"],
                label=f"pearson/r (pass {pass_ix})",
                min_points=int(kcfg.get("min_points_pearson", kcfg.get("min_points_hist", 3))),
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
            def _pearson_sort_key(pid: str) -> Tuple[float, str]:
                try:
                    score = float(pearson_best_by_pid.get(pid, float("-inf")))
                except Exception:
                    score = float("-inf")
                if not math.isfinite(score):
                    score = float("-inf")
                return -score, str(pid)

            sorted_pids = sorted(
                pid_to_cif.keys(),
                key=_pearson_sort_key,
            )[:joint_top_k]
            pid_to_cif = {pid: pid_to_cif[pid] for pid in sorted_pids}
            print(f"[INFO] [pass {pass_ix}] keep top {joint_top_k} by Pearson for compare-run")

        if pearson_defer_export and pid_to_cif:
            print(f"[INFO] [pass {pass_ix}] Exporting refined CIFs for {len(pid_to_cif)} compare survivor(s).")
            exported_pid_to_cif: Dict[str, str] = {}
            for pid in list(pid_to_cif.keys()):
                source_cif = pearson_source_cif_by_pid.get(pid) or pid_to_cif.get(pid)
                if not source_cif or not Path(source_cif).exists():
                    print(f"[WARN] [pass {pass_ix}] Cannot export survivor {pid}: source CIF missing")
                    continue
                try:
                    export_diag = compute_gsas_pearson_for_cif(
                        data_path="",
                        instprm_path=instprm_path,
                        fmthint=None,
                        cif_path=source_cif,
                        work_dir=candidate_work_dir,
                        limits=None,
                        exclude_regions=None,
                        tmp_tag=f"sel_{pid}_export",
                        x_override=pearson_x_native,
                        y_override=pearson_residual_native,
                        template_gpx=template_gpx,
                        cell_refine_min_r=pearson_cell_min_r,
                        export_refined_cif=True,
                        return_diagnostics=True,
                    )
                    refined_path = export_diag.get("refined_cif_path") if isinstance(export_diag, dict) else None
                    if refined_path and Path(refined_path).exists():
                        exported_pid_to_cif[pid] = str(Path(refined_path).resolve())
                    else:
                        exported_pid_to_cif[pid] = str(Path(source_cif).resolve())
                    pearson_details_by_pid.setdefault(pid, {}).setdefault("deferred_export", export_diag)
                except Exception as exc:
                    print(f"[WARN] [pass {pass_ix}] Dropping {pid}: deferred refined-CIF export failed: {type(exc).__name__}: {exc}")
            pid_to_cif = exported_pid_to_cif

        return final_candidates, pid_to_cif, pearson_best_by_pid, result_by_pid

    def _print_pearson_timing_summary(
        self,
        *,
        pass_ix: int,
        wall_s: float,
        details_by_pid: Dict[str, Dict[str, Any]],
        out_dir: Path,
    ) -> None:
        if not details_by_pid:
            return

        def _as_float(value: Any, default: float = 0.0) -> float:
            try:
                v = float(value)
                return v if math.isfinite(v) else default
            except Exception:
                return default

        rows: List[Dict[str, Any]] = []
        for pid, details in sorted(details_by_pid.items()):
            timings = details.get("timings", {}) if isinstance(details, dict) else {}
            row = {
                "pid": pid,
                "pearson": _as_float(details.get("pearson", 0.0)) if isinstance(details, dict) else 0.0,
                "r_scale": _as_float(details.get("r_scale", float("nan")), float("nan")) if isinstance(details, dict) else float("nan"),
                "r_cell": _as_float(details.get("r_cell", float("nan")), float("nan")) if isinstance(details, dict) else float("nan"),
                "cell_refined": bool(details.get("cell_refined", False)) if isinstance(details, dict) else False,
                "cell_skip_reason": details.get("cell_skip_reason", "") if isinstance(details, dict) else "",
                "setup_s": _as_float(timings.get("setup_s", 0.0)),
                "pass1_scale_s": _as_float(timings.get("pass1_scale_s", 0.0)),
                "pass2_cell_s": _as_float(timings.get("pass2_cell_s", 0.0)),
                "export_cif_s": _as_float(timings.get("export_cif_s", 0.0)),
                "total_s": _as_float(timings.get("total_s", 0.0)),
                "error": details.get("error", "") if isinstance(details, dict) else "",
            }
            rows.append(row)

        worker_total = sum(row["total_s"] for row in rows)
        setup_total = sum(row["setup_s"] for row in rows)
        scale_total = sum(row["pass1_scale_s"] for row in rows)
        cell_total = sum(row["pass2_cell_s"] for row in rows)
        export_total = sum(row["export_cif_s"] for row in rows)
        cell_count = sum(1 for row in rows if row["cell_refined"])

        print(
            f"[TIME] [pass {pass_ix}] Pearson refinement wall={wall_s:.3f}s; "
            f"worker_total={worker_total:.3f}s; setup={setup_total:.3f}s; "
            f"scale={scale_total:.3f}s; cell={cell_total:.3f}s; export={export_total:.3f}s; "
            f"cell_refined={cell_count}/{len(rows)}"
        )

        for row in sorted(rows, key=lambda item: item["total_s"], reverse=True)[:3]:
            print(
                f"[TIME] [pass {pass_ix}] Pearson slow candidate {row['pid']}: "
                f"total={row['total_s']:.3f}s, scale={row['pass1_scale_s']:.3f}s, "
                f"cell={row['pass2_cell_s']:.3f}s, export={row['export_cif_s']:.3f}s, "
                f"r1={row['r_scale']:.4f}, r={row['pearson']:.4f}"
            )

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = out_dir / f"pearson_timing_pass{pass_ix}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"[INFO] [pass {pass_ix}] Pearson timing CSV: {csv_path}")
        except Exception as exc:
            print(f"[WARN] [pass {pass_ix}] Could not write Pearson timing CSV: {exc}")

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
        mode_from_auto = str(mode).lower() == "auto"
        if mode_from_auto:
            mode, tag = _guess_mode_and_tag(data_path, ds.get("instprm_path"))
            if not mode:
                raise RuntimeError(f"[{name}] Could not infer instrument mode. Specify CW or TOF.")
        else:
            mode = str(mode).lower()
        tag_keys = _instrument_map_keys(mode, tag)
        tag_display = tag if tag else (tag_keys[0] if tag_keys else None)
        if mode_from_auto:
            print(f"[INFO] Instrument mode: {mode.upper()}, tag: {tag_display}")
        else:
            print(f"[INFO] Instrument mode: {mode.upper()}")

        fmthint = ds.get("fmthint", "auto")
        if fmthint == "auto":
            fmthint = _default_fmthint(mode)

        instprm_path = ds.get("instprm_path")
        if instprm_path == "auto" or not instprm_path:
            imap = self.top_cfg.get("instrument_map", {})
            if "instrument_map" in ds and isinstance(ds["instrument_map"], dict):
                imap = ds["instrument_map"]
            guess_key = None
            for key in tag_keys:
                candidate = _expand(imap.get(key))
                if candidate and Path(candidate).exists():
                    instprm_path = candidate
                    guess_key = key
                    break
            if not instprm_path or not Path(instprm_path).exists():
                raise RuntimeError(
                    f"[{name}] Could not resolve instrument parameter file. "
                    f"Provide one of instrument_map.{', instrument_map.'.join(tag_keys)} or explicit path."
                )
        else:
            instprm_path = _expand(instprm_path)
            if not instprm_path or not Path(instprm_path).exists():
                raise RuntimeError(f"[{name}] Instrument parameter file not found: {instprm_path}")

        print(f"[INFO] Instrument parameters: {instprm_path}")

        # Main phase CIF and name
        main_cif = _expand(ds.get("main_cif"))
        user_supplied_main_cif = bool(main_cif)
        main_phase_name = ds.get("main_phase_name") or "auto"
        if main_phase_name == "auto":
            parsed_name, parsed_sg = _parse_cif_metadata(main_cif)
            if parsed_name:
                clean_name = str(parsed_name).replace(" ", "")
                sg_match = re.search(r"\((\d+)\)|^\s*(\d+)\s*$", str(parsed_sg or ""))
                sg_num = next((grp for grp in (sg_match.groups() if sg_match else ()) if grp), "")
                main_phase_name = f"{clean_name} (SG {sg_num})" if sg_num else clean_name
            else:
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
        tech_cand_dir = str(Path(technical_dir) / "Candidate_Refinements")
        
        # Create all directories
        if bool(self.top_cfg.get("verbose_paths", False)):
            print(f"[DEBUG] results_dir: {results_dir}")
            print(f"[DEBUG] models_dir: {models_dir}")
            print(f"[DEBUG] diagnostics_dir: {diagnostics_dir}")
            print(f"[DEBUG] technical_dir: {technical_dir}")
        
        for d in (work_dir, results_dir, results_plots_dir, 
                  models_dir, models_ref_dir, models_refined_dir,
                  diagnostics_dir, diag_hist_dir, diag_resid_dir, diag_traces_dir,
                  technical_dir, tech_projects_dir, tech_logs_dir, tech_cand_dir):
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
        defer_main_cell_polish = bool(ds.get("polish_defer_main_cell", self.top_cfg.get("polish_defer_main_cell", True)))

        # Stage 4 configuration
        ds_stage4 = ds.get("stage4", {}) or {}
        top_stage4 = self.top_cfg.get("stage4", {}) or {}
        default_score_q_max = ds_stage4.get("score_q_max", top_stage4.get("score_q_max", 8.0))
        len_tol_pct_raw = ds_stage4.get("len_tol_pct", top_stage4.get("len_tol_pct", None))
        if ds_stage4.get("frac_window", top_stage4.get("frac_window", None)) is None and len_tol_pct_raw is not None:
            frac_window_value = float(len_tol_pct_raw) / 100.0
        else:
            frac_window_value = float(ds_stage4.get("frac_window", top_stage4.get("frac_window", 0.025)))
        angle_window_value = float(
            ds_stage4.get(
                "angle_window_deg",
                top_stage4.get(
                    "angle_window_deg",
                    ds_stage4.get("ang_tol_deg", top_stage4.get("ang_tol_deg", 1.5)),
                ),
            )
        )
        s4_cfg = {
            "radiation": str(ds_stage4.get("radiation", top_stage4.get("radiation", "neutron"))).lower(),
            "wavelength": float(ds_stage4.get("wavelength", top_stage4.get("wavelength", 1.50))),
            "two_theta_range": tuple(ds_stage4.get("two_theta_range", top_stage4.get("two_theta_range", [5.0, 160.0]))),
            "frac_window": frac_window_value,
            "angle_window_deg": angle_window_value,
            "len_tol_pct": float(len_tol_pct_raw) if len_tol_pct_raw is not None else float(frac_window_value) * 100.0,
            "ang_tol_deg": angle_window_value,
            "samples": int(ds_stage4.get("samples", top_stage4.get("samples", 500))),
            "reps": int(ds_stage4.get("reps", top_stage4.get("reps", 20))),
            "seed": int(ds_stage4.get("seed", top_stage4.get("seed", 0))),
            "score_q_max": float(default_score_q_max),
            "pearson_q_max": float(ds_stage4.get("pearson_q_max", top_stage4.get("pearson_q_max", default_score_q_max))),
            "pearson_min_points": int(ds_stage4.get("pearson_min_points", top_stage4.get("pearson_min_points", 25))),
            "lattice_tiebreak_score_tol": float(ds_stage4.get("lattice_tiebreak_score_tol", top_stage4.get("lattice_tiebreak_score_tol", 5e-4))),
            "min_score": float(ds_stage4.get("min_score", top_stage4.get("min_score", 0.02))),
            "min_pearson": ds_stage4.get("min_pearson", top_stage4.get("min_pearson", "nan")),
            "pearson_cell_refine_min_r": float(ds_stage4.get("pearson_cell_refine_min_r", top_stage4.get("pearson_cell_refine_min_r", 0.50))),
            "pearson_defer_export": bool(ds_stage4.get("pearson_defer_export", top_stage4.get("pearson_defer_export", True))),
            "pearson_engine": ds_stage4.get("pearson_engine", top_stage4.get("pearson_engine", self.top_cfg.get("hist_filter", {}).get("pearson_engine", "surrogate"))),
            "diagnostics_path": diagnostics_dir,
            "tech_cand_path": tech_cand_dir,
        }

        # Database configuration
        db_cfg = dict(self.top_cfg.get("db", {}) or {})
        ds_db = ds.get("db", {})
        if isinstance(ds_db, dict):
            db_cfg.update(ds_db)
        for k in ("catalog_csv", "original_json", "profiles_dir", "stable_csv", "cif_map_json"):
            if k in ds:
                db_cfg[k] = ds[k]

        if not db_cfg:
            db_source = str(ds.get("db_source", self.top_cfg.get("db_source", "xray"))).strip().lower()
            if db_source == "neutron":
                db_cfg = dict(self.top_cfg.get("db_neutron", {}) or {})
            else:
                db_cfg = dict(self.top_cfg.get("db_xray", {}) or {})

        if not self.db_loader:
            if not self.initialize_database(db_cfg):
                print(f"[ERROR] [{name}] Database initialization failed")
                return False

        # Resolve display names and SG for main phase
        disp_main, sg_main = self._main_phase_display_and_sg(main_phase_name, main_cif)
        sg_main_disp = sg_main if sg_main not in (None, "", "—") else "unknown"

        profiles_dir = _expand(db_cfg.get("profiles_dir"))

        # Element filter configuration
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
        manual_exclude_regions = list(ds.get("exclude_regions", []) or [])
        ref_exclusion_cfg = merge_reference_phase_exclusion_config(
            self.top_cfg.get("reference_phase_exclusions", {}),
            ds.get("reference_phase_exclusions", {}),
        )

        def _reference_exclusion_state(current_instprm_path: str) -> Tuple[Dict[str, Any], List[Any]]:
            try:
                report = build_reference_phase_exclusions(
                    ref_exclusion_cfg,
                    instprm_path=current_instprm_path,
                    mode=mode,
                    limits=limits,
                )
            except Exception as exc:
                raise RuntimeError(f"[{name}] Reference/can phase exclusion setup failed: {exc}") from exc

            generated = report.get("ranges", []) or []
            combined = manual_exclude_regions + generated
            if report.get("enabled"):
                audit_payload = {
                    **report,
                    "manual_ranges": manual_exclude_regions,
                    "combined_ranges": combined,
                }
                audit_path = Path(tech_logs_dir) / "reference_phase_exclusions.json"
                audit_path.write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")
                if self.manifest:
                    self.manifest.add_artifact(str(audit_path))

            if generated:
                print(
                    "[INFO] Reference/can phase exclusions: "
                    f"{len(generated)} generated window(s) from {report.get('presets', [])} "
                    f"on {report.get('native_axis', 'native axis')}"
                )
                if bool(ref_exclusion_cfg.get("include_cu_kbeta", ref_exclusion_cfg.get("include_kbeta", False))):
                    print("[INFO] Reference/can phase exclusions include Cu K-beta companion windows.")
            elif report.get("enabled"):
                print("[INFO] Reference/can phase exclusions enabled; no Bragg windows fell inside the active range.")
            return report, combined

        reference_phase_exclusion_report, exclude_regions = _reference_exclusion_state(instprm_path)

        # Build comprehensive ds_cfg with all redirected paths
        ds_cfg = {
            **ds,
            "exclude_regions": exclude_regions,
            "reference_phase_exclusion_report": reference_phase_exclusion_report,
            "diagnostics_path": diagnostics_dir,
            "diag_hist_path": diag_hist_dir,
            "diag_resid_path": diag_resid_dir,
            "diag_traces_path": diag_traces_dir,
            "models_path": models_dir,
            "models_ref_path": models_ref_dir,
            "models_refined_path": models_refined_dir,
            "technical_path": technical_dir,
            "tech_projects_path": tech_projects_dir,
            "tech_logs_path": tech_logs_dir,
            "tech_cand_path": tech_cand_dir,
            "work_dir": work_dir,
            "active_main_cif": main_cif,
        }

        # =========================
        bench = BenchTimer(run_name=name)
        benchmark_pass_records: List[Dict[str, Any]] = []
        benchmark_final: Dict[str, Any] = {"status": "running"}

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
                            if not pm0.add_histogram(data_path, instprm_path, fmthint=fmthint, instrument_type=mode):
                                raise RuntimeError(f"[{name}] Failed to add histogram")

                            # Apply limits/exclusions
                            hist0 = pm0.main_histogram
                            abs_lo, abs_hi = read_abs_limits_or_bounds(hist0)
                            active_limits = normalize_limits(limits, abs_lo, abs_hi)
                            if active_limits:
                                lo, hi = active_limits
                            elif abs_lo is not None and abs_hi is not None:
                                lo, hi = float(abs_lo), float(abs_hi)
                            else:
                                lo = hi = None
                            normalized_exclusions = normalize_excluded_regions(exclude_regions, lo, hi)
                            if lo is not None and hi is not None:
                                ensure_usable_range(lo, hi, normalized_exclusions)
                            if active_limits:
                                set_limits(hist0, lo, hi)
                                print(f"[INFO] Data limits applied: [{lo:.2f}, {hi:.2f}]")
                            if normalized_exclusions:
                                set_excluded(hist0, normalized_exclusions)
                                print(f"[INFO] Excluded regions: {normalized_exclusions}")

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
                                ds_cfg=ds_cfg,
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
                            pearson_bg_cfg = ds.get("background", self.top_cfg.get("background", {})) or {}
                            try:
                                pearson_cell_min_r = float(s4_cfg.get("pearson_cell_refine_min_r", 0.5))
                            except Exception:
                                pearson_cell_min_r = 0.5

                            def _pearson_raw(cif_path: Optional[str], tag: str) -> Tuple[float, Optional[str]]:
                                if not cif_path or not Path(cif_path).exists():
                                    return float("nan"), None
                                try:
                                    pearson_diag = compute_gsas_pearson_for_cif(
                                        data_path=data_path,
                                        instprm_path=instprm_path,
                                        fmthint=fmthint,
                                        cif_path=cif_path,
                                        work_dir=pear_tmp_dir,
                                        limits=limits,
                                        exclude_regions=exclude_regions,
                                        tmp_tag=tag,
                                        background_config=pearson_bg_cfg,
                                        cell_refine_min_r=pearson_cell_min_r,
                                        export_refined_cif=False,
                                        return_diagnostics=True,
                                    )
                                    if isinstance(pearson_diag, dict):
                                        r = float(pearson_diag.get("pearson", float("nan")))
                                    else:
                                        r = float(pearson_diag)
                                    return r, str(cif_path)
                                except Exception as e:
                                    print(f"[WARN] Stage-0 Pearson failed for tag={tag}, cif={cif_path}: {type(e).__name__}: {e}")
                                    return float("nan"), None

                            def _export_stage0_refined_cif(cif_path: Optional[str], tag: str) -> Optional[str]:
                                if not cif_path or not Path(cif_path).exists():
                                    return cif_path
                                try:
                                    export_diag = compute_gsas_pearson_for_cif(
                                        data_path=data_path,
                                        instprm_path=instprm_path,
                                        fmthint=fmthint,
                                        cif_path=cif_path,
                                        work_dir=pear_tmp_dir,
                                        limits=limits,
                                        exclude_regions=exclude_regions,
                                        tmp_tag=tag,
                                        background_config=pearson_bg_cfg,
                                        cell_refine_min_r=pearson_cell_min_r,
                                        export_refined_cif=True,
                                        return_diagnostics=True,
                                    )
                                    refined_path = export_diag.get("refined_cif_path") if isinstance(export_diag, dict) else None
                                    if refined_path and Path(refined_path).exists():
                                        return str(Path(refined_path).resolve())
                                except Exception as e:
                                    print(f"[WARN] Stage-0 refined CIF export failed for tag={tag}, cif={cif_path}: {type(e).__name__}: {e}")
                                return cif_path

                            pearson_best_by_pid: Dict[str, float] = {}
                            path_choice_by_pid: Dict[str, Optional[str]] = {}

                            for r in (s4_res or []):
                                pid = str(getattr(r, "phase_id", ""))
                                if not pid:
                                    continue

                                nudged_cif = getattr(r, "nudged_cif_path", None)
                                if nudged_cif:
                                    p, final_cif = _pearson_raw(nudged_cif, f"{name}_stage0_sel_{pid}_nudged")
                                    pearson_best_by_pid[pid] = p
                                    path_choice_by_pid[pid]  = final_cif
                                else:
                                    try:
                                        orig_cif = self.db_loader.ensure_cif_on_disk(pid, out_dir=cif_cache_dir)  # type: ignore[union-attr]
                                    except Exception:
                                        orig_cif = None
                                    if orig_cif:
                                        p, final_cif = _pearson_raw(orig_cif, f"{name}_stage0_sel_{pid}_orig")
                                    else:
                                        p, final_cif = float("nan"), None
                                    
                                    pearson_best_by_pid[pid] = p
                                    path_choice_by_pid[pid]  = final_cif


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
                                best_cif = _export_stage0_refined_cif(best_cif, f"{name}_stage0_sel_{best_pid}_export")

                                if best_pid and best_cif and (best_pid != main_phase_name or best_cif != main_cif):
                                    print(f"[INFO] Pearson override: {main_phase_name} → {best_pid} (r={best_p:.4f})")
                                    main_phase_name = best_pid
                                    main_cif = best_cif
                                    ds_cfg["active_main_cif"] = main_cif
                                    ds_cfg["main_phase_name"] = main_phase_name
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
                    
                    self.emitter.emit("Stage 0", "Bootstrap complete", 20, metrics={"main_phase_id": main_phase_name})
                    ds_cfg["active_main_cif"] = main_cif
                    ds_cfg["main_phase_name"] = main_phase_name
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

                if not main_cif:
                    raise RuntimeError(f"[{name}] Main CIF is required for Stage-1")

                def _build_stage1_main_project(project_label: str, cif_path: str):
                    pm_local = GSASProjectManager(tech_projects_dir, project_label)
                    if not pm_local.create_project():
                        raise RuntimeError(f"[{name}] Failed to create GSAS project")
                    if not pm_local.add_histogram(data_path, instprm_path, fmthint=fmthint, instrument_type=mode):
                        raise RuntimeError(f"[{name}] Failed to add histogram")

                    hist_local = pm_local.main_histogram
                    abs_lo_local, abs_hi_local = read_abs_limits_or_bounds(hist_local)
                    active_limits_local = normalize_limits(limits, abs_lo_local, abs_hi_local)
                    if active_limits_local:
                        lo_local, hi_local = active_limits_local
                    elif abs_lo_local is not None and abs_hi_local is not None:
                        lo_local, hi_local = float(abs_lo_local), float(abs_hi_local)
                    else:
                        lo_local = hi_local = None
                    normalized_exclusions_local = normalize_excluded_regions(exclude_regions, lo_local, hi_local)
                    if lo_local is not None and hi_local is not None:
                        ensure_usable_range(lo_local, hi_local, normalized_exclusions_local)
                    if active_limits_local:
                        set_limits(hist_local, lo_local, hi_local)
                        print(f"[INFO] Data limits: [{lo_local:.2f}, {hi_local:.2f}]")
                    if normalized_exclusions_local:
                        set_excluded(hist_local, normalized_exclusions_local)
                        print(f"[INFO] Excluded regions: {normalized_exclusions_local}")

                    if not pm_local.add_phase_from_cif(cif_path, main_phase_name):
                        raise RuntimeError(f"[{name}] Failed to add main phase from CIF: {cif_path}")
                    try:
                        phase_local = pm_local.main_phase
                        phase_local.set_HAP_refinements({"Use": True, "Scale": False}, histograms=[hist_local])
                        phase_local.HAPvalue("Scale", 1.0, targethistlist=[hist_local])
                    except Exception as e:
                        print(f"[WARN] Phase initialization: {e}")

                    return (
                        pm_local,
                        hist_local,
                        GSASMainPhaseRefiner(pm_local),
                        lo_local,
                        hi_local,
                        normalized_exclusions_local,
                    )

                pm, hist, main_ref, lo, hi, normalized_exclusions = _build_stage1_main_project(
                    f"{name}_project",
                    main_cif,
                )
                _bg = ds.get("background", self.top_cfg.get("background", {})) or {}
                calibrated_instprm_path = None
                calibration_status = "not_requested"
                calibration_note = None
                calibration_rwp_before = None
                calibration_rwp_after = None
                baseline_instprm_path = instprm_path

                calib_cfg = dict(self.top_cfg.get("light_calibration", {}) or {})
                if isinstance(ds.get("light_calibration"), dict):
                    calib_cfg.update(ds.get("light_calibration") or {})

                should_light_calibrate = bool(
                    calib_cfg.get("enabled")
                    and main_cif
                    and str(mode).upper() == "CW"
                )
                if should_light_calibrate:
                    calibration_export = str(Path(technical_dir) / f"{name}_light_calibrated.instprm")
                    self.emitter.emit("Stage 1", "PXRD light calibration", 28)
                    with bench.block("S1: light pxrd calibration"):
                        calib_result = main_ref.run_light_instrument_calibration(
                            background_config=_bg,
                            bg_type=_bg.get("type"),
                            bg_terms=int(_bg["terms"]) if _bg.get("terms") is not None else None,
                            bg_coeffs=_bg.get("coeffs"),
                            zero_cycles=int(calib_cfg.get("zero_cycles", 1)),
                            profile_cycles=int(calib_cfg.get("profile_cycles", 2)),
                            profile_terms=calib_cfg.get("terms"),
                            export_path=calibration_export,
                        )
                    calibration_rwp_before = calib_result.rwp_before
                    calibration_rwp_after = calib_result.rwp_after

                    if calib_result.skipped:
                        calibration_status = "skipped"
                        calibration_note = calib_result.error_message
                        print(f"[INFO] Light PXRD calibration skipped: {calib_result.error_message}")
                    elif not calib_result.success:
                        calibration_status = "failed"
                        calibration_note = calib_result.error_message
                        try:
                            main_ref.load_instrument_profile(baseline_instprm_path)
                        except Exception as restore_err:
                            raise RuntimeError(
                                f"[{name}] Light PXRD calibration failed and the original "
                                f"instrument profile could not be restored: {restore_err}"
                            ) from restore_err
                        print(f"[WARN] Light PXRD calibration failed: {calib_result.error_message}")
                    else:
                        accept_rwp_worsen = float(calib_cfg.get("accept_rwp_worsen", 0.15))
                        before_rwp = calib_result.rwp_before
                        after_rwp = calib_result.rwp_after
                        if (
                            before_rwp is not None and after_rwp is not None
                            and math.isfinite(before_rwp) and math.isfinite(after_rwp)
                            and after_rwp <= before_rwp + accept_rwp_worsen
                            and calib_result.exported_instprm
                        ):
                            calibration_status = "adopted"
                            calibration_note = (
                                f"Rwp {before_rwp:.3f}% -> {after_rwp:.3f}% "
                                f"using {','.join(calib_result.refined_terms)}"
                            )
                            calibrated_instprm_path = calib_result.exported_instprm
                            instprm_path = calibrated_instprm_path
                            print(
                                f"[INFO] Adopted calibrated PXRD profile: {instprm_path} "
                                f"(Rwp {before_rwp:.3f}% -> {after_rwp:.3f}%)"
                            )
                            self.manifest.add_artifact(instprm_path)
                        else:
                            calibration_status = "rejected"
                            calibration_note = (
                                f"Calibration not adopted (Rwp {before_rwp:.3f}% -> {after_rwp:.3f}%)"
                                if before_rwp is not None and after_rwp is not None
                                else "Calibration not adopted"
                            )
                            try:
                                main_ref.load_instrument_profile(baseline_instprm_path)
                            except Exception as restore_err:
                                raise RuntimeError(
                                    f"[{name}] Light PXRD calibration was not adopted and the original "
                                    f"instrument profile could not be restored: {restore_err}"
                                ) from restore_err
                            print(
                                "[WARN] Light PXRD calibration was not adopted; "
                                f"Rwp changed from {before_rwp} to {after_rwp}"
                            )

                if calibrated_instprm_path and reference_phase_exclusion_report.get("enabled"):
                    reference_phase_exclusion_report, exclude_regions = _reference_exclusion_state(instprm_path)
                    ds_cfg["exclude_regions"] = exclude_regions
                    ds_cfg["reference_phase_exclusion_report"] = reference_phase_exclusion_report
                    normalized_exclusions = normalize_excluded_regions(exclude_regions, lo, hi)
                    if lo is not None and hi is not None:
                        ensure_usable_range(lo, hi, normalized_exclusions)
                    set_excluded(hist, normalized_exclusions)
                    print("[INFO] Recomputed reference/can phase exclusions after light PXRD calibration.")

                with bench.block("S1: staged refinement"):
                    main_results = main_ref.run_staged_refinement(
                        enable_cell=True,
                        background_config=_bg,
                        bg_type=_bg.get("type"),
                        bg_terms=int(_bg["terms"]) if _bg.get("terms") is not None else None,
                        bg_coeffs=_bg.get("coeffs"),
                    )
                if not main_results.success:
                    raise RuntimeError(f"[{name}] Main-phase refinement failed: {main_results.error_message}")
                print(f"[RESULT] Rwp = {main_results.rwp:.3f}%")

                prenudge_cfg = self._main_prenudge_cfg(ds, s4_cfg)
                prenudge_audit: Dict[str, Any] = {
                    "enabled": bool(prenudge_cfg.get("enabled", True)),
                    "attempted": False,
                    "adopted": False,
                    "user_supplied_main": bool(user_supplied_main_cif),
                }
                if (
                    bool(prenudge_cfg.get("enabled", True))
                    and main_cif
                    and (not bool(prenudge_cfg.get("apply_only_user_main", True)) or user_supplied_main_cif)
                ):
                    normal_fit_audit, nudge_q, nudge_signal = self._assess_main_fit_for_prenudge(
                        main_ref,
                        main_results.rwp,
                        mode,
                        prenudge_cfg,
                        _bg,
                    )
                    prenudge_audit["normal_fit"] = normal_fit_audit
                    prenudge_audit["triggered"] = bool(normal_fit_audit.get("triggered", False))

                    if normal_fit_audit.get("triggered"):
                        print(
                            "[INFO] Main-phase pre-nudge triggered: "
                            f"{normal_fit_audit.get('reason')} "
                            f"(Rwp={main_results.rwp:.3f}%, "
                            f"peak_support={float(normal_fit_audit.get('weighted_peak_support', 0.0)):.2f})"
                        )
                        if self.emitter:
                            self.emitter.emit(
                                "Stage 1",
                                "Main phase lattice pre-nudge",
                                32,
                                metrics={
                                    "reason": normal_fit_audit.get("reason"),
                                    "rwp": float(main_results.rwp),
                                    "peak_support": normal_fit_audit.get("weighted_peak_support"),
                                },
                            )
                        prenudge_audit["attempted"] = True
                        try:
                            from lattice_nudger import LatticeNudger

                            main_xray_doublet_cfg = {"enabled": False}
                            if resolve_xray_doublet_spec is not None:
                                try:
                                    main_xray_doublet = resolve_xray_doublet_spec(
                                        self.top_cfg,
                                        dataset=ds_cfg,
                                        instprm_path=instprm_path,
                                        stage4=s4_cfg,
                                    )
                                    main_xray_doublet_cfg = main_xray_doublet.to_dict()
                                except Exception as exc:
                                    prenudge_audit["xray_doublet_error"] = str(exc)

                            nudger = LatticeNudger(
                                self.db_loader,
                                wavelength_ang=float(s4_cfg["wavelength"]),
                                two_theta_range=tuple(s4_cfg["two_theta_range"]),
                                radiation=str(s4_cfg.get("radiation", "neutron")),
                                score_q_max=float(prenudge_cfg.get("score_q_max", s4_cfg.get("score_q_max", 8.0))),
                                lattice_tiebreak_score_tol=float(s4_cfg.get("lattice_tiebreak_score_tol", 5e-4)),
                                xray_doublet_config=main_xray_doublet_cfg,
                                random_seed=int(s4_cfg.get("seed", 0)),
                            )
                            nudge_result = nudger.optimize_cif(
                                main_cif,
                                f"{main_phase_name}_main",
                                nudge_q,
                                nudge_signal,
                                reps=int(prenudge_cfg.get("reps", 20)),
                                samples=int(prenudge_cfg.get("samples", 2000)),
                                frac_window=float(prenudge_cfg.get("frac_window", 0.01)),
                                angle_window_deg=float(prenudge_cfg.get("angle_window_deg", 1.0)),
                                out_cif_dir=models_refined_dir,
                                allow_inner_parallel=True,
                                score_q_max=float(prenudge_cfg.get("score_q_max", s4_cfg.get("score_q_max", 8.0))),
                            )
                            prenudge_audit["nudge"] = {
                                "score": float(nudge_result.best_score),
                                "elapsed_s": float(nudge_result.elapsed_s),
                                "candidate_count": int(nudge_result.candidate_count),
                                "scored_count": int(nudge_result.scored_count),
                                "lattice_deviation": float(getattr(nudge_result, "lattice_deviation", 0.0)),
                                "tie_count": int(getattr(nudge_result, "score_tie_count", 1)),
                                "cif": nudge_result.nudged_cif_path,
                                "params": dict(nudge_result.best_params or {}),
                            }
                            self.manifest.add_artifact(nudge_result.nudged_cif_path)

                            pm_nudged, hist_nudged, main_ref_nudged, lo_nudged, hi_nudged, exclusions_nudged = _build_stage1_main_project(
                                f"{name}_project_main_prenudged",
                                nudge_result.nudged_cif_path,
                            )
                            with bench.block("S1: staged refinement after main pre-nudge"):
                                nudged_results = main_ref_nudged.run_staged_refinement(
                                    enable_cell=True,
                                    background_config=_bg,
                                    bg_type=_bg.get("type"),
                                    bg_terms=int(_bg["terms"]) if _bg.get("terms") is not None else None,
                                    bg_coeffs=_bg.get("coeffs"),
                                )
                            prenudge_audit["nudged_refinement"] = {
                                "success": bool(nudged_results.success),
                                "rwp": None if nudged_results.rwp is None else float(nudged_results.rwp),
                                "error": nudged_results.error_message,
                            }
                            if nudged_results.success:
                                nudged_fit_audit, _q_after, _sig_after = self._assess_main_fit_for_prenudge(
                                    main_ref_nudged,
                                    nudged_results.rwp,
                                    mode,
                                    prenudge_cfg,
                                    _bg,
                                )
                                prenudge_audit["nudged_fit"] = nudged_fit_audit
                                adopt, adopt_reason = self._should_adopt_prenudged_main(
                                    normal_fit_audit,
                                    nudged_fit_audit,
                                    main_results.rwp,
                                    nudged_results.rwp,
                                    float(nudge_result.best_score),
                                    prenudge_cfg,
                                )
                                prenudge_audit["adoption_reason"] = adopt_reason
                                if adopt:
                                    print(
                                        "[INFO] Adopted pre-nudged main phase CIF: "
                                        f"Rwp {main_results.rwp:.3f}% -> {nudged_results.rwp:.3f}% "
                                        f"({adopt_reason})"
                                    )
                                    main_cif = nudge_result.nudged_cif_path
                                    ds_cfg["active_main_cif"] = main_cif
                                    ds_cfg["main_cif_prenudged_path"] = main_cif
                                    pm = pm_nudged
                                    hist = hist_nudged
                                    main_ref = main_ref_nudged
                                    main_results = nudged_results
                                    lo = lo_nudged
                                    hi = hi_nudged
                                    normalized_exclusions = exclusions_nudged
                                    disp_main, sg_main = self._main_phase_display_and_sg(main_phase_name, main_cif)
                                    sg_main_disp = sg_main if sg_main not in (None, "", "—") else "unknown"
                                    prenudge_audit["adopted"] = True
                                else:
                                    print(f"[INFO] Rejected pre-nudged main phase CIF: {adopt_reason}")
                            else:
                                print(f"[WARN] Pre-nudged main refinement failed: {nudged_results.error_message}")
                        except Exception as prenudge_err:
                            prenudge_audit["error"] = str(prenudge_err)
                            print(f"[WARN] Main-phase pre-nudge failed; continuing with normal Stage-1 result: {prenudge_err}")
                            traceback.print_exc()
                    else:
                        print(
                            "[INFO] Main-phase pre-nudge skipped: "
                            f"{normal_fit_audit.get('reason')} "
                            f"(peak_support={float(normal_fit_audit.get('weighted_peak_support', 0.0)):.2f})"
                        )
                else:
                    prenudge_audit["reason"] = "disabled_or_not_user_supplied_main"

                def _build_stage1_cleanup_project(project_label: str, cif_path: str):
                    pm_cleanup, hist_cleanup, ref_cleanup, lo_cleanup, hi_cleanup, exclusions_cleanup = _build_stage1_main_project(
                        project_label,
                        cif_path,
                    )
                    return pm_cleanup, ref_cleanup, {
                        "hist": hist_cleanup,
                        "lo": lo_cleanup,
                        "hi": hi_cleanup,
                        "excluded_regions": exclusions_cleanup,
                    }

                def _run_stage1_cleanup_refinement(refiner_obj):
                    return refiner_obj.run_staged_refinement(
                        enable_cell=True,
                        background_config=_bg,
                        bg_type=_bg.get("type"),
                        bg_terms=int(_bg["terms"]) if _bg.get("terms") is not None else None,
                        bg_coeffs=_bg.get("coeffs"),
                    )

                cleanup_anchor = run_main_phase_cleanup_if_enabled(
                    pm=pm,
                    main_ref=main_ref,
                    main_results=main_results,
                    main_cif=main_cif,
                    main_phase_name=main_phase_name,
                    top_cfg=self.top_cfg,
                    ds_cfg=ds_cfg,
                    build_project_from_cif=_build_stage1_cleanup_project,
                    run_refinement=_run_stage1_cleanup_refinement,
                    out_dir=Path(models_refined_dir),
                    user_supplied_main=bool(user_supplied_main_cif),
                    log=print,
                    audit_path=Path(tech_logs_dir) / "main_phase_cleanup.json",
                )
                cleanup_audit = cleanup_anchor.audit
                if cleanup_audit.get("adopted"):
                    main_cif = cleanup_anchor.main_cif
                    ds_cfg["active_main_cif"] = main_cif
                    ds_cfg["main_cif_cleanup_path"] = main_cif
                    pm = cleanup_anchor.pm
                    main_ref = cleanup_anchor.refiner
                    main_results = cleanup_anchor.refinement_result
                    hist = pm.main_histogram
                    cleanup_ctx = cleanup_anchor.context or {}
                    lo = cleanup_ctx.get("lo", lo)
                    hi = cleanup_ctx.get("hi", hi)
                    normalized_exclusions = cleanup_ctx.get("excluded_regions", normalized_exclusions)
                    disp_main, sg_main = self._main_phase_display_and_sg(main_phase_name, main_cif)
                    sg_main_disp = sg_main if sg_main not in (None, "", "â€”") else "unknown"
                    if cleanup_audit.get("exported_cif"):
                        self.manifest.add_artifact(str(cleanup_audit["exported_cif"]))

                prenudge_audit_path = Path(tech_logs_dir) / "main_phase_prenudge.json"
                prenudge_audit_path.write_text(json.dumps(prenudge_audit, indent=2, default=str), encoding="utf-8")
                self.manifest.add_artifact(str(prenudge_audit_path))
                cleanup_audit_path = Path(tech_logs_dir) / "main_phase_cleanup.json"
                if cleanup_audit_path.exists():
                    self.manifest.add_artifact(str(cleanup_audit_path))

                with bench.block("S1: plot main refinement"):
                    try:
                        main_plot = str(Path(plots_dir) / "main_phase_fit.png")
                        labels = {main_phase_name: f"{disp_main} — {sg_main_disp}"}
                        plot_gpx_fit_with_ticks(pm.project.filename, main_plot, phase_labels=labels)
                        print(f"[INFO] Main phase plot saved: {main_plot}")
                        self.manifest.add_artifact(main_plot)
                    except Exception as plot_err:
                        print(f"[WARN] Could not generate main phase plot: {plot_err}")

                self.emitter.emit("Stage 1", "Main phase refinement complete", 40)
                self.manifest.update_stage(
                    "Stage 1",
                    "complete",
                    {
                        "rwp": main_results.rwp,
                        "calibrated_instprm": calibrated_instprm_path,
                        "calibration_status": calibration_status,
                        "calibration_note": calibration_note,
                        "calibration_rwp_before": calibration_rwp_before,
                        "calibration_rwp_after": calibration_rwp_after,
                        "main_phase_prenudge": prenudge_audit,
                        "main_phase_cleanup": cleanup_audit,
                    },
                )

            # --------------------------------------------------------------------
            # INITIAL RESIDUAL (pass 1 seed)
            # --------------------------------------------------------------------
            with bench.block("Stage 2: Residual extraction (seed for pass 1)"):
                if self.emitter:
                    self.emitter.emit("Stage 2", "Residual extraction (seed for pass discovery)", 45)
                self.manifest.update_stage("Stage 2", "running")

                try:
                    Q, residual_Q = main_ref.get_residual_q()
                    x_native, residual_native = main_ref.get_residual_native()
                    print(f"[INFO] Extracted {len(Q)} residual points (Q-space)")
                    print(f"[INFO] Extracted {len(x_native)} residual points (native space)")
                except Exception as resid_err:
                    raise RuntimeError(
                        f"[{name}] Could not extract residual from GSAS project: {resid_err}. "
                        "Check that the data file and instrument parameter file are compatible."
                    ) from resid_err
                self.manifest.update_stage("Stage 2", "complete")

            # --------------------------------------------------------------------
            # OPTIONAL MAGNETIC RESIDUAL INDEXING PRECHECK
            # --------------------------------------------------------------------
            magnetic_summary: Dict[str, Any] = {"enabled": False, "status": "skipped"}
            magnetic_cfg = dict(self.top_cfg.get("magnetic_precheck", {}) or {})
            if isinstance(ds.get("magnetic_precheck"), dict):
                magnetic_cfg.update(ds.get("magnetic_precheck") or {})
            if bool(magnetic_cfg.get("enabled", False)):
                with bench.block("Magnetic precheck: residual k-vector indexing"):
                    mag_dir = Path(work_dir) / "Magnetic_Precheck"
                    try:
                        self.emitter.emit(
                            "Magnetic Precheck",
                            "Magnetic residual indexing started",
                            46,
                        )
                        magnetic_summary = run_magnetic_precheck(
                            q=Q,
                            residual=residual_Q,
                            main_cif=main_cif,
                            refined_gpx=str(getattr(pm.project, "filename", Path(tech_projects_dir) / f"{name}_project.gpx")),
                            phase_name=main_phase_name,
                            out_dir=mag_dir,
                            config=magnetic_cfg,
                        )
                        print(
                            "[INFO] Magnetic precheck: "
                            f"evidence={magnetic_summary.get('evidence')} "
                            f"best_k={magnetic_summary.get('best_k')} "
                            f"score={magnetic_summary.get('best_score')}"
                        )
                        self.emitter.emit(
                            "Magnetic Precheck",
                            f"Magnetic evidence: {magnetic_summary.get('evidence', 'unknown')}",
                            48,
                            metrics={
                                "evidence": magnetic_summary.get("evidence"),
                                "best_k": magnetic_summary.get("best_k"),
                                "score": magnetic_summary.get("best_score"),
                            },
                        )
                        self.manifest.update_stage("Magnetic Precheck", "complete", magnetic_summary)
                        for artifact in (magnetic_summary.get("artifacts") or {}).values():
                            self.manifest.add_artifact(str(artifact))
                    except Exception as mag_err:
                        magnetic_summary = {
                            "enabled": True,
                            "status": "failed",
                            "evidence": "not_available",
                            "reason": str(mag_err),
                        }
                        mag_dir.mkdir(parents=True, exist_ok=True)
                        (mag_dir / "magnetic_precheck_summary.json").write_text(
                            json.dumps(magnetic_summary, indent=2),
                            encoding="utf-8",
                        )
                        print(f"[WARN] Magnetic precheck failed: {mag_err}")
                        self.manifest.update_stage("Magnetic Precheck", "failed", magnetic_summary)

            # ========================= SEQUENTIAL PASSES =========================
            accepted: List[str] = []   # accepted impurities in order
            # Use the active Stage-1 project. This may be the original main fit or
            # an adopted pre-nudged main-phase project.
            kept_gpx = str(Path(pm.project.filename))
            kept_rwp = main_results.rwp

            for pass_ix in range(1, int(seq_max_passes) + 1):

                print("\n" + "═" * 80)
                print(f"SEQUENTIAL PASS {pass_ix} — candidate discovery")
                print("═" * 80)
                progress_base = 40 + ((pass_ix - 1) / seq_max_passes) * 50
                progress_step = 50 / seq_max_passes
                self.emitter.emit(f"Pass {pass_ix}", f"Discovery pass {pass_ix} starting", progress_base, metrics={"pass": pass_ix, "event": "pass_start"})
                pass_record: Dict[str, Any] = {
                    "pass": int(pass_ix),
                    "accepted_before": list(accepted),
                    "rwp_before": float(kept_rwp),
                }
                benchmark_pass_records.append(pass_record)

                # For pass > 1, recompute residual from the kept GPX
                if pass_ix > 1:
                    with bench.block(f"Pass {pass_ix}: residual from kept GPX"):
                        x_native, residual_native, Q, residual_Q, kept_rwp, hist_name, _ = extract_residual_from_gpx(kept_gpx)
                        print(f"[INFO] [pass {pass_ix}] Kept GPX: {kept_gpx}, Rwp={kept_rwp:.3f}%")
                        pass_record["rwp_before"] = float(kept_rwp)

                main_phase_db_matches = self._matching_db_ids_for_main_phase(main_cif)
                exclude_ids = {main_phase_name, *accepted, *main_phase_db_matches}
                anchor_ids_for_pass = list(accepted)

                with bench.block(f"Pass {pass_ix}: screen + nudge + pearson"):
                    try:
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
                                    ds_cfg=ds_cfg,
                                    anchor_ids=anchor_ids_for_pass,
                                )
                        finite_pearson_items = []
                        for pid, score in (pearson_best_by_pid or {}).items():
                            try:
                                score_f = float(score)
                            except Exception:
                                continue
                            if math.isfinite(score_f):
                                finite_pearson_items.append((str(pid), score_f))
                        finite_pearson_items.sort(key=lambda item: (-item[1], item[0]))
                        pass_record.update({
                            "screened_candidate_count": len(final_candidates or []),
                            "nudged_candidate_count": len(result_by_pid or {}),
                            "pearson_candidate_count": len(pearson_best_by_pid or {}),
                            "compare_candidate_count": len(pid_to_cif or {}),
                            "compare_candidate_ids": list((pid_to_cif or {}).keys()),
                            "pearson_top": [
                                {"phase_id": pid, "pearson": score}
                                for pid, score in finite_pearson_items[:10]
                            ],
                        })
                    except RuntimeError as e:
                        msg = str(e)
                        if msg.startswith("No candidates"):
                            print(f"[INFO] [pass {pass_ix}] {msg}; stopping discovery.")
                            reason = "no_candidates"
                            if "element filtering" in msg:
                                reason = "no_candidates_element_filter"
                            elif "space-group" in msg:
                                reason = "no_candidates_space_group"
                            elif "stability" in msg:
                                reason = "no_candidates_stability"
                            elif "histogram screening" in msg:
                                reason = "no_candidates_histogram"
                            self.manifest.update_stage(f"Pass {pass_ix}", "complete", {"status": "stopped", "reason": reason})
                            pass_record.update({"status": "stopped", "reason": reason})
                            final_candidates = []
                            pid_to_cif = {}
                            pearson_best_by_pid = {}
                            result_by_pid = {}
                            break
                        raise


                if not pid_to_cif:
                    print(f"[INFO] [pass {pass_ix}] No candidates passed Pearson/score filters. Stopping.")
                    self.manifest.update_stage(f"Pass {pass_ix}", "complete", {"status": "stopped", "reason": "no candidates"})
                    pass_record.update({"status": "stopped", "reason": "no candidates"})
                    break

                # Compare-run: kept GPX + top-K new candidates to decide which ONE to accept
                with bench.block(f"Pass {pass_ix}: compare-run joint refinement (kept + top-K new)"):
                    if self.emitter:
                         self.emitter.emit(f"Pass {pass_ix}", "Joint refinement (compare-run) started", progress_base + 0.4 * progress_step, metrics={"pass": pass_ix, "event": "joint_compare_start"})

                    pass_dir = Path(joint_dir)
                    cmp_gpx = str(pass_dir / f"seq_pass{pass_ix}_compare.gpx")
                    compare_pid_to_cif = dict(pid_to_cif)
                    adaptive_compare_enabled = bool(ds.get("adaptive_compare_enabled", self.top_cfg.get("adaptive_compare_enabled", True)))
                    adaptive_compare_keep = int(ds.get("adaptive_compare_keep", self.top_cfg.get("adaptive_compare_keep", 2)))
                    adaptive_compare_cycles = int(ds.get("adaptive_compare_cycles", self.top_cfg.get("adaptive_compare_cycles", 1)))
                    adaptive_compare_keep = max(1, adaptive_compare_keep)
                    adaptive_compare_cycles = max(1, min(int(max_joint_cycles), adaptive_compare_cycles))

                    if adaptive_compare_enabled and len(compare_pid_to_cif) > adaptive_compare_keep and int(max_joint_cycles) > adaptive_compare_cycles:
                        quick_gpx = str(pass_dir / f"seq_pass{pass_ix}_compare_quick.gpx")
                        print(
                            f"[INFO] [pass {pass_ix}] Adaptive compare quick pass: "
                            f"{len(compare_pid_to_cif)} candidates, {adaptive_compare_cycles} cycle(s)."
                        )
                        quick_fractions = joint_refine_add_phases(
                            base_gpx=kept_gpx,
                            out_gpx=quick_gpx,
                            main_phase_name=main_phase_name,
                            pid_to_cif_new=compare_pid_to_cif,
                            hap_init=hap_init,
                            max_joint_cycles=adaptive_compare_cycles,
                            preserve_existing_scales=True,
                        )
                        try:
                            _, _, _, _, rwp_compare_quick, _, _ = extract_residual_from_gpx(quick_gpx)
                        except Exception:
                            rwp_compare_quick = float("nan")
                        self.manifest.add_artifact(quick_gpx)

                        def _quick_sort_key(pid: str) -> Tuple[float, float, str]:
                            wf = self._phase_weight_fraction(quick_fractions, pid)
                            if wf is None:
                                wf = float("-inf")
                            try:
                                pr = float(pearson_best_by_pid.get(pid, float("-inf")))
                            except Exception:
                                pr = float("-inf")
                            if not math.isfinite(pr):
                                pr = float("-inf")
                            return (-float(wf), -pr, str(pid))

                        quick_ranked = sorted(compare_pid_to_cif.keys(), key=_quick_sort_key)
                        pearson_ranked = sorted(
                            compare_pid_to_cif.keys(),
                            key=lambda pid: (-(float(pearson_best_by_pid.get(pid, float("-inf"))) if math.isfinite(float(pearson_best_by_pid.get(pid, float("-inf")))) else float("-inf")), str(pid)),
                        )
                        keep_order: List[str] = []
                        for pid in [*quick_ranked, *pearson_ranked[:1]]:
                            if pid not in keep_order:
                                keep_order.append(pid)
                            if len(keep_order) >= adaptive_compare_keep:
                                break
                        compare_pid_to_cif = {pid: compare_pid_to_cif[pid] for pid in keep_order}
                        pid_to_cif = compare_pid_to_cif
                        pass_record.update({
                            "adaptive_compare_enabled": True,
                            "adaptive_compare_initial_count": len(quick_ranked),
                            "adaptive_compare_final_count": len(compare_pid_to_cif),
                            "adaptive_compare_candidate_ids": list(compare_pid_to_cif.keys()),
                            "rwp_compare_quick": float(rwp_compare_quick) if math.isfinite(float(rwp_compare_quick)) else None,
                        })
                        print(
                            f"[INFO] [pass {pass_ix}] Adaptive compare kept "
                            f"{len(compare_pid_to_cif)}/{len(quick_ranked)} candidates: {_fmt_list(compare_pid_to_cif.keys())}"
                        )

                    fractions_cmp = joint_refine_add_phases(
                        base_gpx=kept_gpx,
                        out_gpx=cmp_gpx,
                        main_phase_name=main_phase_name,
                        pid_to_cif_new=compare_pid_to_cif,
                        hap_init=hap_init,
                        max_joint_cycles=max_joint_cycles,
                        preserve_existing_scales=True,
                    )
                    # Keep TRIAL BLEND Rwp (kept + top-K)
                    _, _, _, _, rwp_compare, _, _ = extract_residual_from_gpx(cmp_gpx)
                    self.manifest.add_artifact(cmp_gpx)
                    pass_record["rwp_compare"] = float(rwp_compare)
                    pass_record["compare_candidate_count"] = len(pid_to_cif or {})
                    pass_record["compare_candidate_ids"] = list((pid_to_cif or {}).keys())


                # Choose one new impurity by highest Wt%
                new_candidates = list(pid_to_cif.keys())
                best_new = self._choose_top_new_by_wf(fractions_cmp, new_candidates, pearson_best_by_pid)
                if best_new is None:
                    print(f"[INFO] [pass {pass_ix}] No usable candidate in compare-run. Stopping.")
                    self.manifest.update_stage(f"Pass {pass_ix}", "complete", {"status": "stopped", "reason": "no usable candidate in compare-run"})
                    pass_record.update({"status": "stopped", "reason": "no usable candidate in compare-run"})
                    break

                wf_best = self._phase_weight_fraction(fractions_cmp, best_new)
                if wf_best is None:
                    print(f"[INFO] [pass {pass_ix}] Candidate {best_new} has invalid trial Wt%. Stopping.")
                    self.manifest.update_stage(f"Pass {pass_ix}", "complete", {"status": "stopped", "reason": "invalid trial weight fraction", "candidate": best_new})
                    pass_record.update({"status": "stopped", "reason": "invalid trial weight fraction", "candidate": best_new})
                    break
                pearson_best = float(pearson_best_by_pid.get(best_new, float("-inf")))
                pass_record.update({
                    "candidate_selected_by_compare": best_new,
                    "wf_trial": float(wf_best),
                    "pearson_selected": float(pearson_best) if math.isfinite(pearson_best) else None,
                })
                if wf_best < float(min_impurity_percent):
                    print(f"[INFO] [pass {pass_ix}] Top Wt% {wf_best:.3f} < threshold {min_impurity_percent}. Stopping.")
                    self.manifest.update_stage(f"Pass {pass_ix}", "complete", {"status": "stopped", "reason": "impurity below threshold", "wf_best": wf_best})
                    pass_record.update({"status": "stopped", "reason": "impurity below threshold"})
                    break
                if not math.isnan(float(s4_cfg.get("min_pearson", "nan"))) and pearson_best < float(s4_cfg["min_pearson"]):
                    print(f"[INFO] [pass {pass_ix}] Pearson {pearson_best:.3f} < min_pearson {s4_cfg['min_pearson']}. Stopping.")
                    self.manifest.update_stage(f"Pass {pass_ix}", "complete", {"status": "stopped", "reason": "pearson below threshold", "pearson_best": pearson_best})
                    pass_record.update({"status": "stopped", "reason": "pearson below threshold"})
                    break

                # Commit-run: kept = main + accepted + best_new; then POLISH
                with bench.block(f"Pass {pass_ix}: commit-run joint refinement (kept = main + accepted + best_new)"):
                    self.emitter.emit(f"Pass {pass_ix}", "Joint refinement (commit-run) started", progress_base + 0.6 * progress_step, metrics={"pass": pass_ix, "event": "joint_refine_start"})
                    commit_gpx = str(Path(joint_dir) / f"seq_pass{pass_ix}_kept.gpx")
                    pid_to_cif_new = {best_new: pid_to_cif[best_new] if best_new in pid_to_cif else
                                      self.db_loader.ensure_cif_on_disk(best_new, out_dir=models_ref_dir)}
                    commit_reused_compare = False

                    if len(pid_to_cif) == 1 and best_new in pid_to_cif:
                        print(
                            f"[INFO] [pass {pass_ix}] Reusing single-candidate compare-run "
                            "as commit-run; phase set is identical."
                        )
                        self._copy_gpx_with_lst(cmp_gpx, commit_gpx)
                        fractions_kept_quick = fractions_cmp
                        rwp_kept = rwp_compare
                        commit_reused_compare = True
                    else:
                        fractions_kept_quick = joint_refine_add_phases(
                            base_gpx=kept_gpx,   # Build upon the previously polished/kept project
                            out_gpx=commit_gpx,
                            main_phase_name=main_phase_name,
                            pid_to_cif_new=pid_to_cif_new,  # Add ONLY the new phase (others are already in kept_gpx)
                            hap_init=hap_init,
                            max_joint_cycles=max_joint_cycles,
                            preserve_existing_scales=True,  # Critical: don't reset previous phases
                        )
                        # Rwp for quick accept
                        _, _, _, _, rwp_kept, _, _ = extract_residual_from_gpx(commit_gpx)
                    self.manifest.add_artifact(commit_gpx)
                    pass_record.update({
                        "commit_reused_compare": bool(commit_reused_compare),
                        "rwp_commit_quick": float(rwp_kept),
                    })


                    # POLISH
                    if self.emitter:
                         self.emitter.emit(f"Pass {pass_ix}", "Polishing model", progress_base + 0.8 * progress_step, metrics={"pass": pass_ix, "event": "polish_start"})
                    
                    polished_gpx = str(Path(joint_dir) / f"seq_pass{pass_ix}_kept_polished.gpx")
                    polish_trace = str(Path(joint_dir) / f"seq_pass{pass_ix}_kept_polished.polish_trace.json")
                    fractions_polished, rwp_polished = joint_refine_polish(
                        base_gpx=commit_gpx,
                        out_gpx=polished_gpx,
                        main_phase_name=main_phase_name,
                        max_polish_cycles=int(ds.get("polish_cycles", self.top_cfg.get("polish_cycles", 10))),
                        refine_cell_for_all=bool(ds.get("polish_refine_cell", self.top_cfg.get("polish_refine_cell", True))),
                        refine_background=bool(ds.get("polish_refine_background", self.top_cfg.get("polish_refine_background", True))),
                        target_phase_names=[best_new],
                        polish_strategy=str(ds.get("polish_strategy", self.top_cfg.get("polish_strategy", "adaptive"))),
                        refine_main_cell=(False if defer_main_cell_polish else bool(ds.get("polish_refine_main_cell", self.top_cfg.get("polish_refine_main_cell", True)))),
                        refine_existing_cells=bool(ds.get("polish_refine_existing_cells", self.top_cfg.get("polish_refine_existing_cells", False))),
                        escalate_on_failure=bool(ds.get("polish_escalate_on_failure", self.top_cfg.get("polish_escalate_on_failure", True))),
                        stabilization_cycles=int(ds.get("polish_stabilization_cycles", self.top_cfg.get("polish_stabilization_cycles", 1))),
                        cell_trial_cycles=int(ds.get("polish_cell_trial_cycles", self.top_cfg.get("polish_cell_trial_cycles", 1))),
                        final_polish_cycles=int(ds.get("polish_final_cycles", self.top_cfg.get("polish_final_cycles", 0))),
                        skip_fresh_lst_regen=bool(ds.get("polish_skip_fresh_lst_regen", self.top_cfg.get("polish_skip_fresh_lst_regen", True))),
                        trace_path=polish_trace,
                    )

                    wf_polished = self._phase_weight_fraction(fractions_polished, best_new)
                    if wf_polished is None or wf_polished < float(min_impurity_percent):
                        wf_label = "invalid" if wf_polished is None else f"{wf_polished:.3f}"
                        print(
                            f"[INFO] [pass {pass_ix}] Candidate {best_new} did not survive commit/polish "
                            f"(polished Wt%={wf_label}, threshold={min_impurity_percent}). Stopping before accepting."
                        )
                        self.manifest.update_stage(
                            f"Pass {pass_ix}",
                            "complete",
                            {
                                "status": "stopped",
                                "reason": "candidate failed commit polish sanity",
                                "candidate": best_new,
                                "wf_trial": wf_best,
                                "wf_polished": wf_polished,
                            },
                        )
                        pass_record.update({
                            "status": "stopped",
                            "reason": "candidate failed commit polish sanity",
                            "wf_polished": wf_polished,
                        })
                        break

                    main_guard_hit, main_wf_polished, main_guard_cfg = self._main_phase_guard_violation(
                        fractions_polished,
                        main_phase_name,
                        ds,
                        user_supplied_main=user_supplied_main_cif,
                    )
                    pass_record["main_wf_polished"] = (
                        None if main_wf_polished is None else float(main_wf_polished)
                    )
                    if main_guard_hit:
                        min_main_wf = float(main_guard_cfg.get("min_weight_pct", 5.0))
                        print(
                            f"[WARN] [pass {pass_ix}] Main phase collapse guard stopped acceptance: "
                            f"{main_phase_name} refined to {main_wf_polished:.3f} wt%, "
                            f"below {min_main_wf:.3f} wt%. Candidate {best_new} is not accepted."
                        )
                        self.manifest.update_stage(
                            f"Pass {pass_ix}",
                            "complete",
                            {
                                "status": "stopped",
                                "reason": "main phase collapse guard",
                                "candidate": best_new,
                                "main_phase": main_phase_name,
                                "main_wf_polished": main_wf_polished,
                                "min_main_wf": min_main_wf,
                                "wf_trial": wf_best,
                                "wf_polished": wf_polished,
                            },
                        )
                        pass_record.update({
                            "status": "stopped",
                            "reason": "main phase collapse guard",
                            "candidate": best_new,
                            "main_phase": main_phase_name,
                            "main_wf_polished": main_wf_polished,
                            "min_main_wf": min_main_wf,
                            "wf_polished": wf_polished,
                        })
                        break

                    accepted.append(best_new)
                    kept_gpx = polished_gpx
                    kept_rwp_new = rwp_polished
                    fractions_kept = fractions_polished
                    self.manifest.add_artifact(kept_gpx)
                    if Path(polish_trace).exists():
                        self.manifest.add_artifact(polish_trace)
                    pass_record.update({
                        "status": "accepted",
                        "accepted_this_pass": best_new,
                        "accepted_after": list(accepted),
                        "rwp_polished": float(rwp_polished),
                        "wf_polished": float(wf_polished) if wf_polished is not None else None,
                        "polish_trace": polish_trace,
                    })


                # PLOTS: TRIAL BLEND and POLISHED ACCEPTED MODEL
                with bench.block(f"Pass {pass_ix}: plots"):
                    trial_png = str(Path(plots_dir) / f"seq_pass{pass_ix}_trial_blend.png")
                    final_png = str(Path(plots_dir) / f"seq_pass{pass_ix}_accepted_model.png")
                    
                    # Build labels mapping ID -> "Name — SG" for all active phases
                    active_pids = [main_phase_name] + accepted
                    labels = {}
                    for p in active_pids:
                        if p == main_phase_name:
                            labels[p] = f"{disp_main} — {sg_main_disp}"
                        else:
                            d, s = self._safe_db_display_and_sg(p)
                            s_disp = s if s not in (None, "", "—") else "unknown"
                            labels[p] = f"{d} — {s_disp}"
                            
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

                self.emitter.emit(f"Pass {pass_ix}", f"Discovery pass {pass_ix} complete", progress_base + progress_step, metrics={"pass": pass_ix, "event": "pass_end"})
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
                    stop_reason = (
                        "accepted model worsened Rwp"
                        if delta < 0
                        else "Rwp improvement below configured threshold"
                    )
                    if delta < 0:
                        print(f"[INFO] Early stop: Rwp worsened by {abs(delta):.3f} (threshold {rwp_improve_eps}); stopping.")
                    else:
                        print(f"[INFO] Early stop: ΔRwp={delta:.3f} < eps {rwp_improve_eps}; stopping.")
                    pass_record.update({
                        "early_stop": True,
                        "stop_reason": stop_reason,
                        "rwp_improvement": float(delta),
                        "rwp_improvement_threshold": float(rwp_improve_eps),
                    })
                    self.manifest.update_stage(f"Pass {pass_ix}", "complete", {
                        "status": "stopped_after_acceptance",
                        "reason": stop_reason,
                        "rwp_before": kept_rwp,
                        "rwp_accepted_polished": kept_rwp_new,
                        "rwp_improvement": float(delta),
                        "rwp_improvement_threshold": float(rwp_improve_eps),
                        "accepted_this_pass": best_new,
                        "accepted_phases": accepted_labels,
                    })
                    kept_rwp = kept_rwp_new
                    break
                kept_rwp = kept_rwp_new

            pass_stop_reason = next(
                (
                    str(record.get("stop_reason") or record.get("reason"))
                    for record in reversed(benchmark_pass_records)
                    if record.get("stop_reason") or record.get("reason")
                ),
                "pass budget completed" if len(benchmark_pass_records) >= int(seq_max_passes) else "not recorded",
            )
            self.manifest.update_metrics({
                "runtime_profile": str(self.top_cfg.get("runtime_profile", "custom")),
                "passes_requested": int(seq_max_passes),
                "passes_started": len(benchmark_pass_records),
                "passes_accepted": len(accepted),
                "sequential_stop_reason": pass_stop_reason,
            })
            print(
                "[INFO] Sequential search summary: "
                f"profile={self.top_cfg.get('runtime_profile', 'custom')}, "
                f"requested={int(seq_max_passes)}, started={len(benchmark_pass_records)}, "
                f"accepted={len(accepted)}, stop={pass_stop_reason}."
            )

            # --------------------------------------------------------------------
            # FINAL MAIN-PHASE CELL POLISH
            # --------------------------------------------------------------------
            if (
                accepted
                and defer_main_cell_polish
                and bool(ds.get("polish_refine_cell", self.top_cfg.get("polish_refine_cell", True)))
                and bool(ds.get("polish_refine_main_cell", self.top_cfg.get("polish_refine_main_cell", True)))
            ):
                with bench.block("Stage 5: Final main-phase cell polish"):
                    if self.emitter:
                        self.emitter.emit(
                            "Final Polish",
                            "Final main-phase cell polish",
                            92,
                            metrics={"event": "final_main_polish_start", "accepted_phases": list(accepted)},
                        )
                    final_main_polished_gpx = str(Path(joint_dir) / "seq_final_main_polished.gpx")
                    final_main_trace = str(Path(joint_dir) / "seq_final_main_polished.polish_trace.json")
                    try:
                        fractions_final_main, rwp_final_main = joint_refine_polish(
                            base_gpx=kept_gpx,
                            out_gpx=final_main_polished_gpx,
                            main_phase_name=main_phase_name,
                            max_polish_cycles=int(ds.get("polish_cycles", self.top_cfg.get("polish_cycles", 10))),
                            refine_cell_for_all=True,
                            refine_background=bool(ds.get("polish_refine_background", self.top_cfg.get("polish_refine_background", True))),
                            target_phase_names=[main_phase_name],
                            polish_strategy=str(ds.get("polish_strategy", self.top_cfg.get("polish_strategy", "adaptive"))),
                            refine_main_cell=True,
                            refine_existing_cells=False,
                            escalate_on_failure=bool(ds.get("polish_escalate_on_failure", self.top_cfg.get("polish_escalate_on_failure", True))),
                            stabilization_cycles=int(ds.get("polish_stabilization_cycles", self.top_cfg.get("polish_stabilization_cycles", 1))),
                            cell_trial_cycles=int(ds.get("polish_cell_trial_cycles", self.top_cfg.get("polish_cell_trial_cycles", 1))),
                            final_polish_cycles=int(ds.get("polish_final_cycles", self.top_cfg.get("polish_final_cycles", 0))),
                            skip_fresh_lst_regen=bool(ds.get("polish_skip_fresh_lst_regen", self.top_cfg.get("polish_skip_fresh_lst_regen", True))),
                            trace_path=final_main_trace,
                        )
                        kept_gpx = final_main_polished_gpx
                        kept_rwp = float(rwp_final_main)
                        fractions_kept = fractions_final_main
                        self.manifest.add_artifact(kept_gpx)
                        if Path(final_main_trace).exists():
                            self.manifest.add_artifact(final_main_trace)
                        if self.emitter:
                            self.emitter.emit(
                                "Final Polish",
                                "Final main-phase cell polish complete",
                                95,
                                metrics={"event": "final_main_polish_done", "rwp": float(rwp_final_main)},
                            )
                    except Exception as exc:
                        print(f"[WARN] Final main-phase polish failed; keeping last accepted model: {type(exc).__name__}: {exc}")
                        if self.emitter:
                            self.emitter.emit(
                                "Final Polish",
                                "Final main-phase cell polish skipped after failure",
                                95,
                                level="WARN",
                                metrics={"event": "final_main_polish_failed", "error": str(exc)},
                            )

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
                        "is_main": is_main,
                    })

                rows_sorted = sorted(rows, key=lambda r: (not r["is_main"], -r["wf"]))  # main first, then by wf
                if len(rows_sorted) == 1:
                    main_row = rows_sorted[0]
                    try:
                        wf_val = float(main_row.get("wf", 0.0))
                    except Exception:
                        wf_val = 0.0
                    if wf_val == 0.0:
                        main_row["wf"] = 100.0
                        rows_sorted[0] = main_row

                total_imp = sum(r["wf"] for r in rows_sorted if not r["is_main"] and r["wf"] >= min_impurity_percent)

                hdr = f"Sequential Phase Quantification for {name}"
                cols = f"{'#':>2}  {'Phase ID':<18}  {'Compound Name':<30}  {'SG':>6}  {'Wt%':>7}  {'Notes':<20}"
                rule = "─" * len(cols)
                print(f"\n{hdr}")
                print(rule); print(cols); print(rule)
                for i, r in enumerate(rows_sorted, 1):
                    sg_str = r["sg"] if r["sg"] not in (None, "", "unknown") else "—"
                    # Include ID in the name column for clarity, but keep width in check
                    full_name = f"{r['pid']} ({r['display_name']})"
                    name_disp = full_name[:30]
                    note = "MAIN PHASE" if r["is_main"] else ("" if r["wf"] >= min_impurity_percent else f"<{min_impurity_percent}% (trace)")
                    print(f"{i:>2}  {r['pid']:<18}  {name_disp:<30}  {sg_str:>6}  {r['wf']:7.2f}  {note:<20}")
                print(rule)
                print(f"Final kept GPX: {kept_gpx}  (Rwp={rwp_final:.3f}%)")
                print(f"Total impurity (Wt% ≥ {min_impurity_percent}%): {total_imp:.2f}%")
                print(f"{'═' * 80}\n")

                csv_path = str(Path(results_dir) / "Summary_Fractions.csv")
                try:
                    import pandas as pd
                    csv_rows = [{
                        "phase_id": r["pid"],
                        "compound_name": r["display_name"],
                        "space_group": r["sg"],
                        "weight_fraction_pct": r["wf"],
                        "is_main": int(r["is_main"]),
                    } for r in rows_sorted]
                    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
                    print(f"[INFO] Final fractions CSV: {csv_path}")
                    self.manifest.add_artifact(csv_path)
                except Exception as e:
                    print(f"[WARN] Pandas CSV write failed ({csv_path}); using manual CSV fallback: {type(e).__name__}: {e}")

                self.emitter.emit("Final", "Pipeline completed successfully", 100)
                self.manifest.set_status("complete")
                with open(csv_path, "w") as f:
                    f.write("phase_id,compound_name,space_group,weight_fraction_pct,is_main\n")
                    for r in rows_sorted:
                        f.write(f"{r['pid']},{r['display_name']},{r['sg']},{r['wf']:.6f},{int(r['is_main'])}\n")
                print(f"[INFO] Phase quantification saved: {csv_path}")
                benchmark_final.update({
                    "status": "complete",
                    "final_rwp": float(rwp_final),
                    "accepted_phases": list(accepted),
                    "summary_csv": csv_path,
                    "total_impurity_pct": float(total_imp),
                })

            return True

        finally:
            try:
                report_extra = {
                    "dataset": name,
                    "work_dir": work_dir,
                    "mode": mode,
                    "instrument_parameter_file": instprm_path,
                    "data_file": data_path,
                    "main_cif": main_cif,
                    "allowed_elements": list(allowed_elements or []),
                    "stage4": dict(s4_cfg),
                    "max_passes": int(seq_max_passes),
                    "joint_top_k": int(joint_top_k),
                    "passes": benchmark_pass_records,
                    "final": benchmark_final,
                    "manifest_metrics": (self.manifest.data.get("metrics", {}) if self.manifest else {}),
                }
                report_json, report_csv = bench.write_report(
                    str(Path(tech_logs_dir) / "benchmark_report.json"),
                    report_extra,
                )
                print(f"[INFO] Benchmark report written: {report_json}")
                print(f"[INFO] Benchmark timing CSV written: {report_csv}")
                if self.manifest:
                    self.manifest.add_artifact(report_json)
                    self.manifest.add_artifact(report_csv)
            except Exception as exc:
                print(f"[WARN] Benchmark report write failed: {type(exc).__name__}: {exc}")
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
    instprm_path: str,
    template_gpx: Optional[str] = None,
    engine: str = "surrogate",
    cell_refine_min_r: float = 0.5,
    export_refined_cif: bool = False,
) -> Tuple[float, str, str, Dict[str, Any]]:
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
        pearson_diag = compute_gsas_pearson_for_cif(
            data_path="",
            instprm_path=instprm_path,
            fmthint=None,
            cif_path=cand_cif,
            work_dir=work_dir,
            limits=None,
            exclude_regions=None,
            tmp_tag=f"sel_{pid}",
            x_override=x_native,
            y_override=residual_native,
            template_gpx=template_gpx,
            cell_refine_min_r=cell_refine_min_r,
            export_refined_cif=export_refined_cif,
            return_diagnostics=True,
        )
        if isinstance(pearson_diag, dict):
            r_val = float(pearson_diag.get("pearson", 0.0))
        else:
            r_val = float(pearson_diag)
            pearson_diag = {"pearson": r_val, "timings": {}}

        if export_refined_cif:
            refined_cif_path = pearson_diag.get("refined_cif_path") if isinstance(pearson_diag, dict) else None
            if not refined_cif_path:
                stem = Path(cand_cif).stem
                refined_cif_path = str(Path(cand_cif).with_name(f"{stem}_refined.cif"))
            final_cif = refined_cif_path if refined_cif_path and Path(refined_cif_path).exists() else cand_cif
        else:
            # Scoring workers deliberately do not emit refined CIFs. The caller
            # exports only the post-pruning survivors before joint compare.
            pearson_diag["source_cif_path"] = cand_cif
            final_cif = ""

        timings = pearson_diag.get("timings", {}) if isinstance(pearson_diag, dict) else {}
        print(
            f"[INFO] Stage-4 Pearson (GSAS - {label}): {pid} -> "
            f"r={float(r_val):.4f}, r1={float(pearson_diag.get('r_scale', float('nan'))):.4f}, "
            f"cell={bool(pearson_diag.get('cell_refined', False))}, "
            f"total={float(timings.get('total_s', 0.0)):.3f}s "
            f"(scale={float(timings.get('pass1_scale_s', 0.0)):.3f}s, "
            f"cell={float(timings.get('pass2_cell_s', 0.0)):.3f}s, "
            f"export={float(timings.get('export_cif_s', 0.0)):.3f}s; "
            f"exported={bool(export_refined_cif)}; "
            f"using {Path(final_cif).name if final_cif else Path(cand_cif).name})"
        )

        return float(r_val), label, final_cif, pearson_diag

    except Exception as e:
        print(f"[ERROR] Pearson computation failed for {pid}: {e}")
        traceback.print_exc()
        return 0.0, "orig", cand_cif, {"pearson": 0.0, "error": str(e), "timings": {}}

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
            if pipe.manifest:
                pipe.manifest.set_status("interrupted")
            if pipe.emitter:
                pipe.emitter.emit("Interrupted", "Pipeline interrupted by user", 100, level="WARN")
            return False
        except Exception as e:
            print(f"[FATAL] Dataset '{args.dataset}' failed: {e}")
            traceback.print_exc()
            if pipe.manifest:
                pipe.manifest.set_status("failed")
            if pipe.emitter:
                pipe.emitter.emit("Error", str(e), 100, level="ERROR")
            return False
    else:
        for ds in datasets:
            name = ds.get("name", "<unnamed>")
            try:
                ok = pipe.run_dataset(ds)
                success &= ok
            except KeyboardInterrupt:
                print("\n[INFO] Interrupted by user")
                if pipe.manifest:
                    pipe.manifest.set_status("interrupted")
                if pipe.emitter:
                    pipe.emitter.emit("Interrupted", "Pipeline interrupted by user", 100, level="WARN")
                return False
            except Exception as e:
                print(f"[ERROR] Dataset '{name}' failed: {e}")
                traceback.print_exc()
                if pipe.manifest:
                    pipe.manifest.set_status("failed")
                    pipe.emitter.emit("Error", str(e), 100, level="ERROR")
                success = False

    out_dir = (
        _expand(cfg.get("work_dir"))
        or _expand(cfg.get("work_root"))
        or _expand(cfg.get("WORK_ROOT"))
        or str(Path(args.config).resolve().parent)
    )
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
