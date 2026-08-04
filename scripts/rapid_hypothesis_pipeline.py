#!/usr/bin/env python3
"""Live rapid-hypothesis runner for RADAR-PD.

This is the first app-wired rapid path. It reads the normal RADAR-PD
``pipeline_config.yaml`` and writes rapid-mode artifacts into the selected run
folder so the Streamlit Rapid Results panel can inspect them.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import nnls

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _default_gsas_root() -> Path:
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return Path(sys.prefix) / "lib" / pyver / "site-packages" / "radar_pd_gsasii_runtime" / "gsasii"


os.environ.setdefault("RADAR_PD_GSASII_ROOT", str(_default_gsas_root()))

from aniso_db_loader import CatalogPaths, DBLoader
from auto_background_points import coerce_auto_background_params, estimate_background
from gsas_core_infrastructure import GSASProjectManager
from gsas_main_phase_refiner import (
    GSASDataExtractor,
    GSASMainPhaseRefiner,
    HISTOGRAM_HOLD_VARS,
    _add_histogram_hold_constraint,
    apply_safe_limits,
    clone_gpx,
    ensure_usable_range,
    joint_refine_polish,
    normalize_excluded_regions,
    normalize_limits,
    parse_gsas_lst,
    plot_gpx_fit_with_ticks,
    read_abs_limits_or_bounds,
    set_excluded,
    set_limits,
    set_phase_cell_refine,
)
from ratio_filter import _load_profiles64_metadata, _residual_hist_from_continuous_parts
from magnetic_precheck import run_magnetic_precheck
from xray_doublet import (
    apply_doublet_to_peaks,
    apply_doublet_to_profiles,
    describe_doublet,
    resolve_xray_doublet_spec,
)
from main_phase_anchor import (
    assess_main_fit_for_prenudge,
    main_anchor_reliability_from_audit,
    main_anchor_reliability_from_fit_audit,
    main_prenudge_cfg,
    main_shadow_filter_decision,
    main_phase_guard_cfg,
    main_phase_shadow_cfg,
    main_shadow_peaks_from_arrays,
    run_main_phase_prenudge_if_needed,
    run_main_phase_cleanup_if_enabled,
)

try:
    from reference_phase_masks import build_reference_phase_exclusions, merge_reference_phase_exclusion_config
except Exception:
    build_reference_phase_exclusions = None
    merge_reference_phase_exclusion_config = None


def _expand_cfg_path(value: Any, cfg: dict[str, Any]) -> Any:
    """Expand RADAR-PD config placeholders in a filesystem path value."""
    if value in (None, ""):
        return value
    text = os.path.expanduser(os.path.expandvars(str(value)))
    replacements = {
        "CONFIG_DIR": cfg.get("CONFIG_DIR"),
        "PROJECT_ROOT": cfg.get("PROJECT_ROOT"),
        "DATA_ROOT": cfg.get("DATA_ROOT"),
        "WORK_ROOT": cfg.get("WORK_ROOT"),
    }
    for key, replacement in replacements.items():
        if replacement:
            text = text.replace("${" + key + "}", str(replacement))
    path = Path(text)
    if not path.is_absolute():
        base = Path(str(cfg.get("CONFIG_DIR") or PROJECT_ROOT))
        path = base / path
    return str(path.resolve())


def _database_config_for_dataset(cfg: dict[str, Any], dataset: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Resolve the database block used by rapid mode.

    Rapid mode is invoked from both the web app and the API. The shared
    RADAR-PD config normally stores database paths as db_neutron/db_xray, while
    older rapid-only configs used a top-level db block. Support both so the API
    can run the same exported configs as the hosted UI.
    """
    dataset = dataset or {}
    db_cfg = dict(cfg.get("db") or {})
    ds_db = dataset.get("db") or {}
    if isinstance(ds_db, dict):
        db_cfg.update(ds_db)

    for key in ("catalog_csv", "original_json", "profiles_dir", "stable_csv", "cif_map_json"):
        if dataset.get(key):
            db_cfg[key] = dataset[key]

    if not db_cfg:
        db_source = str(dataset.get("db_source") or cfg.get("db_source") or "xray").strip().lower()
        source_key = "db_neutron" if db_source == "neutron" else "db_xray"
        db_cfg = dict(cfg.get(source_key) or {})

    for key in ("catalog_csv", "original_json", "profiles_dir", "stable_csv", "cif_map_json"):
        if key in db_cfg and db_cfg[key]:
            db_cfg[key] = _expand_cfg_path(db_cfg[key], cfg)

    missing = [key for key in ("catalog_csv", "profiles_dir") if not db_cfg.get(key)]
    if missing:
        raise KeyError(
            "Rapid mode database configuration is missing "
            + ", ".join(missing)
            + ". Provide db: or db_neutron/db_xray with catalog_csv and profiles_dir."
        )
    return db_cfg


@dataclass
class CandidateView:
    ids: np.ndarray
    formulas: np.ndarray
    formula_keys: np.ndarray
    space_groups: np.ndarray
    base_profiles: np.ndarray
    variant_profiles: np.ndarray
    variant_unit: np.ndarray
    variant_candidate: np.ndarray
    variant_shift: np.ndarray


@dataclass
class BeamState:
    formulas: Tuple[str, ...]
    variants: Tuple[int, ...]
    coefs: Tuple[float, ...]
    sse: float
    residual: np.ndarray
    fit: np.ndarray


@dataclass
class GSASStageResult:
    label: str
    ok: bool
    rwp: float
    stdout: str
    reason: str = ""


def _log(message: str) -> None:
    print(message, flush=True)


def _safe_name(value: Any, max_len: int = 64) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return (text or "phase")[:max_len]


def _split_pipe(value: Any) -> Tuple[str, ...]:
    return tuple(part.strip() for part in str(value or "").split("|") if part.strip())


def _unit_rows(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-8)


def _unit_vec(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    return vec / max(float(np.linalg.norm(vec)), 1e-8)


def _formula_key(formula: Any) -> str:
    text = str(formula or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return "unknown"
    parts = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", text)
    if not parts:
        return text
    vals: list[tuple[str, float]] = []
    for elem, amount in parts:
        try:
            vals.append((elem, float(amount) if amount else 1.0))
        except Exception:
            return text
    min_val = min((v for _e, v in vals if v > 0), default=1.0)
    scaled = [(e, v / min_val) for e, v in vals]
    rendered = []
    for elem, amount in scaled:
        rounded = round(amount)
        if abs(amount - rounded) < 0.05:
            amount = float(max(1, rounded))
        suffix = "" if abs(amount - 1.0) < 1e-6 else (str(int(amount)) if amount.is_integer() else f"{amount:.3g}")
        rendered.append(f"{elem}{suffix}")
    return "".join(rendered) or text


def _formula_amount_map(formula: Any) -> dict[str, float]:
    text = str(formula or "").strip()
    parts = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", text)
    amounts: dict[str, float] = {}
    for elem, amount in parts:
        try:
            value = float(amount) if amount else 1.0
        except Exception:
            continue
        if value > 0:
            amounts[elem] = amounts.get(elem, 0.0) + float(value)
    return amounts


def _pure_element_formula(formula: Any) -> str:
    amounts = _formula_amount_map(formula)
    if len(amounts) != 1:
        return ""
    return next(iter(amounts))


def _near_elemental_dominant(formula: Any, *, trace_fraction_max: float = 0.02) -> str:
    amounts = _formula_amount_map(formula)
    if len(amounts) <= 1:
        return ""
    total = float(sum(amounts.values()))
    if total <= 0:
        return ""
    elem, amount = max(amounts.items(), key=lambda item: item[1])
    if float(amount) / total >= (1.0 - float(trace_fraction_max)):
        return elem
    return ""


def _is_missing_formula(value: Any, pid: str | None = None) -> bool:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null", "unknown"}:
        return True
    if pid is not None and text == str(pid):
        return True
    return bool(re.fullmatch(r"(?:mp|mvc|cod)-\d+", text, flags=re.IGNORECASE))


def _render_formula_amount(amount: float) -> str:
    rounded = round(float(amount))
    if abs(float(amount) - rounded) < 0.03:
        amount = float(max(1, rounded))
    if abs(amount - 1.0) < 1e-8:
        return ""
    return str(int(amount)) if float(amount).is_integer() else f"{amount:.4g}"


def _formula_from_composition(comp: Any) -> str | None:
    if not isinstance(comp, dict) or not comp:
        return None
    vals: list[tuple[str, float]] = []
    for elem, amount in comp.items():
        try:
            val = float(amount)
        except Exception:
            continue
        if val > 0:
            vals.append((str(elem), val))
    if not vals:
        return None
    scale = min(val for _elem, val in vals if val > 0)
    parts = [f"{elem}{_render_formula_amount(val / scale)}" for elem, val in vals]
    return "".join(parts) or None


def _formula_from_cif_content(cif_text: Any) -> str | None:
    text = str(cif_text or "")
    if not text:
        return None
    for tag in ("_chemical_formula_structural", "_chemical_formula_sum"):
        match = re.search(rf"^{re.escape(tag)}\s+(.+?)\s*$", text, flags=re.MULTILINE)
        if not match:
            continue
        raw = match.group(1).strip().strip("'\"")
        if not raw:
            continue
        if tag == "_chemical_formula_structural" and not _is_missing_formula(raw):
            return raw.replace(" ", "")
        parts = re.findall(r"([A-Z][a-z]?)\s*([0-9]*\.?[0-9]*)", raw)
        if parts:
            comp = {elem: float(amount) if amount else 1.0 for elem, amount in parts}
            formula = _formula_from_composition(comp)
            if formula:
                return formula
    return None


def _formula_from_metadata_record(record: Any) -> str | None:
    if not isinstance(record, dict):
        return None
    for key in ("formula_pretty", "pretty_formula", "formula", "pretty_name"):
        val = str(record.get(key) or "").strip()
        if not _is_missing_formula(val):
            return val
    formula = _formula_from_cif_content(record.get("cif_content"))
    if formula:
        return formula
    return _formula_from_composition(record.get("composition"))


def _load_formula_metadata(cfg: dict) -> dict[str, Any]:
    metadata_path = ((cfg.get("db") or {}).get("original_json") or "").strip()
    if not metadata_path:
        return {}
    path = Path(metadata_path)
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _catalog_formula(catalog: pd.DataFrame, pid: str, metadata: dict[str, Any] | None = None) -> str:
    row = catalog[catalog["id"].astype(str).eq(str(pid))]
    if not row.empty:
        rec = row.iloc[0]
        for col in ("pretty_formula", "formula_pretty", "formula", "pretty_name"):
            if col in rec.index:
                val = str(rec.get(col) or "").strip()
                if not _is_missing_formula(val, pid):
                    return val
    if metadata:
        formula = _formula_from_metadata_record(metadata.get(str(pid)))
        if formula:
                return formula
    return str(pid)


def _space_group_from_cif_text(cif_text: str) -> Optional[int]:
    for tag in (
        "_space_group_IT_number",
        "_symmetry_Int_Tables_number",
        "_space_group.it_number",
    ):
        match = re.search(rf"^{re.escape(tag)}\s+([0-9]+)", cif_text or "", flags=re.MULTILINE | re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return None
    return None


def _space_group_symbol_from_cif_text(cif_text: str) -> Optional[str]:
    for tag in (
        "_space_group_name_H-M_alt",
        "_symmetry_space_group_name_H-M",
        "_space_group.name_H-M_alt",
    ):
        match = re.search(rf"^{re.escape(tag)}\s+(.+?)\s*$", cif_text or "", flags=re.MULTILINE | re.IGNORECASE)
        if not match:
            continue
        value = match.group(1).strip().strip("'\"")
        if value and value not in {"?", "."}:
            return value
    return None


def _main_phase_display_from_cif(cif_path: Any) -> dict[str, Any]:
    path = Path(str(cif_path or ""))
    info: dict[str, Any] = {
        "label": "Main phase",
        "formula": "",
        "formula_key": "",
        "space_group": None,
        "space_group_symbol": "",
        "path": str(path) if str(cif_path or "").strip() else "",
    }
    if not cif_path or not path.exists():
        return info

    formula: Optional[str] = None
    sgnum: Optional[int] = None
    sgsym: Optional[str] = None
    try:
        from pymatgen.io.cif import CifParser
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        structure = CifParser(str(path), occupancy_tolerance=2.0).parse_structures(primitive=False)[0]
        try:
            formula = str(structure.composition.reduced_formula)
        except Exception:
            formula = None
        try:
            sga = SpacegroupAnalyzer(structure, symprec=0.1)
            sgnum = int(sga.get_space_group_number())
            sgsym = str(sga.get_space_group_symbol())
        except Exception:
            pass
    except Exception:
        pass

    try:
        cif_text = path.read_text(errors="ignore")
    except Exception:
        cif_text = ""
    if not formula:
        formula = _formula_from_cif_content(cif_text)
    if sgnum is None:
        sgnum = _space_group_from_cif_text(cif_text)
    if not sgsym:
        sgsym = _space_group_symbol_from_cif_text(cif_text)

    if not formula:
        formula = path.stem
    label = f"{formula} (SG {sgnum})" if sgnum else str(formula)
    info.update(
        {
            "label": label,
            "formula": str(formula),
            "formula_key": _formula_key(formula),
            "space_group": sgnum,
            "space_group_symbol": str(sgsym or ""),
        }
    )
    return info


def _main_phase_display_from_dataset(dataset: dict) -> dict[str, Any]:
    info = _main_phase_display_from_cif(dataset.get("main_cif"))
    for key in ("main_cif_prenudged_path", "main_cif_cleanup_path"):
        if dataset.get(key):
            updated = _main_phase_display_from_cif(dataset.get(key))
            if updated.get("label") != "Main phase":
                return updated
    return info


def _main_phase_signature(dataset: dict) -> Optional[dict[str, Any]]:
    path_raw = dataset.get("main_cif")
    if not path_raw:
        return None
    path = Path(str(path_raw))
    if not path.exists():
        return None
    formula: Optional[str] = None
    sgnum: Optional[int] = None
    try:
        from pymatgen.io.cif import CifParser
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        structure = CifParser(str(path), occupancy_tolerance=2.0).parse_structures(primitive=False)[0]
        try:
            formula = str(structure.composition.reduced_formula)
        except Exception:
            formula = None
        try:
            sgnum = int(SpacegroupAnalyzer(structure, symprec=0.1).get_space_group_number())
        except Exception:
            sgnum = None
    except Exception:
        pass
    try:
        cif_text = path.read_text(errors="ignore")
    except Exception:
        cif_text = ""
    if not formula:
        formula = _formula_from_cif_content(cif_text)
    if sgnum is None:
        sgnum = _space_group_from_cif_text(cif_text)
    if not formula:
        return None
    return {
        "formula": formula,
        "formula_key": _formula_key(formula),
        "space_group": sgnum,
        "path": str(path),
    }


def _is_main_phase_duplicate(candidate_formula: str, candidate_sg: int, main_sig: Optional[dict[str, Any]]) -> bool:
    if not main_sig:
        return False
    main_key = str(main_sig.get("formula_key") or "").strip()
    if not main_key or main_key == "unknown":
        return False
    if _formula_key(candidate_formula) != main_key:
        return False
    main_sg = main_sig.get("space_group")
    try:
        main_sg_i = int(main_sg) if main_sg is not None else None
    except Exception:
        main_sg_i = None
    try:
        cand_sg_i = int(candidate_sg)
    except Exception:
        cand_sg_i = 0
    if main_sg_i is not None and cand_sg_i > 0:
        return cand_sg_i == main_sg_i
    return True


def _is_xray_run(cfg: dict) -> bool:
    stage4 = cfg.get("stage4") or {}
    radiation = str(stage4.get("radiation") or cfg.get("radiation") or "").strip().lower()
    return radiation in {"xray", "x-ray", "pxrd"}


def _light_calibration_config(cfg: dict, dataset: dict) -> dict[str, Any]:
    merged = dict(cfg.get("light_calibration") or {})
    if isinstance(dataset.get("light_calibration"), dict):
        merged.update(dataset.get("light_calibration") or {})
    return merged


def _run_rapid_light_calibration(
    refiner: GSASMainPhaseRefiner,
    cfg: dict,
    dataset: dict,
    work_dir: Path,
    mode: str,
) -> Optional[str]:
    calib_cfg = _light_calibration_config(cfg, dataset)
    if not (
        calib_cfg.get("enabled")
        and dataset.get("main_cif")
        and str(mode).lower() == "cw"
        and _is_xray_run(cfg)
    ):
        return None

    bg_cfg = cfg.get("background") or {}
    export_path = work_dir / "rapid_light_calibrated.instprm"
    report_path = work_dir / "rapid_light_calibration.json"
    _log("[rapid] running PXRD light calibration before residual extraction")
    report: dict[str, Any] = {
        "requested": True,
        "status": "started",
        "exported_instprm": "",
        "rwp_before": None,
        "rwp_after": None,
        "refined_terms": [],
        "note": "",
    }
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = refiner.run_light_instrument_calibration(
                background_config=bg_cfg,
                bg_type=bg_cfg.get("type"),
                bg_terms=int(bg_cfg["terms"]) if bg_cfg.get("terms") is not None else None,
                bg_coeffs=bg_cfg.get("coeffs"),
                zero_cycles=int(calib_cfg.get("zero_cycles", 1)),
                profile_cycles=int(calib_cfg.get("profile_cycles", 2)),
                profile_terms=calib_cfg.get("terms"),
                export_path=str(export_path),
            )
        report.update(
            {
                "rwp_before": result.rwp_before,
                "rwp_after": result.rwp_after,
                "refined_terms": list(result.refined_terms or ()),
                "exported_instprm": str(result.exported_instprm or ""),
            }
        )
        if result.skipped:
            report["status"] = "skipped"
            report["note"] = result.error_message or "calibration skipped"
            _log(f"[rapid] PXRD light calibration skipped: {report['note']}")
            return None
        if not result.success:
            report["status"] = "failed"
            report["note"] = result.error_message or "calibration failed"
            try:
                refiner.load_instrument_profile(str(dataset.get("instprm_path") or ""))
            except Exception as restore_err:
                report["note"] = f"{report['note']}; restore failed: {restore_err}"
            _log(f"[rapid][WARN] PXRD light calibration failed: {report['note']}")
            return None
        before = result.rwp_before
        after = result.rwp_after
        accept_worsen = float(calib_cfg.get("accept_rwp_worsen", 0.15))
        accepted = (
            result.exported_instprm
            and before is not None
            and after is not None
            and math.isfinite(float(before))
            and math.isfinite(float(after))
            and float(after) <= float(before) + accept_worsen
        )
        if accepted:
            report["status"] = "adopted"
            report["note"] = f"Rwp {float(before):.3f}% -> {float(after):.3f}%"
            dataset["calibrated_instprm_path"] = str(result.exported_instprm)
            _log(f"[rapid] adopted PXRD light calibration: {report['note']}")
            return str(result.exported_instprm)
        report["status"] = "rejected"
        report["note"] = (
            f"not adopted: Rwp {before} -> {after}"
            if before is not None and after is not None
            else "not adopted"
        )
        try:
            refiner.load_instrument_profile(str(dataset.get("instprm_path") or ""))
        except Exception as restore_err:
            report["note"] = f"{report['note']}; restore failed: {restore_err}"
        _log(f"[rapid][WARN] PXRD light calibration rejected: {report['note']}")
        return None
    except Exception as exc:
        report["status"] = "error"
        report["note"] = str(exc)
        try:
            refiner.load_instrument_profile(str(dataset.get("instprm_path") or ""))
        except Exception:
            pass
        _log(f"[rapid][WARN] PXRD light calibration error: {exc}")
        return None
    finally:
        try:
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except Exception:
            pass


def _effective_limits_and_exclusions(
    cfg: dict,
    dataset: dict,
    *,
    mode: str,
    instprm_path: str,
    hist: Any,
    audit_dir: Optional[Path] = None,
) -> tuple[Optional[tuple[float, float]], list[list[float]]]:
    """Return and apply the same native-axis limits/exclusions used by full RADAR-PD."""
    limits = dataset.get("limits")
    manual_excludes = list(dataset.get("exclude_regions", []) or [])
    ref_cfg = cfg.get("reference_phase_exclusions", {}) or {}
    if merge_reference_phase_exclusion_config is not None:
        try:
            ref_cfg = merge_reference_phase_exclusion_config(
                cfg.get("reference_phase_exclusions", {}),
                dataset.get("reference_phase_exclusions", {}),
            )
        except Exception:
            ref_cfg = cfg.get("reference_phase_exclusions", {}) or {}

    generated: list[list[float]] = []
    ref_report: dict[str, Any] = {"enabled": bool(ref_cfg.get("enabled"))}
    if ref_cfg.get("enabled") and build_reference_phase_exclusions is not None:
        ref_report = build_reference_phase_exclusions(
            ref_cfg,
            instprm_path=instprm_path,
            mode=mode,
            limits=limits,
        )
        generated = list(ref_report.get("ranges", []) or [])
    combined = manual_excludes + generated

    abs_lo, abs_hi = read_abs_limits_or_bounds(hist)
    active_limits = normalize_limits(limits, abs_lo, abs_hi)
    if active_limits:
        lo, hi = active_limits
    else:
        lo, hi = abs_lo, abs_hi
    normalized_excludes = normalize_excluded_regions(combined, lo, hi)
    if active_limits:
        ensure_usable_range(float(active_limits[0]), float(active_limits[1]), normalized_excludes)
        set_limits(hist, float(active_limits[0]), float(active_limits[1]))
    if normalized_excludes:
        set_excluded(hist, normalized_excludes)

    if audit_dir is not None and (manual_excludes or generated or active_limits):
        audit_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "limits": list(active_limits) if active_limits else None,
            "manual_ranges": manual_excludes,
            "generated_reference_ranges": generated,
            "combined_ranges": normalized_excludes,
            "reference_phase_exclusion_report": ref_report,
        }
        (audit_dir / "rapid_limits_and_exclusions.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return active_limits, normalized_excludes


def _background_rows(n_bins: int) -> list[np.ndarray]:
    x = np.linspace(-1.0, 1.0, int(n_bins), dtype=np.float32)
    return [
        np.ones_like(x),
        x,
        2.0 * x * x - 1.0,
    ]


def _fit_rows(rows: Sequence[np.ndarray], y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    y = np.asarray(y, dtype=np.float32)
    if not rows:
        fit = np.zeros_like(y)
        return np.zeros(0, dtype=np.float32), fit, float(np.sum((y - fit) ** 2))
    active = np.vstack(rows).astype(np.float32)
    cols = active.T
    norms = np.maximum(np.linalg.norm(cols, axis=0, keepdims=True), 1e-8)
    cols = cols / norms
    coef, _ = nnls(cols, y)
    fit = (cols @ coef).astype(np.float32)
    return coef.astype(np.float32), fit, float(np.sum((y - fit) ** 2))


def _shift_profiles(profiles: np.ndarray, shifts: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    profiles = np.asarray(profiles, dtype=np.float32)
    x = np.arange(profiles.shape[1], dtype=np.float32)
    rows: list[np.ndarray] = []
    cand_idx: list[np.ndarray] = []
    shift_vals: list[np.ndarray] = []
    for shift in shifts:
        shifted = np.empty_like(profiles)
        src_x = x - float(shift)
        for i, row in enumerate(profiles):
            shifted[i] = np.interp(x, src_x, row, left=0.0, right=0.0)
        rows.append(shifted)
        cand_idx.append(np.arange(profiles.shape[0], dtype=np.int32))
        shift_vals.append(np.full(profiles.shape[0], float(shift), dtype=np.float32))
    return np.vstack(rows).astype(np.float32), np.concatenate(cand_idx), np.concatenate(shift_vals)


def _make_view(
    profiles: np.ndarray,
    ids: Sequence[Any],
    formulas: Sequence[Any],
    sgs: Sequence[int],
    shifts: Sequence[float],
) -> CandidateView:
    variant_profiles, variant_candidate, variant_shift = _shift_profiles(profiles, shifts)
    formula_keys = np.asarray([_formula_key(f) for f in formulas], dtype=object)
    return CandidateView(
        ids=np.asarray(ids, dtype=object),
        formulas=np.asarray(formulas, dtype=object),
        formula_keys=formula_keys,
        space_groups=np.asarray(sgs, dtype=np.int32),
        base_profiles=np.asarray(profiles, dtype=np.float32),
        variant_profiles=variant_profiles,
        variant_unit=_unit_rows(variant_profiles),
        variant_candidate=variant_candidate,
        variant_shift=variant_shift,
    )


def _best_variants_for_residual(
    view: CandidateView,
    residual: np.ndarray,
    used_formulas: set[str],
    *,
    top_formulas: int,
    shift_penalty: float,
) -> pd.DataFrame:
    r = np.maximum(np.asarray(residual, dtype=np.float32), 0.0)
    scores = view.variant_unit @ _unit_vec(r)
    if shift_penalty:
        scores = scores - float(shift_penalty) * np.abs(view.variant_shift)
    order = np.argsort(-scores)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for vi in order:
        ci = int(view.variant_candidate[vi])
        key = str(view.formula_keys[ci])
        if key in used_formulas or key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "formula_key": key,
                "phase_id": str(view.ids[ci]),
                "formula": str(view.formulas[ci]),
                "space_group": int(view.space_groups[ci]),
                "variant_index": int(vi),
                "candidate_index": ci,
                "shift_bins": float(view.variant_shift[vi]),
                "residual_score": float(scores[vi]),
            }
        )
        if len(rows) >= int(top_formulas):
            break
    out = pd.DataFrame(rows)
    if not out.empty:
        out.insert(0, "residual_rank", np.arange(1, len(out) + 1))
    return out


def _beam_search(
    y64: np.ndarray,
    view: CandidateView,
    *,
    depth: int,
    beam_width: int,
    branch_top: int,
    shift_penalty: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = np.maximum(np.asarray(y64, dtype=np.float32), 0.0)
    background = _background_rows(y.size)
    _coef0, fit0, sse0 = _fit_rows(background, y)
    beams: list[BeamState] = [BeamState(tuple(), tuple(), tuple(), sse0, np.maximum(y - fit0, 0.0), fit0)]
    all_rows: list[dict[str, Any]] = []
    rank_rows: list[pd.DataFrame] = []
    for d in range(1, int(depth) + 1):
        proposals: list[BeamState] = []
        for parent_rank, state in enumerate(beams, start=1):
            ranked = _best_variants_for_residual(
                view,
                state.residual,
                set(state.formulas),
                top_formulas=branch_top,
                shift_penalty=shift_penalty,
            )
            if not ranked.empty:
                rr = ranked.copy()
                rr.insert(0, "depth", d)
                rr.insert(1, "parent_rank", parent_rank)
                rr.insert(2, "parent_formulas", "|".join(state.formulas))
                rank_rows.append(rr)
            for rec in ranked.to_dict("records"):
                vi = int(rec["variant_index"])
                new_variants = tuple([*state.variants, vi])
                new_formulas = tuple([*state.formulas, str(rec["formula_key"])])
                coefs, fit, sse = _fit_rows([*background, *[view.variant_profiles[i] for i in new_variants]], y)
                phase_coefs = coefs[-len(new_variants):] if new_variants else np.zeros(0, dtype=np.float32)
                proposals.append(
                    BeamState(
                        formulas=new_formulas,
                        variants=new_variants,
                        coefs=tuple(float(x) for x in phase_coefs),
                        sse=sse,
                        residual=np.maximum(y - fit, 0.0),
                        fit=fit,
                    )
                )
        if not proposals:
            break
        proposals.sort(key=lambda item: item.sse)
        beams = []
        seen_sets: set[Tuple[str, ...]] = set()
        for state in proposals:
            key = tuple(sorted(state.formulas))
            if key in seen_sets:
                continue
            seen_sets.add(key)
            beams.append(state)
            if len(beams) >= int(beam_width):
                break
        for rank, state in enumerate(beams, start=1):
            cis = [int(view.variant_candidate[i]) for i in state.variants]
            all_rows.append(
                {
                    "depth": d,
                    "rank": rank,
                    "rank64": rank,
                    "formula_keys": "|".join(state.formulas),
                    "phase_ids": "|".join(str(view.ids[i]) for i in cis),
                    "formulas": "|".join(str(view.formulas[i]) for i in cis),
                    "space_groups": "|".join(str(int(view.space_groups[i])) for i in cis),
                    "shifts": "|".join(f"{float(view.variant_shift[i]):.2f}" for i in state.variants),
                    "coefs": "|".join(f"{c:.6g}" for c in state.coefs),
                    "sse": float(state.sse),
                    "sse64": float(state.sse),
                    "relative_gain": float((sse0 - state.sse) / max(sse0, 1e-8)),
                    "gain64": float((sse0 - state.sse) / max(sse0, 1e-8)),
                    "residual_sum": float(state.residual.sum()),
                }
            )
    states = pd.DataFrame(all_rows)
    residual_rank = pd.concat(rank_rows, ignore_index=True) if rank_rows else pd.DataFrame()
    return states, residual_rank


def _run_beam64_search(
    *,
    y64: np.ndarray,
    profiles: np.ndarray,
    ids: np.ndarray,
    formulas: np.ndarray,
    sgs: np.ndarray,
    shifts: Sequence[float],
    rapid_cfg: dict,
    depth: int,
    excluded_phase_ids: Optional[set[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    excluded = {str(pid) for pid in (excluded_phase_ids or set()) if str(pid)}
    if excluded:
        keep = np.asarray([str(pid) not in excluded for pid in ids], dtype=bool)
        ids_use = np.asarray(ids, dtype=object)[keep]
        formulas_use = np.asarray(formulas, dtype=object)[keep]
        sgs_use = np.asarray(sgs, dtype=np.int32)[keep]
        profiles_use = np.asarray(profiles, dtype=np.float32)[keep]
    else:
        ids_use = np.asarray(ids, dtype=object)
        formulas_use = np.asarray(formulas, dtype=object)
        sgs_use = np.asarray(sgs, dtype=np.int32)
        profiles_use = np.asarray(profiles, dtype=np.float32)

    if profiles_use.size == 0 or len(ids_use) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    view = _make_view(profiles_use, ids_use, formulas_use, sgs_use, sorted(set(float(s) for s in shifts)))
    states, residual_rank = _beam_search(
        y64,
        view,
        depth=depth,
        beam_width=int(rapid_cfg.get("beam_width", 40)),
        branch_top=int(rapid_cfg.get("branch_top", 160)),
        shift_penalty=float(rapid_cfg.get("shift_penalty", 0.01)),
    )
    if states.empty:
        return pd.DataFrame(), states, residual_rank
    final_depth = int(states["depth"].max())
    beam64 = (
        states[states["depth"].eq(final_depth)]
        .sort_values("rank")
        .head(int(rapid_cfg.get("beam_output_limit", 100)))
        .copy()
        .reset_index(drop=True)
    )
    return beam64, states, residual_rank


def _near_elemental_duplicate_records(
    beam_rows: pd.DataFrame,
    *,
    trace_fraction_max: float = 0.02,
) -> tuple[set[str], list[dict[str, Any]]]:
    banned: set[str] = set()
    records: list[dict[str, Any]] = []
    if beam_rows is None or beam_rows.empty:
        return banned, records

    for row in beam_rows.itertuples(index=False):
        pids = _split_pipe(getattr(row, "phase_ids", ""))
        formulas = _split_pipe(getattr(row, "formula_keys", "")) or _split_pipe(getattr(row, "formulas", ""))
        raw_formulas = _split_pipe(getattr(row, "formulas", ""))
        pure_by_element: dict[str, int] = {}
        for idx, formula in enumerate(formulas):
            elem = _pure_element_formula(formula)
            if elem:
                pure_by_element.setdefault(elem, idx)
        if not pure_by_element:
            continue
        for idx, formula in enumerate(formulas):
            dominant = _near_elemental_dominant(formula, trace_fraction_max=trace_fraction_max)
            if not dominant or dominant not in pure_by_element:
                continue
            pid = pids[idx] if idx < len(pids) else ""
            if not pid:
                continue
            banned.add(str(pid))
            pure_idx = pure_by_element[dominant]
            records.append(
                {
                    "rank64": getattr(row, "rank64", getattr(row, "rank", "")),
                    "phase_id": str(pid),
                    "formula_key": str(formula),
                    "formula": raw_formulas[idx] if idx < len(raw_formulas) else str(formula),
                    "dominant_element": dominant,
                    "pure_phase_id": pids[pure_idx] if pure_idx < len(pids) else "",
                    "pure_formula": formulas[pure_idx] if pure_idx < len(formulas) else dominant,
                    "reason": "near_elemental_duplicate_of_pure_element_in_same_hypothesis",
                }
            )
    return banned, records


def _apply_near_elemental_duplicate_filter(
    *,
    beam64: pd.DataFrame,
    states: pd.DataFrame,
    residual_rank: pd.DataFrame,
    y64: np.ndarray,
    profiles: np.ndarray,
    ids: np.ndarray,
    formulas: np.ndarray,
    sgs: np.ndarray,
    shifts: Sequence[float],
    rapid_cfg: dict,
    depth: int,
    out_dir: Path,
    excluded_phase_ids: Optional[set[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[str], dict[str, Any]]:
    cfg = rapid_cfg.get("near_elemental_duplicate_filter") or {}
    enabled = bool(cfg.get("enabled", True))
    trace_fraction_max = float(cfg.get("trace_fraction_max", 0.02))
    audit: dict[str, Any] = {
        "enabled": enabled,
        "trace_fraction_max": trace_fraction_max,
        "banned_phase_ids": [],
        "effective_banned_phase_ids": [],
        "fail_open": False,
        "records": [],
    }
    if not enabled or beam64 is None or beam64.empty:
        audit["skipped"] = True
        return beam64, states, residual_rank, set(), audit

    banned, records = _near_elemental_duplicate_records(
        beam64,
        trace_fraction_max=trace_fraction_max,
    )
    audit["records"] = records
    audit["banned_phase_ids"] = sorted(banned)
    if not banned:
        pd.DataFrame(records).to_csv(out_dir / "near_elemental_duplicate_filter.csv", index=False)
        (out_dir / "near_elemental_duplicate_filter.json").write_text(
            json.dumps(audit, indent=2, default=str),
            encoding="utf-8",
        )
        return beam64, states, residual_rank, set(), audit

    excluded = {str(pid) for pid in (excluded_phase_ids or set()) if str(pid)}
    excluded.update(banned)
    filtered_beam, filtered_states, filtered_residual = _run_beam64_search(
        y64=y64,
        profiles=profiles,
        ids=ids,
        formulas=formulas,
        sgs=sgs,
        shifts=shifts,
        rapid_cfg=rapid_cfg,
        depth=depth,
        excluded_phase_ids=excluded,
    )
    if filtered_beam.empty or filtered_states.empty:
        audit["fail_open"] = True
        audit["fail_open_reason"] = "would_remove_all_hypotheses"
        pd.DataFrame(records).to_csv(out_dir / "near_elemental_duplicate_filter.csv", index=False)
        (out_dir / "near_elemental_duplicate_filter.json").write_text(
            json.dumps(audit, indent=2, default=str),
            encoding="utf-8",
        )
        _log(
            "[rapid][WARN] near-elemental duplicate filter would remove all hypotheses; "
            "keeping the original coarse shortlist."
        )
        return beam64, states, residual_rank, set(), audit

    audit["effective_banned_phase_ids"] = sorted(banned)
    pd.DataFrame(records).to_csv(out_dir / "near_elemental_duplicate_filter.csv", index=False)
    (out_dir / "near_elemental_duplicate_filter.json").write_text(
        json.dumps(audit, indent=2, default=str),
        encoding="utf-8",
    )
    filtered_beam.to_csv(out_dir / "beam64_after_near_elemental_filter.csv", index=False)
    filtered_states.to_csv(out_dir / "beam_states_after_near_elemental_filter.csv", index=False)
    filtered_residual.to_csv(out_dir / "residual_rank_history_after_near_elemental_filter.csv", index=False)
    _log(
        "[rapid] near-elemental duplicate filter removed "
        f"{len(banned)} phase(s): {', '.join(sorted(banned))}"
    )
    return filtered_beam, filtered_states, filtered_residual, banned, audit


def _extract_q_signal(
    cfg: dict,
    dataset: dict,
    out_dir: Path,
    *,
    db_loader: Optional[DBLoader] = None,
    xray_doublet_config: Optional[dict] = None,
    run_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, str, str]:
    work = out_dir / "rapid_base"
    pm = GSASProjectManager(str(work), "rapid_base")
    if not pm.create_project():
        raise RuntimeError("Could not create rapid base project")
    mode = str(dataset.get("mode") or cfg.get("instrument_mode") or "auto").lower()
    instrument_type = "TOF" if mode == "tof" else ("CW" if mode == "cw" else None)
    if not pm.add_histogram(str(dataset["data_path"]), str(dataset["instprm_path"]), instrument_type=instrument_type):
        raise RuntimeError("Could not import diffraction data for rapid mode")
    _effective_limits_and_exclusions(
        cfg,
        dataset,
        mode=mode,
        instprm_path=str(dataset["instprm_path"]),
        hist=pm.main_histogram,
        audit_dir=out_dir,
    )
    signal_kind = "raw data"
    if dataset.get("main_cif"):
        try:
            ok = pm.add_phase_from_cif(str(dataset["main_cif"]), "Main phase", link_to_histogram=True)
            if not ok:
                raise RuntimeError(f"GSAS-II could not add the supplied main CIF: {dataset['main_cif']}")
            refiner = GSASMainPhaseRefiner(pm)

            def _configure_main_anchor_refiner(anchor_refiner: GSASMainPhaseRefiner) -> None:
                try:
                    anchor_refiner.phase.set_refinements({"Cell": False})
                    anchor_refiner.phase.set_HAP_refinements(
                        {"Use": True, "Scale": False},
                        histograms=[anchor_refiner.histogram],
                    )
                    anchor_refiner.phase.HAPvalue(
                        "Scale",
                        1.0,
                        targethistlist=[anchor_refiner.histogram],
                    )
                except Exception:
                    pass

            def _run_anchor_refinement(anchor_refiner: GSASMainPhaseRefiner):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    return anchor_refiner.run_staged_refinement(
                        enable_cell=True,
                        background_config=bg_cfg,
                        bg_type=bg_cfg.get("type"),
                        bg_terms=int(bg_cfg["terms"]) if bg_cfg.get("terms") is not None else None,
                        bg_coeffs=bg_cfg.get("coeffs"),
                    )

            _configure_main_anchor_refiner(refiner)
            bg_cfg = cfg.get("background") or {}
            _run_rapid_light_calibration(refiner, cfg, dataset, work, mode)
            main_results = _run_anchor_refinement(refiner)
            if not main_results.success:
                raise RuntimeError(main_results.error_message or "staged main-phase refinement failed")
            current_main_cif = str(dataset["main_cif"])

            def _build_prenudged_rapid_project(project_name: str, cif_path: str):
                project_name_safe = _safe_name(project_name, max_len=72)
                project_work = out_dir / project_name_safe
                pm_nudged = GSASProjectManager(str(project_work), project_name_safe)
                if not pm_nudged.create_project():
                    raise RuntimeError("Could not create rapid pre-nudged main project")
                inst_path = str(dataset.get("calibrated_instprm_path") or dataset["instprm_path"])
                if not pm_nudged.add_histogram(str(dataset["data_path"]), inst_path, instrument_type=instrument_type):
                    raise RuntimeError("Could not import diffraction data for rapid pre-nudged main project")
                _effective_limits_and_exclusions(
                    cfg,
                    dataset,
                    mode=mode,
                    instprm_path=inst_path,
                    hist=pm_nudged.main_histogram,
                    audit_dir=project_work,
                )
                ok2 = pm_nudged.add_phase_from_cif(str(cif_path), "Main phase", link_to_histogram=True)
                if not ok2:
                    raise RuntimeError(f"GSAS-II could not add the pre-nudged main CIF: {cif_path}")
                refiner_nudged = GSASMainPhaseRefiner(pm_nudged)
                _configure_main_anchor_refiner(refiner_nudged)
                return pm_nudged, refiner_nudged, {"project_work": str(project_work)}

            if db_loader is not None:
                anchor = run_main_phase_prenudge_if_needed(
                    pm=pm,
                    main_ref=refiner,
                    main_results=main_results,
                    main_cif=str(dataset["main_cif"]),
                    main_phase_name="Main phase",
                    top_cfg=cfg,
                    ds_cfg=dataset,
                    s4_cfg=cfg.get("stage4") or {},
                    background_config=bg_cfg,
                    mode=mode,
                    db_loader=db_loader,
                    out_cif_dir=out_dir / "main_phase_nudged_cifs",
                    build_project_from_cif=_build_prenudged_rapid_project,
                    run_refinement=_run_anchor_refinement,
                    user_supplied_main=True,
                    log=_log,
                    event_callback=(
                        None
                        if run_dir is None
                        else lambda message, metrics: _write_event(
                            run_dir,
                            "Stage 1 Rapid",
                            message,
                            18,
                            metrics=metrics,
                        )
                    ),
                    audit_path=out_dir / "main_phase_prenudge.json",
                    xray_doublet_config=xray_doublet_config or {"enabled": False},
                )
                pm = anchor.pm
                refiner = anchor.refiner
                main_results = anchor.refinement_result
                if anchor.audit.get("adopted"):
                    dataset["main_cif_prenudged_path"] = anchor.main_cif
                    current_main_cif = anchor.main_cif
                    signal_kind = "main-phase residual (pre-nudged main)"

            cleanup_anchor = run_main_phase_cleanup_if_enabled(
                pm=pm,
                main_ref=refiner,
                main_results=main_results,
                main_cif=current_main_cif,
                main_phase_name="Main phase",
                top_cfg=cfg,
                ds_cfg=dataset,
                build_project_from_cif=_build_prenudged_rapid_project,
                run_refinement=_run_anchor_refinement,
                out_dir=out_dir / "main_phase_cleanup_cifs",
                user_supplied_main=True,
                log=_log,
                audit_path=out_dir / "main_phase_cleanup.json",
            )
            pm = cleanup_anchor.pm
            refiner = cleanup_anchor.refiner
            main_results = cleanup_anchor.refinement_result
            if cleanup_anchor.audit.get("adopted"):
                dataset["main_cif_cleanup_path"] = cleanup_anchor.main_cif
                current_main_cif = cleanup_anchor.main_cif
                signal_kind = "main-phase residual (cleaned main)"

            final_anchor_cfg = main_prenudge_cfg(cfg, dataset, cfg.get("stage4") or {})
            final_fit_audit, _anchor_q, _anchor_signal = assess_main_fit_for_prenudge(
                refiner,
                getattr(main_results, "rwp", None),
                mode,
                final_anchor_cfg,
                bg_cfg,
            )
            final_reliable, final_reason = main_anchor_reliability_from_fit_audit(
                final_fit_audit,
                main_phase_shadow_cfg(cfg, dataset),
            )
            final_anchor_audit = {
                "reliable": bool(final_reliable),
                "reason": final_reason,
                "fit": final_fit_audit,
                "active_main_cif": current_main_cif,
                "signal_kind_if_accepted": signal_kind if signal_kind != "raw data" else "main-phase residual",
            }
            if not final_reliable:
                warning = (
                    "Supplied main CIF fit is weak, but RADAR-PD will continue because a high "
                    "main-only Rwp can be caused by real impurity phases. Downstream main-shadow "
                    "filters will treat the main anchor as unreliable and fail open instead of "
                    "removing candidates aggressively."
                )
                final_anchor_audit["warning"] = warning
                _log(f"[rapid][WARN] {warning} Reason: {final_reason}")
                if run_dir is not None:
                    _write_event(
                        run_dir,
                        "Stage 1 Rapid",
                        "Weak supplied-main fit; continuing with cautious residual search",
                        20,
                        metrics={"level": "WARN", "reason": final_reason},
                    )
            (out_dir / "main_phase_anchor_final.json").write_text(
                json.dumps(final_anchor_audit, indent=2, default=str),
                encoding="utf-8",
            )
            dataset["_main_anchor_reliable"] = bool(final_reliable)
            dataset["_main_anchor_reliability_reason"] = final_reason
            if not final_reliable and bool(final_anchor_cfg.get("strict_fail_unresolved_main", False)):
                raise RuntimeError(
                    "Supplied main CIF could not be anchored reliably enough for strict residual phase search. "
                    f"Reason: {final_reason}. strict_fail_unresolved_main is enabled."
                )

            pm.save_project()
            arrays = GSASDataExtractor.get_all_arrays(pm.main_histogram)
            q = np.asarray(arrays.get("Q", []), dtype=float)
            residual = np.asarray(arrays.get("residual", []), dtype=float)
            if q.size and residual.size:
                if signal_kind == "raw data":
                    signal_kind = "main-phase residual"
                project_path = str(getattr(pm, "project_path", "") or getattr(pm.project, "filename", ""))
                return q, np.maximum(residual, 0.0), signal_kind, project_path
            raise RuntimeError("Main-phase fit did not produce residual arrays")
        except Exception as exc:
            raise RuntimeError(f"Rapid mode could not prepare the supplied main CIF residual: {exc}") from exc
    arrays = GSASDataExtractor.get_all_arrays(pm.main_histogram)
    q = np.asarray(arrays.get("Q", []), dtype=float)
    y = np.asarray(arrays.get("yobs", []), dtype=float)
    pm.save_project()
    signal, signal_kind = _prepare_no_main_rapid_signal(q, y, cfg, out_dir)
    return q, signal, signal_kind, str(pm.project_path)


def _main_shadow_peaks_from_gpx(gpx_path: str, cfg: dict, dataset: dict) -> list[float]:
    if not gpx_path or not Path(gpx_path).exists() or not dataset.get("main_cif"):
        return []
    shadow_cfg = main_phase_shadow_cfg(cfg, dataset)
    if not bool(shadow_cfg.get("enabled", True)):
        return []
    try:
        from GSASII import GSASIIscriptable as G2sc

        proj = G2sc.G2Project(gpxfile=str(gpx_path))
        hist = proj.histograms()[0] if proj.histograms() else None
        if hist is None:
            return []
        arrays = GSASDataExtractor.get_all_arrays(hist)
        q_arr = np.asarray(arrays.get("Q", []), dtype=float)
        anchors = list(main_shadow_peaks_from_arrays(
            q_arr,
            np.asarray(arrays.get("ycalc", []), dtype=float),
            shadow_cfg,
        ))
        main_cif_path = ""
        for key in ("main_cif_cleanup_path", "main_cif_prenudged_path", "main_cif"):
            candidate = str(dataset.get(key) or "").strip()
            if candidate and Path(candidate).exists():
                main_cif_path = candidate
                break
        if q_arr.size >= 3 and main_cif_path:
            q_grid = np.linspace(float(np.nanmin(q_arr)), float(np.nanmax(q_arr)), 512, dtype=np.float64)
            _profile, raw_q, raw_i = _render_cif_profile512_with_peaks(main_cif_path, q_grid, cfg)
            if raw_q.size and raw_i.size == raw_q.size:
                q_min = float(np.nanmin(q_arr))
                q_max = float(np.nanmax(q_arr))
                order = np.argsort(np.asarray(raw_i, dtype=float))[::-1]
                tol = float(shadow_cfg.get("peak_match_tolerance_q", 0.040))
                for idx in order:
                    qv = float(raw_q[int(idx)])
                    if not math.isfinite(qv) or qv < q_min or qv > q_max:
                        continue
                    if any(abs(qv - existing) <= tol for existing in anchors):
                        continue
                    anchors.append(qv)
                    if len(anchors) >= int(shadow_cfg.get("top_main_peaks", 8)):
                        break
        return anchors[: max(1, int(shadow_cfg.get("top_main_peaks", 8)))]
    except Exception as exc:
        _log(f"[rapid][WARN] Could not build main-shadow peak windows: {exc}")
        return []


def _prepare_no_main_rapid_signal(
    q: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    out_dir: Path | None = None,
) -> Tuple[np.ndarray, str]:
    """Return the search target for rapid runs that do not have a main phase.

    Raw lab PXRD patterns often contain a large smooth background/envelope. If
    that envelope is sent directly into the 64-bin hypothesis search, the broad
    background can dominate the match and push simple host phases far down the
    ranking. For no-main rapid mode, use the same low-envelope estimator already
    used for GSAS-II fixed background points, then search the positive residual.
    """
    q = np.asarray(q, dtype=float)
    y = np.asarray(y, dtype=float)
    y_positive = np.maximum(y, 0.0)
    rapid_cfg = cfg.get("rapid_hypothesis") or {}
    bg_cfg = cfg.get("background") or {}
    enabled = bool(rapid_cfg.get("background_subtract_no_main", True))
    if not enabled or str(bg_cfg.get("mode") or "auto_fixed_points") != "auto_fixed_points":
        return y_positive, "raw data"

    finite = np.isfinite(q) & np.isfinite(y_positive)
    if int(finite.sum()) < 50:
        return y_positive, "raw data"

    try:
        params = coerce_auto_background_params(bg_cfg.get("auto_params") or {})
        background, fixed_points, resolved = estimate_background(q[finite], y_positive[finite], params=params)
        residual_finite = np.maximum(y_positive[finite] - background, 0.0)

        raw_sum = float(np.nansum(y_positive[finite]))
        residual_sum = float(np.nansum(residual_finite))
        residual_max = float(np.nanmax(residual_finite)) if residual_finite.size else 0.0
        positive_bins = int(np.count_nonzero(residual_finite > max(1e-9, residual_max * 1e-4)))
        if residual_max <= 0.0 or residual_sum <= max(1e-8, raw_sum * 1e-4) or positive_bins < 10:
            return y_positive, "raw data"

        signal = np.zeros_like(y_positive, dtype=float)
        signal[finite] = residual_finite
        if out_dir is not None:
            try:
                np.savez_compressed(
                    Path(out_dir) / "target_background.npz",
                    Q=q[finite],
                    raw=y_positive[finite],
                    background=np.asarray(background, dtype=float),
                    residual=residual_finite,
                    fixed_points=np.asarray(fixed_points, dtype=float),
                    snip_iterations=int(getattr(resolved, "snip_iterations", 0) or 0),
                )
            except Exception as exc:
                _log(f"[rapid][WARN] could not save background-subtraction audit: {exc}")
        _log(
            "[rapid] no-main background subtraction active: "
            f"residual_sum/raw_sum={residual_sum / max(raw_sum, 1e-8):.4f}, "
            f"points={len(fixed_points)}"
        )
        return signal, "background-subtracted raw data"
    except Exception as exc:
        _log(f"[rapid][WARN] no-main background subtraction failed; using raw data: {exc}")
        return y_positive, "raw data"


def _histogram_signal(q: np.ndarray, y: np.ndarray, meta: dict) -> np.ndarray:
    edges = np.linspace(float(meta["q_min"]), float(meta["q_max"]), int(meta["n_bins"]) + 1, dtype=np.float64)
    hist, _observed, _counts = _residual_hist_from_continuous_parts(
        np.asarray(q, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.array([], dtype=np.float64),
        edges,
        float(meta.get("sigma_bins", 0.7)),
        peak_mask_width=0.0,
        debug_plot=False,
    )
    hist = np.maximum(np.asarray(hist, dtype=np.float32), 0.0)
    if hist.max() > 0:
        hist = hist / (float(hist.max()) + 1e-8)
    return hist.astype(np.float32)


def _candidate_matrix(cfg: dict, db_loader: DBLoader, profile_ctx: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    catalog = db_loader.catalog.copy()
    formula_metadata = _load_formula_metadata(cfg)
    ids_all = catalog["id"].astype(str).tolist()
    dataset = (cfg.get("datasets") or [{}])[0] if isinstance(cfg.get("datasets"), list) else {}
    main_sig = _main_phase_signature(dataset or {})
    duplicate_main_count = 0
    ef = cfg.get("element_filter") or {}
    allowed = list(cfg.get("allowed_elements") or [])
    if allowed:
        ids = db_loader.filter_by_element_mask(
            allowed,
            ids_all,
            ignore_elements=ef.get("ignore_elements"),
            require_base=bool(ef.get("require_base", True)),
            max_offlist_elements=int(ef.get("max_offlist_elements", 0)),
            disallow_offlist=ef.get("disallow_offlist"),
            wildcard_relation=str(ef.get("wildcard_relation", "any")),
            sample_env=ef.get("sample_env"),
            disallow_pure=ef.get("disallow_pure"),
        )
    else:
        ids = ids_all
    exclude_sg = cfg.get("exclude_sg", [1, 2])
    if exclude_sg:
        try:
            ids = db_loader.exclude_space_groups(ids, exclude_sg=exclude_sg)
        except Exception:
            pass
    pid_to_row = profile_ctx["pid_to_row"]
    keep_ids: list[str] = []
    rows: list[int] = []
    formulas: list[str] = []
    sgs: list[int] = []
    for pid in ids:
        if str(pid) not in pid_to_row:
            continue
        formula = _catalog_formula(catalog, str(pid), formula_metadata)
        try:
            sgnum = int(db_loader.get_space_group_number(str(pid)) or 0)
        except Exception:
            sgnum = 0
        if _is_main_phase_duplicate(formula, sgnum, main_sig):
            duplicate_main_count += 1
            continue
        keep_ids.append(str(pid))
        rows.append(int(pid_to_row[str(pid)]))
        formulas.append(formula)
        sgs.append(sgnum)
    if main_sig and duplicate_main_count:
        _log(
            "[rapid] removed "
            f"{duplicate_main_count} candidate(s) duplicating supplied main phase "
            f"{main_sig.get('formula')} (SG {main_sig.get('space_group') or 'unknown'})"
        )
    profiles = np.asarray(profile_ctx["profiles"], dtype=np.float32)[np.asarray(rows, dtype=np.int64), :]
    return np.asarray(keep_ids, dtype=object), np.asarray(formulas, dtype=object), np.asarray(sgs, dtype=np.int32), profiles


def _gaussian_deposit(profile: np.ndarray, center_bin: float, weight: float, sigma_bins: float) -> None:
    half = int(max(1, math.ceil(3.0 * float(sigma_bins))))
    j0 = int(round(center_bin))
    lo = max(0, j0 - half)
    hi = min(profile.shape[0] - 1, j0 + half)
    if hi < lo:
        return
    js = np.arange(lo, hi + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * ((js - float(center_bin)) / max(float(sigma_bins), 1e-6)) ** 2)
    profile[lo : hi + 1] += float(weight) * kernel


def _profile_from_qi(q: np.ndarray, inten: np.ndarray, q_grid: np.ndarray, sigma_bins: float = 1.0) -> np.ndarray:
    profile = np.zeros(int(q_grid.size), dtype=np.float32)
    if q_grid.size < 2:
        return profile
    dq = float(q_grid[1] - q_grid[0])
    q_min = float(q_grid[0] - 0.5 * dq)
    q_max = float(q_grid[-1] + 0.5 * dq)
    inten = np.asarray(inten, dtype=np.float32)
    if inten.size and float(np.max(inten)) > 0:
        inten = inten / float(np.max(inten))
    for qv, iv in zip(np.asarray(q, dtype=float), inten):
        if q_min <= float(qv) < q_max:
            center = (float(qv) - q_min) / dq - 0.5
            _gaussian_deposit(profile, center, float(iv), sigma_bins)
    if profile.max() > 0:
        profile /= float(profile.max())
    return profile.astype(np.float32)


def _render_cif_profile512_with_peaks(
    cif_path: str,
    q_grid: np.ndarray,
    cfg: dict,
    doublet_spec: Optional[dict] = None,
) -> tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
    try:
        from pymatgen.io.cif import CifParser
        from database_catalog_builder import simulate_topM_peaks

        stage4 = cfg.get("stage4") or {}
        if doublet_spec is None:
            dataset = (cfg.get("datasets") or [{}])[0] if isinstance(cfg.get("datasets"), list) else {}
            doublet_spec_obj = resolve_xray_doublet_spec(
                cfg,
                dataset=dataset,
                instprm_path=(dataset or {}).get("instprm_path"),
                stage4=stage4,
            )
        else:
            doublet_spec_obj = doublet_spec
        structure = CifParser(str(cif_path), occupancy_tolerance=2.0).parse_structures(primitive=False)[0]
        _tt, _d, q, inten = simulate_topM_peaks(
            structure,
            float(stage4.get("two_theta_range", [0.0, 160.0])[0]),
            float(stage4.get("two_theta_range", [0.0, 160.0])[1]),
            700,
            radiation=str(stage4.get("radiation", "neutron")),
            wavelength=float(stage4.get("wavelength", 1.54)),
        )
        q_profile, i_profile = apply_doublet_to_peaks(q, inten, doublet_spec_obj, apply_key="apply_to_512")
        return _profile_from_qi(q_profile, i_profile, q_grid, sigma_bins=1.1), np.asarray(q, dtype=np.float32), np.asarray(inten, dtype=np.float32)
    except Exception:
        return None, np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)


def _render_cif_profile512(cif_path: str, q_grid: np.ndarray, cfg: dict) -> Optional[np.ndarray]:
    profile, _q, _inten = _render_cif_profile512_with_peaks(cif_path, q_grid, cfg)
    return profile


def _scaled_fit_components(rows: Sequence[np.ndarray], coefs: Sequence[float]) -> list[np.ndarray]:
    if not rows:
        return []
    active = np.vstack(rows).astype(np.float32)
    norms = np.maximum(np.linalg.norm(active, axis=1), 1e-8)
    out: list[np.ndarray] = []
    for row, norm, coef in zip(active, norms, np.asarray(coefs, dtype=np.float32)):
        out.append((row / float(norm) * float(coef)).astype(np.float32))
    return out


def _best_modeled_peaks(
    *,
    q_grid: np.ndarray,
    target: np.ndarray,
    total_fit: np.ndarray,
    component: np.ndarray,
    raw_q: np.ndarray,
    raw_i: np.ndarray,
    phase_label: str,
    max_peaks: int = 5,
) -> list[dict[str, Any]]:
    if q_grid.size < 2 or component.size != q_grid.size:
        return []
    q_min = float(q_grid[0])
    q_max = float(q_grid[-1])
    raw_q = np.asarray(raw_q, dtype=float)
    raw_i = np.asarray(raw_i, dtype=float)
    if raw_q.size and raw_i.size == raw_q.size:
        keep = np.where((raw_q >= q_min) & (raw_q <= q_max) & np.isfinite(raw_i))[0]
        if keep.size:
            order = keep[np.argsort(raw_i[keep])[::-1]]
            selected_q = raw_q[order[:max_peaks]]
            selected_i = raw_i[order[:max_peaks]]
        else:
            selected_q = np.zeros(0, dtype=float)
            selected_i = np.zeros(0, dtype=float)
    else:
        # Fallback when only the broadened component is available: use local
        # maxima in the scaled contribution, which is still a useful inspection
        # proxy for the strongest modeled Bragg features.
        y = np.asarray(component, dtype=float)
        local = np.where((y[1:-1] >= y[:-2]) & (y[1:-1] >= y[2:]) & (y[1:-1] > 0))[0] + 1
        order = local[np.argsort(y[local])[::-1]] if local.size else np.array([], dtype=int)
        selected_q = q_grid[order[:max_peaks]]
        selected_i = y[order[:max_peaks]]
    peaks: list[dict[str, Any]] = []
    if selected_q.size == 0:
        return peaks
    dq = abs(float(q_grid[1] - q_grid[0]))
    for qv, strength in zip(selected_q[:max_peaks], selected_i[:max_peaks]):
        idx = int(np.argmin(np.abs(q_grid - float(qv))))
        lo = max(0, idx - 2)
        hi = min(q_grid.size, idx + 3)
        observed_peak = float(np.nanmax(target[lo:hi])) if hi > lo else float(target[idx])
        component_peak = float(np.nanmax(component[lo:hi])) if hi > lo else float(component[idx])
        total_peak = float(np.nanmax(total_fit[lo:hi])) if hi > lo else float(total_fit[idx])
        if component_peak <= 1e-8:
            support = "not visible"
        else:
            ratio = observed_peak / max(component_peak, 1e-8)
            if total_peak > observed_peak * 1.35 and component_peak > observed_peak * 0.35:
                support = "overfit risk"
            elif ratio >= 0.75:
                support = "supported"
            elif ratio >= 0.35:
                support = "weak support"
            else:
                support = "missing/overfit"
        peaks.append(
            {
                "phase": phase_label,
                "q": float(qv),
                "grid_q": float(q_grid[idx]),
                "window_q": float(2.0 * dq),
                "relative_peak_strength": float(strength),
                "observed_peak": observed_peak,
                "component_peak": component_peak,
                "total_peak": total_peak,
                "support": support,
            }
        )
    return peaks


def _write_512_payload(
    *,
    out_dir: Path,
    row_key: str,
    q_grid: np.ndarray,
    y512: np.ndarray,
    total_fit: np.ndarray,
    background_fit: np.ndarray,
    phase_components: Sequence[np.ndarray],
    phase_labels: Sequence[str],
    phase_coefs: Sequence[float],
    raw_peaks: Sequence[tuple[np.ndarray, np.ndarray]],
    score: float,
    sse: float,
    formulas: str,
    space_groups: str,
    phase_ids: str,
) -> tuple[str, str]:
    payload_dir = out_dir / "component_payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)
    peak_groups: list[dict[str, Any]] = []
    summary_bits: list[str] = []
    for idx, (label, component) in enumerate(zip(phase_labels, phase_components)):
        raw_q, raw_i = raw_peaks[idx] if idx < len(raw_peaks) else (np.zeros(0), np.zeros(0))
        peaks = _best_modeled_peaks(
            q_grid=q_grid,
            target=y512,
            total_fit=total_fit,
            component=np.asarray(component, dtype=np.float32),
            raw_q=np.asarray(raw_q, dtype=np.float32),
            raw_i=np.asarray(raw_i, dtype=np.float32),
            phase_label=str(label),
            max_peaks=5,
        )
        supported = sum(1 for peak in peaks if str(peak.get("support")) == "supported")
        weak = sum(1 for peak in peaks if str(peak.get("support")) == "weak support")
        total = len(peaks)
        if total:
            missing = max(0, total - supported - weak)
            summary_bits.append(f"{label}: {supported} supported, {weak} weak, {missing} missing/review")
        peak_groups.append({"phase": str(label), "peaks": peaks})
    coefs = np.asarray(phase_coefs, dtype=float)
    positive = np.maximum(coefs, 0.0)
    denom = float(np.sum(positive))
    relative = positive / denom if denom > 0 else np.zeros_like(positive)
    payload = {
        "plot_kind": "rapid_refined_pattern_match",
        "title": "Refined pattern match",
        "formulas": formulas,
        "space_groups": space_groups,
        "phase_ids": phase_ids,
        "score": float(score),
        "sse": float(sse),
        "q": np.asarray(q_grid, dtype=float).round(6).tolist(),
        "target": np.asarray(y512, dtype=float).round(7).tolist(),
        "total_fit": np.asarray(total_fit, dtype=float).round(7).tolist(),
        "background": np.asarray(background_fit, dtype=float).round(7).tolist(),
        "residual": np.asarray(y512 - total_fit, dtype=float).round(7).tolist(),
        "phases": [
            {
                "label": str(label),
                "coefficient": float(coefs[idx]) if idx < len(coefs) else 0.0,
                "relative_scale": float(relative[idx]) if idx < len(relative) else 0.0,
                "component": np.asarray(component, dtype=float).round(7).tolist(),
                "top_peaks": peak_groups[idx]["peaks"] if idx < len(peak_groups) else [],
            }
            for idx, (label, component) in enumerate(zip(phase_labels, phase_components))
        ],
        "peak_support_summary": "; ".join(summary_bits) if summary_bits else "-",
        "created_at": time.time(),
    }
    safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", row_key).strip("_") or "hypothesis"
    path = payload_dir / f"{safe_key}.plotdata.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path), str(payload.get("peak_support_summary") or "-")


def _write_rerank512_progress(out_dir: Path, rows: list[dict[str, Any]], *, total: int, started_at: float, done: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    elapsed = max(0.0, time.perf_counter() - started_at)
    processed = len(rows)
    if rows:
        view = pd.DataFrame(rows).sort_values(["score512", "rank64"], ascending=[False, True]).reset_index(drop=True)
        view.insert(0, "rank512", np.arange(1, len(view) + 1))
        view.to_csv(out_dir / "reranked_512_after_radar_nudge.partial.csv", index=False)
        best = view.iloc[0].to_dict()
    else:
        best = {}
    rate = processed / elapsed if elapsed > 0 and processed else 0.0
    remaining = (max(0, total - processed) / rate) if rate > 0 else None
    status = {
        "stage": "refined_pattern_match",
        "processed": int(processed),
        "total": int(total),
        "done": bool(done),
        "elapsed_seconds": float(elapsed),
        "estimated_remaining_seconds": None if remaining is None else float(remaining),
        "best_hypothesis": best,
        "updated_at": time.time(),
    }
    (out_dir / "rerank512_status.json").write_text(json.dumps(status, default=str, indent=2), encoding="utf-8")


def _nudge_unique_phases(
    rows: pd.DataFrame,
    db_loader: DBLoader,
    q: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    out_dir: Path,
) -> pd.DataFrame:
    stage4 = cfg.get("stage4") or {}
    rapid_cfg = cfg.get("rapid_hypothesis") or {}
    max_phases = int(rapid_cfg.get("nudge_unique_phases", 0) or 0)
    phase_order: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for row in rows.itertuples(index=False):
        for pid, fkey, formula in zip(_split_pipe(row.phase_ids), _split_pipe(row.formula_keys), _split_pipe(row.formulas)):
            if pid not in seen:
                seen.add(pid)
                phase_order.append((pid, fkey, formula))
            if max_phases > 0 and len(phase_order) >= max_phases:
                break
        if max_phases > 0 and len(phase_order) >= max_phases:
            break
    nudge_dir = out_dir / "nudged_cifs"
    nudge_dir.mkdir(parents=True, exist_ok=True)

    def _base_rec(pid: str, fkey: str, formula: str) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "phase_id": pid,
            "formula_key": fkey,
            "formula": formula,
            "space_group": "",
            "best_score": np.nan,
            "distance_from_start": 0.0,
            "a": np.nan,
            "b": np.nan,
            "c": np.nan,
            "alpha": np.nan,
            "beta": np.nan,
            "gamma": np.nan,
            "nudged_cif": "",
            "seconds": 0.0,
            "error": "",
        }
        try:
            rec["space_group"] = int(db_loader.get_space_group_number(pid) or 0)
        except Exception:
            pass
        return rec

    def _record_from_result(pid: str, fkey: str, formula: str, res: Any | None, elapsed: float = 0.0, error: str = "") -> dict[str, Any]:
        rec = _base_rec(pid, fkey, formula)
        if res is not None:
            params = getattr(res, "best_params", {}) or {}
            rec.update(
                {
                    "best_score": float(getattr(res, "best_score", np.nan)),
                    "nudged_cif": str(getattr(res, "nudged_cif_path", "") or ""),
                    "seconds": float(getattr(res, "elapsed_s", elapsed) or elapsed),
                    "distance_from_start": float(getattr(res, "lattice_deviation", 0.0) or 0.0),
                    "a": float(params.get("a", np.nan)),
                    "b": float(params.get("b", np.nan)),
                    "c": float(params.get("c", np.nan)),
                    "alpha": float(params.get("alpha", np.nan)),
                    "beta": float(params.get("beta", np.nan)),
                    "gamma": float(params.get("gamma", np.nan)),
                }
            )
        if error:
            rec["error"] = error
        if not rec["nudged_cif"]:
            try:
                rec["nudged_cif"] = db_loader.ensure_cif_on_disk(pid, out_dir=str(nudge_dir), overwrite=False)
            except Exception as exc:
                rec["error"] = "; ".join(part for part in [str(rec.get("error") or ""), str(exc)] if part)
        return rec

    results: list[dict[str, Any]] = []
    try:
        from lattice_nudger import LatticeNudger

        dataset = (cfg.get("datasets") or [{}])[0] if isinstance(cfg.get("datasets"), list) else {}
        doublet_spec = resolve_xray_doublet_spec(
            cfg,
            dataset=dataset,
            instprm_path=(dataset or {}).get("instprm_path"),
            stage4=stage4,
        )
        nudger = LatticeNudger(
            db_loader,
            wavelength_ang=float(stage4.get("wavelength", 1.54)),
            two_theta_range=tuple(stage4.get("two_theta_range", [5.0, 160.0])),
            radiation=str(stage4.get("radiation", "neutron")),
            score_q_max=float(stage4.get("score_q_max", 8.0)),
            xray_doublet_config=doublet_spec.to_dict(),
        )
    except Exception as exc:
        nudger = None
        _log(f"[rapid][WARN] Lattice nudge unavailable: {exc}")

    _log(f"[rapid] lattice nudge queue: {len(phase_order)} unique phase(s) from coarse hypotheses")
    use_parallel = bool(rapid_cfg.get("parallel_nudge", True))
    if nudger is not None and use_parallel and len(phase_order) > 1:
        try:
            t0 = time.perf_counter()
            nudged = nudger.optimize_many(
                [pid for pid, _fkey, _formula in phase_order],
                np.asarray(q, dtype=float),
                np.asarray(y, dtype=float),
                reps=int(stage4.get("reps", 7)),
                samples=int(stage4.get("samples", 500)),
                frac_window=float(stage4.get("len_tol_pct", 1.0)) / 100.0,
                angle_window_deg=float(stage4.get("ang_tol_deg", 3.0)),
                out_cif_dir=str(nudge_dir),
                score_q_max=float(stage4.get("score_q_max", 8.0)),
            )
            by_pid = {str(getattr(res, "phase_id", "")): res for res in nudged}
            elapsed = time.perf_counter() - t0
            for pid, fkey, formula in phase_order:
                res = by_pid.get(pid)
                err = "" if res is not None else "parallel lattice nudge did not return a result; using original CIF"
                rec = _record_from_result(pid, fkey, formula, res, elapsed=0.0, error=err)
                results.append(rec)
                _log(f"[rapid] nudged {fkey} ({pid}) score={rec.get('best_score')}")
            _log(f"[rapid] parallel lattice nudge completed in {elapsed:.2f}s")
            return pd.DataFrame(results)
        except Exception as exc:
            _log(f"[rapid][WARN] Parallel lattice nudge failed; falling back to serial mode: {exc}")

    for pid, fkey, formula in phase_order:
        t0 = time.perf_counter()
        rec = _base_rec(pid, fkey, formula)
        try:
            res = None
            if nudger is not None:
                res = nudger.optimize_one(
                    pid,
                    np.asarray(q, dtype=float),
                    np.asarray(y, dtype=float),
                    reps=int(stage4.get("reps", 7)),
                    samples=int(stage4.get("samples", 500)),
                    frac_window=float(stage4.get("len_tol_pct", 1.0)) / 100.0,
                    angle_window_deg=float(stage4.get("ang_tol_deg", 3.0)),
                    out_cif_dir=str(nudge_dir),
                    allow_inner_parallel=False,
                    score_q_max=float(stage4.get("score_q_max", 8.0)),
                )
                rec = _record_from_result(pid, fkey, formula, res, elapsed=time.perf_counter() - t0)
            if not rec["nudged_cif"]:
                rec["nudged_cif"] = db_loader.ensure_cif_on_disk(pid, out_dir=str(nudge_dir), overwrite=False)
        except Exception as exc:
            rec["error"] = str(exc)
            try:
                rec["nudged_cif"] = db_loader.ensure_cif_on_disk(pid, out_dir=str(nudge_dir), overwrite=False)
            except Exception:
                pass
        rec["seconds"] = float(rec.get("seconds") or (time.perf_counter() - t0))
        results.append(rec)
        _log(f"[rapid] nudged {fkey} ({pid}) score={rec.get('best_score')}")
    return pd.DataFrame(results)


def _filter_rapid_main_shadow_after_nudge(
    beam_rows: pd.DataFrame,
    nudge_df: pd.DataFrame,
    db_loader: DBLoader,
    q: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    out_dir: Path,
    *,
    main_shadow_q: Sequence[float],
    doublet_spec: Optional[dict] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str], dict[str, Any]]:
    dataset = (cfg.get("datasets") or [{}])[0] if isinstance(cfg.get("datasets"), list) else {}
    shadow_cfg = main_phase_shadow_cfg(cfg, dataset)
    main_shadow_q = list(main_shadow_q or [])
    if (
        beam_rows.empty
        or nudge_df.empty
        or not main_shadow_q
        or not bool(shadow_cfg.get("enabled", True))
        or not bool(shadow_cfg.get("nudge_filter_enabled", True))
    ):
        return beam_rows, nudge_df, set(), {"enabled": bool(shadow_cfg.get("enabled", True)), "skipped": True}

    q_arr = np.asarray(q, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if q_arr.size < 3 or y_arr.size < 3:
        return beam_rows, nudge_df, set(), {"enabled": True, "skipped": True, "reason": "insufficient_signal"}
    q_grid = np.linspace(float(np.nanmin(q_arr)), float(np.nanmax(q_arr)), 512, dtype=np.float64)
    records: list[dict[str, Any]] = []
    filtered_pids: set[str] = set()
    main_anchor_reliable = False
    main_anchor_reason = "missing_main_anchor_audit"
    try:
        final_anchor_path = out_dir / "main_phase_anchor_final.json"
        if final_anchor_path.exists():
            final_anchor = json.loads(final_anchor_path.read_text(encoding="utf-8"))
            main_anchor_reliable = bool(final_anchor.get("reliable", False))
            main_anchor_reason = str(final_anchor.get("reason") or "final_anchor_audit")
        else:
            prenudge_path = out_dir / "main_phase_prenudge.json"
            if prenudge_path.exists():
                prenudge_audit = json.loads(prenudge_path.read_text(encoding="utf-8"))
                main_anchor_reliable, main_anchor_reason = main_anchor_reliability_from_audit(
                    prenudge_audit,
                    shadow_cfg,
                )
    except Exception as exc:
        main_anchor_reliable = False
        main_anchor_reason = f"could_not_read_main_anchor_audit: {exc}"

    for row in nudge_df.itertuples(index=False):
        pid = str(getattr(row, "phase_id", "") or "")
        if not pid:
            continue
        raw_q = np.zeros(0, dtype=np.float32)
        raw_i = np.zeros(0, dtype=np.float32)
        cif_path = str(getattr(row, "nudged_cif", "") or "")
        if cif_path and Path(cif_path).exists():
            _profile, raw_q, raw_i = _render_cif_profile512_with_peaks(
                cif_path,
                q_grid,
                cfg,
                doublet_spec=doublet_spec,
            )
        if raw_q.size == 0 or raw_i.size == 0:
            try:
                raw_q = np.asarray(db_loader.load_q0(pid), dtype=np.float32)
                raw_i = np.asarray(db_loader.load_I0(pid), dtype=np.float32)
            except Exception:
                raw_q = np.zeros(0, dtype=np.float32)
                raw_i = np.zeros(0, dtype=np.float32)
        filtered, metrics = main_shadow_filter_decision(
            raw_q,
            raw_i,
            main_shadow_q,
            q_arr,
            y_arr,
            shadow_cfg,
        )
        if filtered:
            filtered_pids.add(pid)
        records.append({
            "phase_id": pid,
            "formula_key": str(getattr(row, "formula_key", "") or ""),
            "formula": str(getattr(row, "formula", "") or ""),
            "space_group": str(getattr(row, "space_group", "") or ""),
            "best_score": getattr(row, "best_score", None),
            **{k: v for k, v in metrics.items() if k != "candidate_peaks"},
        })

    if filtered_pids and not main_anchor_reliable:
        for record in records:
            if str(record.get("phase_id", "")) in filtered_pids:
                record["filtered_effective"] = False
                record["filter_fail_open_reason"] = "main_anchor_not_reliable"
        audit = {
            "config": shadow_cfg,
            "main_peak_q": [float(qv) for qv in main_shadow_q],
            "filtered_phase_ids": sorted(filtered_pids),
            "effective_filtered_phase_ids": [],
            "fail_open": True,
            "fail_open_reason": "main_anchor_not_reliable",
            "main_anchor_reliable": False,
            "main_anchor_reliability_reason": main_anchor_reason,
            "records": records,
        }
        pd.DataFrame(records).to_csv(out_dir / "main_shadow_nudge_filter.csv", index=False)
        (out_dir / "main_shadow_nudge_filter.json").write_text(
            json.dumps(audit, indent=2, default=str),
            encoding="utf-8",
        )
        _log(
            "[rapid][WARN] main-shadow nudge filter flagged "
            f"{len(filtered_pids)} phase(s), but kept them because the supplied main-phase anchor "
            f"is not reliable ({main_anchor_reason})."
        )
        return beam_rows, nudge_df, set(), audit

    if not filtered_pids:
        _log("[rapid] main-shadow nudge filter removed 0 phase(s)")
        audit = {
            "config": shadow_cfg,
            "main_peak_q": [float(qv) for qv in main_shadow_q],
            "filtered_phase_ids": [],
            "effective_filtered_phase_ids": [],
            "fail_open": False,
            "main_anchor_reliable": bool(main_anchor_reliable),
            "main_anchor_reliability_reason": main_anchor_reason,
            "records": records,
        }
        pd.DataFrame(records).to_csv(out_dir / "main_shadow_nudge_filter.csv", index=False)
        (out_dir / "main_shadow_nudge_filter.json").write_text(
            json.dumps(audit, indent=2, default=str),
            encoding="utf-8",
        )
        return beam_rows, nudge_df, filtered_pids, audit

    keep_mask = []
    for row in beam_rows.itertuples(index=False):
        pids = set(_split_pipe(getattr(row, "phase_ids", "")))
        keep_mask.append(not bool(pids & filtered_pids))
    filtered_beam = beam_rows.loc[keep_mask].copy().reset_index(drop=True)
    filtered_nudge = nudge_df[~nudge_df["phase_id"].astype(str).isin(filtered_pids)].copy().reset_index(drop=True)
    if filtered_beam.empty:
        for record in records:
            if str(record.get("phase_id", "")) in filtered_pids:
                record["filtered_effective"] = False
                record["filter_fail_open_reason"] = "would_remove_all_hypotheses"
        audit = {
            "config": shadow_cfg,
            "main_peak_q": [float(qv) for qv in main_shadow_q],
            "filtered_phase_ids": sorted(filtered_pids),
            "effective_filtered_phase_ids": [],
            "fail_open": True,
            "fail_open_reason": "filter_would_remove_all_hypotheses",
            "main_anchor_reliable": bool(main_anchor_reliable),
            "main_anchor_reliability_reason": main_anchor_reason,
            "records": records,
        }
        pd.DataFrame(records).to_csv(out_dir / "main_shadow_nudge_filter.csv", index=False)
        (out_dir / "main_shadow_nudge_filter.json").write_text(
            json.dumps(audit, indent=2, default=str),
            encoding="utf-8",
        )
        _log(
            "[rapid][WARN] main-shadow nudge filter would remove all hypotheses; "
            "requesting a shadow-free coarse-search refill."
        )
        return beam_rows, nudge_df, set(), audit
    for record in records:
        if str(record.get("phase_id", "")) in filtered_pids:
            record["filtered_effective"] = True
    audit = {
        "config": shadow_cfg,
        "main_peak_q": [float(qv) for qv in main_shadow_q],
        "filtered_phase_ids": sorted(filtered_pids),
        "effective_filtered_phase_ids": sorted(filtered_pids),
        "fail_open": False,
        "main_anchor_reliable": bool(main_anchor_reliable),
        "main_anchor_reliability_reason": main_anchor_reason,
        "records": records,
    }
    pd.DataFrame(records).to_csv(out_dir / "main_shadow_nudge_filter.csv", index=False)
    (out_dir / "main_shadow_nudge_filter.json").write_text(
        json.dumps(audit, indent=2, default=str),
        encoding="utf-8",
    )
    filtered_beam.to_csv(out_dir / "beam64_after_main_shadow_filter.csv", index=False)
    filtered_nudge.to_csv(out_dir / "nudge_results_after_main_shadow_filter.csv", index=False)
    _log(
        "[rapid] main-shadow nudge filter removed "
        f"{len(filtered_pids)} phase(s) and {len(beam_rows) - len(filtered_beam)} hypothesis/hypotheses"
    )
    return filtered_beam, filtered_nudge, filtered_pids, audit


def _rerank512(
    beam_rows: pd.DataFrame,
    nudge_df: pd.DataFrame,
    db_loader: DBLoader,
    q: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    meta: dict,
    out_dir: Path,
    doublet_spec: Optional[dict] = None,
) -> pd.DataFrame:
    q_grid = np.linspace(float(meta["q_min"]), float(meta["q_max"]), 512, dtype=np.float64)
    y512 = _histogram_signal(q, y, {"q_min": q_grid[0], "q_max": q_grid[-1], "n_bins": 512, "sigma_bins": 1.1})
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "target_512.npz", q_grid=q_grid, y512=y512)
    nudge_by_pid = {str(r.phase_id): r for r in nudge_df.itertuples(index=False)} if not nudge_df.empty else {}
    profile_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def profile_for(pid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if pid in profile_cache:
            return profile_cache[pid]
        prof = None
        raw_q = np.zeros(0, dtype=np.float32)
        raw_i = np.zeros(0, dtype=np.float32)
        rec = nudge_by_pid.get(pid)
        if rec is not None and str(getattr(rec, "nudged_cif", "") or ""):
            prof, raw_q, raw_i = _render_cif_profile512_with_peaks(str(rec.nudged_cif), q_grid, cfg, doublet_spec=doublet_spec)
        if prof is None:
            try:
                raw_q = np.asarray(db_loader.load_q0(pid), dtype=np.float32)
                raw_i = np.asarray(db_loader.load_I0(pid), dtype=np.float32)
                q_profile, i_profile = apply_doublet_to_peaks(raw_q, raw_i, doublet_spec, apply_key="apply_to_512")
                prof = _profile_from_qi(q_profile, i_profile, q_grid, sigma_bins=1.1)
            except Exception:
                prof = np.zeros(q_grid.size, dtype=np.float32)
        profile_cache[pid] = (np.asarray(prof, dtype=np.float32), raw_q, raw_i)
        return profile_cache[pid]

    bg = _background_rows(q_grid.size)
    _c0, _fit0, baseline_sse = _fit_rows(bg, y512)
    rows: list[dict[str, Any]] = []
    t512 = time.perf_counter()
    total_rows = int(len(beam_rows))
    _write_rerank512_progress(out_dir, rows, total=total_rows, started_at=t512, done=False)
    for row_idx, row in enumerate(beam_rows.itertuples(index=False), start=1):
        pids = _split_pipe(row.phase_ids)
        profile_records = [profile_for(pid) for pid in pids]
        profiles = [rec[0] for rec in profile_records]
        raw_peaks = [(rec[1], rec[2]) for rec in profile_records]
        formulas = _split_pipe(row.formulas)
        sgs = _split_pipe(getattr(row, "space_groups", ""))
        phase_labels = [
            f"{formulas[idx] if idx < len(formulas) else pid} (SG {sgs[idx]})"
            if idx < len(sgs) and str(sgs[idx]).strip()
            else (formulas[idx] if idx < len(formulas) else pid)
            for idx, pid in enumerate(pids)
        ]
        cif_paths: list[str] = []
        for pid in pids:
            rec = nudge_by_pid.get(pid)
            path = str(getattr(rec, "nudged_cif", "") or "") if rec is not None else ""
            if not path:
                try:
                    path = db_loader.ensure_cif_on_disk(pid, out_dir=str(out_dir / "fallback_cifs"), overwrite=False)
                except Exception:
                    path = ""
            cif_paths.append(path)
        coefs, fit, sse = _fit_rows([*bg, *profiles], y512)
        phase_coefs = coefs[-len(profiles):] if profiles else np.zeros(0, dtype=np.float32)
        scaled_components = _scaled_fit_components([*bg, *profiles], coefs)
        bg_components = scaled_components[:len(bg)]
        phase_components = scaled_components[len(bg):]
        background_fit = np.sum(np.vstack(bg_components), axis=0) if bg_components else np.zeros_like(y512)
        score = float((baseline_sse - sse) / max(baseline_sse, 1e-8))
        rank64 = int(getattr(row, "rank64", getattr(row, "rank", 0)))
        row_key = f"rank64_{rank64:04d}_{hashlib.md5(str(row.phase_ids).encode('utf-8')).hexdigest()[:8]}"
        payload_path, peak_support_summary = _write_512_payload(
            out_dir=out_dir,
            row_key=row_key,
            q_grid=q_grid,
            y512=y512,
            total_fit=fit,
            background_fit=background_fit,
            phase_components=phase_components,
            phase_labels=phase_labels,
            phase_coefs=phase_coefs,
            raw_peaks=raw_peaks,
            score=score,
            sse=float(sse),
            formulas=str(row.formulas),
            space_groups=str(getattr(row, "space_groups", "")),
            phase_ids=str(row.phase_ids),
        )
        rows.append(
            {
                "rank64": rank64,
                "formula_keys": str(row.formula_keys),
                "phase_ids": str(row.phase_ids),
                "formulas": str(row.formulas),
                "space_groups": str(getattr(row, "space_groups", "")),
                "cif_paths": "|".join(cif_paths),
                "component_payload": payload_path,
                "peak_support_summary": peak_support_summary,
                "score512": score,
                "r2_512": score,
                "sse512": float(sse),
                "phase_coefs512": "|".join(f"{float(c):.6g}" for c in phase_coefs),
                "overshoot512": 0.0,
                "active_phase_fraction512": float(np.count_nonzero(phase_coefs > 1e-6) / max(len(phase_coefs), 1)),
            }
        )
        _write_rerank512_progress(out_dir, rows, total=total_rows, started_at=t512, done=False)
        if row_idx == 1 or row_idx % 10 == 0 or row_idx == total_rows:
            _log(f"[rapid] refined-pattern match {row_idx}/{total_rows}")
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["score512", "rank64"], ascending=[False, True]).reset_index(drop=True)
        out.insert(0, "rank512", np.arange(1, len(out) + 1))
        out.to_csv(out_dir / "reranked_512_after_radar_nudge.partial.csv", index=False)
    _write_rerank512_progress(out_dir, rows, total=total_rows, started_at=t512, done=True)
    return out


def _extract_curve(
    gpx: Path,
    out_csv: Path,
    out_png: Path,
    title: str,
    *,
    phase_labels: Optional[dict[str, str]] = None,
) -> bool:
    wrote_any = False
    curve: pd.DataFrame | None = None
    try:
        from GSASII import GSASIIscriptable as G2sc

        proj = G2sc.G2Project(gpxfile=str(gpx))
        hist = proj.histograms()[0]
        arrays = GSASDataExtractor.get_all_arrays(hist)
        q = np.asarray(arrays.get("Q", []), dtype=float)
        yobs = np.asarray(arrays.get("yobs", []), dtype=float)
        ycalc = np.asarray(arrays.get("ycalc", []), dtype=float)
        residual = np.asarray(arrays.get("residual", []), dtype=float)
        n = min(q.size, yobs.size, ycalc.size, residual.size)
        if n > 0:
            curve = pd.DataFrame(
                {
                    "Q": q[:n],
                    "yobs": yobs[:n],
                    "ycalc": ycalc[:n],
                    "residual": residual[:n],
                }
            ).sort_values("Q")
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            curve.to_csv(out_csv, index=False)
            wrote_any = True
    except Exception:
        curve = None

    try:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plot_gpx_fit_with_ticks(str(gpx), str(out_png), phase_labels=phase_labels or {})
        if out_png.exists():
            wrote_any = True
    except Exception:
        pass

    if out_png.exists() or curve is None or curve.empty:
        return wrote_any

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = curve
        fig, (ax, axr) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        ax.plot(df["Q"], df["yobs"], ".", ms=1.8, color="#263238", alpha=0.65, label="Observed")
        ax.plot(df["Q"], df["ycalc"], "-", lw=1.0, color="#d62728", label="Calculated")
        ax.set_ylabel("Intensity")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.2)
        axr.plot(df["Q"], df["residual"], "-", lw=0.8, color="#006d77")
        axr.axhline(0, color="#555", lw=0.8)
        axr.set_xlabel("Q (1/A)")
        axr.set_ylabel("Obs-Calc")
        axr.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        wrote_any = True
    except Exception:
        pass
    return wrote_any


_RWP_RE = re.compile(r"(?:Rwp|wR)\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*%", re.IGNORECASE)


def _rwp_from_text(text: str) -> float:
    matches = _RWP_RE.findall(text or "")
    if not matches:
        return float("nan")
    try:
        return float(matches[-1])
    except Exception:
        return float("nan")


def _stdout_has_divergence(text: str) -> bool:
    lowered = (text or "").lower()
    return any(token in lowered for token in ("divergence:", "ouch", "singular matrix", "invalid value"))


def _project_rwp(hist: Any, stdout_text: str = "") -> float:
    try:
        rwp = float(hist.get_wR())
        if math.isfinite(rwp):
            return rwp
    except Exception:
        pass
    return _rwp_from_text(stdout_text)


def _set_histogram_scale(hist: Any, refine: bool) -> None:
    try:
        sample_params = hist.getHistEntryValue(["Sample Parameters"])
        if isinstance(sample_params, dict):
            raw_scale = sample_params.get("Scale", [1.0, False])
            if isinstance(raw_scale, (list, tuple)) and raw_scale:
                value = float(raw_scale[0])
            elif isinstance(raw_scale, (int, float)):
                value = float(raw_scale)
            else:
                value = 1.0
            sample_params["Scale"] = [value, bool(refine)]
            hist.setHistEntryValue(["Sample Parameters"], sample_params)
        if refine:
            hist.set_refinements({"Sample Parameters": ["Scale"]})
        else:
            hist.clear_refinements({"Sample Parameters": ["Scale"]})
    except Exception:
        pass


def _set_background_refine(hist: Any, refine: bool) -> None:
    try:
        hist.set_refinements({"Background": {"refine": bool(refine)}})
    except Exception:
        pass


def _set_phase_hap_scale(phase: Any, hist: Any, value: Optional[float] = None, refine: bool = False) -> None:
    try:
        phase.set_refinements({"Cell": False})
    except Exception:
        pass
    try:
        phase.set_HAP_refinements({"Use": True, "Scale": bool(refine)}, histograms=[hist])
    except Exception:
        pass
    if value is not None:
        try:
            clean = max(float(value), 1e-6)
            phase.HAPvalue("Scale", clean, targethistlist=[hist])
        except Exception:
            pass


def _current_hap_scale(phase: Any, hist_name: str) -> float:
    try:
        hist_cfg = phase.data.get("Histograms", {}).get(hist_name, {})
        raw = hist_cfg.get("Scale", [0.0, False])
        if isinstance(raw, (list, tuple)) and raw:
            return float(raw[0])
        return float(raw)
    except Exception:
        return 0.0


def _initial_hap_scales(coefs_pipe: Any, n_phases: int, n_candidate_phases: int) -> list[float]:
    raw: list[float] = []
    for item in _split_pipe(coefs_pipe):
        try:
            val = float(item)
        except Exception:
            val = 0.0
        raw.append(max(val, 0.0) if math.isfinite(val) else 0.0)
    if len(raw) < n_candidate_phases:
        raw.extend([0.0] * (n_candidate_phases - len(raw)))
    raw = raw[:n_candidate_phases]
    if sum(raw) <= 0:
        raw = [1.0] * max(n_candidate_phases, 1)

    candidate_total = 1.0
    extra_main = max(0, int(n_phases) - int(n_candidate_phases))
    if extra_main:
        candidate_total = 0.25
    cand_norm = [candidate_total * v / max(sum(raw), 1e-12) for v in raw]
    main_norm = [0.75 / extra_main] * extra_main if extra_main else []
    out = [*main_norm, *cand_norm]
    if len(out) < n_phases:
        out.extend([1.0 / max(n_phases, 1)] * (n_phases - len(out)))
    return [max(float(v), 1e-6) for v in out[:n_phases]]


def _row_value(row: Any, key: str, default: Any = "") -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    try:
        return getattr(row, key)
    except Exception:
        return default


def _row_formula_keys(row: Any, n_candidate_phases: int) -> list[str]:
    keys = [str(item).strip() for item in _split_pipe(_row_value(row, "formula_keys", ""))]
    if len(keys) < n_candidate_phases:
        labels = [str(item).strip() for item in _split_pipe(_row_value(row, "formulas", ""))]
        keys.extend(_formula_key(label) for label in labels[len(keys):n_candidate_phases])
    return keys[:n_candidate_phases]


def _frozen_formula_keys(row: Any) -> set[str]:
    return {
        str(item).strip()
        for item in _split_pipe(_row_value(row, "frozen_formula_keys", ""))
        if str(item).strip()
    }


def _candidate_refine_flags(row: Any, phases: Sequence[Any], n_candidate_phases: int) -> list[bool]:
    flags = [True] * len(phases)
    frozen = _frozen_formula_keys(row)
    if not frozen:
        return flags
    extra_main = max(0, len(phases) - int(n_candidate_phases))
    keys = _row_formula_keys(row, n_candidate_phases)
    for cand_ix, key in enumerate(keys):
        phase_ix = extra_main + cand_ix
        if phase_ix < len(flags) and key in frozen:
            flags[phase_ix] = False
    return flags


def _candidate_cif_paths(row: Any, db_loader: DBLoader, run_dir: Path) -> list[str]:
    pids = _split_pipe(_row_value(row, "phase_ids", ""))
    supplied = list(_split_pipe(_row_value(row, "cif_paths", "")))
    paths: list[str] = []
    for idx, pid in enumerate(pids):
        path = supplied[idx] if idx < len(supplied) else ""
        if path and Path(path).exists():
            paths.append(str(Path(path).resolve()))
            continue
        paths.append(db_loader.ensure_cif_on_disk(pid, out_dir=str(run_dir / "cifs"), overwrite=False))
    return paths


def _configure_project_background(proj: Any, hist: Any, cfg: dict, dataset: dict) -> None:
    try:
        refiner = GSASMainPhaseRefiner.__new__(GSASMainPhaseRefiner)
        refiner.project = proj
        refiner.histogram = hist
        refiner.phase = proj.phases()[0] if proj.phases() else None
        mode = str(dataset.get("mode") or cfg.get("instrument_mode") or "auto").lower()
        refiner.instrument_type = "TOF" if mode == "tof" else "CW"
        refiner._configure_background(cfg.get("background") or {})
    except Exception:
        pass


def _hold_histogram_scale_and_profile(proj: Any, hist: Any) -> None:
    _set_histogram_scale(hist, refine=False)
    for var in HISTOGRAM_HOLD_VARS:
        try:
            _add_histogram_hold_constraint(proj, hist, var)
        except Exception:
            pass
    try:
        cons = proj.data.setdefault("Constraints", {})
        if "HAP" in cons:
            cons["HAP"] = []
    except Exception:
        pass


def _clone_with_lst(src_gpx: str | Path, dst_gpx: str | Path) -> None:
    src = Path(src_gpx)
    dst = Path(dst_gpx)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    src_lst = src.with_suffix(".lst")
    if src_lst.exists():
        shutil.copy2(src_lst, dst.with_suffix(".lst"))


def _save_gsas_project(proj: Any, gpx: Path) -> None:
    try:
        proj.save(str(gpx))
    except TypeError:
        proj.save()


def _checkpoint_project(proj: Any, gpx: Path, label: str) -> Path:
    _save_gsas_project(proj, gpx)
    checkpoint = gpx.with_name(f"{gpx.stem}_{label}.gpx")
    shutil.copy2(gpx, checkpoint)
    lst = gpx.with_suffix(".lst")
    if lst.exists():
        shutil.copy2(lst, checkpoint.with_suffix(".lst"))
    return checkpoint


def _restore_project_from_checkpoint(pm: GSASProjectManager, gpx: Path, checkpoint: Path) -> tuple[Any, Any]:
    shutil.copy2(checkpoint, gpx)
    checkpoint_lst = checkpoint.with_suffix(".lst")
    lst = gpx.with_suffix(".lst")
    if checkpoint_lst.exists():
        shutil.copy2(checkpoint_lst, lst)
    elif lst.exists():
        lst.unlink()
    from GSASII import GSASIIscriptable as G2sc

    pm.project = G2sc.G2Project(gpxfile=str(gpx))
    pm.main_histogram = pm.project.histograms()[0] if pm.project.histograms() else None
    return pm.project, pm.main_histogram


def _fit_is_reasonable(hist: Any, rwp: float, stdout_text: str) -> tuple[bool, str]:
    if _stdout_has_divergence(stdout_text):
        return False, "GSAS-II reported divergence"
    if not math.isfinite(float(rwp)):
        return False, "Rwp is not finite"
    if float(rwp) >= 99.0:
        return False, f"Rwp is {float(rwp):.2f}%"
    try:
        arrays = GSASDataExtractor.get_all_arrays(hist)
        yobs = np.asarray(arrays.get("yobs", []), dtype=float)
        ycalc = np.asarray(arrays.get("ycalc", []), dtype=float)
        if yobs.size and ycalc.size:
            obs_hi = float(np.nanpercentile(np.abs(yobs), 99.5))
            calc_hi = float(np.nanpercentile(np.abs(ycalc), 99.5))
            if math.isfinite(obs_hi) and math.isfinite(calc_hi) and obs_hi > 0 and calc_hi > 100.0 * obs_hi:
                return False, f"calculated intensity is too large ({calc_hi:.3g} vs obs {obs_hi:.3g})"
    except Exception:
        pass
    return True, ""


def _run_gsas_stage(proj: Any, hist: Any, gpx: Path, label: str, cycles: int) -> GSASStageResult:
    try:
        proj.data["Controls"]["data"]["max cyc"] = int(max(1, cycles))
    except Exception:
        pass
    stdout_buf = io.StringIO()
    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stdout_buf):
        try:
            proj.do_refinements([{"refine": True}])
        except Exception as exc:
            stdout_buf.write(f"\n[rapid-gsas] exception in {label}: {exc}\n")
    stdout_text = stdout_buf.getvalue()
    _save_gsas_project(proj, gpx)
    rwp = _project_rwp(hist, stdout_text)
    ok, reason = _fit_is_reasonable(hist, rwp, stdout_text)
    return GSASStageResult(label=label, ok=ok, rwp=rwp, stdout=stdout_text, reason=reason)


def _phase_weight_fallback(proj: Any, hist_name: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for phase in proj.phases():
        scale = _current_hap_scale(phase, hist_name)
        if math.isfinite(scale) and scale > 0:
            values[str(phase.name)] = scale
    total = sum(values.values())
    if total <= 0:
        return {}
    return {name: 100.0 * value / total for name, value in values.items()}


def _read_gsas_weights(proj: Any, gpx: Path, hist_name: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    try:
        parsed = parse_gsas_lst(gpx.with_suffix(".lst"), hist_name) if gpx.with_suffix(".lst").exists() else {}
        for phase in proj.phases():
            vals = parsed.get(phase.name, {})
            weights[str(phase.name)] = float(vals.get("weight_fraction_pct", 0.0) or 0.0)
    except Exception:
        weights = {}
    if not weights or sum(v for v in weights.values() if math.isfinite(v) and v > 0) <= 0:
        weights = _phase_weight_fallback(proj, hist_name)
    return weights


def _display_weight_labels(weights: dict[str, float], phase_labels: dict[str, str]) -> dict[str, float]:
    if not weights:
        return {}
    remapped: dict[str, float] = {}
    for key, value in weights.items():
        label = str(phase_labels.get(str(key)) or key)
        remapped[label] = value
    return remapped


def _weights_are_plausible(
    weights: dict[str, float],
    *,
    has_main_phase: bool,
    min_main_phase_pct: float = 0.1,
    min_any_candidate: float = 0.01,
) -> tuple[bool, str]:
    finite_weights: dict[str, float] = {}
    for key, value in (weights or {}).items():
        try:
            clean_value = float(value)
        except Exception:
            continue
        if math.isfinite(clean_value):
            finite_weights[str(key)] = clean_value
    if not finite_weights:
        return False, "no readable phase fractions"
    total = sum(max(0.0, v) for v in finite_weights.values())
    if total < 50.0 or total > 150.0:
        return False, f"weight-fraction sum {total:.3f}% outside [50, 150]"
    if has_main_phase and finite_weights.get("Main phase", 0.0) < float(min_main_phase_pct):
        return False, f"main phase collapsed to {finite_weights.get('Main phase', 0.0):.4f}%"
    candidate_weights = [
        v for k, v in finite_weights.items()
        if not (has_main_phase and k == "Main phase")
    ]
    if candidate_weights and max(candidate_weights) < float(min_any_candidate):
        return False, f"all candidate phases are below {min_any_candidate:g} wt%"
    bad = [k for k, v in finite_weights.items() if v < -1e-6 or v > 1000.0]
    if bad:
        return False, f"non-physical phase fraction(s): {bad[:5]}"
    return True, ""


def _run_stable_gsas_validation(
    pm: GSASProjectManager,
    gpx: Path,
    row: Any,
    n_candidate_phases: int,
    cfg: dict,
) -> tuple[str, float, dict[str, float], str, list[str], bool]:
    """Run a staged, rollback-safe final refinement for a rapid hypothesis."""
    hist = pm.main_histogram
    proj = pm.project
    phases = list(proj.phases())
    if not phases:
        raise RuntimeError("No phases were added to the refinement project")
    _configure_project_background(proj, hist, cfg, {})

    initial_scales = _initial_hap_scales(_row_value(row, "phase_coefs512", ""), len(phases), n_candidate_phases)
    phase_refine_flags = _candidate_refine_flags(row, phases, n_candidate_phases)
    for phase, scale in zip(phases, initial_scales):
        _set_phase_hap_scale(phase, hist, scale, refine=False)
    initial_checkpoint = _checkpoint_project(proj, gpx, "initial")

    stdout_parts: list[str] = []
    errors: list[str] = []
    stable_checkpoint: Optional[Path] = None
    stable_label = ""
    best_rwp = float("nan")
    has_main_phase = any(str(getattr(phase, "name", "")) == "Main phase" for phase in phases)
    main_guard_min_pct = 0.1
    if has_main_phase:
        try:
            ds_cfg = (cfg.get("datasets") or [{}])[0] if isinstance(cfg.get("datasets"), list) else {}
            guard_cfg = main_phase_guard_cfg(cfg, ds_cfg or {})
            if bool(guard_cfg.get("enabled", True)):
                main_guard_min_pct = float(guard_cfg.get("min_weight_pct", 20.0))
        except Exception:
            main_guard_min_pct = 0.1

    # Stage 1: mature joint-style refinement. Hold sample/profile terms so HAP
    # phase scales carry mixture information; refine background and HAP scales.
    _hold_histogram_scale_and_profile(proj, hist)
    _set_background_refine(hist, refine=True)
    for phase, refine_scale in zip(phases, phase_refine_flags):
        _set_phase_hap_scale(phase, hist, None, refine=refine_scale)
        try:
            set_phase_cell_refine(phase, refine=False)
        except Exception:
            pass
    stage1 = _run_gsas_stage(
        proj,
        hist,
        gpx,
        "hap_scales_background_fixed_cells",
        cycles=int((cfg.get("rapid_hypothesis") or {}).get("final_refine_cycles", cfg.get("max_joint_cycles", 6))),
    )
    stdout_parts.append(f"[{stage1.label}]\n{stage1.stdout}")
    weights1 = _read_gsas_weights(proj, gpx, hist.name)
    weights1_ok, weights1_reason = _weights_are_plausible(
        weights1,
        has_main_phase=has_main_phase,
        min_main_phase_pct=main_guard_min_pct,
    )
    if stage1.ok and weights1_ok:
        stable_checkpoint = _checkpoint_project(proj, gpx, "stable_hap_background")
        stable_label = stage1.label
        best_rwp = stage1.rwp
    else:
        errors.append(f"{stage1.label}: {stage1.reason or weights1_reason}")
        # Fallback: keep the same HAP scale setup but decouple background.
        _restore_project_from_checkpoint(pm, gpx, initial_checkpoint)
        proj, hist = pm.project, pm.main_histogram
        _hold_histogram_scale_and_profile(proj, hist)
        _set_background_refine(hist, refine=False)
        for phase, refine_scale in zip(proj.phases(), phase_refine_flags):
            _set_phase_hap_scale(phase, hist, None, refine=refine_scale)
            try:
                set_phase_cell_refine(phase, refine=False)
            except Exception:
                pass
        stage1b = _run_gsas_stage(proj, hist, gpx, "hap_scales_fixed_background", cycles=3)
        stdout_parts.append(f"[{stage1b.label}]\n{stage1b.stdout}")
        weights1b = _read_gsas_weights(proj, gpx, hist.name)
        weights1b_ok, weights1b_reason = _weights_are_plausible(
            weights1b,
            has_main_phase=has_main_phase,
            min_main_phase_pct=main_guard_min_pct,
        )
        if stage1b.ok and weights1b_ok:
            stable_checkpoint = _checkpoint_project(proj, gpx, "stable_hap_only")
            stable_label = stage1b.label
            best_rwp = stage1b.rwp
        else:
            errors.append(f"{stage1b.label}: {stage1b.reason or weights1b_reason}")

    if stable_checkpoint is None:
        return "error", _project_rwp(hist, "\n".join(stdout_parts)), {}, "\n".join(stdout_parts), errors, True

    # Stage 2: one guarded cleanup from the stable checkpoint. Cells remain fixed.
    proj, hist = _restore_project_from_checkpoint(pm, gpx, stable_checkpoint)
    _set_histogram_scale(hist, refine=False)
    _hold_histogram_scale_and_profile(proj, hist)
    _set_background_refine(hist, refine=True)
    for phase, refine_scale in zip(proj.phases(), phase_refine_flags):
        _set_phase_hap_scale(phase, hist, None, refine=refine_scale)
        try:
            set_phase_cell_refine(phase, refine=False)
        except Exception:
            pass
    stage2 = _run_gsas_stage(proj, hist, gpx, "hap_scales_background_cleanup", cycles=2)
    stdout_parts.append(f"[{stage2.label}]\n{stage2.stdout}")
    weights2 = _read_gsas_weights(proj, gpx, hist.name)
    weights2_ok, weights2_reason = _weights_are_plausible(
        weights2,
        has_main_phase=has_main_phase,
        min_main_phase_pct=main_guard_min_pct,
    )
    if stage2.ok and weights2_ok and (not math.isfinite(best_rwp) or stage2.rwp <= best_rwp + 2.0):
        stable_checkpoint = _checkpoint_project(proj, gpx, "stable_cleanup")
        stable_label = stage2.label
        best_rwp = stage2.rwp
    else:
        errors.append(f"{stage2.label}: {stage2.reason or weights2_reason or 'rejected by rollback guard'}")
        proj, hist = _restore_project_from_checkpoint(pm, gpx, stable_checkpoint)

    # Stage 3: optional mature transactional cell polish, only when a supplied
    # main phase exists. No-main rapid mode does not have a stable "main" anchor.
    rapid_cfg = cfg.get("rapid_hypothesis") or {}
    if has_main_phase and bool(rapid_cfg.get("final_polish_enabled", False)):
        fixed_checkpoint = stable_checkpoint
        polished_gpx = gpx.with_name(f"{gpx.stem}_polished.gpx")
        trace_path = gpx.with_name(f"{gpx.stem}_polish_trace.json")
        candidate_names = [
            str(getattr(phase, "name", ""))
            for phase in proj.phases()
            if str(getattr(phase, "name", "")) != "Main phase"
        ]
        try:
            _clone_with_lst(fixed_checkpoint, gpx)
            _save_gsas_project(proj, gpx)
            fractions_polished, rwp_polished = joint_refine_polish(
                base_gpx=str(gpx),
                out_gpx=str(polished_gpx),
                main_phase_name="Main phase",
                max_polish_cycles=int(rapid_cfg.get("final_polish_cycles", cfg.get("polish_cycles", 4))),
                refine_cell_for_all=bool(rapid_cfg.get("final_polish_refine_cell", cfg.get("polish_refine_cell", True))),
                refine_background=bool(rapid_cfg.get("final_polish_refine_background", cfg.get("polish_refine_background", True))),
                target_phase_names=candidate_names,
                polish_strategy=str(rapid_cfg.get("final_polish_strategy", cfg.get("polish_strategy", "adaptive"))),
                refine_main_cell=bool(rapid_cfg.get("final_polish_main_cell", False)),
                refine_existing_cells=bool(rapid_cfg.get("final_polish_existing_cells", False)),
                escalate_on_failure=bool(rapid_cfg.get("final_polish_escalate", cfg.get("polish_escalate_on_failure", True))),
                stabilization_cycles=int(rapid_cfg.get("final_polish_stabilization_cycles", cfg.get("polish_stabilization_cycles", 0))),
                cell_trial_cycles=int(rapid_cfg.get("final_polish_cell_trial_cycles", cfg.get("polish_cell_trial_cycles", 1))),
                final_polish_cycles=int(rapid_cfg.get("final_polish_final_cycles", cfg.get("polish_final_cycles", 0))),
                skip_fresh_lst_regen=bool(rapid_cfg.get("final_polish_skip_fresh_lst_regen", cfg.get("polish_skip_fresh_lst_regen", True))),
                trace_path=str(trace_path),
            )
            polished_ok, polished_reason = _weights_are_plausible(
                {k: float(v.get("weight_fraction_pct", 0.0)) for k, v in fractions_polished.items()},
                has_main_phase=True,
                min_main_phase_pct=main_guard_min_pct,
            )
            if polished_ok and math.isfinite(float(rwp_polished)) and float(rwp_polished) <= best_rwp + 2.0:
                _clone_with_lst(polished_gpx, gpx)
                from GSASII import GSASIIscriptable as G2sc

                pm.project = G2sc.G2Project(gpxfile=str(gpx))
                pm.main_histogram = pm.project.histograms()[0] if pm.project.histograms() else None
                proj, hist = pm.project, pm.main_histogram
                stable_checkpoint = _checkpoint_project(proj, gpx, "stable_polished")
                stable_label = "transactional_polish"
                best_rwp = float(rwp_polished)
            else:
                errors.append(f"transactional_polish: {polished_reason or 'rejected by Rwp guard'}")
                proj, hist = _restore_project_from_checkpoint(pm, gpx, stable_checkpoint)
        except Exception as exc:
            errors.append(f"transactional_polish: {exc}")
            proj, hist = _restore_project_from_checkpoint(pm, gpx, stable_checkpoint)
    elif has_main_phase:
        stdout_parts.append(
            "[transactional_polish]\n"
            "Skipped in rapid quick final ranking. Enable rapid final polish for a deeper, slower check.\n"
        )

    final_status = "ok" if stable_checkpoint is not None else "error"
    _save_gsas_project(proj, gpx)
    final_rwp = _project_rwp(hist, "\n".join(stdout_parts))
    weights = _read_gsas_weights(proj, gpx, hist.name)
    had_divergence = _stdout_has_divergence("\n".join(stdout_parts))
    if not math.isfinite(final_rwp) or final_rwp >= 99.0:
        final_status = "error"
        errors.append(f"final Rwp is {final_rwp!r}")
    return final_status, final_rwp, weights, "\n".join(stdout_parts), errors, had_divergence


def _loader_from_config(cfg: dict) -> DBLoader:
    db_cfg = _database_config_for_dataset(cfg)
    return DBLoader(CatalogPaths(
        catalog_csv=str(db_cfg["catalog_csv"]),
        cif_map_json=db_cfg.get("cif_map_json"),
        original_json=db_cfg.get("original_json"),
    ))


def _gsas_skip_row(
    row_data: dict[str, Any],
    *,
    blocked_keys: Sequence[str],
    low_weight_prune_pct: float,
    skip_threshold: int,
) -> dict[str, Any]:
    return {
        "source_scenario": "live_run",
        "rank512": int(row_data.get("rank512", 0)),
        "rank64": int(row_data.get("rank64", 0)),
        "formula_keys": str(row_data.get("formula_keys", "")),
        "phase_ids": str(row_data.get("phase_ids", "")),
        "formulas": str(row_data.get("formulas", "")),
        "space_groups": str(row_data.get("space_groups", "")),
        "cif_paths": str(row_data.get("cif_paths", "")),
        "score512": float(row_data.get("score512", 0.0)),
        "r2_512": float(row_data.get("r2_512", 0.0)),
        "sse512": float(row_data.get("sse512", 0.0)),
        "phase_coefs512": str(row_data.get("phase_coefs512", "")),
        "status": "skipped",
        "rwp": float("nan"),
        "weights_json": "{}",
        "low_weight_formula_keys": "",
        "gpx": "",
        "curve_png": "",
        "curve_csv": "",
        "errors": (
            "Skipped because this hypothesis contains "
            f"{len(blocked_keys)} already-low phase(s): {', '.join(blocked_keys)}. "
            f"Threshold is {skip_threshold}; low means < {low_weight_prune_pct:g} wt% "
            "in an earlier successful final refinement."
        ),
        "stdout_tail": "",
        "seconds": 0.0,
        "has_divergence": False,
        "has_ouch": False,
        "has_refinement_error": False,
        "has_success": False,
    }


def _validate_gsas_one_row(
    row_data: dict[str, Any],
    cfg: dict,
    dataset: dict,
    out_dir: str | Path,
    *,
    low_weight_prune_pct: float,
    db_loader: DBLoader | None = None,
    base_gpx: str | Path | None = None,
) -> dict[str, Any]:
    """Validate one rapid hypothesis in its own project directory."""
    if db_loader is None:
        db_loader = _loader_from_config(cfg)
    row = row_data
    t0 = time.perf_counter()
    run_dir = Path(out_dir) / f"live_rank512_{int(row.get('rank512', 0)):02d}_rank64_{int(row.get('rank64', 0)):02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    status = "ok"
    errors: list[str] = []
    rwp = float("nan")
    weights: dict[str, float] = {}
    curve_csv = ""
    curve_png = ""
    stdout_text = ""
    gpx = run_dir / "rapid_validate.gpx"
    had_divergence = False
    low_keys: list[str] = []
    try:
        base_path = Path(base_gpx) if base_gpx else None
        if base_path is None or not base_path.exists():
            _q_tmp, _signal_tmp, _kind_tmp, rebuilt = _extract_q_signal(cfg, dataset, run_dir / "rebuilt_base")
            base_path = Path(rebuilt)
        if not base_path.exists():
            raise RuntimeError(f"rapid base GPX not found: {base_path}")

        pm = GSASProjectManager(str(run_dir), "rapid_validate")
        if not pm.create_project(template_gpx=str(base_path)):
            raise RuntimeError(f"could not clone rapid base GPX: {base_path}")
        pm.main_histogram = pm.project.histograms()[0] if pm.project.histograms() else None
        if pm.main_histogram is None:
            raise RuntimeError("rapid base GPX has no histogram")
        existing_phases = list(pm.project.phases())
        pm.main_phase = next((p for p in existing_phases if str(getattr(p, "name", "")) == "Main phase"), None)
        if pm.main_phase is None and existing_phases:
            pm.main_phase = existing_phases[0]
        try:
            apply_safe_limits(pm.project)
        except Exception:
            pass
        phase_labels: dict[str, str] = {}
        main_phase_display = _main_phase_display_from_dataset(dataset)
        main_phase_label = str(main_phase_display.get("label") or "Main phase")
        if any(str(getattr(p, "name", "")) == "Main phase" for p in existing_phases):
            phase_labels["Main phase"] = main_phase_label

        candidate_pids = _split_pipe(row.get("phase_ids", ""))
        candidate_formula_labels = _split_pipe(row.get("formulas", ""))
        candidate_keys = _split_pipe(row.get("formula_keys", ""))
        candidate_space_groups = _split_pipe(row.get("space_groups", ""))
        candidate_cifs = _candidate_cif_paths(row, db_loader, run_dir)
        if len(candidate_formula_labels) != len(candidate_pids):
            candidate_formula_labels = candidate_keys
        if len(candidate_keys) != len(candidate_pids):
            candidate_keys = candidate_formula_labels
        phase_name_to_formula_key: dict[str, str] = {}
        for idx, (pid, fkey, label) in enumerate(zip(candidate_pids, candidate_keys, candidate_formula_labels)):
            cif = candidate_cifs[idx] if idx < len(candidate_cifs) else db_loader.ensure_cif_on_disk(pid, out_dir=str(run_dir / "cifs"), overwrite=False)
            phase_name = _safe_name(label, 24)
            existing_names = {str(getattr(p, "name", "")) for p in pm.project.phases()}
            if phase_name in existing_names:
                phase_name = _safe_name(f"{phase_name}_{idx + 1}", 24)
            if not pm.add_phase_from_cif(cif, phase_name, link_to_histogram=True):
                raise RuntimeError(f"could not add candidate CIF for {label} ({pid})")
            added_phase = pm.project.phases()[-1] if pm.project.phases() else None
            added_name = str(getattr(added_phase, "name", phase_name))
            phase_name_to_formula_key[added_name] = str(fkey)
            sg = candidate_space_groups[idx] if idx < len(candidate_space_groups) else ""
            phase_labels[added_name] = f"{label} (SG {sg})" if str(sg).strip() else str(label)
        pm.project.save(str(gpx))
        status, rwp, weights, stdout_text, errors, had_divergence = _run_stable_gsas_validation(
            pm,
            gpx,
            row,
            n_candidate_phases=len(candidate_pids),
            cfg=cfg,
        )
        if status == "ok" and math.isfinite(float(rwp)):
            for phase_name, formula_key in phase_name_to_formula_key.items():
                weight = weights.get(phase_name)
                if weight is None:
                    weight = weights.get(phase_labels.get(phase_name, ""))
                try:
                    weight_value = float(weight)
                except Exception:
                    continue
                if math.isfinite(weight_value) and weight_value < low_weight_prune_pct:
                    low_keys.append(formula_key)
        gpx = Path(pm.project.filename)
        curve_csv_path = run_dir / "curve.csv"
        curve_png_path = run_dir / "curve.png"
        if _extract_curve(gpx, curve_csv_path, curve_png_path, str(row.get("formulas", "")), phase_labels=phase_labels):
            curve_csv = str(curve_csv_path)
            curve_png = str(curve_png_path)
    except Exception as exc:
        status = "error"
        errors.append(str(exc))
        had_divergence = _stdout_has_divergence(stdout_text)
    return {
        "source_scenario": "live_run",
        "rank512": int(row.get("rank512", 0)),
        "rank64": int(row.get("rank64", 0)),
        "formula_keys": str(row.get("formula_keys", "")),
        "phase_ids": str(row.get("phase_ids", "")),
        "formulas": str(row.get("formulas", "")),
        "space_groups": str(row.get("space_groups", "")),
        "cif_paths": str(row.get("cif_paths", "")),
        "score512": float(row.get("score512", 0.0)),
        "r2_512": float(row.get("r2_512", 0.0)),
        "sse512": float(row.get("sse512", 0.0)),
        "phase_coefs512": str(row.get("phase_coefs512", "")),
        "status": status,
        "rwp": rwp,
        "weights_json": json.dumps(_display_weight_labels(weights, phase_labels), sort_keys=True),
        "low_weight_formula_keys": "|".join(sorted(set(low_keys))),
        "gpx": str(gpx),
        "curve_png": curve_png,
        "curve_csv": curve_csv,
        "errors": "; ".join(errors),
        "stdout_tail": "\n".join(stdout_text.splitlines()[-60:]),
        "seconds": float(time.perf_counter() - t0),
        "has_divergence": bool(had_divergence),
        "has_ouch": "ouch" in (stdout_text or "").lower(),
        "has_refinement_error": bool(errors),
        "has_success": status == "ok",
    }


def _resolve_gsas_workers(limit: int, rapid_cfg: dict) -> int:
    requested = int(rapid_cfg.get("gsas_parallel_workers", 0) or 0)
    if requested > 0:
        workers = requested
    else:
        cpu = os.cpu_count() or 1
        workers = min(2, max(1, cpu // 4))
    return max(1, min(int(limit), workers))


def _emit_gsas_progress(
    run_dir: Path | None,
    *,
    completed: int,
    target: int,
    processed_rows: int,
    skipped: int,
    workers: int,
) -> None:
    if run_dir is None:
        return
    pct = 80 + int(15.0 * min(completed, target) / max(target, 1))
    _write_event(
        run_dir,
        "Stage 5 Rapid",
        (
            f"Final refinement ranking {completed}/{target} refinements complete "
            f"({processed_rows} hypotheses processed, {skipped} skipped, {workers} worker(s))"
        ),
        pct,
    )


def _validate_gsas(
    rerank_df: pd.DataFrame,
    db_loader: DBLoader,
    cfg: dict,
    dataset: dict,
    out_dir: Path,
    top_n: int,
    run_dir: Path | None = None,
    base_gpx: str | Path | None = None,
    progress_csv: str | Path | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rapid_cfg = cfg.get("rapid_hypothesis") or {}
    low_weight_prune_pct = float(rapid_cfg.get("low_weight_prune_pct", 0.2))
    low_weight_skip_min_phases = max(1, int(rapid_cfg.get("low_weight_skip_min_phases", 2)))
    low_weight_formula_keys: set[str] = set()
    target_validations = int(top_n)
    workers = _resolve_gsas_workers(target_validations, rapid_cfg)
    completed_validations = 0
    launched_validations = 0
    skipped_count = 0
    row_iter = iter(rerank_df.itertuples(index=False))
    exhausted = False
    progress_path = Path(progress_csv) if progress_csv else None

    def write_progress_rows() -> None:
        if progress_path is None or not rows:
            return
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress = pd.DataFrame(rows)
        if "rwp" in progress.columns:
            progress["gsas_rwp_rank"] = progress["rwp"].rank(method="first", ascending=True, na_option="bottom").astype(int)
            sort_cols = [col for col in ["gsas_rwp_rank", "rank512"] if col in progress.columns]
            if sort_cols:
                progress = progress.sort_values(sort_cols).reset_index(drop=True)
        tmp_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
        progress.to_csv(tmp_path, index=False)
        tmp_path.replace(progress_path)

    _emit_gsas_progress(
        run_dir,
        completed=0,
        target=target_validations,
        processed_rows=0,
        skipped=0,
        workers=workers,
    )

    def row_to_data(row: Any) -> dict[str, Any]:
        if hasattr(row, "_asdict"):
            return dict(row._asdict())
        return dict(row)

    while completed_validations < target_validations and not exhausted:
        batch: list[dict[str, Any]] = []
        while len(batch) < workers and launched_validations < target_validations:
            try:
                row = next(row_iter)
            except StopIteration:
                exhausted = True
                break
            row_data = row_to_data(row)
            candidate_keys_for_skip = _split_pipe(row_data.get("formula_keys", ""))
            blocked_keys = [key for key in candidate_keys_for_skip if key in low_weight_formula_keys]
            skip_threshold = min(len(candidate_keys_for_skip), low_weight_skip_min_phases)
            if blocked_keys and len(blocked_keys) >= skip_threshold:
                rows.append(
                    _gsas_skip_row(
                        row_data,
                        blocked_keys=blocked_keys,
                        low_weight_prune_pct=low_weight_prune_pct,
                        skip_threshold=skip_threshold,
                    )
                )
                skipped_count += 1
                write_progress_rows()
                _log(f"[rapid] skipped rank512={int(row_data.get('rank512', 0))} after low-weight pruning: {blocked_keys}")
                _emit_gsas_progress(
                    run_dir,
                    completed=completed_validations,
                    target=target_validations,
                    processed_rows=len(rows),
                    skipped=skipped_count,
                    workers=workers,
                )
                continue
            batch.append(row_data)
            launched_validations += 1

        if not batch:
            continue

        _log(
            f"[rapid] launching final refinement batch: "
            f"{completed_validations + 1}-{completed_validations + len(batch)} of {target_validations} "
            f"using {workers} worker(s)"
        )
        batch_results: list[dict[str, Any]] = []
        if workers > 1 and len(batch) > 1:
            import concurrent.futures

            with concurrent.futures.ProcessPoolExecutor(max_workers=min(workers, len(batch))) as executor:
                future_map = {
                    executor.submit(
                        _validate_gsas_one_row,
                        row_data,
                        cfg,
                        dataset,
                        str(out_dir),
                        low_weight_prune_pct=low_weight_prune_pct,
                        base_gpx=str(base_gpx) if base_gpx else None,
                    ): row_data
                    for row_data in batch
                }
                for future in concurrent.futures.as_completed(future_map):
                    try:
                        result = future.result()
                    except Exception as exc:
                        row_data = future_map[future]
                        result = _gsas_skip_row(
                            row_data,
                            blocked_keys=[],
                            low_weight_prune_pct=low_weight_prune_pct,
                            skip_threshold=low_weight_skip_min_phases,
                        )
                        result["status"] = "error"
                        result["errors"] = f"Parallel GSAS-II worker failed: {exc}"
                    batch_results.append(result)
                    completed_validations += 1
                    _log(
                        f"[rapid] refined rank512={int(result.get('rank512', 0))} "
                        f"status={result.get('status')} rwp={result.get('rwp')}"
                    )
                    _emit_gsas_progress(
                        run_dir,
                        completed=completed_validations,
                        target=target_validations,
                        processed_rows=len(rows) + len(batch_results),
                        skipped=skipped_count,
                        workers=workers,
                    )
        else:
            for row_data in batch:
                result = _validate_gsas_one_row(
                    row_data,
                    cfg,
                    dataset,
                    out_dir,
                    low_weight_prune_pct=low_weight_prune_pct,
                    db_loader=db_loader,
                    base_gpx=base_gpx,
                )
                batch_results.append(result)
                completed_validations += 1
                _log(
                    f"[rapid] refined rank512={int(result.get('rank512', 0))} "
                    f"status={result.get('status')} rwp={result.get('rwp')}"
                )
                _emit_gsas_progress(
                    run_dir,
                    completed=completed_validations,
                    target=target_validations,
                    processed_rows=len(rows) + len(batch_results),
                    skipped=skipped_count,
                    workers=workers,
                )

        batch_results.sort(key=lambda item: int(item.get("rank512", 0)))
        rows.extend(batch_results)
        write_progress_rows()
        for result in batch_results:
            if str(result.get("status")) != "ok":
                continue
            try:
                rwp = float(result.get("rwp", float("nan")))
            except Exception:
                rwp = float("nan")
            if not math.isfinite(rwp):
                continue
            for formula_key in _split_pipe(result.get("low_weight_formula_keys", "")):
                low_weight_formula_keys.add(formula_key)
                _log(
                    f"[rapid] low-weight note: {formula_key} refined below "
                    f"{low_weight_prune_pct:g} wt%"
                )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["gsas_rwp_rank"] = out["rwp"].rank(method="first", ascending=True, na_option="bottom").astype(int)
        out = out.sort_values(["gsas_rwp_rank", "rank512"]).reset_index(drop=True)
    return out


def _write_event(
    run_dir: Path,
    stage: str,
    message: str,
    percent: int,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    log_dir = run_dir / "Technical" / "Logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    event = {"stage": stage, "message": message, "percent": int(percent), "time": time.time()}
    if metrics:
        event["metrics"] = metrics
    with (log_dir / "run_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def _write_failed_manifest_from_cli(exc: BaseException, argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config")
    parser.add_argument("--dataset")
    try:
        args, _ = parser.parse_known_args(argv)
    except Exception:
        return
    if not args.config:
        return
    try:
        cfg_path = Path(args.config)
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        run_dir = Path(cfg.get("work_root") or cfg_path.parent).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        tb = traceback.format_exc()
        manifest = {
            "status": "failed",
            "analysis_mode": "rapid_hypothesis",
            "dataset": args.dataset,
            "returncode": 1,
            "error": str(exc),
            "failed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "traceback_tail": tb[-12000:],
            "rapid_results": str(run_dir / "rapid_results"),
        }
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        _write_event(
            run_dir,
            "Failed Rapid",
            f"Rapid Hypothesis Mode failed: {exc}",
            100,
            metrics={"error": str(exc), "returncode": 1},
        )
    except Exception:
        pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args(argv)

    t_all = time.perf_counter()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset = next((d for d in cfg.get("datasets", []) if str(d.get("name")) == str(args.dataset)), None)
    if dataset is None:
        raise SystemExit(f"Dataset not found in config: {args.dataset}")
    run_dir = Path(cfg.get("work_root") or Path(args.config).parent).resolve()
    rapid_root = run_dir / "rapid_results"
    scenario_dir = rapid_root / "nudge" / "live_run"
    gsas_out = rapid_root / "live_run" / "gsas"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    rapid_root.mkdir(parents=True, exist_ok=True)
    partial_gsas_csv = rapid_root / "all_gsas_validation_summary.partial.csv"
    if partial_gsas_csv.exists():
        partial_gsas_csv.unlink()

    _log("[rapid] Starting live Rapid Hypothesis Mode")
    _write_event(run_dir, "Stage 0 Rapid", "Rapid setup started", 5)
    db_cfg = _database_config_for_dataset(cfg, dataset)
    db_loader = DBLoader(CatalogPaths(
        catalog_csv=str(db_cfg["catalog_csv"]),
        cif_map_json=db_cfg.get("cif_map_json"),
        original_json=db_cfg.get("original_json"),
    ))
    profile_ctx = _load_profiles64_metadata(str(db_cfg["profiles_dir"]))
    doublet_spec = resolve_xray_doublet_spec(
        cfg,
        dataset=dataset,
        instprm_path=dataset.get("instprm_path"),
        stage4=cfg.get("stage4") or {},
    )
    if doublet_spec.enabled:
        _log(f"[rapid] PXRD doublet correction active: {describe_doublet(doublet_spec)}")

    _log("[rapid] Importing diffraction signal")
    t_stage = time.perf_counter()
    q, signal, signal_kind, signal_gpx = _extract_q_signal(
        cfg,
        dataset,
        scenario_dir,
        db_loader=db_loader,
        xray_doublet_config=doublet_spec.to_dict(),
        run_dir=run_dir,
    )
    main_phase_display = _main_phase_display_from_dataset(dataset)
    signal_seconds = float(time.perf_counter() - t_stage)
    mask = np.isfinite(q) & np.isfinite(signal)
    q = np.asarray(q[mask], dtype=float)
    signal = np.asarray(signal[mask], dtype=float)
    if q.size < 20:
        raise RuntimeError("Rapid mode could not extract enough Q/intensity points from the uploaded data")
    np.savez_compressed(scenario_dir / "target_signal.npz", Q=q, signal=signal)
    main_shadow_q = _main_shadow_peaks_from_gpx(signal_gpx, cfg, dataset)
    if main_shadow_q:
        _log(f"[rapid] main-shadow anchor peaks available: {len(main_shadow_q)} anchor peak(s)")
    _write_event(run_dir, "Stage 1 Rapid", "Signal imported", 20)

    magnetic_summary: Dict[str, Any] = {"enabled": False, "status": "skipped"}
    magnetic_cfg = cfg.get("magnetic_precheck") or {}
    if bool(magnetic_cfg.get("enabled", False)):
        _log("[rapid] Running magnetic residual indexing precheck")
        _write_event(run_dir, "Magnetic Precheck", "Magnetic residual indexing started", 22)
        t_mag = time.perf_counter()
        try:
            magnetic_summary = run_magnetic_precheck(
                q=q,
                residual=signal,
                main_cif=dataset.get("main_cif"),
                refined_gpx=signal_gpx,
                phase_name="Main phase",
                out_dir=run_dir / "Magnetic_Precheck",
                config=magnetic_cfg,
            )
            magnetic_summary["main_phase"] = main_phase_display
            magnetic_summary["wall_seconds"] = float(time.perf_counter() - t_mag)
            _log(
                "[rapid] magnetic precheck: "
                f"evidence={magnetic_summary.get('evidence')} best_k={magnetic_summary.get('best_k')} "
                f"score={magnetic_summary.get('best_score')}"
            )
            _write_event(
                run_dir,
                "Magnetic Precheck",
                f"Magnetic evidence: {magnetic_summary.get('evidence', 'unknown')}",
                25,
                metrics={
                    "evidence": magnetic_summary.get("evidence"),
                    "best_k": magnetic_summary.get("best_k"),
                    "score": magnetic_summary.get("best_score"),
                },
            )
        except Exception as exc:
            magnetic_summary = {
                "enabled": True,
                "status": "failed",
                "evidence": "not_available",
                "reason": str(exc),
                "main_phase": main_phase_display,
            }
            (run_dir / "Magnetic_Precheck").mkdir(parents=True, exist_ok=True)
            (run_dir / "Magnetic_Precheck" / "magnetic_precheck_summary.json").write_text(
                json.dumps(magnetic_summary, indent=2),
                encoding="utf-8",
            )
            _log(f"[rapid][WARN] magnetic precheck failed: {exc}")

    _log("[rapid] Loading coarse candidate profiles")
    t_stage = time.perf_counter()
    ids, formulas, sgs, profiles = _candidate_matrix(cfg, db_loader, profile_ctx)
    profiles = apply_doublet_to_profiles(
        profiles,
        profile_ctx,
        doublet_spec,
        apply_key="apply_to_64_ml_input",
    )
    if profiles.size == 0:
        raise RuntimeError("No candidate profiles remain after the selected chemistry/library filters")
    y64 = _histogram_signal(q, signal, profile_ctx)
    np.savez_compressed(scenario_dir / "target_64.npz", y64=y64)

    rapid_cfg = cfg.get("rapid_hypothesis") or {}
    max_shift = float(rapid_cfg.get("max_shift_bins", 1.0))
    shift_step = float(rapid_cfg.get("shift_step_bins", 1.0))
    shifts = np.arange(-max_shift, max_shift + 1e-6, shift_step, dtype=np.float32).tolist()
    if 0.0 not in shifts:
        shifts.append(0.0)
    depth = int(rapid_cfg.get("beam_depth", 3 if not dataset.get("main_cif") else 2))
    beam64, states, residual_rank = _run_beam64_search(
        y64=y64,
        profiles=profiles,
        ids=ids,
        formulas=formulas,
        sgs=sgs,
        shifts=shifts,
        rapid_cfg=rapid_cfg,
        depth=depth,
    )
    if beam64.empty or states.empty:
        raise RuntimeError("Rapid coarse search produced no hypotheses")
    beam64.to_csv(scenario_dir / "beam64_initial_input_top.csv", index=False)
    near_elemental_banned_phase_ids: set[str] = set()
    near_elemental_filter_audit: dict[str, Any] = {}
    beam64, states, residual_rank, near_elemental_banned_phase_ids, near_elemental_filter_audit = _apply_near_elemental_duplicate_filter(
        beam64=beam64,
        states=states,
        residual_rank=residual_rank,
        y64=y64,
        profiles=profiles,
        ids=ids,
        formulas=formulas,
        sgs=sgs,
        shifts=shifts,
        rapid_cfg=rapid_cfg,
        depth=depth,
        out_dir=scenario_dir,
    )
    if beam64.empty or states.empty:
        raise RuntimeError("Rapid coarse search produced no hypotheses after near-elemental duplicate filtering")
    beam64.to_csv(scenario_dir / "beam64_input_top.csv", index=False)
    states.to_csv(scenario_dir / "beam_states.csv", index=False)
    residual_rank.to_csv(scenario_dir / "residual_rank_history.csv", index=False)
    search64_seconds = float(time.perf_counter() - t_stage)
    _log(f"[rapid] coarse search complete: {len(beam64)} hypotheses")
    _write_event(run_dir, "Stage 2 Rapid", "Coarse hypotheses generated", 45)

    t_stage = time.perf_counter()
    rapid_shadow_filtered: set[str] = set()
    shadow_banned_phase_ids: set[str] = set()
    shadow_refill_audits: list[dict[str, Any]] = []
    max_shadow_refills = int(rapid_cfg.get("main_shadow_refill_attempts", 2))
    refill_attempt = 0
    while True:
        attempt_label = "initial" if refill_attempt == 0 else f"refill_{refill_attempt}"
        nudge_df = _nudge_unique_phases(beam64, db_loader, q, signal, cfg, scenario_dir)
        nudge_df.to_csv(scenario_dir / "nudge_results.csv", index=False)
        if refill_attempt:
            nudge_df.to_csv(scenario_dir / f"nudge_results_{attempt_label}.csv", index=False)
        beam64_filtered, nudge_df_filtered, rapid_shadow_filtered, shadow_audit = _filter_rapid_main_shadow_after_nudge(
            beam64,
            nudge_df,
            db_loader,
            q,
            signal,
            cfg,
            scenario_dir,
            main_shadow_q=main_shadow_q,
            doublet_spec=doublet_spec.to_dict(),
        )
        shadow_audit = dict(shadow_audit or {})
        shadow_audit["attempt"] = attempt_label
        shadow_refill_audits.append(shadow_audit)
        try:
            (scenario_dir / f"main_shadow_nudge_filter_{attempt_label}.json").write_text(
                json.dumps(shadow_audit, indent=2, default=str),
                encoding="utf-8",
            )
            records = list(shadow_audit.get("records") or [])
            if records:
                pd.DataFrame(records).to_csv(scenario_dir / f"main_shadow_nudge_filter_{attempt_label}.csv", index=False)
        except Exception:
            pass

        if (
            bool(shadow_audit.get("fail_open"))
            and shadow_audit.get("filtered_phase_ids")
            and str(shadow_audit.get("fail_open_reason") or "") != "main_anchor_not_reliable"
        ):
            newly_flagged = {str(pid) for pid in (shadow_audit.get("filtered_phase_ids") or []) if str(pid)}
            new_bans = newly_flagged - shadow_banned_phase_ids
            shadow_banned_phase_ids.update(newly_flagged)
            _log(
                "[rapid] main-shadow refill requested; banning "
                f"{len(shadow_banned_phase_ids)} shadow phase(s): {', '.join(sorted(shadow_banned_phase_ids))}"
            )
            _write_event(
                run_dir,
                "Stage 3 Rapid",
                "Refilling coarse hypotheses without main-phase lookalikes",
                58,
                metrics={
                    "attempt": refill_attempt + 1,
                    "banned_phase_ids": sorted(shadow_banned_phase_ids),
                },
            )
            if refill_attempt >= max_shadow_refills:
                raise RuntimeError(
                    "Main-phase lookalike filter exhausted the rapid shortlist after "
                    f"{max_shadow_refills} refill attempt(s). Flagged phase IDs: "
                    f"{', '.join(sorted(shadow_banned_phase_ids))}"
                )
            if not new_bans:
                raise RuntimeError(
                    "Main-phase lookalike filter repeatedly flagged the same phases and could not "
                    "produce an independent rapid shortlist."
                )
            refill_attempt += 1
            beam64, refill_states, refill_residual_rank = _run_beam64_search(
                y64=y64,
                profiles=profiles,
                ids=ids,
                formulas=formulas,
                sgs=sgs,
                shifts=shifts,
                rapid_cfg=rapid_cfg,
                depth=depth,
                excluded_phase_ids=set(shadow_banned_phase_ids) | set(near_elemental_banned_phase_ids),
            )
            if beam64.empty or refill_states.empty:
                raise RuntimeError(
                    "Main-phase lookalike filter removed the current shortlist and no replacement "
                    "hypotheses could be generated after excluding flagged phases."
                )
            beam64.to_csv(scenario_dir / "beam64_input_top.csv", index=False)
            beam64.to_csv(scenario_dir / f"beam64_refill_without_main_shadow_attempt_{refill_attempt}.csv", index=False)
            refill_states.to_csv(scenario_dir / f"beam_states_refill_without_main_shadow_attempt_{refill_attempt}.csv", index=False)
            refill_residual_rank.to_csv(
                scenario_dir / f"residual_rank_history_refill_without_main_shadow_attempt_{refill_attempt}.csv",
                index=False,
            )
            _log(
                "[rapid] rebuilt coarse shortlist without main-shadow phases: "
                f"{len(beam64)} replacement hypotheses"
            )
            continue

        beam64, nudge_df = beam64_filtered, nudge_df_filtered
        break

    if rapid_shadow_filtered:
        beam64.to_csv(scenario_dir / "beam64_input_top.csv", index=False)
        nudge_df.to_csv(scenario_dir / "nudge_results.csv", index=False)
        _write_event(
            run_dir,
            "Stage 3 Rapid",
            f"Main-phase lookalike filter removed {len(rapid_shadow_filtered)} phase(s)",
            64,
            metrics={"filtered_phase_ids": sorted(rapid_shadow_filtered)},
        )
    if beam64.empty:
        raise RuntimeError("Rapid mode has no hypotheses available after lattice nudging")
    nudge_seconds = float(time.perf_counter() - t_stage)
    _write_event(run_dir, "Stage 3 Rapid", "Lattice nudge completed", 65)

    t_stage = time.perf_counter()
    rerank = _rerank512(
        beam64,
        nudge_df,
        db_loader,
        q,
        signal,
        cfg,
        profile_ctx,
        scenario_dir,
        doublet_spec=doublet_spec.to_dict(),
    )
    rerank.to_csv(scenario_dir / "reranked_512_after_radar_nudge.csv", index=False)
    rerank512_seconds = float(time.perf_counter() - t_stage)
    _log(f"[rapid] high-resolution pattern scoring complete: {len(rerank)} hypotheses")
    _write_event(run_dir, "Stage 4 Rapid", "High-resolution pattern scoring completed", 80)

    t_stage = time.perf_counter()
    gsas_limit = int(rapid_cfg.get("gsas_validation_limit", 10))
    gsas_df = _validate_gsas(
        rerank,
        db_loader,
        cfg,
        dataset,
        gsas_out,
        top_n=gsas_limit,
        run_dir=run_dir,
        base_gpx=signal_gpx,
        progress_csv=partial_gsas_csv,
    )
    gsas_df.to_csv(rapid_root / "all_gsas_validation_summary.csv", index=False)
    gsas_wall_seconds = float(time.perf_counter() - t_stage)
    _write_event(run_dir, "Stage 5 Rapid", "Final refinement ranking completed", 95)

    timings = {
        "total_seconds": float(time.perf_counter() - t_all),
        "signal_seconds": signal_seconds,
        "search64_seconds": search64_seconds,
        "nudge_seconds": nudge_seconds,
        "rerank512_seconds": rerank512_seconds,
        "nudge_512_total_seconds": float(nudge_seconds + rerank512_seconds),
        "gsas_wall_seconds": gsas_wall_seconds,
        "gsas_total_seconds": float(gsas_df["seconds"].sum()) if not gsas_df.empty else 0.0,
    }
    main_phase_prenudge_summary: Dict[str, Any] = {}
    main_phase_cleanup_summary: Dict[str, Any] = {}
    main_phase_anchor_summary: Dict[str, Any] = {}
    try:
        prenudge_path = scenario_dir / "main_phase_prenudge.json"
        if prenudge_path.exists():
            main_phase_prenudge_summary = json.loads(prenudge_path.read_text(encoding="utf-8"))
    except Exception as exc:
        main_phase_prenudge_summary = {"error": str(exc)}
    try:
        cleanup_path = scenario_dir / "main_phase_cleanup.json"
        if cleanup_path.exists():
            main_phase_cleanup_summary = json.loads(cleanup_path.read_text(encoding="utf-8"))
    except Exception as exc:
        main_phase_cleanup_summary = {"error": str(exc)}
    try:
        anchor_path = scenario_dir / "main_phase_anchor_final.json"
        if anchor_path.exists():
            main_phase_anchor_summary = json.loads(anchor_path.read_text(encoding="utf-8"))
    except Exception as exc:
        main_phase_anchor_summary = {"error": str(exc)}
    summary = {
        "live_run": {
            "label": f"Live run: {args.dataset}",
            "signal_kind": signal_kind,
            "signal_gpx": signal_gpx,
            "main_phase": main_phase_display,
            "main_shadow_q": [float(qv) for qv in main_shadow_q],
            "main_shadow_banned_phase_ids": sorted(shadow_banned_phase_ids),
            "main_shadow_refill_attempts": int(refill_attempt),
            "near_elemental_banned_phase_ids": sorted(near_elemental_banned_phase_ids),
            "near_elemental_duplicate_filter": near_elemental_filter_audit,
            "main_shadow_filter_audits": [
                {
                    "attempt": audit.get("attempt"),
                    "fail_open": bool(audit.get("fail_open", False)),
                    "filtered_phase_ids": list(audit.get("filtered_phase_ids") or []),
                    "effective_filtered_phase_ids": list(audit.get("effective_filtered_phase_ids") or []),
                    "fail_open_reason": audit.get("fail_open_reason"),
                    "main_anchor_reliable": audit.get("main_anchor_reliable"),
                    "main_anchor_reliability_reason": audit.get("main_anchor_reliability_reason"),
                }
                for audit in shadow_refill_audits
            ],
            "main_phase_prenudge": main_phase_prenudge_summary,
            "main_phase_cleanup": main_phase_cleanup_summary,
            "main_phase_anchor": main_phase_anchor_summary,
            "magnetic_precheck": magnetic_summary,
            "candidate_count": int(len(ids)),
            "target_rank64": "-",
            "target_rank_loo": "-",
            "target_rank512": "-",
            "target_rank_gsas": "-",
            "timings": timings,
        }
    }
    (rapid_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {
        "status": "complete",
        "analysis_mode": "rapid_hypothesis",
        "metrics": {
            "hypotheses_64": int(len(beam64)),
            "hypotheses_512": int(len(rerank)),
            "gsas_validations": int(len(gsas_df)),
            "best_rwp": None if gsas_df.empty else float(gsas_df["rwp"].min()),
            "main_shadow_banned_phase_ids": sorted(shadow_banned_phase_ids),
            "main_shadow_refill_attempts": int(refill_attempt),
            "near_elemental_banned_phase_ids": sorted(near_elemental_banned_phase_ids),
        },
        "rapid_results": str(rapid_root),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_event(run_dir, "Complete Rapid", "Rapid Hypothesis Mode finished", 100)
    _log("[rapid] Rapid Hypothesis Mode finished successfully")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        _write_failed_manifest_from_cli(exc, sys.argv[1:])
        traceback.print_exc()
        raise SystemExit(1)



