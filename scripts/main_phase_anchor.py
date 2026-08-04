"""Shared main-phase anchoring helpers for full and rapid RADAR-PD runs.

The supplied main CIF is an anchor, not just another candidate. These helpers
first assess whether the normal GSAS-II main-phase fit actually supports the
strongest observed peaks. If it does not, they run the same symmetry-aware
lattice nudger used by the main pipeline and only adopt the nudged CIF when the
refined fit has clear evidence of improvement.
"""

from __future__ import annotations

import math
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from auto_background_points import coerce_auto_background_params, estimate_background
except Exception:  # pragma: no cover - import fallback for unusual launch paths
    coerce_auto_background_params = None
    estimate_background = None


@dataclass
class MainPhaseAnchorResult:
    pm: Any
    refiner: Any
    refinement_result: Any
    main_cif: str
    audit: Dict[str, Any]
    context: Dict[str, Any] = field(default_factory=dict)


def _merged_named_cfg(
    top_cfg: Dict[str, Any],
    ds_cfg: Dict[str, Any],
    key: str,
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = dict(defaults)
    top_value = (top_cfg or {}).get(key, {})
    ds_value = (ds_cfg or {}).get(key, {})
    if isinstance(top_value, dict):
        cfg.update(top_value)
    if isinstance(ds_value, dict):
        cfg.update(ds_value)
    return cfg


def _stage4_frac_window(s4_cfg: Dict[str, Any], default: float = 0.01) -> float:
    if "frac_window" in (s4_cfg or {}):
        return float(s4_cfg.get("frac_window", default))
    if "len_tol_pct" in (s4_cfg or {}):
        return float(s4_cfg.get("len_tol_pct", default * 100.0)) / 100.0
    return float(default)


def _stage4_angle_window(s4_cfg: Dict[str, Any], default: float = 1.0) -> float:
    if "angle_window_deg" in (s4_cfg or {}):
        return float(s4_cfg.get("angle_window_deg", default))
    if "ang_tol_deg" in (s4_cfg or {}):
        return float(s4_cfg.get("ang_tol_deg", default))
    return float(default)


def main_prenudge_cfg(
    top_cfg: Dict[str, Any],
    ds_cfg: Dict[str, Any],
    s4_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    s4_cfg = s4_cfg or {}
    return _merged_named_cfg(
        top_cfg,
        ds_cfg,
        "main_phase_prenudge",
        {
            "enabled": True,
            "apply_only_user_main": True,
            "top_observed_peaks": 8,
            "top_calculated_peaks": 30,
            "peak_match_tolerance_q": 0.035,
            "min_peak_support": 0.50,
            "trigger_rwp": 18.0,
            "hard_rwp": 35.0,
            "min_rwp_for_strongest_trigger": 8.0,
            "min_rwp_for_peak_support_trigger": 8.0,
            "strongest_barely_supported_fraction": 0.75,
            "min_peak_prominence_fraction": 0.04,
            "reps": max(8, min(int(s4_cfg.get("reps", 20)), 30)),
            "samples": max(500, min(int(s4_cfg.get("samples", 2000)), 3000)),
            "frac_window": min(_stage4_frac_window(s4_cfg, 0.01), 0.01),
            "angle_window_deg": min(_stage4_angle_window(s4_cfg, 1.0), 1.0),
            "score_q_max": float(s4_cfg.get("score_q_max", 8.0)),
            "accept_rwp_worsen": 0.50,
            "accept_min_support_gain": 0.10,
            "accept_min_nudge_score": 0.02,
            # A poor one-phase main fit is expected for many real multiphase
            # samples. Treat this as an uncertainty flag for downstream
            # filtering, not as a hard stop.
            "fail_unresolved_main": False,
        },
    )


def main_phase_guard_cfg(top_cfg: Dict[str, Any], ds_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return _merged_named_cfg(
        top_cfg,
        ds_cfg,
        "main_phase_guard",
        {
            "enabled": True,
            "apply_only_user_main": True,
            "min_weight_pct": 20.0,
        },
    )


def main_phase_cleanup_cfg(top_cfg: Dict[str, Any], ds_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return _merged_named_cfg(
        top_cfg,
        ds_cfg,
        "main_phase_cleanup",
        {
            "enabled": False,
            "apply_only_user_main": True,
            "refine_u_iso": False,
            "refine_positions": False,
            "cycles": 1,
            "accept_rwp_worsen": 0.15,
            "min_rwp_improvement": 0.05,
            "max_position_shift": 0.15,
            "min_u_iso": 0.0,
            "max_u_iso": 0.20,
        },
    )


def main_phase_shadow_cfg(top_cfg: Dict[str, Any], ds_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return _merged_named_cfg(
        top_cfg,
        ds_cfg,
        "main_phase_shadow",
        {
            "enabled": True,
            "top_main_peaks": 8,
            "top_candidate_peaks": 10,
            "peak_match_tolerance_q": 0.040,
            "min_target_prominence_fraction": 0.03,
            "nudge_filter_enabled": True,
            "filter_top_main_peaks": 5,
            "filter_top_candidate_peaks": 5,
            "filter_min_overlap_count": 3,
            "filter_min_overlap_fraction": 0.60,
            "filter_min_shadow_intensity_fraction": 0.60,
            "filter_max_unique_supported_count": 1,
            "filter_max_unique_supported_fraction": 0.25,
            "refill_attempts": 2,
            "require_reliable_main_rwp_max": 25.0,
            "require_reliable_main_peak_support": 0.55,
        },
    )


def main_anchor_reliability_from_fit_audit(
    fit_audit: Dict[str, Any] | None,
    cfg: Dict[str, Any] | None = None,
) -> Tuple[bool, str]:
    """Return whether a main-phase fit is safe to use as a subtraction anchor."""
    cfg = cfg or {}
    audit = fit_audit or {}
    if not audit:
        return True, "not_assessed"
    if int(audit.get("points", 0) or 0) and int(audit.get("points", 0) or 0) < 50:
        return False, "too_few_points"
    triggered = bool(audit.get("triggered", False))
    reason = str(audit.get("reason") or "not_evaluated")
    try:
        rwp = float(audit.get("rwp", 0.0) or 0.0)
    except Exception:
        rwp = float("nan")
    try:
        support = float(audit.get("weighted_peak_support", audit.get("peak_support_fraction", 0.0)) or 0.0)
    except Exception:
        support = 0.0
    reliable_rwp = float(cfg.get("require_reliable_main_rwp_max", 25.0))
    reliable_support = float(cfg.get("require_reliable_main_peak_support", 0.55))
    if triggered:
        return False, f"{reason}; rwp={rwp:.3g}; support={support:.3g}"
    if math.isfinite(rwp) and rwp > reliable_rwp:
        return False, f"rwp_above_reliable_limit; rwp={rwp:.3g}; support={support:.3g}"
    if support < reliable_support:
        return False, f"peak_support_below_reliable_limit; rwp={rwp:.3g}; support={support:.3g}"
    return True, f"{reason}; rwp={rwp:.3g}; support={support:.3g}"


def main_anchor_reliability_from_audit(
    anchor_audit: Dict[str, Any] | None,
    cfg: Dict[str, Any] | None = None,
) -> Tuple[bool, str]:
    """Return whether a pre-nudge audit ended with a trustworthy main anchor."""
    audit = anchor_audit or {}
    if bool(audit.get("adopted", False)):
        nudged_fit = audit.get("nudged_fit") or audit.get("normal_fit") or {}
        reliable, reason = main_anchor_reliability_from_fit_audit(nudged_fit, cfg)
        if reliable:
            return True, "adopted_prenudge"
        return False, f"adopted_prenudge_but_unreliable: {reason}"
    normal_fit = audit.get("normal_fit") or {}
    if not normal_fit:
        return True, str(audit.get("reason") or "not_assessed")
    return main_anchor_reliability_from_fit_audit(normal_fit, cfg)

def phase_weight_fraction(fractions: Dict[str, Dict[str, float]], pid: str) -> Optional[float]:
    try:
        wf = float(fractions.get(pid, {}).get("weight_fraction_pct", 0.0))
    except Exception:
        return None
    if not math.isfinite(wf) or wf < -1e-6 or wf > 1000.0:
        return None
    return max(0.0, wf)


def main_phase_guard_violation(
    fractions: Dict[str, Dict[str, float]],
    main_phase_name: str,
    top_cfg: Dict[str, Any],
    ds_cfg: Dict[str, Any],
    *,
    user_supplied_main: bool,
) -> Tuple[bool, Optional[float], Dict[str, Any]]:
    cfg = main_phase_guard_cfg(top_cfg, ds_cfg)
    if not bool(cfg.get("enabled", True)):
        return False, None, cfg
    if bool(cfg.get("apply_only_user_main", True)) and not user_supplied_main:
        return False, None, cfg
    main_wf = phase_weight_fraction(fractions, main_phase_name)
    if main_wf is None:
        return False, None, cfg
    return bool(main_wf < float(cfg.get("min_weight_pct", 20.0))), main_wf, cfg


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


def select_top_peaks(
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
    ys = _smooth_1d(y, smooth_width)
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


def main_shadow_peaks_from_arrays(
    q: np.ndarray,
    calculated_signal: np.ndarray,
    cfg: Dict[str, Any],
) -> List[float]:
    peaks = select_top_peaks(
        q,
        np.maximum(np.asarray(calculated_signal, dtype=float), 0.0),
        top_n=int(cfg.get("top_main_peaks", 8)),
        min_rel_height=float(cfg.get("min_peak_prominence_fraction", 0.03)),
        min_sep_q=float(cfg.get("peak_match_tolerance_q", 0.040)),
    )
    return [float(p["q"]) for p in peaks]


def _signal_support_at_q(
    q_grid: np.ndarray,
    signal: np.ndarray,
    q0: float,
    half_width: float,
    threshold: float,
) -> bool:
    q_grid = np.asarray(q_grid, dtype=float).ravel()
    signal = np.asarray(signal, dtype=float).ravel()
    n = min(q_grid.size, signal.size)
    if n <= 0:
        return False
    q_grid = q_grid[:n]
    signal = np.maximum(signal[:n], 0.0)
    mask = np.isfinite(q_grid) & np.isfinite(signal) & (np.abs(q_grid - float(q0)) <= float(half_width))
    if not np.any(mask):
        return False
    return bool(float(np.nanmax(signal[mask])) >= float(threshold))


def compute_main_shadow_metrics(
    candidate_q: np.ndarray,
    candidate_i: np.ndarray,
    main_peak_q: List[float] | np.ndarray,
    target_q: np.ndarray,
    target_signal: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Quantify whether a candidate's evidence sits mostly on main-phase peaks."""
    cand_q = np.asarray(candidate_q, dtype=float).ravel()
    cand_i = np.asarray(candidate_i, dtype=float).ravel()
    n = min(cand_q.size, cand_i.size)
    main_q = np.asarray(main_peak_q, dtype=float).ravel()
    result: Dict[str, Any] = {
        "enabled": bool(cfg.get("enabled", True)),
        "main_shadow_fraction": 0.0,
        "unique_peak_fraction": 0.0,
        "unique_supported_fraction": 0.0,
        "candidate_peak_count": 0,
        "main_peak_count": int(main_q.size),
    }
    if not bool(cfg.get("enabled", True)) or n <= 0 or main_q.size <= 0:
        result["reason"] = "disabled_or_no_peaks"
        return result

    cand_q = cand_q[:n]
    cand_i = np.maximum(cand_i[:n], 0.0)
    finite = np.isfinite(cand_q) & np.isfinite(cand_i) & (cand_i > 0)
    if int(finite.sum()) <= 0:
        result["reason"] = "no_candidate_intensity"
        return result
    cand_q = cand_q[finite]
    cand_i = cand_i[finite]
    order = np.argsort(cand_i)[::-1][: max(1, int(cfg.get("top_candidate_peaks", 10)))]
    cand_q = cand_q[order]
    cand_i = cand_i[order]
    total_i = float(np.sum(cand_i))
    if total_i <= 0:
        result["reason"] = "zero_candidate_intensity"
        return result

    tol = float(cfg.get("peak_match_tolerance_q", 0.040))
    target = np.maximum(np.asarray(target_signal, dtype=float).ravel(), 0.0)
    target_hi = float(np.nanmax(target)) if target.size else 0.0
    target_threshold = max(1e-12, target_hi * float(cfg.get("min_target_prominence_fraction", 0.03)))
    overlap_i = 0.0
    unique_i = 0.0
    unique_supported_i = 0.0
    peak_rows: List[Dict[str, Any]] = []
    for qv, iv in zip(cand_q, cand_i):
        nearest_delta = float(np.nanmin(np.abs(main_q - float(qv)))) if main_q.size else float("inf")
        overlaps_main = bool(nearest_delta <= tol)
        target_supported = False
        if overlaps_main:
            overlap_i += float(iv)
        else:
            unique_i += float(iv)
            target_supported = _signal_support_at_q(target_q, target_signal, float(qv), tol, target_threshold)
            if target_supported:
                unique_supported_i += float(iv)
        peak_rows.append({
            "q": float(qv),
            "relative_intensity": float(iv),
            "nearest_main_delta_q": None if not math.isfinite(nearest_delta) else nearest_delta,
            "overlaps_main": overlaps_main,
            "target_supported": bool(target_supported),
        })

    shadow_fraction = float(overlap_i / max(total_i, 1e-12))
    unique_fraction = float(unique_i / max(total_i, 1e-12))
    unique_supported_fraction = float(unique_supported_i / max(total_i, 1e-12))
    result.update({
        "reason": "evaluated",
        "main_shadow_fraction": shadow_fraction,
        "unique_peak_fraction": unique_fraction,
        "unique_supported_fraction": unique_supported_fraction,
        "candidate_peak_count": int(cand_q.size),
        "candidate_peaks": peak_rows,
    })
    return result


def main_shadow_filter_decision(
    candidate_q: np.ndarray,
    candidate_i: np.ndarray,
    main_peak_q: List[float] | np.ndarray,
    target_q: np.ndarray,
    target_signal: np.ndarray,
    cfg: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    """Return whether a candidate should be filtered as a main-phase lookalike.

    It only filters when the candidate's strongest peaks mostly sit on the
    strongest main-phase peaks and there is little independently supported
    evidence left in the residual.
    """
    if not bool(cfg.get("enabled", True)) or not bool(cfg.get("nudge_filter_enabled", True)):
        return False, {"enabled": bool(cfg.get("enabled", True)), "filtered": False, "filter_reason": "disabled"}

    local_cfg = dict(cfg or {})
    local_cfg["top_candidate_peaks"] = int(local_cfg.get("filter_top_candidate_peaks", 5))
    main_q = list(np.asarray(main_peak_q, dtype=float).ravel())
    main_q = main_q[: max(1, int(local_cfg.get("filter_top_main_peaks", 5)))]
    metrics = compute_main_shadow_metrics(
        candidate_q,
        candidate_i,
        main_q,
        target_q,
        target_signal,
        local_cfg,
    )
    peaks = list(metrics.get("candidate_peaks") or [])
    candidate_count = max(1, int(metrics.get("candidate_peak_count", len(peaks)) or len(peaks) or 1))
    overlap_count = int(sum(1 for p in peaks if bool(p.get("overlaps_main"))))
    unique_supported_count = int(
        sum(1 for p in peaks if (not bool(p.get("overlaps_main"))) and bool(p.get("target_supported")))
    )
    main_anchor_count = max(1, len(main_q))
    min_overlap_count = min(
        int(local_cfg.get("filter_min_overlap_count", 3)),
        main_anchor_count,
        candidate_count,
    )
    overlap_fraction = float(overlap_count / candidate_count)
    shadow_intensity_fraction = float(metrics.get("main_shadow_fraction", 0.0) or 0.0)
    unique_supported_fraction = float(metrics.get("unique_supported_fraction", 0.0) or 0.0)
    filtered = (
        str(metrics.get("reason")) == "evaluated"
        and overlap_count >= max(1, min_overlap_count)
        and (
            overlap_fraction >= float(local_cfg.get("filter_min_overlap_fraction", 0.60))
            or shadow_intensity_fraction >= float(local_cfg.get("filter_min_shadow_intensity_fraction", 0.60))
        )
        and unique_supported_count <= int(local_cfg.get("filter_max_unique_supported_count", 1))
        and unique_supported_fraction <= float(local_cfg.get("filter_max_unique_supported_fraction", 0.25))
    )
    if filtered:
        reason = "strong_candidate_peaks_overlap_main_phase"
    elif str(metrics.get("reason")) != "evaluated":
        reason = str(metrics.get("reason"))
    else:
        reason = "has_independent_peak_evidence_or_low_overlap"
    metrics.update({
        "filtered": bool(filtered),
        "filter_reason": reason,
        "filter_main_peak_count": int(len(main_q)),
        "filter_candidate_peak_count": int(candidate_count),
        "filter_overlap_count": int(overlap_count),
        "filter_min_overlap_count_effective": int(max(1, min_overlap_count)),
        "filter_overlap_fraction": overlap_fraction,
        "filter_unique_supported_count": int(unique_supported_count),
    })
    return bool(filtered), metrics


def _as_finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _sanitize_label(text: str, fallback: str = "main_phase") -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or fallback)).strip("._-")
    return safe[:64] or fallback


def _snapshot_atom_state(phase: Any) -> List[Dict[str, Any]]:
    """Best-effort atom coordinate/U snapshot for guarded cleanup sanity checks."""
    rows: List[Dict[str, Any]] = []
    try:
        atom_ptrs = (phase.data.get("General") or {}).get("AtomPtrs") or []
        cx = int(atom_ptrs[0]) if len(atom_ptrs) > 0 else 3
        cia = int(atom_ptrs[3]) if len(atom_ptrs) > 3 else None
        atom_rows = phase.data.get("Atoms") or []
        for idx, atom in enumerate(atom_rows):
            label = str(atom[0]) if len(atom) > 0 else f"atom_{idx}"
            xyz = None
            try:
                xyz = tuple(float(atom[cx + j]) for j in range(3))
            except Exception:
                xyz = None
            u_iso = None
            if cia is not None:
                for offset in (1, 0, 2):
                    pos = cia + offset
                    if 0 <= pos < len(atom):
                        u_iso = _as_finite_float(atom[pos])
                        if u_iso is not None:
                            break
            rows.append({"index": idx, "label": label, "xyz": xyz, "u_iso": u_iso})
        if rows:
            return rows
    except Exception:
        rows = []

    try:
        for idx, atom in enumerate(phase.atoms()):
            label = str(getattr(atom, "label", f"atom_{idx}"))
            xyz = None
            coords = getattr(atom, "coordinates", None) or getattr(atom, "xyz", None)
            if coords is not None:
                try:
                    xyz = tuple(float(coords[j]) for j in range(3))
                except Exception:
                    xyz = None
            u_iso = (
                _as_finite_float(getattr(atom, "uiso", None))
                or _as_finite_float(getattr(atom, "Uiso", None))
                or _as_finite_float(getattr(atom, "u_iso", None))
            )
            rows.append({"index": idx, "label": label, "xyz": xyz, "u_iso": u_iso})
    except Exception:
        pass
    return rows


def _set_all_atom_flags(phase: Any, flags: str) -> Tuple[int, List[str]]:
    valid = set("XUF")
    clean = "".join(ch for ch in str(flags or "").upper() if ch in valid)
    labels: List[str] = []
    try:
        atoms = list(phase.atoms())
        for atom in atoms:
            atom.refinement_flags = clean
            labels.append(str(getattr(atom, "label", f"atom_{len(labels)}")))
        return len(labels), labels
    except Exception:
        pass

    try:
        phase.set_refinements({"Atoms": {"all": clean}})
        return 0, []
    except Exception:
        return 0, []


def _wrapped_position_delta(before_xyz: Any, after_xyz: Any) -> Optional[float]:
    if before_xyz is None or after_xyz is None:
        return None
    try:
        b = np.asarray(before_xyz, dtype=float)
        a = np.asarray(after_xyz, dtype=float)
        if b.size != 3 or a.size != 3 or not np.all(np.isfinite(b)) or not np.all(np.isfinite(a)):
            return None
        delta = np.abs(a - b)
        delta = np.minimum(delta, np.abs(delta - np.round(delta)))
        return float(np.max(delta))
    except Exception:
        return None


def _atom_cleanup_sanity(
    before_atoms: List[Dict[str, Any]],
    after_atoms: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    max_position_shift = 0.0
    bad_u: List[Dict[str, Any]] = []
    min_u = float(cfg.get("min_u_iso", 0.0))
    max_u = float(cfg.get("max_u_iso", 0.20))
    before_by_index = {int(row["index"]): row for row in before_atoms if "index" in row}
    for row in after_atoms:
        idx = int(row.get("index", -1))
        before = before_by_index.get(idx)
        if before:
            delta = _wrapped_position_delta(before.get("xyz"), row.get("xyz"))
            if delta is not None:
                max_position_shift = max(max_position_shift, delta)
        u_iso = row.get("u_iso")
        if u_iso is not None and (float(u_iso) < min_u or float(u_iso) > max_u):
            bad_u.append({
                "label": row.get("label"),
                "u_iso": float(u_iso),
            })
    max_allowed_shift = float(cfg.get("max_position_shift", 0.15))
    ok = bool(max_position_shift <= max_allowed_shift and not bad_u)
    return {
        "ok": ok,
        "max_position_shift": float(max_position_shift),
        "max_allowed_position_shift": max_allowed_shift,
        "bad_u_iso": bad_u,
    }


def _run_main_phase_internal_cleanup_step(
    refiner: Any,
    cfg: Dict[str, Any],
) -> Tuple[Any, Dict[str, Any]]:
    refine_positions = bool(cfg.get("refine_positions", False))
    refine_u_iso = bool(cfg.get("refine_u_iso", False))
    flags = ("X" if refine_positions else "") + ("U" if refine_u_iso else "")
    if not flags:
        raise RuntimeError("No internal main-phase parameters selected for cleanup")

    phase = refiner.phase
    before_atoms = _snapshot_atom_state(phase)
    atom_count, atom_labels = _set_all_atom_flags(phase, flags)
    if atom_count <= 0:
        raise RuntimeError("GSAS-II phase atom wrappers did not expose refinable atom flags")

    try:
        original_cycles = int(refiner.project.data["Controls"]["data"].get("max cyc", 3))
    except Exception:
        original_cycles = None
    try:
        refiner._disable_cell_refinement()
        refiner._enable_scale_refinement()
        refiner._enable_background_refinement()
        try:
            phase.set_HAP_refinements({"Use": True, "Scale": False}, histograms=[refiner.histogram])
            phase.HAPvalue("Scale", 1.0, targethistlist=[refiner.histogram])
        except Exception:
            pass
        refiner._set_max_cyc(max(1, int(cfg.get("cycles", 1))))
        refiner.project.refine()
        result = refiner._extract_refinement_results("MainPhaseInternalCleanup")
    finally:
        _set_all_atom_flags(phase, "")
        refiner._disable_cell_refinement()
        if original_cycles is not None:
            refiner._set_max_cyc(original_cycles)

    after_atoms = _snapshot_atom_state(phase)
    sanity = _atom_cleanup_sanity(before_atoms, after_atoms, cfg)
    audit = {
        "flags": flags,
        "atom_count": int(atom_count),
        "atoms": atom_labels[:50],
        "sanity": sanity,
    }
    return result, audit


def _export_cleanup_cif(phase: Any, out_dir: Path, label: str) -> Optional[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{_sanitize_label(label)}_main_cleanup.cif"
    try:
        phase.export_CIF(str(target), quickmode=True)
        text = target.read_text(encoding="utf-8", errors="ignore")
        if "_cell_length_a" not in text or "_atom_site_" not in text:
            return None
        safe_header = f"data_{_sanitize_label(label)}_main_cleanup"
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            if line.strip().startswith("data_"):
                lines[idx] = safe_header
                break
        else:
            lines.insert(0, safe_header)
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(target)
    except Exception:
        return None


def run_main_phase_cleanup_if_enabled(
    *,
    pm: Any,
    main_ref: Any,
    main_results: Any,
    main_cif: str,
    main_phase_name: str,
    top_cfg: Dict[str, Any],
    ds_cfg: Dict[str, Any],
    build_project_from_cif: Callable[[str, str], Tuple[Any, Any, Dict[str, Any]]],
    run_refinement: Callable[[Any], Any],
    out_dir: Path,
    user_supplied_main: bool,
    log: Callable[[str], None] = print,
    audit_path: Optional[Path] = None,
) -> MainPhaseAnchorResult:
    """Optionally refine main-phase atom U/positions on a clone and adopt only if stable."""
    cfg = main_phase_cleanup_cfg(top_cfg, ds_cfg)
    audit: Dict[str, Any] = {
        "enabled": bool(cfg.get("enabled", False)),
        "attempted": False,
        "adopted": False,
        "user_supplied_main": bool(user_supplied_main),
        "config": dict(cfg),
    }

    def _finish(result: MainPhaseAnchorResult) -> MainPhaseAnchorResult:
        if audit_path is not None:
            try:
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                audit_path.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
            except Exception:
                pass
        return result

    base_result = MainPhaseAnchorResult(pm, main_ref, main_results, main_cif, audit)
    if not bool(cfg.get("enabled", False)):
        audit["reason"] = "disabled"
        return _finish(base_result)
    if bool(cfg.get("apply_only_user_main", True)) and not user_supplied_main:
        audit["reason"] = "not_user_supplied_main"
        return _finish(base_result)
    if not (bool(cfg.get("refine_u_iso", False)) or bool(cfg.get("refine_positions", False))):
        audit["reason"] = "no_internal_parameters_selected"
        return _finish(base_result)
    if not main_cif:
        audit["reason"] = "missing_main_cif"
        return _finish(base_result)

    audit["attempted"] = True
    try:
        pm_cleanup, refiner_cleanup, context = build_project_from_cif(
            f"{_sanitize_label(main_phase_name)}_main_internal_cleanup",
            main_cif,
        )
        baseline_result = run_refinement(refiner_cleanup)
        audit["baseline_refinement"] = {
            "success": bool(getattr(baseline_result, "success", False)),
            "rwp": _as_finite_float(getattr(baseline_result, "rwp", None)),
            "error": getattr(baseline_result, "error_message", None),
        }
        if not getattr(baseline_result, "success", False):
            audit["reason"] = "baseline_clone_refinement_failed"
            return _finish(base_result)

        cleanup_result, step_audit = _run_main_phase_internal_cleanup_step(refiner_cleanup, cfg)
        cleanup_rwp = _as_finite_float(getattr(cleanup_result, "rwp", None))
        baseline_rwp = _as_finite_float(getattr(baseline_result, "rwp", None))
        current_rwp = _as_finite_float(getattr(main_results, "rwp", None))
        reference_candidates = [v for v in (baseline_rwp, current_rwp) if v is not None]
        if not reference_candidates:
            audit["reason"] = "missing_reference_rwp"
            return _finish(base_result)
        reference_rwp = min(reference_candidates)
        step_audit["result"] = {
            "success": bool(getattr(cleanup_result, "success", False)),
            "rwp": cleanup_rwp,
            "error": getattr(cleanup_result, "error_message", None),
        }
        audit["cleanup_step"] = step_audit

        if cleanup_rwp is None or not getattr(cleanup_result, "success", False):
            audit["reason"] = "cleanup_refinement_failed"
            return _finish(base_result)
        if not bool(step_audit.get("sanity", {}).get("ok", False)):
            audit["reason"] = "cleanup_sanity_failed"
            return _finish(base_result)
        accept_worsen = float(cfg.get("accept_rwp_worsen", 0.15))
        min_improve = float(cfg.get("min_rwp_improvement", 0.05))
        if cleanup_rwp > reference_rwp + accept_worsen:
            audit["reason"] = "cleanup_worsened_rwp"
            return _finish(base_result)
        if min_improve > 0 and cleanup_rwp > reference_rwp - min_improve:
            audit["reason"] = "cleanup_improvement_too_small"
            return _finish(base_result)

        cleanup_cif = _export_cleanup_cif(refiner_cleanup.phase, out_dir, main_phase_name)
        if cleanup_cif:
            audit["exported_cif"] = cleanup_cif
        audit["adopted"] = True
        audit["reason"] = "accepted"
        log(
            "[INFO] Adopted main-phase internal cleanup: "
            f"Rwp {reference_rwp:.3f}% -> {cleanup_rwp:.3f}%"
        )
        return _finish(MainPhaseAnchorResult(
            pm_cleanup,
            refiner_cleanup,
            cleanup_result,
            cleanup_cif or main_cif,
            audit,
            context=context,
        ))
    except Exception as exc:
        audit["reason"] = "exception"
        audit["error"] = str(exc)
        log(f"[WARN] Main-phase internal cleanup failed; keeping original fit: {exc}")
        return _finish(base_result)


def observed_signal_for_prenudge(
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


def assess_main_fit_for_prenudge(
    main_ref: Any,
    rwp: Optional[float],
    mode: str,
    cfg: Dict[str, Any],
    bg_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
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
    obs_signal, bg_meta = observed_signal_for_prenudge(q, yobs, bg_cfg)
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
    obs_peaks = select_top_peaks(
        q,
        obs_signal,
        top_n=int(cfg.get("top_observed_peaks", 8)),
        min_rel_height=min_rel,
        min_sep_q=tol_q,
    )
    calc_peaks = select_top_peaks(
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
        nearest = min(calc_peaks, key=lambda c: abs(float(c["q"]) - float(p["q"]))) if calc_peaks else None
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


def should_adopt_prenudged_main(
    before: Dict[str, Any],
    after: Dict[str, Any],
    before_rwp: Optional[float],
    after_rwp: Optional[float],
    nudge_score: Optional[float],
    cfg: Dict[str, Any],
) -> Tuple[bool, str]:
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
    if float(after_rwp) > float(before_rwp) + float(cfg.get("accept_rwp_worsen", 0.50)):
        return False, "rwp_worsened_too_much"
    before_support = float(before.get("weighted_peak_support", 0.0) or 0.0)
    after_support = float(after.get("weighted_peak_support", 0.0) or 0.0)
    if bool(after.get("strongest_peak_supported")) and not bool(before.get("strongest_peak_supported")):
        return True, "strongest_peak_support_fixed"
    if after_support >= before_support + float(cfg.get("accept_min_support_gain", 0.10)):
        return True, "peak_support_improved"
    if after_support >= before_support and bool(before.get("triggered")):
        return True, "triggered_fit_not_worse"
    return False, "no_fit_evidence_gain"


def run_main_phase_prenudge_if_needed(
    *,
    pm: Any,
    main_ref: Any,
    main_results: Any,
    main_cif: str,
    main_phase_name: str,
    top_cfg: Dict[str, Any],
    ds_cfg: Dict[str, Any],
    s4_cfg: Dict[str, Any],
    background_config: Dict[str, Any],
    mode: str,
    db_loader: Any,
    out_cif_dir: str | Path,
    build_project_from_cif: Callable[[str, str], Tuple[Any, Any, Dict[str, Any]]],
    run_refinement: Callable[[Any], Any],
    user_supplied_main: bool = True,
    log: Callable[[str], None] = print,
    event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    add_artifact: Optional[Callable[[str], None]] = None,
    audit_path: Optional[str | Path] = None,
    xray_doublet_config: Optional[Dict[str, Any]] = None,
) -> MainPhaseAnchorResult:
    s4_cfg = s4_cfg or {}
    prenudge_cfg = main_prenudge_cfg(top_cfg, ds_cfg, s4_cfg)
    audit: Dict[str, Any] = {
        "enabled": bool(prenudge_cfg.get("enabled", True)),
        "attempted": False,
        "adopted": False,
        "user_supplied_main": bool(user_supplied_main),
    }

    def _write_audit() -> None:
        if audit_path:
            path = Path(audit_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(__import__("json").dumps(audit, indent=2, default=str), encoding="utf-8")
            if add_artifact is not None:
                try:
                    add_artifact(str(path))
                except Exception:
                    pass

    if not (
        bool(prenudge_cfg.get("enabled", True))
        and main_cif
        and (not bool(prenudge_cfg.get("apply_only_user_main", True)) or user_supplied_main)
    ):
        audit["reason"] = "disabled_or_not_user_supplied_main"
        _write_audit()
        return MainPhaseAnchorResult(pm, main_ref, main_results, main_cif, audit)

    normal_fit_audit, nudge_q, nudge_signal = assess_main_fit_for_prenudge(
        main_ref,
        getattr(main_results, "rwp", None),
        mode,
        prenudge_cfg,
        background_config,
    )
    audit["normal_fit"] = normal_fit_audit
    audit["triggered"] = bool(normal_fit_audit.get("triggered", False))
    if not normal_fit_audit.get("triggered"):
        log(
            "[INFO] Main-phase pre-nudge skipped: "
            f"{normal_fit_audit.get('reason')} "
            f"(peak_support={float(normal_fit_audit.get('weighted_peak_support', 0.0)):.2f})"
        )
        _write_audit()
        return MainPhaseAnchorResult(pm, main_ref, main_results, main_cif, audit)

    log(
        "[INFO] Main-phase pre-nudge triggered: "
        f"{normal_fit_audit.get('reason')} "
        f"(Rwp={float(getattr(main_results, 'rwp', float('nan'))):.3f}%, "
        f"peak_support={float(normal_fit_audit.get('weighted_peak_support', 0.0)):.2f})"
    )
    if event_callback is not None:
        event_callback(
            "Main phase lattice pre-nudge",
            {
                "reason": normal_fit_audit.get("reason"),
                "rwp": getattr(main_results, "rwp", None),
                "peak_support": normal_fit_audit.get("weighted_peak_support"),
            },
        )

    audit["attempted"] = True
    try:
        from lattice_nudger import LatticeNudger

        nudger = LatticeNudger(
            db_loader,
            wavelength_ang=float(s4_cfg.get("wavelength", 1.54)),
            two_theta_range=tuple(s4_cfg.get("two_theta_range", [5.0, 160.0])),
            radiation=str(s4_cfg.get("radiation", "neutron")),
            score_q_max=float(prenudge_cfg.get("score_q_max", s4_cfg.get("score_q_max", 8.0))),
            lattice_tiebreak_score_tol=float(s4_cfg.get("lattice_tiebreak_score_tol", 5e-4)),
            xray_doublet_config=xray_doublet_config or {"enabled": False},
            random_seed=int(s4_cfg.get("seed", 0)),
        )
        out_cif_dir = Path(out_cif_dir)
        out_cif_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{main_phase_name}_main").strip("._-") or "main_phase"
        nudge_result = nudger.optimize_cif(
            main_cif,
            safe_name,
            nudge_q,
            nudge_signal,
            reps=int(prenudge_cfg.get("reps", 20)),
            samples=int(prenudge_cfg.get("samples", 2000)),
            frac_window=float(prenudge_cfg.get("frac_window", 0.01)),
            angle_window_deg=float(prenudge_cfg.get("angle_window_deg", 1.0)),
            out_cif_dir=str(out_cif_dir),
            allow_inner_parallel=True,
            score_q_max=float(prenudge_cfg.get("score_q_max", s4_cfg.get("score_q_max", 8.0))),
        )
        audit["nudge"] = {
            "score": float(nudge_result.best_score),
            "elapsed_s": float(nudge_result.elapsed_s),
            "candidate_count": int(nudge_result.candidate_count),
            "scored_count": int(nudge_result.scored_count),
            "lattice_deviation": float(getattr(nudge_result, "lattice_deviation", 0.0)),
            "tie_count": int(getattr(nudge_result, "score_tie_count", 1)),
            "cif": nudge_result.nudged_cif_path,
            "params": dict(nudge_result.best_params or {}),
        }
        if add_artifact is not None:
            try:
                add_artifact(str(nudge_result.nudged_cif_path))
            except Exception:
                pass

        pm_nudged, main_ref_nudged, context = build_project_from_cif(
            f"{safe_name}_prenudged",
            str(nudge_result.nudged_cif_path),
        )
        nudged_results = run_refinement(main_ref_nudged)
        audit["nudged_refinement"] = {
            "success": bool(getattr(nudged_results, "success", False)),
            "rwp": None if getattr(nudged_results, "rwp", None) is None else float(nudged_results.rwp),
            "error": getattr(nudged_results, "error_message", None),
        }
        if getattr(nudged_results, "success", False):
            nudged_fit_audit, _q_after, _sig_after = assess_main_fit_for_prenudge(
                main_ref_nudged,
                getattr(nudged_results, "rwp", None),
                mode,
                prenudge_cfg,
                background_config,
            )
            audit["nudged_fit"] = nudged_fit_audit
            adopt, adopt_reason = should_adopt_prenudged_main(
                normal_fit_audit,
                nudged_fit_audit,
                getattr(main_results, "rwp", None),
                getattr(nudged_results, "rwp", None),
                float(nudge_result.best_score),
                prenudge_cfg,
            )
            audit["adoption_reason"] = adopt_reason
            if adopt:
                log(
                    "[INFO] Adopted pre-nudged main phase CIF: "
                    f"Rwp {float(getattr(main_results, 'rwp', float('nan'))):.3f}% -> "
                    f"{float(getattr(nudged_results, 'rwp', float('nan'))):.3f}% "
                    f"({adopt_reason})"
                )
                audit["adopted"] = True
                _write_audit()
                return MainPhaseAnchorResult(
                    pm_nudged,
                    main_ref_nudged,
                    nudged_results,
                    str(nudge_result.nudged_cif_path),
                    audit,
                    context or {},
                )
            log(f"[INFO] Rejected pre-nudged main phase CIF: {adopt_reason}")
        else:
            log(f"[WARN] Pre-nudged main refinement failed: {getattr(nudged_results, 'error_message', None)}")
    except Exception as exc:
        audit["error"] = str(exc)
        log(f"[WARN] Main-phase pre-nudge failed; continuing with normal Stage-1 result: {exc}")

    _write_audit()
    return MainPhaseAnchorResult(pm, main_ref, main_results, main_cif, audit)
