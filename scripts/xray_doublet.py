#!/usr/bin/env python3
"""Runtime Cu K-alpha doublet correction for cheap PXRD matching.

The X-ray catalog remains a single-wavelength Q-space catalog. For lab PXRD
comparisons, this module creates a lightweight runtime approximation of the
K-alpha doublet by adding a weaker, fractionally shifted copy of each candidate
pattern. It is intentionally applied at scoring/rendering time so custom packs
and the built-in database do not need to be regenerated.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import re
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np


_NUM_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")


@dataclass(frozen=True)
class XrayDoubletSpec:
    enabled: bool
    lam1: Optional[float] = None
    lam2: Optional[float] = None
    intensity_ratio: float = 0.0
    q_ratio: float = 1.0
    source: str = "none"
    reason: str = ""
    apply_to_64_ml_input: bool = True
    apply_to_64_similarity: bool = True
    apply_to_512: bool = True
    apply_to_lattice_nudge: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "auto"}:
        return True
    if text in {"0", "false", "no", "off", "none"}:
        return False
    return bool(default)


def _first_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    match = _NUM_RE.search(str(value))
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def parse_instprm_numbers(instprm_path: str | Path | None) -> tuple[dict[str, float], dict[str, str]]:
    if not instprm_path:
        return {}, {}
    path = Path(instprm_path)
    if not path.exists():
        return {}, {}
    numeric: dict[str, float] = {}
    raw_values: dict[str, str] = {}
    for raw_line in path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        # GSAS-II .instprm files commonly pack multiple key/value pairs onto
        # one semicolon-delimited line, e.g.
        # ``Lam1:1.5405;Lam2:1.5443;I(L2)/I(L1):0.5``. Parse each segment
        # independently so Lam2 and the intensity ratio are not hidden inside
        # the Lam1 value.
        for segment in line.split(";"):
            segment = segment.strip()
            if not segment or ":" not in segment:
                continue
            key, value = segment.split(":", 1)
            key_norm = key.strip().lower()
            value = value.strip()
            if not key_norm:
                continue
            raw_values[key_norm] = value
            number = _first_number(value)
            if number is not None:
                numeric[key_norm] = number
    return numeric, raw_values


def _is_tof(raw_values: Mapping[str, str], mode: str | None) -> bool:
    requested = str(mode or "auto").strip().lower()
    if requested in {"tof", "time_of_flight", "time-of-flight"}:
        return True
    if requested in {"cw", "constant_wavelength", "constant-wavelength", "xray", "neutron_cw"}:
        return False
    type_token = str(raw_values.get("type", "") or "").upper()
    return "T" in type_token


def resolve_xray_doublet_spec(
    cfg: Mapping[str, Any] | None,
    *,
    dataset: Mapping[str, Any] | None = None,
    instprm_path: str | Path | None = None,
    stage4: Mapping[str, Any] | None = None,
) -> XrayDoubletSpec:
    """Resolve the doublet model from config and a GSAS-II instrument file."""
    cfg = cfg or {}
    dataset = dataset or {}
    section = dict(cfg.get("xray_doublet") or {})
    stage4_cfg = dict(stage4 or cfg.get("stage4") or {})

    enabled_value = str(section.get("enabled", "auto")).strip().lower()
    if enabled_value in {"0", "false", "no", "off", "none", "disabled"}:
        return XrayDoubletSpec(enabled=False, reason="disabled in config")

    radiation = str(stage4_cfg.get("radiation") or section.get("radiation") or "").strip().lower()
    if not radiation:
        return XrayDoubletSpec(enabled=False, reason="radiation is not declared as X-ray")
    if radiation not in {"xray", "x-ray", "pxrd"}:
        return XrayDoubletSpec(enabled=False, reason=f"radiation is {radiation!r}, not X-ray")

    instprm = instprm_path or dataset.get("instprm_path")
    params, raw_values = parse_instprm_numbers(instprm)
    mode = str(dataset.get("mode") or cfg.get("instrument_mode") or section.get("mode") or "auto")
    if _is_tof(raw_values, mode):
        return XrayDoubletSpec(enabled=False, reason="instrument mode is TOF")

    lam1 = _first_number(section.get("lambda1"))
    lam2 = _first_number(section.get("lambda2"))
    ratio_i = _first_number(section.get("intensity_ratio"))
    source = "config"

    if lam1 is None:
        lam1 = params.get("lam1")
        source = "instprm"
    if lam2 is None:
        lam2 = params.get("lam2")
    if ratio_i is None:
        ratio_i = params.get("i(l2)/i(l1)")
    if ratio_i is None and lam1 is not None and lam2 is not None:
        ratio_i = float(section.get("default_intensity_ratio", 0.5))
        source = f"{source}+default_intensity_ratio"

    if lam1 is None or lam2 is None:
        if enabled_value in {"1", "true", "yes", "on"}:
            return XrayDoubletSpec(enabled=False, reason="Lam1/Lam2 not found")
        return XrayDoubletSpec(enabled=False, reason="single-wavelength instrument")

    try:
        lam1_f = float(lam1)
        lam2_f = float(lam2)
        ratio_f = float(ratio_i if ratio_i is not None else 0.0)
    except Exception:
        return XrayDoubletSpec(enabled=False, reason="invalid doublet numeric values")

    if not (math.isfinite(lam1_f) and math.isfinite(lam2_f) and lam1_f > 0.0 and lam2_f > 0.0):
        return XrayDoubletSpec(enabled=False, reason="invalid Lam1/Lam2")
    if not (math.isfinite(ratio_f) and ratio_f > 0.0):
        return XrayDoubletSpec(enabled=False, reason="non-positive secondary wavelength intensity")

    q_ratio = lam2_f / lam1_f
    if not math.isfinite(q_ratio) or abs(q_ratio - 1.0) < 1e-6:
        return XrayDoubletSpec(enabled=False, reason="Lam1 and Lam2 are indistinguishable")

    return XrayDoubletSpec(
        enabled=True,
        lam1=lam1_f,
        lam2=lam2_f,
        intensity_ratio=ratio_f,
        q_ratio=q_ratio,
        source=source,
        reason="active",
        apply_to_64_ml_input=_as_bool(section.get("apply_to_64_ml_input"), True),
        apply_to_64_similarity=_as_bool(section.get("apply_to_64_similarity"), True),
        apply_to_512=_as_bool(section.get("apply_to_512"), True),
        apply_to_lattice_nudge=_as_bool(section.get("apply_to_lattice_nudge"), True),
    )


def _coerce_spec(spec: XrayDoubletSpec | Mapping[str, Any] | None) -> XrayDoubletSpec:
    if isinstance(spec, XrayDoubletSpec):
        return spec
    if isinstance(spec, Mapping):
        return XrayDoubletSpec(
            enabled=bool(spec.get("enabled", False)),
            lam1=float(spec["lam1"]) if spec.get("lam1") is not None else None,
            lam2=float(spec["lam2"]) if spec.get("lam2") is not None else None,
            intensity_ratio=float(spec.get("intensity_ratio") or 0.0),
            q_ratio=float(spec.get("q_ratio") or 1.0),
            source=str(spec.get("source") or "dict"),
            reason=str(spec.get("reason") or ""),
            apply_to_64_ml_input=_as_bool(spec.get("apply_to_64_ml_input"), True),
            apply_to_64_similarity=_as_bool(spec.get("apply_to_64_similarity"), True),
            apply_to_512=_as_bool(spec.get("apply_to_512"), True),
            apply_to_lattice_nudge=_as_bool(spec.get("apply_to_lattice_nudge"), True),
        )
    return XrayDoubletSpec(enabled=False)


def is_active_for(spec: XrayDoubletSpec | Mapping[str, Any] | None, key: str) -> bool:
    s = _coerce_spec(spec)
    return bool(s.enabled and getattr(s, key, True))


def _centers_from_meta(meta: Mapping[str, Any], n_bins: int) -> np.ndarray:
    if "centers" in meta:
        centers = np.asarray(meta["centers"], dtype=np.float64).reshape(-1)
        if centers.size == n_bins:
            return centers
    q_min = float(meta["q_min"])
    q_max = float(meta["q_max"])
    edges = np.linspace(q_min, q_max, int(n_bins) + 1, dtype=np.float64)
    return 0.5 * (edges[:-1] + edges[1:])


def shifted_profile_copy(profiles: np.ndarray, centers: np.ndarray, q_ratio: float) -> np.ndarray:
    """Return f(Q / q_ratio) on the same uniform Q centers for each profile row."""
    arr = np.asarray(profiles, dtype=np.float32)
    if arr.ndim == 1:
        arr2 = arr.reshape(1, -1)
        squeeze = True
    else:
        arr2 = arr
        squeeze = False

    centers = np.asarray(centers, dtype=np.float64).reshape(-1)
    if arr2.shape[1] != centers.size or centers.size < 2:
        return arr.copy()

    source_q = centers / float(q_ratio)
    pos = np.interp(source_q, centers, np.arange(centers.size, dtype=np.float64), left=np.nan, right=np.nan)
    safe_pos = np.where(np.isfinite(pos), pos, -1.0)
    lo = np.floor(safe_pos).astype(np.int64, copy=False)
    frac = (pos - lo).astype(np.float32, copy=False)
    valid = np.isfinite(pos) & (lo >= 0) & (lo < centers.size)
    hi = np.minimum(lo + 1, centers.size - 1)

    shifted = np.zeros_like(arr2, dtype=np.float32)
    valid_idx = np.where(valid)[0]
    if valid_idx.size:
        lo_v = lo[valid_idx]
        hi_v = hi[valid_idx]
        frac_v = frac[valid_idx]
        shifted[:, valid_idx] = (
            arr2[:, lo_v] * (1.0 - frac_v)[None, :]
            + arr2[:, hi_v] * frac_v[None, :]
        )
    return shifted[0] if squeeze else shifted


def apply_doublet_to_profiles(
    profiles: np.ndarray,
    meta: Mapping[str, Any],
    spec: XrayDoubletSpec | Mapping[str, Any] | None,
    *,
    apply_key: str = "apply_to_64_ml_input",
    renormalize: bool = True,
) -> np.ndarray:
    s = _coerce_spec(spec)
    if not (s.enabled and getattr(s, apply_key, True)):
        return np.asarray(profiles)
    arr = np.asarray(profiles, dtype=np.float32)
    if arr.size == 0:
        return arr
    centers = _centers_from_meta(meta, arr.shape[-1])
    shifted = shifted_profile_copy(arr, centers, s.q_ratio)
    primary_w = 1.0 / (1.0 + s.intensity_ratio)
    secondary_w = s.intensity_ratio / (1.0 + s.intensity_ratio)
    out = (primary_w * arr + secondary_w * shifted).astype(np.float32, copy=False)
    if renormalize:
        if out.ndim == 1:
            mx = float(np.max(out)) if out.size else 0.0
            if mx > 0.0 and math.isfinite(mx):
                out = out / mx
        else:
            mx = np.max(out, axis=1, keepdims=True)
            out = np.divide(out, np.maximum(mx, 1e-8), out=np.zeros_like(out), where=mx > 0.0)
    return out.astype(np.float32, copy=False)


def apply_doublet_to_peaks(
    q: np.ndarray,
    intensity: np.ndarray,
    spec: XrayDoubletSpec | Mapping[str, Any] | None,
    *,
    apply_key: str = "apply_to_512",
) -> tuple[np.ndarray, np.ndarray]:
    s = _coerce_spec(spec)
    q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
    i_arr = np.asarray(intensity, dtype=np.float64).reshape(-1)
    n = min(q_arr.size, i_arr.size)
    q_arr = q_arr[:n]
    i_arr = i_arr[:n]
    mask = np.isfinite(q_arr) & np.isfinite(i_arr) & (q_arr > 0.0) & (i_arr > 0.0)
    q_arr = q_arr[mask]
    i_arr = i_arr[mask]
    if not (s.enabled and getattr(s, apply_key, True)) or q_arr.size == 0:
        return q_arr.astype(np.float32), i_arr.astype(np.float32)
    q2 = q_arr * float(s.q_ratio)
    i2 = i_arr * float(s.intensity_ratio)
    q_out = np.concatenate([q_arr, q2])
    i_out = np.concatenate([i_arr, i2])
    order = np.argsort(q_out, kind="stable")
    return q_out[order].astype(np.float32), i_out[order].astype(np.float32)


def describe_doublet(spec: XrayDoubletSpec | Mapping[str, Any] | None) -> str:
    s = _coerce_spec(spec)
    if not s.enabled:
        return f"off ({s.reason or 'inactive'})"
    return (
        f"Lam1={s.lam1:.6g}, Lam2={s.lam2:.6g}, "
        f"I2/I1={s.intensity_ratio:.3g}, Q ratio={s.q_ratio:.6g}"
    )
