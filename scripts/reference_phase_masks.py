#!/usr/bin/env python3
"""Generate native-axis exclusion windows for common container/reference phases."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_REFERENCE_PHASE_DIR = REPO_ROOT / "data" / "reference_phases"
SCRIPT_REFERENCE_PHASE_DIR = Path(__file__).resolve().parent / "reference_phases"
REFERENCE_PHASE_DIR = (
    DATA_REFERENCE_PHASE_DIR
    if DATA_REFERENCE_PHASE_DIR.exists()
    else SCRIPT_REFERENCE_PHASE_DIR
)
CU_KBETA_WAVELENGTH = 1.39225
DEFAULT_FWHM_FACTOR = 6.0
DEFAULT_FRACTIONAL_D_TOLERANCE = 0.003
DEFAULT_CW_ZERO_TOLERANCE_DEG = 0.05
DEFAULT_TOF_ZERO_TOLERANCE = 25.0


@dataclass(frozen=True)
class ReferencePreset:
    canonical: str
    aliases: frozenset[str]
    formula: str
    lattice: str
    a: float
    cif_name: str

    @property
    def cif_path(self) -> Path:
        return REFERENCE_PHASE_DIR / self.cif_name


@dataclass(frozen=True)
class InstrumentModel:
    mode: str
    type_token: str
    zero: float
    params: Dict[str, float]
    wavelengths: Tuple[Tuple[str, float], ...] = ()


REFERENCE_PHASE_PRESETS: Dict[str, ReferencePreset] = {
    "Al_fcc": ReferencePreset(
        canonical="Al_fcc",
        aliases=frozenset({"Al", "Al_fcc", "al", "al_fcc", "aluminum"}),
        formula="Al",
        lattice="fcc",
        a=4.0495,
        cif_name="Al_fcc.cif",
    ),
    "Cu_fcc": ReferencePreset(
        canonical="Cu_fcc",
        aliases=frozenset({"Cu", "Cu_fcc", "cu", "cu_fcc", "copper"}),
        formula="Cu",
        lattice="fcc",
        a=3.6149,
        cif_name="Cu_fcc.cif",
    ),
    "V_bcc": ReferencePreset(
        canonical="V_bcc",
        aliases=frozenset({"V", "V_bcc", "v", "v_bcc", "vanadium"}),
        formula="V",
        lattice="bcc",
        a=3.0270,
        cif_name="V_bcc.cif",
    ),
}


def normalize_reference_preset(name: str) -> str:
    needle = str(name).strip()
    for canonical, preset in REFERENCE_PHASE_PRESETS.items():
        if needle == canonical or needle in preset.aliases:
            return canonical
    supported = ", ".join(sorted(REFERENCE_PHASE_PRESETS))
    raise ValueError(f"Unsupported reference phase preset '{name}'. Supported presets: {supported}")


def _first_number(text: str) -> Optional[float]:
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    return float(match.group(0)) if match else None


def parse_instprm_parameters(instprm_path: str | Path) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Parse numeric GSAS-II instprm values and raw string values."""
    path = Path(instprm_path)
    if not path.exists():
        raise FileNotFoundError(f"Instrument parameter file not found: {path}")

    numeric: Dict[str, float] = {}
    raw_values: Dict[str, str] = {}
    for raw_line in path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key_norm = key.strip().lower()
        value = value.strip()
        raw_values[key_norm] = value
        number = _first_number(value)
        if number is not None:
            numeric[key_norm] = number
    return numeric, raw_values


def _detected_mode_from_type(type_token: str) -> Optional[str]:
    token = str(type_token or "").upper()
    if not token:
        return None
    return "tof" if "T" in token else "cw"


def _normalize_mode(mode: str, raw_values: Dict[str, str]) -> str:
    requested = str(mode or "auto").strip().lower()
    detected = _detected_mode_from_type(raw_values.get("type", ""))
    if requested == "auto":
        if not detected:
            raise ValueError("Could not determine instrument mode from config or instprm Type")
        return detected
    if requested in {"constant_wavelength", "constant-wavelength", "xray", "neutron_cw"}:
        requested = "cw"
    if requested in {"time_of_flight", "time-of-flight"}:
        requested = "tof"
    if requested not in {"cw", "tof"}:
        raise ValueError(f"Unsupported instrument mode for reference masks: {mode!r}")
    if detected and requested != detected:
        raise ValueError(
            f"Reference mask mode {requested!r} conflicts with instprm Type={raw_values.get('type')!r}"
        )
    return requested


def _instrument_wavelengths(params: Dict[str, float], cfg: Dict[str, Any]) -> Tuple[Tuple[str, float], ...]:
    wavelengths: List[Tuple[str, float]] = []
    if "lam1" in params:
        wavelengths.append(("Lam1", params["lam1"]))
        include_secondary = bool(cfg.get("include_secondary_wavelengths", True))
        intensity_ratio = float(params.get("i(l2)/i(l1)", 0.0))
        if include_secondary and "lam2" in params and intensity_ratio > 0.0:
            wavelengths.append(("Lam2", params["lam2"]))
    elif "lam" in params:
        wavelengths.append(("Lam", params["lam"]))
    elif cfg.get("wavelength") is not None:
        wavelengths.append(("config", float(cfg["wavelength"])))

    cleaned = [(label, float(wl)) for label, wl in wavelengths if math.isfinite(float(wl)) and float(wl) > 0.0]
    if not cleaned:
        raise ValueError("Could not determine CW wavelength from instprm Lam/Lam1 or config wavelength")
    return tuple(cleaned)


def instrument_model_from_instprm(
    instprm_path: str | Path,
    *,
    mode: str,
    config: Optional[Dict[str, Any]] = None,
) -> InstrumentModel:
    cfg = dict(config or {})
    params, raw_values = parse_instprm_parameters(instprm_path)
    mode_norm = _normalize_mode(mode, raw_values)
    zero = float(params.get("zero", 0.0))

    if mode_norm == "cw":
        wavelengths = _instrument_wavelengths(params, cfg)
        return InstrumentModel(
            mode="cw",
            type_token=raw_values.get("type", ""),
            zero=zero,
            params=params,
            wavelengths=wavelengths,
        )

    required = ("difc",)
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError(f"TOF reference masks require instprm parameter(s): {', '.join(missing)}")
    return InstrumentModel(
        mode="tof",
        type_token=raw_values.get("type", ""),
        zero=zero,
        params=params,
        wavelengths=(),
    )


def _reflection_allowed(lattice: str, h: int, k: int, l: int) -> bool:
    if h == 0 and k == 0 and l == 0:
        return False
    if lattice == "fcc":
        parities = {h % 2, k % 2, l % 2}
        return len(parities) == 1
    if lattice == "bcc":
        return (h + k + l) % 2 == 0
    return True


def _cubic_reflections(
    lattice: str,
    a: float,
    max_index: int = 16,
) -> Iterable[Tuple[Tuple[int, int, int], float]]:
    """Generate unique cubic reflections with systematic absences applied."""
    seen_n: set[int] = set()
    rows: List[Tuple[int, Tuple[int, int, int], float]] = []
    for h in range(max_index + 1):
        for k in range(h + 1):
            for l in range(k + 1):
                if not _reflection_allowed(lattice, h, k, l):
                    continue
                n = h * h + k * k + l * l
                if n in seen_n:
                    continue
                seen_n.add(n)
                d_spacing = a / math.sqrt(n)
                rows.append((n, (h, k, l), d_spacing))
    rows.sort(key=lambda row: row[0])
    for _, hkl, d_spacing in rows:
        yield hkl, d_spacing


def _two_theta_from_d(d_spacing: float, wavelength: float, zero: float = 0.0) -> Optional[float]:
    arg = wavelength / (2.0 * d_spacing)
    if arg <= 0.0 or arg >= 1.0:
        return None
    return 2.0 * math.degrees(math.asin(arg)) + float(zero)


def _tof_from_d(d_spacing: float, instrument: InstrumentModel) -> Optional[float]:
    if d_spacing <= 0.0:
        return None
    params = instrument.params
    difc = float(params.get("difc", 0.0))
    if not math.isfinite(difc) or difc <= 0.0:
        return None
    difa = float(params.get("difa", 0.0))
    difb = float(params.get("difb", 0.0))
    return difc * d_spacing + difa * d_spacing * d_spacing + difb / d_spacing + instrument.zero


def _as_limits(limits: Any) -> Tuple[Optional[float], Optional[float]]:
    if not limits or len(limits) != 2:
        return None, None
    lo = float(min(limits[0], limits[1]))
    hi = float(max(limits[0], limits[1]))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return None, None
    return lo, hi


def _extract_preset_names(raw_refs: Any) -> List[str]:
    if raw_refs is None:
        return []
    if isinstance(raw_refs, str):
        return [raw_refs]
    if not isinstance(raw_refs, list):
        raise ValueError("reference_phase_exclusions presets/references must be a string or list")

    names: List[str] = []
    for item in raw_refs:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            preset = item.get("preset") or item.get("name")
            if preset:
                names.append(str(preset))
        elif item is not None:
            names.append(str(item))
    return names


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _cw_profile_fwhm(center: float, instrument: InstrumentModel) -> Optional[float]:
    params = instrument.params
    corrected = center - instrument.zero
    theta = math.radians(corrected / 2.0)
    if theta <= 0.0 or not math.isfinite(theta):
        return None
    tan_theta = math.tan(theta)
    cos_theta = math.cos(theta)
    if abs(cos_theta) < 1e-8:
        return None

    gaussian_var = (
        float(params.get("u", 0.0)) * tan_theta * tan_theta
        + float(params.get("v", 0.0)) * tan_theta
        + float(params.get("w", 0.0))
    )
    candidates: List[float] = []
    if math.isfinite(gaussian_var) and gaussian_var > 0.0:
        # GSAS CW U/V/W are in centidegrees squared.
        candidates.append(math.sqrt(gaussian_var) / 100.0)

    lorentz = float(params.get("x", 0.0)) / cos_theta + float(params.get("y", 0.0)) * tan_theta
    if math.isfinite(lorentz) and lorentz > 0.0:
        # GSAS CW X/Y are in centidegrees.
        candidates.append(lorentz / 100.0)

    if not candidates:
        return None
    return max(candidates)


def _tof_profile_fwhm(d_spacing: float, instrument: InstrumentModel) -> Optional[float]:
    params = instrument.params
    sig0 = float(params.get("sig-0", params.get("sig0", 0.0)))
    sig1 = float(params.get("sig-1", params.get("sig1", 0.0)))
    sig2 = float(params.get("sig-2", params.get("sig2", 0.0)))
    sigq = float(params.get("sig-q", params.get("sigq", 0.0)))
    variance = sig2 * d_spacing**4 + sig1 * d_spacing**2 + sigq * d_spacing + sig0
    if not math.isfinite(variance) or variance <= 0.0:
        return None
    return 2.354820045 * math.sqrt(variance)


def _position_tolerance_native(center: float, d_spacing: float, instrument: InstrumentModel, cfg: Dict[str, Any]) -> float:
    """Estimate extra half-width for reference cell/alloy and zero-offset mismatch."""
    frac_tol = float(cfg.get("fractional_d_tolerance", DEFAULT_FRACTIONAL_D_TOLERANCE))
    if not math.isfinite(frac_tol) or frac_tol < 0.0:
        raise ValueError("Reference mask fractional_d_tolerance must be non-negative")

    if instrument.mode == "cw":
        zero_tol = _specific_float(
            cfg,
            instrument.mode,
            "zero_tolerance",
            default=DEFAULT_CW_ZERO_TOLERANCE_DEG,
        )
        corrected = center - instrument.zero
        theta = math.radians(corrected / 2.0)
        if theta <= 0.0 or not math.isfinite(theta):
            lattice_tol = 0.0
        else:
            # d(2theta)/d(ln d) = -2 tan(theta), in radians.
            lattice_tol = abs(math.degrees(2.0 * math.tan(theta)) * frac_tol)
        return max(0.0, lattice_tol) + max(0.0, zero_tol)

    zero_tol = _specific_float(
        cfg,
        instrument.mode,
        "zero_tolerance",
        default=DEFAULT_TOF_ZERO_TOLERANCE,
    )
    lattice_tol = 0.0
    if frac_tol > 0.0 and d_spacing > 0.0:
        centers = []
        for scale in (1.0 - frac_tol, 1.0 + frac_tol):
            shifted = _tof_from_d(d_spacing * scale, instrument)
            if shifted is not None and math.isfinite(shifted):
                centers.append(abs(float(shifted) - float(center)))
        if centers:
            lattice_tol = max(centers)
    return max(0.0, lattice_tol) + max(0.0, zero_tol)


def _specific_float(cfg: Dict[str, Any], mode: str, *names: str, default: float) -> float:
    suffix_names = []
    if mode == "cw":
        suffix_names = [f"{name}_deg" for name in names]
    elif mode == "tof":
        suffix_names = [f"{name}_tof" for name in names] + [f"{name}_us" for name in names]
    for key in (*names, *suffix_names):
        if key in cfg and cfg[key] is not None:
            return float(cfg[key])
    return float(default)


def _half_width_native(
    center: float,
    d_spacing: float,
    instrument: InstrumentModel,
    cfg: Dict[str, Any],
) -> Tuple[float, str]:
    has_fixed = any(
        key in cfg
        for key in (
            "half_width",
            "half_width_deg",
            "half_width_tof",
            "half_width_us",
        )
    )
    mode_cfg = cfg.get("window_mode", cfg.get("width_mode"))
    window_mode = str(mode_cfg or ("fixed" if has_fixed else "auto")).strip().lower()

    if window_mode == "fixed":
        default_fixed = 0.30 if instrument.mode == "cw" else 75.0
        width = _specific_float(cfg, instrument.mode, "half_width", default=default_fixed)
        if not math.isfinite(width) or width <= 0.0:
            raise ValueError("Reference mask fixed half-width must be positive")
        return width, "fixed"

    if window_mode != "auto":
        raise ValueError("reference_phase_exclusions.window_mode must be 'auto' or 'fixed'")

    if instrument.mode == "cw":
        fwhm = _cw_profile_fwhm(center, instrument)
        default_min = 0.35
        default_max = 2.00
        default_fallback = 0.75
    else:
        fwhm = _tof_profile_fwhm(d_spacing, instrument)
        default_min = 75.0
        default_max = 750.0
        default_fallback = 200.0

    min_width = _specific_float(cfg, instrument.mode, "min_half_width", default=default_min)
    max_width = _specific_float(cfg, instrument.mode, "max_half_width", default=default_max)
    fallback = _specific_float(cfg, instrument.mode, "fallback_half_width", default=default_fallback)
    if min_width <= 0.0 or max_width <= 0.0 or max_width < min_width:
        raise ValueError("Reference mask auto width limits must be positive and ordered")

    if fwhm is None or not math.isfinite(fwhm) or fwhm <= 0.0:
        return _clamp(fallback, min_width, max_width), "fallback"

    factor = float(cfg.get("fwhm_factor", cfg.get("window_fwhm_factor", DEFAULT_FWHM_FACTOR)))
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("Reference mask fwhm_factor must be positive")
    profile_half_width = 0.5 * factor * fwhm
    position_tolerance = _position_tolerance_native(center, d_spacing, instrument, cfg)
    width_source = "auto_profile+tolerance" if position_tolerance > 0.0 else "auto_profile"
    return _clamp(profile_half_width + position_tolerance, min_width, max_width), width_source


def _range_in_limits(center: float, half_width: float, lo: Optional[float], hi: Optional[float]) -> Optional[List[float]]:
    start = center - half_width
    end = center + half_width
    if lo is not None:
        start = max(start, lo)
    if hi is not None:
        end = min(end, hi)
    if end <= start:
        return None
    return [float(start), float(end)]


def _reference_lines_for_d(
    d_spacing: float,
    instrument: InstrumentModel,
    cfg: Dict[str, Any],
) -> Iterable[Tuple[str, Optional[float], Optional[float]]]:
    if instrument.mode == "cw":
        for label, wavelength in instrument.wavelengths:
            yield label, wavelength, _two_theta_from_d(d_spacing, wavelength, zero=instrument.zero)
        if bool(cfg.get("include_cu_kbeta", cfg.get("include_kbeta", False))):
            kbeta = float(cfg.get("kbeta_wavelength", CU_KBETA_WAVELENGTH))
            yield "CuKbeta", kbeta, _two_theta_from_d(d_spacing, kbeta, zero=instrument.zero)
    else:
        yield "TOF", None, _tof_from_d(d_spacing, instrument)


def build_reference_phase_exclusions(
    config: Optional[Dict[str, Any]],
    *,
    instprm_path: str | Path,
    mode: str,
    limits: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Build reference-phase exclusion ranges in the histogram native axis."""
    cfg = dict(config or {})
    if not bool(cfg.get("enabled", False)):
        return {"enabled": False, "ranges": [], "reflections": [], "warnings": []}

    phase_names = _extract_preset_names(cfg.get("presets", cfg.get("phases", cfg.get("references", []))))
    presets = [normalize_reference_preset(name) for name in phase_names]
    if not presets:
        raise ValueError("reference_phase_exclusions.enabled=true requires at least one preset")

    instrument = instrument_model_from_instprm(instprm_path, mode=mode, config=cfg)
    max_index = int(cfg.get("max_hkl_index", 16))
    if max_index < 1:
        raise ValueError("reference_phase_exclusions.max_hkl_index must be positive")
    lo, hi = _as_limits(limits)

    ranges: List[List[float]] = []
    reflections: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for preset_name in presets:
        preset = REFERENCE_PHASE_PRESETS[preset_name]
        if not preset.cif_path.exists():
            warnings.append(f"Bundled reference CIF missing for {preset_name}: {preset.cif_path}")
        for hkl, d_spacing in _cubic_reflections(preset.lattice, preset.a, max_index=max_index):
            for line_label, wavelength, center in _reference_lines_for_d(d_spacing, instrument, cfg):
                if center is None or not math.isfinite(center):
                    continue
                half_width, width_source = _half_width_native(center, d_spacing, instrument, cfg)
                window = _range_in_limits(center, half_width, lo, hi)
                if window is None:
                    continue
                ranges.append(window)
                reflections.append(
                    {
                        "preset": preset_name,
                        "formula": preset.formula,
                        "reference_cif": str(preset.cif_path),
                        "hkl": list(hkl),
                        "d_spacing": float(d_spacing),
                        "line": line_label,
                        "wavelength": float(wavelength) if wavelength is not None else None,
                        "center": float(center),
                        "half_width": float(half_width),
                        "width_source": width_source,
                        "range": window,
                    }
                )

    return {
        "enabled": True,
        "presets": presets,
        "mode": instrument.mode,
        "native_axis": "2theta_deg" if instrument.mode == "cw" else "tof_us",
        "instrument": {
            "instprm_path": str(Path(instprm_path)),
            "type": instrument.type_token,
            "zero": instrument.zero,
            "wavelengths": [
                {"label": label, "value": value}
                for label, value in instrument.wavelengths
            ],
            "difC": instrument.params.get("difc"),
            "difA": instrument.params.get("difa"),
            "difB": instrument.params.get("difb"),
        },
        "window_mode": str(cfg.get("window_mode", cfg.get("width_mode", "auto" if not any(k in cfg for k in ("half_width", "half_width_deg", "half_width_tof", "half_width_us")) else "fixed"))),
        "include_cu_kbeta": bool(cfg.get("include_cu_kbeta", cfg.get("include_kbeta", False))),
        "kbeta_wavelength": float(cfg.get("kbeta_wavelength", CU_KBETA_WAVELENGTH))
        if bool(cfg.get("include_cu_kbeta", cfg.get("include_kbeta", False))) else None,
        "ranges": ranges,
        "reflections": reflections,
        "warnings": warnings,
    }


def merge_reference_phase_exclusion_config(
    global_cfg: Optional[Dict[str, Any]],
    dataset_cfg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    merged = dict(global_cfg or {})
    if dataset_cfg:
        merged.update(dataset_cfg)
    return merged
