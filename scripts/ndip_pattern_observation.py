"""Export measured/fit arrays from a trusted pipeline project as portable JSON.

This function receives records already read by the output collector. NOVA never
unpickles a user project to inspect curves. Invalid/excluded bins remain masks;
missing values are not converted into evidence for absent peaks.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
try:
    from gpx_recipe.fit_domain import NATIVE_FIT_CONVENTION, native_powder_fit_mask
except ModuleNotFoundError as exc:
    if exc.name not in {"gpx_recipe", "gpx_recipe.fit_domain"}:
        raise
    # Match ndip_cell_covariance's source-tree compatibility path: use the
    # same implementation without requiring an editable NOVA installation.
    import importlib.util
    _source = Path(__file__).resolve().parents[1] / "nova/src/gpx_recipe/fit_domain.py"
    _spec = importlib.util.spec_from_file_location("_radar_native_fit_domain", _source)
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    NATIVE_FIT_CONVENTION = _module.NATIVE_FIT_CONVENTION
    native_powder_fit_mask = _module.native_powder_fit_mask

try:
    from radar_pd_nova.signal_quality import supplied_error_signal_quality
except ModuleNotFoundError as exc:
    if exc.name not in {"radar_pd_nova", "radar_pd_nova.signal_quality"}:
        raise
    import importlib.util
    _source = Path(__file__).resolve().parents[1] / "nova/src/radar_pd_nova/signal_quality.py"
    _spec = importlib.util.spec_from_file_location("_radar_signal_quality", _source)
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    supplied_error_signal_quality = _module.supplied_error_signal_quality

SCHEMA = "radar-pd-pattern-observation/v1"
MAX_POINTS = 100_000
MAX_BYTES = 16_000_000


def pattern_from_records(path: Path, records, *, input_provenance=None, mode="full"):
    histograms = [r for r in records if r and str(r[0][0]).startswith("PWDR")]
    if len(histograms) != 1:
        raise ValueError("Curve comparison requires one explicit histogram per bank analysis")
    record = histograms[0]
    name, main = record[0]
    components = dict(record[1:])
    data = main[1]
    if len(data) < 6:
        raise ValueError("Powder histogram lacks measured, weight or calculated arrays")
    x, y, weights, calculated = [np.ma.asarray(v, dtype=float) for v in data[:4]]
    if (
        x.ndim != 1
        or not 16 <= len(x) <= MAX_POINTS
        or any(a.shape != x.shape for a in [y, weights, calculated])
    ):
        raise ValueError("Powder histogram exceeds the curve bounds or has inconsistent arrays")
    # Masked bins still have physical coordinates. Preserve that grid while
    # keeping them out of every measured/fitted comparison via explicit masks.
    xx = np.asarray(np.ma.getdata(x))
    if not np.all(np.isfinite(xx)) or np.any(np.diff(xx) <= 0):
        raise ValueError("Powder coordinates must be finite and strictly increasing")
    yy, ww, cc = (np.asarray(a.filled(np.nan)) for a in [y, weights, calculated])
    valid = ~(np.ma.getmaskarray(x) | np.ma.getmaskarray(y) | np.ma.getmaskarray(weights))
    valid &= np.isfinite(yy) & np.isfinite(ww) & (ww > 0)
    limits = components.get("Limits")
    if not isinstance(limits, (list, tuple)) or len(limits) < 2 or len(limits[1]) != 2:
        raise ValueError("Actual histogram fit limits are unavailable")
    lo, hi = map(float, limits[1])
    if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
        raise ValueError("Actual histogram fit interval is invalid")
    excluded = []
    for interval in limits[2:]:
        a, b = map(float, interval)
        if not math.isfinite(a) or not math.isfinite(b) or a > b:
            raise ValueError("Histogram exclusion interval is invalid")
        excluded.append([a, b])
    # errRefine includes the first bin at/above the upper limit. Reuse the
    # same domain contract as native recipe exports so the published mask
    # reproduces its Nobs and weighted residual, including between-bin limits.
    fitted = valid & native_powder_fit_mask(xx, [lo, hi], excluded)
    parameters = components.get("Instrument Parameters", [{}, {}])[0]
    initial = {
        key: (value[0] if isinstance(value, (list, tuple)) else value)
        for key, value in parameters.items()
    }
    instrument_type = str(initial.get("Type", ""))
    if "T" in instrument_type:
        coordinate = {"kind": "tof", "unit": "us"}
    elif instrument_type.startswith("P") and "C" in instrument_type:
        coordinate = {"kind": "two_theta", "unit": "deg"}
    else:
        coordinate = {"kind": "unknown", "unit": ""}
    original = input_provenance or {}
    sigma = [float(1 / math.sqrt(v)) if ok else None for v, ok in zip(ww, valid)]
    signal_quality = supplied_error_signal_quality(yy, sigma, fitted)
    packet = {
        "$schema": SCHEMA,
        "extractor_version": 2,
        "source_gpx_sha256": sha256(Path(path).read_bytes()).hexdigest(),
        "histogram": name,
        "coordinate": coordinate,
        "ordinate": {
            "basis": "as_imported_by_GSAS-II",
            "errors": "from_histogram_weights; conditional on the reduction error model",
        },
        "context": {
            "profile_sha256": sha256(
                json.dumps([initial], sort_keys=True, default=str).encode()
            ).hexdigest(),
            "recipe_key": f"radar-pd/{mode}/observables-v1",
            "instrument": original.get("instrument"),
            "source_bank": original.get("source_bank"),
            "reduction_family": original.get("reduction_family"),
            "calibration": original.get("calibration"),
            "reduction_setup": original.get("reduction_setup"),
            "source_product_sha256": original.get("original_product_sha256")
            or original.get("source_sha256"),
            "normalization": "not independently qualified by GPX extraction",
        },
        "fit_domain": {
            "limits": [lo, hi],
            "excluded": excluded,
            "convention": NATIVE_FIT_CONVENTION,
            "weight_factor": float(main[0].get("wtFactor", 1)) if isinstance(main[0], dict) else 1,
        },
        "signal_quality": signal_quality,
        "x": xx.tolist(),
        "y": np.where(np.isfinite(yy), yy, 0).tolist(),
        "sigma": sigma,
        "ycalc": [float(v) if ok and math.isfinite(v) else None for v, ok in zip(cc, fitted)],
        "valid": valid.tolist(),
        "fit_valid": fitted.tolist(),
    }
    raw = json.dumps(packet, sort_keys=True, allow_nan=False).encode()
    if len(raw) > MAX_BYTES:
        raise ValueError("Pattern observation exceeds its portable size limit")
    return packet


def publish_pattern(path, records, destination, *, input_provenance=None, mode="full"):
    packet = pattern_from_records(path, records, input_provenance=input_provenance, mode=mode)
    raw = json.dumps(packet, sort_keys=True, allow_nan=False).encode()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial")
    temporary.write_bytes(raw)
    temporary.replace(destination)
    return {
        "$schema": SCHEMA,
        "name": destination.name,
        "sha256": sha256(raw).hexdigest(),
        "points": len(packet["x"]),
        "source_gpx_sha256": packet["source_gpx_sha256"],
        "signal_quality": packet["signal_quality"],
    }
