import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    from gpx_recipe.fit_domain import NATIVE_FIT_CONVENTION, native_powder_fit_mask
except ModuleNotFoundError as exc:
    if exc.name not in {"gpx_recipe", "gpx_recipe.fit_domain"}:
        raise
    # Reuse the native contract in source checkouts without installing NOVA.
    import importlib.util
    _source = Path(__file__).resolve().parents[1] / "nova/src/gpx_recipe/fit_domain.py"
    _spec = importlib.util.spec_from_file_location("_radar_plot_fit_domain", _source)
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    NATIVE_FIT_CONVENTION = _module.NATIVE_FIT_CONVENTION
    native_powder_fit_mask = _module.native_powder_fit_mask

logger = logging.getLogger(__name__)


def histogram_fit_region(histogram):
    """Read explicit GSAS-II active limits/exclusions, never infer from ycalc."""
    limits = histogram.data.get("Limits") or []
    def interval(value, *, allow_point=False):
        if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 2:
            return None
        try:
            pair = [float(v) for v in value]
        except (TypeError, ValueError):
            return None
        return pair if all(np.isfinite(pair)) and (pair[0] <= pair[1] if allow_point else pair[0] < pair[1]) else None
    active = interval(limits[1]) if len(limits) > 1 else None
    excluded = [pair for value in limits[2:] if (pair := interval(value, allow_point=True)) is not None]
    return active, excluded


def native_plot_mask(arrays, limits, exclusions):
    """Validate a complete native-grid plot mask, without inferring from ycalc."""
    for interval, allow_point in [(limits, False), *((value, True) for value in exclusions)]:
        pair = np.asarray(interval, dtype=float)
        if (pair.shape != (2,) or not np.all(np.isfinite(pair))
                or (pair[0] > pair[1] if allow_point else pair[0] >= pair[1])):
            raise ValueError("Native plot fit limits or exclusions are invalid")
    x = np.asarray(arrays["x"], dtype=float)
    if x.ndim != 1 or not len(x) or not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError("A native plot needs finite, increasing histogram coordinates")
    masks = []
    for key in ("valid", "fit_valid"):
        value = np.asarray(arrays[key])
        if value.shape != x.shape or value.dtype.kind != "b":
            raise ValueError("A native plot needs explicit aligned boolean masks")
        masks.append(value)
    valid, fitted = masks
    weights = np.asarray(arrays["weights"], dtype=float)
    observed = np.asarray(arrays["yobs"], dtype=float)
    calculated = np.asarray(arrays["ycalc"], dtype=float)
    if any(value.shape != x.shape for value in (weights, observed, calculated)):
        raise ValueError("Native plot arrays do not share the histogram grid")
    if np.any(valid & (~np.isfinite(weights) | (weights <= 0) | ~np.isfinite(observed))):
        raise ValueError("Native plot validity conflicts with observed values or weights")
    if not np.array_equal(fitted, valid & native_powder_fit_mask(x, limits, exclusions)):
        raise ValueError("Native plot fit mask differs from the declared fit domain")
    if np.any(fitted & ~np.isfinite(calculated)):
        raise ValueError("Native plot has missing calculated values in its fitted domain")
    return fitted


def histogram_plot_arrays(histogram):
    """Read plotting arrays on the original grid before any masks are compressed.

    This is separate from the scientific extractor: changing its compressed
    arrays would also change candidate scoring and refinement inputs.
    """
    limits, exclusions = histogram_fit_region(histogram)
    raw_limits = histogram.data.get("Limits") or []
    if limits is None or len(exclusions) != max(0, len(raw_limits) - 2):
        raise ValueError("Native plot fit limits or exclusions are unavailable")
    raw = [np.ma.asarray(value, dtype=float) for value in histogram.data["data"][1][:4]]
    if len(raw) != 4 or any(value.ndim != 1 or value.shape != raw[0].shape for value in raw):
        raise ValueError("Native plot needs aligned x, observed, weight and calculated arrays")
    x, observed, weights, calculated = [np.asarray(np.ma.getdata(value)) for value in raw]
    valid = ~(np.ma.getmaskarray(raw[0]) | np.ma.getmaskarray(raw[1]) | np.ma.getmaskarray(raw[2]))
    valid &= np.isfinite(observed) & np.isfinite(weights) & (weights > 0)
    fitted = valid & native_powder_fit_mask(x, limits, exclusions)
    arrays = {"x": x, "yobs": observed, "weights": weights, "ycalc": calculated,
              "resid": observed - calculated, "valid": valid, "fit_valid": fitted}
    native_plot_mask(arrays, limits, exclusions)
    if np.any(fitted & np.ma.getmaskarray(raw[3])):
        raise ValueError("Native plot has masked calculated values in its fitted domain")
    return arrays


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def save_plot_payload(
    plot_path: str,
    payload: Optional[Dict[str, Any]] = None,
    arrays: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """Save sidecar payload next to a static plot.

    Produces:
      - <plot>.plotdata.json
      - <plot>.plotdata.npz   (optional, if arrays provided)
    """
    try:
        plot_file = Path(plot_path)
        json_file = Path(str(plot_file) + ".plotdata.json")

        data: Dict[str, Any] = dict(payload or {})
        data.setdefault("schema_version", "1.0")
        data.setdefault("source_plot", plot_file.name)
        data.setdefault("source_plot_path", str(plot_file.resolve()))

        if arrays:
            npz_file = Path(str(plot_file) + ".plotdata.npz")
            serial_arrays: Dict[str, np.ndarray] = {}
            for key, arr in arrays.items():
                if arr is None:
                    continue
                serial_arrays[str(key)] = np.asarray(arr)
            if serial_arrays:
                np.savez_compressed(npz_file, **serial_arrays)
                data["arrays_npz"] = npz_file.name
                data["array_keys"] = sorted(serial_arrays.keys())

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(data), f, ensure_ascii=False, indent=2)

        return json_file
    except Exception as exc:
        logger.warning(f"[plot-payload] save failed for {plot_path}: {exc}")
        return None


def load_plot_payload(json_path: str) -> Dict[str, Any]:
    """Load sidecar payload and optional NPZ arrays."""
    payload_path = Path(json_path)
    with open(payload_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    arrays_npz = payload.get("arrays_npz")
    if arrays_npz:
        npz_path = payload_path.parent / arrays_npz
        if npz_path.exists():
            with np.load(npz_path, allow_pickle=False) as z:
                payload["arrays"] = {k: z[k] for k in z.files}
    return payload
