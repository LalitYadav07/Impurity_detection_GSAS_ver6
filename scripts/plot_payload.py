import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


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
