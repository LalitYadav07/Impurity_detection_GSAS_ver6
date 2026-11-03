"""Run the frozen Focus U-Net over compatible full diffraction curves.

The model proposes local changes on overlapping 151-native-point windows.
Probabilities are fused on the measured coordinate grid. Source support and
reported point uncertainties then validate proposed boxes; they never create
a peak-change box. The former indicator-vote detector is not called here.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from threading import RLock

import numpy as np

from . import morphology_type_fallback, native_match, native_type_policy
from . import peak_fate_policy, source_support
from .normalization import transform_pair
from .runner import run_pair
from .stitch import fuse, native_grid, stitch


HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE / "assets" / "best.pt"
RECEIPT = HERE / "assets" / "receipt.json"
CHECKPOINT_SHA256 = "bd13855acc1c1a0e142ab111140fbe80a13b19878e9877910fb087439eeae236"
FEATURE_TYPES = (
    "peak_emergence", "peak_disappearance", "relative_intensity_change",
    "peak_splitting", "peak_merging", "peak_broadening", "peak_narrowing",
    "peak_asymmetry_change", "peak_shift", "background_change",
    "diffuse_scattering_change",
)
SCHEMA = "radar-pd-unet-transition-pair/v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _as_pattern(value) -> dict:
    pattern = getattr(value, "pattern", value)
    if isinstance(pattern, dict):
        result = dict(pattern)
        result.setdefault("y", result.get("intensity"))
        return result
    scope = pattern.scope
    scope = asdict(scope) if is_dataclass(scope) else dict(scope)
    return {"id": str(pattern.observation_id), "scope": scope,
            "x": np.asarray(pattern.x), "y": np.asarray(pattern.intensity),
            "intensity": np.asarray(pattern.intensity),
            "sigma": np.asarray(pattern.sigma),
            "valid": np.asarray(pattern.valid, dtype=bool)}


def _covered_fraction(box: tuple[float, float], support: list[list[float]]) -> float:
    lo, hi = box
    return sum(max(0., min(hi, b) - max(lo, a)) for a, b in support) / (hi - lo)


def _paired_native(before: dict, after: dict, target: tuple[float, float]) -> dict:
    """Use exact matching when possible; otherwise align on the coarser grid.

    The fallback never extrapolates or crosses a measured-support gap. It
    treats interpolated values as one coarser native bin for evidence counts.
    """
    exact = native_match.directly_matched(before, after)
    def usable(curve):
        x = np.asarray(curve["x"], dtype=float)
        y = np.asarray(curve["y"], dtype=float)
        sigma = np.asarray(curve["sigma"], dtype=float)
        return (np.asarray(curve["valid"], dtype=bool) &
                (x >= target[0]) & (x <= target[1]) &
                np.isfinite(y) & np.isfinite(sigma) & (sigma > 0))

    a, b = usable(before), usable(after)
    inside = (exact["x"] >= target[0]) & (exact["x"] <= target[1])
    exact = {name: values[inside] for name, values in exact.items()}
    minimum = min(int(a.sum()), int(b.sum()))
    if int(np.sum(exact["good"])) >= max(32, int(.8 * minimum)):
        return exact
    first, valid = (before, a) if int(a.sum()) <= int(b.sum()) else (after, b)
    x = np.asarray(first["x"], dtype=float)
    x = x[valid]
    if len(x) < 2:
        return {"x": np.array([]), "good": np.array([], dtype=bool),
                "direct_z": np.array([])}
    support = source_support.measured_support(before, after, target)["supported_intervals"]
    good = np.zeros(len(x), dtype=bool)
    for lo, hi in support:
        good |= (x >= lo) & (x <= hi)

    def interp(curve: dict, key: str) -> np.ndarray:
        cx = np.asarray(curve["x"], dtype=float)
        cv = np.asarray(curve[key], dtype=float)
        keep = np.asarray(curve["valid"], dtype=bool) & np.isfinite(cv)
        return np.interp(x, cx[keep], cv[keep])

    ay, by = interp(before, "y"), interp(after, "y")
    ae, be = interp(before, "sigma"), interp(after, "sigma")
    good &= (np.isfinite(ay) & np.isfinite(by) & np.isfinite(ae) &
             np.isfinite(be) & (ae > 0) & (be > 0))
    z = np.zeros(len(x), dtype=float)
    z[good] = (by[good] - ay[good]) / np.hypot(ae[good], be[good])
    return {"x": x, "good": good, "before_y": ay, "after_y": by,
            "before_sigma": ae, "after_sigma": be, "direct_z": z}


class _Capture:
    def __init__(self, predictor):
        self.predictor = predictor
        self.attention = []
        self.types = {name: [] for name in FEATURE_TYPES}

    def __call__(self, batch: np.ndarray) -> dict:
        output = self.predictor(batch)
        self.attention.append(np.asarray(output["attention"]).copy())
        for name in FEATURE_TYPES:
            self.types[name].append(
                np.asarray(output["type_probabilities"][name]).copy())
        return output

    def windows(self, inferred: list[dict]) -> list[dict]:
        if not inferred:
            return []
        attention = np.concatenate(self.attention)
        types = {name: np.concatenate(rows) for name, rows in self.types.items()}
        if len(attention) != len(inferred):
            raise ValueError("Model tile count differs from inference receipt")
        return [{**row, "attention": attention[i],
                 "type_probabilities": {name: values[i]
                                        for name, values in types.items()}}
                for i, row in enumerate(inferred)]


class FocusTransitionPredictor:
    """Lazy, hash-pinned CPU inference for one RadarPD process."""

    def __init__(self):
        self._model = None
        self._lock = RLock()
        self.metadata = {"name": "FocusUNet-2x156",
                         "version": CHECKPOINT_SHA256[:16],
                         "checkpoint_sha256": CHECKPOINT_SHA256,
                         "transform": "robust_linear",
                         "status": "experimental_local_morphology",
                         "label_basis": "Luna silver labels; not expert phase assignments",
                         "postprocessing_sha256": _fingerprint({
                             name: _sha(HERE / name) for name in (
                                 "production.py", "runner.py", "adapter.py", "stitch.py",
                                 "source_support.py", "native_match.py", "native_type_policy.py",
                                 "morphology_type_fallback.py", "peak_fate_policy.py", "lag_morphology.py")})}

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            if _sha(CHECKPOINT) != CHECKPOINT_SHA256:
                raise ValueError("Packaged U-Net checkpoint differs from the reviewed model")
            receipt = json.loads(RECEIPT.read_text())
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("PyTorch CPU runtime is unavailable for the U-Net tracker") from exc
            payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
            config = payload.get("config")
            if (not isinstance(config, dict) or receipt.get("config") != config or
                    payload.get("config_fingerprint") != _fingerprint(config) or
                    receipt.get("config_fingerprint") != payload.get("config_fingerprint") or
                    config.get("transform") != "robust_linear" or
                    config.get("head") != "joint" or
                    config.get("model", {}).get("architecture") != "FocusUNet" or
                    config["model"].get("base_channels") != 16 or
                    config["model"].get("input_shape") != [2, 156] or
                    config["model"].get("feature_types") != len(FEATURE_TYPES) or
                    config["model"].get("code_sha256") != _sha(HERE / "model.py") or
                    config.get("normalization_code_sha256") !=
                    {"focus_v2_normalization": _sha(HERE / "normalization.py")}):
                raise ValueError("Packaged U-Net model source or receipt differs")
            from .model import FocusUNet
            model = FocusUNet(base_channels=16, feature_types=len(FEATURE_TYPES))
            model.load_state_dict(payload["model"], strict=True)
            model.eval()
            torch.set_num_threads(min(2, torch.get_num_threads()))
            self._model = model
            return model

    def predict_batch(self, batch: np.ndarray) -> dict:
        inputs = np.asarray(batch)
        if inputs.ndim != 3 or inputs.shape[1:] != (2, 156) or \
                inputs.dtype != np.float32 or not np.isfinite(inputs).all():
            raise ValueError("U-Net input must be finite float32 [B, 2, 156]")
        model = self._load()
        import torch
        with self._lock, torch.inference_mode():
            probabilities = model(torch.from_numpy(np.ascontiguousarray(inputs))).sigmoid().numpy()
        return {"attention": probabilities[:, 0, :],
                "type_probabilities": {
                    name: probabilities[:, i + 1, :]
                    for i, name in enumerate(FEATURE_TYPES)}}

    def predict_pair(self, before, after) -> dict:
        a, b = _as_pattern(before), _as_pattern(after)
        capture = _Capture(self.predict_batch)
        base = run_pair(
            a, b, capture, self.metadata,
            window_points=151, overlap_fraction=.5, batch_size=64,
            preprocess=lambda y0, y1: transform_pair(y0, y1, "robust_linear"),
            min_native_points=32, max_windows=2048,
            attention_threshold=.8, type_threshold=.5,
            uncertainty_floor=.35, min_coverage=.98, smooth_bins=5)
        if base["status"] == "unsupported" or base["target_interval"] is None:
            return {**base, "schema": SCHEMA}
        target = tuple(base["target_interval"])
        windows = capture.windows(base["inference"]["inferred_windows"])
        grid = native_grid(a, target)
        fused = fuse(windows, grid, FEATURE_TYPES)
        raw = stitch(windows, grid, FEATURE_TYPES, fused=fused)
        inferred = [row["focus_interval"] for row in
                    base["inference"]["inferred_windows"]]
        source = source_support.measured_support(a, b, target)
        supported = source_support.intersect_intervals(inferred, source["supported_intervals"])
        matched = _paired_native(a, b, target)
        kept = []
        rejected = {"missing_source_support": 0, "insufficient_native_evidence": 0}
        for event in raw:
            lo, hi = float(event["x0"]), float(event["x1"])
            if _covered_fraction((lo, hi), supported) < 1 - 1e-6:
                rejected["missing_source_support"] += 1
                continue
            inside = (matched["good"] & (matched["x"] >= lo) &
                      (matched["x"] <= hi))
            z = np.abs(matched["direct_z"][inside])
            if (len(z) < 2 or int(np.sum(z >= 2)) < 2 or
                    float(np.sqrt(np.mean(np.minimum(z, 5)**2))) < 2.5):
                rejected["insufficient_native_evidence"] += 1
                continue
            tagged, _ = native_type_policy.tag_event(event, matched)
            tagged, _ = morphology_type_fallback.tag_unresolved_event(tagged, matched)
            kept.append(tagged)
        if kept:
            tagged, _ = peak_fate_policy.retag_prediction(
                {"prediction": {"events": kept}}, matched)
            kept = tagged["prediction"]["events"]
        coverage = sum(hi - lo for lo, hi in supported) / (target[1] - target[0])
        native_error_bins = int(np.sum(matched["good"]))
        # The runner's borderline decision describes individual tiles. The
        # displayed decision must use the same fused scores as its final boxes.
        reasons = [reason for reason in (base.get("reasons") or [])
                   if reason != "borderline_attention_score"]
        qualified = np.zeros(len(grid), dtype=bool)
        for lo, hi in supported:
            qualified |= (grid >= lo) & (grid <= hi)
        qualified &= fused["covered"]
        maximum = float(np.max(fused["attention"][qualified])) if np.any(qualified) else 0.
        if not kept and maximum >= base["model"]["uncertainty_floor"]:
            reasons.append("borderline_attention_score" if not raw else
                           "local_model_proposals_rejected_by_measured_evidence")
        normalization = str(a.get("scope", {}).get("normalization", "")).lower()
        if normalization.startswith("not_") or "not independently" in normalization:
            reasons.append("normalization_unqualified: counting or monitor changes can mimic intensity changes")
        if native_error_bins < 2:
            reasons.append("insufficient_joint_native_point_errors")
        if coverage < .98:
            reasons.append("incomplete_measured_source_support")
        status = "uncertain" if reasons else "valid"
        attention = "yes" if kept else "no" if status == "valid" else "uncertain"
        return {**base, "schema": SCHEMA, "status": status, "reasons": reasons,
                "attention": attention, "events": kept,
                "coverage_fraction": round(coverage, 6),
                "supported_intervals": supported,
                "postprocessing": {"raw_stitched_boxes": len(raw),
                                   "retained_local_boxes": len(kept),
                                   "rejected_local_boxes": rejected,
                                   "maximum_supported_fused_attention": round(maximum, 6),
                                   "native_evidence": "reported_sigma_source_support_v1",
                                   "model_created_all_local_proposals": True}}
