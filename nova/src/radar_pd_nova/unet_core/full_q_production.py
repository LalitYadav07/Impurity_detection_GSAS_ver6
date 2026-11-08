"""Profile-verified, whole-Q inference for two measured diffraction curves."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from threading import RLock

import numpy as np

from ..diffraction_coordinates import coordinate_values


HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE / "assets" / "full_q_best.pt"
RECEIPT = HERE / "assets" / "full_q_receipt.json"
CHECKPOINT_SHA256 = "3bc7a885f8e961b926933f98ad7c2f5de5866456215012945d19e2ab213c4a8b"
SCHEMA = "radar-pd-full-q-transition-pair/v1"
Q_MIN, Q_MAX, BINS = 0.25, 12.0, 1024
Q_GRID = np.linspace(Q_MIN, Q_MAX, BINS, dtype=np.float64)
Q_STEP = float(Q_GRID[1] - Q_GRID[0])
Q_EDGES = np.r_[Q_GRID - Q_STEP / 2, Q_GRID[-1] + Q_STEP / 2]
LOCAL_NAMES = ("attention", "peak_emergence", "relative_intensity_change",
               "peak_broadening", "peak_shift")
CAUSE_NAMES = ("lattice_change_candidate", "magnetic_order_candidate",
               "new_phase_candidate", "structural_change_candidate")
LOCAL_THRESHOLDS = {"attention": 0.85, "peak_emergence": 0.95,
                    "relative_intensity_change": 0.90,
                    "peak_broadening": 0.95, "peak_shift": 0.95}
CAUSE_THRESHOLDS = {"lattice_change_candidate": 0.95,
                    "magnetic_order_candidate": 0.40,
                    "new_phase_candidate": 0.95,
                    "structural_change_candidate": 0.10}


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _spans(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return [(int(a), int(b)) for a, b in edges.reshape(-1, 2)]


def _primitive(x, y, points):
    idx = np.clip(np.searchsorted(x, points, side="right") - 1, 0, len(x) - 2)
    widths = np.diff(x)
    cumulative = np.r_[0.0, np.cumsum((y[:-1] + y[1:]) * widths * 0.5)]
    dx = points - x[idx]
    slope = (y[idx + 1] - y[idx]) / widths[idx]
    return cumulative[idx] + y[idx] * dx + slope * dx * dx * 0.5


def rebin_scan(q, intensity, valid):
    """Match training's area-averaged Q bins without bridging invalid runs."""
    q = np.asarray(q, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if (q.ndim != 1 or len(q) < 2 or q.shape != intensity.shape or
            q.shape != valid.shape or not np.isfinite(q).all() or
            not np.isfinite(intensity[valid]).all()):
        raise ValueError("Measured Q curve has invalid coordinates or intensities")
    order = np.argsort(q)
    q, intensity, valid = q[order], intensity[order], valid[order]
    if np.any(np.diff(q) <= 0):
        raise ValueError("Converted measured Q coordinates are not unique")
    result = np.zeros(BINS, dtype=np.float32)
    covered = np.zeros(BINS, dtype=bool)
    for start, stop in _spans(valid):
        if stop - start < 2:
            continue
        x, y = q[start:stop], intensity[start:stop]
        bins = np.flatnonzero((Q_EDGES[:-1] >= x[0]) & (Q_EDGES[1:] <= x[-1]))
        if not len(bins):
            continue
        left = _primitive(x, y, Q_EDGES[bins])
        right = _primitive(x, y, Q_EDGES[bins + 1])
        result[bins] = ((right - left) / Q_STEP).astype(np.float32)
        covered[bins] = True
    return result, covered


def prepare_pair(before, after, calibration_before=None, calibration_after=None):
    """Return the exact three training channels and their joint support mask."""
    a = getattr(before, "pattern", before)
    b = getattr(after, "pattern", after)
    if a.scope != b.scope:
        raise ValueError("Measured scans have incompatible comparison provenance")
    for pattern, calibration in ((a, calibration_before), (b, calibration_after)):
        if pattern.scope.coordinate in {"tof_us", "two_theta_deg"}:
            if calibration is None:
                raise ValueError("Exact instrument profile is required for whole-Q inference")
            if str(calibration.source_bank) != pattern.scope.bank:
                raise ValueError("Instrument profile belongs to another physical bank")
    if (calibration_before is not None and calibration_after is not None and
            calibration_before.profile_sha256 != calibration_after.profile_sha256):
        raise ValueError("Compared scans use different instrument profiles")
    qa = coordinate_values(a.x, a.scope.coordinate, "q_inv_angstrom", calibration_before)
    qb = coordinate_values(b.x, b.scope.coordinate, "q_inv_angstrom", calibration_after)
    before_y, before_mask = rebin_scan(qa, a.intensity, a.valid)
    after_y, after_mask = rebin_scan(qb, b.intensity, b.valid)
    coverage = before_mask & after_mask
    if int(coverage.sum()) < 32:
        raise ValueError("Fewer than 32 jointly measured Q bins between 0.25 and 12 Å⁻¹")
    before_y[~coverage] = 0
    after_y[~coverage] = 0
    joint = np.r_[before_y[coverage], after_y[coverage]].astype(np.float64)
    baseline = float(np.percentile(joint, 5.0))
    residual = joint - baseline
    divisor = float(np.percentile(residual, 99.5))
    if not np.isfinite(divisor) or divisor <= 1e-6:
        divisor = float(np.max(residual))
    if not np.isfinite(divisor) or divisor <= 1e-6:
        divisor = 1.0
    first = (before_y - baseline) / divisor
    second = (after_y - baseline) / divisor
    x = np.stack((first, second, second - first)).astype(np.float32)
    x *= coverage[None, :]
    if not np.isfinite(x).all():
        raise ValueError("Whole-Q model input contains a nonfinite value")
    return np.ascontiguousarray(x), np.ascontiguousarray(coverage.astype(np.float32)), {
        "baseline": baseline, "divisor": divisor,
        "covered_bins": int(coverage.sum()), "coverage_fraction": float(np.mean(coverage)),
        "q_coverage_intervals": [[float(Q_EDGES[lo]), float(Q_EDGES[hi])]
                                 for lo, hi in _spans(coverage)],
    }


class FullQTransitionPredictor:
    """Lazy, checksum-pinned CPU inference with validation-selected thresholds."""

    def __init__(self):
        self._model = None
        self._lock = RLock()
        self.metadata = {"name": "FullQUNet-3x1024",
                         "version": CHECKPOINT_SHA256[:16],
                         "checkpoint_sha256": CHECKPOINT_SHA256,
                         "transform": "pair_shared_affine_p995",
                         "status": "experimental_synthetic_cause_and_local_morphology",
                         "label_basis": "deterministic synthetic labels; cause scores are not calibrated probabilities",
                         "postprocessing_sha256": sha256(
                             (Path(__file__).read_bytes()
                              + (HERE / "full_q_model.py").read_bytes()
                              + RECEIPT.read_bytes())).hexdigest()}

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            if _sha(CHECKPOINT) != CHECKPOINT_SHA256:
                raise ValueError("Packaged whole-Q checkpoint differs from the reviewed model")
            receipt = json.loads(RECEIPT.read_text())
            if (receipt.get("checkpoint_sha256") != CHECKPOINT_SHA256 or
                    receipt.get("local_thresholds") != LOCAL_THRESHOLDS or
                    receipt.get("cause_thresholds") != CAUSE_THRESHOLDS):
                raise ValueError("Whole-Q model receipt or thresholds changed")
            import torch
            from .full_q_model import FullCurveUNet
            payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
            config = payload.get("config") or {}
            if (config.get("base") != 32 or config.get("multiscale") is not False or
                    config.get("attention") is not False or config.get("scale") != "p995" or
                    int(payload.get("epoch", -1)) != 20):
                raise ValueError("Whole-Q checkpoint architecture or normalization changed")
            model = FullCurveUNet()
            model.load_state_dict(payload["model"], strict=True)
            model.eval()
            self._model = model
            return model

    def predict_pair(self, before, after, *, calibration_before=None,
                     calibration_after=None):
        x, coverage, stats = prepare_pair(before, after,
                                           calibration_before, calibration_after)
        model = self._load()
        import torch
        with self._lock, torch.inference_mode():
            local_logits, cause_logits = model(
                torch.from_numpy(x[None]), torch.from_numpy(coverage[None]))
            local = torch.sigmoid(local_logits)[0].numpy()
            cause = torch.sigmoid(cause_logits)[0].numpy()
        valid = coverage.astype(bool)
        events = []
        for start, stop in _spans((local[0] >= LOCAL_THRESHOLDS["attention"]) & valid):
            labels = [(name, float(np.max(local[index, start:stop])))
                      for index, name in enumerate(LOCAL_NAMES[1:], 1)]
            labels = [name for name, score in sorted(labels, key=lambda item: item[1], reverse=True)
                      if score >= LOCAL_THRESHOLDS[name]][:2]
            q0, q1 = float(Q_EDGES[start]), float(Q_EDGES[stop])
            pattern = getattr(after, "pattern", after)
            native = coordinate_values([q0, q1], "q_inv_angstrom",
                                       pattern.scope.coordinate, calibration_after)
            x0, x1 = sorted(map(float, native))
            events.append({"x0": x0, "x1": x1, "q0": q0, "q1": q1,
                           "feature_types": labels,
                           "attention_score": float(np.max(local[0, start:stop])),
                           "type_scores": {name: score for name, score in
                                           [(name, float(np.max(local[i, start:stop])))
                                            for i, name in enumerate(LOCAL_NAMES[1:], 1)]},
                           "source_windows": ["full-Q scan"]})
        causes = [{"name": name, "score": float(cause[i]),
                   "selected": bool(cause[i] >= CAUSE_THRESHOLDS[name])}
                  for i, name in enumerate(CAUSE_NAMES)]
        profiles = [{"source_bank": cal.source_bank,
                     "profile_bank": cal.profile_bank,
                     "sha256": cal.profile_sha256}
                    for cal in (calibration_before, calibration_after) if cal is not None]
        reasons = []
        if stats["covered_bins"] < BINS:
            reasons.append("The model evaluated only jointly measured Q bins")
        return {"schema": SCHEMA, "status": "valid", "attention": "yes" if events else "no",
                "events": events, "causes": causes, "profiles": profiles,
                "reasons": reasons,
                "coverage_fraction": round(stats["coverage_fraction"], 6),
                "q_coverage_intervals": stats["q_coverage_intervals"],
                "supported_intervals": stats["q_coverage_intervals"],
                "inference": {"passes": 1, "input_bins": BINS,
                              "covered_bins": stats["covered_bins"]},
                "postprocessing": {"attention_threshold": LOCAL_THRESHOLDS["attention"],
                                   "cause_thresholds": CAUSE_THRESHOLDS,
                                   "all_regions_from_model": True}}
