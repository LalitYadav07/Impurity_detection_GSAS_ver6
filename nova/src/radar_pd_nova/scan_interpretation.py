"""Replayable, evidence-linked interpretation of completed scan observations.

Detectors report observed features and competing explanations, not chemical
truth, calibrated probabilities or instrument commands. Arrays and fitted
models have separate compatibility checks. No GPX pickle or LLM is consumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .facility_metadata import parse_nexus_timestamp

PATTERN_SCHEMA = "radar-pd-pattern-observation/v1"
REPORT_SCHEMA = "radar-pd-scan-interpretation/v1"
ENGINE_VERSION = "scan-interpretation/1"
MAX_BYTES = 16_000_000


def digest(value):
    return sha256(
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


class InterpretationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1] = 1
    enabled: bool = True
    min_points: int = Field(default=64, ge=32, le=10000)
    peak_error_ratio: float = Field(default=8, ge=3, le=100)
    difference_error_ratio: float = Field(default=6, ge=3, le=100)
    min_peak_fraction: float = Field(default=0.02, gt=0, le=0.5)
    shape_effect: float = Field(default=0.03, gt=0, le=1)
    peak_shift_bins: float = Field(default=0.5, ge=0.2, le=10)
    width_fraction: float = Field(default=0.3, ge=0.1, le=3)
    intensity_fraction: float = Field(default=0.25, ge=0.05, le=3)
    scale_fraction: float = Field(default=0.15, ge=0.01, le=3)
    background_fraction: float = Field(default=0.1, ge=0.01, le=2)
    cell_fraction: float = Field(default=0.0001, gt=0, le=0.1)
    fraction_points: float = Field(default=1, gt=0, le=50)
    max_history: int = Field(default=2000, ge=2, le=10000)
    history_search_limit: int = Field(default=32, ge=2, le=128)
    condition_tolerances: dict[str, float] = Field(
        default_factory=lambda: {"temperature": 1, "magnetic_field": 0.02, "voltage": 0.05}
    )
    condition_bindings: dict[str, dict[str, str]] = Field(default_factory=dict)
    references: dict[str, str] = Field(default_factory=dict)
    reference_hashes: dict[str, str] = Field(default_factory=dict)
    active_revisions: dict[str, str] = Field(default_factory=dict)
    reviewed_candidate_equivalences: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def finite_tolerances(self):
        if any(not math.isfinite(v) or v <= 0 for v in self.condition_tolerances.values()):
            raise ValueError("Condition tolerances must be finite and positive")
        if set(self.condition_tolerances) - {
            "temperature",
            "magnetic_field",
            "voltage",
            "pressure",
            "gas_flow",
        }:
            raise ValueError("Choose a registered condition quantity")
        if any(
            not str(k).strip() or not str(v).strip()
            for k, v in self.reviewed_candidate_equivalences.items()
        ):
            raise ValueError("Reviewed candidate equivalences need explicit identities")
        for binding in self.condition_bindings.values():
            if set(binding) - {
                "temperature",
                "magnetic_field",
                "voltage",
                "pressure",
                "gas_flow",
            } or any(not v.strip() for v in binding.values()):
                raise ValueError(
                    "Condition bindings require registered quantities and explicit readback channels"
                )
        return self


@dataclass
class Pattern:
    packet: dict
    key: str
    x: np.ndarray
    y: np.ndarray
    sigma: np.ndarray
    valid: np.ndarray
    fitted: np.ndarray
    calculated: np.ndarray

    @classmethod
    def parse(cls, packet):
        if packet.get("$schema") != PATTERN_SCHEMA:
            raise ValueError(
                "A current pattern observation is required; republish the result with a current Analyze image"
            )
        x = np.asarray(packet.get("x"), dtype=float)
        n = len(x) if x.ndim == 1 else 0
        if not 16 <= n <= 100000 or not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
            raise ValueError(
                "Curve coordinates must be finite, increasing and within the supported bounds"
            )
        arrays = {}
        for name in ["y", "sigma", "ycalc", "valid", "fit_valid"]:
            raw = packet.get(name)
            if not isinstance(raw, list) or len(raw) != n:
                raise ValueError(f"Pattern {name} does not match the coordinate grid")
            if name in {"valid", "fit_valid"}:
                if any(not isinstance(v, bool) for v in raw):
                    raise ValueError("Curve masks must contain explicit booleans")
                arrays[name] = np.asarray(raw, dtype=bool)
            else:
                arrays[name] = np.asarray(raw, dtype=float)
        valid = arrays["valid"]
        if np.any(~np.isfinite(arrays["y"][valid])) or np.any(arrays["fit_valid"] & ~valid):
            raise ValueError("Measured data or fitted masks are inconsistent")
        positive = np.isfinite(arrays["sigma"]) & (arrays["sigma"] > 0)
        arrays["sigma"] = np.where(positive, arrays["sigma"], np.nan)
        return cls(
            packet,
            digest(packet),
            x,
            arrays["y"],
            arrays["sigma"],
            valid,
            arrays["fit_valid"],
            arrays["ycalc"],
        )


def load_pattern(path, *, expected_sha256=None):
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Curve observation exceeds its bounded size")
    raw = path.read_bytes()
    if len(raw) > MAX_BYTES:
        raise ValueError("Curve observation changed or exceeds its bounded size")
    if expected_sha256 and sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Curve observation checksum disagrees with its result receipt")
    return json.loads(raw)


def _segments(mask):
    padded = np.r_[False, mask, False].astype(int)
    return list(zip(np.where(np.diff(padded) == 1)[0], np.where(np.diff(padded) == -1)[0]))


def _peaks(pattern, settings):
    """Local feature measurements; thresholds are engineering sensitivity settings.

    Work separately inside valid contiguous segments. No smoothing across a
    masked/excluded gap, nor inference of physical size/strain from widths.
    """
    peaks = []
    for begin, end in _segments(pattern.valid):
        if end - begin < 16:
            continue
        y = pattern.y[begin:end]
        width = min(51, (len(y) // 3) * 2 + 1)
        half = width // 2
        windows = np.lib.stride_tricks.sliding_window_view(np.pad(y, half, mode="edge"), width)
        baseline = np.quantile(windows, 0.2, axis=1)
        excess = y - baseline
        smooth = np.convolve(
            np.pad(excess, 2, mode="edge"), [1 / 9, 2 / 9, 3 / 9, 2 / 9, 1 / 9], mode="valid"
        )
        span = max(float(np.max(smooth)), 1e-30)
        sigma = pattern.sigma[begin:end]
        # Reported errors can be conditional/correlated after reduction. The
        # ratio is diagnostic, never a p-value or a calibrated confidence.
        threshold = np.where(np.isfinite(sigma), settings.peak_error_ratio * sigma, 0)
        threshold = np.maximum(threshold, settings.min_peak_fraction * span)
        indices = np.where((smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] >= smooth[2:]))[0] + 1
        for i in indices:
            if i < 3 or i >= len(y) - 3 or smooth[i] <= threshold[i]:
                continue
            left, right = i, i
            while left > 0 and smooth[left] > smooth[i] / 2:
                left -= 1
            while right < len(y) - 1 and smooth[right] > smooth[i] / 2:
                right += 1
            if left == 0 or right == len(y) - 1 or right - left < 2:
                continue
            ii = begin + i
            xx = pattern.x
            denominator = smooth[i - 1] - 2 * smooth[i] + smooth[i + 1]
            displacement = 0.5 * (smooth[i - 1] - smooth[i + 1]) / denominator if denominator else 0
            position = (
                xx[ii] + float(np.clip(displacement, -0.5, 0.5)) * (xx[ii + 1] - xx[ii - 1]) / 2
            )
            interval = [float(xx[begin + left]), float(xx[begin + right])]
            area = float(np.sum(np.maximum(excess[left : right + 1], 0)))
            peaks.append(
                {
                    "x": float(position),
                    "height": float(smooth[i]),
                    "area": area,
                    "width": interval[1] - interval[0],
                    "region": interval,
                    "bin_width": float((xx[ii + 1] - xx[ii - 1]) / 2),
                    "reported_error_ratio": (
                        float(smooth[i] / sigma[i]) if np.isfinite(sigma[i]) else None
                    ),
                }
            )
    # Closely spaced maxima from a noisy shoulder are not independent peaks.
    chosen = []
    for peak in sorted(peaks, key=lambda v: -v["height"]):
        if all(
            abs(peak["x"] - other["x"]) > min(peak["width"], other["width"]) * 0.45
            for other in chosen
        ):
            chosen.append(peak)
    return sorted(chosen[:256], key=lambda v: v["x"])


def compare_patterns(left, right, settings):
    unit = right.packet.get("coordinate", {})
    if (
        unit != left.packet.get("coordinate")
        or unit.get("kind") not in {"tof", "two_theta", "d", "q"}
        or not unit.get("unit")
    ):
        return {
            "status": "unavailable",
            "reason": "Coordinate units or meanings are not compatible",
        }
    # Match existing bins; no arbitrary interpolation that could manufacture
    # splitting, hide motion or invent independent uncertainty is performed.
    positions = np.searchsorted(right.x, left.x)
    ok = positions < len(right.x)
    safe = np.minimum(positions, len(right.x) - 1)
    ok &= np.isclose(left.x, right.x[safe], rtol=1e-9, atol=1e-8)
    a = np.where(ok)[0]
    b = positions[ok]
    mask = left.valid[a] & right.valid[b]
    a, b = a[mask], b[mask]
    if len(a) < settings.min_points or len(a) < 0.5 * min(np.sum(left.valid), np.sum(right.valid)):
        return {
            "status": "unavailable",
            "reason": "Too few matching valid bins; review binning, coverage and masks before comparing",
        }
    y1, y2 = left.y[a], right.y[b]
    if np.ptp(y1) <= 1e-12 or np.ptp(y2) <= 1e-12:
        return {
            "status": "unavailable",
            "reason": "Insufficient resolved pattern variation for a shape comparison",
        }
    scale, offset = np.linalg.lstsq(np.column_stack([y1, np.ones(len(y1))]), y2, rcond=None)[0]
    if not np.isfinite(scale) or scale <= 0:
        return {
            "status": "unavailable",
            "reason": "A positive shared intensity scale could not be established",
        }
    residual = y2 - scale * y1 - offset
    denominator = max(float(np.std(y2)), 1e-30)
    result = {
        "status": "compared",
        "points": len(a),
        "coordinate": unit,
        "fit_domain_compatible": left.packet.get("fit_domain") == right.packet.get("fit_domain"),
        "pearson": float(np.corrcoef(y1, y2)[0, 1]),
        "scale": float(scale),
        "offset": float(offset),
        "shape_effect": float(np.sqrt(np.mean(residual**2)) / denominator),
        "background_effect": float(offset / denominator),
        "diagnostic_only": True,
        "new_features": [],
        "lost_features": [],
        "matched_peaks": [],
        "splitting_candidates": [],
        "ambiguous_feature_matches": 0,
        "unexplained_features": [],
    }
    sigma = np.sqrt(right.sigma[b] ** 2 + scale**2 * left.sigma[a] ** 2)
    errors_valid = np.isfinite(sigma) & (sigma > 0)
    result["error_coverage_fraction"] = float(np.mean(errors_valid))
    if np.all(errors_valid):
        result["reported_error_scaled_difference_rms"] = float(
            np.sqrt(np.mean((residual / sigma) ** 2))
        )
        combined = float(np.median(sigma))
        result["offset_error_ratio"] = float(abs(offset) / combined)
        result["scale_error_ratio"] = float(
            abs(scale - 1) * np.quantile(np.abs(y1 - np.median(y1)), 0.99) / combined
        )

    def evidence(region):
        selected = (left.x[a] >= region[0]) & (left.x[a] <= region[1]) & errors_valid
        # Contiguous native bins only. A conservative linear combination of
        # reported errors avoids pretending rebinned bins are independent.
        indices = np.where(selected)[0]
        scores = []
        for contiguous in np.split(indices, np.where(np.diff(a[indices]) != 1)[0] + 1):
            if len(contiguous) < 3:
                continue
            width = min(5, len(contiguous))
            numerator = np.convolve(residual[contiguous], np.ones(width) / width, mode="valid")
            denominator = np.convolve(sigma[contiguous], np.ones(width) / width, mode="valid")
            scores.extend(np.abs(numerator / denominator).tolist())
        return float(max(scores)) if scores else 0.0

    # Restrict each feature detector to the same bins and do not bridge gaps.
    def common(pattern, indices):
        valid = np.zeros(len(pattern.x), dtype=bool)
        valid[indices] = True
        return Pattern(
            pattern.packet,
            pattern.key,
            pattern.x,
            pattern.y,
            pattern.sigma,
            valid,
            pattern.fitted,
            pattern.calculated,
        )

    p1, p2 = _peaks(common(left, a), settings), _peaks(common(right, b), settings)
    matches, used, ambiguous = [], set(), set()
    for old in p1:
        candidates = sorted(
            [(abs(new["x"] - old["x"]), j, new) for j, new in enumerate(p2) if j not in used],
            key=lambda v: v[0],
        )
        bin_width = old["bin_width"]
        tolerance = max(old["width"] * 2, abs(old["x"]) * 0.01, 3 * bin_width)
        if not candidates or candidates[0][0] > tolerance:
            result["lost_features"].append(old)
            continue
        if (
            len(candidates) > 1
            and candidates[1][0] < tolerance
            and candidates[1][0] < 1.5 * max(candidates[0][0], bin_width)
        ):
            ambiguous.update(j for distance, j, _ in candidates if distance < tolerance)
            result["ambiguous_feature_matches"] += 1
            continue
        _, j, new = candidates[0]
        used.add(j)
        matches.append(
            {
                "previous": old,
                "current": new,
                "shift": new["x"] - old["x"],
                "difference_error_ratio": evidence(
                    [
                        min(old["region"][0], new["region"][0]),
                        max(old["region"][1], new["region"][1]),
                    ]
                ),
                "relative_shift": (new["x"] - old["x"]) / abs(old["x"]) if old["x"] else None,
                "shift_resolution": settings.peak_shift_bins
                * max(old["bin_width"], new["bin_width"]),
                "width_ratio": new["width"] / old["width"],
                "relative_area": new["area"] / (scale * old["area"]) if old["area"] > 0 else None,
            }
        )
    result["matched_peaks"] = matches
    result["new_features"] = [peak for j, peak in enumerate(p2) if j not in used | ambiguous]
    for name in ["new_features", "lost_features"]:
        for peak in result[name]:
            peak["difference_error_ratio"] = evidence(peak["region"])
        result[name] = [
            p
            for p in result[name]
            if p["difference_error_ratio"] >= settings.difference_error_ratio
        ]
    for old in p1:
        near = [peak for peak in p2 if abs(peak["x"] - old["x"]) < old["width"] * 1.5]
        old_near = [peak for peak in p1 if abs(peak["x"] - old["x"]) < old["width"] * 1.5]
        if (
            len(near) == 2
            and len(old_near) == 1
            and near[1]["x"] - near[0]["x"] > 2 * old["bin_width"]
        ):
            score = evidence(
                [
                    min(old["region"][0], near[0]["region"][0]),
                    max(old["region"][1], near[-1]["region"][1]),
                ]
            )
            if score >= settings.difference_error_ratio:
                result["splitting_candidates"].append(
                    {"previous": old, "current": near, "difference_error_ratio": score}
                )

    # Persistent positive residual features are evidence of a model mismatch,
    # not proof of a missing chemical phase. Only common fitted coverage counts.
    def residual_peaks(pattern):
        mask = pattern.fitted & np.isfinite(pattern.calculated)
        y = np.where(mask, pattern.y - pattern.calculated, 0)
        return _peaks(
            Pattern(
                pattern.packet,
                pattern.key,
                pattern.x,
                y,
                pattern.sigma,
                mask,
                mask,
                pattern.calculated,
            ),
            settings,
        )

    if left.packet.get("fit_domain") == right.packet.get("fit_domain"):
        old_residual = residual_peaks(left)
        for peak in residual_peaks(right):
            if any(
                abs(peak["x"] - old["x"]) < max(peak["width"], old["width"]) / 2
                for old in old_residual
            ):
                result["unexplained_features"].append(peak)
    return result


def _time(scan):
    try:
        value = parse_nexus_timestamp((scan.get("metadata") or {}).get("start_time") or "")
        return value.timestamp() if value.tzinfo else None
    except ValueError:
        return None


def _run(scan):
    return str(scan.get("run_id") or scan.get("run_name") or scan.get("run_number") or "unknown")


def condition_scope(scan):
    return ":".join(
        str(v or "")
        for v in [
            scan.get("instrument"),
            scan.get("ipts"),
            (scan.get("metadata") or {}).get("sample_id"),
        ]
    )


def _conditions(scan, settings=None):
    from .scan_conditions import qualified_metadata
    metadata = qualified_metadata(scan.get("metadata") or {})
    packet = metadata.get("conditions") or {}
    result = {}
    bindings = settings.condition_bindings.get(condition_scope(scan), {}) if settings else {}
    for channel in packet.get("channels", []):
        quantity = channel.get("quantity")
        selected = bindings.get(quantity)
        if channel.get("role") != "measured" or (selected and selected != channel.get("channel")):
            continue
        if channel.get("status") == "recorded":
            result[quantity] = channel
        elif (
            channel.get("status") == "needs_mapping"
            and selected == channel.get("channel")
            and channel.get("statistic") == "time_weighted_step_hold"
            and channel.get("coverage_fraction", 0) >= 0.95
        ):
            result[quantity] = {
                **channel,
                "status": "recorded",
                "specimen_mapping": "explicit user-verified device binding",
            }
    declared = {c.get("quantity") for c in packet.get("channels", [])}
    for quantity in ["temperature", "magnetic_field", "voltage", "pressure", "gas_flow"]:
        legacy = metadata.get(quantity)
        if (
            quantity not in declared
            and isinstance(legacy, dict)
            and legacy.get("value") is not None
        ):
            result[quantity] = {
                **legacy,
                "quantity": quantity,
                "status": "summary_only",
                "role": "measured",
            }
    return result


def _group(scan, pattern):
    metadata = scan.get("metadata") or {}
    context = pattern.packet.get("context") or {}
    observable_context = (scan.get("observables") or {}).get("context") or {}
    model_project = (scan.get("observables") or {}).get("source_gpx_sha256")
    curve_project = pattern.packet.get("source_gpx_sha256")
    if model_project and curve_project and model_project != curve_project:
        raise ValueError("Measured curve and fitted summary belong to different projects")
    for field, declared in [
        ("instrument", scan.get("instrument")),
        ("source_bank", scan.get("bank")),
    ]:
        if (
            declared is not None
            and context.get(field) is not None
            and str(declared) != str(context[field])
        ):
            raise ValueError(
                "Pattern receipt disagrees with the selected instrument or physical bank"
            )
    identity = str(metadata.get("sample_id") or "").strip()
    if metadata.get("identity_status") == "conflict":
        identity = ""
    bank = scan.get("bank") or metadata.get("source_bank") or context.get("source_bank")
    profile = context.get("profile_sha256") or observable_context.get("profile_sha256")
    if (
        context.get("profile_sha256")
        and observable_context.get("profile_sha256")
        and context["profile_sha256"] != observable_context["profile_sha256"]
    ):
        raise ValueError("Pattern and fitted-observation profile receipts disagree")
    family = (
        context.get("reduction_family")
        or scan.get("source_family")
        or observable_context.get("reduction_key")
    )
    values = [
        scan.get("instrument") or context.get("instrument"),
        scan.get("ipts"),
        identity,
        bank,
        profile,
        family,
        pattern.packet.get("coordinate"),
        context.get("calibration"),
        context.get("normalization"),
        context.get("reduction_setup"),
        scan.get("comparison_setup_id"),
        metadata.get("sample_role"),
    ]
    if (
        not identity
        or not values[0]
        or not values[1]
        or not bank
        or not profile
        or not family
        or str(family).lower() in {"unknown", "n/a", "unavailable"}
    ):
        raise ValueError(
            "Specimen, physical bank, profile and reduction-family identity are required for a physical series comparison"
        )
    if metadata.get("source_revision_warning"):
        raise ValueError(
            "The source reduction changed; review the retained result before comparing"
        )
    return digest(values)


def _candidates(scan):
    rows = (scan.get("observables") or {}).get("phases") or scan.get("lattice_parameters") or []
    return {
        str(row.get("phase_id") or row.get("phase_key")): row
        for row in rows
        if row.get("phase_id") or row.get("phase_key")
    }


def _finding(code, title, evidence, alternatives=(), regions=(), status="observed"):
    return {
        "code": code,
        "title": title,
        "status": status,
        "evidence": list(evidence),
        "alternatives": list(alternatives),
        "regions": list(regions),
    }


def _model_changes(before, after, settings):
    left, right = _candidates(before), _candidates(after)
    canonical = lambda values: {
        settings.reviewed_candidate_equivalences.get(key, key) for key in values
    }
    changed = set(left) != set(right)
    result = {
        "candidate_ids_changed": changed,
        "reviewed_identity_changed": canonical(left) != canonical(right),
        "added": sorted(set(right) - set(left)),
        "removed": sorted(set(left) - set(right)),
        "cell_changes": [],
        "fraction_changes": [],
        "fraction_basis_compatible": False,
        "rwp_difference": None,
    }
    old_obs, new_obs = before.get("observables") or {}, after.get("observables") or {}
    recipe = old_obs.get("context", {}).get("recipe_key")
    same_recipe = bool(recipe) and recipe == new_obs.get("context", {}).get("recipe_key")
    same_recipe = (
        same_recipe
        and bool(before.get("configuration_id"))
        and before.get("configuration_id") == after.get("configuration_id")
    )
    same_recipe = same_recipe and before.get("candidate_library_id") == after.get(
        "candidate_library_id"
    )
    same_basis = bool(old_obs.get("fraction_basis")) and old_obs.get(
        "fraction_basis"
    ) == new_obs.get("fraction_basis")
    result["fraction_basis_compatible"] = same_basis and same_recipe and not changed
    if same_recipe:
        try:
            delta = float(after["rwp"]) - float(before["rwp"])
            if math.isfinite(delta):
                result["rwp_difference"] = delta
        except (KeyError, ValueError, TypeError):
            pass
    for key in set(left) & set(right):
        a, b = left[key], right[key]
        if (
            result["fraction_basis_compatible"]
            and a.get("fraction_state") == b.get("fraction_state") == "refined"
        ):
            try:
                delta = float(b["weight_percent"]) - float(a["weight_percent"])
                if math.isfinite(delta):
                    result["fraction_changes"].append(
                        {"phase_id": key, "percentage_point_difference": delta}
                    )
            except (KeyError, TypeError, ValueError):
                pass
        if not a.get("basis_key") or a.get("basis_key") != b.get("basis_key") or not same_recipe:
            continue
        for metric in ["a", "b", "c", "volume"]:
            if a.get("metric_states", {}).get(metric) not in {
                "refined",
                "constraint-derived",
            } or b.get("metric_states", {}).get(metric) not in {"refined", "constraint-derived"}:
                continue
            try:
                old, new = float(a[metric]), float(b[metric])
                if not math.isfinite(old) or not math.isfinite(new) or old <= 0:
                    continue
                result["cell_changes"].append(
                    {
                        "phase_id": key,
                        "metric": metric,
                        "previous": old,
                        "current": new,
                        "relative_change": (new - old) / old,
                        "uncertainty_qualified": bool(
                            a.get("covariance_fresh")
                            and b.get("covariance_fresh")
                            and a.get("covariance_status")
                            == b.get("covariance_status")
                            == "checked_against_gsas_esds"
                        ),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return result


def explain_comparison(before, after, features, settings):
    findings = []
    if features.get("status") != "compared":
        return [
            _finding(
                "comparison_unresolved",
                "Curve comparison needs review",
                [features.get("reason", "No comparable curve")],
                status="unassessed",
            )
        ]
    model = _model_changes(before, after, settings)
    if not features.get("fit_domain_compatible"):
        model.update(cell_changes=[], fraction_changes=[])
        findings.append(
            _finding(
                "fit_domain_changed",
                "The fitted domains differ",
                ["Fit intervals, excluded regions or weight factors changed"],
                [
                    "Model fractions, cells and Rwp differences are withheld; measured curves still use common valid bins"
                ],
                status="review_required",
            )
        )
    if features.get("error_coverage_fraction", 0) < 1:
        findings.append(
            _finding(
                "difference_errors_unavailable",
                "Curve differences have incomplete reported errors",
                [
                    f"Error coverage {100*features.get('error_coverage_fraction',0):.3g}% of compared bins"
                ],
                [
                    "Local events require adequate errors; no missing error was replaced with an invented value"
                ],
                status="unassessed",
            )
        )
    peaks = features["matched_peaks"]
    moving = [
        peak
        for peak in peaks
        if abs(peak["shift"]) > peak["shift_resolution"]
        and peak["difference_error_ratio"] >= settings.difference_error_ratio
    ]
    if len(moving) >= 2:
        findings.append(
            _finding(
                "peak_movement",
                "Matched diffraction features moved",
                [f"{len(moving)} local features moved beyond the configured bin-scale threshold"],
                ["Lattice response and calibration drift can both move peaks"],
                [p["current"]["region"] for p in moving],
                status="candidate",
            )
        )
    for key, title in [
        ("new_features", "Additional local diffraction features"),
        ("lost_features", "Previously resolved features are weaker or unresolved"),
    ]:
        if features[key]:
            findings.append(
                _finding(
                    key,
                    title,
                    [f"{len(features[key])} candidate features in common measured coverage"],
                    [
                        "Conditional on local background, resolution and reported errors",
                        "Feature persistence is reported separately; phase identity is not established",
                    ],
                    [p["region"] for p in features[key]],
                    status="candidate",
                )
            )
    widening = [
        p
        for p in peaks
        if abs(p["width_ratio"] - 1) > settings.width_fraction
        and p["difference_error_ratio"] >= settings.difference_error_ratio
    ]
    if widening or features["splitting_candidates"]:
        findings.append(
            _finding(
                "profile_shape_change",
                "Local peak width or shape changed",
                [
                    f"{len(widening)} width changes; {len(features['splitting_candidates'])} possible two-maximum features"
                ],
                [
                    "Overlapping reflections, mixed acquisition conditions and instrument resolution remain alternatives",
                    "This does not measure crystallite size, strain or symmetry",
                ],
                [p["current"]["region"] for p in widening]
                + [p["previous"]["region"] for p in features["splitting_candidates"]],
                status="candidate",
            )
        )
    intensity = [
        p
        for p in peaks
        if p["relative_area"] is not None
        and abs(p["relative_area"] - 1) > settings.intensity_fraction
        and p["difference_error_ratio"] >= settings.difference_error_ratio
    ]
    if len(peaks) >= 3 and intensity:
        findings.append(
            _finding(
                "relative_intensity_redistribution",
                "Relative peak intensities changed",
                [
                    f"{len(intensity)} matched feature areas changed after retaining a separate global scale"
                ],
                [
                    "Phase populations, texture, magnetic scattering and structural changes can affect intensities"
                ],
                [p["current"]["region"] for p in intensity],
                status="candidate",
            )
        )
    if (
        abs(features["scale"] - 1) > settings.scale_fraction
        and features.get("scale_error_ratio", 0) >= settings.difference_error_ratio
    ):
        findings.append(
            _finding(
                "global_scale_change",
                "Overall intensity scale changed",
                [
                    f"Least-squares intensity scale {features['scale']:.4g}; additive offset {features['offset']:.4g}"
                ],
                [
                    "Check exposure, normalization, attenuation and assembly before interpreting a material change"
                ],
            )
        )
    if (
        abs(features["background_effect"]) > settings.background_fraction
        and features.get("offset_error_ratio", 0) >= settings.difference_error_ratio
    ):
        findings.append(
            _finding(
                "background_change",
                "Global additive difference in the pattern",
                [
                    f"Additive offset {features['offset']:.4g}; {features['background_effect']:.3g} of pattern standard deviation"
                ],
                [
                    "The fitted offset also couples to intensity redistribution; diffuse scattering, assembly and reduction remain alternatives"
                ],
                status="candidate",
            )
        )
    cells = [
        v for v in model["cell_changes"] if abs(v["relative_change"]) >= settings.cell_fraction
    ]
    if cells:
        findings.append(
            _finding(
                "refined_cell_change",
                "Refined cell metrics changed on the same candidate basis",
                [f"{c['phase_id']} {c['metric']}: {100*c['relative_change']:+.4g}%" for c in cells],
                [
                    "Practical effect threshold, not a significance test; calibration and covariance review still apply"
                ],
                status="candidate",
            )
        )
    fractions = [
        v
        for v in model["fraction_changes"]
        if abs(v["percentage_point_difference"]) >= settings.fraction_points
    ]
    if fractions:
        findings.append(
            _finding(
                "modeled_fraction_change",
                "Modeled crystalline fractions changed on the same basis",
                [
                    f"{c['phase_id']}: {c['percentage_point_difference']:+.3g} percentage points"
                    for c in fractions
                ],
                [
                    "Fractions exclude omitted/amorphous material; no absolute conversion or composition is inferred"
                ],
                status="candidate",
            )
        )
    if features["unexplained_features"]:
        findings.append(
            _finding(
                "persistent_unexplained_features",
                "Localized measured features remain unexplained in both fits",
                [
                    f"{len(features['unexplained_features'])} positive residual features recur in common fitted coverage"
                ],
                [
                    "Incomplete candidate library, magnetic/diffuse scattering, background, profile and assembly effects remain alternatives"
                ],
                [p["region"] for p in features["unexplained_features"]],
                status="review_required",
            )
        )
    if model["candidate_ids_changed"]:
        findings.append(
            _finding(
                "candidate_assignment_changed",
                "RADAR-PD selected different candidate identities",
                [
                    "Added: " + (", ".join(model["added"]) or "none"),
                    "Removed: " + (", ".join(model["removed"]) or "none"),
                ],
                [
                    "Candidate changes alone do not establish phase appearance/disappearance",
                    "Fraction changes across different modeled-phase denominators are withheld",
                ],
                status="ambiguous",
            )
        )
    if (
        features["shape_effect"] > settings.shape_effect
        and features.get("reported_error_scaled_difference_rms", 0) > 2
        and not any(
            f["code"]
            in {
                "peak_movement",
                "new_features",
                "lost_features",
                "profile_shape_change",
                "relative_intensity_redistribution",
            }
            for f in findings
        )
    ):
        findings.append(
            _finding(
                "unresolved_pattern_change",
                "Measured pattern changed beyond the configured shape threshold",
                [f"Shape residual / pattern standard deviation: {features['shape_effect']:.4g}"],
                ["The current local-feature model does not identify a unique explanation"],
                status="ambiguous",
            )
        )
    if not findings:
        findings.append(
            _finding(
                "no_resolved_event",
                "No event resolved at the current detector settings",
                [f"{len(peaks)} matched features; shape difference {features['shape_effect']:.4g}"],
                ["This is not evidence of phase absence, equilibrium or experiment completion"],
                status="descriptive",
            )
        )
    return findings


def _condition_deltas(before, after, settings):
    left, right = _conditions(before, settings), _conditions(after, settings)
    result = []
    for quantity in sorted(set(left) | set(right)):
        a, b = left.get(quantity), right.get(quantity)
        item = {"quantity": quantity, "previous": a, "current": b, "difference": None}
        if a and b and a.get("unit") == b.get("unit"):
            try:
                delta = float(b["value"]) - float(a["value"])
                if math.isfinite(delta):
                    item.update(difference=delta, unit=b["unit"])
            except (KeyError, ValueError, TypeError):
                pass
        result.append(item)
    return result


def _resolved_difference(features, settings):
    return (
        features.get("status") == "compared"
        and features.get("shape_effect", 0) > settings.shape_effect
        and features.get("reported_error_scaled_difference_rms", 0) > 2
    )


def _matched_conditions(before, after, settings):
    a, b = _conditions(before, settings), _conditions(after, settings)
    if not a or set(a) != set(b):
        return False
    for key, current in b.items():
        old = a[key]
        tolerance = settings.condition_tolerances.get(key)
        if (
            not tolerance
            or current.get("status") != "recorded"
            or old.get("status") != "recorded"
            or current.get("unit") != old.get("unit")
        ):
            return False
        if current.get("channel") != old.get("channel"):
            return False
        if abs(current["value"] - old["value"]) > tolerance:
            return False
        if any(
            v.get("maximum", v["value"]) - v.get("minimum", v["value"]) > tolerance
            for v in [old, current]
        ):
            return False
    return True


def _measurement(scan):
    metadata = scan.get("metadata") or {}
    return json.dumps(
        [
            scan.get("instrument"),
            scan.get("ipts"),
            scan.get("run_number") or _run(scan),
            scan.get("bank"),
            metadata.get("start_time"),
            metadata.get("end_time"),
        ]
    )


def _revision(scan):
    return (
        (scan.get("observables") or {}).get("source_gpx_sha256")
        or (scan.get("pattern_observation") or {}).get("source_gpx_sha256")
        or (scan.get("pattern_observation") or {}).get("sha256")
        or _run(scan)
    )


def _earlier_pending(pending, scan):
    meta, current = pending.get("metadata") or {}, scan.get("metadata") or {}
    for key in ["instrument", "ipts", "bank"]:
        if pending.get(key) is not None and str(pending[key]) != str(scan.get(key)):
            return False
    if (
        meta.get("sample_id")
        and current.get("sample_id")
        and str(meta["sample_id"]) != str(current["sample_id"])
    ):
        return False
    if _time(pending) is not None and _time(scan) is not None:
        return _time(pending) <= _time(scan)
    if pending.get("run_number") and scan.get("run_number"):
        return int(pending["run_number"]) <= int(scan["run_number"])
    # Unknown ordering cannot certify that the reference is the first scan.
    return True


def _independent(before, after):
    try:
        end = parse_nexus_timestamp((before.get("metadata") or {})["end_time"])
        return (
            end.tzinfo is not None and _time(after) is not None and end.timestamp() <= _time(after)
        )
    except (KeyError, ValueError, TypeError):
        return False


class ScanInterpreter:
    def __init__(self, loader=None):
        self.loader = loader or self._default_loader
        self._patterns = {}
        self._pairs = {}

    @staticmethod
    def _default_loader(scan):
        if isinstance(scan.get("pattern"), dict):
            return scan["pattern"]
        if scan.get("pattern_path"):
            return load_pattern(
                scan["pattern_path"],
                expected_sha256=(scan.get("pattern_observation") or {}).get("sha256"),
            )
        raise ValueError(
            "Measured-curve observation unavailable; publish/extract it with the current Analyze image"
        )

    def pattern(self, scan):
        key = None
        if scan.get("pattern_path"):
            path = Path(scan["pattern_path"])
            stat = path.stat()
            key = (
                str(path.resolve()),
                stat.st_mtime_ns,
                stat.st_size,
                (scan.get("pattern_observation") or {}).get("sha256"),
            )
        elif scan.get("pattern"):
            key = digest(scan["pattern"])
        if key and key in self._patterns:
            return self._patterns[key]
        pattern = Pattern.parse(self.loader(scan))
        if len(self._patterns) >= 256:
            self._patterns.pop(next(iter(self._patterns)))
        self._patterns[key or pattern.key] = pattern
        return pattern

    def compare(self, before, after, settings):
        a, b = self.pattern(before), self.pattern(after)
        key = (
            a.key,
            b.key,
            digest(
                settings.model_dump(exclude={"references", "reference_hashes", "active_revisions"})
            ),
        )
        if key not in self._pairs:
            if len(self._pairs) >= 4096:
                self._pairs.pop(next(iter(self._pairs)))
            self._pairs[key] = compare_patterns(a, b, settings)
        return self._pairs[key]

    def similar(self, previous, scan, settings):
        """Rank recent compatible curves with a small signature, then compare fully."""
        current = self.pattern(scan)
        indices = np.linspace(0, len(current.x) - 1, min(512, len(current.x))).astype(int)
        scores = []
        for old in previous[-settings.history_search_limit :]:
            pattern = self.pattern(old)
            if not np.array_equal(pattern.x, current.x):
                continue
            valid = pattern.valid[indices] & current.valid[indices]
            if np.sum(valid) < 32:
                continue
            a, b = pattern.y[indices][valid], current.y[indices][valid]
            if np.std(a) > 0 and np.std(b) > 0:
                scores.append((float(np.corrcoef(a, b)[0, 1]), old))
        return max(scores, key=lambda pair: pair[0])[1] if scores else None

    def evaluate(self, scans, settings=None, *, pending=()):
        settings = settings or InterpretationSettings()
        if not isinstance(settings, InterpretationSettings):
            settings = InterpretationSettings.model_validate(settings)
        if len(scans) > settings.max_history:
            raise ValueError(
                "Series exceeds the configured history bound; select a narrower specimen/bank series"
            )
        report = {
            "$schema": REPORT_SCHEMA,
            "engine_version": ENGINE_VERSION,
            "settings": settings.model_dump(),
            "detector_calibration": "Engineering thresholds; scientific event sensitivity and false-alert rates are not certified",
            "records": [],
            "references": dict(settings.references),
            "reference_hashes": dict(settings.reference_hashes),
            "history_search_limit": settings.history_search_limit,
        }
        if not settings.enabled:
            return report
        ordered = sorted(
            scans,
            key=lambda s: (_time(s) is None, _time(s) or float(s.get("run_number") or 0), _run(s)),
        )
        revisions = {}
        for scan in ordered:
            measurement = _measurement(scan)
            revisions.setdefault(measurement, []).append(scan)
        history, prior_entries = {}, {}
        seen = set()
        for scan in ordered:
            run_id = _run(scan)
            metadata = scan.get("metadata") or {}
            measurement = _measurement(scan)
            revision = _revision(scan)
            identities = {_revision(row) for row in revisions[measurement]}
            entry = {
                "run_id": run_id,
                "measurement_key": measurement,
                "source_gpx_sha256": revision,
                "conditions": list(_conditions(scan, settings).values()),
                "condition_receipt": metadata.get("conditions") or {},
                "comparisons": [],
                "findings": [],
                "notices": [],
                "status": "unassessed",
                "scientific_warnings": list(scan.get("scientific_warnings") or []),
            }
            if scan.get("scientific_status") == "needs_review":
                entry["notices"].append(
                    "Input scientific review remains required; interpretations do not qualify the refinement"
                )
            if str(entry["condition_receipt"].get("window_scope", "")).startswith("parent_run"):
                entry["notices"].append(
                    "Condition trajectories describe the parent run; confirm the reduced product uses the same time window"
                )
            if len(identities) > 1 and settings.active_revisions.get(measurement) not in identities:
                entry["findings"] = [
                    _finding(
                        "revision_conflict",
                        "Select one result revision before comparing",
                        ["Different fitted projects belong to the same measurement"],
                        status="unassessed",
                    )
                ]
                report["records"].append(entry)
                continue
            if len(identities) > 1 and settings.active_revisions.get(measurement) != revision:
                continue
            if (measurement, revision) in seen:
                continue
            seen.add((measurement, revision))
            for channel in entry["conditions"]:
                tolerance = settings.condition_tolerances.get(channel["quantity"])
                if (
                    tolerance
                    and channel.get("status") == "recorded"
                    and channel.get("maximum", 0) - channel.get("minimum", 0) > tolerance
                ):
                    entry["findings"].append(
                        _finding(
                            "condition_sweep",
                            "Acquisition context spans a condition sweep",
                            [
                                f"{channel['quantity']}: {channel['minimum']:.5g} to {channel['maximum']:.5g} {channel.get('unit', '')}"
                            ],
                            [
                                "Verify the reduction time window before interpreting a fitted pattern at one condition",
                                "Averaged states can affect peak shapes and apparent coexistence",
                            ],
                            status="review_required",
                        )
                    )
            try:
                pattern = self.pattern(scan)
                group = _group(scan, pattern)
                entry.update(group_key=group, pattern_sha256=pattern.key, status="observed")
                previous = history.setdefault(group, [])
                candidates = []
                if previous:
                    candidates.append(("previous", previous[-1]))
                barrier = any(_earlier_pending(p, scan) for p in pending)
                entry["provisional"] = barrier
                if barrier:
                    entry["notices"].append(
                        "Earlier analysis is pending; reference/persistence conclusions remain provisional"
                    )
                reference_id = report["references"].get(group)
                if reference_id:
                    reference = next(
                        (p for p in [*previous, scan] if _run(p) == reference_id), None
                    )
                    if reference is None:
                        entry["notices"].append(
                            "Pinned reference is unavailable; choose a replacement explicitly"
                        )
                    elif report["reference_hashes"].get(group) not in {
                        None,
                        self.pattern(reference).key,
                    }:
                        entry["notices"].append(
                            "Pinned reference content changed; select a reference revision explicitly"
                        )
                    elif reference is not scan:
                        candidates.append(("reference", reference))
                        report["reference_hashes"][group] = self.pattern(reference).key
                elif not barrier:
                    report["references"][group] = _run(previous[0] if previous else scan)
                    report["reference_hashes"][group] = self.pattern(
                        previous[0] if previous else scan
                    ).key
                    if previous:
                        candidates.append(("reference", previous[0]))
                # Matched-condition history is useful across branches, but one
                # unknown/unmatched condition cannot certify the same state.
                matched = [old for old in previous[:-1] if _matched_conditions(old, scan, settings)]
                if matched:
                    candidates.append(("matched_conditions", matched[-1]))
                similar = self.similar(previous[:-1], scan, settings)
                if similar is not None:
                    candidates.append(("similar_history", similar))
                used = {}
                for role, old in candidates:
                    old_id = _run(old)
                    if old_id in used:
                        used[old_id]["roles"].append(role)
                        continue
                    features = self.compare(old, scan, settings)
                    change = _model_changes(old, scan, settings)
                    if not features.get("fit_domain_compatible"):
                        change.update(
                            cell_changes=[],
                            fraction_changes=[],
                            fraction_basis_compatible=False,
                            rwp_difference=None,
                            notice="Fit intervals, masks or weight factors differ; quantitative model changes are withheld",
                        )
                    comparison = {
                        "reference_run_id": old_id,
                        "reference_pattern_sha256": self.pattern(old).key,
                        "roles": [role],
                        "features": features,
                        "model_changes": change,
                        "condition_changes": _condition_deltas(old, scan, settings),
                        "independent_windows": _independent(old, scan),
                    }
                    used[old_id] = comparison
                    entry["comparisons"].append(comparison)
                for old_id, comparison in used.items():
                    roles, features = comparison["roles"], comparison["features"]
                    old = next(p for p in previous if _run(p) == old_id)
                    if "previous" in roles:
                        findings = explain_comparison(old, scan, features, settings)
                        for finding in findings:
                            finding["comparison_reference_run_id"] = old_id
                        entry["findings"] += findings
                    if (
                        "reference" in roles
                        and "previous" not in roles
                        and _resolved_difference(features, settings)
                    ):
                        entry["findings"].append(
                            _finding(
                                "reference_difference",
                                "The pattern differs from the pinned reference",
                                [
                                    f"Shape difference {features['shape_effect']:.4g} versus {old_id}"
                                ],
                                [
                                    "Small adjacent changes can accumulate; inspect the reference overlay"
                                ],
                                status="candidate",
                            )
                        )
                    if "matched_conditions" in roles and _resolved_difference(features, settings):
                        entry["findings"].append(
                            _finding(
                                "history_difference",
                                "A previous matching recorded condition has a different pattern",
                                [f"Comparison with {old_id}"],
                                [
                                    "Check branch, cycle, aging and unrecorded conditions before assigning a mechanism"
                                ],
                                status="ambiguous",
                            )
                        )
                last_entry = prior_entries.get(group)
                if last_entry and previous and not barrier and _independent(previous[-1], scan):
                    current_pair = next(
                        (c for c in entry["comparisons"] if "previous" in c["roles"]), {}
                    )
                    old_pair = next(
                        (c for c in last_entry["comparisons"] if "previous" in c["roles"]), {}
                    )
                    if (
                        len(previous) >= 2
                        and _independent(previous[-2], previous[-1])
                        and _matched_conditions(previous[-2], previous[-1], settings)
                        and _matched_conditions(previous[-1], scan, settings)
                        and all(
                            _resolved_difference(c.get("features", {}), settings)
                            for c in [current_pair, old_pair]
                        )
                    ):
                        entry["findings"].append(
                            _finding(
                                "continuing_pattern_evolution",
                                "Pattern continues changing at matching recorded conditions",
                                [
                                    f"Two successive changes across {_run(previous[-2])}, {_run(previous[-1])}, {run_id}"
                                ],
                                [
                                    "Only recorded channels are matched; equilibration and unrecorded conditions remain alternatives"
                                ],
                                status="candidate",
                            )
                        )
                    reference_pair = next(
                        (c for c in entry["comparisons"] if "reference" in c["roles"]), {}
                    )
                    old_reference = next(
                        (c for c in last_entry["comparisons"] if "reference" in c["roles"]), {}
                    )
                    if reference_pair.get("reference_pattern_sha256") == old_reference.get(
                        "reference_pattern_sha256"
                    ):
                        persistent = [
                            p
                            for p in reference_pair.get("features", {}).get("new_features", [])
                            if any(
                                abs(p["x"] - q["x"]) < max(p["width"], q["width"]) / 2
                                for q in old_reference.get("features", {}).get("new_features", [])
                            )
                        ]
                        if persistent:
                            entry["findings"].append(
                                _finding(
                                    "features_persist",
                                    "Additional features persist in two nonoverlapping acquisitions",
                                    [
                                        f"{len(persistent)} features relative to the pinned reference"
                                    ],
                                    [
                                        "Persistence strengthens the observation; it does not identify a chemical phase"
                                    ],
                                    [p["region"] for p in persistent],
                                    status="candidate",
                                )
                            )
                if not previous:
                    entry["notices"].append(
                        "First compatible measured pattern; waiting for another acquisition"
                    )
                previous.append(scan)
                prior_entries[group] = entry
            except (ValueError, TypeError, KeyError, OSError) as exc:
                entry["notices"].append(str(exc))
                entry["findings"].append(
                    _finding(
                        "comparison_unresolved",
                        "Curve comparison is unavailable",
                        [str(exc)],
                        status="unassessed",
                    )
                )
            report["records"].append(entry)
        report["settings"]["references"] = report["references"]
        report["settings"]["reference_hashes"] = report["reference_hashes"]
        report["evidence_sha256"] = digest(report)
        return report


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Replay saved RADAR-PD curve/condition observations without a browser"
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    document = json.loads(args.manifest.read_text())
    settings = InterpretationSettings.model_validate(
        json.loads(args.settings.read_text()) if args.settings else {}
    )
    root = args.manifest.resolve().parent
    for scan in document["scans"]:
        if scan.get("pattern_path"):
            path = (root / scan["pattern_path"]).resolve()
            path.relative_to(root)
            scan["pattern_path"] = str(path)
    result = ScanInterpreter().evaluate(
        document["scans"], settings, pending=document.get("pending", [])
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
