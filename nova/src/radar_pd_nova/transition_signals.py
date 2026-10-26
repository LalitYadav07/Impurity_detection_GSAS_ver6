"""Measured indicators and an explicit user boundary rule.

Profile widths are conditional local measurements, not instrumental-resolution
deconvolution or probabilities. No future acquisition or fitted phase is used.
The original pair detector remains unchanged; its measurements are supplemented
with local native-grid profile fits instead of its 45% width-change gate.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math

import numpy as np
from scipy.optimize import least_squares

SIGNALS = {
    "appearance": "New or lost signal",
    "splitting": "Splitting / doublet evolution",
    "width": "Peak broadening / narrowing",
    "shape": "Peak shape / asymmetry",
    "intensity": "Relative intensity redistribution",
    "rwp": "Final Rwp change · refinement",
}
LOCAL_SIGNALS = tuple(k for k in SIGNALS if k!="rwp")
SIGNAL_VERSION = "native-profile-indicators/3"


@dataclass(frozen=True)
class BoundaryPolicy:
    selected: tuple[str, ...] = ("appearance", "splitting", "width")
    minimum_signals: int = 1
    comparison: str = "reference"
    width_fraction: float = 0.03
    error_ratio: float = 3.0
    intensity_fraction: float = 0.15
    shape_fraction: float = 0.08
    minimum_features: int = 2
    fixed_broadening: bool = True
    rwp_delta_pp: float = 1.0
    rwp_relative: float = .10
    rwp_direction: str = "either"
    split_separation_fwhm: float = .10

    def __post_init__(self):
        if not isinstance(self.selected, (list, tuple)) or any(s not in SIGNALS for s in self.selected):
            raise ValueError("Choose supported boundary indicators")
        object.__setattr__(self, "selected", tuple(s for s in SIGNALS if s in self.selected))
        if isinstance(self.minimum_signals, bool) or not isinstance(self.minimum_signals, int) or not 1 <= self.minimum_signals <= len(SIGNALS):
            raise ValueError("The boundary rule needs between one and six indicators")
        if self.selected and self.minimum_signals > len(self.selected):
            raise ValueError("The required count exceeds the selected indicators")
        if self.comparison not in {"reference", "previous"}:
            raise ValueError("Choose the current group reference or preceding scan")
        for field, low, high in (("width_fraction", .005, .5), ("error_ratio", 2, 10),
                                 ("intensity_fraction", .02, 2), ("shape_fraction", .01, .5),
                                 ("rwp_delta_pp",.01,100), ("rwp_relative",.01,10),
                                 ("split_separation_fwhm",.02,2)):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"Unsupported {field}")
        if isinstance(self.minimum_features, bool) or not isinstance(self.minimum_features, int) or not 1 <= self.minimum_features <= 10:
            raise ValueError("Choose one to ten supporting peak regions")
        if not isinstance(self.fixed_broadening, bool):
            raise ValueError("The fixed-broadening assumption must be explicit")
        if self.rwp_direction not in {"either","increase"}:
            raise ValueError("Choose either direction or increasing Rwp")

    @property
    def fingerprint(self):
        return sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    @classmethod
    def restore(cls, value):
        if not isinstance(value, dict):
            return cls()
        try:
            return cls(**{k: v for k, v in value.items() if k in cls.__dataclass_fields__})
        except (TypeError, ValueError):
            return cls()


def profile_width(prepared, feature):
    """Fit one isolated peak once, using native samples and an affine background.

    A pseudo-Voigt is a local width estimator, independent of phase/hkl. A
    conditional covariance inflated by residual scatter and a 0.5% width floor
    prevent tiny formal fit errors from dominating the width comparison.
    Crowded, masked, undersampled and unsuccessful estimates are withheld.
    """
    cache = getattr(prepared, "_profile_width_measurements", None)
    if cache is None:
        cache = prepared._profile_width_measurements = {}
    if feature.position in cache:
        return cache[feature.position]
    result = None
    cache[feature.position] = None
    if feature.crowded or feature.width_bins < 4 or feature.snr < 12:
        return None
    p = prepared.observation
    use = (p.x >= feature.left) & (p.x <= feature.right)
    if use.sum() < 12 or not np.all(p.valid[use]):
        return None
    x = (p.x[use] - feature.position) / feature.width
    height = max(feature.height, 1e-30)
    base = float(np.interp(feature.position, p.x, prepared.baseline))
    y = (p.intensity[use] - base) / height
    sigma = np.maximum(prepared.noise[use] / height, 1e-6)

    def residual(v):
        amplitude, center, width, mixing, offset, slope = v
        z = (x - center) / width
        model = amplitude * ((1 - mixing) * np.exp(-4 * math.log(2) * z * z)
                             + mixing / (1 + 4 * z * z)) + offset + slope * x
        return (model - y) / sigma

    try:
        fit = least_squares(residual, [1, 0, 1, .2, 0, 0],
            bounds=([0, -.75, .35, 0, -2, -1], [5, .75, 3, 1, 2, 1]),
            max_nfev=65, ftol=1e-6, xtol=1e-6, gtol=1e-6)
        if not fit.success or not .36 < fit.x[2] < 2.99 or abs(fit.x[1]) > .74:
            return None
        scatter = max(1.0, float(np.dot(fit.fun, fit.fun)) / max(1, len(x) - len(fit.x)))
        covariance = np.linalg.pinv(fit.jac.T @ fit.jac) * scatter
        width = float(fit.x[2] * feature.width)
        error = max(math.sqrt(max(0, float(covariance[2, 2]))) * feature.width, width * .005)
        if not math.isfinite(error) or error / width > .25:
            return None
        result = {"width": width, "error": error, "position": feature.position,
                  "native_points": int(len(x)), "residual_scatter": scatter,
                  "method": "local pseudo-Voigt FWHM; conditional native-grid covariance"}
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        pass
    cache[feature.position] = result
    return result


def measure_signals(before, after, report):
    """Cache measurements independently of which indicators a user selects."""
    coverage = report.get("metrics", {}).get("common_native_coverage", 0)
    quality = (coverage >= report.get("settings",{}).get("min_coverage",.75)
               and len(before.features) >= 3 and len(after.features) >= 3
               and before.observation.scope == after.observation.scope
               and not before.quality["feature_cap_hit"] and not after.quality["feature_cap_hit"])
    measured = {"quality_ok": bool(quality), "width": [], "shape": [], "intensity": [],
                "appearance": [], "splitting": [], "width_method": "relative FWHM / peak position"}
    if not quality:
        return measured
    fa = {f.position: f for f in before.features}
    fb = {f.position: f for f in after.features}
    for match in report.get("matches", []):
        if match["ambiguous"] or match["crowded"]:
            continue
        p, q = fa[match["before_position"]], fb[match["after_position"]]
        interval = [min(p.left, q.left), max(p.right, q.right)]
        common = {"interval": interval, "before_position": p.position, "after_position": q.position}
        a, b = profile_width(before, p), profile_width(after, q)
        if a is not None and b is not None:
            delta = math.log((b["width"] / b["position"]) / (a["width"] / a["position"]))
            error = math.hypot(a["error"] / a["width"], b["error"] / b["width"])
            measured["width"].append({**common, "fraction": math.expm1(delta),
                "error_ratio": abs(delta) / error, "log_error": error,
                "before_width": a["width"], "after_width": b["width"]})
        shape = match.get("shape_distance")
        if shape is not None and min(p.snr, q.snr) >= 12 and min(p.width_bins, q.width_bins) >= 3:
            measured["shape"].append({**common, "fraction": float(shape)})
        delta, error_ratio = match.get("log_relative_intensity"), match.get("intensity_error_ratio")
        if delta is not None and error_ratio is not None:
            measured["intensity"].append({**common, "fraction": math.expm1(delta), "error_ratio": error_ratio})
    for region in report.get("regions", []):
        kinds = region.get("kinds", [])
        if "resolved_component_count_change" in kinds:
            measured["splitting"].append(region)
        # A tentative daughter/redistribution hint is not an established new
        # or lost signal. Keep it in the raw diagnostics without counting it.
        if "additional_or_lost_signal" in kinds:
            measured["appearance"].append(region)
    from .splitting_profiles import component_changes
    measured["splitting"].extend(component_changes(before,after,report))
    return measured


def apply_policy(measurements, policy):
    """Count selected indicators, without treating correlated votes as probabilities."""
    indicators = {}
    quality = measurements.get("quality_ok", False)
    for key, title in SIGNALS.items():
        if key=="rwp":
            evidence=measurements.get("rwp") or {}
            available=bool(evidence.get("available"))
            detected=bool(available and evidence.get("detected"))
            indicators[key]={"title":title,"available":available,"detected":detected,
                "support":int(detected),"measured":int(available),"required_features":1,
                "regions":[],"localizable":False,"model_dependent":True,
                "selected":key in policy.selected,"details":evidence,"reason":evidence.get("reason", "Select Rwp to compare completed final refinements")}
            continue
        values = measurements.get(key, [])
        required = policy.minimum_features if key in {"width", "shape", "intensity"} else 1
        available = quality and (len(values) >= required if key in {"width", "shape", "intensity"} else True)
        if key == "width":
            available &= policy.fixed_broadening
            hits = [v for v in values if abs(v["fraction"]) >= policy.width_fraction and v["error_ratio"] >= policy.error_ratio]
        elif key == "intensity":
            hits = [v for v in values if abs(v["fraction"]) >= policy.intensity_fraction and v["error_ratio"] >= policy.error_ratio]
        elif key == "shape":
            hits = [v for v in values if v["fraction"] >= policy.shape_fraction]
        elif key == "splitting":
            hits=[v for v in values if v.get("kind")!="local_component_separation_change"
                or v["separation_change_fwhm"]>=policy.split_separation_fwhm and v["error_ratio"]>=policy.error_ratio]
        else:
            hits = values
        indicators[key] = {"title": title, "available": bool(available),
            "detected": bool(available and len(hits) >= required), "support": len(hits),
            "measured": len(values), "required_features": required, "regions": hits,
            "maximum_effect": max((abs(v["fraction"]) for v in hits if "fraction" in v), default=None),
            "maximum_measured_effect": max((abs(v["fraction"]) for v in values if "fraction" in v), default=None),
            "selected": key in policy.selected,"localizable":True,"model_dependent":False}
        if key=="splitting":
            indicators[key]["features"]=[{**v,"passes_rule":v in hits} for v in values
                if v.get("kind")=="local_component_separation_change"]
    fired = [key for key in policy.selected if indicators[key]["detected"]]
    unavailable = sum(not indicators[key]["available"] for key in policy.selected)
    if not policy.selected:
        status, reason = "uncertain", "no_boundary_indicators_selected"
    elif len(fired) >= policy.minimum_signals:
        status, reason = "transition_evidence", "selected_indicator_rule_met"
    elif not quality or len(fired) + unavailable >= policy.minimum_signals:
        status, reason = "uncertain", "insufficient_available_selected_indicators"
    else:
        status, reason = "no_resolved_change", "selected_indicator_rule_not_met"
    return {"status": status, "reason": reason, "indicators": indicators,
            "signal_engine": SIGNAL_VERSION,
            "triggered": fired, "votes": len(fired), "required": policy.minimum_signals,
            "selected": list(policy.selected), "policy_sha256": policy.fingerprint,
            "fixed_broadening_assumed": policy.fixed_broadening,
            "regions": [r for key in fired for r in indicators[key]["regions"]]}
