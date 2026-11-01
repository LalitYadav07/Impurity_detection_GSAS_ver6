"""Phase-independent evidence of changes between observed powder patterns.

The input contract deliberately has no CIF, hkl, phase, cell, Rwp, ycalc, or
refinement fields. Each resolved feature can move independently, in either
direction. A partial bipartite assignment permits unmatched features on BOTH
sides and does not impose an order-preserving warp on the pattern.

Outputs describe visible changes, not thermodynamic truth or magnetic k.
Counting-error ratios are diagnostics, not p-values: reduced bins can be
correlated and systematic reduction errors are not fully known. Callers must
establish specimen/bank/reduction compatibility before making a comparison.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from itertools import pairwise
from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter, percentile_filter
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

ENGINE_VERSION = "observed-pattern-transition/2"
MAX_POINTS = 100_000


@dataclass(frozen=True)
class PatternScope:
    """Explicit upstream provenance, never inferred from a similar curve.

    ``specimen`` must distinguish actual specimens, including differing samples
    with the same name. ``geometry`` covers calibration/profile/setup revision;
    ``reduction`` covers vanadium/background, processing family and masks.
    A shared, reviewed manual scope is allowed, but must be declared by caller.
    """

    specimen: str
    instrument: str
    bank: str
    coordinate: Literal["tof_us", "two_theta_deg", "d_angstrom", "q_inv_angstrom"]
    geometry: str
    reduction: str
    normalization: str
    intensity_semantics: str

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in asdict(self).values()):
            raise ValueError("Every compatibility field requires an explicit value")
        if self.coordinate not in {"tof_us", "two_theta_deg", "d_angstrom", "q_inv_angstrom"}:
            raise ValueError("Unsupported or unspecified coordinate units")


@dataclass(frozen=True)
class TransitionSettings:
    """Sensitivity settings. A calibration receipt must accompany defaults.

    Motion is a search bound for EACH feature, not a common lattice scale.
    Baseline/smoothing/noise windows are in native bins; the caller must retain
    native resolution and may not silently oversample a low-resolution curve.
    """

    baseline_bins: int = 151
    noise_bins: int = 61
    smooth_bins: float = 0.8
    peak_snr: float = 6.0
    min_prominence_fraction: float = 0.004
    min_width_bins: float = 1.5
    max_feature_relative_width: float = 0.06
    motion_limit: float = 0.02
    unmatched_cost: float = 0.8
    assignment_margin: float = 0.12
    change_snr: float = 6.0
    area_systematic_fraction: float = 0.04
    intensity_log_effect: float = math.log(1.6)
    width_log_effect: float = math.log(1.45)
    shape_effect: float = 0.20
    min_changed_mass: float = 0.015
    single_region_mass: float = 0.025
    min_feature_height_fraction: float = 0.008
    min_independent_regions: int = 2
    min_coverage: float = 0.75
    min_segment_bins: int = 24
    max_features: int = 500
    smooth_intensity_nuisance: bool = True

    def __post_init__(self):
        for k, v in asdict(self).items():
            if (
                isinstance(v, (float, int))
                and not isinstance(v, bool)
                and (not math.isfinite(v) or v <= 0)
            ):
                raise ValueError(f"{k} must be finite and positive")
        for k in [
            "baseline_bins",
            "noise_bins",
            "min_independent_regions",
            "min_segment_bins",
            "max_features",
        ]:
            if not isinstance(getattr(self, k), int):
                raise TypeError(f"{k} must be an integer")
        if (
            self.baseline_bins < 15
            or self.baseline_bins % 2 != 1
            or self.noise_bins < 7
            or self.noise_bins % 2 != 1
        ):
            raise ValueError("Baseline/noise windows must be odd and sufficiently wide")
        if not 0 < self.motion_limit <= 0.15 or not 0 < self.min_coverage <= 1:
            raise ValueError("Unsupported motion or coverage bounds")
        if self.max_features > 600 or self.min_segment_bins < 16:
            raise ValueError("Feature/segment bounds exceed the supported limits")

    @property
    def fingerprint(self):
        return sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    @property
    def preparation_fingerprint(self):
        keys = [
            "baseline_bins",
            "noise_bins",
            "smooth_bins",
            "peak_snr",
            "min_prominence_fraction",
            "min_width_bins",
            "max_feature_relative_width",
            "min_segment_bins",
            "max_features",
        ]
        return sha256(
            json.dumps({k: getattr(self, k) for k in keys}, sort_keys=True).encode()
        ).hexdigest()


@dataclass(frozen=True)
class ObservedPattern:
    scope: PatternScope
    x: np.ndarray
    intensity: np.ndarray
    sigma: np.ndarray | None = None
    valid: np.ndarray | None = None
    observation_id: str = ""

    def __post_init__(self):
        if not isinstance(self.scope, PatternScope):
            raise TypeError("An explicit PatternScope is required")
        x = np.asarray(self.x, dtype=float).copy()
        y = np.asarray(self.intensity, dtype=float).copy()
        if x.ndim != 1 or not 32 <= len(x) <= MAX_POINTS or y.shape != x.shape:
            raise ValueError(
                "Observed arrays need matching, bounded one-dimensional shapes"
            )
        if not np.all(np.isfinite(x)) or np.any(x <= 0) or np.any(np.diff(x) <= 0):
            raise ValueError(
                "Coordinates must be finite, positive and strictly increasing"
            )
        mask = (
            np.ones(len(x), dtype=bool)
            if self.valid is None
            else np.asarray(self.valid)
        )
        if mask.dtype != bool or mask.shape != x.shape:
            raise ValueError("The validity mask must contain explicit booleans")
        mask = mask.copy()
        if not np.all(np.isfinite(y[mask])):
            raise ValueError("Unmasked measured intensities must be finite")
        sigma = (
            np.full(len(x), np.nan)
            if self.sigma is None
            else np.asarray(self.sigma, dtype=float).copy()
        )
        if sigma.shape != x.shape:
            raise ValueError(
                "Errors and measured coordinates must have matching shapes"
            )
        sigma[~np.isfinite(sigma) | (sigma <= 0)] = np.nan
        for name, value in [
            ("x", x),
            ("intensity", y),
            ("sigma", sigma),
            ("valid", mask),
        ]:
            value.flags.writeable = False
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class Feature:
    position: float
    width: float
    width_bins: float
    area: float
    area_error: float
    height: float
    snr: float
    left: float
    right: float
    segment: int
    bin_width: float
    crowded: bool


@dataclass
class PreparedPattern:
    observation: ObservedPattern
    settings_fingerprint: str
    baseline: np.ndarray
    excess: np.ndarray
    noise: np.ndarray
    segments: list[tuple[int, int]]
    features: list[Feature]
    quality: dict


def _segments(x, mask, minimum):
    # Do not bridge excluded regions or a large gap in a coordinate grid.
    delta = np.diff(np.log(x))
    local = median_filter(delta, size=min(31, len(delta) // 2 * 2 + 1), mode="nearest")
    gaps = delta > 5 * np.maximum(local, np.finfo(float).eps)
    starts = mask & ~np.r_[False, mask[:-1] & ~gaps]
    ends = mask & ~np.r_[mask[1:] & ~gaps, False]
    return [
        (int(a), int(b) + 1)
        for a, b in zip(np.where(starts)[0], np.where(ends)[0])
        if b + 1 - a >= minimum
    ]


def prepare_pattern(
    pattern: ObservedPattern, settings: TransitionSettings | None = None
):
    """Measure native-grid features once for reuse across comparison anchors."""
    settings = settings or TransitionSettings()
    x, y = pattern.x, pattern.intensity
    baseline = np.full(len(x), np.nan)
    excess = np.full(len(x), np.nan)
    noise = np.full(len(x), np.nan)
    segments = _segments(x, pattern.valid, settings.min_segment_bins)
    candidates, broad = [], []
    for segment, (a, b) in enumerate(segments):
        yy, xx = y[a:b], x[a:b]
        smoothed = gaussian_filter1d(yy, settings.smooth_bins, mode="nearest")
        window = min(settings.baseline_bins, (len(yy) // 2) * 2 - 1)
        bg = gaussian_filter1d(
            percentile_filter(smoothed, 20, size=window, mode="nearest"),
            max(window / 18, 1),
            mode="nearest",
        )
        residual = smoothed - bg
        # A local robust high-frequency estimate protects against unrealistically
        # tiny conditional errors. It is not an independent Poisson estimate.
        differences = np.r_[
            yy[0] - 2 * yy[1] + yy[2], np.diff(yy, 2), yy[-3] - 2 * yy[-2] + yy[-1]
        ]
        empirical = median_filter(
            np.abs(differences), size=settings.noise_bins, mode="nearest"
        ) / (0.67448975 * math.sqrt(6))
        empirical_floor = max(
            float(np.median(np.abs(differences))) / (0.67448975 * math.sqrt(6)) * 0.2,
            float(np.max(np.abs(residual))) * 1e-8,
            np.finfo(float).tiny,
        )
        empirical = np.maximum(empirical, empirical_floor)
        reported = pattern.sigma[a:b]
        sigma = np.maximum(empirical, np.where(np.isfinite(reported), reported, 0))
        baseline[a:b], excess[a:b], noise[a:b] = bg, residual, sigma
        threshold = np.maximum(
            settings.peak_snr * sigma,
            settings.min_prominence_fraction * max(float(np.max(residual)), 1e-30),
        )
        peaks, props = find_peaks(
            residual,
            prominence=(threshold, None),
            width=(settings.min_width_bins, None),
            distance=2,
        )
        for k, i in enumerate(peaks):
            width = float(
                np.interp(props["right_ips"][k], np.arange(len(xx)), xx)
                - np.interp(props["left_ips"][k], np.arange(len(xx)), xx)
            )
            # Features too close to a mask/frame boundary have unknown support.
            if (
                i < 3
                or i >= len(yy) - 3
                or xx[i] - 1.25 * width <= xx[0]
                or xx[i] + 1.25 * width >= xx[-1]
            ):
                continue
            curvature = residual[i - 1] - 2 * residual[i] + residual[i + 1]
            sub = (
                float(
                    np.clip(
                        0.5 * (residual[i - 1] - residual[i + 1]) / curvature, -0.5, 0.5
                    )
                )
                if curvature
                else 0
            )
            position = float(np.interp(i + sub, np.arange(len(xx)), xx))
            feature = {
                "position": position,
                "width": width,
                "width_bins": float(props["widths"][k]),
                "height": float(props["prominences"][k]),
                "snr": float(props["prominences"][k] / sigma[i]),
                "segment": segment,
                "bin_width": float((xx[i + 1] - xx[i - 1]) / 2),
                "index": a + int(i),
            }
            if width / position > settings.max_feature_relative_width:
                broad.append(
                    {k: feature[k] for k in ["position", "width", "height", "snr"]}
                )
            else:
                candidates.append(feature)
    truncated = len(candidates) > settings.max_features
    if truncated:
        candidates = sorted(candidates, key=lambda f: f["height"], reverse=True)[
            : settings.max_features
        ]
    candidates.sort(key=lambda f: f["position"])
    features = []
    for k, f in enumerate(candidates):
        a, b = segments[f["segment"]]
        pos, width = f["position"], f["width"]
        left, right = max(pos - 2.25 * width, x[a]), min(pos + 2.25 * width, x[b - 1])
        crowded = False
        if k and candidates[k - 1]["segment"] == f["segment"]:
            neighbor = candidates[k - 1]
            left = max(left, (pos + neighbor["position"]) / 2)
            crowded |= pos - neighbor["position"] < 1.5 * max(width, neighbor["width"])
        if k + 1 < len(candidates) and candidates[k + 1]["segment"] == f["segment"]:
            neighbor = candidates[k + 1]
            right = min(right, (pos + neighbor["position"]) / 2)
            crowded |= neighbor["position"] - pos < 1.5 * max(width, neighbor["width"])
        use = (x >= left) & (x <= right) & np.isfinite(excess)
        indices = np.where(use)[0]
        if len(indices) < 3:
            continue
        # Native coordinate widths are essential for SLOG/variable binning.
        dx = np.gradient(x[indices])
        area = float(np.dot(np.maximum(excess[indices], 0), dx))
        area_error = float(2 * np.sqrt(np.sum((noise[indices] * dx) ** 2)))
        if area <= 0:
            continue
        features.append(
            Feature(
                **{
                    name: f[name]
                    for name in [
                        "position",
                        "width",
                        "width_bins",
                        "height",
                        "snr",
                        "segment",
                        "bin_width",
                    ]
                },
                area=area,
                area_error=area_error,
                left=float(left),
                right=float(right),
                crowded=bool(crowded),
            )
        )
    usable = sum(b - a for a, b in segments)
    reported_fraction = (
        float(np.mean(np.isfinite(pattern.sigma[pattern.valid])))
        if np.any(pattern.valid)
        else 0
    )
    quality = {
        "points": len(x),
        "valid_fraction": float(np.mean(pattern.valid)),
        "usable_segment_fraction": usable / len(x),
        "features": len(features),
        "feature_cap_hit": truncated,
        "reported_error_fraction": reported_fraction,
        "broad_components_for_separate_review": broad,
        "empirical_error_floor": True,
        "errors_are_independent_poisson": False,
    }
    return PreparedPattern(
        pattern,
        settings.preparation_fingerprint,
        baseline,
        excess,
        noise,
        segments,
        features,
        quality,
    )


def partial_assignment(cost, unmatched_cost):
    """Optimal partial matching with independent unmatched choices on both sides.

    An ordinary square or rectangular assignment forces inappropriate matches.
    Dummy rows AND columns are required here, even when feature counts agree.
    """
    cost = np.asarray(cost, dtype=float)
    if (
        cost.ndim != 2
        or not math.isfinite(unmatched_cost)
        or unmatched_cost <= 0
        or np.any(np.isnan(cost))
    ):
        raise ValueError("A finite penalty and a non-NaN cost matrix are required")
    n, m = cost.shape
    if n > 600 or m > 600:
        raise ValueError("Assignment exceeds bounded feature count")
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    augmented = np.full((n + m, n + m), np.inf)
    augmented[:n, :m] = cost
    augmented[np.arange(n), m + np.arange(n)] = unmatched_cost
    augmented[n + np.arange(m), np.arange(m)] = unmatched_cost
    augmented[n:, m:] = 0
    rows, cols = linear_sum_assignment(augmented)
    matches = [
        (int(i), int(j))
        for i, j in zip(rows, cols)
        if i < n and j < m and cost[i, j] < 2 * unmatched_cost
    ]
    used_a, used_b = {i for i, j in matches}, {j for i, j in matches}
    return (
        matches,
        [i for i in range(n) if i not in used_a],
        [j for j in range(m) if j not in used_b],
    )


def _complete_feature_assignment(result, costs, settings):
    """Complete feasible local correspondences before declaring a new signal.

    A cheap chain of neighboring matches can leave two endpoints unmatched
    even when all features have an admissible one-to-one explanation. Within
    small affected components, first maximize explained features, then retain
    the original cost preference. No previously forbidden edge is admitted,
    and an unchanged cardinality keeps the original assignment. Individual
    identities can remain ambiguous; their existing margin checks still apply.
    """
    matches, unmatched_a, unmatched_b = result
    if not unmatched_a or not unmatched_b:
        return result
    n, m = costs.shape
    eligible = np.isfinite(costs) & (costs < 2 * settings.unmatched_cost)
    starts = [*unmatched_a, *(n + j for j in unmatched_b)]
    visited, completed = set(), list(matches)
    for start in starts:
        if start in visited:
            continue
        pending, component = [start], set()
        while pending:
            node = pending.pop()
            if node in component:
                continue
            component.add(node)
            neighbors = (n + int(j) for j in np.flatnonzero(eligible[node])) if node < n else (
                int(i) for i in np.flatnonzero(eligible[:, node - n]))
            pending.extend(neighbors)
        visited.update(component)
        left = sorted(i for i in component if i < n)
        right = sorted(i - n for i in component if i >= n)
        if not left or not right or len(component) > 64:
            continue
        previous = [pair for pair in completed if pair[0] in left and pair[1] in right]
        local = costs[np.ix_(left, right)].copy()
        local[~eligible[np.ix_(left, right)]] = np.inf
        # One unmatched endpoint pair must cost more than every feasible local
        # edge combined. This changes priority, not the admissibility threshold.
        penalty = (min(len(left), len(right)) + 1) * 2 * settings.unmatched_cost
        proposed, _, _ = partial_assignment(local, penalty)
        if len(proposed) <= len(previous):
            continue
        completed = [pair for pair in completed if pair not in previous]
        completed.extend((left[i], right[j]) for i, j in proposed)
    used_a, used_b = {i for i, j in completed}, {j for i, j in completed}
    return (sorted(completed), [i for i in range(n) if i not in used_a],
            [j for j in range(m) if j not in used_b])


def _match_features(a, b, settings):
    fa, fb = a.features, b.features
    if not fa or not fb:
        return partial_assignment(
            np.empty((len(fa), len(fb))), settings.unmatched_cost
        ), np.empty((len(fa), len(fb)))
    xa, xb = np.array([f.position for f in fa]), np.array([f.position for f in fb])
    aa, ab = np.array([f.area for f in fa]), np.array([f.area for f in fb])
    wa, wb = np.array([f.width for f in fa]), np.array([f.width for f in fb])
    # A scale-invariant preliminary rank/area prior discourages mapping a tiny
    # new shoulder to a nearby dominant reflection. It does not require equal
    # intensities or a shared isotropic strain.
    gain = float(np.quantile(ab, 0.65) / np.quantile(aa, 0.65))
    displacement = np.abs(np.log(xb[None, :] / xa[:, None])) / math.log1p(
        settings.motion_limit
    )
    area_delta = np.abs(np.log(ab[None, :] / (gain * aa[:, None])))
    width_delta = np.abs(np.log(wb[None, :] / wa[:, None]))
    cost = (
        0.8 * displacement**2
        + 0.10 * np.minimum(area_delta, 3) ** 2
        + 0.12 * np.minimum(width_delta, 3) ** 2
    )
    cost[displacement > 1] = np.inf
    result = partial_assignment(cost, settings.unmatched_cost)
    return _complete_feature_assignment(result, cost, settings), cost


def _gain_model(fa, fb, matches, coordinate, settings, measurements):
    good = [
        (i, j, measurements[k])
        for k, (i, j) in enumerate(matches)
        if measurements[k] is not None
    ]
    if not good:
        return 0.0, 0.0, 0.0, 1.0
    pos = np.array([math.sqrt(fa[i].position * fb[j].position) for i, j, m in good])
    ratios = np.array([math.log(m["gain"]) for i, j, m in good])
    q2 = pos**2 if coordinate == "q_inv_angstrom" else 1 / pos**2
    center = float(np.median(q2))
    span = max(float(np.quantile(q2, 0.9) - np.quantile(q2, 0.1)), 1e-30)
    q = np.clip((q2 - center) / span, -2, 2)
    intercept, slope = float(np.median(ratios)), 0.0
    if settings.smooth_intensity_nuisance and len(good) >= 8:
        design = np.c_[np.ones(len(q)), q]
        for _ in range(8):
            residual = ratios - intercept - slope * q
            spread = max(
                float(np.median(np.abs(residual - np.median(residual)))) * 1.4826, 0.06
            )
            weight = np.minimum(1, (1.5 * spread) / np.maximum(np.abs(residual), 1e-30))
            fit = np.linalg.lstsq(
                design * np.sqrt(weight[:, None]), ratios * np.sqrt(weight), rcond=None
            )[0]
            intercept, slope = float(fit[0]), float(fit[1])
    return intercept, slope, center, span


def _local_intensity(a, b, fa, fb):
    """Symmetric, background-projected template intensity on COMMON support.

    Independent detection windows are not comparable integrals: a weak nearby
    maximum can truncate only one of them. Here both observed templates use
    exactly the same offsets. A local constant and slope are projected out.
    Errors-in-both-axes (Deming) regression avoids treating the earlier noisy
    spectrum as an exact calculated curve. No peak width warp is fitted.
    """
    radius = min(
        2.25 * max(fa.width, fb.width),
        fa.position - fa.left,
        fa.right - fa.position,
        fb.position - fb.left,
        fb.right - fb.position,
    )
    # Reserve one native bin so the subsequent sub-bin search cannot query
    # outside observed support, even when every frame-edge bin is marked valid.
    radius -= max(fa.bin_width, fb.bin_width)
    count = min(128, int(2 * radius / max(fa.bin_width, fb.bin_width)))
    if count < 7:
        return None
    u = np.linspace(-radius, radius, count)
    design = np.c_[np.ones(count), u / radius]
    raw_a = np.interp(u + fa.position, a.observation.x, a.observation.intensity)
    sa = np.interp(u + fa.position, a.observation.x, a.noise)
    if not np.all(np.isfinite(raw_a)) or not np.all(np.isfinite(sa)):
        return None
    ca = raw_a - design @ np.linalg.lstsq(design, raw_a, rcond=None)[0]
    va = float(np.dot(ca, ca))
    na = float(np.mean(sa**2))
    if va <= 4 * count * na:
        return None
    candidates = []
    for shift in np.linspace(-0.75, 0.75, 7) * max(fa.bin_width, fb.bin_width):
        raw_b = np.interp(
            u + fb.position + shift, b.observation.x, b.observation.intensity
        )
        sb = np.interp(u + fb.position + shift, b.observation.x, b.noise)
        if not np.all(np.isfinite(raw_b)) or not np.all(np.isfinite(sb)):
            continue
        cb = raw_b - design @ np.linalg.lstsq(design, raw_b, rcond=None)[0]
        vb, covariance = float(np.dot(cb, cb)), float(np.dot(ca, cb))
        if covariance <= 0:
            continue
        nb = float(np.mean(sb**2))
        ratio = nb / max(na, 1e-30)
        diff = vb - ratio * va
        gain = (diff + math.sqrt(diff**2 + 4 * ratio * covariance**2)) / (
            2 * covariance
        )
        if gain <= 0 or not math.isfinite(gain):
            continue
        residual = cb - gain * ca
        combined_variance = nb + gain**2 * na
        log_error = 2 * math.sqrt(combined_variance / max(va * gain**2, 1e-30))
        score = float(np.dot(residual, residual) / max(vb + gain**2 * va, 1e-30))
        candidates.append(
            {
                "gain": gain,
                "log_error": log_error,
                "shape_residual": math.sqrt(score),
                "score": score,
                "native_points": count,
                "radius": radius,
            }
        )
    return min(candidates, key=lambda m: m["score"]) if candidates else None


def _local_shape(a, b, fa, fb):
    """Compare centered profiles without stretching their widths or reordering bins.

    A sub-bin translation search reduces discretization artifacts. No width
    warp is allowed: otherwise a resolved split could be registered away.
    """
    radius = min(
        2 * max(fa.width, fb.width),
        fa.position - fa.left,
        fa.right - fa.position,
        fb.position - fb.left,
        fb.right - fb.position,
    )
    radius -= max(fa.bin_width, fb.bin_width)
    if radius <= max(fa.bin_width, fb.bin_width):
        return None
    offsets = np.linspace(-radius, radius, 81)
    xa = offsets + fa.position
    ya = np.maximum(np.interp(xa, a.observation.x, a.excess), 0)
    if not np.all(np.isfinite(ya)) or np.sum(ya) <= 0:
        return None
    ya /= np.sum(ya)
    effects = []
    for shift in np.linspace(-0.5, 0.5, 5) * max(fa.bin_width, fb.bin_width):
        yb = np.maximum(
            np.interp(offsets + fb.position + shift, b.observation.x, b.excess), 0
        )
        if not np.all(np.isfinite(yb)) or np.sum(yb) <= 0:
            continue
        yb /= np.sum(yb)
        effects.append(float(0.5 * np.sum(np.abs(ya - yb))))
    return min(effects) if effects else None


def _supported(pattern, position, width):
    for a, b in pattern.segments:
        if (
            pattern.observation.x[a] <= position - width
            and position + width <= pattern.observation.x[b - 1]
        ):
            return True
    return False


def _unmatched_source_support(own, source, feature, matches, settings, source_gain, *, side):
    """Seek an unassigned predecessor on native samples, with uncertainty.

    A matched source peak is already accounted for and cannot be reused to
    explain an additional weak peak. Local profile fits span the allowed
    independent motion; they do not warp the pattern. New means newly resolved
    signal, never proof that a phase or an unresolved component did not exist.
    """
    pattern = source.observation
    x = pattern.x
    usable = pattern.valid & np.isfinite(source.noise) & (source.noise > 0)
    lower = feature.position / (1 + settings.motion_limit)
    upper = feature.position * (1 + settings.motion_limit)
    candidates = np.flatnonzero((x >= lower) & (x <= upper))
    result = {"status": "unresolved", "reason": "insufficient_native_source_support",
              "appearance_available": False, "appearance_status": "unresolved",
              "search_interval": [lower, upper], "source_gain": source_gain,
              "source_candidates": 0, "best_source": None, "source_upper_bound": None,
              "matched_source_peaks": 0, "occupied_source_fraction": 0.,
              "meaning": "Observable signal support only, not physical phase creation or absence."}
    if len(candidates) < 2:
        return result
    used = {m["before_position" if side == "after" else "after_position"] for m in matches}
    occupied = np.zeros(len(x), dtype=bool)
    competing = []
    for other in source.features:
        if other.position not in used:
            continue
        left = np.searchsorted(x, other.position - 2.25 * other.width, side="left")
        right = np.searchsorted(x, other.position + 2.25 * other.width, side="right")
        occupied[left:right] = True
        if lower <= other.position <= upper:
            result["matched_source_peaks"] += 1
            height_ratio = feature.height / max(other.height * source_gain, 1e-30)
            width_ratio = (feature.width / feature.position) / (other.width / other.position)
            if (abs(math.log(height_ratio)) <= settings.intensity_log_effect
                    and abs(math.log(width_ratio)) <= settings.width_log_effect):
                competing.append(other.position)
    result["occupied_source_fraction"] = float(np.mean(occupied[candidates]))
    result["prior_overlap_possible"] = bool(np.any(occupied[candidates]))
    result["competing_assigned_positions"] = competing
    if competing:
        # A dense assignment can consume the actual counterpart. It cannot
        # then manufacture absence for this unmatched feature. Conversely, a
        # much stronger unrelated matched peak is not a weak predecessor.
        result["reason"] = "compatible_source_feature_is_already_assigned"
        return result
    missing = lower < x[0] or upper > x[-1] or not np.all(usable[candidates])
    # The bounded center grid must still resolve the measured feature. A
    # capacity limit is unknown support, not evidence of absence.
    if len(candidates) > 128:
        candidates = candidates[np.unique(np.linspace(0, len(candidates) - 1, 128, dtype=int))]
        if np.max(np.diff(x[candidates])) > feature.width / 2:
            result["reason"] = "source_motion_search_exceeds_native_resolution_bound"
            return result
    fits = []
    for index in candidates:
        center = float(x[index])
        width = feature.width * center / feature.position
        left = np.searchsorted(x, center - 2.25 * width, side="left")
        right = np.searchsorted(x, center + 2.25 * width, side="right")
        xx, valid = x[left:right], usable[left:right]
        if len(xx) < 8 or not np.all(valid):
            missing = True
            continue
        available = ~occupied[left:right]
        if np.count_nonzero(available) < 8:
            continue
        z = (xx[available] - center) / width
        template = np.exp(-4 * math.log(2) * z * z)
        full_template_energy = np.sum(np.exp(-8 * math.log(2) * ((xx-center)/width)**2))
        if np.dot(template, template) / max(float(full_template_energy), 1e-30) < settings.min_coverage:
            continue
        design = np.c_[template, np.ones(len(z)), z]
        sigma = np.maximum(source.noise[left:right][available], np.finfo(float).tiny)
        matrix, target = design / sigma[:, None], pattern.intensity[left:right][available] / sigma
        normal = matrix.T @ matrix
        if not np.all(np.isfinite(normal)):
            continue
        try:
            if np.linalg.cond(normal) > 1e10:
                continue
            covariance = np.linalg.inv(normal)
        except np.linalg.LinAlgError:
            continue
        coefficients = covariance @ (matrix.T @ target)
        residual = target - matrix @ coefficients
        scatter = max(1., float(np.dot(residual, residual)) / max(1, len(z) - 3))
        amplitude = float(coefficients[0])
        error = math.hypot(math.sqrt(max(float(covariance[0, 0]) * scatter, 0.)),
                           settings.area_systematic_fraction * abs(amplitude))
        if math.isfinite(error) and error > 0:
            fits.append({"position": center, "height": amplitude, "error": error,
                         "error_ratio": amplitude / error, "native_points": len(z),
                         "residual_scatter": scatter})
    result.update(source_candidates=len(fits), missing_source_support=bool(missing))
    if not fits:
        return result
    best = max(fits, key=lambda row: row["error_ratio"])
    upper_bound = max(max(0., row["height"]) + settings.change_snr * row["error"] for row in fits)
    own_error = feature.height / max(feature.snr, 1e-30)
    least_change = min((feature.height - max(0., row["height"]) * source_gain) / math.sqrt(
        own_error**2 + (row["error"] * source_gain)**2 +
        (settings.area_systematic_fraction * feature.height)**2) for row in fits)
    result.update(best_source=best, source_upper_bound=upper_bound, minimum_change_error_ratio=least_change)
    if feature.crowded:
        result["reason"] = "new_feature_blends_with_an_existing_resolved_peak"
    elif best["error_ratio"] >= settings.change_snr:
        result.update(status="preexisting", reason="supported_unassigned_source_profile",
                      appearance_available=True, appearance_status="no_resolved_change")
    elif missing:
        result["reason"] = "unobserved_source_region_could_contain_counterpart"
    elif (least_change < settings.change_snr or feature.height < math.exp(settings.intensity_log_effect) *
          source_gain * max(max(0., row["height"]) for row in fits)):
        # Complete observed support can fail the change rule even when the
        # physical origin is unresolved. This is a negative under the selected
        # evidence rule, not a claim that the predecessor or a phase was absent.
        result.update(reason="consistent_with_source_uncertainty", appearance_available=True,
                      appearance_status="no_resolved_change")
    elif best["error_ratio"] >= settings.change_snr / 2:
        # A significant change with weaker positive predecessor support still
        # leaves origin unresolved. Do not lower the detection threshold.
        result["reason"] = "weak_source_profile_requires_review"
    else:
        result.update(status="new", reason="new_resolved_signal_without_supported_unassigned_counterpart",
                      appearance_available=True, appearance_status="new_resolved_signal")
    return result


def _union_regions(evidence):
    groups = []
    for event in sorted(evidence, key=lambda e: e["interval"][0]):
        lo, hi = event["interval"]
        if groups and lo <= groups[-1]["interval"][1]:
            group = groups[-1]
            group["interval"][1] = max(hi, group["interval"][1])
            group["mass"] = max(group["mass"], event["mass"])
            group["kinds"] = sorted(set(group["kinds"] + [event["kind"]]))
            group["evidence"].append(event)
        else:
            groups.append(
                {
                    "interval": [lo, hi],
                    "mass": event["mass"],
                    "kinds": [event["kind"]],
                    "evidence": [event],
                }
            )
    return groups


def _component_changes(fa, fb, costs, total, gain_at, height, settings):
    """Find well-resolved one-to-many feature groups, without assigning hkl.

    Ambiguous daughter correspondence must not hide an unambiguous change in
    the number of resolved components. Conversely, a larger/crowded cluster
    cannot support this simple counterfactual and remains locally ambiguous.
    Coalescence can also be caused by independent moving peaks, so the output
    is an observed component change, never proof of a structural transition.
    """
    edges = np.argwhere(costs < 2 * settings.unmatched_cost)
    adjacency = {}
    n = len(fa)
    for i, j in edges:
        adjacency.setdefault(int(i), set()).add(n + int(j))
        adjacency.setdefault(n + int(j), set()).add(int(i))
    visited = set()
    events = []
    for node in adjacency:
        if node in visited:
            continue
        stack = [node]
        component = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(adjacency[current] - component)
        visited |= component
        left = [fa[i] for i in component if i < n]
        right = [fb[i - n] for i in component if i >= n]
        if not (
            (len(left) == 1 and 2 <= len(right) <= 3)
            or (len(right) == 1 and 2 <= len(left) <= 3)
        ):
            continue
        children = right if len(left) == 1 else left
        children = sorted(children, key=lambda f: f.position)
        if min(f.snr for f in left + right) < 2 * settings.peak_snr:
            continue
        if min(f.height for f in children) < 0.15 * max(f.height for f in children):
            continue
        if (
            min(
                (q.position - p.position) / min(p.width, q.width)
                for p, q in pairwise(children)
            )
            < 1.1
        ):
            continue
        mass = (
            max(
                sum(f.area * gain_at(f.position) for f in left),
                sum(f.area for f in right),
            )
            / total
        )
        if (
            max(
                [f.height * gain_at(f.position) for f in left]
                + [f.height for f in right]
            )
            / height
            < settings.min_feature_height_fraction
        ):
            continue
        events.append(
            {
                "kind": "resolved_component_count_change",
                "interval": [
                    min(f.position - f.width for f in left + right),
                    max(f.position + f.width for f in left + right),
                ],
                "mass": mass,
                "before_components": len(left),
                "after_components": len(right),
                "physical_cause": "undetermined; includes coalescence of independently moving peaks",
            }
        )
    return events


def compare_observed_patterns(
    before: ObservedPattern | PreparedPattern,
    after: ObservedPattern | PreparedPattern,
    settings: TransitionSettings | None = None,
):
    """Return a replayable three-way evidence assessment with localized reasons.

    ``transition_evidence`` means a visible change warrants investigation, not
    a certified transition. ``no_resolved_change`` never asserts absence of a
    weak/unresolved transition. ``uncertain`` covers compatibility/quality or
    ambiguous effects. No fitted RADAR-PD information is consulted.
    """
    settings = settings or TransitionSettings()
    a = (
        before
        if isinstance(before, PreparedPattern)
        else prepare_pattern(before, settings)
    )
    b = (
        after
        if isinstance(after, PreparedPattern)
        else prepare_pattern(after, settings)
    )
    if (
        a.settings_fingerprint != settings.preparation_fingerprint
        or b.settings_fingerprint != settings.preparation_fingerprint
    ):
        raise ValueError(
            "Prepared features were made with different sensitivity settings"
        )
    report = {
        "schema": "radar-pd-observed-transition/v1",
        "engine": ENGINE_VERSION,
        "settings": asdict(settings),
        "settings_sha256": settings.fingerprint,
        "before": a.observation.observation_id,
        "after": b.observation.observation_id,
        "status": "uncertain",
        "reasons": [],
        "regions": [],
        "matches": [],
        "quality": {"before": a.quality, "after": b.quality},
        "metrics": {},
        "qualifications": [
            "Unmatched features are unmatched only within the declared per-feature movement bound.",
            "Broad/diffuse components beyond the configured relative width are retained for separate review, not classified here.",
            "Changing texture, absorption, illumination or reduction can also change relative intensities.",
            "Peak coalescence/splitting alone does not establish a space-group change.",
        ],
        "interpretation_limit": "Observed-pattern evidence; no chemical phase, magnetic origin, k vector, or thermodynamic transition is identified.",
    }
    scope_a, scope_b = asdict(a.observation.scope), asdict(b.observation.scope)
    incompatible = [k for k in scope_a if scope_a[k] != scope_b[k]]
    if incompatible:
        report["reasons"] = ["incompatible_acquisition_scope:" + ",".join(incompatible)]
        return report
    if a.quality["feature_cap_hit"] or b.quality["feature_cap_hit"]:
        report["reasons"] = ["feature_cap_hit"]
        return report
    if (
        min(a.quality["usable_segment_fraction"], b.quality["usable_segment_fraction"])
        < settings.min_coverage
    ):
        report["reasons"] = ["fragmented_or_insufficient_valid_data"]
        return report
    # Common valid support is measured on BOTH native grids; interpolating a
    # mask to a union grid must never turn an excluded region into an observed one.
    support = []
    for left, right in [(a, b), (b, a)]:
        good = np.zeros(len(left.observation.x), dtype=bool)
        for begin, end in right.segments:
            good |= (left.observation.x >= right.observation.x[begin]) & (
                left.observation.x <= right.observation.x[end - 1]
            )
        support.append(
            float(
                np.count_nonzero(good & left.observation.valid)
                / max(np.count_nonzero(left.observation.valid), 1)
            )
        )
    report["metrics"]["common_native_coverage"] = min(support)
    if min(support) < settings.min_coverage:
        report["reasons"] = ["insufficient_common_coverage"]
        return report
    (matches, unmatched_a, unmatched_b), costs = _match_features(a, b, settings)
    fa, fb = a.features, b.features
    local_intensities = [_local_intensity(a, b, fa[i], fb[j]) for i, j in matches]
    gain0, gain1, center, span = _gain_model(
        fa, fb, matches, a.observation.scope.coordinate, settings, local_intensities
    )

    def gain_at(position):
        q2 = (
            position**2
            if a.observation.scope.coordinate == "q_inv_angstrom"
            else 1 / position**2
        )
        return math.exp(
            float(
                np.clip(gain0 + gain1 * np.clip((q2 - center) / span, -2, 2), -50, 50)
            )
        )

    # Matched-feature areas calibrate the nuisance gain. If there are no matches,
    # normalize total measured feature mass, not a single strongest peak.
    if not matches and fa and fb:
        gain0 = math.log(sum(f.area for f in fb) / sum(f.area for f in fa))
    total = max(
        sum(f.area * gain_at(f.position) for f in fa), sum(f.area for f in fb), 1e-30
    )
    height = max(
        [f.height * gain_at(f.position) for f in fa] + [f.height for f in fb] + [1e-30]
    )
    evidence = _component_changes(fa, fb, costs, total, gain_at, height, settings)
    hints, motion, ambiguity_mass = [], [], 0.0
    for pair_index, (i, j) in enumerate(matches):
        p, q = fa[i], fb[j]
        scale = gain_at(math.sqrt(p.position * q.position))
        intensity = local_intensities[pair_index]
        area_delta = math.log(intensity["gain"] / scale) if intensity else None
        area_uncertainty = (
            math.hypot(intensity["log_error"], settings.area_systematic_fraction)
            if intensity
            else None
        )
        area_z = abs(area_delta) / max(area_uncertainty, 1e-30) if intensity else None
        width_delta = math.log(q.width / p.width)
        shape = _local_shape(a, b, p, q)
        mass = max(p.area * scale, q.area) / total
        delta = q.position / p.position - 1
        motion.append(delta)
        # Local alternative margins are conservative ambiguity diagnostics, not
        # probabilities. Global assignment may deliberately forgo a local best.
        alternatives = [2 * settings.unmatched_cost]
        alternatives += [
            float(c) for k, c in enumerate(costs[i]) if k != j and math.isfinite(c)
        ]
        alternatives += [
            float(c) for k, c in enumerate(costs[:, j]) if k != i and math.isfinite(c)
        ]
        margin = min(alternatives) - float(costs[i, j])
        ambiguous = margin < settings.assignment_margin
        if ambiguous:
            ambiguity_mass += mass
        result = {
            "before_position": p.position,
            "after_position": q.position,
            "relative_motion": delta,
            "before_width": p.width,
            "after_width": q.width,
            "log_width_ratio": width_delta,
            "log_relative_intensity": area_delta,
            "intensity_error_ratio": area_z,
            "shape_distance": shape,
            "common_window_intensity": intensity,
            "mass": mass,
            "assignment_margin": margin,
            "ambiguous": ambiguous,
            "before_snr": p.snr,
            "after_snr": q.snr,
            "crowded": p.crowded or q.crowded,
        }
        report["matches"].append(result)
        interval = [
            min(p.position - p.width, q.position - q.width),
            max(p.position + p.width, q.position + q.width),
        ]
        common = {
            "interval": interval,
            "mass": mass,
            "before_position": p.position,
            "after_position": q.position,
        }
        enough_height = (
            max(p.height * scale, q.height) / height
            >= settings.min_feature_height_fraction
        )
        if (
            intensity
            and enough_height
            and abs(area_delta) >= settings.intensity_log_effect
            and area_z >= settings.change_snr
        ):
            event = {
                **common,
                "kind": "relative_intensity_redistribution",
                "effect": abs(area_delta),
                "error_ratio": area_z,
            }
            (hints if ambiguous else evidence).append(event)
        if (
            enough_height
            and min(p.snr, q.snr) >= 2 * settings.peak_snr
            and min(p.width_bins, q.width_bins) >= 3
            and abs(width_delta) >= settings.width_log_effect
            and shape is not None
            and shape >= settings.shape_effect
        ):
            event = {
                **common,
                "kind": "local_shape_or_component_change",
                "effect": shape,
                "log_width_ratio": width_delta,
            }
            (hints if ambiguous else evidence).append(event)
    unmatched = []
    for side, indices, own, other, own_prepared, other_prepared in [
        ("before", unmatched_a, fa, fb, a, b),
        ("after", unmatched_b, fb, fa, b, a),
    ]:
        for index in indices:
            p = own[index]
            own_scale = gain_at(p.position) if side == "before" else 1.0
            mass = p.area * own_scale / total
            item = {
                "side": side,
                "position": p.position,
                "width": p.width,
                "height_fraction": p.height * own_scale / height,
                "snr": p.snr,
                "mass": mass,
                "supported_in_both": _supported(
                    other_prepared, p.position, 1.5 * p.width
                ),
            }
            unmatched.append(item)
            if item["height_fraction"] < settings.min_feature_height_fraction:
                continue
            if not item["supported_in_both"]:
                item["source_support"] = {"status": "unresolved",
                    "reason": "insufficient_coincident_native_support", "missing_source_support": True}
                continue
            support = _unmatched_source_support(own_prepared, other_prepared, p, report["matches"], settings,
                gain_at(p.position) if side == "after" else 1 / gain_at(p.position), side=side)
            item["source_support"] = support
            kind = ("additional_or_lost_signal" if support["status"] == "new" else
                    "preexisting_signal_motion_or_intensity_change" if support["status"] == "preexisting" else
                    "unresolved_signal_origin")
            interval = [p.position - 1.5 * p.width, p.position + 1.5 * p.width]
            event = {
                "interval": interval,
                "kind": kind,
                "mass": mass,
                "position": p.position,
                "side": side,
                "snr": p.snr,
                "source_support": support,
            }
            if support["status"] == "new":
                evidence.append(event)
            else:
                hints.append(event)
    regions = _union_regions(evidence)
    report["regions"] = regions
    changed_mass = min(sum(r["mass"] for r in regions), 1.0)
    max_mass = max([r["mass"] for r in regions] + [0])
    report["unmatched"] = unmatched
    # Retain this compact evidence even when a caller drops the detailed
    # assignment and hint lists. Unknown support must not become a negative.
    report["source_support"] = [{"side": row["side"], "position": row["position"],
        "interval": [row["position"] - 1.5*row["width"], row["position"] + 1.5*row["width"]],
        **row["source_support"]} for row in unmatched if "source_support" in row]
    report["hints"] = _union_regions(hints)
    report["metrics"].update(
        matched_features=len(matches),
        before_features=len(fa),
        after_features=len(fb),
        unmatched_before=len(unmatched_a),
        unmatched_after=len(unmatched_b),
        independent_changed_regions=len(regions),
        changed_mass=changed_mass,
        largest_changed_region_mass=max_mass,
        ambiguous_matched_mass=min(ambiguity_mass, 1.0),
        median_relative_motion=float(np.median(motion)) if motion else None,
        min_relative_motion=min(motion) if motion else None,
        max_relative_motion=max(motion) if motion else None,
        gain_intercept_log=gain0,
        gain_smooth_q2_slope=gain1,
        intensity_nuisance="robust global gain + smooth position-dependent intensity slope"
        if settings.smooth_intensity_nuisance
        else "robust global gain",
        hints=len(report["hints"]),
    )
    if (
        len(regions) >= settings.min_independent_regions
        and changed_mass >= settings.min_changed_mass
    ) or max_mass >= settings.single_region_mass:
        report["status"] = "transition_evidence"
        report["reasons"] = sorted({kind for r in regions for kind in r["kinds"]})
    elif not fa and not fb:
        report["reasons"] = ["no_resolved_features_for_transition_comparison"]
    elif len(matches) < 3 and (not regions or changed_mass < settings.min_changed_mass):
        report["reasons"] = ["insufficient_stable_correspondence"]
    elif changed_mass >= settings.min_changed_mass / 3 or ambiguity_mass > 0.35:
        report["reasons"] = ["localized_or_ambiguous_change_requires_review"]
    else:
        report["status"] = "no_resolved_change"
        report["reasons"] = [
            "existing_features_explain_patterns_with_independent_local_motion"
        ]
    return report


@dataclass
class TransitionMonitor:
    """Causal previous-scan AND fixed-anchor comparisons, separately per scope.

    The anchor does not silently follow the latest scan: doing that would hide
    a gradual onset. It changes only through explicit ``reset_anchor``. This
    class never uses a future scan to retrospectively raise a live alert.
    """

    settings: TransitionSettings = field(default_factory=TransitionSettings)
    max_scopes: int = 128
    _previous: dict = field(default_factory=dict, init=False, repr=False)
    _anchors: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.max_scopes, int) or not 1 <= self.max_scopes <= 256:
            raise ValueError("Monitor capacity must be between 1 and 256 scopes")

    def observe(self, pattern: ObservedPattern):
        key = pattern.scope
        if key not in self._previous and len(self._previous) >= self.max_scopes:
            raise ValueError(
                "Monitor scope limit reached; explicitly close an inactive series"
            )
        prepared = prepare_pattern(pattern, self.settings)
        if key not in self._previous:
            if prepared.quality["usable_segment_fraction"] < self.settings.min_coverage:
                return {
                    "status": "uncertain",
                    "reason": "first_scan_quality_insufficient",
                    "observation": pattern.observation_id,
                }
            self._previous[key] = self._anchors[key] = prepared
            return {
                "status": "reference_initialized",
                "observation": pattern.observation_id,
            }
        previous = compare_observed_patterns(
            self._previous[key], prepared, self.settings
        )
        anchor = compare_observed_patterns(self._anchors[key], prepared, self.settings)
        if prepared.quality["usable_segment_fraction"] >= self.settings.min_coverage:
            self._previous[key] = prepared
        return {
            # "What just happened?" and "Have we changed since the reference?"
            # are different questions. A stable new state must not repeatedly
            # announce a new event, and returning to the reference can itself
            # be a new change relative to the preceding scan.
            "status": previous["status"],
            "new_change_evidence": previous["status"] == "transition_evidence",
            "change_since_anchor": anchor["status"] == "transition_evidence",
            "observation": pattern.observation_id,
            "previous_comparison": previous,
            "anchor_comparison": anchor,
            "provisional": True,
            "physical_transition_confirmed": False,
        }

    def reset_anchor(self, pattern: ObservedPattern):
        if pattern.scope not in self._anchors and len(self._anchors) >= self.max_scopes:
            raise ValueError("Monitor scope limit reached")
        prepared = prepare_pattern(pattern, self.settings)
        if prepared.quality["usable_segment_fraction"] < self.settings.min_coverage:
            raise ValueError("Cannot use a fragmented/invalid scan as a reference")
        self._anchors[pattern.scope] = prepared
        self._previous[pattern.scope] = prepared

    def close_scope(self, scope: PatternScope):
        self._previous.pop(scope, None)
        self._anchors.pop(scope, None)
