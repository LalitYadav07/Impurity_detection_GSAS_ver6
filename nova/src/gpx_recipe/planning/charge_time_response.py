"""Conditional raw-count response with separate recorded-dose and time terms.

Zero recorded charge is an input value, not evidence of neutron beam-off. These
working Poisson fits do not establish the physical origin of either component.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import erf, erfc

from .aperture_response import poisson_deviance

VERSION = "powgen-charge-time-response/1"
APERTURE_VERSION = "powgen-aperture-charge-time-response/1"


def _vector(value, name, *, positive=False):
    raw = np.asarray(value)
    if raw.dtype.kind == "b":
        raise ValueError(f"{name} cannot be boolean")
    out = np.asarray(value, dtype=float)
    if out.ndim != 1 or not 1 <= out.size <= 8192 or not np.isfinite(out).all():
        raise ValueError(f"Finite bounded one-dimensional {name} is required")
    if np.any(out <= 0 if positive else out < 0):
        raise ValueError(f"{'Positive' if positive else 'Nonnegative'} {name} is required")
    return out


def _counts(value):
    out = _vector(value, "raw counts")
    if np.any(out != np.floor(out)) or out.sum() > 2**53:
        raise ValueError("Exactly representable nonnegative integer raw counts are required")
    return out


def fit_charge_time(signal_exposure, elapsed_s, counts):
    """Fit mean = signal_exposure * signal_rate + elapsed_s * time_rate.

    Profiling the total count leaves a convex one-dimensional mixture problem.
    Both boundary solutions are admitted. Collinear exposures are explicitly
    unidentified and cannot be used to predict a new exposure ratio.
    """
    x = _vector(signal_exposure, "signal exposure")
    t = _vector(elapsed_s, "elapsed seconds")
    n = _counts(counts)
    if not len(x) == len(t) == len(n):
        raise ValueError("Matching exposure, time and raw counts are required")
    sx, st = float(x.sum()), float(t.sum())
    if not np.isfinite(sx + st) or sx <= 0 or st <= 0:
        raise ValueError("Both components require positive finite total exposure")
    if np.any((x == 0) & (t == 0) & (n > 0)):
        raise ValueError("Positive counts cannot have zero exposure to both components")
    design = np.column_stack((x/sx, t/st))
    singular = np.linalg.svd(design, compute_uv=False)
    rank_tolerance = np.finfo(float).eps * max(design.shape) * singular[0]
    rank = int(np.count_nonzero(singular > rank_tolerance))
    total = float(n.sum())
    result = {
        "schema": VERSION, "signal_exposure_sum": sx, "elapsed_seconds_sum": st,
        "raw_count_sum": total, "normalized_design_singular_values": singular.tolist(),
        "normalized_design_rank": rank,
        "normalized_design_condition": float(singular[0]/singular[-1]) if rank == 2 else None,
        "experimental_accuracy_certified": False,
        "time_component_physical_origin_identified": False,
        "zero_recorded_charge_is_neutron_beam_off": False,
    }
    positive = n > 0
    relative_counts = n[positive]/total if total else n[positive]
    s, u = design[positive].T
    delta = u-s

    def slope(weight):
        probability = (1-weight)*s + weight*u
        if np.any(probability == 0):
            return -float("inf") if weight == 0 else float("inf")
        return -float(np.dot(relative_counts, delta/probability))

    if total == 0:
        weight, status = 0., "zero_counts_no_interval_coverage_established"
    elif rank != 2:
        weight, status = 0., "unidentified_collinear_exposures"
    elif slope(0.) >= 0:
        weight, status = 0., "fitted_signal_only_boundary"
    elif slope(1.) <= 0:
        weight, status = 1., "fitted_time_only_boundary"
    else:
        weight = float(brentq(slope, 0., 1., xtol=np.nextafter(0., 1.),
                              rtol=4*np.finfo(float).eps, maxiter=2000))
        status = "fitted_interior"
    signal_rate = total*(1-weight)/sx
    time_rate = total*weight/st
    mean = x*signal_rate + t*time_rate
    covariance = None
    covariance_status = "unavailable_at_boundary_or_unidentified"
    if status == "fitted_interior":
        exposed = mean > 0
        information = design[exposed].T @ (design[exposed]/mean[exposed, None])
        condition = np.linalg.cond(information)
        if condition < 1/np.sqrt(np.finfo(float).eps):
            normalized_covariance = np.linalg.inv(information)
            covariance = (normalized_covariance/np.outer([sx, st], [sx, st])).tolist()
            covariance_status = "interior_fixed_design_expected_fisher"
        else:
            covariance_status = "ill_conditioned_fisher_not_inverted"
    result.update(
        status=status, signal_rate=signal_rate, time_rate_per_s=time_rate,
        time_component_count_fraction=weight, training_mean=mean.tolist(),
        working_poisson_deviance=poisson_deviance(n, mean),
        mixture_slope_per_count=float(slope(weight)) if total and rank == 2 else None,
        conditional_local_rate_covariance=covariance,
        covariance_status=covariance_status,
        covariance_scope="Expected Fisher, interior rates and fixed design only; excludes shape/source/state/model uncertainty; no calibrated predictive coverage",
    )
    return result


def predict_charge_time(calibration, signal_exposure, elapsed_s):
    if calibration.get("schema") != VERSION:
        raise ValueError("A matching charge-time response is required")
    if calibration.get("normalized_design_rank") != 2:
        raise ValueError("Unidentified component rates cannot predict a new exposure")
    x = _vector(signal_exposure, "target signal exposure")
    t = _vector(elapsed_s, "target elapsed seconds")
    if len(x) != len(t):
        raise ValueError("Matching target exposure and time are required")
    rates = np.asarray([calibration["signal_rate"], calibration["time_rate_per_s"]], float)
    if not np.isfinite(rates).all() or np.any(rates < 0):
        raise ValueError("Finite nonnegative component rates are required")
    mean = x*rates[0] + t*rates[1]
    if not np.isfinite(mean).all():
        raise ValueError("A finite target mean is required")
    return mean


def gaussian_interval_acceptance(heights, sigma, *, centers=None):
    """Normalized Gaussian integral over each declared aperture interval."""
    h = _vector(heights, "aperture heights", positive=True)
    c = np.zeros_like(h) if centers is None else np.asarray(centers, float)
    if c.shape != h.shape or not np.isfinite(c).all() or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Matching finite centers and a positive finite width are required")
    low, high = (c-h/2)/(np.sqrt(2)*sigma), (c+h/2)/(np.sqrt(2)*sigma)
    out = np.empty_like(h)
    right, left = low >= 0, high <= 0
    middle = ~(right | left)
    out[right] = .5*(erfc(low[right])-erfc(high[right]))
    out[left] = .5*(erfc(-high[left])-erfc(-low[left]))
    out[middle] = .5*(erf(high[middle])-erf(low[middle]))
    return out


def fit_aperture_charge_time(model, heights, recorded_dose, elapsed_s, counts,
                             *, centers=None, sigma_bounds=(.5, 200.)):
    """Fit uniform/Gaussian aperture signal with an aperture-independent time rate.

    Repeated heights and zero-dose blocks are permitted. Heights and centers
    remain recorded coordinates; their physical metrology is not inferred.
    """
    if model not in ("uniform", "gaussian"):
        raise ValueError("Uniform or Gaussian aperture response is required")
    h = _vector(heights, "aperture heights", positive=True)
    q = _vector(recorded_dose, "recorded dose")
    t = _vector(elapsed_s, "elapsed seconds")
    n = _counts(counts)
    c = np.zeros_like(h) if centers is None else np.asarray(centers, float)
    if not len(h) == len(q) == len(t) == len(n) or c.shape != h.shape or not np.isfinite(c).all():
        raise ValueError("Matching aperture, center, dose, time and raw counts are required")
    reference_height = float(h.max())
    search = None
    sigma = None
    if model == "uniform":
        response = fit_charge_time(q*h/reference_height, t, n)
    else:
        if len(np.unique(np.column_stack((h, c)), axis=0)) < 2:
            raise ValueError("Gaussian width requires distinct aperture intervals")
        bounds = np.asarray(sigma_bounds, float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 < bounds[0] < bounds[1]:
            raise ValueError("Increasing positive finite Gaussian width bounds are required")

        def evaluate(logsigma):
            shape = gaussian_interval_acceptance(h, np.exp(logsigma), centers=c)
            if np.dot(q, shape) == 0:
                return None
            return fit_charge_time(q*shape, t, n)

        grid = np.linspace(*np.log(bounds), 41)
        points = [(float(v), evaluate(float(v))) for v in grid]
        candidates = [(v, r) for v, r in points if r is not None]
        if not candidates:
            raise ValueError("Gaussian acceptance underflow over the entire width domain")
        refinements = []
        for i in range(1, len(grid)-1):
            if any(points[j][1] is None for j in (i-1, i, i+1)):
                continue
            if points[i][1]["working_poisson_deviance"] <= min(points[j][1]["working_poisson_deviance"] for j in (i-1, i+1)):
                opt = minimize_scalar(lambda v: evaluate(float(v))["working_poisson_deviance"],
                                      bounds=(float(grid[i-1]), float(grid[i+1])), method="bounded",
                                      options={"xatol": 1e-10, "maxiter": 500})
                candidates.append((float(opt.x), evaluate(float(opt.x))))
                refinements.append({"bracket": [float(grid[i-1]), float(grid[i+1])],
                                    "success": bool(opt.success)})
        coordinate, response = min(candidates, key=lambda p: p[1]["working_poisson_deviance"])
        sigma = float(np.exp(coordinate))
        search = {"sigma_bounds": bounds.tolist(), "fixed_log_grid_points": 41,
                  "grid_profile": [{"sigma": float(np.exp(v)),
                                    "deviance": r["working_poisson_deviance"] if r is not None else None}
                                   for v, r in points], "refinements": refinements,
                  "sigma_at_boundary": bool(coordinate == grid[0] or coordinate == grid[-1]),
                  "all_refinements_succeeded": all(v["success"] for v in refinements)}
    return {
        "schema": APERTURE_VERSION, "model": model, "response": response,
        "sigma": sigma, "reference_height": reference_height, "profile_search": search,
        "height_range": [float(h.min()), float(h.max())],
        "aperture_coordinate_envelope": [float(np.min(c-h/2)), float(np.max(c+h/2))],
        "calibration_center_range": [float(c.min()), float(c.max())],
        "coordinate_units": "Declared shared readback; physical units not inferred",
        "experimental_accuracy_certified": False, "pure_incident_beam_identified": False,
        "covariance_excludes_fitted_width_and_source_state": True,
    }


def predict_aperture_charge_time(calibration, heights, recorded_dose, elapsed_s,
                                 *, centers=None):
    if calibration.get("schema") != APERTURE_VERSION or calibration.get("model") not in ("uniform", "gaussian"):
        raise ValueError("A matching aperture charge-time response is required")
    h = _vector(heights, "target aperture heights", positive=True)
    q = _vector(recorded_dose, "target recorded dose")
    c = np.zeros_like(h) if centers is None else np.asarray(centers, float)
    if len(h) != len(q) or c.shape != h.shape or not np.isfinite(c).all():
        raise ValueError("Matching target aperture, center and dose are required")
    low, high = calibration["aperture_coordinate_envelope"]
    if (np.any(h < calibration["height_range"][0]) or np.any(h > calibration["height_range"][1])
            or np.any(c-h/2 < low) or np.any(c+h/2 > high)):
        raise ValueError("Target aperture must lie inside the calibration coordinate envelope")
    shape = (h/calibration["reference_height"] if calibration["model"] == "uniform"
             else gaussian_interval_acceptance(h, calibration["sigma"], centers=c))
    return predict_charge_time(calibration["response"], q*shape, elapsed_s)
