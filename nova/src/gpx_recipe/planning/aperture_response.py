"""Conditional accepted-aperture response models for recorded raw counts.

Height uses an explicitly shared readback coordinate, not an inferred physical
unit. These models include specimen/detector acceptance and do not identify a
pure incident beam. Counts, recorded dose and source drift remain distinct.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import erf

VERSION = "powgen-accepted-aperture-response/1"
MODELS = ("uniform", "uniform_background", "gaussian", "gaussian_background",
          "interpolation", "time_only", "uniform_time_drift")


def _vector(value, name, *, positive=False):
    a = np.asarray(value)
    if a.dtype.kind == "b":
        raise ValueError(f"{name} cannot be boolean")
    a = np.asarray(a, dtype=float)
    if a.ndim != 1 or not 1 <= len(a) <= 512 or not np.isfinite(a).all():
        raise ValueError(f"Finite bounded one-dimensional {name} is required")
    if positive and np.any(a <= 0):
        raise ValueError(f"Positive {name} is required")
    return a


def poisson_deviance(counts, mean):
    """Stable working Poisson deviance; it does not certify a Poisson process."""
    n, mu = np.broadcast_arrays(np.asarray(counts, float), np.asarray(mean, float))
    if (not np.isfinite(n).all() or not np.isfinite(mu).all() or np.any(n < 0) or np.any(mu < 0)):
        raise ValueError("Nonnegative finite counts and means are required")
    if np.any((n > 0) & (mu == 0)):
        return float("inf")
    positive = n > 0
    relative = (mu[positive]-n[positive])/n[positive]
    small_mean = mu[positive] < .5*n[positive]
    terms = np.empty_like(relative)
    terms[small_mean] = (mu[positive][small_mean]-n[positive][small_mean]
                        +n[positive][small_mean]*(np.log(n[positive][small_mean])-np.log(mu[positive][small_mean])))
    terms[~small_mean] = n[positive][~small_mean]*(relative[~small_mean]-np.log1p(relative[~small_mean]))
    return float(2*(mu[~positive].sum()+terms.sum()))


def interpolation_weights(calibration_heights, target_heights):
    h = _vector(calibration_heights, "calibration heights", positive=True)
    target = _vector(target_heights, "target heights", positive=True)
    if len(h) < 2 or np.any(np.diff(h) <= 0):
        raise ValueError("Strictly increasing distinct calibration heights are required")
    if np.any(target < h[0]) or np.any(target > h[-1]):
        raise ValueError("Aperture interpolation cannot extrapolate")
    right = np.clip(np.searchsorted(h, target, side="left"), 1, len(h)-1)
    left = right-1
    fraction = (target-h[left])/(h[right]-h[left])
    weights = np.zeros((len(target), len(h)))
    weights[np.arange(len(target)), left] = 1-fraction
    weights[np.arange(len(target)), right] = fraction
    return weights


def gaussian_acceptance(height, sigma):
    """Integral of a centered normalized Gaussian over [-height/2,height/2]."""
    h = np.asarray(height, float)
    if not np.isfinite(h).all() or np.any(h <= 0) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Positive finite height and Gaussian width are required")
    return erf(h/(2*np.sqrt(2)*sigma))


def _scale(n, dose, shape):
    amplitude = float(n.sum()/np.dot(dose, shape))
    mean = dose*amplitude*shape
    if not np.isfinite(amplitude) or not np.isfinite(mean).all():
        raise ValueError("A finite conditional rate calibration is required")
    return amplitude, mean, poisson_deviance(n, mean)


def _mix(n, dose, shape):
    def evaluate(w):
        amplitude, mean, loss = _scale(n, dose, (1-w)*shape+w)
        return {"weight": float(w), "amplitude": amplitude, "mean": mean, "loss": loss}
    optimized = minimize_scalar(lambda w: evaluate(w)["loss"], bounds=(0., 1.),
                                method="bounded", options={"xatol": 1e-12, "maxiter": 500})
    candidates = [evaluate(0.), evaluate(1.), evaluate(float(optimized.x))]
    result = min(candidates, key=lambda x: x["loss"])
    result["optimizer_success"] = bool(optimized.success)
    return result


def _profile_search(function, bounds):
    """Retain a fixed grid and refine every interior grid minimum."""
    grid = np.linspace(*bounds, 41)
    evaluated = [(float(x), function(float(x))) for x in grid]
    candidates = list(evaluated)
    refinements = []
    for i in range(1, len(grid)-1):
        if evaluated[i][1]["loss"] <= min(evaluated[i-1][1]["loss"], evaluated[i+1][1]["loss"]):
            opt = minimize_scalar(lambda x: function(float(x))["loss"],
                                  bounds=(float(grid[i-1]), float(grid[i+1])), method="bounded",
                                  options={"xatol": 1e-10, "maxiter": 500})
            candidates.append((float(opt.x), function(float(opt.x))))
            refinements.append({"bracket": [float(grid[i-1]), float(grid[i+1])], "success": bool(opt.success)})
    x, result = min(candidates, key=lambda row: row[1]["loss"])
    return x, result, {"bounds": list(bounds), "grid_points": 41, "refinements": refinements,
                       "grid_profile": [{"coordinate": v, "deviance": r["loss"]} for v, r in evaluated]}


def fit(model, heights, counts, recorded_dose, *, times=None, sigma_bounds=(.5, 200.)):
    if model not in MODELS:
        raise ValueError("Unknown accepted-aperture response model")
    h = _vector(heights, "heights", positive=True)
    n = _vector(counts, "raw counts")
    dose = _vector(recorded_dose, "recorded dose", positive=True)
    if len(h) < 2 or len(h) != len(n) or len(h) != len(dose) or len(np.unique(h)) != len(h):
        raise ValueError("At least two distinct heights with matching counts and dose are required")
    if np.any(n < 0) or np.any(n != np.floor(n)) or np.any(n > 2**53):
        raise ValueError("Raw nonnegative integer counts are required")
    result = {"schema": VERSION, "model": model, "height_range": [float(h.min()), float(h.max())],
              "height_coordinate": "shared declared readback; physical units not inferred",
              "experimental_accuracy_certified": False, "pure_incident_beam_identified": False}
    reference_height = float(h.max())
    result["reference_height"] = reference_height
    if model == "interpolation":
        order = np.argsort(h)
        result.update(heights=h[order].tolist(), raw_counts=n[order].tolist(), recorded_dose=dose[order].tolist(),
                      rates=(n[order]/dose[order]).tolist(), status="calibrated_conditional_interpolation",
                      training_mean=n.tolist(), working_poisson_deviance=0.)
        return result
    if n.sum() == 0:
        result.update(status="zero_counts_shape_unidentified", amplitude=0., background=0.,
                      training_mean=n.tolist(), working_poisson_deviance=0.)
        return result
    search = None
    if model in ("uniform", "uniform_background"):
        shape = h/reference_height
        if model == "uniform":
            amplitude, mean, loss = _scale(n, dose, shape)
            background, success = 0., True
        else:
            mixed = _mix(n, dose, shape)
            amplitude = mixed["amplitude"]*(1-mixed["weight"])
            background = mixed["amplitude"]*mixed["weight"]
            mean, loss, success = mixed["mean"], mixed["loss"], mixed["optimizer_success"]
    elif model in ("gaussian", "gaussian_background"):
        bounds = np.asarray(sigma_bounds, float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 < bounds[0] < bounds[1]:
            raise ValueError("Finite increasing positive Gaussian width bounds are required")
        def evaluate(logsigma):
            shape = gaussian_acceptance(h, np.exp(logsigma))
            if model == "gaussian_background":
                return _mix(n, dose, shape)
            amplitude, mean, loss = _scale(n, dose, shape)
            return {"amplitude": amplitude, "mean": mean, "loss": loss, "weight": 0., "optimizer_success": True}
        logsigma, best, search = _profile_search(evaluate, tuple(np.log(bounds)))
        amplitude, background = best["amplitude"]*(1-best["weight"]), best["amplitude"]*best["weight"]
        mean, loss = best["mean"], best["loss"]
        success = best["optimizer_success"] and all(r["success"] for r in search["refinements"])
        result.update(sigma=float(np.exp(logsigma)), sigma_bounds=bounds.tolist(),
                      sigma_at_search_boundary=bool(np.isclose(logsigma, np.log(bounds)).any()))
    else:
        t = _vector(times, "recorded dose-weighted times")
        if len(t) != len(h) or np.ptp(t) <= 0:
            raise ValueError("Matching distinct finite acquisition times are required")
        origin, scale = float(t.mean()), float(np.ptp(t))
        coordinate = (t-origin)/scale
        base = np.ones_like(h) if model == "time_only" else h/reference_height
        def evaluate(beta):
            amplitude, mean, loss = _scale(n, dose, base*np.exp(beta*coordinate))
            return {"amplitude": amplitude, "mean": mean, "loss": loss}
        beta, best, search = _profile_search(evaluate, (-20., 20.))
        amplitude, background, mean, loss = best["amplitude"], 0., best["mean"], best["loss"]
        success = all(r["success"] for r in search["refinements"])
        result.update(beta=beta, time_origin=origin, time_scale=scale,
                      time_range=[float(t.min()), float(t.max())], beta_bounds=[-20., 20.])
    result.update(amplitude=float(amplitude), background=float(background), training_mean=mean.tolist(),
                  working_poisson_deviance=float(loss),
                  status="fitted_conditional_response" if success else "optimizer_failure_retained")
    if search is not None:
        result["profile_search"] = search
    return result


def predict(calibration, heights, recorded_dose, *, times=None):
    if calibration.get("schema") != VERSION or calibration.get("model") not in MODELS:
        raise ValueError("A matching conditional aperture calibration is required")
    h = _vector(heights, "target heights", positive=True)
    dose = _vector(recorded_dose, "target recorded dose", positive=True)
    if len(h) != len(dose) or np.any(h < calibration["height_range"][0]) or np.any(h > calibration["height_range"][1]):
        raise ValueError("Target heights and dose must match inside the calibrated range")
    if calibration.get("status") == "zero_counts_shape_unidentified":
        return np.zeros_like(h)
    model = calibration["model"]
    if model == "interpolation":
        return dose*(interpolation_weights(calibration["heights"], h)@np.asarray(calibration["rates"]))
    if model.startswith("gaussian"):
        shape = gaussian_acceptance(h, calibration["sigma"])
    elif model == "time_only":
        shape = np.ones_like(h)
    else:
        shape = h/calibration["reference_height"]
    if model in ("time_only", "uniform_time_drift"):
        t = _vector(times, "target dose-weighted times")
        if len(t) != len(h) or np.any(t < calibration["time_range"][0]) or np.any(t > calibration["time_range"][1]):
            raise ValueError("Target times must lie inside the calibration time range")
        shape *= np.exp(calibration["beta"]*(t-calibration["time_origin"])/calibration["time_scale"])
    return dose*(calibration["amplitude"]*shape+calibration["background"])


def interpolation_count_covariance(calibration, heights, recorded_dose):
    """Plug-in covariance from shared independent-Poisson training counts only.

    This excludes target counting variance and all source/model uncertainty.
    Zero-count plug-in variance is not a predictive upper bound.
    """
    if calibration.get("schema") != VERSION or calibration.get("model") != "interpolation":
        raise ValueError("A matching interpolation calibration is required")
    predict(calibration, heights, recorded_dose)
    weights = interpolation_weights(calibration["heights"], heights)
    dose = np.asarray(recorded_dose, float)
    rate_variance = np.asarray(calibration["raw_counts"])/np.asarray(calibration["recorded_dose"])**2
    return (weights*rate_variance)@weights.T*np.outer(dose, dose)
