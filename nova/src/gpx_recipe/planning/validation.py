"""Frozen conditional repeatability models and measured-data diagnostics.

No target scale/background is fitted. A fixed measured reference checks noise
and repeatability; it is never described as arbitrary-CIF intensity validation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import StringIO
from pathlib import Path
import numpy as np
from scipy.stats import chi2


def read_xye(path, maximum_bytes=20 * 1024 * 1024):
    path = Path(path)
    before = path.stat()
    if not 0 < before.st_size <= maximum_bytes:
        raise ValueError("Reduced profile exceeds the declared bounded read")
    raw = path.read_bytes()
    numeric = []
    for line in raw.decode("utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith(("#", "'")):
            continue
        values = text.split()
        if len(values) != 3:
            raise ValueError("Expected three explicit XYE columns")
        try:
            numeric.append([float(v) for v in values])
        except ValueError as exc:
            raise ValueError("Unrecognized nonnumeric XYE content") from exc
    arrays = np.asarray(numeric, float)
    if arrays.ndim != 2 or arrays.shape[0] < 3 or arrays.shape[1] != 3:
        raise ValueError("At least three XYE bins required")
    if not np.isfinite(arrays).all() or np.any(np.diff(arrays[:, 0]) <= 0) or np.any(arrays[:, 2] < 0):
        raise ValueError("Invalid grid, intensity or uncertainty")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Reduced observation changed while being read")
    return arrays, {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                    "bins": len(arrays), "negative_y_bins": int((arrays[:, 1] < 0).sum()),
                    "zero_error_bins": int((arrays[:, 2] == 0).sum())}


def vectors(y, error, charge):
    y, error, charge = (np.asarray(a, float) for a in (y, error, charge))
    if y.ndim != 2 or error.shape != y.shape or charge.shape != (len(y),):
        raise ValueError("Matching runs-by-bins arrays and per-run exposures required")
    if not np.isfinite(y).all() or not np.isfinite(error).all() or not np.isfinite(charge).all():
        raise ValueError("Nonfinite measured values")
    if np.any(error < 0) or np.any(charge <= 0):
        raise ValueError("Errors must be nonnegative; charges strictly positive")
    return y, error, charge


@dataclass(frozen=True)
class RepeatReference:
    mean: np.ndarray
    variance_coefficient: np.ndarray
    reference_diagonal_variance: np.ndarray
    training_ids: tuple[str, ...]
    usable: np.ndarray


def fit_repeat_reference(y, error, charge, ids):
    y, error, charge = vectors(y, error, charge)
    if len(y) < 3 or len(ids) != len(y) or len(set(ids)) != len(ids):
        raise ValueError("At least three distinct training acquisitions required")
    weights = charge / charge.sum()
    mean = weights @ y
    a = np.mean(error**2 * charge[:, None], axis=0)
    usable = a > 0
    if usable.sum() < 3:
        raise ValueError("Training data do not establish nonzero uncertainty")
    return RepeatReference(mean, a, weights**2 @ error**2, tuple(ids), usable)


def score_repeats(reference, y, error, charge, ids, *, alpha=.05):
    y, error, charge = vectors(y, error, charge)
    if y.shape[1:] != reference.mean.shape or len(ids) != len(y) or len(set(ids)) != len(ids):
        raise ValueError("Distinct evaluation acquisitions on the exact reference grid required")
    if set(ids) & set(reference.training_ids):
        raise ValueError("Calibration/evaluation acquisition leakage")
    if len(y) < 3 or not 0 < alpha < 1:
        raise ValueError("At least three evaluation acquisitions and a valid error level required")
    use = reference.usable
    acquisition_variance = reference.variance_coefficient[use] / charge[:, None]
    # This marginal reference estimate does not identify shared normalization
    # covariance. Its finite uncertainty is retained, not called exact truth.
    predictive = acquisition_variance + reference.reference_diagonal_variance[use]
    standardized = (y[:, use] - reference.mean[use]) / np.sqrt(predictive)
    regions = {"all_bins": np.flatnonzero(use)}
    for i, indices in enumerate(np.array_split(np.flatnonzero(use), 4)):
        regions[f"range_quartile_{i + 1}"] = indices
    upper = np.quantile(reference.mean[use], .9)
    regions["strongest_reference_decile"] = np.flatnonzero(use & (reference.mean >= upper))
    region_results = {}
    for name, indices in regions.items():
        if not len(indices):
            continue
        values = y[:, indices].sum(axis=1)
        variance = reference.variance_coefficient[indices].sum() / charge
        weights = 1 / variance
        center = np.dot(weights, values) / weights.sum()
        statistic = float(np.sum((values - center)**2 / variance))
        df = len(values) - 1
        # The fitted diagnostic mean removes the fixed reference offset. It
        # is used only to test repeat variance, never to repair predictions.
        p_upper, p_lower = float(chi2.sf(statistic, df)), float(chi2.cdf(statistic, df))
        region_results[name] = {
            "bins": len(indices), "runs": len(values), "variance_ratio": statistic / df,
            "ratio_interval_under_independent_Gaussian_runs": [
                statistic / float(chi2.ppf(1 - alpha / (2 * len(regions)), df)),
                statistic / float(chi2.ppf(alpha / (2 * len(regions)), df))],
            "two_sided_p_under_independent_Gaussian_runs": min(1., 2 * min(p_upper, p_lower)),
            "diagnostic_mean_removed": True,
        }
    rejects = [name for name, result in region_results.items()
               if result["two_sided_p_under_independent_Gaussian_runs"] < alpha / len(region_results)]
    return {
        "schema": "powgen-planning-repeat-diagnostics/v1",
        "training_ids": list(reference.training_ids), "evaluation_ids": list(ids),
        "training_only_usable_bins": int(use.sum()), "omitted_zero_training_variance_bins": int((~use).sum()),
        "zero_exported_error_observations_retained": int((error[:, use] == 0).sum()),
        "negative_observations_retained": int((y[:, use] < 0).sum()),
        "charge_range_uah": [float(charge.min()), float(charge.max())],
        "marginal_standardized_residual_rms": float(np.sqrt(np.mean(standardized**2))),
        "marginal_95_percent_fraction": float(np.mean(abs(standardized) <= 1.959963984540054)),
        "marginal_over_3_sigma_fraction": float(np.mean(abs(standardized) > 3)),
        "marginal_statistics_are_descriptive_bins_are_not_independent_trials": True,
        "variance_coefficient_estimation_uncertainty_in_test": False,
        "regional_test_interpretation": "Conditional Gaussian diagnostic with training-estimated variances; p values and intervals are approximate. Rejection can reflect covariance, normalization drift, or specimen changes.",
        "regions": region_results, "rejected_independent_bin_variance_regions": rejects,
        "familywise_diagnostic_alpha": alpha,
        "target_scale_or_background_fit": False,
        "experimental_accuracy_certified": False,
        "scope": "Conditional repeatability with fixed measured mean; no absolute intensity, material transfer, or stopping-rule validation",
    }
