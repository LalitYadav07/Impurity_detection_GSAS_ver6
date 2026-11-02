"""Read baseline convergence without changing a native project or fit policy."""
from __future__ import annotations

import math
from numbers import Real


SCHEMA = "radar-baseline-convergence/v1"


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _vector(value):
    if value is None or isinstance(value, (str, bytes, dict)):
        return []
    try:
        return list(value)
    except TypeError:
        return []


def native_convergence(records, *, source_gpx, source_sha256, published_sha256):
    """Bind native flags and *final-cycle* shifts to the retained final GPX.

    The historical ``Max shft/sig`` is a different quantity. It is retained
    separately and is never used to fill missing final-cycle evidence. No
    shift/sigma threshold is imposed on the baseline pipeline here.
    """
    entries = {}
    for record in records:
        if record and record[0] and len(record[0]) >= 2:
            name, value = record[0][:2]
            if name in {"Covariance", "Controls"}:
                entries.setdefault(name, []).append(value)
    covs, controls = entries.get("Covariance", []), entries.get("Controls", [])
    cov = covs[0] if len(covs) == 1 and isinstance(covs[0], dict) else {}
    control = controls[0] if len(controls) == 1 and isinstance(controls[0], dict) else {}
    rvals = cov.get("Rvals") if isinstance(cov.get("Rvals"), dict) else {}
    names = [str(value) for value in _vector(cov.get("varyList"))]
    unique_names = bool(names) and len(names) == len(set(names)) and all(names)
    sigmas = _vector(cov.get("sig"))
    raw = rvals.get("lastShifts")
    if isinstance(raw, dict):
        shifts = {str(key): value for key, value in raw.items()}
        if len(shifts) != len(raw):
            shifts = {}
    else:
        values = _vector(raw)
        shifts = dict(zip(names, values)) if len(values) == len(names) else {}
    ratios = {}
    if unique_names and len(sigmas) == len(names):
        for name, sigma in zip(names, sigmas):
            shift = shifts.get(name)
            if _finite(shift) and _finite(sigma) and sigma > 0:
                ratio = float(shift) / float(sigma)
                if math.isfinite(ratio):
                    ratios[name] = ratio
    missing = [name for name in names if name not in ratios]
    complete = unique_names and not missing and len(ratios) == len(names)
    maximum = max((abs(value) for value in ratios.values()), default=None) if complete else None
    native = rvals.get("converged") if type(rvals.get("converged")) is bool else None
    aborted = rvals.get("Aborted") is True
    fresh = cov.get("freshCOV") is True
    status = ("needs_review" if aborted or native is False or (native is True and (not complete or not fresh))
              else "converged" if native is True and fresh and complete else "unknown")
    historical = rvals.get("Max shft/sig")
    cycles = control.get("max cyc")
    return {
        "schema": SCHEMA, "source_gpx": str(source_gpx),
        "source_sha256": source_sha256, "published_sha256": published_sha256,
        "status": status, "native_converged": native, "aborted": aborted,
        "covariance_fresh": fresh,
        "maximum_cycles": cycles if type(cycles) is int and cycles >= 0 else None,
        "method": str(control.get("deriv type") or ""),
        "independent_parameters": len(names),
        "last_cycle_complete": complete,
        "missing_last_cycle_parameters": missing,
        "last_cycle_ratios": ratios,
        "last_cycle_max_shift_over_sigma": maximum,
        "total_run_max_shift_over_sigma": float(historical) if _finite(historical) else None,
    }


def convergence_warnings(evidence):
    if evidence["status"] == "converged":
        return []
    if evidence["aborted"]:
        message = "The native refinement reported an abort. The retained candidate requires scientific review."
    elif evidence["native_converged"] is False:
        message = "The native refinement did not reach convergence. The retained candidate requires scientific review."
    else:
        message = "Native convergence is unverified for the retained candidate."
    if not evidence["covariance_fresh"]:
        message += " A current covariance is unavailable."
    if not evidence["last_cycle_complete"]:
        message += " Final-cycle shifts and positive sigmas do not cover every independent parameter."
    return [{"code": "native_convergence_review", "message": message}]
