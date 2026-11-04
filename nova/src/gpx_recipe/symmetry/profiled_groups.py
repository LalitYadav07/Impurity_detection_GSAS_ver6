"""Search one immutable, checked magnetic objective with a finite seed set.

This layer performs no GSAS-II calls and never labels a result native-verified.
All starts and magnetic parameter solutions survive profile grouping. A group
only says that calculated powder profiles agree numerically; it does not say
that the magnetic structures are equivalent.
"""
from __future__ import annotations

from dataclasses import asdict
import math
import time
from typing import Callable

import numpy as np

from ..snapshot import json_safe


SCHEMA = "radar-profiled-group-screen/v1"
PROFILE_RTOL = 1e-8
PROFILE_ATOL = 1e-10


def native_seed(template, seed, component_maps, labels):
    """Map the reviewed start into the one prepared native constraint basis.

    Uses the same site initializer as native magnetic construction, including
    parent-to-child directions, coupled components and independent orbit signs.
    No new randomized portfolio is introduced by grouped execution.
    """
    from .moment_starts import site_moment_start
    if template.get('schema') != 'radar-magnetic-seed-template/v1':
        raise ValueError('Unsupported prepared magnetic seed template')
    sites={row['label']:row for row in template['sites']}
    if len(sites)!=len(template['sites']) or set(sites)!=set(labels):
        raise ValueError('Prepared sites differ from the native moment basis')
    direction=None
    if seed.get('parent_direction') is not None:
        direction=np.asarray(template['parent_to_child'],float) @ np.asarray(seed['parent_direction'],float)
        metric=np.asarray(template['sites'][0]['metric'],float)
        magnitude=float(np.sqrt(direction @ metric @ direction))
        if not np.isfinite(magnitude) or magnitude<1e-10:
            raise ValueError('Invalid transformed start direction')
        direction=direction/magnitude
    values=[]
    for label in labels:
        row=sites[label]
        values.append(site_moment_start(row['ids'],row['coefficients'],row['basis_ids'],
            row['refined_axes'],magnitude=seed['magnitude'],axis=seed['axis'],
            direction=direction,site_sign=seed.get('site_signs',{}).get(row['parent_label'],1),
            symmetry_allowed=True,independent_seed=seed.get('independent_mode_seed'),
            orbit=row['orbit'],metric=row['metric']))
    matrix=np.asarray(component_maps,float).reshape(len(labels)*3,-1)
    target=np.asarray(values,float).ravel()
    coefficients=np.linalg.lstsq(matrix,target,rcond=1e-12)[0]
    if not np.allclose(matrix@coefficients,target,rtol=1e-9,atol=1e-10):
        raise ValueError('The reviewed seed cannot be represented by the prepared native constraints')
    return coefficients


def calculated_residual(model, fit):
    """The unchanged weighted residual, including the declared nuisances."""
    m = np.asarray(fit.coefficients, dtype=float)
    return (model.offset + fit.scale * (model.nuclear +
        np.einsum("i,nij,j->n", m, model.quadratic, m)) +
        model.background @ fit.background)


def same_profile(first, second, *, rtol=PROFILE_RTOL, atol=PROFILE_ATOL):
    """Compare all fitted bins, never Rwp alone or model/space-group names."""
    a, b = np.asarray(first), np.asarray(second)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return False
    # Both arrays are weighted residuals, so this also checks the observed
    # noise scale and cannot silently compare different binnings or masks.
    scale = max(1., float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    return bool(np.max(np.abs(a-b)) <= atol + rtol*scale)


def screen_group(model, starts, *, max_iterations, objective_ftol=1e-14,
                 on_progress: Callable | None = None, cancelled: Callable | None = None):
    """Evaluate every declared start against an already compiled objective.

    ``starts`` is a finite sequence of (trial ID, native independent moments).
    Returning early on cancellation retains all remaining IDs as pending.
    Numerical failure of one start does not prevent the others from running.
    The returned fits stay in memory for subsequent native verification; the
    serializable screen records contain no false native/covariance claims.
    """
    starts = [(str(uid), np.asarray(seed, dtype=float).copy()) for uid, seed in starts]
    ids = [uid for uid, _ in starts]
    if not starts or len(set(ids)) != len(ids) or any(not uid for uid in ids):
        raise ValueError("Declare a finite nonempty set of unique trial IDs")
    if any(seed.shape != (model.dimension,) or not np.isfinite(seed).all()
           for _, seed in starts):
        raise ValueError("Every seed must use this exact independent magnetic basis")
    if (isinstance(max_iterations, bool) or not isinstance(max_iterations, int)
            or max_iterations < 1):
        raise ValueError("Declare a positive optimizer iteration limit")
    began = time.monotonic()
    records, fits, profiles, groups = [], {}, {}, []
    for uid, seed in starts:
        if cancelled is not None and cancelled():
            break
        started = time.monotonic()
        try:
            fit = model.fit(seed, max_iterations=max_iterations,
                            objective_ftol=objective_ftol, callback=(
                                (lambda _: _check_cancel(cancelled)) if cancelled else None))
            residual = calculated_residual(model, fit)
            if not np.isfinite(residual).all() or not math.isfinite(fit.weighted_cost):
                raise ValueError("Surrogate returned a nonfinite calculated profile")
            recomputed = .5 * float(residual @ residual)
            if not math.isclose(recomputed, fit.weighted_cost, rel_tol=1e-9, abs_tol=1e-8):
                raise ValueError("Stored optimizer cost differs from its complete weighted profile")
            rwp = 100 * math.sqrt(2*recomputed/model.normalization)
            fits[uid], profiles[uid] = fit, residual
            qualified = bool(fit.success and not fit.bound_active)
            record = dict(id=uid, stage="surrogate_screen", state="finished",
                optimizer_qualified=qualified, native_verified=False,
                uncertainty_available=False, rwp=rwp,
                fit=json_safe(asdict(fit)), wall_seconds=time.monotonic()-started,
                initial_coefficients=seed.tolist(), profile_group=None)
            # Only qualified local solutions proceed automatically to native
            # verification. Failed starts remain explicit, with their vectors
            # and diagnostics; they never borrow a successful member's status.
            if qualified:
                group = next((g for g in groups
                    if same_profile(residual, profiles[g["representative"]])), None)
                if group is None:
                    group = dict(id=uid, representative=uid, members=[])
                    groups.append(group)
                group["members"].append(uid)
                record["profile_group"] = group["id"]
            records.append(record)
        except InterruptedError:
            break
        except Exception as exc:
            records.append(dict(id=uid, stage="surrogate_screen", state="failed",
                optimizer_qualified=False, native_verified=False,
                uncertainty_available=False, reason=str(exc),
                error_type=type(exc).__name__, wall_seconds=time.monotonic()-started,
                initial_coefficients=seed.tolist(), profile_group=None))
        if on_progress is not None:
            on_progress(dict(completed=len(records), total=len(starts),
                distinct_profiles=len(groups), last_id=uid))
    completed = {record["id"] for record in records}
    report = dict(schema=SCHEMA, declared_ids=ids, records=records, groups=groups,
        pending=[uid for uid in ids if uid not in completed],
        wall_seconds=time.monotonic()-began,
        profile_equivalence=dict(rtol=PROFILE_RTOL, atol=PROFILE_ATOL,
            meaning="Calculated-profile agreement only; magnetic parameter solutions remain distinct"),
        native_verified=False, covariance_supplied=False)
    return report, fits


def _check_cancel(cancelled):
    if cancelled():
        raise InterruptedError("Group screening cancelled; unstarted seeds remain pending")
