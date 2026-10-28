"""Report actual native parameter changes; independently detect fixed-value drift."""
import math
from numbers import Real


ORDINARY_GROUPS = {
    "background", "histogram_scale", "phase_fraction", "preferred_orientation",
    "lattice", "atomic_coordinates", "atomic_displacement", "occupancy",
    "instrument_zero", "instrument_profile", "sample_displacement",
    "hydrostatic_strain", "crystallite_size", "microstrain", "correction",
    "histogram_or_instrument", "rigid_body",
}
DERIVED_FAMILIES = {"Vol", "WgtFrac"}


def numeric(value):
    # Native solvers return NumPy real scalars, including refined HAP scales.
    # Reject flags and arrays, but never silently discard those scalar values.
    return isinstance(value,Real) and not isinstance(value,bool) and math.isfinite(value)


def parameter_changes(before, after):
    """Before is the configured preflight, so explicitly requested edits are recorded separately.

    Native configured flags include symmetry-dependent coordinates. Do not call
    a mathematically derived volume or weight fraction a freely refined variable.
    A zero change can be a legitimate already-converged refinement.
    """
    previous = {p["name"]: p for p in before["parameter_space"]["parameters"]}
    current = {p['name']:p for p in after['parameter_space']['parameters']}
    missing=[p['name'] for p in previous.values() if p.get('configured_now') and
             (not numeric(p.get('current_value')) or p['name'] not in current or not numeric(current[p['name']].get('current_value')))]
    missing_fixed=[p['name'] for p in previous.values() if not p.get('configured_now') and numeric(p.get('current_value'))
                   and p['group'] in ORDINARY_GROUPS and p.get('family') not in DERIVED_FAMILIES and p.get('status')!='derived'
                   and p.get('source_group')!='derived'
                   and (p['name'] not in current or not numeric(current[p['name']].get('current_value')))]
    released, fixed, derived, unexpected, historical = [], [], [], [], []
    # The read-only extractor retains last-cycle covariance-only values. An
    # explicitly replaced model (e.g. MD -> SH texture) can legitimately remove
    # such historical names; they are not parameters of the configured model.
    historical=[p['name'] for p in previous.values() if p.get('source_group')=='derived' and not p.get('configured_now')]
    for row in after["parameter_space"]["parameters"]:
        old = previous.get(row["name"])
        if old is None or not numeric(old.get("current_value")) or not numeric(row.get("current_value")):
            continue
        a, b = old["current_value"], row["current_value"]
        changed = not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)
        record = {"variable": row["name"], "group": row["group"], "before": a,
                  "after": b, "delta": b-a, "changed": changed, "sigma": row.get("sigma"),
                  "effective_independent": bool(row.get("effective_now")),
                  "native_status": row.get("status"), "location": row.get("location"),
                  "meaning": row.get("meaning"), "refinement_variable": row.get("refinement_variable"),
                  "family": row.get("family")}
        if old.get("configured_now"):
            released.append(record)
        elif row.get("family") in DERIVED_FAMILIES or row.get("status") == "derived":
            if changed: derived.append(record)
        elif old.get('source_group')=='derived':
            continue
        elif row["group"] in ORDINARY_GROUPS:
            fixed.append(record)
            if changed: unexpected.append(record)
    return {"released_parameters": released, "fixed_parameters_checked": len(fixed),
            "unobserved_released_parameters":missing,
            "unobserved_fixed_parameters":missing_fixed,
            "historical_only_parameters":historical,
            "released_parameters_changed": sum(p["changed"] for p in released),
            "derived_changes": derived, "unexpected_fixed_changes": unexpected,
            "comparison_tolerance": {"relative": 1e-10, "absolute": 1e-12}}


def require_fixed_values_unchanged(ledger):
    if ledger.get('unobserved_released_parameters'):
        raise ValueError('Could not audit every requested released parameter: '+str(ledger['unobserved_released_parameters']))
    if ledger.get('unobserved_fixed_parameters'):
        raise ValueError('Could not audit every numeric fixed parameter: '+str(ledger['unobserved_fixed_parameters']))
    if ledger["unexpected_fixed_changes"]:
        raise ValueError("Native refinement changed parameters held fixed by this stage: " +
                         str([p["variable"] for p in ledger["unexpected_fixed_changes"]]))


def require_native_covariance(covariance,variable_count,summary):
    import numpy as np
    matrix=np.asarray(covariance.get('covMatrix',[]),dtype=float)
    sigmas=np.asarray(covariance.get('sig',[]),dtype=float)
    if matrix.shape!=(variable_count,variable_count) or not np.all(np.isfinite(matrix)):
        raise ValueError('Fresh refinement covariance is missing, nonfinite or has the wrong dimensions')
    if sigmas.shape!=(variable_count,) or not np.all(np.isfinite(sigmas)) or np.any(sigmas<0):
        raise ValueError('Fresh refinement covariance has missing or invalid uncertainty values')
    diagonal=np.diag(matrix)
    if np.any(diagonal<=0) or not np.allclose(sigmas*sigmas,diagonal,rtol=1e-8,atol=1e-16):
        raise ValueError('Fresh refinement covariance variances are invalid or disagree with native independent uncertainties')
    correlation=matrix/np.sqrt(np.outer(diagonal,diagonal))
    if not np.allclose(correlation,correlation.T,rtol=0,atol=1e-8):
        raise ValueError('Fresh refinement covariance is not symmetric')
    if np.min(np.linalg.eigvalsh((correlation+correlation.T)/2)) < -1e-10:
        raise ValueError('Fresh refinement covariance is not positive semidefinite')
    if summary.get('rank') is not None and summary['rank']<variable_count:
        raise ValueError('Refinement covariance is rank deficient; the requested independent parameters are not all identifiable')


def constraint_bound_audit(before,covariance,reconstruct):
    """Check the solver's dependent values before native save-time clipping.

    Native GSAS-II can clip a negative dependent HAP scale to 1e-12 while
    leaving a sum-to-one equation and its covariance on the unconstrained fit.
    Reconstruct from the actual solver's mapping and independent values.
    """
    rows=before['parameter_space']['parameters']
    values={p['name']:p['current_value'] for p in rows if numeric(p.get('current_value'))}
    for name,value in zip(covariance.get('varyList',[]),covariance.get('variables',[])):values[str(name)]=float(value)
    reconstruct(values)
    checks=[];violations=[]
    for p in rows:
        if not p.get('configured_now') or p['group'] not in {'phase_fraction','histogram_scale','occupancy','atomic_displacement','preferred_orientation'}:continue
        if p['name'] not in values:continue
        lo,hi=p.get('lower_bound'),p.get('upper_bound')
        lo=lo.get('value') if isinstance(lo,dict) else lo;hi=hi.get('value') if isinstance(hi,dict) else hi
        if lo is None and hi is None:continue
        value=float(values[p['name']]);row={'variable':p['name'],'solver_value_before_native_clipping':value,'lower':lo,'upper':hi}
        checks.append(row)
        if not math.isfinite(value) or lo is not None and value<float(lo)-1e-9 or hi is not None and value>float(hi)+1e-9:violations.append(row)
    return {'checked':len(checks),'parameters':checks,'violations':violations}


def require_constraint_bounds(audit):
    if audit['violations']:
        raise ValueError('Native dependent parameters violated physical/requested bounds before save-time clipping: '+str([(r['variable'],r['solver_value_before_native_clipping']) for r in audit['violations']])+'. Later stages were not executed; no phases were silently removed or fixed.')


def blocking_frozen_variables(before,after):
    prior=set(before['parameter_space'].get('frozen_variables',[]))
    final=set(after['parameter_space'].get('frozen_variables',[]))
    requested=set(before['parameter_space'].get('configured_vary',[]))|set(before['parameter_space'].get('effective_vary',[]))
    return sorted((final-prior)|(final & requested))
