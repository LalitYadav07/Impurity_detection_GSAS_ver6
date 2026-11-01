"""Checkpoint convergence evidence, separate from residual and model adequacy."""
import math
from numbers import Real

SCHEMA = 'radar-native-convergence/v1'


def finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def convergence_evidence(state):
    """Use final-cycle shifts only; a total-run displacement is not a substitute."""
    fit = state.get('fit_statistics') or {}
    covariance = state.get('covariance') or {}
    shifts = covariance.get('shift_over_sigma') or {}
    names = list((state.get('parameter_space') or {}).get('last_vary') or [])
    ratios = (shifts.get('last_cycle') or {}).get('ratios') or {}
    available = {name: float(ratios[name]) for name in names if finite(ratios.get(name))}
    missing = [name for name in names if name not in available]
    worst = max(available, key=lambda name: abs(available[name])) if available else None
    method = str((state.get('controls') or {}).get('deriv type') or '')
    change = fit.get('DelChi2')
    # GSAS-II supplies a -1 sentinel for methods that do not report this
    # Hessian diagnostic. Do not label that sentinel as objective worsening.
    change = float(change) if 'Hessian' in method and finite(change) else None
    tolerance = (state.get('controls') or {}).get('min dM/M')
    tolerance = float(tolerance) if 'Hessian' in method and finite(tolerance) and tolerance > 0 else None
    return dict(schema=SCHEMA, method=method,
        native_converged=fit.get('converged') if type(fit.get('converged')) is bool else None,
        aborted=fit.get('Aborted') is True, native_message=str(fit.get('msg') or '')[:2000],
        maximum_cycles=(state.get('controls') or {}).get('max cyc'),
        relative_objective_change=change, relative_objective_tolerance=tolerance,
        independent_parameters=len(names),
        last_cycle_complete=bool(names) and not missing,
        missing_last_cycle_parameters=missing,
        last_cycle_ratios=available,
        last_cycle_max_shift_over_sigma=abs(available[worst]) if worst else None,
        largest_shift_parameter=worst,
        total_run_max_shift_over_sigma=(shifts.get('total_run') or {}).get('reported_maximum'),
        covariance_rank=covariance.get('rank'),
        high_correlation_count=covariance.get('high_correlation_count_0_9'),
        top_correlations=covariance.get('top_correlations') or [])


def assess_convergence(evidence, maximum):
    """Apply a reviewed screening threshold without accepting the science."""
    if maximum is None:
        return dict(status='not_reviewed', message='No convergence screen was recorded in this review.', reasons=['screen_not_reviewed'])
    if not finite(maximum) or maximum < 0:
        raise ValueError('Declare a finite nonnegative final-cycle shift/sigma ceiling')
    if not isinstance(evidence, dict) or evidence.get('schema') != SCHEMA:
        return dict(status='unavailable', message='Final-cycle convergence evidence is unavailable for this saved result.', reasons=['evidence_unavailable'])
    issues, missing = [], []
    if evidence.get('aborted') is True:
        issues.append(('native_aborted', 'Native refinement reported an abort.'))
    if evidence.get('native_converged') is False:
        issues.append(('native_not_converged', 'Native convergence was not reached.'))
    elif evidence.get('native_converged') is not True:
        missing.append(('native_convergence_unavailable', 'Native convergence status is unavailable.'))
    value = evidence.get('last_cycle_max_shift_over_sigma')
    if evidence.get('last_cycle_complete') is not True or not finite(value) or value < 0:
        missing.append(('last_cycle_shifts_unavailable', 'Final-cycle shifts and sigmas do not cover every independent parameter.'))
    elif value > maximum:
        issues.append(('last_cycle_shift_exceeded', f'Final-cycle maximum shift/sigma {value:.5g} exceeds {maximum:g}.'))
    change = evidence.get('relative_objective_change')
    # HessianLSQ accepts a step with chi2_new <= chi2_old * (1 + ftol),
    # where ftol is Controls['min dM/M']. Use that recorded tolerance, not
    # a near-zero threshold that rejects already converged native projects.
    # Keep legacy evidence conservative when the native tolerance is absent.
    tolerance = evidence.get('relative_objective_tolerance')
    tolerance = tolerance if finite(tolerance) and tolerance > 0 else 1e-12
    if finite(change) and change < -tolerance:
        issues.append(('objective_increased', f'The native last-cycle relative objective increased by {-change:.5g}, beyond tolerance {tolerance:.5g}.'))
    reasons = issues + missing
    message = f'Final-cycle maximum shift/sigma {value:.5g} is within {maximum:g}; native convergence reported.' if not reasons else ''
    if not reasons and finite(change) and change < 0:
        message += f' Relative objective increase {-change:.5g} is within the recorded tolerance {tolerance:.5g}.'
    return dict(status='needs_review' if issues else 'unavailable' if missing else 'passed',
        message=' '.join(detail for _, detail in reasons) if reasons else message,
        reasons=[key for key, _ in reasons])


def require_convergence(evidence, policy):
    """An explicit recipe screen is a stop condition, including missing data."""
    assessment=assess_convergence(evidence,policy['max_shift_over_sigma'])
    if assessment['status']!='passed':
        raise ValueError('Stage failed its required convergence screen: '+assessment['message'])
    return assessment
