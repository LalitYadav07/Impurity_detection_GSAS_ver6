"""Observe native numerical termination without changing solver arguments/results.

GSAS-II's numerical/Jacobian branches discard MINPACK's ``ier`` on save.
This diagnostic is deliberately separate from Rvals.converged, final-cycle
shift evidence and scientific qualification. Use only in an owned GSAS-II
process: its refinement and constraint maps are already process-global.
"""
from contextlib import contextmanager
from numbers import Integral
import threading


SCHEMA = 'radar-native-solver-termination/v1'
_LOCK = threading.Lock()
_TERMINATIONS = {
    0: 'improper_input',
    1: 'relative_objective_tolerance',
    2: 'relative_parameter_tolerance',
    3: 'objective_and_parameter_tolerances',
    4: 'residual_jacobian_orthogonality',
    5: 'function_evaluation_limit',
    6: 'objective_tolerance_too_small',
    7: 'parameter_tolerance_too_small',
    8: 'orthogonality_tolerance_too_small',
}


def _integer(value):
    return int(value) if isinstance(value, Integral) and not isinstance(value, bool) else None


def _record(result):
    if not isinstance(result, tuple) or len(result) != 5 or not isinstance(result[2], dict):
        return dict(state='unavailable', reason='native_full_output_not_returned')
    code = _integer(result[4])
    return dict(state='observed', code=code,
                termination=_TERMINATIONS.get(code, 'unknown_native_code'),
                solver_success=code in (1, 2, 3, 4) if code in _TERMINATIONS else None,
                message=str(result[3])[:2000],
                function_evaluations=_integer(result[2].get('nfev')))


@contextmanager
def observe_leastsq(main):
    """Yield a receipt; delegate every call and return the identical result.

    Only calls to this GSAS-II module's errRefine in the owning thread count.
    The proxy replaces its local ``so`` reference, never scipy.optimize's
    global leastsq. Missing adapters/calls remain unavailable, not successful.
    """
    receipt = dict(schema=SCHEMA, solver='scipy.optimize.leastsq', state='unavailable',
                   reason='native_solver_not_observed', calls=[], final=None,
                   qualification_changed=False, final_cycle_shifts_supplied=False)
    original = getattr(main, 'so', None)
    target = getattr(getattr(main, 'G2stMth', None), 'errRefine', None)
    leastsq = getattr(original, 'leastsq', None)
    if target is None or not callable(leastsq):
        receipt['reason'] = 'native_adapter_unavailable'
        yield receipt
        return
    if not _LOCK.acquire(blocking=False):
        receipt['reason'] = 'native_observer_already_active'
        yield receipt
        return
    owner = threading.get_ident()

    class ObservedOptimize:
        def __getattr__(self, name):
            return getattr(original, name)

        def leastsq(self, function, *args, **kwargs):
            result = leastsq(function, *args, **kwargs)
            if function is target and threading.get_ident() == owner:
                record = _record(result)
                receipt['calls'].append(record)
                receipt['final'] = record
                receipt['state'] = record['state']
                receipt['reason'] = record.get('reason')
            return result

    try:
        main.so = ObservedOptimize()
        yield receipt
    finally:
        main.so = original
        _LOCK.release()
