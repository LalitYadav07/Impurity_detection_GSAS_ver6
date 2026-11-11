"""Bound native MINPACK residual calls in an owned GSAS-II worker process."""
from contextlib import contextmanager
import threading


NUMERICAL_METHODS = frozenset({'numeric', 'analytic Jacobian'})
DEFAULT_NUMERICAL_EVALUATIONS = 1000
_LOCK = threading.Lock()


class EvaluationBudgetExceeded(RuntimeError):
    pass


def stage_limits(context, *, hessian_cycles):
    """Use one visible numerical budget across deterministic recipe builders."""
    method = str((context.get('controls') or {}).get('deriv type') or '')
    if method in NUMERICAL_METHODS:
        return dict(cycles=1, max_function_evaluations=DEFAULT_NUMERICAL_EVALUATIONS)
    return dict(cycles=hessian_cycles)


@contextmanager
def residual_budget(main, maximum):
    """Leave unrelated calls untouched; count every native residual evaluation."""
    receipt = dict(schema='radar-native-evaluation-budget/v1', maximum=maximum,
                   evaluations=0, exhausted=False)
    if maximum is None:
        yield receipt
        return
    if type(maximum) is not int or not 1 <= maximum <= 100000:
        raise ValueError('Native evaluation budget must be an integer from 1 to 100000')
    original = getattr(main, 'so', None)
    target = getattr(getattr(main, 'G2stMth', None), 'errRefine', None)
    if target is None or not callable(getattr(original, 'leastsq', None)):
        raise ValueError('This native runtime cannot enforce the reviewed evaluation budget')
    if not _LOCK.acquire(blocking=False):
        raise ValueError('Another native evaluation budget is already active')
    owner = threading.get_ident()

    class BoundedOptimize:
        def __getattr__(self, name):
            return getattr(original, name)

        def leastsq(self, function, *args, **kwargs):
            if function is not target or threading.get_ident() != owner:
                return original.leastsq(function, *args, **kwargs)

            def counted(*values, **options):
                if receipt['evaluations'] >= maximum:
                    receipt['exhausted'] = True
                    raise EvaluationBudgetExceeded(
                        f'Native numerical refinement exhausted its reviewed budget of {maximum} residual evaluations')
                receipt['evaluations'] += 1
                return function(*values, **options)

            return original.leastsq(counted, *args, **kwargs)

    try:
        main.so = BoundedOptimize()
        yield receipt
    finally:
        main.so = original
        _LOCK.release()


def iteration_caption(stage, context):
    method = str((context.get('controls') or {}).get('deriv type') or '')
    if method in NUMERICAL_METHODS:
        limit = stage.get('max_function_evaluations')
        return (f'At most {limit:,} residual evaluations · {method} minimizer' if limit is not None
                else f'{method} minimizer · native tolerance stopping; no evaluation limit recorded')
    return f"At most {stage['cycles']} cycles"
