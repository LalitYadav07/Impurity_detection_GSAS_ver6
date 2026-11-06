"""Choose and independently check a native fixed-nuisance magnetic compiler.

Native amplitudes and peak profiles are reused before trying the derivative
compiler. Unsupported or numerically unqualified accelerators fall back to
the independently checked native profile path.
This changes computation only: it provides no model-selection or stopping rule.
"""
import time
import numpy as np
from .central_quadratic import compile_even_quadratic
from .native_quadratic import compile_native_magnetic_jacobian
from .native_reflection import compile_profile_reflections, ReflectionCompilerUnavailable


def compile_native_magnetic_kernel(objective, reference, indices, *,
                                   method='auto', seed=192, checks=5, factor_cache=None):
    if method not in {'auto', 'central', 'derivative', 'reflection'}:
        raise ValueError('Unknown magnetic compiler')
    reference = np.asarray(reference, float)
    indices = list(indices)
    if (reference.ndim != 1 or not np.isfinite(reference).all() or checks < 1 or
            not indices or len(set(indices)) != len(indices) or
            any(not isinstance(i, (int, np.integer)) or i < 0 or i >= len(reference) for i in indices)):
        raise ValueError('Finite reference, valid distinct indices and positive checks required')
    started = time.monotonic()
    selected = 'reflection' if method in {'auto', 'reflection'} else method
    fallback = None
    reflection_diagnostics = None

    def forward(p):
        x = reference.copy()
        x[indices] = p
        return np.asarray(objective.profile(x), float)

    def validate(base, q):
        rng = np.random.default_rng(seed + 1)
        total_errors, magnetic_errors = [], []
        for _ in range(checks):
            p = rng.normal(size=len(indices)) * 3.
            direct = forward(p)
            predicted = base + np.einsum('i,ijb,j->b', p, q, p, optimize=True)
            difference = float(np.max(abs(direct - predicted)))
            total_errors.append(difference / max(float(np.max(abs(direct))), 1e-12))
            magnetic_errors.append(difference / max(float(np.max(abs(direct - base))), 1e-12))
        if not np.isfinite(total_errors + magnetic_errors).all() or max(total_errors + magnetic_errors) > 1e-7:
            raise ValueError('Compiled kernel fails independent total/magnetic-signal validation')
        return dict(total_profile_errors=total_errors, magnetic_signal_errors=magnetic_errors)

    try:
        if selected == 'reflection':
            try:
                compiled = compile_profile_reflections(objective, reference, indices,
                    factor_cache=factor_cache)
                base, q = compiled['base'], compiled['quadratic']
                validation = validate(base, q)
                reflection_diagnostics = compiled['diagnostics']
                native_diagnostics = None
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                if method != 'auto':
                    raise
                fallback = f'{type(error).__name__}: {error}'
                selected = 'derivative' if len(indices) >= 6 else 'central'
        if selected == 'derivative':
            try:
                compiled = compile_native_magnetic_jacobian(objective, reference, indices, seed=seed, checks=checks)
                base, q = compiled['base'], compiled['quadratic']
                validation = validate(base, q)
                native_diagnostics = compiled['diagnostics']
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                if method != 'auto':
                    raise
                fallback = '; '.join(filter(None, [fallback, f'{type(error).__name__}: {error}']))
                selected = 'central'
        if selected == 'central':
            base = forward(np.zeros(len(indices)))
            q = compile_even_quadratic(forward, len(indices))
            validation = validate(base, q)
            native_diagnostics = None
    finally:
        objective.profile(reference)
    return dict(base=base, quadratic=q, diagnostics=dict(
        requested_method=method, selected_method=selected, dimensions=len(indices),
        fallback_reason=fallback, independent_validation=validation,
        native_derivative_diagnostics=native_diagnostics,
        native_reflection_diagnostics=reflection_diagnostics,
        seconds=time.monotonic()-started, global_optimum_certified=False))
