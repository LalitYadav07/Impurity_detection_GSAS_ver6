"""Share qualified native mode derivatives without replacing native scores.

This owned-process proxy keeps ``native`` as the original full native oracle.
Only the displacement Jacobian can use already-captured native responses.
An independent full native curve anchors every returned derivative. Native
constraint ownership, all physical coordinates and fallback checks remain.
"""
import numpy as np

from .captured_cell_derivative import CapturedCellDerivative


class NativeDerivativeSource:
    def __init__(self, source, evaluator=None):
        self.source = source
        self.evaluator = evaluator if evaluator is not None else CapturedCellDerivative(source, max_cells=2)
        if self.evaluator.source is not source:
            raise ValueError('Native derivative reuse belongs to another source')
        self.derivative_requests = 0
        self.native_derivative_fallbacks = 0
        self.maximum_relative_error = 0.
        self.response_at_admission = self.evaluator.diagnostics()

    def __getattr__(self, name):
        return getattr(self.source, name)

    def native(self, point):
        # Acceptance, nuisance checks, secants and final scoring remain raw
        # native calls. Do not compile unused matrices at cell probe points.
        return self.source.native(point)

    def native_mode_jacobian(self, point):
        point = self.source.check(point)
        self.derivative_requests += 1
        cache = self.evaluator
        if not cache.has_response(point) and not cache.needs_capture(point):
            self.native_derivative_fallbacks += 1
            return self.source.native_mode_jacobian(point)
        exact = cache.native_evaluate(point)
        if not cache.has_response(point):
            self.native_derivative_fallbacks += 1
            return self.source.native_mode_jacobian(point)
        reconstructed, jacobian = cache.mode_jacobian(point)
        exact = np.asarray(exact, float)
        reconstructed = np.asarray(reconstructed, float)
        jacobian = np.asarray(jacobian, float)
        if (exact.ndim != 1 or reconstructed.shape != exact.shape or jacobian.shape != (self.source.mode_dimensions, len(exact))
                or not np.isfinite(exact).all() or not np.isfinite(reconstructed).all()
                or not np.isfinite(jacobian).all()):
            raise ValueError('Native structural derivative response is incomplete')
        error = float(np.linalg.norm(reconstructed-exact)/max(np.linalg.norm(exact), 1e-30))
        self.maximum_relative_error = max(self.maximum_relative_error, error)
        if error > 1e-9:
            raise ValueError('Native structural derivative curve disagrees with its full native anchor')
        return exact, jacobian

    def diagnostics(self):
        return dict(derivative_requests=self.derivative_requests,
            native_derivative_fallbacks=self.native_derivative_fallbacks,
            maximum_relative_curve_error=self.maximum_relative_error,
            scores_use_original_native=True, all_mode_derivatives_retained=True,
            response_at_admission=self.response_at_admission,
            response_counts_scope='Owned evaluator lifetime, including any preceding main fit',
            response=self.evaluator.diagnostics())


def derivative_source(source, *, evaluator=None, force_native=False):
    """Use the same native engine for real sources; retain explicit oracles.

    Lightweight non-native test/analytical sources keep their own derivative
    contract. They are not silently labelled as accelerated GSAS-II models.
    """
    if force_native:
        return source
    if isinstance(source, NativeDerivativeSource):
        if evaluator is not None and evaluator is not source.evaluator:
            raise ValueError('Cannot replace an owned native derivative evaluator')
        return source
    if evaluator is None:
        from .full_mode_slice import FullModeSlice
        if not isinstance(source, FullModeSlice):
            return source
    return NativeDerivativeSource(source, evaluator)
