"""Accelerator limits must not reject a complete native scientific model."""
from types import SimpleNamespace

import numpy as np

from gpx_recipe.structural_live.optimization.captured_cell_derivative import CapturedCellDerivative


def test_large_complete_model_keeps_native_curve_and_every_mode_derivative():
    dimensions = 257
    point = np.linspace(-.1, .1, dimensions)
    expected = np.array([2., 3., 5., 7.])
    jacobian = np.arange((dimensions - 1) * len(expected), dtype=float).reshape(
        dimensions - 1, len(expected))
    calls = []

    def native(value):
        np.testing.assert_array_equal(value, point)
        calls.append('curve')
        return expected.copy()

    def derivative(value):
        np.testing.assert_array_equal(value, point)
        calls.append('derivative')
        return expected.copy(), jacobian.copy()

    owner = SimpleNamespace(_check=lambda: None, math=SimpleNamespace(
        StructureFactor2=object(), StructureFactorDerv2=object()))
    source = SimpleNamespace(radii=np.ones(dimensions), mode_dimensions=dimensions - 1,
        cell_dimensions=1, fast=SimpleNamespace(native=owner), check=np.asarray,
        native=native, native_mode_jacobian=derivative)
    cache = CapturedCellDerivative(source)

    np.testing.assert_array_equal(cache.native_evaluate(point), expected)
    np.testing.assert_array_equal(cache.evaluate(point), expected)
    actual, gradients = cache.mode_jacobian(point)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(gradients, jacobian)
    assert not cache.has_response(point) and not cache.needs_capture(point)
    assert calls == ['curve', 'curve', 'derivative']
    assert not cache.cache and cache.capture_attempts == 0
    assert 'response_cache_dimension_limit' in cache.diagnostics()['capture_fallbacks']
