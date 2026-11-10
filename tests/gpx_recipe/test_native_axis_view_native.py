"""Compare the unchanged native calculation, including real masks/cell moves."""
import numpy as np
import pytest

pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_profile_capture import native_source


@pytest.mark.parametrize('kind',['PNC','PNT'])
def test_axis_view_keeps_native_profiles_and_derivatives_identical(tmp_path,kind):
    _,source=native_source(tmp_path/'inputs',kind)
    native=source.fast.native;driver=native.math.getPowderProfile
    points=[np.zeros(len(source.radii)),np.linspace(-.025,.035,len(source.radii))]
    native.axis_view.enabled=False
    expected=[source.native_mode_jacobian(p*source.radii) for p in points]
    native.axis_view.enabled=True
    for p,(profile,jacobian) in zip(points,expected):
        actual,derivative=source.native_mode_jacobian(p*source.radii)
        np.testing.assert_array_equal(actual,profile)
        np.testing.assert_array_equal(derivative,jacobian)
        assert native.math.getPowderProfile is driver
    assert native.axis_view.converted_calls>=len(points)


def test_active_native_capture_retains_its_original_driver_contract(tmp_path):
    from gpx_recipe.structural_live.optimization.captured_cell_derivative import CapturedCellDerivative
    _,source=native_source(tmp_path/'inputs','PNT')
    point=np.zeros(len(source.radii));cache=CapturedCellDerivative(source)
    actual=cache.native_evaluate(point)
    np.testing.assert_array_equal(actual,source.native(point))
    assert cache.diagnostics()['captured_native_calls']==1
    assert source.fast.native.axis_view.bypasses['unreviewed native driver or active capture hook']>=1
