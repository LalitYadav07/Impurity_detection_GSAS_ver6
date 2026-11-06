"""Measurement storage must survive physical rendering and its exceptions."""
from types import SimpleNamespace

import numpy as np
import pytest

from gpx_recipe.preparation.native_magnetic import _unmasked_physical_context


@pytest.mark.parametrize('stacked', [False, True])
@pytest.mark.parametrize('fails', [False, True])
def test_unmasked_rendering_owns_a_separate_container_and_restores_all_source_storage(stacked, fails):
    source = np.ma.array(np.arange(60.).reshape(6, 10), mask=False)
    source.mask[:3, 3] = True
    if not stacked:
        source = [row.copy() for row in source]
    values = np.array([np.ma.getdata(row).copy() for row in source])
    masks = np.array([np.ma.getmaskarray(row).copy() for row in source])
    exclusions = [[2., 4.]]
    histogram = dict(hId=0, Data=source)
    native = SimpleNamespace(parts=[('PWDR test', slice(None))],
        histo_phases=[{'PWDR test': histogram}], controls={':0:Exclude': exclusions},
        _check=lambda: None)
    def calculation():
        with _unmasked_physical_context(native):
            assert histogram['Data'] is not source
            assert not any(np.ma.getmaskarray(a).any() for a in histogram['Data'][:3])
            assert ':0:Exclude' not in native.controls
            histogram['Data'][0][:] = -9.
            histogram['Data'][2][:] = 1.
            if fails:
                raise RuntimeError('native failure')
    if fails:
        with pytest.raises(RuntimeError, match='native failure'):
            calculation()
    else:
        calculation()
    assert histogram['Data'] is source
    assert native.controls[':0:Exclude'] is exclusions
    np.testing.assert_array_equal([np.ma.getdata(row) for row in source], values)
    np.testing.assert_array_equal([np.ma.getmaskarray(row) for row in source], masks)
