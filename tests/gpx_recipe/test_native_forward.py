import pytest
import numpy as np
pytest.importorskip('GSASII')
from test_native_profiled import artificial_project
from GSASII import GSASIIstrMain as main
from gpx_recipe.native_forward import forward_profile,tick_arrays


def test_forward_array_and_exact_reflection_annotations_match_native(tmp_path):
    project,path=artificial_project(tmp_path,sites=1)
    histogram=project.histograms()[0].name
    expected=np.asarray(main.DoNoFit(str(path),histogram))
    curve,ticks=forward_profile(path,histogram)
    np.testing.assert_allclose(curve,expected,rtol=1e-12,atol=1e-12)
    assert ticks and {t['kind'] for t in ticks}=={'nuclear','magnetic'}
    assert all(len(t['hkl'])==3 and t['d']>0 for t in ticks)
    arrays=tick_arrays(ticks)
    assert arrays['tick_hkl'].shape==(len(ticks),3)
    assert arrays['tick_phase'].dtype.kind=='U'
