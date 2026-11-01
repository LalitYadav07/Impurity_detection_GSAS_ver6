import numpy as np
from gpx_recipe.artifacts import finite_scalar


def test_uncomputed_native_rwp_is_not_a_failed_refinement_or_zero():
    assert finite_scalar(None) is None
    assert finite_scalar('') is None
    assert finite_scalar(float('nan')) is None
    assert finite_scalar(True) is None
    assert finite_scalar(np.float64(4.5))==4.5
    assert finite_scalar(0.)==0.
