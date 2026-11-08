"""Projection reuse must preserve weak signals, gaps and input ownership."""
import hashlib

import numpy as np
import pytest

from gpx_recipe.preparation.structural_features import ThreeCurveProjector


def test_reused_projection_matches_prechange_feature_bytes_across_grids_and_masks():
    # Full byte digest from the eb86a04 implementation, including gap changes,
    # signed signals, changing backgrounds and contrasts 1, 30 and 1000.
    rng = np.random.default_rng(34519); digest = hashlib.sha256()
    for n in (11, 71, 2001, 3818):
        q = np.cumsum(rng.uniform(.001, .005, n))
        edges = np.linspace(q[0]-.1, q[-1]+.1, 257)
        projector = ThreeCurveProjector(q, edges)
        for case in range(30):
            before = rng.normal(20, 2, n); after = rng.normal(20, 3, n)
            bg1 = rng.normal(10, 1, n); bg2 = rng.normal(10, 2, n)
            mask = rng.random(n) > .2
            if mask.sum() < 2: mask[:2] = True
            reference = np.ma.array(before, mask=~mask) if case % 2 else before
            current = np.ma.array(after, mask=~np.roll(mask, 1)) if case % 3 else after
            result = projector.features(current, reference, background=bg1, reference_background=bg2,
                valid=mask if case % 5 else None, contrast=(1., 30., 1000.)[case % 3])
            digest.update(result['features'].tobytes())
    assert digest.hexdigest() == '7bbd9548564094f2db4141faa06aa5275c98d540fac04351ce70881c1c078c17'


def test_grid_and_mask_reuse_cannot_freeze_measurements_or_modify_returned_arrays():
    q = np.linspace(.5, 6, 101); edges = np.linspace(1, 5, 65)
    original_q = q.copy(); original_edges = edges.copy()
    projector = ThreeCurveProjector(q, edges)
    q[:] = 0.; edges[:] = 0.  # The admitted plan owns the original grid.
    y = 1+np.exp(-((original_q-2)/.1)**2)
    first = projector.features(y, y)
    plan = projector._projection
    first['q_edges'][:] = 0.; first['coverage'][:] = 0.; first['features'][:] = 99.
    second = projector.features(y+.01, y)
    assert projector._projection is plan
    np.testing.assert_array_equal(second['q_edges'], original_edges)
    assert second['coverage'].max() == 1 and (second['features'][2] > 0).all()
    valid = np.ones(101, bool); valid[20:30] = False
    masked = projector.features(y, y, valid=valid)
    assert projector._projection is not plan and (masked['coverage'] < 1).any()
    valid[:] = True
    again = projector.features(y, y, valid=valid)
    assert projector._projection.mask.all() and again['coverage'].min() == 1
    with pytest.raises(ValueError): projector.features(y, y, valid=valid.astype(float))
    with pytest.raises(ValueError): projector.features(y, y, valid=np.zeros(101, bool))
    y[40] = np.nan
    with pytest.raises(ValueError, match='nonfinite'): projector.features(y, y)
