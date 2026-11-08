"""Paired features must use real native parent controls, not warped curves."""
from pathlib import Path

import numpy as np
import pytest

G = pytest.importorskip('GSASII.GSASIIscriptable')
from gpx_recipe.preparation.structural_pairs import generate
from gpx_recipe.symmetry.compiler import file_sha
from test_native_profile_capture import native_source


@pytest.mark.parametrize('kind', ['PNC', 'PNT'])
def test_native_pairs_preserve_physical_controls_and_whole_trajectory_shards(tmp_path, kind):
    source, _ = native_source(tmp_path/'input', kind)
    parent = tmp_path/'parent.cif'
    G.G2Project(gpxfile=str(source)).phase(0).export_CIF(str(parent))
    child = Path(__file__).with_name('data')/'synthetic-ersi-cmcm-isodistort.cif'
    profile = tmp_path/'input/profile.instprm'
    hashes = [file_sha(p) for p in (source, parent, profile, child)]
    request = dict(source=str(parent), instrument=str(profile), child=str(child),
        grid=dict(kind=kind, lower=12. if kind=='PNC' else 3000.,
                  upper=125. if kind=='PNC' else 40000., bins=701, bank=1),
        settings=dict(candidate=dict(id='public-ersi',
            transformation_matrix=[[-1., 0, 0], [0, -1., 0], [0, 0, 1.]], origin_shift=[0, 0, 0],
            child=dict(path='child.cif', sha256=file_sha(child))), mode_radius=.05, cell_radius=.01),
        destination=str(tmp_path/'pairs'))
    result = generate(request, seed=1947, parent_seed=8341, trajectories=3, steps=2)
    assert result['schema'] == 'radar-structural-ranking-paired-pilot/v1'
    assert result['examples'] == 10 and result['parent_native_calls'] == 4
    assert result['parent_controls_shape_qualified'], result['parent_control_comparisons']
    stages=result['timing']['stages_seconds'];rendering=result['timing']['child_rendering_seconds']
    assert all(np.isfinite(v) and v>=0 for v in [*stages.values(),*rendering.values()])
    assert sum(stages.values())<=result['seconds']
    assert sum(rendering.values())<=stages['rendering_children']
    assert stages['rendering_parent_references']>0 and rendering['forward_curves']>0
    assert all(abs(c['intensity_scale']-1) < 1e-6 for c in result['parent_control_comparisons'])
    assert not any(result[k] for k in ('observations_included', 'screening_qualified', 'classifier_trained', 'scientific_result'))
    with np.load(tmp_path/'pairs/pairs.npz', allow_pickle=False) as data:
        original = {k: data[k].copy() for k in data.files}
    assert original['role'].tolist() == [0, 1, 2, 2, 1, 2, 2, 1, 2, 2]
    assert original['trajectory'].tolist() == [-1, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    for index in range(3):
        rows = np.flatnonzero(original['trajectory']==index)
        for row in rows:
            np.testing.assert_array_equal(original['parent_a'][row], original['parent_a'][rows[0]])
            np.testing.assert_array_equal(original['parent_intensity'][row], original['parent_intensity'][rows[0]])
        assert len({result['groups'][i] for i in rows}) == 1
    pieces = []
    for i, interval in enumerate(((0, 1), (1, 3))):
        request['destination'] = str(tmp_path/f'shard-{i}')
        shard = generate(request, seed=1947, parent_seed=8341, trajectories=3, steps=2, trajectory_slice=interval)
        assert shard['parent_controls_shape_qualified']
        with np.load(tmp_path/f'shard-{i}/pairs.npz', allow_pickle=False) as data:
            pieces.append({k: data[k].copy() for k in data.files})
    for name in original.keys()-{'q', 'background'}:
        np.testing.assert_array_equal(np.concatenate([pieces[0][name], pieces[1][name][1:]]), original[name])
    for piece in pieces:
        np.testing.assert_array_equal(piece['q'], original['q'])
        np.testing.assert_array_equal(piece['background'], original['background'])
    assert hashes == [file_sha(p) for p in (source, parent, profile, child)]
