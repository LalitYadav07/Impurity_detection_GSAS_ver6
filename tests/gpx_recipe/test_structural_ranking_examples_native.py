"""Public, observation-free examples use the original native forward model."""
from pathlib import Path
import json
import numpy as np
import pytest

pytest.importorskip('GSASII.GSASIIscriptable')
from gpx_recipe.preparation.structural_examples import generate
from gpx_recipe.symmetry.compiler import file_sha


def test_public_example_block_has_actual_geometry_labels_and_no_observations(tmp_path):
    parent = tmp_path/'parent.cif'
    parent.write_text('''data_public_ersi
_symmetry_space_group_name_H-M 'C m c m'
_cell_length_a 4.195
_cell_length_b 10.353
_cell_length_c 3.779
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_U_iso_or_equiv
Si1 Si 0 .413 .25 1 .01
Er1 Er 0 .142 .25 1 .01
''')
    instrument = tmp_path/'bank.instprm'
    instrument.write_text('''#GSAS-II instrument parameter file
Type:PNT
Bank:1
difC:6200
difA:2
difB:.4
Zero:1
2-theta:105
fltPath:15.5
alpha:.99
beta-0:.02
beta-1:.03
beta-q:0
sig-0:1
sig-1:66
sig-2:0
sig-q:0
X:0
Y:0
Z:0
Azimuth:0
''')
    child = Path(__file__).with_name('data')/'synthetic-ersi-cmcm-isodistort.cif'
    before = [file_sha(p) for p in (parent, instrument, child)]
    request = dict(source=str(parent), instrument=str(instrument), child=str(child),
        grid=dict(kind='PNT', lower=3000., upper=40000., bins=701, bank=1),
        settings=dict(candidate=dict(id='public-ersi',
            transformation_matrix=[[-1., 0, 0], [0, -1., 0], [0, 0, 1.]], origin_shift=[0, 0, 0],
            child=dict(path='child.cif', sha256=file_sha(child))), mode_radius=.05, cell_radius=.01),
        destination=str(tmp_path/'examples'))
    from gpx_recipe.preparation.worker import execute
    dispatched = execute('structural-examples', dict(request,
        settings=dict(request['settings'], seed=1947, trajectories=3, steps=2)))
    assert dispatched['status'] == 'prepared'
    assert dispatched['artifacts'] == ['examples.json', 'examples.npz', 'entry.json']
    result = json.loads((tmp_path/'examples/examples.json').read_text())
    assert result['examples'] == 7
    assert result['declared_child_group']['number'] == 63
    assert result['observations_included'] is False
    assert result['scientific_result'] is False
    assert result['classifier_trained'] is False
    assert result['screening_qualified'] is False
    assert result['partition_assigned'] is False
    progress = json.loads((tmp_path/'examples/generation-progress.json').read_text())
    assert progress['state'] == 'complete'
    assert progress['processed'] == progress['denominator'] == 7
    assert result['schema'] == 'radar-structural-ranking-examples/v3'
    assert before == [file_sha(p) for p in (parent, instrument, child)]
    with np.load(tmp_path/'examples/examples.npz', allow_pickle=False) as arrays:
        assert arrays['theory_intensity'].shape == (7, result['bins'])
        assert np.isfinite(arrays['theory_intensity']).all()
        assert arrays['background'].shape == arrays['q'].shape
        assert np.isfinite(arrays['background']).all()
        assert np.all(np.diff(arrays['q']) > 0.)
        assert arrays['geometry_groups'][0].tolist() == [63, 63]
        assert arrays['trajectory'].tolist() == [-1, 0, 0, 1, 1, 2, 2]
        assert np.all(arrays['parameters'][0] == 0.)
        assert 'split' not in arrays.files
        for group in range(3):
            indices = np.flatnonzero(arrays['trajectory'] == group)
            assert len({result['groups'][i] for i in indices}) == 1
        assert result['groups'][1].startswith('parent:'+file_sha(parent)+':')
    assert result['renderer']['profile_interpolation'] is False
    assert result['renderer']['qualified_anchor'] is True
    assert result['renderer']['anchor_comparison']['array_identical'] is True
    # Shard expensive children across isolated workers without changing a
    # single physical point, whole-trajectory identity, label or native curve.
    pieces = []
    for i, interval in enumerate([(0, 1), (1, 3)]):
        request['destination'] = str(tmp_path/f'shard-{i}')
        detail = generate(request, seed=1947, trajectories=3, steps=2, trajectory_slice=interval)
        assert detail['trajectory_slice'] == list(interval)
        assert detail['generated_trajectories'] == interval[1]-interval[0]
        with np.load(tmp_path/f'shard-{i}/examples.npz', allow_pickle=False) as a:
            pieces.append({k: a[k].copy() for k in a.files})
    with np.load(tmp_path/'examples/examples.npz', allow_pickle=False) as original:
        for name in ('theory_intensity', 'parameters', 'trajectory', 'geometry_groups',
                     'ambiguous_geometry', 'displacement_rms', 'relative_metric_norm'):
            joined = np.concatenate([pieces[0][name], pieces[1][name][1:]])
            np.testing.assert_array_equal(joined, original[name])
        np.testing.assert_array_equal(pieces[0]['background'], original['background'])
        np.testing.assert_array_equal(pieces[1]['background'], original['background'])
