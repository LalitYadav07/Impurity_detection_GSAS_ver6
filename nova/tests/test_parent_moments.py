import copy
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from radar_pd_nova.parent_moments import parent_frame_moments
from radar_pd_nova.symmetry_workspace import parameter_rows


def direct_axes(cell):
    a, b, c, alpha, beta, gamma = cell
    ca, cb, cg = np.cos(np.deg2rad([alpha, beta, gamma]))
    sg = math.sin(math.radians(gamma))
    return np.array([[a, b*cg, c*cb], [0, b*sg, c*(ca-cb*cg)/sg],
                     [0, 0, c*math.sqrt(1-cb*cb-((ca-cb*cg)/sg)**2)]])


def fixture(parent_cell=None, transform=None, parent_moment=None):
    cell = parent_cell or [5., 5., 14., 90., 90., 120.]
    t = np.eye(3) if transform is None else np.asarray(transform, float)
    parent_moment = np.asarray([1., 1., 2.] if parent_moment is None else parent_moment, float)
    axes = direct_axes(cell)
    child_metric = t @ axes.T @ axes @ t.T
    child_lengths = np.sqrt(np.diag(child_metric))
    unit = child_metric / np.outer(child_lengths, child_lengths)
    child_cell = [*child_lengths, *np.rad2deg(np.arccos(np.clip([unit[1,2], unit[0,2], unit[0,1]], -1, 1)))]
    # Native passive axial-vector convention: parent unit-axis coefficients
    # become child unit-axis coefficients for a' = T a.
    child_moment = np.sign(np.linalg.det(t)) * child_lengths * (np.linalg.inv(t).T @ (parent_moment / cell[:3]))
    magnitude = float(np.linalg.norm(axes @ (parent_moment / cell[:3])))
    native = dict(label='Fe_m1', moment=child_moment.tolist(), magnitude=magnitude,
                  component_esd=dict(mx=.1, my=.2, mz=None), magnitude_esd=None)
    row = dict(kind='magnetic', parent='Parent', histogram='h', status='valid',
               moment_phase_names=dict(parent='Parent', magnetic='Magnetic'),
               embedding=dict(basis=t.tolist(), origin=[0,0,0]), refined_moments=[native])
    view = dict(schema='radar-symmetry-view/v1', histogram='h', phases=[
        dict(name='Parent', type='nuclear', cell=list(cell)),
        dict(name='Magnetic', type='magnetic', cell=child_cell,
             moments=[dict(label='Fe_m1', components=child_moment.tolist(), magnitude=magnitude)])])
    return row, view


def test_hexagonal_native_components_are_not_cartesian_or_independent_magnitudes():
    row, view = fixture(parent_moment=[1., 1., 0.])
    values, note = parent_frame_moments(row, view)
    assert not note
    assert values == [dict(label='Fe_m1', magnitude=1., parallel_c=0., transverse_c=1., angle_c=90.)]
    row, view = fixture(parent_moment=[1., -1., 0.])
    assert parent_frame_moments(row, view)[0][0]['magnitude'] == pytest.approx(math.sqrt(3), abs=1e-6)


def test_monoclinic_parent_c_uses_its_vector_instead_of_cartesian_z():
    row, view = fixture(parent_cell=[4., 5., 6., 90., 103., 90.], parent_moment=[0,0,3])
    values, note = parent_frame_moments(row, view)
    assert not note
    assert values == [dict(label='Fe_m1', magnitude=3., parallel_c=3., transverse_c=0., angle_c=0.)]
    assert not math.isclose(direct_axes(view['phases'][0]['cell'])[2,2] / 6., 1.)


def test_monoclinic_child_mx_and_mz_can_describe_a_parent_c_moment():
    t = [[1/3,-1/3,2/3], [1,1,0], [-1/3,1/3,1/3]]
    row, view = fixture(transform=t, parent_moment=[0,0,4])
    components = row['refined_moments'][0]['moment']
    assert abs(components[0]) > 1 and abs(components[2]) > 1
    values, note = parent_frame_moments(row, view)
    assert not note
    assert values[0]['parallel_c'] == 4 and values[0]['transverse_c'] == 0 and values[0]['angle_c'] == 0


@pytest.mark.parametrize('axis,index',[('a',0),('b',1),('c',2)])
def test_selected_parent_axis_is_a_geometric_projection(axis,index):
    cell=[4.,5.,6.,76.,103.,117.]
    moment=np.array([1.,-2.,3.]);axes=direct_axes(cell)
    row,view=fixture(cell,[[0,1,0],[0,0,-1],[-1,0,0]],moment)
    before=copy.deepcopy((row,view))
    values,note=parent_frame_moments(row,view,axis)
    assert not note and (row,view)==before
    cartesian=axes@(moment/cell[:3]);direction=axes[:,index]/cell[index]
    parallel=float(cartesian@direction)
    transverse=float(np.linalg.norm(cartesian-parallel*direction))
    assert values[0]['parallel_'+axis]==pytest.approx(parallel,abs=1e-6)
    assert values[0]['transverse_'+axis]==pytest.approx(transverse,abs=1e-6)
    assert values[0]['angle_'+axis]==pytest.approx(math.degrees(math.atan2(transverse,abs(parallel))),abs=1e-6)


def test_invalid_display_axis_does_not_guess():
    row,view=fixture()
    values,note=parent_frame_moments(row,view,'Q')
    assert not values and 'select parent axis a, b or c' in note


def test_120_forward_inverse_cases_preserve_magnitude_direction_and_time_reversal():
    cells = [[5.,5.,14.,90.,90.,120.], [4.,5.,6.,90.,103.,90.], [4.,5.,6.,76.,103.,117.]]
    transforms = [np.eye(3), np.diag([-1.,1.,1.]), np.diag([2.,1.,1.]),
                  [[0,1,0],[0,0,-1],[-1,0,0]], [[1/3,-1/3,2/3],[1,1,0],[-1/3,1/3,1/3]]]
    rng = np.random.default_rng(1842)
    count = 0
    for cell in cells:
        axes = direct_axes(cell); c_axis = axes[:,2] / cell[2]
        for transform in transforms:
            for moment in rng.normal(size=(8,3)):
                row, view = fixture(cell, transform, moment)
                before = copy.deepcopy((row, view))
                values, note = parent_frame_moments(row, view)
                assert not note and (row, view) == before
                actual = values[0]
                expected = axes @ (moment / cell[:3])
                parallel = float(expected @ c_axis)
                transverse = float(np.linalg.norm(expected-parallel*c_axis))
                angle = math.degrees(math.atan2(transverse, abs(parallel)))
                assert actual['magnitude'] == pytest.approx(np.linalg.norm(expected), abs=1e-6)
                assert actual['parallel_c'] == pytest.approx(parallel, abs=1e-6)
                assert actual['transverse_c'] == pytest.approx(transverse, abs=1e-6)
                assert actual['angle_c'] == pytest.approx(angle, abs=1e-6)
                reversed_row, reversed_view = fixture(cell, transform, -moment)
                reversed_value = parent_frame_moments(reversed_row, reversed_view)[0][0]
                assert reversed_value['parallel_c'] == -actual['parallel_c']
                for key in ['magnitude', 'transverse_c', 'angle_c']:
                    assert reversed_value[key] == actual[key]
                count += 1
    assert count == 120


@pytest.mark.parametrize('bad', ['parent_cell', 'child_cell', 'cell_angles', 'singular', 'nonfinite_basis',
                                 'nonfinite_moment', 'string_moment', 'zero', 'native_names', 'parent_name',
                                 'child_name', 'duplicate_child', 'histogram', 'duplicate_labels',
                                 'unmatched_labels', 'moment_mismatch', 'magnitude_mismatch'])
def test_invalid_or_unbound_direction_is_unavailable(bad):
    row, view = fixture()
    if bad == 'parent_cell':view['phases'][0].pop('cell')
    elif bad == 'child_cell':view['phases'][1].pop('cell')
    elif bad == 'cell_angles':view['phases'][1]['cell'][5] = 119.
    elif bad == 'singular':row['embedding']['basis'] = [[0,0,0]]*3
    elif bad == 'nonfinite_basis':row['embedding']['basis'][0][0] = float('inf')
    elif bad == 'nonfinite_moment':view['phases'][1]['moments'][0]['components'][0] = float('nan')
    elif bad == 'string_moment':view['phases'][1]['moments'][0]['components'][0] = '1'
    elif bad == 'zero':row, view = fixture(parent_moment=[0,0,0])
    elif bad == 'native_names':row.pop('moment_phase_names')
    elif bad == 'parent_name':row['moment_phase_names']['parent'] = 'Other'
    elif bad == 'child_name':row['moment_phase_names']['magnetic'] = 'Other'
    elif bad == 'duplicate_child':view['phases'].append(copy.deepcopy(view['phases'][1]))
    elif bad == 'histogram':view['histogram'] = 'Other'
    elif bad == 'duplicate_labels':view['phases'][1]['moments'] *= 2
    elif bad == 'unmatched_labels':row['refined_moments'][0]['label'] = 'Other'
    elif bad == 'moment_mismatch':row['refined_moments'][0]['moment'][0] += .1
    elif bad == 'magnitude_mismatch':row['refined_moments'][0]['magnitude'] += .1
    values, note = parent_frame_moments(row, view)
    assert values == [] and 'unavailable' in note.lower()


def test_origin_does_not_rotate_the_vector_and_native_uncertainties_stay_untouched():
    row, view = fixture()
    row['status'] = 'optimization_unresolved'
    row['native_convergence'] = dict(converged=False, Aborted=True, SVD0=0)
    before = copy.deepcopy((row, view))
    raw = parameter_rows(row, view)
    values, note = parent_frame_moments(row, view)
    assert values and not note
    assert (row, view) == before and parameter_rows(row, view) == raw
    assert all(value['uncertainty'] == '—' for value in raw)
    row['embedding']['origin'] = [0.5,0.25,-0.5]
    assert parent_frame_moments(row, view) == (values, note)
    assert not any('uncertainty' in key or 'esd' in key for key in values[0])


def test_nonmagnetic_view_has_no_parent_moment_panel():
    row, view = fixture(); row['kind'] = 'structural'
    assert parent_frame_moments(row, view) == ([], '')


def test_verified_native_phase_names_are_retained_for_exact_view_binding(tmp_path, monkeypatch):
    import radar_pd_nova.symmetry_workspace as module
    candidate = dict(id='c', magnetic_space_group='P1', irrep='', bns_number='1.1',
                     transformation_matrix=np.eye(3).tolist(), origin_shift=[0,0,0])
    protocol = dict(candidates=[candidate], kind='magnetic', source={'sha256':'a'*64},
                    parent_phase='Parent', histogram='h', k=['0','0','0'])
    (tmp_path/'job.json').write_text(json.dumps(dict(id='trial', protocol=protocol, protocol_sha256='p', candidate_id='c', seed_id='s')))
    (tmp_path/'results').mkdir()
    (tmp_path/'results/native.json').write_text(json.dumps(dict(nuclear_phase_name='Parent', magnetic_phase_name='Magnetic')))
    result = dict(status='optimization_unresolved', native_receipt='native.json', files={'results/native.json':'verified'})
    (tmp_path/'result.json').write_text(json.dumps(result))
    monkeypatch.setattr(module, 'checked_receipt', lambda *args: result)
    assert module.read_trial(tmp_path)['moment_phase_names'] == dict(parent='Parent', magnetic='Magnetic')


def test_selection_change_clears_parent_directions_and_keeps_the_raw_table(monkeypatch):
    import radar_pd_nova.symmetry_controller as controller
    row, view = fixture()
    row.update(id='trial', source='source', source_sha256='a'*64, metadata={}, name='P1', bns='1.1',
               rwp=1., has_gpx=True, has_view=True, seed='x', worker_revision='a'*64, k=['0','0','0'])
    state = SimpleNamespace(sx_selected='trial', sx_pins=[], sx_axis='d', sx_reset=0,
                            sx_intensity_scale='linear', rf_experiment='experiment', sx_error='')
    app = controller.SymmetryMixin(); app.server = SimpleNamespace(state=state)
    app._symmetry_rows = [row]; app._refinement_store = lambda: None
    monkeypatch.setattr(controller.plots, 'overview', lambda *_: {})
    monkeypatch.setattr(controller.plots, 'fit', lambda *a, **kw: dict(layout={}, data=[]))
    monkeypatch.setattr(controller, 'embedding_evidence', lambda *_: ([], ''))
    monkeypatch.setattr(controller, 'acquisition_notices', lambda *_: [])
    monkeypatch.setattr(controller, 'acquisition_details', lambda *_: [])
    monkeypatch.setattr(controller, 'acquisition_temperature', lambda *_: '')
    monkeypatch.setattr(controller, 'view_for', lambda *_: view)
    app._symmetry_plot_changed()
    assert state.sx_parent_moments and not state.sx_error
    assert state.sx_parameters == parameter_rows(row, view)
    view['phases'][0].pop('cell')
    app._symmetry_plot_changed()
    assert state.sx_parent_moments == [] and 'unavailable' in state.sx_parent_moment_note
    app._symmetry_rows = []
    app._symmetry_plot_changed()
    assert state.sx_parent_moments == [] and state.sx_parent_moment_note == ''


def test_unreadable_selected_view_cannot_retain_previous_parent_directions(monkeypatch):
    import radar_pd_nova.symmetry_controller as controller
    state=SimpleNamespace(sx_selected='unreadable',sx_pins=[],sx_axis='d',sx_reset=0,
        sx_intensity_scale='linear',rf_experiment='experiment',sx_error='',
        sx_parent_moments=[dict(label='previous trial',parallel_c=4.)],sx_parent_moment_note='previous note')
    row=dict(id='unreadable',name='C2/c',status='optimization_unresolved',rwp=10.84,has_view=True)
    app=controller.SymmetryMixin();app.server=SimpleNamespace(state=state);app._symmetry_rows=[row]
    monkeypatch.setattr(controller.plots,'overview',lambda *_:{})
    def unreadable(*args,**kwargs):raise OSError('Selected view is unreadable')
    monkeypatch.setattr(controller.plots,'fit',unreadable)
    app._symmetry_plot_changed()
    assert state.sx_error=='Selected view is unreadable'
    assert state.sx_parent_moments==[] and state.sx_parent_moment_note==''
