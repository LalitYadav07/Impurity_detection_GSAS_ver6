"""Baseline diagnostics must not impose recipe thresholds or fill missing data."""
from copy import deepcopy
import hashlib
import json
import pickle
from unittest.mock import patch

import pytest

from scripts.ndip_convergence import native_convergence, convergence_warnings
from scripts.ndip_outputs import collect_outputs
from scripts import ndip_outputs


def records():
    return [[['Controls', {'max cyc': 3, 'deriv type': 'analytic Hessian'}]],
            [['Covariance', {'freshCOV': True, 'varyList': ['0::A0', ':0:Scale'],
                'sig': [2., 4.], 'Rvals': {'converged': True, 'Aborted': False,
                    'lastShifts': {'0::A0': 3.2, ':0:Scale': -1.}, 'Max shft/sig': 147.89}}]]]


def evidence(data):
    return native_convergence(data, source_gpx='final.gpx', source_sha256='a' * 64,
                              published_sha256='b' * 64)


def test_final_cycle_is_not_historical_and_has_no_implicit_one_sigma_gate():
    value = evidence(records())
    assert value['status'] == 'converged'
    assert value['last_cycle_max_shift_over_sigma'] == 1.6
    assert value['total_run_max_shift_over_sigma'] == 147.89
    assert value['last_cycle_ratios'] == {'0::A0': 1.6, ':0:Scale': -.25}
    assert convergence_warnings(value) == []


@pytest.mark.parametrize('field,value', [('converged', False), ('Aborted', True)])
def test_native_failure_is_review_without_discarding_diagnostics(field, value):
    data = records(); data[1][0][1]['Rvals'][field] = value
    out = evidence(data)
    assert out['status'] == 'needs_review'
    assert out['last_cycle_max_shift_over_sigma'] == 1.6
    assert convergence_warnings(out)[0]['code'] == 'native_convergence_review'


@pytest.mark.parametrize('edit', [
    lambda c: c['Rvals'].pop('lastShifts'),
    lambda c: c['Rvals']['lastShifts'].pop('0::A0'),
    lambda c: c.update(sig=[0., 4.]),
    lambda c: c.update(sig=[-1., 4.]),
    lambda c: c.update(sig=[float('nan'), 4.]),
    lambda c: c.update(sig=[2.]),
    lambda c: c.update(varyList=['0::A0', '0::A0']),
    lambda c: c['Rvals']['lastShifts'].update({'0::A0': float('inf')}),
    lambda c: c['Rvals']['lastShifts'].update({'0::A0': True}),
    lambda c: c['Rvals']['lastShifts'].update({'0::A0': '3.2'}),
    lambda c: c['Rvals'].update(lastShifts=[3.2]),
])
def test_incomplete_or_malformed_vectors_never_use_historical_fallback(edit):
    data = records(); edit(data[1][0][1]); out = evidence(data)
    assert out['last_cycle_complete'] is False
    assert out['last_cycle_max_shift_over_sigma'] is None
    assert out['total_run_max_shift_over_sigma'] == 147.89
    assert out['status'] == 'needs_review'


def test_list_shift_vector_is_matched_to_exact_native_variable_order():
    data = records(); data[1][0][1]['Rvals']['lastShifts'] = [3.2, -1.]
    assert evidence(data)['last_cycle_ratios'] == {'0::A0': 1.6, ':0:Scale': -.25}


def test_stale_and_ambiguous_covariance_cannot_claim_current_convergence():
    data = records(); data[1][0][1]['freshCOV'] = False
    assert evidence(data)['status'] == 'needs_review'
    assert evidence(records() + [deepcopy(records()[1])])['status'] == 'unknown'
    assert evidence([])['native_converged'] is None
    assert evidence([])['status'] == 'unknown'


def test_publication_binds_diagnostics_to_retained_not_rejected_project(tmp_path):
    root = tmp_path / 'run'; projects = root / 'Technical/GSAS_Projects'; projects.mkdir(parents=True)
    final = projects / 'seq_final_accepted_model.gpx'
    original = records()
    with final.open('wb') as stream:
        for row in original: pickle.dump(row, stream, protocol=1)
    rejected = deepcopy(original); rejected[1][0][1]['Rvals']['converged'] = False
    with (projects / 'seq_final_joint_polished.rejected.gpx').open('wb') as stream:
        for row in rejected: pickle.dump(row, stream, protocol=1)
    before = final.read_bytes(); portal = tmp_path / 'portal'
    with patch('scripts.ndip_lattice.extract_lattice', return_value=([], 2.0)):
        result = collect_outputs(root, portal, mode='full', include_archive=False)
    out = result['native_convergence']; project = result['gpx_projects'][0]
    assert out['source_gpx'] == final.relative_to(root).as_posix()
    assert out['source_sha256'] == project['sha256'] == hashlib.sha256(before).hexdigest()
    assert out['published_sha256'] == project['published_sha256']
    assert out['status'] == 'converged'
    assert out['last_cycle_max_shift_over_sigma'] == 1.6
    assert final.read_bytes() == before
    assert json.loads((portal / 'summary.json').read_text())['native_convergence'] == out
    report = (portal / 'report.html').read_text()
    assert 'Pipeline accepted' in report and 'Final-cycle max shift' in report
    assert 'uncertainties' in report and 'conditional' in report


def test_false_native_flag_becomes_scientific_review(tmp_path):
    root = tmp_path / 'run'; path = root / 'Technical/GSAS_Projects/seq_final_accepted_model.gpx'; path.parent.mkdir(parents=True)
    data = records(); data[1][0][1]['Rvals']['converged'] = False
    with path.open('wb') as stream:
        for row in data: pickle.dump(row, stream, protocol=1)
    with patch('scripts.ndip_lattice.extract_lattice', return_value=([], 2.0)):
        result = collect_outputs(root, tmp_path / 'portal', mode='full', include_archive=False)
    assert result['scientific_status'] == 'needs_review'
    assert result['final_rwp'] == 2.0
    assert result['native_convergence']['status'] == 'needs_review'
    assert any(row['code'] == 'native_convergence_review' for row in result['warnings'])


def test_rapid_without_final_native_checkpoint_is_unchanged(tmp_path):
    root = tmp_path / 'run'; root.mkdir()
    result = collect_outputs(root, tmp_path / 'portal', mode='rapid', include_archive=False)
    assert result['native_convergence'] == {}
    assert not any(row['code'].startswith('native_convergence') for row in result['warnings'])


def test_changed_published_file_cannot_claim_verified_native_convergence(tmp_path):
    root = tmp_path / 'run'; path = root / 'Technical/GSAS_Projects/seq_final_accepted_model.gpx'; path.parent.mkdir(parents=True)
    with path.open('wb') as stream:
        for row in records(): pickle.dump(row, stream, protocol=1)
    original_copy = ndip_outputs._copy_collections
    def change_after_publication(run, portal, **kwargs):
        result = original_copy(run, portal, **kwargs)
        (portal / result['gpx'][0]['path']).write_bytes(b'changed after publication')
        return result
    with patch('scripts.ndip_lattice.extract_lattice', return_value=([], 2.0)), \
            patch('scripts.ndip_outputs._copy_collections', side_effect=change_after_publication):
        result = collect_outputs(root, tmp_path / 'portal', mode='full', include_archive=False)
    assert result['native_convergence'] == {}
    assert any(row['code'] == 'native_convergence_unavailable' for row in result['warnings'])
