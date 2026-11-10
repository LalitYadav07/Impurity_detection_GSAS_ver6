"""Preparation transport admits only typed scientific data."""
import copy
from pathlib import Path

import pytest

from gpx_recipe.preparation.worker import validate_request
from gpx_recipe.symmetry.native_protocol import validate_native_payload
from gpx_recipe.symmetry.bundle import pack, unpack


def reference():
    return dict(source='parent.cif', instrument='bank.instprm',
        grid=dict(kind='PNT', lower=3000., upper=40000., bins=701, bank=1),
        settings=dict(maximum_denominator=8), destination='prepared')


@pytest.mark.parametrize('field', ['api_key', 'python', 'score', 'observed'])
def test_preparation_rejects_extra_secrets_code_and_measurement_fields(field):
    request = reference()
    request['settings'][field] = 'not allowed'
    with pytest.raises(ValueError):
        validate_native_payload('preparation-reference', request)


def test_request_validation_does_not_mutate_the_captured_payload():
    request = reference()
    original = copy.deepcopy(request)
    validate_request('reference', request)
    assert request == original
    with pytest.raises(ValueError):
        validate_native_payload('preparation-reference', {**request, 'child': 'unrelated.cif'})


def test_instrument_profiles_cross_the_same_data_only_packet(tmp_path):
    root = tmp_path/'inputs';root.mkdir()
    (root/'parent.cif').write_text('data_public\n')
    (root/'bank.instprm').write_text('#GSAS-II instrument parameter file\nType:PNT\n')
    pack(root, tmp_path/'packet.zip', ['parent.cif', 'bank.instprm'])
    unpack(tmp_path/'packet.zip', tmp_path/'restored')
    assert (tmp_path/'restored/bank.instprm').read_bytes() == (root/'bank.instprm').read_bytes()
    (root/'code.py').write_text('raise RuntimeError')
    with pytest.raises(ValueError, match='scientific data'):
        pack(root, tmp_path/'invalid.zip', ['code.py'])


@pytest.mark.parametrize('changed', [False, True])
def test_structural_child_content_is_bound_before_native_construction(tmp_path, monkeypatch, changed):
    from gpx_recipe.preparation import worker
    from gpx_recipe.symmetry.compiler import file_sha
    for name in ('parent.cif', 'bank.instprm', 'actual-child.cif'):
        (tmp_path/name).write_text('artificial input '+name)
    candidate=dict(id='struct:fixture', transformation_matrix=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
        origin_shift=[0.,0.,0.], child=dict(path='transport/declared-child.cif',
            sha256='a'*64 if changed else file_sha(tmp_path/'actual-child.cif')))
    request=reference()
    request.update(source=str(tmp_path/'parent.cif'), instrument=str(tmp_path/'bank.instprm'),
        child=str(tmp_path/'actual-child.cif'), destination=str(tmp_path/'prepared'),
        settings=dict(candidate=candidate, mode_radius=.05, cell_radius=.002))
    def native(*args, **kwargs):
        raise AssertionError('native construction reached')
    monkeypatch.setattr(worker, 'create_reference', native)
    if changed:
        with pytest.raises(ValueError, match='child differs from the declared candidate'):
            worker.execute('structural', request)
        assert not (tmp_path/'prepared').exists()
    else:
        # A relocated input with identical content is still the same child.
        with pytest.raises(AssertionError, match='native construction reached'):
            worker.execute('structural', request)
