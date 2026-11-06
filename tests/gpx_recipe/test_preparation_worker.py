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
