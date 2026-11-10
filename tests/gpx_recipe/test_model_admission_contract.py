"""Outer admission contract stays outside the pinned scientific package."""
import ast
from pathlib import Path

import pytest

from radar_pd_nova.model_admission import OPERATION, validate_payload


def payload():
    return dict(source='source.gpx',parent='parent.cif',instrument='profile.instprm',
        model='model.json',weights='weights.npz',children=['child.cif'],phase='Parent',
        histogram='PWDR source',k=['0','0','0'],grid=dict(kind='PNT',lower=3000.,upper=40000.,bins=701,bank=1),
        library_id='a'*24,destination='admitted')


@pytest.mark.parametrize('change',[
    lambda p:p.update(extra='untrusted'),lambda p:p.update(children=[]),
    lambda p:p.update(children=['child.cif','child.cif']),lambda p:p.update(k=['nan','0','0']),
    lambda p:p.update(k=['1/100','0','0']),lambda p:p['grid'].update(bins=True)])
def test_admission_rejects_open_or_invalid_contract(change):
    p=payload();change(p)
    with pytest.raises((ValueError,TypeError,ZeroDivisionError)):validate_payload(p)


def test_outer_contract_does_not_widen_shared_native_operations():
    from gpx_recipe.symmetry.native_protocol import validate_native_payload as shared
    validate_payload(payload())
    with pytest.raises(ValueError):shared(OPERATION,payload())


def test_packaged_adapter_is_the_module_used_by_outer_worker():
    # This runs with the built image's PYTHONPATH first; the checkout supplies
    # only the test, so a missing Docker COPY cannot be hidden by a source mount.
    import radar_pd_nova.model_admission as adapter
    assert Path(adapter.__file__).is_file()
    assert callable(adapter.execute) and adapter.OPERATION==OPERATION
    ast.parse(Path(adapter.__file__).read_text())
    if __import__('os').environ.get('RADAR_ADMISSION_PACKAGED')=='1':
        assert str(Path(adapter.__file__).resolve()).startswith('/opt/radar-symmetry/radar_pd_nova/')


@pytest.mark.parametrize('requested,waves,status',[
    (['0','0','0'],['0,0,0'],'eligible'),
    (['0','0','1/2'],['0,0,0'],'excluded'),
    (['0','0','1/2'],['0,0,0','0,0,1/2'],'eligible'),
    (['0','0','1/2'],['0,0,-1/2'],'eligible'),
    (['1/2','0','0'],['0,1/2,0'],'eligible'),
    (['0','0','0'],['0,0,1/2'],'excluded'),
    (['0','0','1/2'],['0,0,1/2','1/3,0,0'],'excluded'),
    (['0','0','1/2'],['1/3,0,0'],'excluded'),
])
def test_native_k_scope_classifies_each_candidate_with_explicit_reason(requested,waves,status):
    from radar_pd_nova.model_admission import _native_k
    operations=[dict(rotation=[[1,0,0],[0,1,0],[0,0,1]],translation=[0,0,0]),
                dict(rotation=[[0,-1,0],[1,0,0],[0,0,1]],translation=[0,0,0])]
    result=_native_k(['Parent['+wave+']mode' for wave in waves],requested,operations)
    assert result['eligibility']==status and result['eligibility_reason']
    assert len(result['mode_k_vectors'])==len(waves)


def test_missing_native_wavevector_is_integrity_failure_not_scope_exclusion():
    from radar_pd_nova.model_admission import _native_k
    with pytest.raises(ValueError,match='unambiguous'):
        _native_k(['arbitrary display name'],['0','0','0'],[])
