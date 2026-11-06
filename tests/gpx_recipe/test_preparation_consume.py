"""Portable numeric assets remain unqualified until a native consumer checks."""
import json
import shutil

import numpy as np
import pytest

from gpx_recipe.live.reflection_factors import ScatteringFactorCache
from gpx_recipe.preparation.consume import restore_magnetic_entry
from gpx_recipe.preparation.magnetic import MagneticResponse
from gpx_recipe.preparation.operations import portable_artifacts
from gpx_recipe.symmetry.compiler import file_sha


def entry(root, *, factors=True):
    root.mkdir()
    response=MagneticResponse(x=np.arange(1.,5.),fixed=np.zeros(4),nuclear=np.ones(4),
        quadratic=np.ones((4,1,1)),background=np.empty((4,0)),
        moment_maps=np.array([[[1.],[0.],[0.]]]),physics_sha256='a'*64,proof={})
    response.save(root/'response.npz')
    detail=dict(physics_sha256=response.physics_sha256)
    artifacts={'response.npz':file_sha(root/'response.npz')}
    if factors:
        a=np.ones((2,6,1));cache=ScatteringFactorCache(root/'scratch')
        cache._admit_native('b'*64,a,dict(all_coefficient_directions_checked=True,dimensions=1,reflections=2))
        shutil.copyfile(root/'scratch'/('b'*64+'.npz'),root/'scattering.npz')
        artifacts['scattering.npz']=file_sha(root/'scattering.npz')
        detail['scattering']=dict(status='prepared',identity='b'*64,artifact='scattering.npz')
    manifest=dict(schema='radar-experiment-preparation-entry/v1',kind='magnetic',
        observations_included=False,scientific_result=False,detail=detail,artifacts=artifacts)
    (root/'entry.json').write_text(json.dumps(manifest))
    return manifest


@pytest.mark.parametrize('factors',[False,True])
def test_legacy_and_factorized_preparations_load_without_trusting_old_native_proofs(tmp_path,factors):
    root=tmp_path/'entry';entry(root,factors=factors)
    before={p.name:p.read_bytes() for p in root.iterdir() if p.is_file()}
    response,cache,receipt=restore_magnetic_entry(root)
    assert response._native_seal is None
    assert receipt['native_qualification_required'] and receipt['scattering_available']==factors
    if factors:
        block,qualified=cache.lookup('b'*64)
        assert block is not None and not qualified
    assert before=={p.name:p.read_bytes() for p in root.iterdir() if p.is_file()}


@pytest.mark.parametrize('mutation',['counts','science','path','extra','hash','identity','missing'])
def test_substituted_or_unsafe_portable_entries_are_not_consumed(tmp_path,mutation):
    root=tmp_path/'entry';manifest=entry(root)
    if mutation=='counts':manifest['observations_included']=True
    elif mutation=='science':manifest['scientific_result']=True
    elif mutation=='path':manifest['artifacts']['../outside.npz']='a'*64
    elif mutation=='extra':manifest['artifacts']['measurement.gpx']='a'*64
    elif mutation=='hash':manifest['artifacts']['scattering.npz']='a'*64
    elif mutation=='identity':manifest['detail']['scattering']['identity']='invalid'
    else:manifest['detail'].pop('scattering')
    (root/'entry.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):restore_magnetic_entry(root)


@pytest.mark.parametrize('kind,names',[('reference',[]),('magnetic',['response.npz']),
    ('magnetic',['response.npz','scattering.npz']),('structural',['starts.npz','starts.json'])])
def test_supported_portable_inventories_are_closed(kind,names):
    assert portable_artifacts(kind,names)==set(names)


@pytest.mark.parametrize('kind,names',[('reference',['scattering.npz']),('magnetic',[]),
    ('magnetic',['scattering.npz']),('magnetic',['response.npz','response.npz']),
    ('magnetic',['response.npz',None]),('structural',['starts.npz','starts.json','scattering.npz'])])
def test_incomplete_or_cross_kind_assets_are_not_accepted(kind,names):
    with pytest.raises(ValueError):portable_artifacts(kind,names)
