"""Acquisition lineage and fresh supplied-variance likelihoods, without GSAS."""
from dataclasses import asdict
import copy
import hashlib
import json

import numpy as np
import pytest

from gpx_recipe.execution import write_json
from gpx_recipe.structural_live.acquisition import StructuralAcquisition,SCHEMA
from gpx_recipe.structural_live.native import sha
from gpx_recipe.structural_live.service import SearchSettings
from gpx_recipe.symmetry.contracts import digest
from test_structural_reduced_service import saved_result


RAW=b'# X , Y , E Distribution=true\n0\n1,-10,2\n2,20,3\n3,30,4\n'


def fixture(tmp_path):
    prepared=tmp_path/'prepared';(prepared/'model').mkdir(parents=True)
    (prepared/'model'/'prepared.json').write_text('{}')
    (prepared/'prepared').mkdir();(prepared/'inputs').mkdir()
    (prepared/'prepared'/'source.gpx').write_bytes(b'public source-integrity fixture')
    (prepared/'inputs'/'source.dat').write_bytes(RAW)
    write_json(prepared/'transfer.json',dict(source_gpx='prepared/source.gpx',
        source_sha256=sha(prepared/'prepared'/'source.gpx'),source_dat='inputs/source.dat',
        source_dat_sha256=sha(prepared/'inputs'/'source.dat')))
    np.savez_compressed(prepared/'grid.npz',x=np.array([100.,200.,300.]),initial_invalid_mask=np.zeros(3,bool))
    scope=dict(run='2',sample_id='known-specimen',sample='Public artificial',wavelength='1.5')
    mapping=dict(mapping_kind='user_reviewed',source_gpx_sha256='a'*64,physical_bank=1,
        export_geometry=dict(kind='linear_tof_from_d',microseconds_per_angstrom=100.),
        normalization='Reviewed density normalization',producer_config_sha256='b'*64,
        identity=dict(facility='SNS',instrument='PG3',ipts=1,run=2,epoch='synthetic'))
    physical={k:v for k,v in mapping.items() if k!='mapping_kind'}
    contract=dict(schema=SCHEMA,prepared_sha256=sha(prepared/'model'/'prepared.json'),
        transfer_sha256=sha(prepared/'transfer.json'),grid_sha256=sha(prepared/'grid.npz'),
        mapping=physical,source_scope=scope,spectrum_marker=0,
        limits=[[100.,300.],[100.,300.],[200.,200.]],settings=asdict(SearchSettings(seconds=3.,starts=2)),
        acquisition=digest([physical,scope]),forecast_enabled=False,
        statistical_policy_qualified=False,kind='structural_point_fit')
    write_json(prepared/'acquisition.json',contract)
    calls=[]
    def runner(p,path,**kwargs):
        calls.append(kwargs)
        if kwargs['observation'].revision==3:raise RuntimeError('injected export failure')
        path.mkdir();return saved_result(path,[.1,.2])
    owner=StructuralAcquisition(prepared,tmp_path/'session',runner=runner)
    def receipt(raw=RAW,exposure=None):
        return dict(identity=mapping['identity'],content_sha256=hashlib.sha256(raw).hexdigest(),
                    metadata=dict(cumulative_exposure=exposure,sample_id=scope['sample_id'],physical_bank=1,
                                  producer_config_sha256='b'*64))
    return owner,scope,mapping,receipt,calls,runner


def test_signed_density_variances_and_exclusions_survive_replacement_and_restore(tmp_path):
    owner,scope,mapping,receipt,calls,runner=fixture(tmp_path)
    first=owner.update(RAW,receipt(),revision=1,current_scope=scope,current_mapping=mapping)
    assert first['kind']=='structural_point_fit' and not first['forecast_enabled']
    old=calls[-1]['observation']
    np.testing.assert_array_equal(old.x,[100.,200.,300.])
    np.testing.assert_allclose(old.y,[-.1,.2,.3])
    np.testing.assert_allclose(old.variance,[.0004,.0009,.0016])
    np.testing.assert_array_equal(old.mask,[True,False,True])
    np.testing.assert_array_equal(old.normalization_mask,[True,True,True])
    assert old.exposure is None
    assert owner.update(RAW,receipt(),revision=1,current_scope=scope,current_mapping=mapping)['new_fit'] is False
    assert len(calls)==1
    restored=StructuralAcquisition(owner.prepared,owner.root,runner=runner)
    updated=RAW.replace(b'1,-10,2',b'1,-12,1')
    restored.update(updated,receipt(updated),revision=2,current_scope=scope,current_mapping=mapping)
    assert len(calls)==2 and calls[-1]['warm_starts']==[[.1,.2]]
    fresh=calls[-1]['observation']
    np.testing.assert_allclose(fresh.y,[-.12,.2,.3])
    np.testing.assert_allclose(fresh.variance,[.0001,.0009,.0016])
    before=(restored.session.root/'latest.json').read_bytes()
    with pytest.raises(RuntimeError,match='export failure'):
        restored.update(updated,receipt(updated),revision=3,current_scope=scope,current_mapping=mapping)
    assert (restored.session.root/'latest.json').read_bytes()==before
    with pytest.raises(ValueError,match='older|Stale'):
        restored.update(RAW,receipt(),revision=1,current_scope=scope,current_mapping=mapping)


@pytest.mark.parametrize('change',['grid','spectrum','receipt','run','sample','bank','producer','geometry','normalization','captured_bank'])
def test_incompatible_snapshots_cannot_start_a_structural_fit(tmp_path,change):
    owner,scope,mapping,receipt,calls,_=fixture(tmp_path)
    raw=RAW;row=receipt();scope=copy.deepcopy(scope);mapping=copy.deepcopy(mapping)
    if change=='grid':raw=raw.replace(b'3,30,4',b'4,30,4')
    if change=='spectrum':raw=raw.replace(b'\n0\n',b'\n1\n')
    if change=='receipt':row['content_sha256']='c'*64
    else:row['content_sha256']=hashlib.sha256(raw).hexdigest()
    if change=='run':row['identity']=dict(row['identity'],run=3)
    if change=='sample':scope['sample_id']='other-specimen'
    if change=='bank':mapping['physical_bank']=2
    if change=='producer':mapping['producer_config_sha256']='c'*64
    if change=='geometry':mapping['export_geometry']['microseconds_per_angstrom']=101.
    if change=='normalization':mapping['normalization']='Changed normalization'
    if change=='captured_bank':row['metadata']['physical_bank']=2
    with pytest.raises(ValueError):
        owner.update(raw,row,revision=1,current_scope=scope,current_mapping=mapping)
    assert not calls and not (owner.session.root/'latest.json').exists()


def test_changed_same_revision_and_exposure_reset_leave_publication_intact(tmp_path):
    owner,scope,mapping,receipt,calls,_=fixture(tmp_path)
    owner.update(RAW,receipt(exposure=10.),revision=1,current_scope=scope,current_mapping=mapping)
    before=(owner.session.root/'latest.json').read_bytes()
    changed=RAW.replace(b'1,-10,2',b'1,-12,2')
    with pytest.raises(ValueError,match='without a new owner revision'):
        owner.update(changed,receipt(changed,10.),revision=1,current_scope=scope,current_mapping=mapping)
    with pytest.raises(ValueError,match='exposure'):
        owner.update(RAW,receipt(exposure=5.),revision=2,current_scope=scope,current_mapping=mapping)
    assert len(calls)==1 and (owner.session.root/'latest.json').read_bytes()==before


def test_newly_measured_bins_change_the_current_mask_without_repreparing_physics(tmp_path):
    owner,scope,mapping,receipt,calls,_=fixture(tmp_path)
    initial=RAW.replace(b'3,30,4',b'3,0,0')
    owner.update(initial,receipt(initial),revision=1,current_scope=scope,current_mapping=mapping)
    first=calls[-1]['observation']
    assert not first.mask[2] and not first.normalization_mask[2]
    owner.update(RAW,receipt(),revision=2,current_scope=scope,current_mapping=mapping)
    current=calls[-1]['observation']
    assert current.mask[2] and current.normalization_mask[2]
    assert current.protocol_sha256==first.protocol_sha256
    assert len(calls)==2


def test_restore_checks_the_raw_snapshot_against_the_published_observation(tmp_path):
    owner,scope,mapping,receipt,_,runner=fixture(tmp_path)
    owner.update(RAW,receipt(),revision=1,current_scope=scope,current_mapping=mapping)
    path=owner.root/'snapshots'/'revision-1'/'source.dat'
    path.write_bytes(RAW.replace(b'1,-10,2',b'1,-12,2'))
    with pytest.raises(ValueError,match='receipt'):
        StructuralAcquisition(owner.prepared,owner.root,runner=runner)


def test_restore_recomputes_observation_even_if_input_receipt_was_rehashed(tmp_path):
    owner,scope,mapping,receipt,_,runner=fixture(tmp_path)
    owner.update(RAW,receipt(),revision=1,current_scope=scope,current_mapping=mapping)
    path=owner.root/'snapshots'/'revision-1'
    changed=RAW.replace(b'1,-10,2',b'1,-12,2')
    (path/'source.dat').write_bytes(changed)
    document=json.loads((path/'binding.json').read_text());document['receipt']=receipt(changed)
    write_json(path/'binding.json',document)
    with pytest.raises(ValueError,match='published|Published'):
        StructuralAcquisition(owner.prepared,owner.root,runner=runner)


def test_one_acquisition_owner_must_serialize_native_updates(tmp_path):
    owner,scope,mapping,receipt,calls,_=fixture(tmp_path)
    owner.lock.acquire()
    try:
        with pytest.raises(RuntimeError,match='serialize'):
            owner.update(RAW,receipt(),revision=1,current_scope=scope,current_mapping=mapping)
    finally:owner.lock.release()
    assert not calls


def test_separate_owners_cannot_write_concurrently_or_overwrite_a_newer_revision(tmp_path):
    owner,scope,mapping,receipt,calls,runner=fixture(tmp_path)
    other=StructuralAcquisition(owner.prepared,owner.root,runner=runner)
    with owner._writer():
        with pytest.raises(RuntimeError,match='Another acquisition owner'):
            other.update(RAW,receipt(),revision=1,current_scope=scope,current_mapping=mapping)
    assert not calls
    owner.update(RAW,receipt(),revision=2,current_scope=scope,current_mapping=mapping)
    # The second object existed before revision 2; it must reload the durable
    # ledger rather than trust its older in-memory view of this acquisition.
    before=(owner.session.root/'latest.json').read_bytes()
    with pytest.raises(ValueError,match='Stale'):
        other.update(RAW,receipt(),revision=1,current_scope=scope,current_mapping=mapping)
    assert len(calls)==1 and (owner.session.root/'latest.json').read_bytes()==before
    assert other.update(RAW,receipt(),revision=2,current_scope=scope,current_mapping=mapping)['new_fit'] is False


def test_preparation_change_after_admission_requires_review_before_a_fit(tmp_path):
    owner,scope,mapping,receipt,calls,_=fixture(tmp_path)
    (owner.prepared/'transfer.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='changed'):
        owner.update(RAW,receipt(),revision=1,current_scope=scope,current_mapping=mapping)
    assert not calls
