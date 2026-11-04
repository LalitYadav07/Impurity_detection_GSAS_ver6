"""Public synthetic controls for links from screened starts to native fits."""
import json
from pathlib import Path

import numpy as np
import pytest

from gpx_recipe.symmetry.compiler import file_sha
from gpx_recipe.symmetry.contracts import digest
from gpx_recipe.symmetry.group_receipts import reference_paths, verify_equivalent


@pytest.fixture
def linked(tmp_path):
    owner='1'*24;native_id='2'*24;uid='3'*24;identity='a'*64;protocol='b'*64
    data=tmp_path/owner/'group-data';data.mkdir(parents=True)
    reference=tmp_path/native_id;reference.mkdir()
    directory=tmp_path/uid;output=directory/'results';output.mkdir(parents=True)
    job=dict(id=uid,candidate_id='mag:test',protocol_sha256=protocol,
        protocol={'policy':{'maximum_moment':3.}})
    common=dict(candidate_id='mag:test',protocol_sha256=protocol)
    (data.parent/'job.json').write_text(json.dumps(common))
    (reference/'job.json').write_text(json.dumps(common))
    np.savez_compressed(data/'kernel.npz',identity=identity,offset=np.array([-3.,-4.]),
        nuclear=np.ones(2),quadratic=np.array([[[1.]],[[2.]]]),background=np.empty((2,0)),
        maps=np.array([[[1.],[0.],[0.]]]),linear_error=0.)
    kernel=dict(owner=owner,identity=identity,sha256=file_sha(data/'kernel.npz'))
    (data/'kernel.json').write_text(json.dumps(dict(schema='radar-profiled-group-kernel/v1',
        candidate_id='mag:test',protocol_sha256=protocol,kernel=kernel,normalization=25.,
        labels=['Mn1'],component_maps=[[[1.],[0.],[0.]]])))
    kernel=dict(kernel,metadata_sha256=file_sha(data/'kernel.json'))
    fit=dict(coefficients=[1.],background=[],scale=1.)
    proof=dict(objective_identity={'cache_identity':identity},qualification={'passed':True},raw_solution=fit)
    (reference/'results').mkdir()
    proof_path=reference/'results/optimizer-receipt.json';proof_path.write_text(json.dumps(proof))
    native=dict(status='inadequate_fit',rwp=100*np.sqrt(2/25),audit={'passed':True},effects_verified=True,
        files={'results/optimizer-receipt.json':file_sha(proof_path)})
    (reference/'result.json').write_text(json.dumps(native))
    screen=dict(id=uid,native_verified=False,fit={**fit,'coefficients':[-1.]})
    (output/'surrogate-screening.json').write_text(json.dumps(screen))
    result=dict(status='profile_equivalent',effects_verified=False,
        native_reference={'id':native_id,'receipt_sha256':digest(native)},group_kernel=kernel,
        rwp=100*np.sqrt(2/25),moments=[dict(label='Mn1',moment=[-1.,0.,0.],magnitude=1.)])
    return directory,result,job,native,reference,data


def check(linked):
    directory,result,job,native,*_=linked
    return verify_equivalent(directory,result,job,lambda *_:native)


def test_time_reversed_screened_parameters_keep_native_reference_without_native_claim(linked):
    check(linked)
    directory,result,*_=linked
    assert not result['effects_verified']
    screen=json.loads((directory/'results/surrogate-screening.json').read_text())
    assert screen['fit']['coefficients']==[-1.]
    paths=reference_paths(directory,result)
    assert any(path=='kernel.npz' for _,path in paths)
    assert any(path=='results/optimizer-receipt.json' for _,path in paths)


def test_same_score_is_not_enough_if_the_complete_profile_differs(linked):
    directory,result,*_=linked
    path=directory/'results/surrogate-screening.json'
    data=json.loads(path.read_text());data['fit']['coefficients']=[1.2];path.write_text(json.dumps(data))
    result['moments'][0].update(moment=[1.2,0.,0.],magnitude=1.2)
    with pytest.raises(ValueError,match='profile'):
        check(linked)


def test_screened_display_rwp_uses_audited_native_mask_convention(linked):
    directory,result,job,native,reference,_=linked
    native['rwp']=7.5
    (reference/'result.json').write_text(json.dumps(native))
    result['native_reference']['receipt_sha256']=digest(native)
    result['rwp']=7.5
    check(linked)


@pytest.mark.parametrize('mutation',["candidate","native_status","receipt_hash","bounds","path","rwp","moments","metadata","missing_moments"])
def test_changed_reference_or_screen_is_rejected(linked,mutation):
    directory,result,job,native,reference,data=linked
    if mutation=='candidate':
        path=reference/'job.json';value=json.loads(path.read_text());value['candidate_id']='mag:other';path.write_text(json.dumps(value))
    elif mutation=='native_status':
        native['status']='technical_failure';(reference/'result.json').write_text(json.dumps(native))
    elif mutation=='receipt_hash':result['native_reference']['receipt_sha256']='0'*64
    elif mutation=='bounds':job['protocol']['policy']['maximum_moment']=.5
    elif mutation=='path':result['group_kernel']['owner']='../outside'
    elif mutation=='rwp':result['rwp']+=1
    elif mutation=='moments':result['moments'][0]['moment']=[2.,0.,0.]
    elif mutation=='missing_moments':result['moments']=[]
    elif mutation=='metadata':
        path=data/'kernel.json';value=json.loads(path.read_text());value['normalization']=26.;path.write_text(json.dumps(value))
    with pytest.raises(ValueError):check(linked)
