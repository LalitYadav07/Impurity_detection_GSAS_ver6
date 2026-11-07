"""Rendered response scope and portable trust boundaries; artificial arrays."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from gpx_recipe.preparation import authority, rendered_magnetic as module
from gpx_recipe.preparation.magnetic import MagneticResponse, load
from gpx_recipe.live.moment_basis import MomentBasis
from gpx_recipe.live.reflection_factors import ScatteringFactorCache
from gpx_recipe.symmetry.compiler import file_sha


def context():
    x=np.linspace(100.,200.,5)
    return dict(math=None,histogram=dict(hId=0,Data=[x,np.ones(5),np.ones(5)],
        **{'Reflection Lists':{'mag':{'RefList':np.array([[1.,0.,0.,2.,1.,9.]])}}}),
        phases={'mag':dict(pId=0,General=dict(Type='magnetic',Name='mag',Cell=[False,3.,4.,5.,90.,90.,90.,60.]),
            Histograms={'PWDR arbitrary':dict(Scale=[1.,False])})},histdict={'PWDR arbitrary':{'Type':'PNT'}},
        parameters={'0::AMx:0':1.,'0::AMy:0':0.,'0::AMz:0':0.,'0::Ax:0':.25,
            '0::AUiso:0':.01,'0::A0':1/9.,':0:Scale':2.,':0:Back;0':8.,':0:X':.1,'::constr0':1.},
        controls={':0:histType':'PNT',':0:Limits':[100.,200.],':0:wtFactor':1.,':0:Exclude':[]},
        basis=MomentBasis((('mag',0,('0::AMx:0','0::AMy:0','0::AMz:0')),),np.eye(3)[None],np.eye(3)[None]))


def key(monkeypatch,values):
    monkeypatch.setattr(module,'native_identity',lambda _:dict(reviewed_native='a'*64))
    return module.identity(**values)


@pytest.mark.parametrize('condition',['Temperature','Pressure','FreePrm1','FreePrm2','FreePrm3',
                                     'Temperature_model','0::Temperature'])
def test_recorded_conditions_do_not_hide_actual_fixed_forward_parameters(monkeypatch,condition):
    values=context();name=condition if condition.startswith('0::') else ':0:'+condition
    values['parameters'][name]=17.;before=key(monkeypatch,values)
    values['parameters'][name]=117.
    unchanged=condition in {'Temperature','Pressure','FreePrm1','FreePrm2','FreePrm3'}
    assert (key(monkeypatch,values)==before)==unchanged


@pytest.mark.parametrize('change',['observed','weights','excluded','scale','background','moments','constraint_helper','cycles'])
def test_only_observation_and_declared_external_nuisance_changes_preserve_response_identity(monkeypatch,change):
    values=context();before=key(monkeypatch,values)
    if change=='observed':values['histogram']['Data'][1]*=7.
    elif change=='weights':values['histogram']['Data'][2]*=100.
    elif change=='excluded':
        values['controls'][':0:Exclude']=[[120.,130.]]
        values['controls'][':0:wtFactor']=2.
        values['histogram']['Data'][0]=np.ma.array(values['histogram']['Data'][0],mask=[False,True,False,False,False])
    elif change=='scale':values['parameters'][':0:Scale']=4.
    elif change=='background':values['parameters'][':0:Back;0']=90.
    elif change=='moments':values['parameters']['0::AMx:0']=-2.
    elif change=='constraint_helper':values['parameters']['::constr0']=-2.
    else:values['controls']['max cyc']=99
    assert key(monkeypatch,values)==before


@pytest.mark.parametrize('change',['cell','coordinate','adp','profile','axis','range','reflection','basis','phase_scale'])
def test_every_rendered_physics_dependency_invalidates_the_identity(monkeypatch,change):
    values=context();before=key(monkeypatch,values)
    if change=='cell':values['parameters']['0::A0']*=1.00001
    elif change=='coordinate':values['parameters']['0::Ax:0']+=1e-8
    elif change=='adp':values['parameters']['0::AUiso:0']+=1e-8
    elif change=='profile':values['parameters'][':0:X']+=1e-8
    elif change=='axis':values['histogram']['Data'][0][1]+=1e-8
    elif change=='range':values['controls'][':0:Limits'][0]+=1e-8
    elif change=='reflection':values['histogram']['Reflection Lists']['mag']['RefList'][0,3]+=1.
    elif change=='basis':values['basis'].components[0,0,0]*=-1.
    else:values['phases']['mag']['Histograms']['PWDR arbitrary']['Scale'][0]=.5
    assert key(monkeypatch,values)!=before


def artifact(tmp_path):
    x=np.linspace(100.,200.,5);maps=np.eye(3)[None]
    response=MagneticResponse(x=x,fixed=np.ones(5),nuclear=np.ones(5)*2.,
        quadratic=np.repeat(np.eye(3)[None],5,axis=0),background=np.ones((5,1)),
        moment_maps=maps,physics_sha256='a'*64,
        proof=dict(compiler=dict(native_reflection_diagnostics=dict(rendered_identity='b'*64))))
    path=tmp_path/'response.npz';response.save(path)
    cache=ScatteringFactorCache();cache.attach_response_source(path,identity='b'*64,
        sha256=file_sha(path),physics_sha256='a'*64)
    return path,response,cache


def test_uploaded_proof_is_not_native_authority(tmp_path):
    _,response,cache=artifact(tmp_path)
    assert cache.rendered_response('b'*64,axis=response.x,moment_maps=response.moment_maps) is None
    assert 'no verified native producer' in cache.rendered_response_note


@pytest.mark.parametrize('change',['none','bytes','axis','basis','physics','identity'])
def test_exact_authorized_arrays_and_current_context_are_required(tmp_path,monkeypatch,change):
    path,response,cache=artifact(tmp_path)
    monkeypatch.setattr(authority,'magnetic_response_origin',lambda *args:dict(native_receipt='reviewed'))
    axis=response.x.copy();maps=response.moment_maps.copy()
    if change=='bytes':path.write_bytes(path.read_bytes()+b'changed')
    elif change=='axis':axis[1]+=1e-8
    elif change=='basis':maps[0,0,0]*=-1.
    elif change=='physics':
        cache.attach_response_source(path,identity='b'*64,sha256=file_sha(path),physics_sha256='c'*64)
    elif change=='identity':
        cache.attach_response_source(path,identity='c'*64,sha256=file_sha(path),physics_sha256='a'*64)
    result=cache.rendered_response('c'*64 if change=='identity' else 'b'*64,axis=axis,moment_maps=maps)
    if change=='none':
        q,proof=result;np.testing.assert_array_equal(q,response.quadratic)
        q[:]=0.;np.testing.assert_array_equal(load(path,physics_sha256='a'*64).quadratic,response.quadratic)
        assert proof['native_receipt']=='reviewed'
    else:assert result is None and cache.rendered_response_note


def test_decoded_bytes_must_match_the_certified_file_digest(tmp_path):
    path,_,_=artifact(tmp_path)
    with pytest.raises(ValueError,match='bytes differ'):
        load(path,physics_sha256='a'*64,expected_file_sha256='f'*64)


def test_rendered_response_authority_is_bound_to_both_file_and_physical_scope(tmp_path):
    from test_preparation_authority import certified,envelope
    *_,certificate,document=certified(tmp_path,kind='magnetic')
    path,_=envelope(tmp_path,certificate);entry=document['entries'][0]
    sha=entry['artifacts']['response.npz'];physics=entry['physics_sha256']
    assert authority.magnetic_response_origin(sha,physics) is None
    with authority.server_scope(path,tmp_path):
        assert authority.magnetic_response_origin(sha,physics)['entry_sha256']==entry['entry_sha256']
        assert authority.magnetic_response_origin('f'*64,physics) is None
        assert authority.magnetic_response_origin(sha,'f'*64) is None
    assert authority.magnetic_response_origin(sha,physics) is None
