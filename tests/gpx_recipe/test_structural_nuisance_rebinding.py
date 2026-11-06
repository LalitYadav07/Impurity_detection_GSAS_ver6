"""Affine baseline changes never authorize geometry or unchecked predictions."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from gpx_recipe.preparation import structural_nuisance as module


def native_record(monkeypatch):
    histogram=dict(hId=0,Data=[np.zeros(5) for _ in range(3)])
    phase=dict(pId=0,General=dict(Type='nuclear'),Histograms={'hist':{}})
    native=SimpleNamespace(_check=lambda:None,histo_phases=[{'hist':histogram},{'phase':phase}],
        controls={':0:histType':'PNT'},parms={'0:0:Extinction':0.},parts=[('hist',slice(None))])
    source=SimpleNamespace(fast=SimpleNamespace(native=native,slow=SimpleNamespace(hist=SimpleNamespace(name='hist'),phase=SimpleNamespace(name='phase'))))
    record=dict(parameters={':bank:Scale':120.,'primary:bank:Scale':1.,':bank:Back;0':3.,':bank:Back;1':.2,'primary::A0':.1,':bank:X':0.},
        background=[['chebyschev-1',True,2,3.,.2],{'nDebye':0,'nPeaks':0}],sample={'Scale':[120.,True]},
        physical_modes=[1.,2.],instrument={'X':0.},phases=[{'cell':[4.,5.,6.],'histograms':{'bank':{'Scale':[1.,False]}}}])
    monkeypatch.setattr(module,'_physics_record',lambda source:copy.deepcopy(record))
    return source,record,histogram,phase,native


def test_shape_identity_changes_only_for_declared_affine_nuisance_values(monkeypatch):
    source,record,*_=native_record(monkeypatch)
    before=copy.deepcopy(record);first=module.affine_context(source)
    assert record==before
    record['parameters'][':bank:Scale']=300.;record['sample']['Scale'][0]=300.
    record['parameters'][':bank:Back;0']=-5.;record['background'][0][3]=-5.
    record['parameters'][':bank:Back;1']=.7;record['background'][0][4]=.7
    second=module.affine_context(source)
    assert first['shape_sha256']==second['shape_sha256'] and second['scale']==300.
    record['parameters']['primary:bank:Scale']=.5;record['phases'][0]['histograms']['bank']['Scale'][0]=.5
    assert module.affine_context(source)['shape_sha256']==first['shape_sha256']
    assert module.affine_context(source)['scale']==150.
    record['parameters'][':bank:X']=.1
    assert module.affine_context(source)['shape_sha256']!=first['shape_sha256']


@pytest.mark.parametrize('change',['phase_scale_flag','cell','modes','background_kind','background_count','sample_flag'])
def test_other_physical_or_definition_changes_remain_dependencies(monkeypatch,change):
    source,record,*_=native_record(monkeypatch);before=module.affine_context(source)
    if change=='phase_scale_flag':record['phases'][0]['histograms']['bank']['Scale'][1]=True
    elif change=='cell':record['parameters']['primary::A0']+=.01
    elif change=='modes':record['physical_modes'][0]+=.1
    elif change=='background_kind':record['background'][0][0]='cosine'
    elif change=='sample_flag':record['sample']['Scale'][1]=False
    else:record['background'][0][2]=1
    after=module.affine_context(source)
    assert after is None or after['shape_sha256']!=before['shape_sha256']


def test_scalar_histogram_diagnostics_do_not_disable_rebinding(monkeypatch):
    source,_,_,_,native=native_record(monkeypatch);before=module.affine_context(source)
    native.histo_phases[0].update(Nobs=1000,chisq=20.,sumwYo=4.)
    assert module.affine_context(source)==before


@pytest.mark.parametrize('change',['extinction','secondary','mask','exclusion','debye','peaks','floor','modulated'])
def test_unsupported_affine_context_keeps_the_native_path(monkeypatch,change):
    source,record,histogram,phase,native=native_record(monkeypatch)
    if change=='extinction':native.parms['0:0:Extinction']=.01
    elif change=='secondary':native.histo_phases[1]['second']=phase
    elif change=='mask':histogram['Data'][0]=np.ma.array(np.zeros(5),mask=[False,True,False,False,False])
    elif change=='exclusion':native.controls[':0:Exclude']=[[1,2]]
    elif change=='debye':record['background'][1]['nDebye']=1
    elif change=='peaks':record['background'][1]['nPeaks']=1
    elif change=='floor':record['parameters'][':bank:Scale']=1e-12
    else:phase['General']['Modulated']=True
    assert module.affine_context(source) is None


def test_affine_rebinding_preserves_signed_background_and_continuous_curves():
    old=np.array([3.,.2,-1.,.1]);new=np.array([-2.,.7,.5,-.3])
    signals=np.array([[1.,4.,2.,.5],[4.,2.,.2,3.]])
    curves=120*signals+old
    stored=dict(schema=module.SCHEMA,shape_sha256='a'*64,scale=120.)
    current=dict(stored,scale=300.)
    report=module.rebind(curves,old,stored,current,new)
    np.testing.assert_allclose(curves,300*signals+new,rtol=1e-14,atol=1e-14)
    assert report['maximum_relative_roundoff_bound']<1e-11 and report['observations_used'] is False


def test_cancellation_cannot_hide_an_unreliable_foreground():
    background=np.ones(4)*1e18;curves=np.array([background+1.])
    context=dict(schema=module.SCHEMA,shape_sha256='a'*64,scale=1.)
    with pytest.raises(ValueError,match='loses precision'):
        module.rebind(curves,background,context,context,np.zeros(4))


@pytest.mark.parametrize('value',[None,[],{},dict(schema=module.SCHEMA,shape_sha256='a'*64,scale=True),
    dict(schema=module.SCHEMA,shape_sha256='a'*64,scale=float('inf'))])
def test_affine_metadata_is_closed_and_typed(value):
    with pytest.raises(ValueError):module.validate_context(value)
