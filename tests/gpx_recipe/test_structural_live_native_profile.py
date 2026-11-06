from types import SimpleNamespace

import numpy as np
import pytest

import gpx_recipe.live.native_profile as module
from gpx_recipe.live.native_peak_memo import NativePeakMemo


@pytest.mark.parametrize('array_storage',[False,True])
def test_zero_observation_forward_call_never_generates_dummy_counts(array_storage):
    native=module.NativeReducedProfile.__new__(module.NativeReducedProfile)
    native.peak_memo=NativePeakMemo()
    data=[np.arange(4.),np.zeros(4),np.ones(4),np.zeros(4),np.zeros(4),np.zeros(4)]
    if array_storage:data=np.array(data)
    histogram={'Data':data,'Reflection Lists':{}}
    native.histo_phases=[{'PWDR test':histogram},{},{},{}]
    native.parts=[('PWDR test',slice(None))];native.parms={};native.vary=[]
    native.maps=SimpleNamespace(Dict2Map=lambda p:None)
    native._constraint_signature=module.constraint_state_signature(native.maps)
    native._reflection_cell_keys=[];native._reflection_cell_signature=()
    native.refresh_cell_reflections=False;native.histdict={};native.controls={};native.pawley={}
    native.counts=np.zeros(4);native.factor=np.ones(4);native.numerical_indices=[]
    def err(values,*args):
        assert np.any(data[1]),'Native dummy-data branch must never be entered'
        data[3][:]=np.arange(4.)+2
        return np.zeros(4)
    native.math=SimpleNamespace(errRefine=err,dervRefine=lambda *args:np.zeros((0,4)))
    module._active=native
    first,jac=native.profile([],jacobian=True)
    np.testing.assert_array_equal(data[1],np.zeros(4))
    np.testing.assert_array_equal(data[2],np.ones(4))
    np.testing.assert_array_equal(first,np.arange(4.)+2)
    np.testing.assert_array_equal(native.profile([]),first)
    assert jac.shape==(0,4)


def test_observations_are_restored_even_when_native_profile_fails():
    native=module.NativeReducedProfile.__new__(module.NativeReducedProfile)
    native.peak_memo=NativePeakMemo()
    data=[np.arange(2.),np.zeros(2),np.ones(2),np.zeros(2),np.zeros(2),np.zeros(2)]
    native.histo_phases=[{'PWDR test':{'Data':data,'Reflection Lists':{}}},{},{},{}]
    native.parts=[('PWDR test',slice(None))];native.parms={};native.vary=[]
    native.maps=SimpleNamespace(Dict2Map=lambda p:None);native._reflection_cell_keys=[]
    native._constraint_signature=module.constraint_state_signature(native.maps)
    native._reflection_cell_signature=();native.refresh_cell_reflections=False
    native.histdict={};native.controls={};native.pawley={}
    def fail(*args):raise ValueError('native calculation failed')
    native.math=SimpleNamespace(errRefine=fail);module._active=native
    with pytest.raises(ValueError,match='native calculation'):native.profile([])
    np.testing.assert_array_equal(data[1],np.zeros(2))


@pytest.mark.parametrize('mutation',['reset','in_place_matrix','offset'])
def test_external_native_constraint_mutation_is_rejected(mutation):
    native=module.NativeReducedProfile.__new__(module.NativeReducedProfile)
    native.maps=SimpleNamespace(dependentParmList=[['a']],indParmList=[['b']],
        arrayList=[np.array([[1.]])],invarrayList=[np.array([[1.]])],offsetDict={},
        holdParmList=[],holdParmType={})
    native._constraint_signature=module.constraint_state_signature(native.maps)
    module._active=native;native._check()
    if mutation=='reset':native.maps.dependentParmList=[['a']]
    elif mutation=='in_place_matrix':native.maps.invarrayList[0][0,0]=2.
    else:native.maps.offsetDict['a']=.1
    with pytest.raises(RuntimeError,match='constraint maps changed'):native._check()
