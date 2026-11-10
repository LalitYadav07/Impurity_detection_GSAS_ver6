"""Automatic peak reuse follows actual native work, not scientific labels."""
import numpy as np
import pytest

from gpx_recipe.live.native_profile import dense_tof_inventory,DENSE_TOF_REFLECTIONS


def inventory(count,kind='PNT',hid=0):
    histogram={'hId':hid,'Reflection Lists':{'phase':{'RefList':np.empty((count,12))}}}
    return {f'PWDR bank {hid}':histogram},{f':{hid}:histType':kind}


@pytest.mark.parametrize('count,selected',[(0,False),(1,False),
    (DENSE_TOF_REFLECTIONS-1,False),(DENSE_TOF_REFLECTIONS,True),
    (DENSE_TOF_REFLECTIONS+1,True)])
def test_native_inventory_threshold(count,selected):
    assert dense_tof_inventory(*inventory(count)) is selected


def test_complete_reflection_inventory_includes_every_phase_but_not_other_histograms():
    histograms,controls=inventory(DENSE_TOF_REFLECTIONS//2)
    histograms['PWDR bank 0']['Reflection Lists']['secondary']={'RefList':np.empty((DENSE_TOF_REFLECTIONS//2,12))}
    histograms['metadata']={'irrelevant':'not a powder loop'}
    assert dense_tof_inventory(histograms,controls)
    first,c1=inventory(DENSE_TOF_REFLECTIONS//2,hid=0)
    second,c2=inventory(DENSE_TOF_REFLECTIONS//2,hid=1)
    assert not dense_tof_inventory(dict(first,**second),dict(c1,**c2))


@pytest.mark.parametrize('kind',['PNC','PXC','PNT?',''])
def test_large_non_tof_or_mixed_workloads_retain_the_existing_policy(kind):
    histograms,controls=inventory(DENSE_TOF_REFLECTIONS)
    other,extra=inventory(1,kind=kind,hid=1)
    assert not dense_tof_inventory(dict(histograms,**other),dict(controls,**extra))


@pytest.mark.parametrize('change',['no_controls','no_hid','no_lists','bad_row','not_array','bad_rank','no_powder'])
def test_unresolved_native_inventory_does_not_request_memoization(change):
    histograms,controls=inventory(DENSE_TOF_REFLECTIONS)
    h=histograms['PWDR bank 0']
    if change=='no_controls':controls.clear()
    elif change=='no_hid':h.pop('hId')
    elif change=='no_lists':h.pop('Reflection Lists')
    elif change=='bad_row':h['Reflection Lists']['phase']=None
    elif change=='not_array':h['Reflection Lists']['phase']['RefList']=[1]*DENSE_TOF_REFLECTIONS
    elif change=='bad_rank':h['Reflection Lists']['phase']['RefList']=np.empty(DENSE_TOF_REFLECTIONS)
    else:histograms={'metadata':h}
    assert not dense_tof_inventory(histograms,controls)
