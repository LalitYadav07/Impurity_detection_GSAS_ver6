"""The reusable shape specification excludes unrelated scientific state."""
import copy
import pytest

from gpx_recipe.structural_live.optimization.tof_shape import validate,SCHEMA
from gpx_recipe.structural_live.optimization.tof_profile_table import _bounds


def specification():
    return dict(schema=SCHEMA,
        bank={'difC':6200.,'alpha':.99,'beta-0':.02,'beta-1':.03,'beta-q':0.,
              'sig-0':1.,'sig-1':66.,'sig-2':0.,'sig-q':0.,'X':0.,'Y':0.,'Z':0.},
        sample={'Size;i':1.,'Size;mx':1.,'Mustrain;i':1000.,'Mustrain;mx':1.})


def test_bounded_shape_specification_retains_only_native_shape_inputs():
    value=specification();assert validate(value)==value
    assert set(value)=={'schema','bank','sample'}


@pytest.mark.parametrize('change',['extra','missing','bank_extra','sample_extra','bool',
    'nan','inf','size','strain','mix_low','mix_high','difc','wrong_schema','not_mapping'])
def test_invalid_shape_specification_cannot_request_native_calculation(change):
    value=copy.deepcopy(specification())
    if change=='extra':value['observations']=[1,2,3]
    elif change=='missing':value.pop('sample')
    elif change=='bank_extra':value['bank']['Zero']=1.
    elif change=='sample_extra':value['sample']['lattice']=[1,1,1]
    elif change=='bool':value['bank']['X']=True
    elif change=='nan':value['bank']['X']=float('nan')
    elif change=='inf':value['bank']['X']=float('inf')
    elif change=='size':value['sample']['Size;i']=0.
    elif change=='strain':value['sample']['Mustrain;i']=-1.
    elif change=='mix_low':value['sample']['Size;mx']=-.1
    elif change=='mix_high':value['sample']['Mustrain;mx']=1.1
    elif change=='difc':value['bank']['difC']=0.
    elif change=='wrong_schema':value['schema']='executable'
    else:value=[]
    with pytest.raises(ValueError):validate(value)


@pytest.mark.parametrize('value',[None,[],[1.],[1.,2.,3.],[True,2.],['1',2.],
    [[1.],[2.]],[0.,2.],[-1.,2.],[2.,1.],[1.,1.],[1.,float('inf')],[float('nan'),2.]])
def test_invalid_compilation_envelopes_are_rejected_before_native_work(value):
    with pytest.raises(ValueError,match='envelope'):_bounds(value,'invalid envelope')
