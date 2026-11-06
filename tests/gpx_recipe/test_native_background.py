"""Owned inputs, bounded columns and native background fallback contracts."""
from types import SimpleNamespace
import copy

import numpy as np
import pytest

from gpx_recipe.live import native_background as module


def arguments(monkeypatch):
    monkeypatch.setattr(module,'_reviewed',lambda _:True)
    calls=[]
    def background(prefix,parameters,kind,data_type,x,fixed):
        calls.append(copy.deepcopy(parameters))
        return (parameters[prefix+'Back;0']+parameters[prefix+'Back;1']*x+
            parameters[prefix+'Back;2']*x*x+fixed),[0.,0.,0.]
    return dict(math=SimpleNamespace(G2pwd=SimpleNamespace(getBackground=background)),
        histogram={'hId':0,'Background':[[],{'fixback':np.arange(7.)}]},
        parameters={':0:Back;0':3.,':0:Back;1':.2,':0:Back;2':.1},
        controls={':0:histType':'PNT',':0:bakType':'chebyschev'},axis=np.linspace(1.,2.,7),
        names=[':0:Back;2',':0:Back;0']),calls


def test_native_background_keeps_variable_order_fixed_terms_and_inputs(monkeypatch):
    args,calls=arguments(monkeypatch);parameters=copy.deepcopy(args['parameters'])
    axis=args['axis'].copy();fixed=args['histogram']['Background'][1]['fixback'].copy()
    columns,receipt=module.compile_columns(**args)
    np.testing.assert_allclose(columns,np.column_stack([axis*axis,np.ones(7)]))
    assert len(calls)==receipt['native_background_calls']==3
    assert args['parameters']==parameters
    np.testing.assert_array_equal(args['axis'],axis)
    np.testing.assert_array_equal(args['histogram']['Background'][1]['fixback'],fixed)
    assert all(row[':0:Back;1']==.2 for row in calls)
    assert receipt['observations_used'] is False and receipt['native_functions_replaced'] is False


@pytest.mark.parametrize('change',['foreign','duplicate','grid','fixed_grid','budget','nonfinite'])
def test_unsupported_background_context_requests_original_native_path(monkeypatch,change):
    args,_=arguments(monkeypatch)
    if change=='foreign':args['names']=[':1:Back;0']
    elif change=='duplicate':args['names']*=2
    elif change=='grid':args['axis']=args['axis'][::-1]
    elif change=='fixed_grid':args['histogram']['Background'][1]['fixback']=np.arange(3.)
    elif change=='budget':monkeypatch.setattr(module,'MAX_BYTES',1)
    else:args['math'].G2pwd.getBackground=lambda *a:(np.full(7,np.nan),[])
    with pytest.raises(module.BackgroundCompilerUnavailable):module.compile_columns(**args)


def test_missing_or_active_replacement_native_source_cannot_use_this_compiler(monkeypatch):
    args,_=arguments(monkeypatch);monkeypatch.undo()
    assert module._reviewed(SimpleNamespace()) is False
    with pytest.raises(module.BackgroundCompilerUnavailable,match='needs review'):
        module.compile_columns(**args)
