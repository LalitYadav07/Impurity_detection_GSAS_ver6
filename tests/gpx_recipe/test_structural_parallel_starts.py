import multiprocessing
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

from gpx_recipe.structural_live.service import SearchSettings
from gpx_recipe.structural_live.optimization import parallel_starts as module


def test_serial_path_uses_the_current_shared_optimizer(monkeypatch):
    from gpx_recipe.structural_live.optimization import profiled_native_fit
    called=[]
    def current(*args, **kwargs):
        called.append(kwargs)
        return {'method':'current_optimizer'}, np.ones(3)
    monkeypatch.setattr(profiled_native_fit,'fit',current)
    monkeypatch.setattr(module,'allocation',lambda _:dict(workers=1))
    result,_=module.fit_allocated(None,None,SearchSettings(),None,None,None,None)
    assert called and result['method']=='current_optimizer'
    assert result['parallel_execution']['workers']==1


def failing_start(directory,*unused):
    """Use real child processes to test failure cleanup, without native work."""
    if Path(directory).name=='start-00':raise RuntimeError('deliberate test failure')
    time.sleep(30)


def test_failed_start_is_not_dropped_and_other_owned_processes_are_stopped(tmp_path,monkeypatch):
    before={p.pid for p in multiprocessing.active_children()}
    monkeypatch.setattr(module,'allocation',lambda _:dict(workers=4,memory_mb=8192,reserve_mb=512))
    monkeypatch.setattr(module,'_start_worker',failing_start)
    def screened(*args,**kwargs):
        assert kwargs['screening_only']
        return dict(selected_starts=[[0.,0.,0.,0.]]*4,seconds=0.,full_native_calls=1,
                    best=dict(parameters=[0.]*4,deviance=1.)),np.ones(3)
    monkeypatch.setattr(module,'fit',screened)
    bank=SimpleNamespace(points=np.zeros((1,2)),proposal_cell_steps=np.ones(2))
    with pytest.raises(RuntimeError,match='start 0 failed'):
        module.fit_allocated(tmp_path,tmp_path/'parallel',SearchSettings(seconds=10),
                             None,None,None,bank)
    assert {p.pid for p in multiprocessing.active_children()}<=before
