"""A preparation batch owns one CPU grant, including nested curve pools."""
import copy
import subprocess
import threading
from types import SimpleNamespace

import pytest

from gpx_recipe.preparation import batch


@pytest.mark.parametrize('modes,slots,expected',[(2,4,False),(15,4,False),(16,1,False),(16,4,True),(32,2,True)])
def test_small_banks_avoid_setup_and_complex_banks_respect_grants(monkeypatch,tmp_path,modes,slots,expected):
    from gpx_recipe.preparation import worker
    from gpx_recipe.structural_live.optimization import parallel_bank
    calls=[]
    def allocated(source,output):
        calls.append((source,output));return SimpleNamespace(workers=slots)
    monkeypatch.setattr(parallel_bank,'NativeBankProcessMap',allocated)
    source=SimpleNamespace(mode_dimensions=modes)
    value=worker.starting_curve_renderer(source,tmp_path/'pool')
    assert (value is not None) is expected
    assert bool(calls)==(modes>=16)


@pytest.mark.parametrize('kinds,examples,parallel,workers',[
    (['structural'],False,1,4),
    (['structural']*4,False,1,4),
    (['reference']*4,False,4,1),
    (['reference']*2,False,2,2),
    (['structural','reference'],False,2,2),
    (['structural-examples']*4,True,4,1),
])
def test_nested_pools_share_the_grant_and_keep_requests_unchanged(kinds,examples,parallel,workers):
    plan=dict(tasks=[dict(kind=kind) for kind in kinds]);before=copy.deepcopy(plan)
    limits=batch.task_resource_limits(plan,dict(workers=4,memory_mb=8192,reserve_mb=512),examples=examples)
    assert limits==dict(workers=workers,memory_mb=7680//parallel,maximum_simultaneous_tasks=parallel)
    assert limits['workers']*parallel<=4
    assert limits['memory_mb']*parallel<=7680
    assert plan==before


def test_one_slot_remains_serial_when_a_larger_host_is_visible():
    limits=batch.task_resource_limits(dict(tasks=[dict(kind='structural')]),
        dict(workers=1,memory_mb=None,reserve_mb=512))
    assert limits==dict(workers=1,memory_mb=None,maximum_simultaneous_tasks=1)


def test_child_receives_its_share_without_changing_parent_environment(tmp_path,monkeypatch):
    monkeypatch.setenv('GALAXY_SLOTS','4');monkeypatch.setenv('GALAXY_MEMORY_MB','8192')
    captured={}
    def stop(command,**kwargs):
        captured.update(kwargs['env'])
        raise subprocess.TimeoutExpired(command,kwargs['timeout'])
    monkeypatch.setattr(batch,'run_scientific',stop)
    result=batch.run_task(dict(kind='structural',_resource_limits=dict(workers=2,memory_mb=3840)),
        dict(destination='compiled'),tmp_path/'worker',cancel=threading.Event(),timeout=1.)
    assert result['state']=='interrupted'
    assert captured['GALAXY_SLOTS']=='2' and captured['GALAXY_MEMORY_MB']=='3840'
    import os
    assert os.environ['GALAXY_SLOTS']=='4' and os.environ['GALAXY_MEMORY_MB']=='8192'


def test_legacy_direct_task_cannot_inherit_the_whole_parent_pool(tmp_path,monkeypatch):
    monkeypatch.setenv('GALAXY_SLOTS','4')
    captured={}
    def stop(command,**kwargs):
        captured.update(kwargs['env']);raise subprocess.TimeoutExpired(command,kwargs['timeout'])
    monkeypatch.setattr(batch,'run_scientific',stop)
    batch.run_task(dict(kind='reference'),{},tmp_path/'worker',cancel=threading.Event(),timeout=1.)
    assert captured['GALAXY_SLOTS']=='1'
