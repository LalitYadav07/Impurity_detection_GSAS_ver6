"""Analysis overrides cannot exceed an NDIP job's declared resources."""
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import worker_budget
from gpx_recipe.symmetry import resources


@pytest.fixture
def budget(monkeypatch,tmp_path):
    monkeypatch.setattr(worker_budget,'_resources',lambda:resources)
    original=resources.allocation
    monkeypatch.setattr(resources,'allocation',lambda requested,**kwargs:
        original(requested,cgroup=tmp_path,**kwargs))
    monkeypatch.setattr(resources,'affinity_limit',lambda:64)
    monkeypatch.setattr(os,'cpu_count',lambda:64)
    return {'RADAR_PD_EXECUTION_CONTEXT':'galaxy','GALAXY_SLOTS':'4','GALAXY_MEMORY_MB':'8192'}


def test_visible_64_cpu_host_does_not_authorize_32_analysis_workers(budget):
    result=worker_budget.analysis_allocation(32,tasks=100,environ=budget)
    assert result['workers']==4
    assert result['allocation_source']=='galaxy_slots'


def test_analysis_override_is_bounded_by_cpu_memory_and_work(budget):
    assert worker_budget.analysis_allocation(8,tasks=100,environ={**budget,'GALAXY_SLOTS':'2'})['workers']==2
    assert worker_budget.analysis_allocation(32,tasks=100,environ={**budget,'GALAXY_MEMORY_MB':'3072'})['workers']==2
    assert worker_budget.analysis_allocation(32,tasks=1,environ=budget)['workers']==1


def test_unknown_ndip_allocation_remains_serial(budget):
    result=worker_budget.analysis_allocation(32,tasks=100,environ={'RADAR_PD_EXECUTION_CONTEXT':'galaxy'})
    assert result['workers']==1 and not result['allocation_reported']


def test_standalone_capacity_does_not_claim_a_scheduler_grant(budget):
    result=worker_budget.analysis_allocation(32,tasks=100,environ={})
    assert result['workers']==32
    assert not result['allocation_reported']
    assert result['allocation_source']=='standalone_local_capacity'


@pytest.mark.parametrize('value',[True,0,257,1.5,'4'])
def test_invalid_analysis_override_is_rejected(budget,value):
    with pytest.raises(ValueError):worker_budget.analysis_allocation(value,tasks=10,environ=budget)


def test_rapid_final_fitting_override_cannot_bypass_allocation(budget,monkeypatch):
    from scripts.rapid_hypothesis_pipeline import _resolve_gsas_workers
    for key,value in {**budget,'GALAXY_SLOTS':'2'}.items():monkeypatch.setenv(key,value)
    assert _resolve_gsas_workers(20,{'gsas_parallel_workers':8})==2


def test_library_builders_share_the_analysis_allocation(budget,monkeypatch):
    from scripts.db_pack_builder import _pick_workers as pack_workers
    from scripts.augment_from_cifs import _pick_workers as augment_workers
    for key,value in {**budget,'GALAXY_SLOTS':'2'}.items():monkeypatch.setenv(key,value)
    assert pack_workers(20)==2
    assert augment_workers(20)==2
    assert pack_workers(1)==augment_workers(1)==1
