import pytest
from gpx_recipe.live.resources import kernel_budget
from gpx_recipe.symmetry.resources import allocation


def test_dense_kernel_budget_refuses_before_allocating(monkeypatch):
    monkeypatch.delenv('RADAR_PD_LIVE_WORKERS',raising=False)
    resources={'workers':1,'memory_mb':1536,'reserve_mb':512}
    assert kernel_budget(3,10000,256,resources=resources)['estimated_peak_bytes']<1024**3
    with pytest.raises(ValueError,match='memory budget'):kernel_budget(64,100000,256,resources=resources)
    with pytest.raises(ValueError,match='memory budget'):kernel_budget(1,1000000,4096,resources=resources)


def test_one_galaxy_slot_never_admits_four_workers(tmp_path,monkeypatch):
    monkeypatch.setattr('gpx_recipe.symmetry.resources.affinity_limit',lambda:64)
    resources=allocation(4,environ={'GALAXY_SLOTS':'1','GALAXY_MEMORY_MB':'4096'},cgroup=tmp_path)
    assert resources['workers']==1
    monkeypatch.setenv('RADAR_PD_LIVE_WORKERS','4')
    small=kernel_budget(3,10000,256,resources=resources)
    assert small['per_worker_budget_bytes']==896*1024**2
