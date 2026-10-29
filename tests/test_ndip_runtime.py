import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from ndip_runtime import runtime_limits, process_usage, usage_delta


def test_numeric_limits_include_scheduler_and_cgroup_without_environment_secrets(tmp_path,monkeypatch):
    (tmp_path/'cpu.max').write_text('250000 100000')
    (tmp_path/'memory.max').write_text(str(4*1024**3))
    (tmp_path/'memory.peak').write_text('123456')
    row=runtime_limits(cgroup_root=tmp_path,environ={'GALAXY_SLOTS':'2','AZURE_ILLUMINE_API_KEY':'never-publish-this'})
    assert row['cgroup_cpu_limit']==2.5 and row['scheduler_slots']=={'GALAXY_SLOTS':2}
    assert row['observed_cpu_ceiling']<=2
    assert row['cgroup_memory_limit_bytes']==4*1024**3
    assert row['cgroup_memory_peak_bytes']==123456
    assert 'never-publish-this' not in json.dumps(row)


def test_missing_and_unlimited_cgroups_never_invent_an_allocation(tmp_path):
    row=runtime_limits(cgroup_root=tmp_path,environ={})
    assert row['cgroup_cpu_limit'] is None and row['cgroup_memory_limit_bytes'] is None
    (tmp_path/'cpu.max').write_text('max 100000')
    (tmp_path/'memory.max').write_text('max')
    assert runtime_limits(cgroup_root=tmp_path,environ={})['cgroup_cpu_limit'] is None
    (tmp_path/'cpu').mkdir();(tmp_path/'memory').mkdir()
    (tmp_path/'cpu/cpu.cfs_quota_us').write_text('100000')
    (tmp_path/'cpu/cpu.cfs_period_us').write_text('100000')
    (tmp_path/'cpu.max').unlink()
    (tmp_path/'memory/memory.limit_in_bytes').write_text(str(2**63-4096))
    row=runtime_limits(cgroup_root=tmp_path,environ={'GALAXY_SLOTS':'invalid'})
    assert row['cgroup_cpu_limit']==1 and row['cgroup_memory_limit_bytes'] is None


def test_usage_reports_cpu_and_distinguishes_individual_peak_memory():
    row=usage_delta(process_usage())
    if row:
        assert row['self_cpu_seconds']>=0
        assert row['waited_children_cpu_seconds']>=0
        assert row['self_peak_rss_bytes']>0
        assert 'largest_waited_child_peak_rss_bytes' in row
