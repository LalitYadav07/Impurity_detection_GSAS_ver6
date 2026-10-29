"""Numeric runtime receipts, without credentials or a dump of the environment."""
from __future__ import annotations

import os
from pathlib import Path
import sys


def _text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ''


def _positive(value):
    try:
        number = float(value)
        return number if 0 < number < 2**60 else None
    except (TypeError, ValueError, OverflowError):
        return None


def runtime_limits(*, cgroup_root=Path('/sys/fs/cgroup'), environ=None):
    """Observed process/container ceilings, not an NDIP scheduling entitlement."""
    root = Path(cgroup_root)
    env = os.environ if environ is None else environ
    slots = {name:int(value) for name in ('GALAXY_SLOTS', 'NSLOTS', 'SLURM_CPUS_PER_TASK')
             if (value := env.get(name, '')).isascii() and value.isdecimal() and int(value) > 0}
    try:
        affinity = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    except OSError:
        affinity = None
    cpu = _text(root/'cpu.max').split()
    if len(cpu) != 2:
        cpu = [_text(root/'cpu/cpu.cfs_quota_us'), _text(root/'cpu/cpu.cfs_period_us')]
    quota, period = (_positive(v) for v in cpu)
    cpu_limit = quota/period if quota and period else None
    memory = _positive(_text(root/'memory.max')) or _positive(_text(root/'memory/memory.limit_in_bytes'))
    peak = _positive(_text(root/'memory.peak')) or _positive(_text(root/'memory/memory.max_usage_in_bytes'))
    ceilings = [v for v in (os.cpu_count(), affinity, cpu_limit, *slots.values()) if v]
    return dict(logical_cpus=os.cpu_count(), affinity_cpus=affinity, scheduler_slots=slots,
                cgroup_cpu_limit=cpu_limit, cgroup_memory_limit_bytes=int(memory) if memory else None,
                cgroup_memory_peak_bytes=int(peak) if peak else None,
                observed_cpu_ceiling=min(ceilings) if ceilings else None,
                scope='Observed runtime limits; shared-cluster quotas and scheduling policy require NDIP confirmation')


def process_usage():
    try:
        import resource
        own = resource.getrusage(resource.RUSAGE_SELF)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
    except (ImportError, OSError):
        return {}
    factor = 1 if sys.platform == 'darwin' else 1024
    return dict(self_cpu_seconds=own.ru_utime+own.ru_stime,
                waited_children_cpu_seconds=children.ru_utime+children.ru_stime,
                self_peak_rss_bytes=int(own.ru_maxrss*factor),
                largest_waited_child_peak_rss_bytes=int(children.ru_maxrss*factor))


def usage_delta(before):
    after = process_usage()
    for name in ('self_cpu_seconds', 'waited_children_cpu_seconds'):
        if name in after:
            after[name] = max(0, after[name]-before.get(name, 0))
    return after
