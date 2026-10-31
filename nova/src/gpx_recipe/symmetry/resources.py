"""Respect scheduler allocation; CPU visibility alone grants no worker slots."""
import math
import os
from pathlib import Path


def affinity_limit():
    try:return len(os.sched_getaffinity(0)) or None
    except (AttributeError,OSError):return None


def allocation(requested, *, environ=None, cgroup=Path('/sys/fs/cgroup')):
    env = os.environ if environ is None else environ
    if type(requested) is not int or not 1 <= requested <= 16:
        raise ValueError('Request one to sixteen native workers')
    def integer(name):
        try:
            value=int(env.get(name,''))
            return value if value > 0 else None
        except (TypeError,ValueError):return None
    slots=integer('GALAXY_SLOTS')
    cpu_quota=None
    try:
        quota,period=(cgroup/'cpu.max').read_text().split()
        if quota != 'max':
            cpu_quota=float(quota)/float(period)
    except (OSError,ValueError,ZeroDivisionError):pass
    if cpu_quota is None:
        for base in (cgroup,cgroup/'cpu',cgroup/'cpu,cpuacct'):
            try:
                quota=float((base/'cpu.cfs_quota_us').read_text())
                period=float((base/'cpu.cfs_period_us').read_text())
                if quota>0 and period>0:cpu_quota=quota/period;break
            except (OSError,ValueError):pass
    if cpu_quota is not None and (not math.isfinite(cpu_quota) or cpu_quota<=0):cpu_quota=None
    # An explicit container CPU quota is also an allocation. CPU affinity or
    # the number of visible host cores remains only a limit, never a grant.
    granted=slots or (max(1,math.floor(cpu_quota)) if cpu_quota is not None else 1)
    limits=[requested,granted]
    affinity=affinity_limit()
    if affinity is not None:limits.append(affinity)
    if cpu_quota is not None:limits.append(max(1,math.floor(cpu_quota)))
    memory=integer('GALAXY_MEMORY_MB')
    per_slot=integer('GALAXY_MEMORY_MB_PER_SLOT')
    if per_slot and slots:memory=min(memory or per_slot*slots,per_slot*slots)
    try:
        value=(cgroup/'memory.max').read_text().strip()
        if value!='max':
            quota_memory=int(value)//(1024**2)
            memory=min(memory or quota_memory,quota_memory)
    except (OSError,ValueError):pass
    for base in (cgroup,cgroup/'memory'):
        try:
            quota_memory=int((base/'memory.limit_in_bytes').read_text())//(1024**2)
            # The cgroup-v1 unlimited sentinel is near signed int64 maximum.
            if 0<quota_memory<2**40:memory=min(memory or quota_memory,quota_memory)
        except (OSError,ValueError):pass
    # Conservative initial reserve; retain the measurement in every receipt.
    # Complex supercells may require a larger scheduler resource class.
    reserve_mb,worker_mb=512,1024
    if memory:
        if memory < reserve_mb+worker_mb:
            raise ValueError('The allocated memory is below the model worker minimum (1536 MiB)')
        limits.append(max(1,(memory-reserve_mb)//worker_mb))
    return dict(requested_workers=requested,workers=min(limits),galaxy_slots=slots,
        cpu_quota=cpu_quota,memory_mb=memory,reserve_mb=reserve_mb,minimum_worker_mb=worker_mb,
        cpu_affinity_count=affinity,visible_cpu_count=os.cpu_count(),
        allocation_reported=slots is not None or cpu_quota is not None,
        allocation_source='galaxy_slots' if slots else 'container_cpu_quota' if cpu_quota is not None else 'unreported_single_worker',blas_threads=1)
