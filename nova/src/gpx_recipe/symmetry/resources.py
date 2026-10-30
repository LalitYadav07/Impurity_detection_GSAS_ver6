"""Respect scheduler allocation; CPU visibility alone grants no worker slots."""
import math
import os
from pathlib import Path


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
    limits=[requested,slots or 1]
    cpu_quota=None
    try:
        quota,period=(cgroup/'cpu.max').read_text().split()
        if quota != 'max':
            cpu_quota=float(quota)/float(period)
            limits.append(max(1,math.floor(cpu_quota)))
    except (OSError,ValueError,ZeroDivisionError):pass
    memory=integer('GALAXY_MEMORY_MB')
    per_slot=integer('GALAXY_MEMORY_MB_PER_SLOT')
    if per_slot and slots:memory=min(memory or per_slot*slots,per_slot*slots)
    try:
        value=(cgroup/'memory.max').read_text().strip()
        if value!='max':
            quota_memory=int(value)//(1024**2)
            memory=min(memory or quota_memory,quota_memory)
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
        allocation_reported=slots is not None,blas_threads=1)
