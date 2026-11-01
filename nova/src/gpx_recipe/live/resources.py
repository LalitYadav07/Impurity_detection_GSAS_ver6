"""Bound the dense live kernel before NumPy creates its working arrays."""
from __future__ import annotations
import os
from gpx_recipe.symmetry.resources import allocation


def kernel_budget(dimensions,bins,particles,*,resources=None):
    if any(type(v) is not int or v<1 for v in (dimensions,bins,particles)):
        raise ValueError('Positive integral kernel dimensions, bins and particles required')
    resources=allocation(1) if resources is None else resources
    active=max(1,int(os.environ.get('RADAR_PD_LIVE_WORKERS',resources.get('workers',1))))
    memory=resources.get('memory_mb')
    per_worker=((memory-resources.get('reserve_mb',512))//active if memory else 1024)*1024**2
    cells=dimensions*dimensions*bins
    triangle=dimensions*(dimensions+1)//2
    # Native workspace + quadratic tensor + QR temporaries + display envelopes
    # and particles. No bins or posterior draws are dropped to fit this budget.
    estimated=(256*1024**2 + 8*cells + 24*triangle*bins +
               16*min(64,2*particles)*bins + 80*particles*dimensions + 160*bins)
    if cells>40_000_000 or estimated>per_worker:
        raise ValueError('Live candidate exceeds the allocated dense-kernel memory budget; use a larger resource class or review a smaller candidate scope')
    return dict(dimensions=dimensions,bins=bins,particles=particles,quadratic_cells=cells,
                estimated_peak_bytes=estimated,per_worker_budget_bytes=per_worker,active_workers=active)
