"""Apply observations to an unchanged validated response; never accumulate scans."""
from pathlib import Path
import numpy as np
from .contracts import file_hash


def replace_observations(kernel, request):
    if not request.get('observation'):return False
    if file_hash(request['observation'])!=request['observation_sha256']:raise ValueError('Observation changed')
    with np.load(kernel,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    with np.load(request['observation'],allow_pickle=False) as z:o={k:z[k].copy() for k in z.files}
    x=o['x'];idx=np.minimum(np.searchsorted(x,data['x']),len(x)-1)
    if not np.allclose(x[idx],data['x'],rtol=1e-12,atol=1e-8):raise ValueError('Snapshot grid is incompatible with native kernel')
    if np.any(o['mask'][idx]) or np.any(o['sigma'][idx]<=0):raise ValueError('Snapshot masks differ from compiled response')
    data.update(y=o['y'][idx],sigma=o['sigma'][idx])
    if not np.isfinite(data['y']).all() or not np.isfinite(data['sigma']).all():raise ValueError('Nonfinite observations')
    tmp=Path(kernel).with_suffix('.next.npz');np.savez_compressed(tmp,**data);tmp.replace(kernel)
    return True
