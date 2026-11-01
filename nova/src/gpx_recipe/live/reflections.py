"""Read native reflection tables at saved posterior maxima; never refine."""
from pathlib import Path
import sys,json,hashlib,numpy as np
from contextlib import nullcontext
import threading
from .native_profile import NativeReducedProfile
from .validated_backend import corrected_backend,active_identity

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def export_reflections(dest,source,*,backend_active=False):
 if backend_active and (active_identity() is None or threading.current_thread() is not threading.main_thread()):raise RuntimeError('Reflection export requires the active native backend in its owning process')
 dest,source=Path(dest),Path(source)
 with np.load(dest/'posterior.npz') as a:best=a['samples'][np.argmax(a['loglike'])]
 with np.load(dest/'kernel.npz') as a:
  expected=a['base']+np.einsum('i,ijb,j->b',best,a['quadratic'],best);x=a['x'].copy()
 with (nullcontext() if backend_active else corrected_backend()):
  n=NativeReducedProfile(source/'magnetic-refinement.gpx');direct=n.profile(best)
  native_x=np.concatenate([n.histo_phases[0][name]['Data'][0][sl] for name,sl in n.parts]);weight=np.concatenate([n.original_weights[name][sl] for name,sl in n.parts]);mask=np.isfinite(native_x)&(weight>0)
  if not np.allclose(x,native_x[mask]):raise ValueError('Different masked grid')
  err=float(np.max(abs(expected-direct[mask]))/max(np.max(abs(expected)),1e-12))
  if err>1e-7:raise ValueError(f'Native posterior mismatch {err}')
  tables=[]
  for histname,sl in n.parts:
   for name,r in n.histo_phases[0][histname]['Reflection Lists'].items():
    if r.get('Super'):raise ValueError('Modulated ticks unsupported')
    kind=n.histo_phases[1][name]['General']['Type'];a=r['RefList'];tol=max(1e-12,float(np.max(a[:,9]))*1e-10)
    rows=a[(a[:,9]>tol)&(a[:,5]>=x.min())&(a[:,5]<=x.max())]
    tables.append(dict(phase=name,kind=kind,hkl=rows[:,:3].astype(int).tolist(),d=rows[:,4].tolist(),tof=rows[:,5].tolist(),f2=rows[:,9].tolist()))
 payload=dict(schema='native-reflection-ticks/v1',kernel_sha256=sha(dest/'kernel.npz'),posterior_sha256=sha(dest/'posterior.npz'),source_gpx_sha256=sha(source/'magnetic-refinement.gpx'),native_profile_relative_error=err,scope='Nonzero Fcalc squared at saved posterior maximum; hkl in each phase setting; d and TOF from GSAS-II',tables=tables)
 (dest/'reflections.json').write_text(json.dumps(payload))
 return payload
