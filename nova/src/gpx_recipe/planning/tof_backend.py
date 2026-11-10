"""Scoped, pinned forward-only use of the normalized TOF convolution.

Installed GSAS sources/binaries are unchanged. A dedicated main-thread worker
temporarily supplies the point-density callback and disables native worker
processes that would otherwise import the unmodified kernel. Native profile
derivatives are deliberately unavailable inside this forward-only context.
"""
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import threading

import numpy as np
from . import tof_profile

VERSION='powgen-forward-TOF-convolution-adapter/2'
_LOCK=threading.Lock()
_PINNED={
    ('pwd','getEpsVoigt'):'a299231c52b148b2be785c4fd0b6d6ad8899cf5df67925adc06dc2fa297f73f3',
    ('pwd','getdEpsVoigt'):'cbc214d0890f950f36dd358f65332a5e208204ab5bf21c2c8395b5c992cdd07a',
    ('mp','InitMP'):'7c8589fa4585a8688b6147f2caa1d6992b552246f0a7542fb686bad285df7031',
    ('mp','ComputeFobsSqTOF'):'6eb953eb2f662df69eff4d939d7933be7a16976f3c8bb36b5724c6511a18e8e1',
    ('math','getPowderProfile'):'caa9e0aac821043004b54a2382008b2c0d04abb97ac8b057d57be6044880944d',
    ('main','DoNoFit'):'d62f916a9b164762ca432ff4d6f09e2d1d2c1a71cdfd7c5f86ca76e30299fe1c'}


@contextmanager
def physical_tof_forward(reflection_transform=None):
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Use a dedicated main-thread process for TOF calculations')
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError('Nested or concurrent TOF forward contexts are forbidden')
    saved=None
    try:
        from GSASII import GSASIIpwd as P,GSASIImpsubs as MP,GSASIIstrMath as M,GSASIIstrMain as S
        modules={'pwd':P,'mp':MP,'math':M,'main':S};found={}
        for (key,name),expected in _PINNED.items():
            fn=getattr(modules[key],name)
            actual=hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
            if actual!=expected:raise RuntimeError('Unreviewed native TOF route: '+key+'.'+name)
            found[key+'.'+name]=actual
        if M.G2pwd is not P or MP.G2pwd is not P or M.G2mp is not MP:
            raise RuntimeError('Native TOF module aliases changed')
        had_use=hasattr(MP,'useMP');old_use=getattr(MP,'useMP',None);old_cores=MP.ncores
        saved=(P,MP,P.getEpsVoigt,P.getdEpsVoigt,had_use,old_use,old_cores)
        identity={'version':VERSION,'density_version':tof_profile.VERSION,
            'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'density_source_sha256':hashlib.sha256(Path(tof_profile.__file__).read_bytes()).hexdigest(),
            'native_route_function_sha256':found,'single_process_forward_route_enforced':True,
            'native_finite_reflection_windows_retained':True,
            'density_width_or_probability_floor_applied':False,
            'density_calls':0,'density_points':0,'native_state_restored':False,
            'installed_sources_modified':False,'experimental_accuracy_certified':False}
        def evaluate(pos,alp,bet,sig,gam,xdata):
            x=np.asarray(xdata,float)
            if x.ndim!=1 or len(x)<2 or np.any(np.diff(x)<=0):
                raise ValueError('A TOF peak interval needs at least two increasing coordinates; refine the grid')
            y=tof_profile.density(x-float(pos),float(alp),float(bet),float(sig),float(gam))
            step=np.diff(x);measure=np.r_[step,step[-1]]
            identity['density_calls']+=1;identity['density_points']+=len(x)
            return y,float(y@measure)
        def derivative_unavailable(*args,**kwargs):
            raise RuntimeError('The normalized TOF adapter supports forward calculations only')
        P.getEpsVoigt=evaluate;P.getdEpsVoigt=derivative_unavailable
        MP.useMP=False;MP.ncores=0
        original_powder=M.getPowderProfile
        if reflection_transform is not None:
            if not callable(reflection_transform):raise ValueError('A reflection transform must be callable')
            def transformed(*args,**kwargs):
                calculated,background=original_powder(*args,**kwargs)
                reflections=[v['RefList'] for k,v in args[4]['Reflection Lists'].items() if k in args[5]]
                value=np.asarray(reflection_transform(np.asarray(args[2],float),reflections),float)
                if value.shape!=calculated.shape or not np.isfinite(value).all() or np.any(value<0):
                    raise ValueError('Invalid transformed TOF reflection density')
                return value,background
            M.getPowderProfile=transformed
            identity['reflection_transform_applied']=True
            identity['native_finite_reflection_windows_retained']=False
        try:yield identity
        finally:
            M.getPowderProfile=original_powder
            P.getEpsVoigt=saved[2];P.getdEpsVoigt=saved[3]
            MP.ncores=old_cores
            if had_use:MP.useMP=old_use
            else:delattr(MP,'useMP')
            identity['native_state_restored']=(M.getPowderProfile is original_powder and P.getEpsVoigt is saved[2] and P.getdEpsVoigt is saved[3]
                and MP.ncores==old_cores and hasattr(MP,'useMP')==had_use
                and (not had_use or MP.useMP==old_use))
            saved=None
    finally:
        if saved is not None:
            P,MP,old_profile,old_derivative,had_use,old_use,old_cores=saved
            P.getEpsVoigt=old_profile;P.getdEpsVoigt=old_derivative;MP.ncores=old_cores
            if had_use:MP.useMP=old_use
            elif hasattr(MP,'useMP'):delattr(MP,'useMP')
        _LOCK.release()
