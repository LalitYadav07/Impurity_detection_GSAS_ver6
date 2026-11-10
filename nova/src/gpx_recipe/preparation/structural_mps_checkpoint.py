"""Owned MPS recovery delegating numerical audits to the shared CPU code.

This working envelope is excluded from portable libraries. Both device RNG and the
canonical CPU checkpoint are committed in one atomic, bounded data-only file.
"""
import copy
import hashlib
import io
import json
import os
import platform
import tempfile
from pathlib import Path

import numpy as np
import torch
from . import structural_training_checkpoint as canonical
from ..live.array_limits import validate_npz_bytes
from .tof_kernel_arrays import read_bounded, array_identity

SCHEMA='radar-owned-mps-training-continuation/v1'
STATE='training-mps-checkpoint.npz'
MAX_BYTES=32*1024**2
owned_run=canonical.owned_run


def runtime():
    return dict(device='mps',torch=str(torch.__version__),machine=platform.machine(),macos=platform.mac_ver()[0])


def identity(*args,**kwargs):
    base=canonical.identity(*args,**kwargs)
    return hashlib.sha256((base+json.dumps(runtime(),sort_keys=True)
        +hashlib.sha256(Path(__file__).read_bytes()).hexdigest()).encode()).hexdigest()


def _require_device(model):
    if not torch.backends.mps.is_available() or {p.device.type for p in model.parameters()}!={'mps'}:
        raise ValueError('This private continuation needs a complete MPS model')


def _cpu_views(model,optimizer,scheduler):
    """The same installed AdamW state; no extra fitting or optimizer step."""
    _require_device(model)
    cpu=copy.deepcopy(model).cpu()
    opt=torch.optim.AdamW(cpu.parameters(),lr=.0003,weight_decay=.0001)
    opt.load_state_dict(optimizer.state_dict())
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=scheduler.T_max,eta_min=scheduler.eta_min)
    schedule.load_state_dict(scheduler.state_dict())
    return cpu,opt,schedule


def save(directory,run_identity,model,optimizer,scheduler,rng,visited,**kwargs):
    directory=Path(directory);torch.mps.synchronize()
    device_rng=torch.mps.get_rng_state().cpu().numpy().copy()
    cpu,opt,schedule=_cpu_views(model,optimizer,scheduler)
    with tempfile.TemporaryDirectory(prefix='mps-cpu-stage-',dir=directory) as temporary:
        canonical.save(Path(temporary),run_identity,cpu,opt,schedule,rng,visited,**kwargs)
        raw=read_bounded(Path(temporary)/'training-checkpoint.npz',canonical.MAX_BYTES)
    arrays=dict(cpu_checkpoint=np.frombuffer(raw,np.uint8).copy(),mps_rng=device_rng)
    metadata=dict(schema=SCHEMA,identity=run_identity,runtime=runtime(),arrays_sha256=array_identity(arrays))
    arrays['metadata']=np.frombuffer(json.dumps(metadata,sort_keys=True).encode(),np.uint8)
    fd,name=tempfile.mkstemp(prefix='mps-state-',suffix='.partial',dir=directory)
    try:
        with os.fdopen(fd,'wb') as stream:
            np.savez(stream,**arrays);stream.flush();os.fsync(stream.fileno())
        validate_npz_bytes(read_bounded(name,MAX_BYTES),maximum_bytes=MAX_BYTES)
        os.replace(name,directory/STATE)
    finally:Path(name).unlink(missing_ok=True)


def _read(path,run_identity):
    raw=read_bounded(path,MAX_BYTES);validate_npz_bytes(raw,maximum_bytes=MAX_BYTES)
    with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
        if set(archive.files)!={'metadata','cpu_checkpoint','mps_rng'}:
            raise ValueError('Owned MPS checkpoint inventory changed')
        arrays={name:archive[name] for name in archive.files}
    encoded=arrays.pop('metadata')
    if encoded.dtype!=np.uint8 or encoded.ndim!=1 or len(encoded)>4096:
        raise ValueError('Invalid owned MPS metadata')
    metadata=json.loads(encoded.tobytes())
    if (not isinstance(metadata,dict) or set(metadata)!={'schema','identity','runtime','arrays_sha256'}
            or metadata['schema']!=SCHEMA or metadata['identity']!=run_identity
            or metadata['runtime']!=runtime() or metadata['arrays_sha256']!=array_identity(arrays)
            or any(a.dtype!=np.uint8 or a.ndim!=1 for a in arrays.values())
            or not 0<len(arrays['cpu_checkpoint'])<=canonical.MAX_BYTES
            or arrays['mps_rng'].shape!=tuple(torch.mps.get_rng_state().shape)):
        raise ValueError('Owned MPS continuation identity, bytes or device changed')
    # Validate the installed RNG decoder, then restore the caller's state.
    before=torch.mps.get_rng_state()
    try:torch.mps.set_rng_state(torch.from_numpy(arrays['mps_rng'].copy()))
    finally:torch.mps.set_rng_state(before)
    return arrays


def restore(directory,run_identity,model,optimizer,scheduler,rng,visited,*,epochs,warmup_steps,row_layout=None):
    _require_device(model);directory=Path(directory);path=directory/STATE
    if not path.exists() and not path.is_symlink():return None
    arrays=_read(path,run_identity);cpu,opt,schedule=_cpu_views(model,optimizer,scheduler)
    copied_rng=np.random.default_rng();copied_rng.bit_generator.state=copy.deepcopy(rng.bit_generator.state)
    copied_visited=visited.copy()
    with tempfile.TemporaryDirectory(prefix='mps-cpu-restore-',dir=directory) as temporary:
        (Path(temporary)/'training-checkpoint.npz').write_bytes(arrays['cpu_checkpoint'].tobytes())
        result=canonical.restore(Path(temporary),run_identity,cpu,opt,schedule,copied_rng,copied_visited,
            epochs=epochs,warmup_steps=warmup_steps,row_layout=row_layout)
    # The canonical audit has checked every tensor, optimizer setting,
    # schedule, incumbent and complete boundary before replacing this model.
    model.load_state_dict(cpu.state_dict(),strict=True)
    optimizer.load_state_dict(opt.state_dict());scheduler.load_state_dict(schedule.state_dict())
    rng.bit_generator.state=copied_rng.bit_generator.state;visited[:]=copied_visited
    torch.mps.set_rng_state(torch.from_numpy(arrays['mps_rng'].copy()))
    torch.mps.synchronize()
    return result
