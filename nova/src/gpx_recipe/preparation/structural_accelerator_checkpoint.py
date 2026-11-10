"""Device RNG envelope around the unchanged canonical optimizer audit.

The provider owns hardware placement. Numerical model, AdamW, schedule,
history, visitation and CPU RNG validation remain in the shared CPU engine.
These working checkpoints are not trained-model library artifacts.
"""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from . import structural_training_checkpoint as canonical
from ..live.array_limits import validate_npz_bytes
from .tof_kernel_arrays import read_bounded,array_identity

MAX_BYTES=32*1024**2


def inspect_bytes(raw,*,schema,identity=None):
    """Hardware-independent, bounded transport inspection; no tensor execution."""
    validate_npz_bytes(raw,maximum_bytes=MAX_BYTES)
    with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
        if set(archive.files)!={'metadata','cpu_checkpoint','device_rng'}:
            raise ValueError('Owned accelerator checkpoint inventory changed')
        arrays={key:archive[key] for key in archive.files}
    encoded=arrays.pop('metadata')
    if encoded.dtype!=np.uint8 or encoded.ndim!=1 or len(encoded)>4096:
        raise ValueError('Invalid owned accelerator metadata')
    metadata=json.loads(encoded.tobytes())
    if (not isinstance(metadata,dict) or set(metadata)!={'schema','identity','runtime','arrays_sha256'}
            or metadata['schema']!=schema or identity is not None and metadata['identity']!=identity
            or metadata['arrays_sha256']!=array_identity(arrays)
            or any(a.dtype!=np.uint8 or a.ndim!=1 for a in arrays.values())
            or not 0<len(arrays['cpu_checkpoint'])<=canonical.MAX_BYTES
            or not 0<len(arrays['device_rng'])<=1024**2):
        raise ValueError('Owned accelerator identity or numerical bytes changed')
    return arrays,metadata


class Checkpoint:
    owned_run=staticmethod(canonical.owned_run)

    def __init__(self,provider):self.provider=provider

    def identity(self,*args,**kwargs):
        base=canonical.identity(*args,**kwargs)
        return hashlib.sha256((base+json.dumps(self.provider.runtime(),sort_keys=True)).encode()).hexdigest()

    def cpu_views(self,model,optimizer,scheduler):
        import torch
        self.provider.require_model(model)
        cpu=copy.deepcopy(model).cpu()
        opt=torch.optim.AdamW(cpu.parameters(),lr=.0003,weight_decay=.0001)
        opt.load_state_dict(optimizer.state_dict())
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=scheduler.T_max,eta_min=scheduler.eta_min)
        schedule.load_state_dict(scheduler.state_dict())
        return cpu,opt,schedule

    def save(self,directory,run_identity,model,optimizer,scheduler,rng,visited,**kwargs):
        directory=Path(directory);self.provider.synchronize()
        device_rng=self.provider.get_rng().cpu().numpy().copy()
        cpu,opt,schedule=self.cpu_views(model,optimizer,scheduler)
        with tempfile.TemporaryDirectory(prefix='device-cpu-stage-',dir=directory) as temporary:
            canonical.save(Path(temporary),run_identity,cpu,opt,schedule,rng,visited,**kwargs)
            raw=read_bounded(Path(temporary)/'training-checkpoint.npz',canonical.MAX_BYTES)
        arrays=dict(cpu_checkpoint=np.frombuffer(raw,np.uint8).copy(),device_rng=device_rng)
        metadata=dict(schema=self.provider.SCHEMA,identity=run_identity,runtime=self.provider.runtime(),
            arrays_sha256=array_identity(arrays))
        arrays['metadata']=np.frombuffer(json.dumps(metadata,sort_keys=True).encode(),np.uint8)
        fd,name=tempfile.mkstemp(prefix='device-state-',suffix='.partial',dir=directory)
        try:
            with os.fdopen(fd,'wb') as stream:
                np.savez(stream,**arrays);stream.flush();os.fsync(stream.fileno())
            validate_npz_bytes(read_bounded(name,MAX_BYTES),maximum_bytes=MAX_BYTES)
            os.replace(name,directory/self.provider.STATE)
        finally:Path(name).unlink(missing_ok=True)

    def read(self,path,run_identity):
        import torch
        arrays,metadata=inspect_bytes(read_bounded(path,MAX_BYTES),schema=self.provider.SCHEMA,identity=run_identity)
        if (metadata['runtime']!=self.provider.runtime()
                or arrays['device_rng'].shape!=tuple(self.provider.get_rng().shape)):
            raise ValueError('Owned accelerator identity, bytes or runtime changed')
        before=self.provider.get_rng()
        try:self.provider.set_rng(torch.from_numpy(arrays['device_rng'].copy()))
        finally:self.provider.set_rng(before)
        return arrays

    def restore(self,directory,run_identity,model,optimizer,scheduler,rng,visited,*,epochs,warmup_steps,row_layout=None):
        import torch
        self.provider.require_model(model);directory=Path(directory);path=directory/self.provider.STATE
        if not path.exists() and not path.is_symlink():return None
        arrays=self.read(path,run_identity);cpu,opt,schedule=self.cpu_views(model,optimizer,scheduler)
        copied_rng=np.random.default_rng();copied_rng.bit_generator.state=copy.deepcopy(rng.bit_generator.state)
        copied_visited=visited.copy()
        with tempfile.TemporaryDirectory(prefix='device-cpu-restore-',dir=directory) as temporary:
            (Path(temporary)/'training-checkpoint.npz').write_bytes(arrays['cpu_checkpoint'].tobytes())
            result=canonical.restore(Path(temporary),run_identity,cpu,opt,schedule,copied_rng,copied_visited,
                epochs=epochs,warmup_steps=warmup_steps,row_layout=row_layout)
        model.load_state_dict(cpu.state_dict(),strict=True)
        optimizer.load_state_dict(opt.state_dict());scheduler.load_state_dict(schedule.state_dict())
        rng.bit_generator.state=copied_rng.bit_generator.state;visited[:]=copied_visited
        self.provider.set_rng(torch.from_numpy(arrays['device_rng'].copy()));self.provider.synchronize()
        return result
