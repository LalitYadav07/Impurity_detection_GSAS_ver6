"""Execution placement for one shared three-curve training algorithm.

CPU worker continuations keep their existing format. A local MPS run owns
both CPU and device random state; it cannot resume silently on another device.
The trained tensor layout and portable CPU inference remain unchanged.
"""
from contextlib import contextmanager
import copy
from pathlib import Path
import sys

from . import structural_training_checkpoint as cpu_checkpoint


def device(model):
    import torch
    locations={parameter.device for parameter in model.parameters()}
    if len(locations)>1:
        raise ValueError('Structural training requires one complete model device')
    return next(iter(locations),torch.device('cpu'))


def forward(model,values):
    """Preserve the shared CPU loss/evaluation and its gradient connection."""
    location=device(model)
    if values.device==location:return model(values)
    return model(values.to(location)).to(values.device)


class Backend:
    def __init__(self,kind):
        if kind not in ('cpu','mps','cuda'):raise ValueError('Unknown structural training execution device')
        self.kind=kind
        if kind=='cuda':
            from . import structural_cuda_checkpoint
            structural_cuda_checkpoint.runtime()
            self.checkpoint=structural_cuda_checkpoint
        elif kind=='mps':
            from . import structural_mps_checkpoint
            self.checkpoint=structural_mps_checkpoint
        else:self.checkpoint=cpu_checkpoint

    def place(self,model):
        return model if self.kind=='cpu' else model.to(self.kind)

    @contextmanager
    def random_state(self,seed):
        if self.kind=='cpu':
            yield
            return
        import torch
        if self.kind=='cuda':
            before=torch.cuda.get_rng_state(0).clone()
            deterministic=torch.are_deterministic_algorithms_enabled()
            warn_only=torch.is_deterministic_algorithms_warn_only_enabled()
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32
            cudnn_tf32=torch.backends.cudnn.allow_tf32
            benchmark=torch.backends.cudnn.benchmark
            try:
                torch.use_deterministic_algorithms(True)
                torch.backends.cuda.matmul.allow_tf32=False
                torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
                torch.cuda.manual_seed(seed)
                yield
            finally:
                torch.cuda.synchronize(0);torch.cuda.set_rng_state(before,0)
                torch.use_deterministic_algorithms(deterministic,warn_only=warn_only)
                torch.backends.cuda.matmul.allow_tf32=matmul_tf32
                torch.backends.cudnn.allow_tf32=cudnn_tf32;torch.backends.cudnn.benchmark=benchmark
            return
        before=torch.mps.get_rng_state().clone()
        try:
            torch.mps.manual_seed(seed)
            yield
        finally:
            torch.mps.synchronize()
            torch.mps.set_rng_state(before)

    def portable_model(self,model,rows,indices,mean,scale,*,mode_context=None):
        """Audit GPU -> CPU logits on training representatives before export.

        This checks numerical transport, not accuracy or scientific admission.
        Held-out metrics are subsequently measured with this actual CPU model.
        """
        if self.kind=='cpu':return model,None
        import torch
        from .structural_training_data import batches
        cpu=copy.deepcopy(model).cpu().eval();largest=0.;count=0;mode_queries=0;mode_error=0.
        with torch.no_grad():
            for _,values in batches(rows,indices,size=64):
                inputs=torch.from_numpy((values-mean)/scale)
                if mode_context is None:actual=forward(model,inputs);expected=cpu(inputs)
                else:
                    actual,expected,checked=mode_context.transport(model,cpu,inputs)
                    mode_queries+=checked['mode_queries'];mode_error=max(mode_error,checked['maximum_mode_absolute_error'])
                if (not torch.isfinite(actual).all() or not torch.isfinite(expected).all()
                        or not torch.allclose(actual,expected,rtol=1e-4,atol=1e-5)):
                    raise ValueError('GPU training weights do not reproduce portable CPU logits')
                largest=max(largest,float((actual-expected).abs().max()));count+=len(values)
        if not count:raise ValueError('CPU portability requires training representatives')
        return cpu,dict(device=self.kind,portable_inference_device='cpu',
            representatives=count,maximum_logit_absolute_error=largest,
            **(dict(mode_queries=mode_queries,maximum_mode_absolute_error=mode_error) if mode_context is not None else {}),
            relative_tolerance=1e-4,absolute_tolerance=1e-5,
            scope='Training-only numerical transport; no screening qualification')


def cuda_backend():
    """Fail before cohort assembly when a GPU worker lacks its bound device."""
    import os,torch
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8',':16:8'):
        raise ValueError('CUDA training needs its deterministic cuBLAS workspace configured before launch')
    engine=Backend('cuda')
    # Application VRAM cap; it does not imply exclusive GPU allocation.
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.,4096*1024**2/total),0)
    return engine


def child_environment():
    """Pass managed device controls only to the training subprocess."""
    import os
    required=os.environ.get('RADAR_PREPARATION_TRAINING_DEVICE')
    if required is None:return {}
    if required not in ('cpu','cuda'):raise ValueError('Unknown managed structural training device')
    result={'RADAR_PREPARATION_TRAINING_DEVICE':required}
    if required=='cuda':
        workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG')
        if workspace not in (':4096:8',':16:8'):
            raise ValueError('CUDA training needs its deterministic cuBLAS workspace configured before launch')
        result['CUBLAS_WORKSPACE_CONFIG']=workspace
        for name in ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER'):
            value=os.environ.get(name)
            if value is not None:
                if not 0<len(value)<=512 or any(ord(c)<32 for c in value):
                    raise ValueError('Invalid bounded CUDA device binding')
                result[name]=value
    return result


def select(destination,*,segmented=False,transferred=False,continuation=None):
    """Choose hardware once; retain CPU worker transport and existing owners."""
    import torch
    destination=Path(destination)
    cpu=destination/'training-checkpoint.npz'
    mps=destination/'training-mps-checkpoint.npz'
    has_cpu=cpu.exists() or cpu.is_symlink()
    has_mps=mps.exists() or mps.is_symlink()
    cuda=destination/'training-cuda-checkpoint.npz';has_cuda=cuda.exists() or cuda.is_symlink()
    if sum((has_cpu,has_mps,has_cuda))>1:
        raise ValueError('Conflicting structural training device owners')
    import os
    required=os.environ.get('RADAR_PREPARATION_TRAINING_DEVICE')
    if required not in (None,'cpu','cuda'):raise ValueError('Unknown managed structural training device')
    resume_device=None
    if continuation is not None:
        from .structural_training_transfer import read,CUDA_SCHEMA
        receipt,_=read(continuation)
        resume_device='cuda' if receipt['schema']==CUDA_SCHEMA else 'cpu'
        if required is not None and resume_device!=required:
            raise ValueError('A transferred optimizer cannot move silently to another device')
    if required=='cuda' or has_cuda or resume_device=='cuda':
        if has_cuda and required=='cpu':raise ValueError('An owned CUDA optimizer cannot move silently to CPU')
        if has_cpu or has_mps:raise ValueError('An owned optimizer cannot move silently to CUDA')
        return cuda_backend()
    if required=='cpu':
        if has_mps:raise ValueError('An owned optimizer cannot move silently to CPU')
        return Backend('cpu')
    available=(sys.platform=='darwin' and torch.backends.mps.is_available()
        and __import__('os').environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')=='0')
    if has_mps:
        if not available or segmented or transferred:
            raise ValueError('Resume this owned MPS training state on its original local device')
        return Backend('mps')
    # Numerical worker continuations are intentionally CPU-owned. They use
    # their existing transfer audit and never reinterpret a device envelope.
    if has_cpu or segmented or transferred or not available:return Backend('cpu')
    return Backend('mps')
