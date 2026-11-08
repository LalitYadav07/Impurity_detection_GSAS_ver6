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
        if kind not in ('cpu','mps'):raise ValueError('Unknown structural training execution device')
        self.kind=kind
        if kind=='mps':
            from . import structural_mps_checkpoint
            self.checkpoint=structural_mps_checkpoint
        else:self.checkpoint=cpu_checkpoint

    def place(self,model):
        return model if self.kind=='cpu' else model.to('mps')

    @contextmanager
    def random_state(self,seed):
        if self.kind=='cpu':
            yield
            return
        import torch
        before=torch.mps.get_rng_state().clone()
        try:
            torch.mps.manual_seed(seed)
            yield
        finally:
            torch.mps.synchronize()
            torch.mps.set_rng_state(before)

    def portable_model(self,model,rows,indices,mean,scale):
        """Audit GPU -> CPU logits on training representatives before export.

        This checks numerical transport, not accuracy or scientific admission.
        Held-out metrics are subsequently measured with this actual CPU model.
        """
        if self.kind=='cpu':return model,None
        import torch
        from .structural_training_data import batches
        cpu=copy.deepcopy(model).cpu().eval();largest=0.;count=0
        with torch.no_grad():
            for _,values in batches(rows,indices,size=64):
                inputs=torch.from_numpy((values-mean)/scale)
                actual=forward(model,inputs);expected=cpu(inputs)
                if (not torch.isfinite(actual).all() or not torch.isfinite(expected).all()
                        or not torch.allclose(actual,expected,rtol=1e-4,atol=1e-5)):
                    raise ValueError('GPU training weights do not reproduce portable CPU logits')
                largest=max(largest,float((actual-expected).abs().max()));count+=len(values)
        if not count:raise ValueError('CPU portability requires training representatives')
        return cpu,dict(device='mps',portable_inference_device='cpu',
            representatives=count,maximum_logit_absolute_error=largest,
            relative_tolerance=1e-4,absolute_tolerance=1e-5,
            scope='Training-only numerical transport; no screening qualification')


def select(destination,*,segmented=False,transferred=False):
    """Choose hardware once; retain CPU worker transport and existing owners."""
    import torch
    destination=Path(destination)
    cpu=destination/'training-checkpoint.npz'
    mps=destination/'training-mps-checkpoint.npz'
    has_cpu=cpu.exists() or cpu.is_symlink()
    has_mps=mps.exists() or mps.is_symlink()
    if has_cpu and has_mps:
        raise ValueError('Conflicting CPU and MPS structural training owners')
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
