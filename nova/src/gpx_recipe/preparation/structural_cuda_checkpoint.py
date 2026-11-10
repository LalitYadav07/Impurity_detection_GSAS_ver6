"""One explicitly bound CUDA device; no visible-host capacity assumption."""
import platform
import torch
from .structural_accelerator_checkpoint import Checkpoint

SCHEMA='radar-owned-cuda-training-continuation/v1'
STATE='training-cuda-checkpoint.npz'


def validate_runtime(value):
    expected={'device','torch','cuda','cudnn','machine','capability','name','deterministic','tf32'}
    if (not isinstance(value,dict) or set(value)!=expected or value['device']!='cuda'
            or any(not isinstance(value[k],str) or not 1<=len(value[k])<=256 for k in ('torch','cuda','machine','name'))
            or not isinstance(value['capability'],list) or len(value['capability'])!=2
            or any(type(v)is not int or not 0<=v<=100 for v in value['capability'])
            or type(value['cudnn'])is not int or value['cudnn']<1
            or value['deterministic']is not True or value['tf32']is not False):
        raise ValueError('Invalid bounded CUDA continuation runtime')
    return value


def runtime():
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError('CUDA training requires exactly one bound usable device')
    return validate_runtime(dict(device='cuda',torch=str(torch.__version__),cuda=str(torch.version.cuda),
        cudnn=torch.backends.cudnn.version(),machine=platform.machine(),
        capability=list(torch.cuda.get_device_capability(0)),name=torch.cuda.get_device_name(0),
        deterministic=True,tf32=False))


def require_model(model):
    runtime()
    if {p.device for p in model.parameters()}!={torch.device('cuda',0)}:
        raise ValueError('The complete model must use its one bound CUDA device')


def synchronize():torch.cuda.synchronize(0)
def get_rng():return torch.cuda.get_rng_state(0)
def set_rng(value):torch.cuda.set_rng_state(value,0)

import sys
_checkpoint=Checkpoint(sys.modules[__name__])
identity=_checkpoint.identity
owned_run=_checkpoint.owned_run
save=_checkpoint.save
restore=_checkpoint.restore
