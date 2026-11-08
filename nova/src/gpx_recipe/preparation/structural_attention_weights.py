"""Fixed attention architecture, transported as numerical data, never code."""
import hashlib
import json

import numpy as np

SCHEMA='radar-three-curve-structural-ranking-model/v1'
LEGACY_ARCHITECTURE='parent-q-convolution-attention-3-24-32-t128-h4/v1'
NORMALIZED_ARCHITECTURE='parent-q-convolution-attention-3-24-32-t128-h4-normalized-readout/v2'
ARCHITECTURE='parent-q-convolution-linear-attention-3-24-32-full-local-h4-readout128/v3'
ARCHITECTURES=frozenset({LEGACY_ARCHITECTURE,NORMALIZED_ARCHITECTURE,ARCHITECTURE})
REPRESENTATION='reference-current-signed-difference/shared-Q-asinh/v1'
REFERENCE_KIND='calculated_parent_at_shared_parent_compatible_cell'
NORMALIZATION='training-channel-standard-deviation/no-centering/v1'


def _layout(model):
    """Derive layout from installed code; a saved file cannot choose tensors."""
    state=model.state_dict()
    layout=[(name,list(value.shape)) for name,value in state.items()]
    digest=hashlib.sha256(json.dumps(layout,separators=(',',':')).encode()).hexdigest()
    return state,digest


def pack(model):
    state,digest=_layout(model)
    values=[]
    for tensor in state.values():
        value=tensor.detach().cpu().numpy()
        if value.dtype!=np.float32 or not np.isfinite(value).all():
            raise ValueError('Finite float32 attention parameters required')
        values.append(value.ravel())
    plane=np.concatenate(values)
    if plane.nbytes>8*1024**2:
        raise ValueError('Attention parameters exceed the fixed portable bound')
    return plane,digest


def restore(model,plane,layout_sha256):
    import torch
    state,digest=_layout(model)
    count=sum(value.numel() for value in state.values())
    if (digest!=layout_sha256 or type(plane) is not np.ndarray
            or plane.dtype!=np.float32 or plane.shape!=(count,)
            or plane.nbytes>8*1024**2 or not np.isfinite(plane).all()):
        raise ValueError('Attention parameter plane differs from the installed architecture')
    position=0;values={}
    for name,tensor in state.items():
        size=tensor.numel()
        values[name]=torch.from_numpy(plane[position:position+size].copy().reshape(tensor.shape))
        position+=size
    model.load_state_dict(values,strict=True);model.eval()
