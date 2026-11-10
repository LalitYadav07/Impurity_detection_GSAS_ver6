"""Native adaptive pooling with an exact MPS window implementation.

Some MPS versions require divisible lengths. Floor/ceiling windows preserve
the same average or first-maximum values and gradients for other lengths.
CPU operations retain the existing PyTorch implementation unchanged.
"""
from functools import lru_cache


@lru_cache(maxsize=32)
def _windows(length,tokens):
    import torch
    starts=torch.arange(tokens,dtype=torch.int64)*length//tokens
    ends=((torch.arange(tokens,dtype=torch.int64)+1)*length+tokens-1)//tokens
    counts=ends-starts
    offsets=torch.arange(int(counts.max()),dtype=torch.int64)
    positions=starts[:,None]+offsets[None,:]
    valid=positions<ends[:,None]
    return positions.clamp_max(length-1),valid,counts


def adaptive(values,tokens,*,maximum=False):
    import torch
    from torch.nn import functional as F
    if (values.ndim!=3 or type(tokens)is not int or tokens<1
            or type(maximum)is not bool or values.shape[-1]<1):
        raise ValueError('Adaptive attention pooling needs complete bounded curve tensors')
    native=F.adaptive_max_pool1d if maximum else F.adaptive_avg_pool1d
    if values.device.type=='cuda':
        from .structural_cuda_pool import adaptive as deterministic
        return deterministic(values,tokens,maximum=maximum)
    if values.device.type!='mps' or values.shape[-1]%tokens==0:
        return native(values,tokens)
    positions,valid,counts=_windows(values.shape[-1],tokens)
    positions=positions.to(values.device);valid=valid.to(values.device)
    selected=values[:,:,positions]
    if maximum:
        # max(dim) sends a tie's derivative to its first maximum, as native
        # adaptive_max_pool1d does; amax would distribute that derivative.
        return selected.masked_fill(~valid,float('-inf')).max(-1).values
    return selected.masked_fill(~valid,0.).sum(-1)/counts.to(values.device,values.dtype)
