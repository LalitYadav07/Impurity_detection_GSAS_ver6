"""Adaptive windows with a deterministic gather-only adjoint on CUDA.

Native CUDA adaptive max/average pooling backward uses atomic additions and
rejects deterministic mode. The same floor/ceiling windows and first-maximum
rule are retained here. The adjoint gathers contributing windows per input
coordinate, so no floating-point atomic scatter is required.
"""
from functools import lru_cache


@lru_cache(maxsize=32)
def contributors(length,tokens):
    import torch
    rows=[[] for _ in range(length)]
    for token in range(tokens):
        start=token*length//tokens;end=((token+1)*length+tokens-1)//tokens
        for index in range(start,end):rows[index].append(token)
    width=max(map(len,rows));indices=torch.zeros((length,width),dtype=torch.int64)
    valid=torch.zeros((length,width),dtype=torch.bool)
    for index,row in enumerate(rows):
        indices[index,:len(row)]=torch.tensor(row,dtype=torch.int64);valid[index,:len(row)]=True
    return indices,valid


def adaptive(values,tokens,*,maximum=False):
    import torch
    from .structural_attention_pool import _windows
    length=values.shape[-1]
    positions,valid,counts=_windows(length,tokens)
    positions=positions.to(values.device);valid=valid.to(values.device);counts=counts.to(values.device)
    indices,present=contributors(length,tokens)
    indices=indices.to(values.device);present=present.to(values.device)

    class WindowPool(torch.autograd.Function):
        @staticmethod
        def forward(ctx,source):
            selected=source[:,:,positions]
            if maximum:
                result,local=selected.masked_fill(~valid,float('-inf')).max(-1)
                selected_indices=positions[None,None,:,:].expand(source.shape[:2]+positions.shape).gather(-1,local[...,None]).squeeze(-1)
                ctx.save_for_backward(selected_indices)
                return result
            return selected.masked_fill(~valid,0.).sum(-1)/counts.to(source.dtype)

        @staticmethod
        def backward(ctx,gradient):
            contributions=gradient[:,:,indices]
            mask=present[None,None,:,:]
            if maximum:
                chosen,=ctx.saved_tensors
                locations=torch.arange(length,device=gradient.device)[None,None,:,None]
                mask=mask & (chosen[:,:,indices]==locations)
            else:contributions=contributions/counts[indices].to(gradient.dtype)
            return contributions.masked_fill(~mask,0.).sum(-1)

    return WindowPool.apply(values)
