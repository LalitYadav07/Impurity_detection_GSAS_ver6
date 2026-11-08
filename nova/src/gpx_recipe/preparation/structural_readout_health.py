"""Detect a dead fixed readout without treating activity as model accuracy."""
import numpy as np

from .structural_training_data import batches
from .structural_training_backend import forward as training_forward


def forward(model,values,*,_embedding_sink=None):
    """One actual forward pass, inspecting the final linear layer's input."""
    import torch
    captured=[]
    def inspect(module,args):captured.append(args[0])
    handle=model.head[-1].register_forward_pre_hook(inspect)
    try:logits=training_forward(model,values)
    finally:handle.remove()
    if (len(captured)!=1 or captured[0].ndim!=2 or captured[0].shape[0]!=len(values)
            or not torch.isfinite(captured[0]).all() or not torch.isfinite(logits).all()):
        raise ValueError('The fixed attention readout returned invalid activations')
    if _embedding_sink is not None:_embedding_sink(captured[0])
    return logits,torch.any(captured[0]!=0,dim=1)


def training_health(model,rows,indices,mean,scale):
    """Check training-only representatives; no held-out example is used.

    An active hidden layer does not prove discrimination, calibration or
    generalization. An entirely zero one cannot condition logits on a pattern.
    Preserve train/eval state and consume no dropout/random-number stream.
    """
    import torch
    previous=model.training;active=0;count=0;first=None;difference=0.
    try:
        model.eval()
        with torch.no_grad():
            for _,values in batches(rows,indices,size=64):
                logits,present=forward(model,torch.from_numpy((values-mean)/scale))
                a=logits.numpy()
                if first is None:first=a[0].copy()
                difference=max(difference,float(np.max(abs(a-first))))
                active+=int(present.sum());count+=len(values)
    finally:model.train(previous)
    if not count:raise ValueError('Training readout checks need training representatives')
    return dict(schema='radar-attention-training-readout-health/v1',
        representatives=count,active_readouts=active,inactive_readouts=count-active,
        maximum_logit_difference=difference,readout_active=active>0,
        scope='Training-only numerical activity; no accuracy or scientific qualification claim')
