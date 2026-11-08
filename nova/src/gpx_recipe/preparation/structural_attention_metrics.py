"""Set-valued family recall, including rare families, from bounded batches."""
import numpy as np

from .structural_training_data import batches
from .structural_training_backend import forward
from .structural_compatible_objective import OBJECTIVE,terms


def evaluate(model,rows,targets,indices,mean,scale,classes,*,prior=None,clean=None,_logits_sink=None):
    import torch
    from .structural_observation_metrics import ObservationMetrics
    observations=None if clean is None else ObservationMetrics(clean,len(rows),len(classes))
    counts=np.zeros(len(classes),np.int64);one=np.zeros(len(classes));five=one.copy()
    total=correct_one=correct_five=0;loss_sum=0.;class_nll=np.zeros(len(classes),np.float64)
    for selected,values in batches(rows,indices):
        allowed=targets[selected]
        if prior is None:
            with torch.no_grad():
                logits=forward(model,torch.from_numpy((values-mean)/scale))
                permitted=torch.from_numpy(allowed)
                loss,negative=terms(logits,permitted)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss).all():
                raise ValueError('Nonfinite attention validation output')
            if _logits_sink is not None:_logits_sink(selected,logits.numpy().copy())
            loss_sum+=float(loss.sum());order=np.argsort(-logits.numpy(),axis=1,kind='stable')
            class_nll+=np.where(allowed,negative.numpy(),0.).sum(0,dtype=np.float64)
        else:order=np.broadcast_to(np.argsort(-prior,kind='stable'),allowed.shape)
        hits=np.take_along_axis(allowed,order,axis=1)
        hit_one=hits[:,0];hit_five=hits[:,:min(5,len(classes))].any(1)
        total+=len(selected);correct_one+=int(hit_one.sum());correct_five+=int(hit_five.sum())
        counts+=allowed.sum(0);one+=(allowed&hit_one[:,None]).sum(0);five+=(allowed&hit_five[:,None]).sum(0)
        if observations is not None:observations.add(selected,allowed,hit_one,hit_five)
    if not total or not counts.all():raise ValueError('Every scored family needs independent held-out rows')
    per_class={str(int(group)):dict(rows=int(counts[i]),top1=float(one[i]/counts[i]),top5=float(five[i]/counts[i]))
        for i,group in enumerate(classes)}
    result=dict(top1=correct_one/total,top5=correct_five/total,macro_top5=float(np.mean(five/counts)),
        worst_class_top5=float(np.min(five/counts)),per_class=per_class,loss=None if prior is not None else loss_sum/total)
    if prior is None:
        for i,group in enumerate(classes):per_class[str(int(group))]['compatible_nll']=float(class_nll[i]/counts[i])
        result.update(loss_method=OBJECTIVE,macro_compatible_nll=float(np.mean(class_nll/counts)),
            worst_class_compatible_nll=float(np.max(class_nll/counts)))
    if observations is not None:result['observation_metrics']=observations.finish()
    return result
