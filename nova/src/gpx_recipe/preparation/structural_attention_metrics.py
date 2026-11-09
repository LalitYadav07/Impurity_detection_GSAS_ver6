"""Set-valued family recall, including rare families, from bounded batches."""
import numpy as np

from .structural_training_data import batches
from .structural_training_backend import forward
from .structural_compatible_objective import OBJECTIVE,terms


def evaluate(model,rows,targets,indices,mean,scale,classes,*,prior=None,clean=None,_logits_sink=None,
             _mode_context=None,_mode_interval_evaluation=None,_groups=None):
    import torch
    from .structural_observation_metrics import ObservationMetrics
    from .structural_label_retention import LabelRetention
    labels={key:LabelRetention(classes.tolist(),k) for key,k in [('top1',1),('top5',5)]}
    observations=None if clean is None else ObservationMetrics(clean,len(rows),len(classes),class_values=classes.tolist())
    counts=np.zeros(len(classes),np.int64);one=np.zeros(len(classes));five=one.copy()
    total=correct_one=correct_five=0;loss_sum=0.;class_nll=np.zeros(len(classes),np.float64)
    if _mode_interval_evaluation is not None and _mode_context is None:
        raise ValueError('A class-only model has no conditional mode intervals to evaluate')
    modes=None if _mode_context is None else _mode_context.metrics(interval_evaluation=_mode_interval_evaluation,groups=_groups)
    if modes is not None and prior is not None:raise ValueError('A class prior has no learned mode readout')
    for selected,values in batches(rows,indices):
        allowed=targets[selected]
        if prior is None:
            with torch.no_grad():
                inputs=torch.from_numpy((values-mean)/scale)
                if modes is None:logits=forward(model,inputs)
                else:
                    from .structural_joint_modes import shared
                    logits,embedding=shared(model,inputs)
                    modes.add(model,embedding,selected)
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
        for measured in labels.values():measured.add(allowed,order)
        if observations is not None:observations.add(selected,allowed,hit_one,hit_five,order=order)
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
    result['label_retention']={key:measured.finish() for key,measured in labels.items()}
    if modes is not None:result['mode_validation']=modes.finish()
    return result
