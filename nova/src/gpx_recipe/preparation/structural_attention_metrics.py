"""Set-valued family recall, including rare families, from bounded batches."""
import numpy as np

from .structural_training_data import batches


def evaluate(model,rows,targets,indices,mean,scale,classes,*,prior=None):
    import torch
    counts=np.zeros(len(classes),np.int64);one=np.zeros(len(classes));five=one.copy()
    total=correct_one=correct_five=0;loss_sum=0.
    for selected,values in batches(rows,indices):
        allowed=targets[selected]
        if prior is None:
            with torch.no_grad():
                logits=model(torch.from_numpy((values-mean)/scale))
                permitted=torch.from_numpy(allowed)
                loss=torch.logsumexp(logits,1)-torch.logsumexp(logits.masked_fill(~permitted,-torch.inf),1)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss).all():
                raise ValueError('Nonfinite attention validation output')
            loss_sum+=float(loss.sum());order=np.argsort(-logits.numpy(),axis=1,kind='stable')
        else:order=np.broadcast_to(np.argsort(-prior,kind='stable'),allowed.shape)
        hits=np.take_along_axis(allowed,order,axis=1)
        hit_one=hits[:,0];hit_five=hits[:,:min(5,len(classes))].any(1)
        total+=len(selected);correct_one+=int(hit_one.sum());correct_five+=int(hit_five.sum())
        counts+=allowed.sum(0);one+=(allowed&hit_one[:,None]).sum(0);five+=(allowed&hit_five[:,None]).sum(0)
    if not total or not counts.all():raise ValueError('Every scored family needs independent held-out rows')
    per_class={str(int(group)):dict(rows=int(counts[i]),top1=float(one[i]/counts[i]),top5=float(five[i]/counts[i]))
        for i,group in enumerate(classes)}
    return dict(top1=correct_one/total,top5=correct_five/total,macro_top5=float(np.mean(five/counts)),
        worst_class_top5=float(np.min(five/counts)),per_class=per_class,loss=None if prior is not None else loss_sum/total)
