"""Conditional mode-family amplitudes on the shared three-curve embedding.

This is a trainable readout and fitting-only target normalization. Its outputs
are exploratory intervals until independent sensitivity calibration and native
checks qualify them. Scores, missing targets and small predictions cannot
establish experimental absence or authorize candidate exclusion.
"""
from dataclasses import dataclass
import hashlib

import numpy as np

from .structural_family_targets import TargetRows
from .symmetry_mode_normalization import transform,physical

SCHEMA='radar-conditional-mode-family-readout/v1'
OBJECTIVE='class-balanced-conditional-interval-pinball-10-90/v1'
SCALE_METHOD='fitting-distinct-alias-family-amplitude-envelope/v2'
MAX_FAMILIES=4096
MAX_QUERIES=16*1024**2//40


def _fitting_scale(values):
    """Condition one family's loss without deleting small native targets.

    A median of positive amplitudes can be dominated by roundoff-sized values
    in numerical null components, leaving ordinary amplitudes tens of asinh
    units away. Use the complete fitting envelope instead. Every fitting
    endpoint then lies in [0, asinh(1)]; held-out endpoints are never clipped.
    Unknown targets remain absent, and an all-zero family keeps unit scale.
    """
    values=np.asarray(values)
    if values.dtype!=np.float64 or values.ndim!=1 or not np.isfinite(values).all() or np.any(values<0):
        raise ValueError('Finite nonnegative native float64 fitting endpoints required')
    maximum=float(values.max(initial=0.))
    return maximum if maximum>0 else 1.


@dataclass(frozen=True)
class Normalization:
    scales:np.ndarray
    known:np.ndarray
    metadata:dict

    def targets(self,table,indices):
        if not isinstance(table,TargetRows) or table.metadata['identity']!=self.metadata['targets_identity']:
            raise ValueError('Mode normalization belongs to a different exact target table')
        result=table.batch(indices)
        result['lower']=transform(result['lower'],self.scales[result['family']])
        result['upper']=transform(result['upper'],self.scales[result['family']])
        return result

    def decode(self,values,families):
        values=np.asarray(values);families=np.asarray(families)
        if (values.ndim!=2 or values.shape[1]!=2 or families.dtype.kind not in 'iu'
                or families.shape!=(len(values),) or np.any(families<0) or np.any(families>=len(self.scales))
                or np.any(values[:,0]>values[:,1])):
            raise ValueError('Ordered complete mode bounds and declared family indices required')
        return physical(values,self.scales[families,None])


def fit(table,cohort):
    """Use distinct fitting aliases only; preserve validation/test targets.

    Multi-class training uses the same fixed independent calibration reservation
    as the shared classifier. A single-class mode control keeps its original
    fitting fold; it does not invent another class or a calibration certificate.
    """
    if not isinstance(table,TargetRows):raise ValueError('Checked native family targets required')
    table.verify(cohort)
    families=table.metadata['families'];classes=table.metadata['classes']
    if not 1<=len(families)<=MAX_FAMILIES:raise ValueError('A complete bounded nonempty family inventory is required')
    split=np.asarray(cohort['split']);reservation=None
    if len(classes)>1:
        from .structural_calibration_partition import reserve
        provenance=cohort['example_simulations']
        clean=np.array([any(v.get('kind')=='clean_native' for v in values) for values in provenance],bool)
        reservation=reserve(cohort['groups'],split,clean,cohort['compatible_labels'])
        split=reservation.split
    selected=np.flatnonzero(split==0)
    if not len(selected):raise ValueError('Mode normalization needs original fitting rows')
    a=table.arrays;aliases=np.unique(a['row_group'][selected]);offsets=a['offsets']
    locations=np.concatenate([np.arange(int(offsets[g]),int(offsets[g+1]),dtype=np.int64) for g in aliases])
    c=a['space_group'][locations];f=a['family'][locations];low=a['lower'][locations];high=a['upper'][locations]
    if not len(locations):raise ValueError('No known fitting mode intervals are available')
    shape=(len(classes),len(families));counts=np.zeros(shape,np.int64);positive=counts.copy();zero=counts.copy()
    np.add.at(counts,(c,f),1);np.add.at(positive,(c[low>0],f[low>0]),1);np.add.at(zero,(c[high==0],f[high==0]),1)
    scales=np.ones(len(families),np.float64)
    for family in range(len(families)):
        scales[family]=_fitting_scale(high[f==family])
    if not np.isfinite(scales).all() or np.any(scales<=0):raise ValueError('Finite positive fitting-only mode scales required')
    metadata=dict(schema=SCHEMA,targets_identity=table.metadata['identity'],
        classes=list(classes),families=[v['id'] for v in families],normalization=table.metadata['normalization'],
        target_scale_method=SCALE_METHOD,
        fitting_split_sha256=hashlib.sha256(split.tobytes()).hexdigest(),fitting_rows=len(selected),
        fitting_alias_groups=len(aliases),scales=scales.tolist(),known_intervals=counts.tolist(),
        positive_intervals=positive.tolist(),exact_zero_intervals=zero.tolist(),
        calibration_reserved=bool(reservation is not None and reservation.metadata['available']),
        mode_head_trained=False,experimental_mode_absence_qualified=False,candidate_pruning_authorized=False)
    scales=np.frombuffer(scales.tobytes(),dtype=np.float64)
    known=np.frombuffer((counts>0).tobytes(),dtype=bool).reshape(shape)
    return Normalization(scales,known,metadata)


def network(class_count,family_count):
    """A sparse query head; callers supply the existing attention embedding."""
    import torch
    from torch import nn
    if (type(class_count)is not int or not 1<=class_count<=230
            or type(family_count)is not int or not 1<=family_count<=MAX_FAMILIES):
        raise ValueError('Complete bounded class and mode-family inventories required')

    class Conditional(nn.Module):
        def __init__(self):
            super().__init__();self.symmetry=nn.Embedding(class_count,8)
            self.family=nn.Embedding(family_count,16)
            self.readout=nn.Sequential(nn.Linear(88,64),nn.GELU(),nn.Linear(64,2))

        def forward(self,embedding,row,space_group,family):
            queries=(row,space_group,family)
            if (embedding.ndim!=2 or embedding.shape[1]!=64 or not 1<=len(embedding)<=256
                    or any(v.dtype!=torch.int64 or v.ndim!=1 or v.shape!=row.shape
                           or v.device!=embedding.device for v in queries)
                    or len(row)>MAX_QUERIES or not torch.isfinite(embedding).all()
                    or any(bool(((v<0)|(v>=limit)).any()) for v,limit in
                           zip(queries,(len(embedding),class_count,family_count)))):
                raise ValueError('Finite shared embeddings and complete bounded sparse mode queries required')
            pieces=[]
            for first in range(0,len(row),2048):
                indices=slice(first,first+2048)
                values=torch.cat((embedding[row[indices]],self.symmetry(space_group[indices]),
                    self.family(family[indices])),dim=1)
                raw=self.readout(values)
                lower=torch.nn.functional.softplus(raw[:,0]);width=torch.nn.functional.softplus(raw[:,1])
                pieces.append(torch.stack((lower,lower+width),dim=1))
            return torch.cat(pieces) if pieces else embedding.new_empty((0,2))

    return Conditional()


def loss(prediction,targets,*,rows,class_count):
    """Equal class mass per row; missing targets never become negatives.

    Bounds stay float64 in the objective, including values below float32's
    range. Outputs may be float32; autograd retains their connection. Targets
    describe native ambiguity, not a claimed calibrated prediction interval.
    """
    import torch
    if (type(rows)is not int or not 1<=rows<=256 or type(class_count)is not int or not 1<=class_count<=230
            or prediction.ndim!=2 or prediction.shape!=(len(targets['row']),2)
            or len(prediction)>MAX_QUERIES or not torch.isfinite(prediction).all()
            or bool((prediction[:,0]<0).any()) or bool((prediction[:,1]<prediction[:,0]).any())):
        raise ValueError('Finite ordered conditional mode predictions required')
    row=np.asarray(targets['row']);classes=np.asarray(targets['space_group'])
    lo=np.asarray(targets['lower']);hi=np.asarray(targets['upper'])
    if (any(v.shape!=(len(prediction),) for v in (row,classes,lo,hi))
            or row.dtype.kind not in 'iu' or classes.dtype.kind not in 'iu'
            or np.any(row<0) or np.any(row>=rows) or np.any(classes<0) or np.any(classes>=class_count)
            or lo.dtype!=np.float64 or hi.dtype!=np.float64 or not np.isfinite(lo).all() or not np.isfinite(hi).all()
            or np.any(lo<0) or np.any(hi<lo)):
        raise ValueError('Complete native transformed intervals and conditional row/class labels required')
    if not len(prediction):return prediction.sum()*0.
    # MPS does not implement float64. Transfer this small query result to the
    # existing CPU objective, keeping autograd, rather than narrowing native
    # targets or enabling an implicit device fallback.
    if prediction.device.type=='mps':prediction=prediction.cpu()
    truth=torch.from_numpy(np.column_stack((lo,hi))).to(prediction.device)
    residual=truth-prediction.to(torch.float64)
    quantiles=prediction.new_tensor([.1,.9],dtype=torch.float64)
    per_query=torch.maximum(quantiles*residual,(quantiles-1)*residual).mean(1)
    key=torch.from_numpy((row*class_count+classes).astype(np.int64)).to(prediction.device)
    sums=per_query.new_zeros(rows*class_count).scatter_add_(0,key,per_query)
    counts=per_query.new_zeros(rows*class_count).scatter_add_(0,key,torch.ones_like(per_query))
    means=(sums/counts.clamp_min(1)).reshape(rows,class_count)
    present=(counts>0).reshape(rows,class_count)
    row_loss=means.sum(1)/present.sum(1).clamp_min(1)
    return row_loss[present.any(1)].mean()
