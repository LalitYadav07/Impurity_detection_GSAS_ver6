"""Distinguish any compatible row hit from survival of each native SG label.

These are diagnostic counts, not probabilities or experimental exclusions.
An ambiguous row may be an any-hit success while losing a valid alternative.
"""
import math
import numpy as np
from ..symmetry.contracts import fields
from .structural_replay_limits import MAX_ROWS

SCHEMA='radar-compatible-label-retention/v1'

def _classes(classes):
    if (not isinstance(classes,list) or not 1<=len(classes)<=230
            or any(type(c)is not int or not 1<=c<=230 for c in classes)
            or classes!=sorted(set(classes))):raise ValueError('Complete ordered native class inventory required')

class LabelRetention:
    def __init__(self,classes,top_k):
        _classes(classes)
        if type(top_k)is not int or not 1<=top_k<=230:raise ValueError('A bounded ranking depth is required')
        self.classes=tuple(classes);self.requested=top_k;self.k=min(top_k,len(classes))
        self.targets=np.zeros(len(classes),np.int64);self.retained=self.targets.copy()
        self.rows=self.any_rows=self.all_rows=0

    def add(self,compatible,order):
        y=np.asarray(compatible);rank=np.asarray(order)
        if (y.dtype!=bool or y.ndim!=2 or not 1<=len(y)<=256 or y.shape[1]!=len(self.classes)
                or not y.any(1).all() or rank.dtype.kind not in 'iu' or rank.shape!=y.shape
                or not np.array_equal(np.sort(rank,axis=1),np.broadcast_to(np.arange(len(self.classes)),y.shape))):
            raise ValueError('Every bounded scored row needs all compatible labels and a complete ranking')
        kept=np.zeros_like(y);kept[np.arange(len(y))[:,None],rank[:,:self.k]]=True
        retained=kept&y
        self.rows+=len(y);self.any_rows+=int(retained.any(1).sum());self.all_rows+=int((kept|~y).all(1).sum())
        self.targets+=y.sum(0);self.retained+=retained.sum(0)

    def finish(self):
        rates=self.retained[self.targets>0]/self.targets[self.targets>0]
        return validate(dict(schema=SCHEMA,classes=list(self.classes),rows=self.rows,requested_top_k=self.requested,top_k=self.k,
            compatible_labels=int(self.targets.sum()),retained_compatible_labels=int(self.retained.sum()),
            any_compatible_rows=self.any_rows,all_compatible_rows=self.all_rows,
            any_compatible_hit_rate=self.any_rows/self.rows if self.rows else None,
            all_compatible_retention=self.all_rows/self.rows if self.rows else None,
            macro_label_retention=float(rates.mean()) if len(rates) else None,
            worst_class_label_retention=float(rates.min()) if len(rates) else None,
            per_class={str(c):dict(target_rows=int(n),retained_target_rows=int(r),retention=float(r/n) if n else None)
                for c,n,r in zip(self.classes,self.targets,self.retained)},
            experimental_coverage_qualified=False,candidate_pruning_authorized=False))

def _rate(actual,expected):
    if expected is None:return actual is None
    return (type(expected) in (int,float) and math.isfinite(expected)
        and type(actual)is float and math.isfinite(actual)
        and math.isclose(actual,expected,rel_tol=0,abs_tol=4*np.finfo(float).eps))

def validate(value,*,classes=None,rows=None,class_rows=None):
    d=fields(value,['schema','classes','rows','requested_top_k','top_k','compatible_labels','retained_compatible_labels',
        'any_compatible_rows','all_compatible_rows','any_compatible_hit_rate','all_compatible_retention',
        'macro_label_retention','worst_class_label_retention','per_class',
        'experimental_coverage_qualified','candidate_pruning_authorized'])
    _classes(d['classes']);n=d['rows'];width=len(d['classes'])
    if (d['schema']!=SCHEMA or classes is not None and d['classes']!=classes
            or type(n)is not int or not 0<=n<=MAX_ROWS or rows is not None and n!=rows
            or type(d['requested_top_k'])is not int or not 1<=d['requested_top_k']<=230
            or type(d['top_k'])is not int or d['top_k']!=min(d['requested_top_k'],width)
            or any(type(d[k])is not int for k in ['compatible_labels','retained_compatible_labels','any_compatible_rows','all_compatible_rows'])
            or not n<=d['compatible_labels']<=n*width
            or not 0<=d['retained_compatible_labels']<=d['compatible_labels']
            or not 0<=d['all_compatible_rows']<=d['any_compatible_rows']<=n
            or not d['any_compatible_rows']<=d['retained_compatible_labels']<=d['any_compatible_rows']*d['top_k']
            or (d['all_compatible_rows']==n)!=(d['retained_compatible_labels']==d['compatible_labels'])
            or d['top_k']==width and d['retained_compatible_labels']!=d['compatible_labels']
            or not _rate(d['any_compatible_hit_rate'],d['any_compatible_rows']/n if n else None)
            or not _rate(d['all_compatible_retention'],d['all_compatible_rows']/n if n else None)
            or not isinstance(d['per_class'],dict) or set(d['per_class'])!={str(c) for c in d['classes']}
            or d['experimental_coverage_qualified'] is not False or d['candidate_pruning_authorized'] is not False):
        raise ValueError('Invalid compatible-label retention scope or counts')
    if class_rows is not None and (not isinstance(class_rows,list) or len(class_rows)!=width
            or any(type(v)is not int or not 0<=v<=n for v in class_rows)):
        raise ValueError('Complete native class denominators required')
    totals=retained=0;rates=[]
    for i,c in enumerate(d['classes']):
        r=fields(d['per_class'][str(c)],['target_rows','retained_target_rows','retention'])
        count=r['target_rows'];kept=r['retained_target_rows']
        if (type(count)is not int or not 0<=count<=n or class_rows is not None and count!=class_rows[i]
                or type(kept)is not int or not 0<=kept<=count
                or not _rate(r['retention'],kept/count if count else None)):
            raise ValueError('Per-label retention changed a native class denominator')
        totals+=count;retained+=kept
        if count:rates.append(kept/count)
    if (totals!=d['compatible_labels'] or retained!=d['retained_compatible_labels']
            or not _rate(d['macro_label_retention'],float(np.mean(rates)) if rates else None)
            or not _rate(d['worst_class_label_retention'],float(np.min(rates)) if rates else None)):
        raise ValueError('Compatible-label retention lost its complete class inventory')
    return d

def validate_bundle(value,classes,*,rows=None,class_rows=None,any_top1=None,any_top5=None):
    d=fields(value,['top1','top5'])
    for key,depth,any_rate in [('top1',1,any_top1),('top5',5,any_top5)]:
        row=validate(d[key],classes=classes,rows=rows,class_rows=class_rows)
        if row['requested_top_k']!=depth or any_rate is not None and not _rate(row['any_compatible_hit_rate'],any_rate):
            raise ValueError('Ranking-depth retention differs from the same scored row hits')
    if d['top1']['rows']!=d['top5']['rows'] or d['top1']['compatible_labels']!=d['top5']['compatible_labels']:
        raise ValueError('Ranking depths cannot use different scored populations')
    return d

def validate_metrics(metrics,*,classes=None,rows=None,class_rows=None):
    if not isinstance(metrics,dict):raise ValueError('Checked native validation metrics required')
    if 'label_retention' not in metrics:return metrics
    table=metrics.get('per_class')
    if not isinstance(table,dict):raise ValueError('Native class denominators required for label-retention metrics')
    if classes is None:
        try:classes=sorted(map(int,table))
        except (ValueError,TypeError) as error:raise ValueError('Invalid native class metric inventory') from error
    _classes(classes)
    if set(table)!={str(c) for c in classes}:raise ValueError('Label retention changed the full native class inventory')
    recorded=[table[str(c)].get('rows') if isinstance(table[str(c)],dict) else None for c in classes]
    if class_rows is not None and recorded!=class_rows:raise ValueError('Label-retention metrics differ from the native role counts')
    validate_bundle(metrics['label_retention'],classes,rows=rows,class_rows=recorded,
        any_top1=metrics.get('top1'),any_top5=metrics.get('top5'))
    return metrics
