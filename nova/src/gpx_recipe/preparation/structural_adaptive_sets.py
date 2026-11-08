"""Adaptive, group-calibrated work priorities; never an absence certificate.

Calibration uses the worst required score gap in each independent physical
trajectory, including every compatible target and observation variant. These
scores are model evidence, not probabilities of a space group being present.
Synthetic calibration alone cannot authorize pruning experimental candidates.
"""
import math

import numpy as np

from ..symmetry.contracts import fields,digest
from .tof_kernel_arrays import checksum

SCHEMA='radar-structural-adaptive-score-set/v1'
METHOD='whole-group-all-compatible-maximum-logit-gap/v1'
DEFAULT_ERROR_RATE=.05
# Conservative expansion of the calibrated set: never split a chain of
# adjacent classes with at least 80% of the preceding class's softmax score.
# This is a declared prioritization rule, not a calibrated confidence claim.
ADJACENT_SCORE_RATIO=.8
MAX_ROWS=1_000_000
MAX_GROUPS=4096


def _classes(values):
    if (not isinstance(values,list) or not 2<=len(values)<=230
            or any(type(v)is not int or not 1<=v<=230 for v in values)
            or values!=sorted(set(values))):
        raise ValueError('Adaptive scores need the exact ordered trained space-group classes')
    return values


def _scores(logits,classes):
    a=np.asarray(logits)
    if (a.dtype.kind!='f' or a.ndim!=2 or not 1<=len(a)<=MAX_ROWS
            or a.shape[1]!=len(classes) or not np.isfinite(a).all()):
        raise ValueError('Finite floating model logits on the complete class inventory required')
    a=a.astype(np.float64)
    with np.errstate(over='ignore',invalid='ignore'):
        gaps=np.max(a,axis=1,keepdims=True)-a
    if not np.isfinite(gaps).all():raise ValueError('Adaptive score gaps exceed finite numerical range')
    return a,gaps


def _groups(values):
    if (not isinstance(values,list) or not values or len(values)>MAX_ROWS
            or any(not isinstance(v,str) or not 1<=len(v)<=500 for v in values)):
        raise ValueError('Calibration rows need complete physical trajectory identities')
    return values


def calibrate(logits,compatible,groups,*,classes,identities,used_groups,error_rate=DEFAULT_ERROR_RATE):
    """Calibrate only after freezing the model on separate fitting groups.

    ``used_groups`` includes training, normalization, prototype construction,
    architecture/epoch selection and test groups. The producer must capture
    the true full inventory; file hashes do not independently prove it.
    Keeping the provenance and qualification flags explicit prevents this
    data-only helper being mistaken for permission to discard candidates.
    """
    _classes(classes);groups=_groups(groups);a,gaps=_scores(logits,classes)
    y=np.asarray(compatible)
    if y.dtype!=bool or y.shape!=a.shape or not y.any(1).all() or len(groups)!=len(a):
        raise ValueError('Every calibration row needs all of its compatible native labels')
    if (not isinstance(used_groups,list) or any(not isinstance(v,str) or not 1<=len(v)<=500 for v in used_groups)
            or set(groups).intersection(used_groups)):
        raise ValueError('Calibration groups must be independent of every fitting, selection and test group')
    identities=fields(identities,['weights_sha256','parent_sha256','instrument_sha256','dataset_sha256'])
    if any(not checksum(v) for v in identities.values()):raise ValueError('Adaptive calibration needs exact model and physical identities')
    if type(error_rate) not in (int,float) or not math.isfinite(error_rate) or not 0<error_rate<1:
        raise ValueError('A finite declared calibration error rate is required')
    row_scores=np.max(np.where(y,gaps,-np.inf),axis=1)
    by_group={}
    for group,score in zip(groups,row_scores):by_group[group]=max(by_group.get(group,0.),float(score))
    if len(by_group)>MAX_GROUPS:raise ValueError('Calibration exceeds the independent physical-group inventory')
    n=len(by_group);k=math.ceil((n+1)*(1-error_rate))
    coverage=y.any(0).tolist()
    threshold=None if k>n or not all(coverage) else sorted(by_group.values())[k-1]
    reason=('Insufficient independent calibration groups; retain every class' if k>n else
            'Some classes have no calibration targets; retain every class' if not all(coverage) else
            'Synthetic whole-group calibration; experimental exclusion is not qualified')
    return validate(dict(schema=SCHEMA,method=METHOD,classes=list(classes),identities=identities,
        error_rate=float(error_rate),groups=[dict(id=g,maximum_gap=by_group[g]) for g in sorted(by_group)],
        used_groups_sha256=digest(sorted(set(used_groups))),rows=len(groups),class_target_coverage=coverage,
        quantile_rank=k,maximum_gap=threshold,adjacent_score_ratio=ADJACENT_SCORE_RATIO,
        reason=reason,experimental_coverage_qualified=False,candidate_pruning_authorized=False))


def validate(value):
    d=fields(value,['schema','method','classes','identities','error_rate','groups','used_groups_sha256','rows',
        'class_target_coverage','quantile_rank','maximum_gap','adjacent_score_ratio','reason',
        'experimental_coverage_qualified','candidate_pruning_authorized'])
    _classes(d['classes'])
    ids=fields(d['identities'],['weights_sha256','parent_sha256','instrument_sha256','dataset_sha256'])
    rows=d['groups'];rate=d['error_rate'];coverage=d['class_target_coverage']
    if (d['schema']!=SCHEMA or d['method']!=METHOD or any(not checksum(v) for v in ids.values())
            or not checksum(d['used_groups_sha256']) or type(d['rows'])is not int or not 1<=d['rows']<=MAX_ROWS
            or type(rate) not in (int,float) or not math.isfinite(rate) or not 0<rate<1
            or type(d['adjacent_score_ratio'])is not float or d['adjacent_score_ratio']!=ADJACENT_SCORE_RATIO
            or d['experimental_coverage_qualified'] is not False or d['candidate_pruning_authorized'] is not False
            or not isinstance(d['reason'],str) or not 1<=len(d['reason'])<=500
            or not isinstance(coverage,list) or len(coverage)!=len(d['classes']) or any(type(v)is not bool for v in coverage)
            or not isinstance(rows,list) or not 1<=len(rows)<=min(d['rows'],MAX_GROUPS)):
        raise ValueError('Invalid adaptive calibration identity, scope or authority')
    groups=[];scores=[]
    for row in rows:
        r=fields(row,['id','maximum_gap']);score=r['maximum_gap']
        _groups([r['id']])
        if type(score) not in (int,float) or not math.isfinite(score) or score<0:
            raise ValueError('Every independent calibration group needs its complete finite worst-case gap')
        groups.append(r['id']);scores.append(score)
    k=math.ceil((len(rows)+1)*(1-rate))
    q=None if k>len(rows) or not all(coverage) else sorted(scores)[k-1]
    if (groups!=sorted(set(groups)) or type(d['quantile_rank'])is not int or d['quantile_rank']!=k
            or (d['maximum_gap'] is not None and type(d['maximum_gap']) not in (int,float))
            or d['maximum_gap']!=q):
        raise ValueError('Adaptive calibration cannot change its group denominator or finite-sample cutoff')
    return d


def select(logits,calibration,*,identities,within_support=True):
    """Select a first work set, retaining the entire deferred work inventory."""
    d=validate(calibration)
    if identities!=d['identities']:raise ValueError('Adaptive selection belongs to different weights, parent, profile or dataset')
    if type(within_support)is not bool:raise ValueError('Adaptive selection needs an explicit domain check')
    a,gaps=_scores(logits,d['classes'])
    if len(a)!=1:raise ValueError('Select one arriving pattern at a time')
    order=np.argsort(-a[0],kind='stable');q=d['maximum_gap']
    if not within_support or q is None:
        count=len(order);reason=d['reason'] if within_support else 'Outside prepared support; retain every class'
    else:
        count=int(np.count_nonzero(gaps[0,order]<=q))
        count=max(1,count)
        while count<len(order) and a[0,order[count-1]]-a[0,order[count]]<=-math.log(ADJACENT_SCORE_RATIO):
            count+=1
        reason='Adaptive first work set; deferred classes remain available for native fitting'
    classes=[d['classes'][i] for i in order]
    return dict(schema=SCHEMA,retained_space_groups=classes[:count],deferred_space_groups=classes[count:],
        calibrated_maximum_gap=q,calibration_groups=len(d['groups']),error_rate=d['error_rate'],
        adjacent_score_ratio=ADJACENT_SCORE_RATIO,reason=reason,model_scores_are_probabilities=False,
        experimental_coverage_qualified=False,candidate_pruning_authorized=False,native_refinement_required=True)


def evaluate(logits,compatible,groups,calibration,*,identities):
    """Measure false exclusions and work-set sizes on untouched test groups."""
    d=validate(calibration);groups=_groups(groups);a,_=_scores(logits,d['classes']);y=np.asarray(compatible)
    if y.dtype!=bool or y.shape!=a.shape or not y.any(1).all() or len(groups)!=len(a):
        raise ValueError('Adaptive evaluation needs complete test labels and physical groups')
    if {r['id'] for r in d['groups']}.intersection(groups):
        raise ValueError('Adaptive evaluation must not score its own calibration groups')
    kept=np.zeros(y.shape,bool);positions={c:i for i,c in enumerate(d['classes'])};sizes=[]
    for i,row in enumerate(a):
        result=select(row[None],d,identities=identities)
        retained=result['retained_space_groups'];sizes.append(len(retained))
        kept[i,[positions[c] for c in retained]]=True
    any_hit=np.any(kept&y,axis=1);all_hit=np.all(kept|~y,axis=1);group_hits={}
    for g,hit in zip(groups,all_hit):group_hits[g]=group_hits.get(g,True) and bool(hit)
    per_class={str(c):dict(target_rows=int(y[:,i].sum()),excluded_target_rows=int((y[:,i]&~kept[:,i]).sum()),
        retention=None if not y[:,i].any() else float(kept[y[:,i],i].mean())) for i,c in enumerate(d['classes'])}
    return dict(rows=len(a),physical_groups=len(group_hits),any_compatible_retention=float(any_hit.mean()),
        all_compatible_retention=float(all_hit.mean()),all_compatible_group_retention=sum(group_hits.values())/len(group_hits),
        excluded_compatible_labels=int((y&~kept).sum()),per_class=per_class,
        retained_size=dict(minimum=min(sizes),median=float(np.median(sizes)),
            p95=float(np.quantile(sizes,.95)),maximum=max(sizes)),
        experimental_coverage_qualified=False,candidate_pruning_authorized=False)
