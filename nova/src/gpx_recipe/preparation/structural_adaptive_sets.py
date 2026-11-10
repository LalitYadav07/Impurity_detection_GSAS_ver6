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


def _work_counts(a,gaps,d,within_support):
    order=np.argsort(-a,axis=1,kind='stable');q=d['maximum_gap'];classes=a.shape[1]
    if not within_support or q is None:
        return order,np.full(len(a),classes,np.int64)
    counts=np.maximum(1,np.sum(gaps<=q,axis=1))
    ordered=np.take_along_axis(a,order,axis=1)
    breaks=(ordered[:,:-1]-ordered[:,1:])>-math.log(ADJACENT_SCORE_RATIO)
    positions=np.arange(1,classes)
    stops=np.where(breaks&(positions[None,:]>=counts[:,None]),positions[None,:],classes)
    return order,np.min(stops,axis=1)


def select(logits,calibration,*,identities,within_support=True):
    """Select a first work set, retaining the entire deferred work inventory."""
    d=validate(calibration)
    if identities!=d['identities']:raise ValueError('Adaptive selection belongs to different weights, parent, profile or dataset')
    if type(within_support)is not bool:raise ValueError('Adaptive selection needs an explicit domain check')
    a,gaps=_scores(logits,d['classes'])
    if len(a)!=1:raise ValueError('Select one arriving pattern at a time')
    orders,counts=_work_counts(a,gaps,d,within_support);order=orders[0];count=int(counts[0]);q=d['maximum_gap']
    if not within_support or q is None:
        reason=d['reason'] if within_support else 'Outside prepared support; retain every class'
    else:
        reason='Adaptive first work set; deferred classes remain available for native fitting'
    classes=[d['classes'][i] for i in order]
    return dict(schema=SCHEMA,retained_space_groups=classes[:count],deferred_space_groups=classes[count:],
        calibrated_maximum_gap=q,calibration_groups=len(d['groups']),error_rate=d['error_rate'],
        adjacent_score_ratio=ADJACENT_SCORE_RATIO,reason=reason,model_scores_are_probabilities=False,
        experimental_coverage_qualified=False,candidate_pruning_authorized=False,native_refinement_required=True)


def validate_work_set(value,classes):
    """A source-bound priority set never changes the native candidate scope."""
    d=fields(value,['schema','retained_space_groups','deferred_space_groups','calibrated_maximum_gap',
        'calibration_groups','error_rate','adjacent_score_ratio','reason','model_scores_are_probabilities',
        'experimental_coverage_qualified','candidate_pruning_authorized','native_refinement_required'])
    retained=d['retained_space_groups'];deferred=d['deferred_space_groups'];q=d['calibrated_maximum_gap']
    if (not isinstance(classes,list) or not 1<=len(classes)<=230
            or any(type(c)is not int or not 1<=c<=230 for c in classes) or len(set(classes))!=len(classes)
            or d['schema']!=SCHEMA or not isinstance(retained,list) or not retained or not isinstance(deferred,list)
            or any(type(c)is not int for c in retained+deferred) or retained+deferred!=classes
            or type(d['calibration_groups'])is not int or not 1<=d['calibration_groups']<=MAX_GROUPS
            or type(d['error_rate']) not in (int,float) or not math.isfinite(d['error_rate']) or not 0<d['error_rate']<1
            or type(d['adjacent_score_ratio'])is not float or d['adjacent_score_ratio']!=ADJACENT_SCORE_RATIO
            or (q is not None and (type(q) not in (int,float) or not math.isfinite(q) or q<0))
            or (q is None and deferred)
            or (q is not None and math.ceil((d['calibration_groups']+1)*(1-d['error_rate']))>d['calibration_groups'])
            or not isinstance(d['reason'],str) or not 1<=len(d['reason'])<=1000
            or d['model_scores_are_probabilities'] is not False or d['experimental_coverage_qualified'] is not False
            or d['candidate_pruning_authorized'] is not False or d['native_refinement_required'] is not True):
        raise ValueError('Adaptive priority must retain the complete ordered requested class inventory')
    return d


def requested_scope(value,classes):
    """Restrict a prepared priority to reviewed routes, never discard work."""
    full=value['retained_space_groups']+value['deferred_space_groups']
    d=validate_work_set(value,full)
    if not classes or classes!=[c for c in full if c in classes]:
        raise ValueError('Requested adaptive classes differ from the prepared ranking order')
    retained=[c for c in d['retained_space_groups'] if c in classes]
    if not retained:
        # Expanding an empty intersection schedules work; it does not invent a
        # favorable score or grant authority to reject the remaining classes.
        retained=classes[:1]
    scoped=dict(d,retained_space_groups=retained,deferred_space_groups=classes[len(retained):])
    return validate_work_set(scoped,classes)


def evaluate(logits,compatible,groups,calibration,*,identities):
    """Measure false exclusions and work-set sizes on untouched test groups."""
    d=validate(calibration);groups=_groups(groups);a,gaps=_scores(logits,d['classes']);y=np.asarray(compatible)
    if identities!=d['identities']:raise ValueError('Adaptive evaluation belongs to different model or physical identities')
    if y.dtype!=bool or y.shape!=a.shape or not y.any(1).all() or len(groups)!=len(a):
        raise ValueError('Adaptive evaluation needs complete test labels and physical groups')
    if {r['id'] for r in d['groups']}.intersection(groups):
        raise ValueError('Adaptive evaluation must not score its own calibration groups')
    order,sizes=_work_counts(a,gaps,d,True)
    kept=np.zeros(y.shape,bool)
    kept[np.arange(len(a))[:,None],order]=np.arange(a.shape[1])[None,:]<sizes[:,None]
    any_hit=np.any(kept&y,axis=1);all_hit=np.all(kept|~y,axis=1);group_hits={}
    for g,hit in zip(groups,all_hit):group_hits[g]=group_hits.get(g,True) and bool(hit)
    per_class={str(c):dict(target_rows=int(y[:,i].sum()),excluded_target_rows=int((y[:,i]&~kept[:,i]).sum()),
        retention=None if not y[:,i].any() else float(kept[y[:,i],i].mean())) for i,c in enumerate(d['classes'])}
    return dict(rows=len(a),physical_groups=len(group_hits),any_compatible_retention=float(any_hit.mean()),
        all_compatible_retention=float(all_hit.mean()),all_compatible_group_retention=sum(group_hits.values())/len(group_hits),
        excluded_compatible_labels=int((y&~kept).sum()),per_class=per_class,
        retained_size=dict(minimum=int(sizes.min()),median=float(np.median(sizes)),
            p95=float(np.quantile(sizes,.95)),maximum=int(sizes.max())),
        experimental_coverage_qualified=False,candidate_pruning_authorized=False)


def validate_evaluation(value,classes,*,rows,groups,class_rows):
    d=fields(value,['rows','physical_groups','any_compatible_retention','all_compatible_retention',
        'all_compatible_group_retention','excluded_compatible_labels','per_class','retained_size',
        'experimental_coverage_qualified','candidate_pruning_authorized'])
    _classes(classes)
    if (type(d['rows'])is not int or d['rows']!=rows or type(d['physical_groups'])is not int or d['physical_groups']!=groups
            or not 1<=groups<=rows or d['experimental_coverage_qualified'] is not False
            or d['candidate_pruning_authorized'] is not False
            or any(type(d[k]) not in (int,float) or not math.isfinite(d[k]) or not 0<=d[k]<=1
                   for k in ['any_compatible_retention','all_compatible_retention','all_compatible_group_retention'])
            or d['all_compatible_retention']>d['any_compatible_retention']
            or not isinstance(d['per_class'],dict) or set(d['per_class'])!={str(c) for c in classes}
            or type(d['excluded_compatible_labels'])is not int or d['excluded_compatible_labels']<0):
        raise ValueError('Adaptive test metrics differ from the untouched test denominator')
    excluded=0
    for i,c in enumerate(classes):
        r=fields(d['per_class'][str(c)],['target_rows','excluded_target_rows','retention'])
        n=r['target_rows'];e=r['excluded_target_rows'];rate=r['retention']
        if (type(n)is not int or n!=class_rows[i] or type(e)is not int or not 0<=e<=n
                or (n==0 and rate is not None)
                or (n>0 and (type(rate) not in (int,float) or not math.isfinite(rate)
                    or not math.isclose(rate,(n-e)/n,rel_tol=0,abs_tol=4*np.finfo(float).eps)))):
            raise ValueError('Adaptive class metrics differ from the native compatible-label counts')
        excluded+=e
    sizes=fields(d['retained_size'],['minimum','median','p95','maximum'])
    if (excluded!=d['excluded_compatible_labels']
            or type(sizes['minimum'])is not int or type(sizes['maximum'])is not int
            or any(type(sizes[k]) not in (int,float) or not math.isfinite(sizes[k]) for k in ['median','p95'])
            or not 1<=sizes['minimum']<=sizes['median']<=sizes['p95']<=sizes['maximum']<=len(classes)):
        raise ValueError('Adaptive work sizes or excluded-label totals are inconsistent')
    return d
