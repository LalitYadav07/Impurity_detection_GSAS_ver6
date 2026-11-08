"""Reserve untouched physical groups before any model fitting or selection.

Original validation/test assignments never move. The reservation is determined
by group identity, not model scores. Unsupported reservations leave calibration
unavailable and keep the complete native search; they do not reshuffle groups
until an attractive split or accuracy is obtained.
"""
from dataclasses import dataclass
import hashlib
import math

import numpy as np

from ..symmetry.contracts import digest,fields
from .structural_adaptive_sets import MAX_GROUPS
from .structural_replay_limits import MAX_ROWS
from .tof_kernel_arrays import checksum

SCHEMA='radar-structural-independent-calibration-partition/v1'
METHOD='fixed-hash-original-training-group-reservation/v1'
SEED=8911319
MIN_CALIBRATION_GROUPS=20
MIN_CLEAN_FIT_PAIRS=5001


@dataclass(frozen=True)
class Partition:
    split:np.ndarray
    metadata:dict


def reserve(groups,split,clean,compatible):
    base=np.asarray(split);clean=np.asarray(clean);labels=np.asarray(compatible)
    if (not isinstance(groups,list) or not 3<=len(groups)<=MAX_ROWS
            or any(not isinstance(g,str) or not 1<=len(g)<=500 for g in groups)
            or base.dtype!=np.int8 or base.shape!=(len(groups),) or set(base)!={0,1,2}
            or clean.dtype!=bool or clean.shape!=base.shape
            or labels.dtype!=bool or labels.ndim!=2 or labels.shape[0]!=len(groups)
            or not 2<=labels.shape[1]<=230 or not labels.any(1).all()):
        raise ValueError('Independent calibration needs the complete original physical-group partition')
    assignment={}
    for group,code in zip(groups,base):
        if group in assignment and assignment[group]!=code:
            raise ValueError('Related trajectories cannot cross original holdouts')
        assignment[group]=int(code)
    if len(assignment)>MAX_GROUPS:raise ValueError('Calibration exceeds the independent-group inventory')
    original=sorted(g for g,c in assignment.items() if c==0)
    count=max(MIN_CALIBRATION_GROUPS,math.ceil(.2*len(original)))
    order=sorted(original,key=lambda g:hashlib.sha256(f'{METHOD}\0{SEED}\0{g}'.encode()).digest())
    proposal=order[:count]
    chosen=set(proposal)
    proposed=np.array([code==0 and g in chosen for g,code in zip(groups,base)],bool)
    remaining=(base==0)&~proposed
    reason=None
    if len(original)<=count:reason='Insufficient original training groups for independent calibration'
    elif int(clean[remaining].sum())<MIN_CLEAN_FIT_PAIRS:
        reason='The reservation would leave fewer than 5001 distinct clean fitting pairs'
    elif not labels[remaining&clean].any(0).all():
        reason='The fixed reservation would remove clean fitting support for a native class'
    result=base.copy()
    if reason is None:result[proposed]=3
    actual=sorted(chosen) if reason is None else []
    actual_set=set(actual)
    rows={name:int(np.count_nonzero(result==code)) for code,name in enumerate(['train','validation','test','calibration'])}
    metadata=dict(schema=SCHEMA,method=METHOD,seed=SEED,available=reason is None,
        reason=reason or 'Reserved before normalization, prototypes, model fitting and checkpoint selection',
        original_training_groups=len(original),proposed_groups=sorted(proposal),calibration_groups=actual,
        fitting_clean_pairs=int(clean[result==0].sum()),partition_rows=rows,
        original_split_sha256=hashlib.sha256(base.tobytes()).hexdigest(),
        assignment_sha256=digest(sorted((g,3 if g in actual_set else c) for g,c in assignment.items())),
        used_groups_sha256=digest(sorted(g for g in assignment if g not in actual_set)),
        minimum_calibration_groups=MIN_CALIBRATION_GROUPS,minimum_clean_fitting_pairs=MIN_CLEAN_FIT_PAIRS,
        original_validation_and_test_unchanged=True,experimental_coverage_qualified=False,candidate_pruning_authorized=False)
    return Partition(np.frombuffer(result.tobytes(),dtype=np.int8),validate(metadata))


def validate(value,*,dataset=None):
    d=fields(value,['schema','method','seed','available','reason','original_training_groups','proposed_groups',
        'calibration_groups','fitting_clean_pairs','partition_rows','original_split_sha256','assignment_sha256','used_groups_sha256',
        'minimum_calibration_groups','minimum_clean_fitting_pairs','original_validation_and_test_unchanged',
        'experimental_coverage_qualified','candidate_pruning_authorized'])
    available=d['available'];original=d['original_training_groups'];parts=d['partition_rows']
    if (d['schema']!=SCHEMA or d['method']!=METHOD or type(d['seed'])is not int or d['seed']!=SEED
            or type(available)is not bool or type(original)is not int or not 1<=original<=MAX_GROUPS
            or type(d['fitting_clean_pairs'])is not int or not 1<=d['fitting_clean_pairs']<=MAX_ROWS
            or d['minimum_calibration_groups']!=MIN_CALIBRATION_GROUPS
            or d['minimum_clean_fitting_pairs']!=MIN_CLEAN_FIT_PAIRS
            or not isinstance(d['reason'],str) or not 1<=len(d['reason'])<=500
            or any(not checksum(d[k]) for k in ['original_split_sha256','assignment_sha256','used_groups_sha256'])
            or d['original_validation_and_test_unchanged'] is not True
            or d['experimental_coverage_qualified'] is not False or d['candidate_pruning_authorized'] is not False
            or not isinstance(parts,dict) or set(parts)!={'train','validation','test','calibration'}
            or any(type(n)is not int or not 0<=n<=MAX_ROWS for n in parts.values())
            or not all(parts[k]>0 for k in ['train','validation','test'])
            or d['fitting_clean_pairs']>parts['train']):
        raise ValueError('Invalid independent calibration partition or authority')
    for key in ['proposed_groups','calibration_groups']:
        groups=d[key]
        if (not isinstance(groups,list) or any(not isinstance(g,str) or not 1<=len(g)<=500 for g in groups)
                or groups!=sorted(set(groups))):
            raise ValueError('Calibration needs complete unique physical-group identities')
    count=min(original,max(MIN_CALIBRATION_GROUPS,math.ceil(.2*original)))
    if (len(d['proposed_groups'])!=count or d['calibration_groups']!=(d['proposed_groups'] if available else [])
            or (available and (original<=count or d['fitting_clean_pairs']<MIN_CLEAN_FIT_PAIRS
                              or parts['calibration']<count))
            or (not available and parts['calibration']!=0)):
        raise ValueError('Calibration reservation differs from its fixed policy or fitting denominator')
    if dataset is not None:
        from .structural_training_counts import validate as validate_counts
        counts=validate_counts(dataset);p=counts['partitions']
        if (set(p)!=({'train','validation','test','calibration'} if available else {'train','validation','test'})
                or any(p[name]['rows']!=parts[name] for name in p)
                or p['train']['clean_pairs']!=d['fitting_clean_pairs']
                or p['train']['trajectory_groups']+(p['calibration']['trajectory_groups'] if available else 0)!=original
                or (available and p['calibration']['trajectory_groups']!=len(d['calibration_groups']))):
            raise ValueError('Model calibration assignment differs from its actual fitting/calibration counts')
    return d
