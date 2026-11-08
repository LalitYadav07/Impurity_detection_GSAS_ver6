"""Class-supported grouped holdouts chosen without diffraction intensities.

Whole physical trajectories and their closed aliases must arrive with the
same group identity. Noise siblings cannot create independent class support.
This assignment is for new three-curve cohorts, never a repair of test scores.
"""
import numpy as np
from . import structural_replay_limits as limits


def partition(groups, labels, *, seed=69271):
    """Legacy bounded row API; keep its existing admission limit."""
    groups = list(groups)
    if not 3 <= len(groups) <= 131072:
        raise ValueError('Bounded physical groups and compatible class labels required')
    return partition_rows(groups, labels, seed=seed)


def partition_rows(groups, labels, *, seed=69271):
    """Keep 15% validation/test groups and require every class in each fold.

    Deterministic greedy set coverage preserves at least one independent
    training group per class. Failure requests more sampling; it neither
    drops classes nor relaxes coverage to manufacture a validation score.
    """
    groups=list(groups);labels=np.asarray(labels)
    if (type(seed) is not int or not 0<=seed<2**32
            or not 3<=len(groups)<=limits.MAX_ROWS or labels.ndim!=2
            or labels.shape[0]!=len(groups) or not 2<=labels.shape[1]<=230
            or labels.dtype!=bool or labels.nbytes>limits.MAX_LABEL_BYTES or not labels.any(1).all()
            or any(not isinstance(g,str) or not 1<=len(g)<=500 for g in groups)):
        raise ValueError('Bounded physical groups and compatible class labels required')
    unique=sorted(set(groups));n=len(unique)
    if n>4096:raise ValueError('New structural cohort exceeds the independent-group partition limit')
    positions={g:i for i,g in enumerate(unique)};coverage=np.zeros((n,labels.shape[1]),bool)
    for group,row in zip(groups,labels):coverage[positions[group]]|=row
    support=coverage.sum(0)
    if np.any(support<3):
        missing=np.flatnonzero(support<3).tolist()
        raise ValueError(f'Classes {missing} need at least three independent physical groups before holdout assignment')
    holdout=max(1,int(np.floor(.15*n+.5)))
    for attempt in range(64):
        rng=np.random.default_rng(np.random.SeedSequence([seed,attempt]))
        available=np.ones(n,bool);remaining=support.copy();folds=[];failed=False
        for reserve in (2,1):
            chosen=[];missing=np.ones(labels.shape[1],bool)
            while missing.any():
                if len(chosen)>=holdout:failed=True;break
                allowed=available & ((remaining-coverage)>=reserve).all(1)
                gain=(coverage*(missing/np.maximum(remaining,1))).sum(1)
                gain[~allowed]=-np.inf
                if not np.any(gain>0):failed=True;break
                best=np.flatnonzero(np.isclose(gain,np.max(gain),rtol=1e-12,atol=1e-15))
                index=int(rng.choice(best));chosen.append(index);available[index]=False
                remaining-=coverage[index];missing&=~coverage[index]
            if failed:break
            folds.append(chosen)
        if failed:continue
        # Cover both holdouts first. Extra members may consume only redundant
        # training support; original trajectory identities stay intact.
        for chosen in folds:
            while len(chosen)<holdout:
                allowed=np.flatnonzero(available & ((remaining-coverage)>=1).all(1))
                if not len(allowed):failed=True;break
                index=int(rng.choice(allowed));chosen.append(index);available[index]=False
                remaining-=coverage[index]
            if failed:break
        if failed:continue
        codes=np.zeros(n,np.int8);codes[folds[0]]=2;codes[folds[1]]=1
        if not all(coverage[codes==i].any(0).all() for i in range(3)):
            raise RuntimeError('Internal grouped class-coverage invariant failed')
        return {g:('train','validation','test')[int(code)] for g,code in zip(unique,codes)}
    raise ValueError('Cannot form class-complete 15% grouped holdouts; generate more independent trajectories')
