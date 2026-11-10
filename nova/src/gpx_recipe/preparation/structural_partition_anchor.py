"""Retain original whole-group holdouts when complete ranges are added.

Anchors are computed by the installed assembler from original native arrays.
An uploaded mapping cannot choose training membership. No expanded feature
plane is needed to reproduce original aliases and group/class coverage.
"""
import numpy as np

from ..symmetry.contracts import digest


class PartitionOnly:
    def reserve(self, count, bins):
        self.capacity=count;self.bins=bins;self.rows=0

    def append(self, feature):
        if (self.rows>=self.capacity or feature.shape!=(3,self.bins)
                or feature.dtype!=np.float32 or not np.isfinite(feature).all()):
            raise ValueError('An original partition needs every finite three-curve row')
        self.rows+=1

    def finish(self, rows):
        if rows!=self.rows:raise ValueError('The original partition row inventory changed')
        return None


def capture(cohort, metadata, *, interval, split_seed,additive=False):
    assignments={}
    for code,references in zip(cohort['split'],cohort['references']):
        for block,row in references:
            group=metadata[block]['groups'][row]
            previous=assignments.setdefault(group,int(code))
            if previous!=int(code):raise ValueError('The original partition crosses a physical trajectory')
    if set(assignments)!={g for d in metadata for g in d['groups']}:
        raise ValueError('The original partition must retain every native physical group')
    return dict(assignments=assignments,q_edges=cohort['q_edges'],
        reference_features=cohort['reference_features'],
        trace=dict(schema='radar-owned-original-additive-partition/v1' if additive else 'radar-owned-original-range-partition/v1',
            original_relative_metric_range=interval,split_seed=split_seed,
            original_input_sha256=cohort['input_sha256'],physical_groups=len(assignments),
            group_assignment_sha256=digest(sorted(assignments.items()))))


def assign(anchor, parents, find):
    original=anchor['assignments'];result={}
    if set(original)!=set(parents):
        raise ValueError('A parent-range expansion must retain all original physical groups')
    for group,code in original.items():
        root=find(group)
        if root in result and result[root]!=code:
            raise ValueError('A new alias links original training and holdout groups; refusing to reshuffle them')
        result[root]=code
    return result
