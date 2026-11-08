"""Closed row identities for additive, complete named-mode paired blocks.

These records describe generated hypotheses, not their diffraction labels or
scientific qualification. Old paired schemas retain their fixed row layout.
"""
import copy

import numpy as np

from .structural_mode_coverage import SCHEMA,CONTEXTS_PER_MODE

RAW_SCHEMA='radar-structural-ranking-paired-pilot/v2'
PACKED_SCHEMA='radar-structural-ranking-pairs/v2'
SCHEMAS={RAW_SCHEMA,PACKED_SCHEMA}


def describe(coverage,*,contexts=CONTEXTS_PER_MODE):
    return dict(schema=SCHEMA,contexts=contexts,labels=list(coverage.labels),
        gauge_axes=list(coverage.gauge_axes),base_examples=coverage.base_examples,
        rays=copy.deepcopy(list(coverage.rays)),rejected_amplitudes=coverage.rejected_amplitudes,
        maximum_reconstruction_error=coverage.maximum_reconstruction_error)


def rows(document,count,steps,interval):
    """Validate every axis/sign/context before deriving one whole-parent shard.

    Ray row ranges refer to the unconditioned complete portfolio. Conditioning
    groups those rows beside their unchanged parent-only reference. No shard
    can omit one of its declared rays or retain only convenient amplitudes.
    """
    d=document.get('mode_coverage')
    keys={'schema','contexts','labels','gauge_axes','base_examples','rays',
          'rejected_amplitudes','maximum_reconstruction_error'}
    if not isinstance(d,dict) or set(d)!=keys or d['schema']!=SCHEMA:
        raise ValueError('Named-mode pairs require their complete versioned coverage record')
    contexts=d['contexts'];labels=d['labels'];gauges=d['gauge_axes'];rays=d['rays']
    error=d['maximum_reconstruction_error'];rejected=d['rejected_amplitudes']
    if (type(contexts)is not int or not 1<=contexts<=min(32,count)
            or not isinstance(labels,list) or len(labels)>2048
            or any(not isinstance(v,str) or not 1<=len(v)<=1000 for v in labels)
            or not isinstance(gauges,list) or any(type(i)is not int or not 0<=i<len(labels) for i in gauges)
            or gauges!=sorted(set(gauges))
            or type(d['base_examples'])is not int or d['base_examples']!=count*steps
            or type(rejected)is not int or rejected<0
            or type(error)not in (int,float) or not np.isfinite(error) or not 0<=error<=1e-10
            or not isinstance(rays,list)):
        raise ValueError('Invalid complete named-mode inventory or physical reconstruction')
    physical=[i for i in range(len(labels)) if i not in set(gauges)]
    if len(rays)!=len(physical)*contexts*2 or count*steps+len(rays)*steps>32768:
        raise ValueError('Every non-gauge native mode needs all signed complete context rays')
    cursor=count*steps;attempts=0;parents=[]
    ray_keys={'native_mode','parent','sign','rows','maximum_amplitude','rejected_amplitudes'}
    for axis in physical:
        used=set()
        for _ in range(contexts):
            parent=None
            for sign in (-1,1):
                ray=rays[len(parents)]
                if not isinstance(ray,dict) or set(ray)!=ray_keys:
                    raise ValueError('A named-mode ray has an unexpected identity or inventory')
                p=ray['parent'];a=ray['maximum_amplitude'];tries=ray['rejected_amplitudes']
                if (type(ray['native_mode'])is not int or ray['native_mode']!=axis
                        or type(ray['sign'])is not int or ray['sign']!=sign
                        or type(p)is not int or not 0<=p<count
                        or not isinstance(ray['rows'],list)
                        or any(type(i)is not int for i in ray['rows'])
                        or ray['rows']!=[cursor,cursor+steps]
                        or type(a)not in (int,float) or not np.isfinite(a) or a<=0
                        or type(tries)is not int or not 0<=tries<100000):
                    raise ValueError('Named-mode ray lost its axis, sign, context or complete amplitude steps')
                if sign==-1:
                    if p in used:raise ValueError('Each named mode needs distinct existing parent contexts')
                    parent=p;used.add(p)
                elif p!=parent:
                    raise ValueError('Both signs of a named mode must share the same reviewed parent context')
                parents.append(p);cursor+=steps;attempts+=tries
    if attempts!=rejected:
        raise ValueError('Named-mode amplitude rejections must retain their complete accounting')
    identities=[-1];roles=[0];raw_rows=[-1];ends=[]
    # Each old mixed trajectory is an exact prefix of its enlarged parent group.
    for parent in range(*interval):
        identities.append(parent);roles.append(1);raw_rows.append(-1)
        indices=list(range(parent*steps,(parent+1)*steps))
        for ray_id,p in enumerate(parents):
            if p==parent:
                first=count*steps+ray_id*steps
                indices.extend(range(first,first+steps))
        identities.extend([parent]*len(indices));roles.extend([2]*len(indices));raw_rows.extend(indices)
        # Every retained ray's endpoint gets the original independent check.
        ends.extend(range(len(identities)-len(indices)+steps-1,len(identities),steps))
    return (np.asarray(identities,np.int64),np.asarray(roles,np.int8),
            np.asarray(raw_rows,np.int64),np.asarray(ends,np.int64))
