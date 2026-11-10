"""Check complete signed strain probes without granting scientific authority."""
import numpy as np
from ..symmetry.contracts import fields
from ..structural_live.optimization.physical_prior_geometry import (
    orthonormal_lift,native_reciprocal_matrix,reciprocal_metric_gram,
)

SCHEMA='radar-complete-parent-conditioned-cell-rays/v1'


def validate(document,arrays,projection):
    d=fields(document['cell_coverage'],['schema','dimensions','directions','cell_gram','rays'])
    nc=document['cell_dimensions'];nm=document['mode_dimensions'];dimension=d['dimensions']
    if (d['schema']!=SCHEMA or type(dimension)is not int or not 0<=dimension<=nc
            or not isinstance(d['rays'],list) or len(d['rays'])!=2*dimension
            or 2*dimension>document['trajectories'] or 'supplement' in document):
        raise ValueError('Complete original signed cell-ray coverage required')
    directions=np.asarray(d['directions'],float);gram=np.asarray(d['cell_gram'],float)
    if dimension==0 and directions.shape==(0,):directions=directions.reshape(0,nc)
    if (directions.shape!=(dimension,nc) or gram.shape!=(nc,nc) or not np.isfinite(directions).all()
            or not np.isfinite(gram).all()
            or not np.allclose(directions@gram@directions.T,np.eye(dimension),rtol=1e-10,atol=1e-12)):
        raise ValueError('All declared cell directions need their full native physical normalization')
    orthonormal_lift(gram)
    if projection is not None and 'strain' in projection.metadata:
        context=projection.metadata['strain'];mapping=np.asarray(context['cell_mapping'])
        wanted=reciprocal_metric_gram(native_reciprocal_matrix(context['baseline_a']),
            np.array([native_reciprocal_matrix(v) for v in mapping.T]))
        if not np.allclose(gram,wanted,rtol=1e-10,atol=1e-12):
            raise ValueError('Cell coverage changed its captured native reciprocal-metric units')
    first,last=document['trajectory_slice'];steps=document['steps'];p=arrays['parameters']
    for index,ray in enumerate(d['rays']):
        ray=fields(ray,['trajectory','axis','sign','maximum_relative_metric_norm',
            'requested_relative_metric_norm','box_relative_metric_norm','rejected_amplitudes'])
        maximum=ray['maximum_relative_metric_norm'];requested=ray['requested_relative_metric_norm'];rejected=ray['rejected_amplitudes']
        cap=ray['box_relative_metric_norm']
        if (type(ray['trajectory'])is not int or ray['trajectory']!=index
                or type(ray['axis'])is not int or ray['axis']!=index//2
                or type(ray['sign'])is not int or ray['sign']!=(-1 if index%2==0 else 1)
                or type(rejected)is not int or not 0<=rejected<=100000
                or any(type(v)not in (int,float) or not np.isfinite(v) or v<=0 for v in [maximum,requested,cap])
                or cap>requested or not np.isclose(maximum,cap*2.**(-rejected),rtol=1e-12,atol=0)):
            raise ValueError('Every signed strain ray must retain its axis, parent and admitted endpoint')
        if not first<=index<last:continue
        rows=np.flatnonzero(arrays['trajectory']==index)[:steps+1]
        if len(rows)!=steps+1 or arrays['role'][rows[0]]!=1 or np.any(arrays['role'][rows[1:]]!=2):
            raise ValueError('A strain ray needs its original parent control and every amplitude step')
        if np.any(p[rows,:nm]!=0):raise ValueError('Mandatory strain probes must not add atomic displacement')
        actual=p[rows[1:],nm:]-p[rows[0],nm:]
        expected=np.geomspace(.1,1.,steps)[:,None]*maximum*ray['sign']*directions[index//2]
        scale=np.max(abs(expected),axis=1,initial=0.)
        error=np.max(abs(actual-expected),axis=1,initial=0.)
        if np.any(error>1e-9*np.maximum(scale,np.finfo(float).tiny)):
            raise ValueError('Captured native strain points differ from their declared signed coverage')
    return d
