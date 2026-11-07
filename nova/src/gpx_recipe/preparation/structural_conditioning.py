"""Pair child distortions with independently sampled parent-cell contexts.

Parent-allowed metric variation is shared across candidate families. The
child's symmetry-breaking strain and every displacement coordinate remain.
This prevents a parent's thermal-cell distribution becoming a family label.
"""
from dataclasses import dataclass
import numpy as np

from .structural_parent_reference import ParentMetricSpace,parent_metric,child_metric,coefficients
from .structural_sampling import TrajectorySamples
from ..structural_live.optimization.physical_prior_geometry import native_reciprocal_matrix


@dataclass(frozen=True)
class ConditionedSamples:
    points:np.ndarray
    parent_a:np.ndarray
    trajectory:np.ndarray
    role:np.ndarray  # 0 original anchor; 1 parent-only control; 2 child hypothesis
    displacement_rms:np.ndarray
    relative_metric_norm:np.ndarray
    maximum_reconstruction_error:float


def condition(source,samples,parent_space,basis,parent_points):
    """Replace only the parent-invariant part, without clipping any point.

    Every final configuration is rechecked in the child's complete physical
    box and chemistry domain. An infeasible block fails as a whole; no family,
    trajectory or weak signal is discarded to improve apparent ML accuracy.
    """
    if not isinstance(samples,TrajectorySamples) or not isinstance(parent_space,ParentMetricSpace):
        raise TypeError('Physical child trajectories and a checked parent metric space required')
    nm,nc=source.mode_dimensions,source.cell_dimensions
    if (samples.points.ndim!=2 or not len(samples.points)
            or samples.points.shape[1]!=nm+nc or not np.isfinite(samples.points).all()
            or samples.trajectory.shape!=(len(samples.points),) or samples.trajectory.dtype.kind not in 'iu'
            or np.any(samples.trajectory<0)):
        raise ValueError('A complete finite nonempty physical trajectory portfolio is required')
    geometry=source.fast.empirical_constraint.geometry
    baseline=native_reciprocal_matrix(geometry.a0)
    if not np.allclose(child_metric(parent_space.baseline,basis),baseline,rtol=1e-9,atol=1e-12):
        raise ValueError('Native child baseline and parent basis do not agree')
    native_mapping=geometry.cell_mapping@source.lift[nm:,nm:]
    if native_mapping.shape!=(6,nc) or np.linalg.matrix_rank(native_mapping)!=nc:
        raise ValueError('Every native child cell coordinate must remain represented')
    parents=np.asarray(parent_points,float)
    count=int(samples.trajectory.max())+1
    if (parents.shape!=(count,6)
            or set(map(int,samples.trajectory))!=set(range(count))):
        raise ValueError('One complete, independently sampled parent context per child trajectory required')
    for a in parents:parent_space.require(native_reciprocal_matrix(a))
    displacement_mapping=np.einsum('aim,mk->aik',geometry.mapping,source.lift[:nm,:nm])
    occupancy=np.asarray(geometry.occupancy,float)
    if occupancy.ndim!=1 or occupancy.shape!=(len(displacement_mapping),) or occupancy.sum()<=0:
        raise ValueError('Conditioned examples need their complete physical site occupancy')
    points=[np.zeros(nm+nc)];references=[coefficients(parent_space.baseline)]
    trajectories=[-1];roles=[0];rms=[0.];norms=[0.];maximum=0.

    def coordinates(g):
        nonlocal maximum
        delta=coefficients(g)-geometry.a0
        values,_,rank,_=np.linalg.lstsq(native_mapping,delta,rcond=None)
        error=float(np.max(abs(native_mapping@values-delta)))
        maximum=max(maximum,error)
        if rank!=nc or error>1e-10*max(float(np.max(abs(geometry.a0))),1e-6):
            raise ValueError('Parent-context change escaped the complete allowed child cell space')
        return values

    def append(point,parent,index,role):
        point=source.check(np.asarray(point,float)).copy()
        actual=native_reciprocal_matrix(geometry.a0+native_mapping@point[nm:])
        direct=np.linalg.inv(actual);movement=np.einsum('aim,m->ai',displacement_mapping,point[:nm])
        squared=float(np.einsum('a,ai,ij,aj->',occupancy/occupancy.sum(),movement,direct,movement))
        if squared < -1e-12 or not np.isfinite(squared):raise ValueError('Invalid physical conditioned displacement norm')
        points.append(point);references.append(parent.copy());trajectories.append(index);roles.append(role)
        rms.append(float(np.sqrt(max(0.,squared))))
        norms.append(parent_space.norm(parent_metric(actual,basis)-parent_space.baseline))

    source.check(points[0])
    for index in range(count):
        parent=native_reciprocal_matrix(parents[index])
        common=coordinates(child_metric(parent,basis))
        append(np.r_[np.zeros(nm),common],parents[index],index,1)
        for old in samples.points[samples.trajectory==index]:
            original=parent_metric(native_reciprocal_matrix(geometry.a0+native_mapping@old[nm:]),basis)
            broken=original-parent_space.project(original)
            changed=child_metric(parent+broken,basis)
            append(np.r_[old[:nm],coordinates(changed)],parents[index],index,2)
    arrays=[np.asarray(points,float),np.asarray(references,float),np.asarray(trajectories,np.int64),
            np.asarray(roles,np.int8),np.asarray(rms,float),np.asarray(norms,float)]
    for a in arrays:a.flags.writeable=False
    return ConditionedSamples(*arrays,maximum)
