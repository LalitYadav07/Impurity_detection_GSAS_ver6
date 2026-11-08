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


class ParentConditioner:
    """One metric transformation for trajectory admission and final examples.

    Parent contexts are supplied before any candidate-specific rejection.
    An invalid parent-only control is a scope error, not permission to redraw
    a different parent for this family. Child rays may be redrawn inside the
    same original bounds, with all coordinates and amplitude steps retained.
    """
    def __init__(self,source,parent_space,basis,parent_points):
        if not isinstance(parent_space,ParentMetricSpace):
            raise TypeError('A checked parent metric space is required')
        self.source=source;self.space=parent_space
        self.nm=source.mode_dimensions;self.nc=source.cell_dimensions
        self.geometry=source.fast.empirical_constraint.geometry
        self.basis=np.array(basis,float,copy=True)
        baseline=native_reciprocal_matrix(self.geometry.a0)
        if not np.allclose(child_metric(parent_space.baseline,self.basis),baseline,rtol=1e-9,atol=1e-12):
            raise ValueError('Native child baseline and parent basis do not agree')
        self.mapping=self.geometry.cell_mapping@source.lift[self.nm:,self.nm:]
        if self.mapping.shape!=(6,self.nc) or np.linalg.matrix_rank(self.mapping)!=self.nc:
            raise ValueError('Every native child cell coordinate must remain represented')
        self.parents=np.array(parent_points,float,copy=True)
        if self.parents.ndim!=2 or self.parents.shape[1]!=6 or not 1<=len(self.parents)<=4096:
            raise ValueError('One complete, independently sampled parent context per child trajectory required')
        self.metrics=np.asarray([parent_space.require(native_reciprocal_matrix(a)) for a in self.parents])
        self.maximum_reconstruction_error=0.
        self.controls=np.asarray([np.r_[np.zeros(self.nm),self.coordinates(child_metric(g,self.basis))]
                                  for g in self.metrics])
        # Preserve strict native/chemistry diagnostics instead of turning an
        # impossible shared parent into endless child resampling.
        for control in self.controls:
            if np.any(abs(control)>source.radii):
                from .failures import PreparationScopeError
                raise PreparationScopeError('parent_cell_domain_conflict')
            source.check(control)
        for array in (self.basis,self.mapping,self.parents,self.metrics,self.controls):array.flags.writeable=False

    def coordinates(self,metric):
        delta=coefficients(metric)-self.geometry.a0
        values,_,rank,_=np.linalg.lstsq(self.mapping,delta,rcond=None)
        error=float(np.max(abs(self.mapping@values-delta)))
        self.maximum_reconstruction_error=max(self.maximum_reconstruction_error,error)
        if rank!=self.nc or error>1e-10*max(float(np.max(abs(self.geometry.a0))),1e-6):
            raise ValueError('Parent-context change escaped the complete allowed child cell space')
        return values

    def metric(self,index,point):
        original=parent_metric(native_reciprocal_matrix(self.geometry.a0+self.mapping@point[self.nm:]),self.basis)
        broken=original-self.space.project(original)
        return self.metrics[index]+broken

    def point(self,index,old):
        return np.r_[old[:self.nm],self.coordinates(child_metric(self.metric(index,old),self.basis))]

    def feasible(self,index,rows):
        """Reject physical proposals only; propagate technical check errors."""
        rows=np.asarray(rows,float)
        if (type(index) is not int or not 0<=index<len(self.parents)
                or rows.ndim!=2 or rows.shape[1]!=self.nm+self.nc or not np.isfinite(rows).all()):
            raise ValueError('A finite complete trajectory and its declared parent index are required')
        allowed=np.zeros(len(rows),bool);selected=[];points=[]
        for i,old in enumerate(rows):
            changed=self.metric(index,old)
            if np.linalg.eigvalsh(changed)[0]<=0:continue
            point=np.r_[old[:self.nm],self.coordinates(child_metric(changed,self.basis))]
            if np.any(abs(point)>self.source.radii):continue
            selected.append(i);points.append(point)
        if selected:
            values=np.asarray(self.source.chemistry.log_kernel(np.asarray(points)))
            if (values.shape!=(len(selected),) or values.dtype.kind!='f'
                    or np.isnan(values).any() or np.isposinf(values).any()):
                raise ValueError('Conditioned chemistry must return one finite value or negative infinity per point')
            allowed[selected]=np.isfinite(values)
        return allowed


def condition(source,samples,parent_space,basis,parent_points):
    """Replace only the parent-invariant part, without clipping any point.

    Every final configuration is rechecked in the child's complete physical
    box and chemistry domain. An infeasible block fails as a whole; no family,
    trajectory or weak signal is discarded to improve apparent ML accuracy.
    """
    if not isinstance(samples,TrajectorySamples):
        raise TypeError('Physical child trajectories and a checked parent metric space required')
    nm,nc=source.mode_dimensions,source.cell_dimensions
    if (samples.points.ndim!=2 or not len(samples.points)
            or samples.points.shape[1]!=nm+nc or not np.isfinite(samples.points).all()
            or samples.trajectory.shape!=(len(samples.points),) or samples.trajectory.dtype.kind not in 'iu'
            or np.any(samples.trajectory<0)):
        raise ValueError('A complete finite nonempty physical trajectory portfolio is required')
    context=ParentConditioner(source,parent_space,basis,parent_points)
    parents=context.parents;geometry=context.geometry;native_mapping=context.mapping
    count=int(samples.trajectory.max())+1
    if len(parents)!=count or set(map(int,samples.trajectory))!=set(range(count)):
        raise ValueError('One complete, independently sampled parent context per child trajectory required')
    displacement_mapping=np.einsum('aim,mk->aik',geometry.mapping,source.lift[:nm,:nm])
    occupancy=np.asarray(geometry.occupancy,float)
    if occupancy.ndim!=1 or occupancy.shape!=(len(displacement_mapping),) or occupancy.sum()<=0:
        raise ValueError('Conditioned examples need their complete physical site occupancy')
    points=[np.zeros(nm+nc)];references=[coefficients(parent_space.baseline)]
    trajectories=[-1];roles=[0];rms=[0.];norms=[0.]

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
        append(context.controls[index],parents[index],index,1)
        for old in samples.points[samples.trajectory==index]:
            append(context.point(index,old),parents[index],index,2)
    arrays=[np.asarray(points,float),np.asarray(references,float),np.asarray(trajectories,np.int64),
            np.asarray(roles,np.int8),np.asarray(rms,float),np.asarray(norms,float)]
    for a in arrays:a.flags.writeable=False
    return ConditionedSamples(*arrays,context.maximum_reconstruction_error)
