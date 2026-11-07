"""Parent-symmetry metric references for paired structural training.

Only the parent-invariant metric enters the reference. Symmetry-breaking
strain stays in the changed pattern. This is not registration or a warp of
measured data, and a generated parent curve is not a fitted observation.
"""
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ..structural_live.optimization.physical_prior_geometry import (
    metric_matrix,native_reciprocal_matrix,reciprocal_metric_gram,orthonormal_lift,
)


def coefficients(g,*,positive=True):
    g=np.asarray(g,float)
    if (g.shape!=(3,3) or not np.isfinite(g).all()
            or not np.allclose(g,g.T,rtol=1e-12,atol=0)):
        raise ValueError('A finite symmetric reciprocal metric is required')
    if positive:metric_matrix(g)
    return np.array([g[0,0],g[1,1],g[2,2],2*g[0,1],2*g[0,2],2*g[1,2]])


def rotations(sg):
    """Complete native fractional point operations, without translations."""
    signs=(1,-1) if sg['SGInv'] and not sg.get('SGFixed',False) else (1,)
    unique={tuple((sign*np.asarray(r,dtype=int)).ravel())
            for sign in signs for r,_ in sg['SGOps']}
    return np.asarray([np.asarray(key).reshape(3,3) for key in sorted(unique)],dtype=int)


class ParentMetricSpace:
    def __init__(self,baseline,operations):
        self.baseline=np.array(metric_matrix(baseline),copy=True)
        self.baseline.flags.writeable=False
        r=np.asarray(operations)
        if (r.ndim!=3 or r.shape[1:]!=(3,3) or not 1<=len(r)<=48
                or not np.isfinite(r).all() or not np.array_equal(r,np.rint(r))
                or np.any(abs(r)>16)):
            raise ValueError('A complete finite fractional parent point group is required')
        r=r.astype(int);keys={tuple(x.ravel()) for x in r}
        if (len(keys)!=len(r) or tuple(np.eye(3,dtype=int).ravel()) not in keys
                or any(round(np.linalg.det(x)) not in (-1,1) for x in r)
                or any(tuple((x@y).ravel()) not in keys for x in r for y in r)):
            raise ValueError('Parent point operations must be unique, unimodular and closed')
        r.flags.writeable=False
        self.operations=r
        if not np.allclose(r@self.baseline@r.transpose(0,2,1),self.baseline,rtol=1e-9,atol=1e-12):
            raise ValueError('Parent point operations do not preserve the native baseline metric')
        native_basis=np.array([native_reciprocal_matrix(v) for v in np.eye(6)])
        projection=np.stack([coefficients(self.project(v,positive=False),positive=False) for v in native_basis],axis=1)
        if not np.allclose(projection@projection,projection,rtol=1e-10,atol=1e-12):
            raise ValueError('Parent metric projection is not idempotent')
        rank=int(round(np.trace(projection)))
        vectors,singular,_=np.linalg.svd(projection)
        if (not 1<=rank<=6 or np.any(singular[:rank]<1e-8)
                or np.any(singular[rank:]>1e-8)):
            raise ValueError('Parent metric symmetry has unresolved invariant dimension')
        basis=vectors[:,:rank]
        directions=np.asarray([native_reciprocal_matrix(v) for v in basis.T])
        self.basis=basis@orthonormal_lift(reciprocal_metric_gram(self.baseline,directions))
        self.basis.flags.writeable=False

    def project(self,metric,*,positive=True):
        g=np.asarray(metric,float)
        if (g.shape!=(3,3) or not np.isfinite(g).all()
                or not np.allclose(g,g.T,rtol=1e-12,atol=0)):
            raise ValueError('A finite symmetric parent-basis metric is required')
        if positive:metric_matrix(g)
        result=np.mean(self.operations@g@self.operations.transpose(0,2,1),axis=0)
        result=(result+result.T)*.5
        return metric_matrix(result) if positive else result

    def require(self,metric):
        g=metric_matrix(metric)
        if not np.allclose(self.project(g),g,rtol=1e-9,atol=1e-12):
            raise ValueError('A parent reference must retain the complete parent metric symmetry')
        return g

    def norm(self,delta):
        x=np.linalg.solve(self.baseline,np.asarray(delta,float))
        value=float(np.trace(x@x))
        if not np.isfinite(value) or value < -1e-12:raise ValueError('Invalid parent reciprocal-metric norm')
        return float(np.sqrt(max(value,0.)))

    def draw(self,*,seed,count,relative_range=(.0001,.001)):
        from .structural_sampling import _interval
        low,high=_interval(relative_range,'parent relative metric')
        if type(seed) is not int or not 0<=seed<2**32 or type(count) is not int or not 1<=count<=4096:
            raise ValueError('A bounded parent trajectory seed and count are required')
        rng=np.random.default_rng(seed);points=[]
        for _ in range(count):
            direction=rng.normal(size=self.basis.shape[1]);direction/=np.linalg.norm(direction)
            delta=native_reciprocal_matrix(self.basis@direction)*np.exp(rng.uniform(np.log(low),np.log(high)))
            points.append(coefficients(self.require(self.baseline+delta)))
        return np.asarray(points)


def parent_metric(child_metric,basis):
    """Candidate basis rows are child direct axes in parent coordinates."""
    t=np.asarray(basis,float)
    if t.shape!=(3,3) or not np.isfinite(t).all() or np.linalg.cond(t)>1e8:
        raise ValueError('A finite nonsingular child basis is required')
    result=t.T@metric_matrix(child_metric)@t
    return metric_matrix((result+result.T)*.5)


def child_metric(parent,basis):
    t=np.asarray(basis,float)
    if t.shape!=(3,3) or not np.isfinite(t).all() or np.linalg.cond(t)>1e8:
        raise ValueError('A finite nonsingular child basis is required')
    inverse=np.linalg.inv(t)
    result=inverse.T@metric_matrix(parent)@inverse
    return metric_matrix((result+result.T)*.5)


class NativeParentReference:
    """Owned native forward context; render separately from child adapters.

    Native global constraint maps are process-local. Construct this after the
    child rendering pass, or in a separate process. Existing native ownership
    checks reject accidental interleaving rather than returning stale curves.
    """
    def __init__(self,cif,instrument,destination,*,grid,maximum_relative_metric=.02):
        from GSASII import GSASIIscriptable as G
        from .reference import create_reference
        from ..live.native_profile import NativeReducedProfile
        reference=create_reference(cif,instrument,Path(destination)/'reference',**grid)
        self.project=G.G2Project(gpxfile=reference['path']);self.phase=self.project.phase(reference['parent_phase'])
        self.histogram=self.project.histogram(reference['histogram'])
        self.phase.set_refinements({'Cell':True});self.project.save()
        self.native=NativeReducedProfile(reference['path'],refresh_cell_reflections=True)
        self.prefix=f'{self.phase.id}::A'
        if any(not name.startswith(self.prefix) for name in self.native.vary):
            raise ValueError('Parent-reference forward context released a non-cell parameter')
        a=self.native.lattice.cell2A(self.phase.data['General']['Cell'][1:7])
        self.space=ParentMetricSpace(native_reciprocal_matrix(a),rotations(self.phase.data['General']['SGData']))
        if type(maximum_relative_metric) not in (int,float) or not np.isfinite(maximum_relative_metric) or not 0<maximum_relative_metric<=.1:
            raise ValueError('A bounded parent reference metric envelope is required')
        self.maximum_relative_metric=float(maximum_relative_metric)
        self.x=np.asarray(self.histogram.data['data'][1][0],float).copy()
        d=np.asarray(self.native.lattice.Pos2dsp(self.histogram.InstrumentParameters,self.x),float)
        if not np.isfinite(d).all() or np.any(d<=0):raise ValueError('Parent reference needs finite positive native d values')
        q=2*np.pi/d;self.order=np.argsort(q);self.q=q[self.order]
        if np.any(np.diff(self.q)<=0):raise ValueError('Parent reference needs a unique increasing Q grid')
        self.cache=OrderedDict();self.cache_bytes=0;self.maximum_cache_bytes=64*1024**2;self.native_calls=0

    def evaluate(self,a):
        self.native._check()
        a=coefficients(self.space.require(native_reciprocal_matrix(a)))
        if self.space.norm(native_reciprocal_matrix(a)-self.space.baseline)>self.maximum_relative_metric+1e-12:
            raise ValueError('Parent reference cell exceeds its declared metric envelope')
        # Follow the same native reciprocal -> direct -> reciprocal rounding
        # used when a saved GPX is independently forward calculated.
        a=np.asarray(self.native.lattice.cell2A(self.native.lattice.A2cell(a)),float)
        self.space.require(native_reciprocal_matrix(a))
        key=a.tobytes()
        if key in self.cache:
            self.cache.move_to_end(key)
            return tuple(v.copy() for v in self.cache[key])
        values=self.native.x0.copy()
        for i,name in enumerate(self.native.vary):values[i]=a[int(name[len(self.prefix):])]
        curve=np.zeros_like(self.x)
        profile=self.native.profile(values);self.native_calls+=1
        if not np.allclose([self.native.parms[self.prefix+str(i)] for i in range(6)],a,rtol=1e-10,atol=1e-12):
            raise ValueError('Native parent constraints changed the requested reference cell')
        name,sl=self.native.parts[0];curve[sl]=profile
        background=np.asarray(self.native.histo_phases[0][name]['Data'][4],float).copy()
        if not np.isfinite(curve).all() or not np.isfinite(background).all():
            raise ValueError('Native parent reference was not finite')
        result=(curve[self.order],background[self.order])
        size=sum(v.nbytes for v in result)
        while self.cache and self.cache_bytes+size>self.maximum_cache_bytes:
            _,removed=self.cache.popitem(last=False);self.cache_bytes-=sum(v.nbytes for v in removed)
        if size<=self.maximum_cache_bytes:
            self.cache[key]=tuple(v.copy() for v in result);self.cache_bytes+=size
        return result
