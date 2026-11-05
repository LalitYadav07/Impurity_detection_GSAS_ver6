"""Candidate-specific native in-memory prototype, pending broad qualification.

No scattering approximation: lift physical coordinates into GSAS-II's own
mode constraints and reciprocal-cell coordinates. The checked slow adapter
remains an independent oracle and export path.
"""
import numpy as np
from pathlib import Path

from gpx_recipe.structural_live.native import NativeCandidate
from gpx_recipe.live.native_profile import NativeReducedProfile


class NativeFastCandidate:
    def __init__(self, prepared, work):
        self.slow=NativeCandidate(prepared,work);self.meta=self.slow.meta;self.basis=self.slow.basis
        p=self.slow.project;phase=self.slow.phase;h=self.slow.hist;iso=phase.data.get('ISODISTORT',{})
        self.nm=self.meta['inventory']['physical_rank'];self.nc=self.meta['inventory']['cell_rank']
        self.dimension=self.nm+self.nc+2
        # A single versioned empirical policy guards every new native forward
        # path, including synthetic truths and numerical derivative probes.
        # Historical frozen runs retain their original source and assumptions.
        from .empirical_catalog import load_declared_catalog,EmpiricalNativeConstraint,EmpiricalCheckedSlowForward
        catalog=load_declared_catalog(Path(__file__).with_name('empirical_chemistry_policy.json'))
        self.empirical_constraint=EmpiricalNativeConstraint(self.slow,catalog)
        self.empirical_constraint.require(np.zeros(self.nm),np.zeros(self.nc))
        self.slow=EmpiricalCheckedSlowForward(self.slow,self.empirical_constraint)
        self.empirical_policy_fingerprint=self.empirical_constraint.fingerprint
        from .bounded_chemical_prior import geometry_identity
        g=self.empirical_constraint.geometry
        self.chemical_geometry_key=geometry_identity(g.base,g.mapping,g.a0,g.cell_mapping,g.species,g.occupancy)
        rows=[c for c in p.data['Constraints']['data']['Phase'] if c[-1]=='f']
        names=[v.name for v in iso.get('G2ModeList',[])]
        if len(rows)!=len(names) or [row[-3].name for row in rows]!=names:
            raise ValueError('Native mode-constraint identity/order mismatch')
        cx,ct,_,_=phase.data['General']['AtomPtrs'];atoms={a[-1]:i for i,a in enumerate(phase.data['Atoms'])}
        mode_lift=[]
        for row in rows:
            coefficients=np.zeros(len(names))
            for coefficient,var in row[:-3]:
                if var.name not in ['dAx','dAy','dAz'] or var.atom not in atoms:
                    raise ValueError('Native mode constraint escaped atomic displacements')
                coefficients+=float(coefficient)*self.basis['coordinate_basis'][atoms[var.atom],'xyz'.index(var.name[-1])]
            mode_lift.append(coefficients);row[-2]=True
        if names and not np.allclose(mode_lift,np.eye(len(names)),rtol=1e-9,atol=1e-9):
            raise ValueError('Native mode constraints disagree with native ApplyModeDisp normalization')
        active={v.atom for v in iso.get('G2VarList',[])}
        for atom in phase.data['Atoms']:
            if atom[-1] in active:atom[ct+1]='X'
        phase.set_refinements({'Cell':True});h.set_refinements({'Sample Parameters':['Scale']})
        h.data['Background'][0][1]=True
        # Preserve the existing fixed higher background coefficients; only the
        # constant term is varied in this numerical prototype's declared scope.
        # Native dervRefine requires contiguous background derivative rows.
        # Keep all of those native rows, but map only Back;0 into our external
        # parameter vector. Higher coefficients remain exactly at native x0.
        p.update_ids()
        p.save();self.native=NativeReducedProfile(p.filename,refresh_cell_reflections=True)
        self.lift=np.zeros((len(self.native.vary),self.dimension));self.scale_index=None
        for i,name in enumerate(self.native.vary):
            prefix=f'{phase.id}::nv-'
            if name.startswith(prefix) and name[len(prefix):] in names:
                self.lift[i,:self.nm]=self.basis['physical_basis'][names.index(name[len(prefix):])]
            elif name.startswith(f'{phase.id}::A') and name[-1:].isdigit():
                axis=int(name[-1]);self.lift[i,self.nm:self.nm+self.nc]=self.slow.a_scale*self.basis['cell_basis'][axis]
            elif name==f':{h.id}:Back;0':self.lift[i,-1]=1.
            elif name.startswith(f':{h.id}:Back;') and name.rsplit(';',1)[-1].isdigit():pass
            elif name==f':{h.id}:Scale':self.scale_index=i
            else:raise ValueError(f'Unexpected free native parameter: {name}')
        if self.scale_index is None:raise ValueError('Histogram scale variable is missing')
        self.x=np.asarray(h.data['data'][1][0]);self.parts=self.native.parts
        if len(self.parts)!=1:raise ValueError('One independently analyzed histogram required')
        self.slice=self.parts[0][1];self.last=None

    def _values(self,theta,chemistry_domain=None):
        theta=np.asarray(theta,float)
        if theta.shape!=(self.dimension,) or not np.isfinite(theta).all() or abs(theta[-2])>100:
            raise ValueError('Invalid physical parameter vector')
        a=self.basis['baseline_a']+self.slow.a_scale*(self.basis['cell_basis']@theta[self.nm:self.nm+self.nc])
        metric=np.array([[a[0],a[3]/2,a[4]/2],[a[3]/2,a[1],a[5]/2],[a[4]/2,a[5]/2,a[2]]])
        if np.linalg.eigvalsh(metric).min()<=0:raise ValueError('Nonpositive native cell metric')
        certified=False
        if chemistry_domain is not None:
            from .bounded_chemical_prior import BoundedChemicalGeometry
            compiler,z=chemistry_domain
            if not isinstance(compiler,BoundedChemicalGeometry):raise ValueError('Checked bounded-chemistry domain required')
            certified=compiler.certifies_native(theta[:self.nm+self.nc],z,geometry_key=self.chemical_geometry_key,
                catalog_key=self.empirical_constraint.catalog.fingerprint)
        if not certified:self.empirical_constraint.require(theta[:self.nm],theta[self.nm:self.nm+self.nc])
        values=self.native.x0+self.lift@theta
        # GPX stores direct-cell parameters. Canonicalize through the same native
        # A -> cell -> A conversion as export/DoNoFit; tiny representation
        # differences otherwise cross numerical branches in native CW kernels.
        canonical_a=self.slow.l.cell2A(self.slow.l.A2cell(a))
        prefix=f'{self.slow.phase.id}::A'
        for i,name in enumerate(self.native.vary):
            if name.startswith(prefix) and name[len(prefix):].isdigit():
                values[i]=canonical_a[int(name[len(prefix):])]
        values[self.scale_index]=self.slow.histogram_scale*np.exp(theta[-2])
        return theta,values

    def evaluate_mode_jacobian(self,theta,*,chemistry_domain=None):
        """Native displacement derivatives only; no analytic cell claim.

        This separate API avoids spending cell finite differences when a
        caller needs numerical cell columns in different physical coordinates.
        Every supplied geometry still passes the same chemistry check.
        """
        theta,values=self._values(theta,chemistry_domain);self.last=theta.copy()
        mean,derivative=self.native.profile(values,jacobian=True)
        jac=self.lift[:,:self.nm].T@np.asarray(derivative,float)
        full=np.zeros(len(self.x));full[self.slice]=mean
        full_jac=np.zeros((self.nm,len(self.x)));full_jac[:,self.slice]=jac
        return full,full_jac

    def evaluate(self,theta,*,jacobian=False,cell_step=1e-4,chemistry_domain=None):
        theta,values=self._values(theta,chemistry_domain);self.last=theta.copy()
        if not jacobian:
            mean=self.native.profile(values)
            full=np.zeros(len(self.x));full[self.slice]=mean
            return full
        mean,derivative=self.native.profile(values,jacobian=True)
        derivative=np.asarray(derivative,float)
        mapping=self.lift.copy();mapping[self.scale_index,-2]=values[self.scale_index]
        jac=mapping.T@derivative
        # Native cell analytic derivatives failed independent checks. Their
        # numerical replacement still needs step-size qualification per case.
        for j in range(self.nm,self.nm+self.nc):
            plus=theta.copy();minus=theta.copy();plus[j]+=cell_step;minus[j]-=cell_step
            jac[j]=(self.native.profile(self._values(plus)[1])-self.native.profile(self._values(minus)[1]))/(2*cell_step)
        self.native.profile(values)
        full=np.zeros(len(self.x));full[self.slice]=mean
        full_jac=np.zeros((self.dimension,len(self.x)));full_jac[:,self.slice]=jac
        return full,full_jac
