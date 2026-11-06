"""Native constrained quadratic-profile oracle for fixed-geometry magnetism.

No observation, uncertainty, exclusion, profile, nuclear parameter or native
refinement status is changed here. The caller must independently qualify any
optimizer result and audit the serialized output. This adapter itself declares
neither GSAS-II convergence nor covariance. GSAS-II's constraint maps are
process-global; concurrent adapters in one process are forbidden.
"""
from __future__ import annotations

import io
import time
import uuid
from pathlib import Path
from contextlib import contextmanager
import numpy as np
from .profiled_magnetic import ProfiledMagneticObjective
from ..live.native_profile import constraint_state_signature

_active=None


class _CacheMiss(Exception):
    """Release the reader lock before rebuilding a missing or invalid hint."""


class _CompilerObjective:
    """Present the same weighted native residual to the shared compiler."""
    def __init__(self, native, background):
        self.native=native;self.background=background;self.numerical_indices=[]

    def profile(self, moments, jacobian=False):
        value=self.native.residual(moments,1.,self.background)
        if not jacobian:return value
        return value,self.native.magnetic_jacobian(moments,1.,self.background)


class NativeMagneticStartingPoint:
    def __init__(self,filename,*,refine_scale,refine_background,maximum_moment,deadline=None):
        global _active
        from GSASII import GSASIIstrMain as main,GSASIIstrIO as stio
        from GSASII import GSASIIstrMath as math,GSASIImapvars as maps
        from GSASII import GSASIImpsubs as mp,GSASIIlattice as lattice
        self.math,self.maps,self.lattice=math,maps,lattice
        self.deadline=deadline
        _active=self
        mp.InitMP();main.ptx.pyqlmninit();maps.InitVars();log=io.StringIO()
        controls=stio.GetControls(str(filename));controls['newLeBail']=False
        histograms,phases=stio.GetUsedHistogramsAndPhases(str(filename))
        names=[name for name in histograms if name.startswith('PWDR')]
        if len(names)!=1 or len(histograms)!=1:
            raise ValueError('The independent-bank starting-point adapter requires one powder histogram')
        self.histogram=histograms[names[0]];self.histogram_name=names[0]
        if not np.any(self.histogram['Data'][1]):
            raise ValueError('A native dummy-data histogram cannot be used for refinement')
        self.histogram_id=self.histogram['hId'];self.scale_name=f':{self.histogram_id}:Scale'
        restraints=stio.GetRestraints(str(filename))
        if any(restraints.values()):
            raise ValueError('Additional native restraints need a separate declared objective')
        constraints,fixed=stio.ReadConstraints(str(filename));rigid=stio.GetRigidBodies(str(filename))
        rbids=rigid.get('RBIds',{'Vector':[],'Residue':[]})
        rbvary,rbparm=stio.GetRigidBodyModels(rigid,pFile=log)
        natoms,indices,pvary,pparm,pawley,ff,ef,orb,bl,mf,maxwave=stio.GetPhaseData(phases,restraints,rbids,pFile=log)
        controls.update(atomIndx=indices,Natoms=natoms,FFtables=ff,EFtables=ef,ORBtables=orb,
            BLtables=bl,MFtables=mf,maxSSwave=maxwave)
        hvary,hparm,extra=stio.GetHistogramPhaseData(phases,histograms,Controls=controls,pFile=log)
        tc,tf=stio.makeTwinFrConstr(phases,histograms,hvary)
        constraints+=tc;fixed+=tf;controls.update(extra)
        dvary,dparm,histdict,extra=stio.GetHistogramData(histograms,pFile=log);controls.update(extra)
        if controls[f':{self.histogram_id}:histType'] not in {'PNC','PNT'}:
            raise ValueError('Only reviewed commensurate neutron powder profiles are supported')
        self.vary=rbvary+pvary+hvary+dvary
        self.parms=dict(rbparm,**pparm);self.parms.update(hparm);self.parms.update(dparm)
        stio.GetFprime(controls,histograms)
        message=maps.EvaluateMultipliers(constraints,self.parms)
        if message:raise ValueError(message)
        error,warning,_,_=maps.GenerateConstraints(self.vary,constraints,fixed,self.parms)
        if error:raise ValueError(error)
        maps.normParms(self.parms);maps.Map2Dict(self.parms,self.vary);maps.Dict2Map(self.parms)
        self._constraint_signature=constraint_state_signature(maps)
        self.initial=np.asarray([self.parms[key] for key in self.vary],float)
        self.scale_index=self.vary.index(self.scale_name) if self.scale_name in self.vary else None
        if bool(self.scale_index is not None)!=bool(refine_scale):
            raise ValueError('Declared common-scale freedom differs from the native independent variables')
        self.source_scale=float(self.parms[self.scale_name])
        if not np.isfinite(self.source_scale) or self.source_scale<=0:
            raise ValueError('The source common histogram scale is invalid')
        nbackground=self.histogram['Background'][0][2]
        background_names=[f':{self.histogram_id}:Back;{i}' for i in range(nbackground)]
        actual=[key for key in self.vary if key.startswith(f':{self.histogram_id}:Back;')]
        if set(actual)!=(set(background_names) if refine_background else set()):
            raise ValueError('Declared background freedom differs from the native independent variables')
        self.background_indices=[self.vary.index(key) for key in actual]
        nuisance=set(actual)|({self.scale_name} if refine_scale else set())
        moment_names=[key for key in self.vary if '::AM' in key or key.startswith('::constr')]
        if not moment_names or set(self.vary)!=set(moment_names)|nuisance:
            raise ValueError('Only independent moments and explicitly declared nuisances may vary')
        self.moment_indices=[self.vary.index(key) for key in moment_names]
        self.moment_names=moment_names
        if len(moment_names)>24:raise ValueError('The declared independent magnetic basis exceeds this bounded adapter')
        self.args=([histograms,phases,restraints,rigid],self.parms,histdict,self.vary,controls,pawley,None)
        self.phases=phases;self.calls=0;self.derivative_calls=0;self.residual_mask=None
        from ..live.moment_basis import derive
        basis=derive(phases,self.parms,self.vary,self.initial,self.moment_indices,
            map_values=self._map,lattice=lattice,nuisance=nuisance)
        self.moment_basis=basis
        self.atoms=list(basis.atoms);self.component_maps=basis.components;self.cartesian_maps=basis.cartesian
        self.maximum_moment=maximum_moment
        self.fixed_scale=None if refine_scale else self.source_scale

    def _check(self):
        if self.deadline is not None and time.monotonic()>self.deadline:raise TimeoutError('The declared profiled optimizer time budget expired')
        if _active is not self:raise RuntimeError('The native process-global constraint map was replaced')
        if constraint_state_signature(self.maps)!=self._constraint_signature:
            raise RuntimeError('The native process-global constraint maps changed outside this adapter')

    def _map(self,vector):
        self._check();self.parms.update(zip(self.vary,vector));self.maps.Dict2Map(self.parms)

    def _all_moments(self):
        return np.asarray([[self.parms[key] for key in keys] for _,_,keys in self.atoms],float)

    def residual(self,moments,scale,background):
        self._check();vector=self.initial.copy();vector[self.moment_indices]=moments
        vector[self.background_indices]=background
        if self.scale_index is not None:vector[self.scale_index]=scale
        self.parms[self.scale_name]=scale
        result=np.ma.asarray(self.math.errRefine(vector,*self.args),dtype=float)
        mask=np.ma.getmaskarray(result)
        if self.residual_mask is None:self.residual_mask=mask.copy()
        elif not np.array_equal(mask,self.residual_mask):
            raise ValueError('Native fit masks changed during the starting-point calculation')
        values=np.asarray(np.ma.getdata(result))[~mask]
        if not np.isfinite(values).all():raise ValueError('The native weighted residual is nonfinite')
        self.calls+=1
        return values.copy()

    def magnetic_jacobian(self,moments,scale,background):
        """Use native constrained derivatives with the residual's exact mask.

        GSAS-II's analytic Jacobian applies wtFactor, while errRefine applies
        its square root. One independent histogram makes that conversion
        explicit; no observed value, weight or exclusion is changed.
        """
        self._check();vector=self.initial.copy();vector[self.moment_indices]=moments
        vector[self.background_indices]=background
        if self.scale_index is not None:vector[self.scale_index]=scale
        self.parms[self.scale_name]=scale
        derivative=np.ma.asarray(self.math.dervRefine(vector,*self.args),dtype=float)
        mask=self.residual_mask
        if mask is None or derivative.shape!=(len(self.vary),len(mask)):
            raise ValueError('Native magnetic derivative has a different residual domain')
        if np.ma.getmaskarray(derivative)[:,~mask].any():
            raise ValueError('Native magnetic derivative masks fitted observations')
        factor=float(self.args[4][f':{self.histogram_id}:wtFactor'])
        if not np.isfinite(factor) or factor<=0:
            raise ValueError('Native histogram weight factor is invalid')
        values=np.asarray(np.ma.getdata(derivative))[self.moment_indices][:,~mask]/np.sqrt(factor)
        if not np.isfinite(values).all():raise ValueError('Native magnetic derivative is nonfinite')
        self.derivative_calls+=1
        return values

    @contextmanager
    def _cache_lock(self, path, *, shared=False):
        """Coordinate disposable workers without reusing native globals."""
        try:import fcntl
        except ImportError:
            yield;return  # Atomic cache publication remains safe without locking.
        target=Path(path)
        try:
            target.parent.mkdir(parents=True,exist_ok=True)
            stream=target.with_suffix('.lock').open('a+b')
        except OSError:
            yield;return
        try:
            while True:
                self._check()
                try:fcntl.flock(stream,(fcntl.LOCK_SH if shared else fcntl.LOCK_EX)|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(.05)
            yield
        finally:
            fcntl.flock(stream,fcntl.LOCK_UN);stream.close()

    def build(self,*,heldouts=4,cache_path=None,cache_identity=None):
        if cache_path is not None and cache_identity:
            # Immutable, checked cache readers may validate native holdouts
            # concurrently. Only a rebuild needs the exclusive writer lock.
            if Path(cache_path).is_file():
                with self._cache_lock(cache_path,shared=True):
                    try:
                        return self._build(heldouts=heldouts,cache_path=cache_path,
                            cache_identity=cache_identity,cache_only=True)
                    except _CacheMiss:
                        pass
            with self._cache_lock(cache_path):
                return self._build(heldouts=heldouts,cache_path=cache_path,cache_identity=cache_identity)
        return self._build(heldouts=heldouts)

    def _build(self,*,heldouts=4,cache_path=None,cache_identity=None,skip_cache=False,
               use_derivative=None,compiler_note='',cache_only=False):
        started=time.monotonic();dimension=len(self.moment_indices);nb=len(self.background_indices)
        # GetAtomFXU replaces exact zero components with 1e-8. An evenly
        # spaced anchor can contain 1.0, so a minus-unit interpolation probe
        # would cross that artificial discontinuity. Avoid integral anchors.
        anchor=np.random.default_rng(77831).uniform(.6,1.6,dimension);zero_bg=np.zeros(nb)
        offset=self.residual(anchor,0.,zero_bg)
        cached=None;cache_note=''
        if cache_path is not None and cache_identity and not skip_cache:
            try:
                target=Path(cache_path)
                from .profiled_cache import bounded_npz
                if bounded_npz(target, cache_identity):
                    with np.load(target,allow_pickle=False) as data:
                        if (str(data['identity'])==cache_identity and
                                np.array_equal(data['offset'],offset) and
                                np.array_equal(data['maps'],self.cartesian_maps) and
                                data['quadratic'].shape==(len(offset),dimension,dimension) and
                                data['background'].shape==(len(offset),nb) and
                                data['nuclear'].shape==(len(offset),) and data['linear_error'].shape==() and
                                all(np.isfinite(data[k]).all() for k in ('nuclear','quadratic','background','linear_error'))):
                            cached={key:data[key].copy() for key in ('nuclear','quadratic','background','linear_error')}
            except (OSError,ValueError,KeyError,EOFError) as exc:cache_note=type(exc).__name__
        if cached is None and cache_only:
            raise _CacheMiss
        compilation=None;compiler='transported_cache' if cached is not None else 'native_profiles'
        if cached is None and use_derivative is None:
            try:
                from ..live.native_reflection import compile_reflection_quadratic
                self.residual(anchor,1.,zero_bg)
                physical,proof=compile_reflection_quadratic(math=self.math,lattice=self.lattice,
                    histogram=self.histogram,phases=self.phases,histdict=self.args[2],
                    parameters=self.parms,controls=self.args[4],basis=self.moment_basis,
                    check=self._check,histogram_name=self.histogram_name)
                begin,end=proof['fit_slice'];weights=np.asarray(np.ma.getdata(self.histogram['Data'][2]),float)[begin:end]
                factor=float(self.args[4].get(f':{self.histogram_id}:wtFactor',1.))
                if (self.residual_mask is None or self.residual_mask.shape!=(len(physical),)
                        or not np.isfinite(factor) or factor<=0
                        or not np.isfinite(weights[~self.residual_mask]).all()
                        or np.any(weights[~self.residual_mask]<=0)):
                    raise ValueError('Native reflection weights/masks differ from the measured residual')
                quadratic=physical[~self.residual_mask]*np.sqrt(weights[~self.residual_mask]*factor)[:,None,None]
                f0=self.residual(anchor,1.,zero_bg)-offset
                nuclear=f0-np.einsum('i,nij,j->n',anchor,quadratic,anchor)
                linear=self.magnetic_jacobian(anchor,1.,zero_bg).T
                linear_error=float(np.max(np.abs(linear-2*np.einsum('nij,j->ni',quadratic,anchor)))/max(1.,float(np.max(np.abs(f0)))))
                if not np.isfinite(linear_error) or linear_error>1e-8:
                    raise ValueError('Native reflection response fails the independent derivative check')
                compiler='native_reflection_amplitudes';compilation=proof
            except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
                compiler_note=str(exc)
        # Derivative cost is amortized only by larger bases. Measured cold
        # controls favor forward evaluations for the smaller spin spaces.
        if cached is None and compilation is None and (use_derivative is True or (use_derivative is None and dimension>=12)):
            try:
                from ..live.native_kernel_policy import compile_native_magnetic_kernel
                compiled=compile_native_magnetic_kernel(_CompilerObjective(self,zero_bg),anchor,
                    list(range(dimension)),method='derivative',checks=4)
                quadratic=compiled['quadratic'].transpose(2,0,1)
                nuclear=compiled['base']-offset
                f0=self.residual(anchor,1.,zero_bg)-offset
                linear=self.magnetic_jacobian(anchor,1.,zero_bg).T
                linear_error=float(np.max(np.abs(linear-2*np.einsum('nij,j->ni',quadratic,anchor)))/max(1.,float(np.max(np.abs(f0)))))
                compiler='native_derivatives';compilation=compiled['diagnostics']
            except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
                compiler_note=str(exc);compilation=None;compiler='native_profiles'
        if cached is None and compilation is None:
            f0=self.residual(anchor,1.,zero_bg)-offset
            quadratic=np.zeros((len(f0),dimension,dimension));linear=np.zeros((len(f0),dimension));plus=[]
            for i in range(dimension):
                step=np.eye(dimension)[i]
                fp=self.residual(anchor+step,1.,zero_bg)-offset
                fm=self.residual(anchor-step,1.,zero_bg)-offset
                plus.append(fp);quadratic[:,i,i]=(fp+fm-2*f0)/2;linear[:,i]=(fp-fm)/2
            for i in range(dimension):
                for j in range(i+1,dimension):
                    value=self.residual(anchor+np.eye(dimension)[i]+np.eye(dimension)[j],1.,zero_bg)-offset
                    quadratic[:,i,j]=quadratic[:,j,i]=(value-plus[i]-plus[j]+f0)/2
            nuclear=f0-np.einsum('i,nij,j->n',anchor,quadratic,anchor)
            linear_error=float(np.max(np.abs(linear-2*np.einsum('nij,j->ni',quadratic,anchor)))/max(1.,float(np.max(np.abs(f0)))))
        if cached is None:
            columns=[self.residual(anchor,0.,np.eye(nb)[i])-offset for i in range(nb)]
            background=np.column_stack(columns) if columns else np.empty((len(f0),0))
        else:
            nuclear,quadratic,background=(cached[k] for k in ('nuclear','quadratic','background'))
            linear_error=float(cached['linear_error'])
        normalization=float(self.histogram['Residuals']['sumwYo'])
        model=ProfiledMagneticObjective(offset=offset,nuclear=nuclear,quadratic=quadratic,
            background=background,moment_maps=self.cartesian_maps,maximum_moment=self.maximum_moment,
            normalization=normalization,fixed_scale=self.fixed_scale)
        rng=np.random.default_rng(18019);errors=[]
        for _ in range(heldouts):
            moments=rng.normal(size=dimension)
            maximum=np.max(np.linalg.norm(self.cartesian_maps@moments,axis=1))
            moments*=.6*self.maximum_moment/maximum
            scale=self.source_scale*rng.uniform(.8,1.2)
            bg=self.initial[self.background_indices]*rng.uniform(.8,1.2)
            native=self.residual(moments,scale,bg)
            errors.append(model.heldout_error(moments,scale,bg,native))
        if not errors or max(errors+[linear_error])>1e-8:
            if cached is not None:
                if cache_only:
                    raise _CacheMiss
                # A hint can never turn a recoverable cache mismatch into a
                # rejected scientific model. Rebuild once from native calls.
                return self._build(heldouts=heldouts,cache_path=cache_path,
                                   cache_identity=cache_identity,skip_cache=True)
            if compiler in {'native_derivatives','native_reflection_amplitudes'}:
                return self._build(heldouts=heldouts,cache_path=cache_path,
                    cache_identity=cache_identity,skip_cache=True,use_derivative=False,
                    compiler_note='Accelerated compilation failed the stricter native residual identity check')
            raise ValueError('The native profile failed held-out quadratic identity checks: '+
                str(dict(heldouts=errors,linear_identity_error=linear_error)))
        if cached is None and cache_path is not None and cache_identity:
            target=Path(cache_path);temporary=target.with_name(target.name+'.'+uuid.uuid4().hex)
            try:
                target.parent.mkdir(parents=True,exist_ok=True)
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream,identity=cache_identity,offset=offset,maps=self.cartesian_maps,
                        nuclear=nuclear,quadratic=quadratic,background=background,linear_error=linear_error)
                temporary.replace(target)
            except OSError as exc:cache_note=type(exc).__name__
            finally:temporary.unlink(missing_ok=True)
        self.model=model
        self.build_receipt=dict(schema='radar-native-profiled-start/v1',
            independent_variables=self.moment_names,magnetic_atoms=len(self.atoms),
            free_background_coefficients=nb,common_scale_free=self.fixed_scale is None,
            native_calls=self.calls,native_derivative_calls=self.derivative_calls,
            compiler=compiler,compiler_note=compiler_note,compiler_diagnostics=compilation,
            heldout_relative_errors=errors,linear_identity_error=linear_error,
            cache_hit=cached is not None,cache_identity=cache_identity,cache_note=cache_note,
            interpolation_anchor_seed=77831,
            seconds=time.monotonic()-started,weighted_residual_points=len(offset),
            native_convergence_declared=False,covariance_supplied=False)
        return model,self.build_receipt

    def apply_to_project(self,project,fit):
        """Apply a starting point, never a replacement native result receipt."""
        if not fit.success:raise ValueError('An unqualified numerical starting point cannot be applied')
        native=self.residual(fit.coefficients,fit.scale,fit.background)
        error=self.model.heldout_error(fit.coefficients,fit.scale,fit.background,native)
        if error>1e-8:raise ValueError('The final starting point disagrees with the native weighted objective')
        moments=self._all_moments()
        for (name,index,_),moment in zip(self.atoms,moments):
            phase=project.phase(name);cx=phase.data['General']['AtomPtrs'][0]
            phase.data['Atoms'][index][cx+4:cx+7]=moment.tolist()
        histogram=project.histogram(self.histogram_name)
        if self.scale_index is not None:histogram.data['Sample Parameters']['Scale'][0]=fit.scale
        for index,value in zip(self.background_indices,fit.background):
            coefficient=int(self.vary[index].split('Back;',1)[1])
            histogram.data['Background'][0][3+coefficient]=float(value)
        return dict(final_native_relative_error=error,weighted_cost=float(.5*(native@native)),
            maximum_moment=fit.maximum_moment,iterations=fit.iterations,evaluations=fit.evaluations,
            stationarity=fit.scaled_stationarity_norm,local_descent_found=fit.local_descent_found,
            native_convergence_declared=False,covariance_supplied=False)
