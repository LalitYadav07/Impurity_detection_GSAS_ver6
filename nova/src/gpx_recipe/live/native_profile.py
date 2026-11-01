"""Direct GSAS-II reduced-profile evaluator; no observation likelihood here.
Adapted from the independently checked local native objective. It retains the
native constraint maps and separates TOF profiles from count statistics.
"""
import io
import numpy as np
_active=None

class NativeReducedProfile:
    def __init__(self, filename):
        numerical_cell=False
        global _active
        from GSASII import GSASIIstrMain as main, GSASIIstrIO as stio
        from GSASII import GSASIIstrMath as math, GSASIImapvars as maps
        from GSASII import GSASIImpsubs as mp, GSASIIlattice as lattice
        from .gpx_index import install_fresh_gpx_index
        install_fresh_gpx_index()
        self.math, self.maps, self.lattice = math, maps, lattice
        self.filename = str(filename)
        _active = self
        mp.InitMP(); main.ptx.pyqlmninit(); maps.InitVars()
        log = io.StringIO()
        controls = stio.GetControls(str(filename)); controls['newLeBail'] = False
        self.controls = dict(controls)
        histograms, phases = stio.GetUsedHistogramsAndPhases(str(filename))
        # Resolve GPX random IDs before interpreting user constraints. A GUI or
        # G2Project normally does this first; a fresh direct worker does not.
        constraints, fixed = stio.ReadConstraints(str(filename))
        restraints = stio.GetRestraints(str(filename))
        rigid = stio.GetRigidBodies(str(filename))
        rbids = rigid.get('RBIds', {'Vector': [], 'Residue': []})
        rbvary, rbparm = stio.GetRigidBodyModels(rigid, pFile=log)
        phase_data = stio.GetPhaseData(phases, restraints, rbids, pFile=log)
        natoms, indices, pvary, pparm, self.pawley, ff, ef, orb, bl, mf, maxwave = phase_data
        self.controls.update(atomIndx=indices, Natoms=natoms, FFtables=ff,
                             EFtables=ef, ORBtables=orb, BLtables=bl,
                             MFtables=mf, maxSSwave=maxwave)
        hvary, hparm, extra = stio.GetHistogramPhaseData(phases, histograms,
                                                       Controls=self.controls, pFile=log)
        tc, tf = stio.makeTwinFrConstr(phases, histograms, hvary)
        constraints += tc; fixed += tf; self.controls.update(extra)
        dvary, dparm, self.histdict, extra = stio.GetHistogramData(histograms, pFile=log)
        self.controls.update(extra)
        self.vary = rbvary + pvary + hvary + dvary
        self.parms = dict(rbparm, **pparm); self.parms.update(hparm); self.parms.update(dparm)
        stio.GetFprime(self.controls, histograms)
        message = maps.EvaluateMultipliers(constraints, self.parms)
        if message: raise ValueError(message)
        error, warning, _, _ = maps.GenerateConstraints(self.vary, constraints, fixed, self.parms)
        if error: raise ValueError(error)
        maps.normParms(self.parms); maps.Map2Dict(self.parms, self.vary); maps.Dict2Map(self.parms)
        self.histo_phases = [histograms, phases, restraints, rigid]
        self.x0 = np.array([self.parms[k] for k in self.vary], float)
        self.numerical_cell = numerical_cell
        self.cell_indices = []
        cell_keys = [f"{p['pId']}::A{i}" for p in phases.values() for i in range(6)]
        base_cell = np.array([self.parms[k] for k in cell_keys])
        for j in range(len(self.vary)):
            trial = self.x0.copy(); trial[j] += 1e-5
            self.parms.update(zip(self.vary, trial)); maps.Dict2Map(self.parms)
            if np.max(abs(np.array([self.parms[k] for k in cell_keys])-base_cell))>1e-12:
                self.cell_indices.append(j)
        self.parms.update(zip(self.vary, self.x0)); maps.Dict2Map(self.parms)
        self.numerical_indices = self.cell_indices.copy() if numerical_cell else []
        self.derivative_validation = None
        self.parts=[]; self.original_weights={}
        for name in sorted(histograms):
            if not name.startswith('PWDR'):raise ValueError('Powder histograms required')
            hist=histograms[name];hid=hist['hId'];prefix=f':{hid}:'
            if self.controls[prefix+'histType'][1:2]!='N':raise ValueError('Neutron profile required')
            x,y,w,*_=hist['Data'];self.original_weights[name]=np.asarray(np.ma.filled(w,0.),float).copy()
            self.original_weights[name][np.ma.getmaskarray(x)]=0.
            for lo,hi in self.controls.get(prefix+'Exclude',[]):
                self.original_weights[name][(np.asarray(x)>=lo)&(np.asarray(x)<=hi)]=0.
            limits=self.controls[prefix+'Limits'];sl=slice(np.searchsorted(x,limits[0]),np.searchsorted(x,limits[1])+1)
            w[:]=1.;self.controls[prefix+'wtFactor']=1.;self.parts.append((name,sl))
        self.counts=np.concatenate([histograms[n]['Data'][1][sl] for n,sl in self.parts])
        self.factor=np.ones_like(self.counts)
    def _check(self):
        if _active is not self:
            raise RuntimeError('Native global constraint state was replaced by another adapter')

    def profile(self, values, jacobian=False):
        self._check(); values = np.asarray(values, float)
        # GSAS getPowderProfile evaluates structure factors before GetReflPos
        # updates the cached d spacings. When the cell changes, one-pass calls
        # otherwise use the preceding cell's d values for form factors/ADPs.
        # Synchronize the commensurate reflection cache before any intensity
        # calculation. Do not modify installed GSAS code.
        self.parms.update(zip(self.vary, values)); self.maps.Dict2Map(self.parms)
        histograms, phases = self.histo_phases[:2]
        # errRefine adds scalar summary entries to the histogram dictionary.
        for histogram_name, _ in self.parts:
            hist = histograms[histogram_name]
            for name, reflections in hist['Reflection Lists'].items():
                phase = phases[name]
                if phase['General'].get('Modulated', False):
                    raise ValueError('Modulated reflection synchronization is not implemented')
                prefix = f"{phase['pId']}::"; phfx = f"{phase['pId']}:{hist['hId']}:"
                correction = self.math.GetDij(phfx, phase['General']['SGData'], self.parms)
                a = [self.parms[prefix+f'A{i}']+correction[i] for i in range(6)]
                reciprocal, _ = self.lattice.A2Gmat(a)
                hkl = reflections['RefList'][:,:3]
                reciprocal_d_squared = np.einsum('ni,ij,nj->n',hkl,reciprocal,hkl)
                if np.any(reciprocal_d_squared<=0): raise ValueError('Nonpositive reflection metric')
                reflections['RefList'][:,4] = 1/np.sqrt(reciprocal_d_squared)
        args = (self.histo_phases, self.parms, self.histdict, self.vary, self.controls, self.pawley, None)
        residual = self.math.errRefine(values, *args)
        if len(residual) != len(self.counts):
            raise ValueError('Additional native penalties require a separate observation model')
        histograms = self.histo_phases[0]
        profile = np.concatenate([np.asarray(histograms[n]['Data'][3][sl], float) for n, sl in self.parts])
        mean = profile * self.factor
        if not jacobian: return mean
        derivative = np.asarray(self.math.dervRefine(values, *args), float)
        if derivative.shape != (len(values), len(mean)): raise ValueError('Unexpected native Jacobian shape')
        derivative = derivative * self.factor[None, :]
        if self.numerical_indices:
            for j in self.numerical_indices:
                h = self._difference_step(values, j, 1e-4)
                plus = values.copy(); minus = values.copy(); plus[j] += h; minus[j] -= h
                derivative[j] = (self.profile(plus)-self.profile(minus))/(2*h)
            self.profile(values)
        return mean, derivative

    def _difference_step(self, values, j, relative):
        key = self.vary[j]
        return relative * max(abs(values[j]), .1 if ':dA' in key or ':AUiso:' in key else .001)

