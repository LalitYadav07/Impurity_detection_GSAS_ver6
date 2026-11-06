"""Exact-cell GSAS-II profile response reuse; no profile interpolation."""
from collections import Counter, OrderedDict
import copy,hashlib,json,pickle,time
from pathlib import Path
import numpy as np
from scipy.sparse import csc_matrix
from .native_profile_capture import CaptureUnavailable, capture_response

def physics_policy():
    p=json.loads(Path(__file__).with_name('structural_forward_model_policy.json').read_text())
    if (p['instrument_parameters']['allow_refinement_in_structural_search']
            or p['profile_generation']['allow_custom_peak_shape_or_convolution']
            or p['profile_generation']['allow_interpolated_width_or_asymmetry_parameters']):
        raise ValueError('Fixed-instrument, native-profile policy is required')
    return {k:p[k] for k in ['instrument_parameters','profile_generation','surrogate_scope']}

class NativeProfileBank:
    """Memory-bounded reuse of exact-cell, fully GSAS-II-generated responses."""
    def __init__(self,source,maximum_bytes=64*1024*1024):
        if type(maximum_bytes) is not int or maximum_bytes<1:raise ValueError('Positive cache budget required')
        physics_policy()
        self.source=source;self.n=source.fast.native;self.maximum_bytes=maximum_bytes
        self.dimension=len(source.radii)
        self.cell_dimensions=getattr(source,'cell_dimensions',1)
        # Response generation is linear in reflections, not a tensor over
        # structural parameters. It also supports a pure-strain (zero-mode)
        # model. The byte budget still bounds each response cache.
        if not (1<=self.dimension<=256 and 1<=self.cell_dimensions<=min(6,self.dimension)):
            raise ValueError('Bounded structural and cell coordinates required')
        full,refs,bg,phase,hist=source.snapshot(np.zeros(self.dimension))
        self.hkls=refs[:,:3].copy();self.background=bg.copy();self.background.flags.writeable=False
        self.histogram_name=self.n.parts[0][0]
        self.phase_name=source.fast.slow.phase.name
        self.hfx=f":{hist['hId']}:";self.pfx=f"{phase['pId']}::";self.phfx=f"{phase['pId']}:{hist['hId']}:"
        if self.n.controls[self.hfx+'histType'] not in ('PNC','PNT'):raise ValueError('Qualified neutron powder histogram required')
        inst=hist['Instrument Parameters'][0]
        if any(self.hfx+k in self.n.vary for k in inst):raise ValueError('Instrument refinement is forbidden in this workstream')
        # Native exclusion masks affect its saved calculated arrays. Until a
        # complete masked response reconstruction is qualified, use the exact
        # native path rather than weakening the reconstruction check.
        excluded=bool(self.n.controls.get(self.hfx+'Exclude'))
        masked=any(np.ma.getmaskarray(a).any() for a in hist['Data'][:3])
        self.linear=(len(hist['Reflection Lists'])==1 and float(self.n.parms[self.phfx+'Extinction'])==0. and not excluded and not masked)
        self.strategy='native_profile_response_cache' if self.linear else 'full_native_context_fallback'
        self.profile_function=self.n.math.getPowderProfile
        self.fixed_identity=self._identity()
        self.cache=OrderedDict();self.bytes=0;self.hits=0;self.builds=0;self.evictions=0
        self.native_profile_calls=0;self.maximum_reconstruction_error=0.;self.build_seconds=[]
        self.response_methods=Counter();self.capture_fallbacks=Counter()

    def _identity(self):
        hist=self.n.histo_phases[0][self.histogram_name]
        phase=self.n.histo_phases[1][self.phase_name]
        # Cells and atomic coordinates are the only varying native geometry.
        # Pin all histogram and phase/histogram nuisance terms, instrument
        # dictionaries, fixed sample terms, grid, masks, symmetry and lookup
        # tables, including auxiliary tabulated profile corrections.
        dynamic_cell={self.pfx+f'A{i}' for i in range(6)}|{self.pfx+'Vol'}
        def dynamic(k):
            return k in dynamic_cell or k.startswith(tuple(self.pfx+s for s in ('dAx:','dAy:','dAz:','nv-')))
        # Occupancy, scattering species, ADPs and undisplaced coordinates are
        # table inputs too; pin them even though they do not affect peak shape.
        parms=sorted((k,v) for k,v in self.n.parms.items() if not dynamic(k))
        controls=sorted(self.n.controls.items())
        general={k:v for k,v in phase['General'].items() if k!='Cell'}
        content=(parms,controls,hist['Instrument Parameters'],hist['Sample Parameters'],hist['Background'],
            phase['Histograms'],general,phase.get('Atoms'),self.n.histdict,
            self.source.fast.x,self.source.fast.slice)
        return hashlib.sha256(pickle.dumps(content,protocol=5)).hexdigest()

    def check(self):
        self.n._check()
        if self.n.math.getPowderProfile is not self.profile_function:
            raise ValueError('Native profile implementation changed; reconstruct preparation')
        if self._identity()!=self.fixed_identity:
            raise ValueError('Fixed instrument/profile/grid context changed; reprepare, do not refine it')

    def matrix(self,cell):
        cells=np.atleast_1d(np.asarray(cell,float))
        if cells.shape!=(self.cell_dimensions,) or not np.isfinite(cells).all():
            raise ValueError('Finite exact cell-coordinate vector required')
        point=np.zeros(self.dimension);point[-self.cell_dimensions:]=cells
        self.source.check_domain(point);self.check()
        key=tuple(cells)  # exact values, no nearest-cell rounding
        if key in self.cache:
            self.hits+=1;self.cache.move_to_end(key);return self.cache[key]
        if not self.linear:raise ValueError('Nonlinear extinction cannot use a linear profile cache')
        began=time.perf_counter()
        full,refs,bg,phase,hist=self.source.snapshot(point);self.check()
        if not np.array_equal(refs[:,:3],self.hkls):raise ValueError('Native reflection identities changed; extend preparation first')
        if not np.allclose(bg,self.background,rtol=1e-13,atol=1e-13):raise ValueError('Fixed background changed')
        # Private dictionaries: native Pawley mode supplies unit F² without
        # monkeypatching StructureFactor2 or touching the Rietveld project.
        local_phase=copy.copy(phase);local_phase['General']=copy.copy(phase['General'])
        local_phase['General']['doPawley']=True
        local_hist=copy.copy(hist)
        reflection_dict=copy.copy(hist['Reflection Lists'][self.phase_name])
        local_hist['Reflection Lists']={self.phase_name:reflection_dict}
        parameters=self.n.parms.copy()
        controls=self.n.controls.copy()
        if controls.get('PhasePartials'):raise ValueError('Partial-output side effects not allowed while generating cached responses')
        axis=np.asarray(self.source.fast.x[self.source.fast.slice],float)
        reflection_dict['RefList']=refs.copy()
        lookup={}
        for index,row in enumerate(refs):
            h,k,l=map(int,row[:3]);lookup[self.pfx+f'{h},{k},{l}']=index
            parameters[self.pfx+f'PWLref:{index}']=1.
        try:
            matrix,background,receipt=capture_response(self.profile_function,self.n.math.G2pwd,
                parameters,self.n.histdict,axis,local_hist,{self.phase_name:local_phase},controls,
                lookup,histogram_name=self.histogram_name,phase_name=self.phase_name)
            self.native_profile_calls+=receipt['native_driver_calls']
            self.response_methods[receipt['method']]+=1
        except CaptureUnavailable as exc:
            # An unrecognized native call layout is an accelerator limitation,
            # not grounds to reject a scientifically supported candidate.
            self.native_profile_calls+=exc.native_driver_calls
            self.capture_fallbacks[str(exc)]+=1
            matrix,background=self._individual_responses(refs,axis,reflection_dict,parameters,
                local_hist,local_phase,controls)
            self.response_methods['individual native reflection calls']+=1
        if not np.allclose(background,bg[self.source.fast.slice],rtol=1e-13,atol=1e-13):
            raise ValueError('Native unit-response background differs from full profile')
        rebuilt=matrix@refs[:,9]+bg[self.source.fast.slice]
        err=float(np.linalg.norm(rebuilt-full[self.source.fast.slice])/max(np.linalg.norm((full-bg)[self.source.fast.slice]),1e-30))
        self.maximum_reconstruction_error=max(self.maximum_reconstruction_error,err)
        if not np.isfinite(err) or err>1e-9:raise ValueError(f'Native unit responses are not equivalent to the full Rietveld profile: {err}')
        self.check();self.builds+=1;self.build_seconds.append(time.perf_counter()-began)
        size=matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes
        for a in [matrix.data,matrix.indices,matrix.indptr]:a.flags.writeable=False
        if size<=self.maximum_bytes:
            while self.cache and self.bytes+size>self.maximum_bytes:
                _,old=self.cache.popitem(last=False);self.bytes-=old.data.nbytes+old.indices.nbytes+old.indptr.nbytes;self.evictions+=1
            self.cache[key]=matrix;self.bytes+=size
        return matrix

    def _individual_responses(self,refs,axis,reflection_dict,parameters,local_hist,local_phase,controls):
        """Original native construction for contexts without capture support."""
        indices=[];values=[];pointers=[0];background=self.background[self.source.fast.slice]
        for row in refs:
            reflection_dict['RefList']=row[None,:].copy()
            reflection_dict['RefList'][0,9]=1.
            h,k,l=map(int,row[:3])
            lookup={self.pfx+f'{h},{k},{l}':0}
            parameters[self.pfx+'PWLref:0']=1.
            signal,background=self.profile_function(parameters,self.n.histdict,axis,[],local_hist,
                {self.phase_name:local_phase},controls,lookup,histogram=self.histogram_name)
            self.native_profile_calls+=1
            if not np.allclose(background,self.background[self.source.fast.slice],rtol=1e-13,atol=1e-13):
                raise ValueError('Native unit-response background differs from full profile')
            signal=np.asarray(signal,float)
            if not np.isfinite(signal).all():raise ValueError('Nonfinite native profile response')
            nonzero=np.flatnonzero(signal)
            indices.append(nonzero.astype(np.int32));values.append(signal[nonzero])
            pointers.append(pointers[-1]+len(nonzero))
        matrix=csc_matrix((np.concatenate(values) if values else np.empty(0),
            np.concatenate(indices) if indices else np.empty(0,np.int32),np.asarray(pointers)),
            shape=(len(axis),len(refs)))
        return matrix,np.asarray(background,float)

    def diagnostics(self):
        return dict(strategy=self.strategy,entries=len(self.cache),estimated_bytes=self.bytes,maximum_bytes=self.maximum_bytes,
            hits=self.hits,builds=self.builds,evictions=self.evictions,native_profile_calls=self.native_profile_calls,
            maximum_reconstruction_error=self.maximum_reconstruction_error,
            median_build_seconds=float(np.median(self.build_seconds)) if self.build_seconds else None,
            fixed_context_sha256=self.fixed_identity,
            response_methods=dict(self.response_methods),capture_fallbacks=dict(self.capture_fallbacks),
            native_driver='GSASIIstrMath.getPowderProfile; private native fixed-intensity mode',
            custom_width_shape_position_or_convolution=False,interpolated_profile_parameters=False)

    def clear(self):
        """Start a fit with an empty cache, preserving cumulative diagnostics."""
        self.cache.clear();self.bytes=0
