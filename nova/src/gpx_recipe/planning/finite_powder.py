"""Finite-volume elastic powder counts with the local Bragg condition.

Every supplied scattering position has its own outgoing angle, Bragg
wavelength, source/factor/removal weight and recorded time. Incident rays
default to parallel +x; geometry can supply explicit incident unit directions.
Each detector coordinate is a surface-quadrature point; supplied
solid-angle weights may depend on scattering position. Pixel quadrature,
divergence, source and specimen physics require their own validation.
"""
from dataclasses import dataclass,replace
import math
import numpy as np
from scipy import sparse
from .joint_path_timing import FinitePathTiming
from .energy_response import TOF_US_PER_M_A
from .timing_response import binned_arrival_probability

VERSION='powgen-finite-volume-Bragg-count-response/3'


def _number(value,name,*,positive=True):
    if (isinstance(value,(bool,np.bool_,str,bytes)) or not np.isscalar(value)
            or not math.isfinite(float(value)) or (positive and float(value)<=0)):
        raise ValueError(name+' must be an explicit finite physical scalar')
    return float(value)


def _values(value,shape,name,*,nonnegative=False,maximum=math.inf):
    a=np.array(value,float,copy=True)
    if (a.shape!=shape or not np.isfinite(a).all() or (nonnegative and np.any(a<0))
            or np.any(a>maximum)):
        raise ValueError(name+' must have matching finite physical dimensions')
    return a


def _label(value,name):
    if type(value) is not str or not value.strip() or len(value)>512:
        raise ValueError('Explicit bounded '+name+' provenance is required')
    return value


def _probabilities(law,wavelength,offsets):
    wavelength,offsets=np.broadcast_arrays(np.asarray(wavelength,float),np.asarray(offsets,float))
    cdf,sf=[np.asarray(v,float) for v in law(wavelength,offsets)]
    shape=offsets.shape
    if (cdf.shape!=shape or sf.shape!=shape or any(not np.isfinite(v).all()
            or np.any(v<0) or np.any(v>1) for v in (cdf,sf))
            or not np.allclose(cdf+sf,1.,rtol=0,atol=2e-12)):
        raise ValueError('The wavelength-dependent residual law needs consistent CDF/SF arrays')
    return cdf,sf


@dataclass(frozen=True)
class PowderVolumeBins:
    mean_per_exposure: np.ndarray
    unobserved_covered_rate: float
    covered_before_cuts_rate: float
    metadata: dict


@dataclass(frozen=True)
class VolumeSourceResponse:
    source_nodes_a: np.ndarray
    coefficient_cm2_a: sparse.csr_matrix
    unobserved_cm2_a: np.ndarray
    before_cuts_cm2_a: np.ndarray
    metadata: dict


@dataclass(frozen=True)
class PowderVolumeLine:
    wavelength_a: np.ndarray
    arrival_tof_us: np.ndarray
    coefficient_cm2_a: np.ndarray
    source_density: np.ndarray
    residual_probability: object
    calibration: tuple
    metadata: dict
    max_bytes: int
    max_probability_evaluations: int
    incident_directions: np.ndarray | None = None
    angular_source_weights: np.ndarray | None = None

    def __post_init__(self):
        arrays=[np.array(v,float,copy=True) for v in (self.wavelength_a,self.arrival_tof_us,
            self.coefficient_cm2_a,self.source_density)]
        n=len(arrays[0]) if arrays[0].ndim==1 else -1
        if (not 0<=n<=2**20 or any(a.shape!=(n,) or not np.isfinite(a).all() for a in arrays)
                or np.any(arrays[0]<=0) or np.any(arrays[1]<=0)
                or any(np.any(a<0) for a in arrays[2:])):
            raise ValueError('Consistent finite local Bragg wavelengths, clocks and positive rate factors are required')
        if self.incident_directions is not None:
            directions=np.array(self.incident_directions,float,copy=True)
            if (directions.shape!=(n,3) or not np.isfinite(directions).all()
                    or not np.allclose(np.linalg.norm(directions,axis=1),1.,rtol=0,atol=1e-12)):
                raise ValueError('Actual incident unit directions must match the retained source positions')
            directions.setflags(write=False);object.__setattr__(self,'incident_directions',directions)
        if self.angular_source_weights is not None:
            if self.incident_directions is None:raise ValueError('Angular source weighting requires actual incident directions')
            angular=np.array(_values(self.angular_source_weights,(n,),'Angular source weights',nonnegative=True),copy=True)
            angular.setflags(write=False);object.__setattr__(self,'angular_source_weights',angular)
        with np.errstate(over='raise',invalid='raise'):
            rate=arrays[2]*arrays[3]
            if self.angular_source_weights is not None:rate=rate*self.angular_source_weights
        if not np.isfinite(rate.sum()):raise ValueError('Integrated volume rate exceeds its finite domain')
        for name,a in zip(('wavelength_a','arrival_tof_us','coefficient_cm2_a','source_density'),arrays):
            a.setflags(write=False);object.__setattr__(self,name,a)

    @property
    def position_rate_per_exposure(self):
        rate=self.coefficient_cm2_a*self.source_density
        return rate if self.angular_source_weights is None else rate*self.angular_source_weights

    @property
    def total_covered_rate_per_exposure(self):
        return float(self.position_rate_per_exposure.sum())

    def with_source_density(self,values,*,source_label):
        """Change only a covered source density, reusing the immutable geometry.

        Values are neutrons/(cm²*exposure*Å), at every retained local wavelength.
        Uncovered positions are not recovered or renormalized by this operation.
        """
        _label(source_label,'source-density')
        density=values(self.wavelength_a) if callable(values) else values
        density=_values(density,self.wavelength_a.shape,'Source density',nonnegative=True)
        return replace(self,source_density=density,metadata=dict(self.metadata,source_label=source_label))

    def with_angular_source_weights(self,values,*,angular_source_label):
        """Reweight a FIXED angular proposal, conditional on actual wavelength.

        Supply source angular density / proposal angular density, with both
        densities in the same coordinates. Proposals must cover the complete
        modeled source support. No per-reflection normalization is performed.
        Geometry and clocks remain unchanged; a new spectral density can be
        changed separately. A moving wavelength-dependent direction map would
        change the Bragg constraint and is not implemented by this operation.
        """
        _label(angular_source_label,'angular source')
        if self.incident_directions is None:raise ValueError('Angular source weighting requires actual incident directions')
        weights=values(self.wavelength_a,self.incident_directions) if callable(values) else values
        weights=_values(weights,self.wavelength_a.shape,'Angular source weights',nonnegative=True)
        return replace(self,angular_source_weights=weights,
            metadata=dict(self.metadata,angular_source_label=angular_source_label,angular_source_shape_fixed=True))

    def probability(self,raw_tof_us):
        """Arrival CDF/SF conditional on the modeled covered detection rate.

        The callback receives (local_wavelength, timing_offset). Wavelengths
        vary along the position axis of each chunk; the residual law can thus
        depend on actual energy without averaging an independent marginal.
        """
        rates=self.position_rate_per_exposure;total=float(rates.sum())
        if total==0:raise ValueError('A zero modeled rate has no conditional arrival distribution')
        query=np.asarray(raw_tof_us,float)
        if query.size>2000000 or np.isnan(query).any():raise ValueError('Bounded finite-or-infinite raw-time queries are required')
        active=np.flatnonzero(rates>0);n=len(active);flat=query.ravel()
        required=8*(len(flat)*4+n*6)+min(n,2048)*128
        if required>self.max_bytes:raise ValueError('Volume timing query exceeds its memory budget')
        finite=np.isfinite(flat);targets=np.flatnonzero(finite)
        if len(targets)*n>self.max_probability_evaluations:
            raise ValueError('Volume timing query exceeds its probability-evaluation budget')
        cdf=np.empty(len(flat));sf=np.empty(len(flat))
        cdf[np.isneginf(flat)],sf[np.isneginf(flat)]=0.,1.
        cdf[np.isposinf(flat)],sf[np.isposinf(flat)]=1.,0.
        pblock=min(n,2048);qblock=min(128,max(1,(self.max_bytes-required)//(pblock*128)))
        for begin in range(0,len(targets),qblock):
            ii=targets[begin:begin+qblock];first=np.zeros(len(ii));last=np.zeros(len(ii))
            for start in range(0,n,pblock):
                jj=active[start:start+pblock]
                offsets=flat[ii,None]-self.arrival_tof_us[None,jj]
                left,right=_probabilities(self.residual_probability,self.wavelength_a[None,jj],offsets)
                weights=rates[jj]/total
                first+=left@weights;last+=right@weights
            # A positive weighted sum of normalized laws is a probability,
            # but its near-one tail can overshoot by a summation ulp. Preserve
            # the independently computed SMALL tail and form its complement;
            # this also preserves tiny Gaussian/exponential tail probabilities.
            # No physical tail, source coverage or detection rate is rescaled.
            if not np.allclose(first+last,1.,rtol=0,atol=2e-12):
                raise ValueError('Volume timing accumulation violates complementary probability')
            small_left=first<=last
            cdf[ii]=np.where(small_left,first,1-last)
            sf[ii]=np.where(small_left,1-first,last)
        return cdf.reshape(query.shape),sf.reshape(query.shape)

    def bins(self,d_edges_a,*,window=None,max_intervals=200000):
        """Observed counts/exposure and lost covered rate, with all event cuts.

        Source values outside the declared modeled domain remain unknown in
        metadata. Their rate is not represented by the reported covered loss.
        """
        total=self.total_covered_rate_per_exposure;b,a,z,cosine=self.calibration
        if total>0:law=self.probability
        else:
            wavelength=float(self.wavelength_a[0]) if len(self.wavelength_a) else self.metadata['empty_validation_wavelength_a']
            law=lambda x:_probabilities(self.residual_probability,wavelength,x)
        result=binned_arrival_probability(d_edges_a,0.,law,difc_us_per_a=b,difa_us_per_a2=a,
            tzero_us=z,cos_two_theta=cosine,window=window,max_intervals=max_intervals,
            timing_law_label=self.metadata['timing_law_label'])
        mean=total*result.probability;mean.setflags(write=False)
        return PowderVolumeBins(mean,total*result.unobserved_probability,total,
            dict(self.metadata,**result.metadata,finite_volume_version=VERSION,
                lost_rate_covers_unmodeled_source=False,rate_unit='counts per declared exposure'))

    def source_matrix(self,d_edges_a,source_nodes_a,*,window=None,max_intervals=200000):
        """Positive piecewise-linear source response, without fixing source flux.

        Each position contributes to at most two source nodes. Retiming each
        occupied basis separately preserves source/wavelength/profile coupling.
        This is a bounded reference route, not an interactive kernel claim.
        """
        nodes=np.array(source_nodes_a,float,copy=True)
        domain=self.metadata['source_domain_a'];nbins=len(d_edges_a)-1
        if (nodes.ndim!=1 or not 2<=len(nodes)<=4097 or not np.isfinite(nodes).all()
                or np.any(np.diff(nodes)<=0) or nodes[0]!=domain[0] or nodes[-1]!=domain[1]
                or not 1<=nbins<=100000 or nbins*len(nodes)*32>self.max_bytes):
            raise ValueError('Bounded source nodes must span the exact declared source domain')
        self.with_source_density(np.zeros_like(self.wavelength_a),source_label='zero source validation').bins(
            d_edges_a,window=window,max_intervals=max_intervals)
        left=np.minimum(np.searchsorted(nodes,self.wavelength_a,side='right')-1,len(nodes)-2)
        alpha=(self.wavelength_a-nodes[left])/(nodes[left+1]-nodes[left])
        occupied=np.unique(np.r_[left,left+1]);rows=[];columns=[];values=[]
        before=np.zeros(len(nodes));lost=np.zeros(len(nodes))
        for j in occupied:
            basis=np.where(left==j,1-alpha,0.)+np.where(left+1==j,alpha,0.)
            result=self.with_source_density(basis,source_label='unit linear source node '+str(j)).bins(
                d_edges_a,window=window,max_intervals=max_intervals)
            active=np.flatnonzero(result.mean_per_exposure>0)
            rows.extend(active.tolist());columns.extend([int(j)]*len(active));values.extend(result.mean_per_exposure[active].tolist())
            before[j]=result.covered_before_cuts_rate;lost[j]=result.unobserved_covered_rate
        matrix=sparse.coo_matrix((values,(rows,columns)),shape=(nbins,len(nodes))).tocsr()
        for array in (nodes,before,lost,matrix.data,matrix.indices,matrix.indptr):array.setflags(write=False)
        return VolumeSourceResponse(nodes,matrix,lost,before,dict(self.metadata,
            source_basis='piecewise_linear',source_matrix_unit='cm²*Å',source_flux_fixed=False))


def powder_volume_line(geometry,d_a,multiplicity,cell_volume_a3,number_cells,*,
        solid_angle_sr,solid_angle_label,factor_squared_b,source_density,source_label,
        removal_cm_inv,detector_efficiency,mean_delay_us,residual_probability,timing_law_label,
        source_domain_a,l1_m,difc_us_per_a,calibration_cos_two_theta=None,
        difa_us_per_a2=0.,tzero_us=0.,clock_model='physical',clock_nominal_l2_m=None,
        relative_detector_gain=1.,relative_gain_label='unit relative detector gain',
        solid_angle_measure='solid_angle',
        angular_source_weight=None,angular_source_label='unit source density on supplied angular proposal',
        tof_us_per_m_a=TOF_US_PER_M_A,allow_partial_source_coverage=False,
        max_bytes=256*1024**2,max_probability_evaluations=20000000):
    """One reflection family and detector quadrature point in physical rates.

    A_lambda(r)=m*|F(lambda_h(r))|²*d³*lambda_h(r)/(4*pi*V).
    Multiplicity includes Friedel signs; factors are barns per declared cell.
    Whole-specimen cells, supplied solid angle, detection efficiency, source
    density and exp[-sum(mu(lambda_h)*(Li+Lo))] are integrated jointly with
    illumination f/N. No target scale is fitted. Removal returns (points,regions).

    The physical clock uses the actual elastic wavelength on both real legs.
    The calibrated_elastic_mean alternative evaluates the event calibration at
    lambda_h/(2*sin(theta_cal)) and adds the physical path difference about the
    declared clock reference. It is a phenomenological mean-clock hypothesis.
    Residual CDF/SF takes (wavelength,offset); its origin/mean must be explicit.

    Each detector coordinate is one supplied surface quadrature point. To
    integrate a finite pixel, sum such points with its actual calibration and
    solid-angle weights. A scalar solid angle assumes no specimen-position
    variation; an array supplies one weight per sampled position. Actual source,
    divergence, pixel geometry, removal and material physics remain unqualified.
    """
    if not isinstance(geometry,FinitePathTiming) or geometry.points_cm is None:
        raise ValueError('Finite path geometry with explicit scattering positions is required')
    d=_number(d_a,'Reflection spacing');m=_number(multiplicity,'Multiplicity')
    volume=_number(cell_volume_a3,'Cell volume');cells=_number(number_cells,'Whole specimen cell count')
    l1=_number(l1_m,'Incident reference path');k=_number(tof_us_per_m_a,'TOF conversion')
    b=_number(difc_us_per_a,'DIFC');a=_number(difa_us_per_a2,'DIFA',positive=False);zero=_number(tzero_us,'TZERO',positive=False)
    gain=_number(relative_detector_gain,'Relative detector gain',positive=False)
    if gain<0:raise ValueError('Relative detector gain must be nonnegative')
    _label(relative_gain_label,'relative detector gain')
    _label(angular_source_label,'angular source')
    if angular_source_weight is not None and (not callable(angular_source_weight) or geometry.incident_directions is None):
        raise ValueError('An angular source callback requires explicit incident directions')
    if (type(solid_angle_measure) is not str or solid_angle_measure not in ('solid_angle','unit_face_density')
            or (solid_angle_measure=='unit_face_density' and geometry.detector_targets_cm is None)):
        raise ValueError('Unit-face angular density requires explicit paired detector targets')
    if m!=round(m) or m>1000000:raise ValueError('An integer multiplicity including Friedel signs is required')
    for value,name in [(solid_angle_label,'solid-angle'),(source_label,'source-density'),(timing_law_label,'timing-law')]:_label(value,name)
    domain=np.asarray(source_domain_a,float)
    if (domain.shape!=(2,) or not np.isfinite(domain).all() or domain[0]<=0 or domain[1]<=domain[0]
            or type(allow_partial_source_coverage) is not bool or clock_model not in ('physical','calibrated_elastic_mean')
            or type(max_bytes) is not int or not 1024<=max_bytes<=2*1024**3
            or type(max_probability_evaluations) is not int or not 1<=max_probability_evaluations<=200000000
            or not all(callable(v) for v in (factor_squared_b,source_density,removal_cm_inv,detector_efficiency,mean_delay_us,residual_probability))):
        raise ValueError('Explicit bounded physical source, timing, response and resource contracts are required')
    points=geometry.points_cm;n=len(points);nr=geometry.incident_material_cm.shape[1]
    if n*(512+32*nr)>max_bytes:raise ValueError('Finite-volume line exceeds its working memory budget')
    detector=np.asarray(geometry.metadata.get('detector_position_cm'),float)
    reference=np.asarray(geometry.metadata.get('reference_point_cm'),float)
    if detector.shape!=(3,) or reference.shape!=(3,) or not np.isfinite(detector+reference).all():
        raise ValueError('Explicit detector/reference positions are required for the local Bragg condition')
    nominal=detector-reference;l2=np.linalg.norm(nominal)/100
    if l2<=0:raise ValueError('A nonzero outgoing reference flight path is required')
    cosine=nominal[0]/(l2*100) if calibration_cos_two_theta is None else _number(calibration_cos_two_theta,'Calibration cosine',positive=False)
    if not -1<=cosine<1:raise ValueError('A nonforward calibration angle is required')
    l2clock=l2 if clock_nominal_l2_m is None else _number(clock_nominal_l2_m,'Clock reference L2')
    actual_targets=detector if geometry.detector_targets_cm is None else geometry.detector_targets_cm
    rays=actual_targets-points;length=np.linalg.norm(rays,axis=1)
    if np.any(length<=0) or np.any(l1+geometry.incident_flight_offset_m<=0):
        raise ValueError('Positive actual incident and outgoing flight paths are required')
    direction=rays/length[:,None]
    # |n_out - n_in| = 2*sin(theta), stable also for small forward angles.
    incoming_direction=[1.,0.,0.] if geometry.incident_directions is None else geometry.incident_directions
    wave=d*np.minimum(np.linalg.norm(direction-incoming_direction,axis=1),2.)
    covered=(wave>=domain[0])&(wave<=domain[1]);outside=~covered
    if outside.any() and not allow_partial_source_coverage:
        raise ValueError('Local Bragg wavelengths extend outside the declared source domain; no extrapolation is allowed')
    indices=np.flatnonzero(covered);w=wave[covered]
    # A pointwise dOmega/(du dv) Jacobian on the unit face can exceed4pi
    # near a surface even though its integral is at most2pi. It is not a
    # probability or an already integrated per-target solid angle.
    maximum_omega=4*np.pi if solid_angle_measure=='solid_angle' else np.inf
    if np.isscalar(solid_angle_sr):omega=np.full(len(w),_number(solid_angle_sr,'Solid angle',positive=False))
    else:omega=_values(solid_angle_sr,(n,),'Position-dependent angular measure',nonnegative=True,maximum=maximum_omega)[covered]
    if np.any(omega<0) or np.any(omega>maximum_omega):raise ValueError('Physical solid angles must lie in [0,4*pi]')
    coefficient=np.zeros(len(w));density=np.zeros(len(w));arrival=np.zeros(len(w))
    if len(w):
        factors=_values(factor_squared_b(w),w.shape,'Local squared factors',nonnegative=True)
        density=_values(source_density(w),w.shape,'Local source density',nonnegative=True)
        efficiency=_values(detector_efficiency(w),w.shape,'Local detection efficiency',nonnegative=True,maximum=1.)
        mu=_values(removal_cm_inv(w),(len(w),nr),'Local material removal',nonnegative=True)
        delay=_values(mean_delay_us(w),w.shape,'Local mean delay')
        depth=np.sum((geometry.incident_material_cm[covered]+geometry.outgoing_material_cm[covered])*mu,axis=1)
        if geometry.integration_weights is None:
            coefficient=(cells*1e-24*m*d**3/(4*np.pi*volume)*geometry.illuminated_fraction/n*gain
                *factors*w*omega*efficiency*np.exp(-depth))
        else:
            coefficient=(cells*1e-24*m*d**3/(4*np.pi*volume)*geometry.integration_weights[covered]*gain
                *factors*w*omega*efficiency*np.exp(-depth))
        path_offset=geometry.incident_flight_offset_m[covered]+geometry.outgoing_flight_offset_m[covered]
        if clock_model=='physical':arrival=k*w*(l1+l2+path_offset)+delay
        else:
            dnom=w/np.sqrt(2*(1-cosine))
            if np.any(b+2*a*dnom<=0):raise ValueError('A local Bragg wavelength leaves the monotonic calibration branch')
            arrival=b*dnom+a*dnom**2+zero+k*w*(path_offset+l2-l2clock)+delay
        cdf,sf=_probabilities(residual_probability,w[None,:],np.array([[-np.inf],[np.inf]]))
        if not (np.allclose(cdf[0],0.,rtol=0,atol=2e-12) and np.allclose(cdf[1],1.,rtol=0,atol=2e-12)):
            raise ValueError('Each local wavelength requires a normalized continuous residual timing law')
    else:
        cdf,sf=_probabilities(residual_probability,float(domain.mean()),np.array([-np.inf,np.inf]))
        if not np.allclose(cdf,[0.,1.],rtol=0,atol=2e-12):raise ValueError('A normalized residual law is required')
    metadata=dict(version=VERSION,source_domain_a=domain.tolist(),source_label=source_label,
        paired_detector_surface_targets=geometry.detector_targets_cm is not None,
        explicit_incident_directions=geometry.incident_directions is not None,
        explicit_base_integration_weights=geometry.integration_weights is not None,
        angular_source_label=angular_source_label,angular_source_shape_fixed=angular_source_weight is not None,
        solid_angle_measure=solid_angle_measure,
        solid_angle_label=solid_angle_label,timing_law_label=timing_law_label,clock_model=clock_model,
        relative_detector_gain=gain,relative_gain_label=relative_gain_label,
        reflection_d_a=d,multiplicity=m,cell_volume_a3=volume,whole_specimen_cells=cells,
        integration_points=n,modeled_source_positions=len(indices),unmodeled_source_positions=int(outside.sum()),
        unmodeled_volume_fraction=(None if geometry.detector_targets_cm is not None or geometry.incident_directions is not None or geometry.integration_weights is not None else
            (0. if n==0 else geometry.illuminated_fraction*float(outside.mean()))),
        unmodeled_joint_unit_measure_fraction=(float(geometry.integration_weights[outside].sum()) if geometry.integration_weights is not None else
            (0. if n==0 else geometry.illuminated_fraction*float(outside.mean()))),
        missing_joint_measure_is_base_proposal_measure_not_source_rate=True,
        unmodeled_wavelength_range_a=None if not outside.any() else [float(wave[outside].min()),float(wave[outside].max())],
        missing_source_assumed_zero=False,illumination_included_once=True,partial_source_coverage=bool(outside.any()),
        empty_validation_wavelength_a=float(domain.mean()),position_and_energy_weights_independently_averaged=False,
        beam_divergence_included=geometry.incident_directions is not None,multiple_scattering_included=False,experimental_accuracy_certified=False)
    model=PowderVolumeLine(w,arrival,coefficient,density,residual_probability,(b,a,zero,float(cosine)),metadata,max_bytes,max_probability_evaluations,
        incident_directions=None if geometry.incident_directions is None else geometry.incident_directions[covered])
    return model if angular_source_weight is None else model.with_angular_source_weights(angular_source_weight,angular_source_label=angular_source_label)
