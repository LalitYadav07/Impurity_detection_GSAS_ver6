"""Correlated finite-volume removal and geometric recording-time probability.

The same scattering positions determine the incident/exit material lengths
and flight-path differences. Averaging attenuation and unweighted time shifts
independently is generally a different physical law. The incident wavefront is
parallel along +x; source/divergence and pixel integration remain separate.
The nominal detector solid angle is not changed by this helper.
"""
from dataclasses import dataclass
import math
import numpy as np
from .transport import (CylindricalRegion,RectangularBeam,illuminated_points,
    illuminated_points_from_unit_cube,ray_length_cm,segment_length_cm,_disjoint,_inside)
from .energy_response import TOF_US_PER_M_A
from .timing_response import interval_probability,binned_arrival_probability

VERSION='powgen-joint-finite-volume-path-timing/4'


def _scalar(value,name,*,positive=True):
    if (isinstance(value,(bool,np.bool_,str,bytes)) or not np.isscalar(value)
            or not math.isfinite(float(value)) or (positive and float(value)<=0)):
        raise ValueError(name+' must be an explicit finite physical scalar')
    return float(value)


@dataclass(frozen=True)
class WeightedArrivalBins:
    whole_powder_bin_weight: np.ndarray
    unobserved_whole_powder_weight: float
    metadata: dict


@dataclass(frozen=True)
class CorrelatedTimingLaw:
    whole_powder_weight: float
    normalized_position_weights: np.ndarray
    geometric_offsets_us: np.ndarray
    residual_probability: object
    metadata: dict
    max_bytes: int
    max_probability_evaluations: int

    def probability(self,offsets_us):
        """CDF/SF conditional on this weighted single scatter, without fluence.

        Multiplication by whole_powder_weight restores the joint measure.
        This conditional distribution is undefined when that weight is zero.
        A supplied residual law describes only remaining broadening about its
        declared clock origin; it must not already contain this volume effect.
        """
        if self.whole_powder_weight==0:
            raise ValueError('A zero-weight scatter has no conditional timing distribution')
        offsets=np.asarray(offsets_us,float)
        if np.isnan(offsets).any() or offsets.size>2000000:
            raise ValueError('Bounded timing offsets without NaNs are required')
        flat=offsets.ravel();n=len(self.geometric_offsets_us)
        if 8*(len(flat)*4+n*4)+min(n,2048)*96>self.max_bytes:
            raise ValueError('Correlated timing exceeds its declared working memory budget')
        cdf=np.empty(len(flat));sf=np.empty(len(flat))
        finite=np.isfinite(flat)
        cdf[np.isneginf(flat)],sf[np.isneginf(flat)]=0.,1.
        cdf[np.isposinf(flat)],sf[np.isposinf(flat)]=1.,0.
        targets=np.flatnonzero(finite)
        if len(targets)*n>self.max_probability_evaluations:
            raise ValueError('Correlated timing exceeds its declared probability-evaluation budget')
        # All positions are retained. Chunking changes working storage, not
        # the scattering-position law or a physical count-noise realization.
        position_block=min(n,2048)
        query_block=min(128,max(1,(self.max_bytes-8*(len(flat)*4+n*4))//(position_block*96)))
        if self.max_bytes<8*(len(flat)*4+n*4)+position_block*96:
            raise ValueError('Correlated timing exceeds its declared working memory budget')
        for start in range(0,len(targets),query_block):
            ii=targets[start:start+query_block];first=np.zeros(len(ii));last=np.zeros(len(ii))
            for begin in range(0,n,position_block):
                end=min(n,begin+position_block)
                argument=flat[ii,None]-self.geometric_offsets_us[None,begin:end]
                left,right=[np.asarray(v,float) for v in self.residual_probability(argument)]
                if (left.shape!=argument.shape or right.shape!=argument.shape
                        or any(not np.isfinite(v).all() or np.any(v<0) or np.any(v>1) for v in (left,right))
                        or not np.allclose(left+right,1.,rtol=0,atol=2e-12)):
                    raise ValueError('The residual timing law must return consistent physical CDF/SF arrays')
                weights=self.normalized_position_weights[begin:end]
                first+=left@weights;last+=right@weights
            cdf[ii],sf[ii]=first,last
        return cdf.reshape(offsets.shape),sf.reshape(offsets.shape)

    def bins(self,d_edges_a,center_tof_us,**kwargs):
        """Whole-powder bin weights, with correlated timing and cuts once.

        Supply the same calibration/window arguments as
        binned_arrival_probability, including a timing_law_label.
        This returns an attenuation-weighted measure, not a unit-area profile.
        """
        if self.whole_powder_weight==0:
            # Validate bin/calibration/cut inputs even in a dark geometry.
            result=binned_arrival_probability(d_edges_a,center_tof_us,self.residual_probability,**kwargs)
            weight=np.zeros_like(result.probability);unobserved=0.
        else:
            result=binned_arrival_probability(d_edges_a,center_tof_us,self.probability,**kwargs)
            weight=self.whole_powder_weight*result.probability
            unobserved=self.whole_powder_weight*result.unobserved_probability
        weight.setflags(write=False)
        return WeightedArrivalBins(weight,float(unobserved),dict(result.metadata,
            joint_path_timing_version=VERSION,whole_powder_weight=self.whole_powder_weight,
            attenuation_and_recorded_time_independently_averaged=False,
            illumination_included_once=True,experimental_accuracy_certified=False))


@dataclass(frozen=True)
class FinitePathTiming:
    incident_material_cm: np.ndarray
    outgoing_material_cm: np.ndarray
    incident_flight_offset_m: np.ndarray
    outgoing_flight_offset_m: np.ndarray
    illuminated_fraction: float
    metadata: dict
    points_cm: np.ndarray | None = None
    detector_targets_cm: np.ndarray | None = None
    incident_directions: np.ndarray | None = None
    integration_weights: np.ndarray | None = None

    def __post_init__(self):
        first=np.array(self.incident_material_cm,float,copy=True)
        last=np.array(self.outgoing_material_cm,float,copy=True)
        incoming=np.array(self.incident_flight_offset_m,float,copy=True)
        outgoing=np.array(self.outgoing_flight_offset_m,float,copy=True)
        fraction=_scalar(self.illuminated_fraction,'Illuminated fraction',positive=False)
        if (first.ndim!=2 or not 0<=len(first)<=2**20 or not 1<=first.shape[1]<=17
                or last.shape!=first.shape or incoming.shape!=(len(first),) or outgoing.shape!=incoming.shape
                or any(not np.isfinite(v).all() for v in (first,last,incoming,outgoing))
                or np.any(first<0) or np.any(last<0) or not 0<=fraction<=1
                or (len(first)==0 and fraction!=0)):
            raise ValueError('Consistent bounded nonnegative material paths and finite volume timing are required')
        for name,array in [('incident_material_cm',first),('outgoing_material_cm',last),
                ('incident_flight_offset_m',incoming),('outgoing_flight_offset_m',outgoing)]:
            array.setflags(write=False);object.__setattr__(self,name,array)
        object.__setattr__(self,'illuminated_fraction',fraction)
        if self.points_cm is not None:
            points=np.array(self.points_cm,float,copy=True)
            if points.shape!=(len(first),3) or not np.isfinite(points).all():
                raise ValueError('Explicit scattering positions must match the finite path arrays')
            points.setflags(write=False);object.__setattr__(self,'points_cm',points)
        if self.detector_targets_cm is not None:
            targets=np.array(self.detector_targets_cm,float,copy=True)
            if targets.shape!=(len(first),3) or not np.isfinite(targets).all():
                raise ValueError('Detector targets must match the paired scattering positions')
            targets.setflags(write=False);object.__setattr__(self,'detector_targets_cm',targets)
        if self.incident_directions is not None:
            directions=np.array(self.incident_directions,float,copy=True)
            if (directions.shape!=(len(first),3) or not np.isfinite(directions).all()
                    or not np.allclose(np.linalg.norm(directions,axis=1),1.,rtol=0,atol=1e-12)):
                raise ValueError('Incident directions must be explicit unit vectors at every retained position')
            directions.setflags(write=False);object.__setattr__(self,'incident_directions',directions)
        if self.integration_weights is not None:
            weights=np.array(self.integration_weights,float,copy=True)
            if (weights.shape!=(len(first),) or not np.isfinite(weights).all() or np.any(weights<0)
                    or not math.isclose(float(weights.sum()),fraction,rel_tol=2e-12,abs_tol=2e-12)):
                raise ValueError('Integration weights must contain the declared base illuminated measure once')
            weights.setflags(write=False);object.__setattr__(self,'integration_weights',weights)

    def law(self,incident_wavelength_a,incident_mu_cm_inv,residual_probability,*,
            outgoing_wavelength_a=None,outgoing_mu_cm_inv=None,
            tof_us_per_m_a=TOF_US_PER_M_A,max_bytes=64*1024**2,max_probability_evaluations=20000000):
        """Bind removal and residual timing at the actual incident/final energies.

        Material vectors list the sample first and each factory layer in order.
        Equal outgoing wavelength/removal are defaults for elastic scattering.
        The whole-powder weight contains illumination once. For use with the
        physical continuous count operator, use this weight as path_weight and
        this very law's probability as timing_probability; do not use an
        unweighted geometric marginal or multiply illumination again.
        """
        incoming=_scalar(incident_wavelength_a,'Incident wavelength')
        outgoing=incoming if outgoing_wavelength_a is None else _scalar(outgoing_wavelength_a,'Outgoing wavelength')
        k=_scalar(tof_us_per_m_a,'TOF constant')
        mu=np.asarray(incident_mu_cm_inv,float)
        other=mu if outgoing_mu_cm_inv is None else np.asarray(outgoing_mu_cm_inv,float)
        if (mu.shape!=(self.incident_material_cm.shape[1],) or other.shape!=mu.shape
                or any(not np.isfinite(v).all() or np.any(v<0) for v in (mu,other))
                or not callable(residual_probability) or type(max_bytes) is not int
                or not 1024<=max_bytes<=1024**3 or type(max_probability_evaluations) is not int
                or not 1<=max_probability_evaluations<=200000000):
            raise ValueError('Explicit physical material coefficients, residual law and bounded resources are required')
        normalization=interval_probability(-np.inf,np.inf,residual_probability)
        if abs(float(normalization)-1.)>2e-12:raise ValueError('A normalized continuous residual law is required')
        if 64*len(self.incident_flight_offset_m)>max_bytes:
            raise ValueError('Correlated timing-law construction exceeds its working memory budget')
        with np.errstate(over='raise',invalid='raise'):
            depth=self.incident_material_cm@mu+self.outgoing_material_cm@other
            offsets=k*(incoming*self.incident_flight_offset_m+outgoing*self.outgoing_flight_offset_m)
        if not np.isfinite(depth).all() or not np.isfinite(offsets).all():
            raise ValueError('Material optical depth or geometric timing exceeds its finite domain')
        raw=np.exp(-depth)
        if self.integration_weights is None:
            total=float(raw.sum());whole=0. if not len(raw) else self.illuminated_fraction*total/len(raw)
        else:
            raw=raw*self.integration_weights;total=float(raw.sum());whole=total
        weights=np.zeros_like(raw) if total==0 else raw/total
        for array in (weights,offsets):array.setflags(write=False)
        return CorrelatedTimingLaw(whole,weights,offsets,residual_probability,
            dict(version=VERSION,integration_points=len(raw),illuminated_fraction=self.illuminated_fraction,
                explicit_base_measure_weights=self.integration_weights is not None,
                zero_floating_transmission_points=int((raw==0).sum()),
                incident_wavelength_a=incoming,outgoing_wavelength_a=outgoing,
                multiple_scattering_included=False,source_pulse_law_measured=False,
                experimental_accuracy_certified=False),max_bytes,max_probability_evaluations)


def finite_path_timing(sample,beam,detector_position_cm,*,layers=(),reference_point_cm=(0.,0.,0.),
        power=12,seed=0,max_bytes=256*1024**2):
    """Trace one detector's finite-position attenuation and flight-time support.

    Incident fluence is defined upstream of all supplied material regions on a
    parallel +x wavefront. Reference flight paths refer to reference_point_cm;
    the shifts are dx and |detector-point|-|detector-reference|. Each outgoing
    material path stops at the detector. Regions must be nonoverlapping.
    Geometry, layer materials, beam footprint and source/divergence remain
    explicit unqualified inputs, not inferred from a standard's fitted width.
    """
    if not isinstance(sample,CylindricalRegion) or not isinstance(beam,RectangularBeam):
        raise ValueError('Explicit finite sample and beam geometry are required')
    detector=np.asarray(detector_position_cm,float);reference=np.asarray(reference_point_cm,float)
    if (detector.shape!=(3,) or reference.shape!=(3,) or not np.isfinite(detector).all()
            or not np.isfinite(reference).all() or max(np.max(abs(detector)),np.max(abs(reference)))>1e8
            or np.array_equal(detector,reference) or _inside(sample.outer,detector[None,:]).any()
            or len(layers)>16 or any(not isinstance(r,CylindricalRegion) for r in layers)
            or type(power) is not int or not 8<=power<=20 or type(seed) is not int or not 0<=seed<2**32
            or type(max_bytes) is not int or not 1024<=max_bytes<=1024**3):
        raise ValueError('Bounded external detector, reference, layers and integration settings are required')
    regions=[sample,*layers]
    if any(not _disjoint(a,b) for j,a in enumerate(regions) for b in regions[j+1:]):
        raise ValueError('Material regions overlap or their disjointness is unverified')
    if any(_inside(r.outer,detector[None,:]).any() and (r.void is None or not _inside(r.void,detector[None,:]).any()) for r in layers):
        raise ValueError('A detector cannot lie inside material')
    if 2**power*(384+32*len(regions))>max_bytes:
        raise ValueError('Joint path/timing construction exceeds its working memory budget')
    points,fraction=illuminated_points(sample,beam,power=power,seed=seed)
    incoming=np.column_stack([ray_length_cm(r,points,[-1.,0.,0.]) for r in regions]) if len(points) else np.zeros((0,len(regions)))
    outgoing=np.column_stack([segment_length_cm(r,points,detector) for r in regions]) if len(points) else incoming.copy()
    # Rationalized length difference avoids subtracting two nearly equal long
    # detector distances as the specimen tends to a point or L2 tends to infinity.
    delta=points-reference;nominal=detector-reference
    distance=np.linalg.norm(detector-points,axis=1);base=np.linalg.norm(nominal)
    outgoing_offset=(np.sum(delta*delta,axis=1)-2*delta@nominal)/(distance+base)/100
    incident_offset=delta[:,0]/100
    for array in (incoming,outgoing,incident_offset,outgoing_offset):array.setflags(write=False)
    return FinitePathTiming(incoming,outgoing,incident_offset,outgoing_offset,fraction,
        dict(version=VERSION,integration_power=power,scramble_seed=seed,integration_points=len(points),
            reference_point_cm=reference.tolist(),detector_position_cm=detector.tolist(),
            material_regions=len(regions),uniform_rectangular_parallel_beam=True,
            detector_solid_angle_variation_included=False,beam_divergence_included=False,
            multiple_scattering_included=False,experimental_accuracy_certified=False),points)


def paired_path_timing(sample,beam,nominal_detector_position_cm,detector_targets_cm,unit_volume_points,*,
        layers=(),reference_point_cm=(0.,0.,0.),max_bytes=256*1024**2):
    """Trace paired volume/target quadrature about a FIXED pixel calibration.

    Each target is a detector-surface quadrature location. Its sample-relative
    angle changes Bragg wavelength; it does not acquire its own pixel DIFC or
    nominal scattering angle. Flight offsets are about the one supplied
    nominal detector position, including target displacement exactly once.
    Cubature weights/solid-angle Jacobians belong to the count operator.
    """
    if not isinstance(sample,CylindricalRegion) or not isinstance(beam,RectangularBeam):
        raise ValueError('Explicit finite sample and beam geometry are required')
    detector=np.asarray(nominal_detector_position_cm,float);reference=np.asarray(reference_point_cm,float)
    targets=np.asarray(detector_targets_cm,float);unit=np.asarray(unit_volume_points,float)
    if (detector.shape!=(3,) or reference.shape!=(3,) or not np.isfinite(detector+reference).all()
            or np.array_equal(detector,reference) or targets.ndim!=2 or targets.shape[1]!=3
            or not 1<=len(targets)<=2**20 or unit.shape!=targets.shape or not np.isfinite(targets).all()
            or max(np.max(abs(detector)),np.max(abs(reference)),np.max(abs(targets)))>1e8
            or len(layers)>16 or any(not isinstance(r,CylindricalRegion) for r in layers)
            or type(max_bytes) is not int or not 1024<=max_bytes<=1024**3):
        raise ValueError('Bounded paired targets, nominal detector and volume coordinates are required')
    regions=[sample,*layers]
    if any(not _disjoint(a,b) for j,a in enumerate(regions) for b in regions[j+1:]):
        raise ValueError('Material regions overlap or their disjointness is unverified')
    for r in regions:
        test=np.vstack([detector,targets]);inside=_inside(r.outer,test)
        if r.void is not None:inside&=~_inside(r.void,test)
        if inside.any():raise ValueError('Detector targets cannot lie inside material')
    if len(targets)*(512+32*len(regions))>max_bytes:
        raise ValueError('Paired path construction exceeds its declared working memory budget')
    points,fraction=illuminated_points_from_unit_cube(sample,beam,unit)
    if len(points)==0:targets=targets[:0]
    incoming=np.column_stack([ray_length_cm(r,points,[-1.,0.,0.]) for r in regions]) if len(points) else np.zeros((0,len(regions)))
    outgoing=np.column_stack([segment_length_cm(r,points,targets) for r in regions]) if len(points) else incoming.copy()
    nominal=detector-reference;base=np.linalg.norm(nominal)
    shift=targets-detector-(points-reference)
    distance=np.linalg.norm(targets-points,axis=1)
    outgoing_offset=(np.sum(shift*shift,axis=1)+2*shift@nominal)/(distance+base)/100
    incident_offset=(points[:,0]-reference[0])/100
    return FinitePathTiming(incoming,outgoing,incident_offset,outgoing_offset,fraction,
        dict(version=VERSION,integration_points=len(points),reference_point_cm=reference.tolist(),
            detector_position_cm=detector.tolist(),material_regions=len(regions),
            paired_detector_surface_targets=True,fixed_pixel_calibration_reference=True,
            uniform_rectangular_parallel_beam=True,detector_solid_angle_variation_included=False,
            beam_divergence_included=False,multiple_scattering_included=False,experimental_accuracy_certified=False),
        points,targets)
