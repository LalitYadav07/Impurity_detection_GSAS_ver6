"""Incident angular phase space, clipped illumination and finite ray clocks.

The footprint is defined on the x=reference_x plane. Spectral angular flux is
fluence per area normal to each ray, per declared angular coordinate measure.
Uniform yaw/pitch proposals are fixed independently of wavelength; an actual
conditional source density is evaluated AFTER the local Bragg wavelength.
Neither optical transport nor a measured source/guide clock is inferred here.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.special import erf,log_ndtr,logsumexp
from .transport import (SolidCylinder,CylindricalRegion,RectangularBeam,_disk_area_below,
    _disjoint,_inside,ray_length_cm,segment_length_cm)
from .joint_path_timing import FinitePathTiming

VERSION='powgen-incident-angular-phase-space/1'

def _unit(values,columns):
    a=np.asarray(values,float)
    if (a.ndim!=2 or a.shape[1]!=columns or not 1<=len(a)<=2**20
            or not np.isfinite(a).all() or np.any(a<0) or np.any(a>1)):
        raise ValueError('Bounded explicit unit-coordinate quadrature is required')
    return a

def _directions(values,n):
    d=np.broadcast_to(np.asarray(values,float),(n,3))
    if (not np.isfinite(d).all() or not np.allclose(np.linalg.norm(d,axis=1),1.,rtol=0,atol=1e-12)
            or np.any(d[:,0]<=1e-8)):
        raise ValueError('Explicit forward unit incident directions are required')
    return d

def directions_from_angles(yaw_rad,pitch_rad):
    """Yaw about vertical, then elevation; density coordinates are yaw/pitch."""
    yaw,pitch=np.broadcast_arrays(np.asarray(yaw_rad,float),np.asarray(pitch_rad,float))
    if (yaw.size>2**20 or not np.isfinite(yaw+pitch).all()
            or np.any(abs(yaw)>=np.pi/2) or np.any(abs(pitch)>=np.pi/2)):
        raise ValueError('Bounded forward yaw/pitch angles in radians are required')
    d=np.stack([np.cos(pitch)*np.cos(yaw),np.cos(pitch)*np.sin(yaw),np.sin(pitch)],axis=-1)
    if np.any(d[...,0]<=1e-8):raise ValueError('Incident directions are too close to the reference plane')
    return d

def _log_normal_interval(low,high):
    """Stable normalized Gaussian mass, including a narrow central interval."""
    low,high=np.broadcast_arrays(low,high);out=np.empty(low.shape)
    central=(low<=0)&(high>=0)
    out[central]=np.log(.5*(erf(high[central]/np.sqrt(2))-erf(low[central]/np.sqrt(2))))
    remaining=~central
    first=np.where(low>0,log_ndtr(-low),log_ndtr(high));last=np.where(low>0,log_ndtr(-high),log_ndtr(low))
    with np.errstate(divide='ignore',invalid='ignore'):out[remaining]=(first+np.log(-np.expm1(last-first)))[remaining]
    # A short interval away from zero needs a local quadrature instead of
    # subtracting two nearly equal CDFs. This is a numerical Gaussian integral,
    # not a normalization of unknown instrument/source support.
    span=high-low;mid=(high+low)/2;small=(span<1e-3)&(abs(mid)*span<.01)&remaining
    if small.any():
        z,w=np.polynomial.legendre.leggauss(5);half=span[small]/2
        out[small]=np.log(half)+logsumexp(np.log(w)-.5*(mid[small,None]+half[:,None]*z)**2,axis=1)-.5*np.log(2*np.pi)
    if not np.isfinite(out).all():raise ValueError('Angular Gaussian normalization leaves its finite domain')
    return out

@dataclass(frozen=True)
class UniformAngleProposal:
    yaw_bounds_rad: tuple
    pitch_bounds_rad: tuple

    def __post_init__(self):
        for name in ('yaw_bounds_rad','pitch_bounds_rad'):
            bounds=np.asarray(getattr(self,name),float)
            if (bounds.shape!=(2,) or not np.isfinite(bounds).all() or bounds[1]<bounds[0]
                    or np.any(abs(bounds)>=np.pi/2-1e-4)):
                raise ValueError('Ordered finite forward angular proposal bounds are required')
            object.__setattr__(self,name,tuple(float(x) for x in bounds))

    def sample(self,unit_angles):
        u=_unit(unit_angles,2)
        yaw=self.yaw_bounds_rad[0]+np.ptp(self.yaw_bounds_rad)*u[:,0]
        pitch=self.pitch_bounds_rad[0]+np.ptp(self.pitch_bounds_rad)*u[:,1]
        return directions_from_angles(yaw,pitch)

    def normal_weight(self,wavelength_a,directions,*,sigma_yaw_rad,sigma_pitch_rad,
            mean_yaw_rad=0.,mean_pitch_rad=0.):
        """Conditional truncated-Gaussian angular density / fixed proposal.

        Parameters may be scalars or functions of the actual wavelength. The
        source distribution is explicitly normalized on these aperture bounds.
        A finite proposal cannot represent a zero-width Dirac source; use a
        matching zero-width proposal for that limit. No reflected intensity or
        sampled source weight is normalized to its computed sample mean.
        """
        wave=np.asarray(wavelength_a,float)
        if wave.ndim!=1 or len(wave)>2**20 or not np.isfinite(wave).all() or np.any(wave<=0):
            raise ValueError('Bounded positive actual wavelengths are required')
        if not len(wave):return np.empty(0)
        d=_directions(directions,len(wave));angles=[np.arctan2(d[:,1],d[:,0]),np.arcsin(np.clip(d[:,2],-1,1))];ratio=np.ones(len(wave))
        for x,bounds,sigma,mean in zip(angles,[self.yaw_bounds_rad,self.pitch_bounds_rad],
                [sigma_yaw_rad,sigma_pitch_rad],[mean_yaw_rad,mean_pitch_rad]):
            sigma=np.broadcast_to(np.asarray(sigma(wave) if callable(sigma) else sigma,float),wave.shape)
            mean=np.broadcast_to(np.asarray(mean(wave) if callable(mean) else mean,float),wave.shape)
            if (not np.isfinite(sigma+mean).all() or np.any(sigma<0) or np.any(sigma>1.) or np.any(abs(mean)>np.pi)):
                raise ValueError('Finite physical angular Gaussian parameters are required')
            lo,hi=bounds;width=hi-lo
            if width==0:
                if np.any(sigma!=0) or np.any(abs(mean-lo)>1e-12):raise ValueError('A degenerate proposal cannot cover a finite-width or displaced angular source')
                ratio*=np.isclose(x,lo,rtol=0,atol=1e-12);continue
            if np.any(sigma<1e-9):raise ValueError('A finite angular proposal cannot resolve a zero-width source')
            logz=_log_normal_interval((lo-mean)/sigma,(hi-mean)/sigma)
            logweight=np.log(width/sigma)-.5*np.log(2*np.pi)-.5*((x-mean)/sigma)**2-logz
            value=np.exp(logweight)
            inside=(x>=lo-8*np.finfo(float).eps)&(x<=hi+8*np.finfo(float).eps)
            ratio*=np.where(inside,value,0.)
        if not np.isfinite(ratio).all():raise ValueError('Angular source weighting exceeds its finite domain')
        return ratio

@dataclass(frozen=True)
class BeamVolumePoints:
    points_cm: np.ndarray
    integration_weights: np.ndarray
    input_indices: np.ndarray
    metadata: dict

    def __post_init__(self):
        p=np.array(self.points_cm,float,copy=True);w=np.array(self.integration_weights,float,copy=True);i=np.array(self.input_indices,np.int64,copy=True)
        if (p.shape!=(len(w),3) or i.shape!=w.shape or not np.isfinite(p).all() or not np.isfinite(w).all()
                or np.any(w<=0) or float(w.sum())>1+2e-12):raise ValueError('Consistent positive base beam-volume measure is required')
        for name,a in [('points_cm',p),('integration_weights',w),('input_indices',i)]:a.setflags(write=False);object.__setattr__(self,name,a)

def illuminated_beam_points(sample,beam,unit_volume_points,incident_directions,*,reference_x_cm=0.,volume_map='strip_cdf'):
    """Weighted uniform-volume map under a tilted x-plane beam footprint.

    A rotated annular strip is sampled in xy, then the EXACT allowed vertical
    interval is sampled. Its variable length remains in the base measure; it
    is not replaced by an average. Zero-volume points are discarded while
    retaining the original generated denominator through their weights.
    """
    u=_unit(unit_volume_points,3);n=len(u);direction=_directions(incident_directions,n)
    if (not isinstance(sample,CylindricalRegion) or not isinstance(beam,RectangularBeam) or not math.isfinite(reference_x_cm)
            or type(volume_map) is not str or volume_map not in ('strip_cdf','polar_overfilled')):
        raise ValueError('Explicit finite powder, footprint and reference plane are required')
    outer,void=sample.outer,sample.void
    if void and (void.center_cm!=outer.center_cm or void.height_cm!=outer.height_cm):raise ValueError('Beam sampling supports solid cylinders or coaxial full-height annuli')
    R=outer.radius_cm;r=0. if void is None else void.radius_cm;cx,cy,cz=outer.center_cm
    slope_y=direction[:,1]/direction[:,0];slope_z=direction[:,2]/direction[:,0];scale=np.hypot(1.,slope_y)
    center_on_plane=cy-slope_y*(cx-reference_x_cm)
    if volume_map=='polar_overfilled':
        if np.any(abs(center_on_plane-beam.center_y_cm)+R*scale>beam.width_cm/2):
            raise ValueError('Polar sampling requires the complete radial section to be illuminated for every direction')
        radial=np.sqrt(r*r+(R*R-r*r)*u[:,0]);angle=2*np.pi*u[:,1]
        x=cx+radial*np.cos(angle);y=cy+radial*np.sin(angle);area_fraction=np.ones(n)
    else:
        low=np.clip((beam.center_y_cm-beam.width_cm/2-center_on_plane)/scale,-R,R)
        high=np.clip((beam.center_y_cm+beam.width_cm/2-center_on_plane)/scale,-R,R)
        def area(x):return _disk_area_below(x,R)-_disk_area_below(x,r)
        first,last=area(low),area(high);area_fraction=(last-first)/(np.pi*(R*R-r*r));area_fraction=np.where((low==-R)&(high==R),1.,area_fraction)
        if np.any(area_fraction < -2e-12) or np.any(area_fraction>1+2e-12):raise ValueError('Projected beam area violates its geometric measure')
        area_fraction=np.clip(area_fraction,0.,1.);target=first+(last-first)*u[:,0];left,right=low.copy(),high.copy()
        for _ in range(48):
            mid=(left+right)/2;greater=area(mid)>target;right=np.where(greater,mid,right);left=np.where(greater,left,mid)
        s=(left+right)/2;xo=np.sqrt(np.maximum(R*R-s*s,0.));xi=np.sqrt(np.maximum(r*r-s*s,0.));chord=xo-xi;offset=2*chord*u[:,1]
        t=np.where(offset<chord,-xo+offset,xi+offset-chord);x=cx+(t-slope_y*s)/scale;y=cy+(slope_y*t+s)/scale
    low_z=np.maximum(cz-outer.height_cm/2,beam.center_z_cm-beam.height_cm/2+(x-reference_x_cm)*slope_z)
    high_z=np.minimum(cz+outer.height_cm/2,beam.center_z_cm+beam.height_cm/2+(x-reference_x_cm)*slope_z)
    length=np.maximum(0.,high_z-low_z);weights=area_fraction*(length/outer.height_cm)/n;indices=np.flatnonzero(weights>0)
    points=np.c_[x[indices],y[indices],low_z[indices]+length[indices]*u[indices,2]]
    return BeamVolumePoints(points,weights[indices],indices,dict(version=VERSION,generated_points=n,retained_points=len(indices),
        reference_plane_x_cm=float(reference_x_cm),base_proposal_illuminated_measure=float(weights.sum()),
        volume_map=volume_map,base_measure_is_not_angular_source_rate=True,footprint_on_x_plane=True,zero_measure_points_not_renormalized=True,experimental_accuracy_certified=False))

def beam_path_timing(sample,beam,nominal_detector_position_cm,detector_targets_cm,unit_volume_points,
        incident_directions,*,layers=(),reference_point_cm=(0.,0.,0.),incident_reference_path_offset_m=0.,
        incident_clock_label='isochronous x-plane reference hypothesis',volume_map='strip_cdf',max_bytes=256*1024**2):
    """Trace actual incident/outgoing rays about a fixed pixel and x plane.

    The reference-plane arrival is an explicit phase-space input, not derived
    from direction alone. Advance from that plane to r is dx/n_x. Supplied
    reference path offsets can represent guide path/direction correlations;
    zero means an isochronous reference plane, NOT a measured guide model.
    Fluence precedes all local material regions; incoming paths are rays.
    """
    unit=_unit(unit_volume_points,3);n=len(unit);d=_directions(incident_directions,n)
    detector=np.asarray(nominal_detector_position_cm,float);reference=np.asarray(reference_point_cm,float);targets=np.asarray(detector_targets_cm,float)
    offsets=np.broadcast_to(np.asarray(incident_reference_path_offset_m,float),(n,))
    if (not isinstance(sample,CylindricalRegion) or not isinstance(beam,RectangularBeam)
            or detector.shape!=(3,) or reference.shape!=(3,) or targets.shape!=(n,3)
            or any(not np.isfinite(a).all() or np.max(abs(a),initial=0)>1e8 for a in [detector,reference,targets])
            or np.array_equal(detector,reference) or not np.isfinite(offsets).all() or np.any(abs(offsets)>1e4)
            or not isinstance(incident_clock_label,str) or not 1<=len(incident_clock_label)<=256
            or len(layers)>16 or any(not isinstance(r,CylindricalRegion) for r in layers)
            or type(max_bytes) is not int or not 1024<=max_bytes<=1024**3):raise ValueError('Explicit bounded angular phase-space geometry and clock inputs are required')
    regions=[sample,*layers]
    if n*(1024+32*len(regions))>max_bytes:raise ValueError('Angular geometry exceeds its declared working memory budget')
    if any(not _disjoint(a,b) for j,a in enumerate(regions) for b in regions[j+1:]):raise ValueError('Material regions overlap or their disjointness is unverified')
    for body in regions:
        test=np.vstack([detector,targets]);inside=_inside(body.outer,test)
        if body.void is not None:inside&=~_inside(body.void,test)
        if inside.any():raise ValueError('Detector targets cannot lie inside material')
    volume=illuminated_beam_points(sample,beam,unit,d,reference_x_cm=float(reference[0]),volume_map=volume_map);ii=volume.input_indices;points=volume.points_cm;actual=targets[ii];inc=d[ii]
    incoming=np.column_stack([ray_length_cm(r,points,-inc) for r in regions]) if len(ii) else np.zeros((0,len(regions)))
    outgoing=np.column_stack([segment_length_cm(r,points,actual) for r in regions]) if len(ii) else incoming.copy()
    nominal=detector-reference;shift=actual-detector-(points-reference);distance=np.linalg.norm(actual-points,axis=1)
    out_offset=(np.sum(shift*shift,axis=1)+2*shift@nominal)/(distance+np.linalg.norm(nominal))/100
    inc_offset=offsets[ii]+(points[:,0]-reference[0])/(100*inc[:,0])
    return FinitePathTiming(incoming,outgoing,inc_offset,out_offset,float(volume.integration_weights.sum()),
        dict(volume.metadata,detector_position_cm=detector.tolist(),reference_point_cm=reference.tolist(),material_regions=len(regions),
            input_indices=ii.tolist(),incident_clock_label=incident_clock_label,incident_clock_advance='dx/n_x from x-plane',
            provided_reference_path_dispersion=bool(np.any(offsets!=0)),incident_guide_path_or_pulse_measured=False,
            paired_detector_surface_targets=True,fixed_pixel_calibration_reference=True,beam_divergence_included=True,
            detector_solid_angle_variation_included=False,multiple_scattering_included=False,experimental_accuracy_certified=False),
        points,actual,inc,volume.integration_weights)
