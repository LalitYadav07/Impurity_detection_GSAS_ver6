"""Reduced diffraction observations: supplied variances, never invented counts.

Measurement validity, likelihood bins and native Rwp normalization are distinct.
``mask`` selects likelihood bins; import adapters name an invalid mask explicitly.
"""
from dataclasses import dataclass
import numpy as np
from ..fit_domain import native_powder_fit_mask
from .observations import array_digest, NewProtocolRequired


def fit_mask(x, limits):
    return native_powder_fit_mask(x, limits[1], limits[2:])


@dataclass(frozen=True, eq=False)
class ReducedObservation:
    acquisition: str
    protocol_sha256: str
    revision: int
    x: np.ndarray
    y: np.ndarray
    variance: np.ndarray
    mask: np.ndarray
    normalization_mask: np.ndarray
    exposure: float | None = None
    measurement_mask: np.ndarray | None = None

    def __post_init__(self):
        if not isinstance(self.acquisition,str) or not self.acquisition or not isinstance(self.protocol_sha256,str) or not self.protocol_sha256 or type(self.revision) is not int or self.revision < 0:
            raise ValueError('Exact acquisition, preparation and nonnegative revision required')
        if self.exposure is not None and (type(self.exposure) not in (int,float) or not np.isfinite(self.exposure) or self.exposure <= 0):
            raise ValueError('Exposure must be unknown or finite and positive')
        if self.measurement_mask is None:
            object.__setattr__(self,'measurement_mask',np.isfinite(self.y)&np.isfinite(self.variance)&(np.asarray(self.variance)>0))
        for name in ('x','y','variance','mask','normalization_mask','measurement_mask'):
            value = np.array(getattr(self,name), dtype=bool if name.endswith('mask') else float, copy=True)
            value.flags.writeable = False
            object.__setattr__(self,name,value)
        n = len(self.x)
        if n < 2 or any(a.shape != (n,) for a in (self.x,self.y,self.variance,self.mask,self.normalization_mask,self.measurement_mask)):
            raise ValueError('All observation arrays must share one grid')
        if not np.isfinite(self.x).all() or not np.all(np.diff(self.x)>0) or not self.mask.any():
            raise ValueError('Ordered finite grid and usable bins required')
        use = self.measurement_mask
        if np.any(self.mask & ~self.normalization_mask) or np.any(self.normalization_mask & ~use) or not np.isfinite(self.y[use]).all() or not np.isfinite(self.variance[use]).all() or np.any(self.variance[use]<=0):
            raise ValueError('Usable and normalization bins need finite values and positive variances')

    @property
    def observation_sha256(self):
        from ..symmetry.contracts import digest
        return digest(dict(acquisition=self.acquisition,protocol=self.protocol_sha256,revision=self.revision,
                           exposure=self.exposure,arrays=array_digest(self.x,self.y,self.variance,self.mask,self.normalization_mask,self.measurement_mask)))

    def require_continuation(self, previous):
        if self.acquisition != previous.acquisition or self.protocol_sha256 != previous.protocol_sha256 or not np.array_equal(self.x,previous.x):
            raise NewProtocolRequired('Acquisition, preparation or grid changed')
        if self.revision < previous.revision:
            raise NewProtocolRequired('An older observation cannot replace a newer result')
        if self.revision == previous.revision:
            if self.observation_sha256 != previous.observation_sha256:
                raise NewProtocolRequired('Observation changed without a new revision')
            return 'duplicate'
        if (self.exposure is None) != (previous.exposure is None) or (self.exposure is not None and self.exposure < previous.exposure):
            raise NewProtocolRequired('Exposure semantics changed or cumulative exposure reset')
        return 'replacement'


def from_arrays(arrays, limits, *, acquisition, protocol_sha256, revision=0, exposure=None):
    x,y,w = (np.asarray(np.ma.getdata(a),float) for a in arrays[:3])
    valid = np.isfinite(x)&np.isfinite(y)&np.isfinite(w)&(w>0)
    for a in arrays[1:3]:
        valid &= ~np.ma.getmaskarray(a)
    domain = fit_mask(x, limits[:2])
    variance = np.divide(1.,w,out=np.full_like(w,np.inf),where=valid)
    return ReducedObservation(acquisition,protocol_sha256,revision,x,y,variance,
                              valid&fit_mask(x,limits)&~np.ma.getmaskarray(arrays[0]),valid&domain,exposure,valid)


def from_snapshot(x,y,sigma,invalid_mask,limits,**identity):
    invalid = np.asarray(invalid_mask)
    if invalid.dtype != bool or invalid.shape != np.shape(x):
        raise ValueError('Snapshot invalid_mask must be an explicit Boolean array on the grid')
    sigma = np.asarray(sigma,float)
    weights = np.divide(1.,sigma**2,out=np.zeros_like(sigma),where=np.isfinite(sigma)&(sigma>0)&~invalid)
    return from_arrays([x,y,weights],limits,**identity)


def native_limits(observation, limits):
    """Represent extra likelihood omissions as exact-bin native exclusions.

    Exclusions leave measured weights and native Rwp normalization intact.
    A changed normalization convention is refused rather than fabricated.
    """
    import copy
    domain=native_powder_fit_mask(observation.x,limits[1])
    normalization=observation.measurement_mask&domain
    if not np.array_equal(observation.normalization_mask,normalization):
        raise ValueError('Rwp normalization must include every valid measurement in the native fit interval')
    permitted=observation.measurement_mask&fit_mask(observation.x,limits)
    if np.any(observation.mask&~permitted):
        raise ValueError('Observation enables bins outside the reviewed native fit domain')
    result=copy.deepcopy(limits)
    omitted=np.flatnonzero(permitted&~observation.mask)
    for group in np.split(omitted,np.flatnonzero(np.diff(omitted)>1)+1):
        if len(group):result.append([float(observation.x[group[0]]),float(observation.x[group[-1]])])
    if not np.array_equal(observation.measurement_mask&fit_mask(observation.x,result),observation.mask):
        raise ValueError('The requested likelihood mask cannot be represented in native limits')
    return result


def verify_native_observation(observation, arrays, limits, calculated, expected_cost, *, weight_factor=1.):
    """Independently check saved measurements, uncertainties, domains and cost."""
    actual=from_arrays(arrays,limits,acquisition=observation.acquisition,
        protocol_sha256=observation.protocol_sha256,revision=observation.revision,exposure=observation.exposure)
    if (not np.array_equal(actual.x,observation.x) or
            not np.array_equal(actual.y,observation.y,equal_nan=True) or
            any(not np.array_equal(getattr(actual,k),getattr(observation,k))
                for k in ('measurement_mask','mask','normalization_mask'))):
        raise ValueError('Saved GPX observations or scientific domains differ from the search')
    use=observation.measurement_mask
    if not np.allclose(actual.variance[use],observation.variance[use],rtol=1e-12,atol=0):
        raise ValueError('Saved GPX measured uncertainties differ from the supplied observation')
    fit=observation.mask;prediction=np.asarray(calculated,float)
    if prediction.shape!=observation.x.shape or not np.isfinite(prediction[fit]).all():
        raise ValueError('Saved GPX has an invalid native prediction')
    cost=float(np.sum((observation.y[fit]-prediction[fit])**2/observation.variance[fit]))
    if not np.isfinite(expected_cost) or not np.isclose(cost,expected_cost,rtol=1e-8,atol=1e-8):
        raise ValueError('Saved GPX weighted objective differs from the fitted search objective')
    if not np.isfinite(weight_factor) or weight_factor<=0:
        raise ValueError('Native histogram weight factor must be finite and positive')
    norm=observation.normalization_mask
    denominator=float(np.sum(observation.y[norm]**2/observation.variance[norm]))
    if not np.isfinite(denominator) or denominator<=0:raise ValueError('Native Rwp normalization is undefined')
    return dict(measurements_verified=True,likelihood_bins_verified=True,normalization_bins_verified=True,
        weighted_objective_verified=True,search_weighted_cost=cost,native_weighted_cost=cost*weight_factor,
        native_histogram_weight_factor=weight_factor,rwp_native=100*float(np.sqrt(cost/denominator)),
        normalization_sum=denominator,measured_points=int(use.sum()),fitted_points=int(fit.sum()),
        normalization_points=int(norm.sum()))
