"""Reduced diffraction observations: supplied variances, never invented counts.

``mask`` means usable here. Import adapters name an invalid mask explicitly.
The normalization mask retains excluded-bin intensity for native Rwp reporting.
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

    def __post_init__(self):
        if not isinstance(self.acquisition,str) or not self.acquisition or not isinstance(self.protocol_sha256,str) or not self.protocol_sha256 or type(self.revision) is not int or self.revision < 0:
            raise ValueError('Exact acquisition, preparation and nonnegative revision required')
        if self.exposure is not None and (type(self.exposure) not in (int,float) or not np.isfinite(self.exposure) or self.exposure <= 0):
            raise ValueError('Exposure must be unknown or finite and positive')
        for name in ('x','y','variance','mask','normalization_mask'):
            value = np.array(getattr(self,name), dtype=bool if name.endswith('mask') else float, copy=True)
            value.flags.writeable = False
            object.__setattr__(self,name,value)
        n = len(self.x)
        if n < 2 or any(a.shape != (n,) for a in (self.x,self.y,self.variance,self.mask,self.normalization_mask)):
            raise ValueError('All observation arrays must share one grid')
        if not np.isfinite(self.x).all() or not np.all(np.diff(self.x)>0) or not self.mask.any():
            raise ValueError('Ordered finite grid and usable bins required')
        use = self.normalization_mask
        if np.any(self.mask & ~use) or not np.isfinite(self.y[use]).all() or not np.isfinite(self.variance[use]).all() or np.any(self.variance[use]<=0):
            raise ValueError('Usable and normalization bins need finite values and positive variances')

    @property
    def observation_sha256(self):
        from ..symmetry.contracts import digest
        return digest(dict(acquisition=self.acquisition,protocol=self.protocol_sha256,revision=self.revision,
                           exposure=self.exposure,arrays=array_digest(self.x,self.y,self.variance,self.mask,self.normalization_mask)))

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
    for a in arrays[:3]:
        valid &= ~np.ma.getmaskarray(a)
    domain = fit_mask(x, limits[:2])
    variance = np.divide(1.,w,out=np.full_like(w,np.inf),where=valid)
    return ReducedObservation(acquisition,protocol_sha256,revision,x,y,variance,
                              valid&fit_mask(x,limits),valid&domain,exposure)


def from_snapshot(x,y,sigma,invalid_mask,limits,**identity):
    invalid = np.asarray(invalid_mask)
    if invalid.dtype != bool or invalid.shape != np.shape(x):
        raise ValueError('Snapshot invalid_mask must be an explicit Boolean array on the grid')
    sigma = np.asarray(sigma,float)
    weights = np.divide(1.,sigma**2,out=np.zeros_like(sigma),where=np.isfinite(sigma)&(sigma>0)&~invalid)
    return from_arrays([x,y,weights],limits,**identity)
