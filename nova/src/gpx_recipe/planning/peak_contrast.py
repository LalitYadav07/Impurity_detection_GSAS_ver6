"""Fixed peak/sideband count contrasts for experimental calibration diagnostics.

These operators remove an affine local count density, not arbitrary background
or overlapping Bragg scattering. Their count covariance is conditional on a
fixed independent-Poisson measure; source/process/model uncertainty is separate.
No measured curve is fitted, clipped or promoted to experimental qualification.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class SymmetricSideband:
    center: float
    core_halfwidth: float
    side_inner: float
    side_outer: float

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.center,self.core_halfwidth,self.side_inner,self.side_outer)):
            raise ValueError("Finite peak and sideband coordinates required")
        if not 0 < self.core_halfwidth < self.side_inner < self.side_outer < self.center:
            raise ValueError("Positive, disjoint core/sidebands on a positive coordinate required")

    @property
    def side_weight(self):
        return -self.core_halfwidth/(self.side_outer-self.side_inner)

    @property
    def intervals(self):
        c,h,i,o = self.center,self.core_halfwidth,self.side_inner,self.side_outer
        return ((c-o,c-i,self.side_weight),(c-h,c+h,1.),(c+i,c+o,self.side_weight))

    def weights(self, coordinates):
        x = np.asarray(coordinates,float)
        if not np.isfinite(x).all():
            raise ValueError("Finite observation coordinates required")
        output = np.zeros_like(x)
        for left,right,weight in self.intervals:
            output[(x>=left)&(x<right)] = weight
        return output

    def centered_moment(self, order):
        """Integral of w(x)*(x-center)**order; useful for background bias bounds."""
        if type(order) is not int or not 0 <= order <= 4:
            raise ValueError("A bounded nonnegative moment order is required")
        h,i,o = self.core_halfwidth,self.side_inner,self.side_outer
        if order % 2:
            return 0.
        return 2*(h**(order+1)+self.side_weight*(o**(order+1)-i**(order+1)))/(order+1)

    def linear_tof_support(self, difc, *, tof_bounds, period_us, prompt_width_us):
        """Pixels with the complete fixed support under TOF=DIFC*d (DIFA=T0=0)."""
        coefficient = np.asarray(difc,float)
        lower,upper = map(float,tof_bounds)
        if (coefficient.ndim != 1 or not np.isfinite(coefficient).all() or np.any(coefficient<=0) or
            not all(math.isfinite(x) for x in (lower,upper,period_us,prompt_width_us)) or
            not 0 <= lower < upper or not 0 <= prompt_width_us < period_us or
            (upper-lower)/period_us > 1000):
            raise ValueError("Finite positive linear calibration and bounded TOF/prompt support required")
        a = (self.center-self.side_outer)*coefficient
        b = (self.center+self.side_outer)*coefficient
        eligible = (a>=lower)&(b<=upper)
        for frame in range(math.floor(lower/period_us),math.ceil(upper/period_us)):
            start = frame*period_us
            if prompt_width_us:
                eligible &= ~((a<start+prompt_width_us)&(b>start))
        return eligible


def nested_count_contrasts(kernels, core_counts, side_counts):
    """Signed contrasts and shared conditional covariance for nested fixed cores.

    Every kernel must have identical wings and the same eligible observations.
    Side counts sum the two disjoint wings. A nested-core Poisson covariance is
    the smaller core count; shared negative wings add their weight products.
    """
    kernels = tuple(kernels)
    counts = np.asarray(core_counts)
    if not kernels or counts.shape != (len(kernels),) or counts.dtype.kind not in "iu":
        raise ValueError("One nonnegative integer count for each nested core required")
    if type(side_counts) not in (int,np.int32,np.int64) or side_counts < 0 or np.any(counts<0):
        raise ValueError("Nonnegative integer side/core counts required")
    first = kernels[0]
    if any((x.center,x.side_inner,x.side_outer)!=(first.center,first.side_inner,first.side_outer) for x in kernels):
        raise ValueError("Nested cores require exactly shared wings and observation support")
    if any(a.core_halfwidth>=b.core_halfwidth for a,b in zip(kernels,kernels[1:])) or np.any(np.diff(counts.astype(float))<0):
        raise ValueError("Core widths and corresponding nested counts must be ordered")
    weights = np.array([x.side_weight for x in kernels])
    contrast = counts.astype(float)+weights*side_counts
    covariance = np.minimum.outer(counts,counts).astype(float)+np.outer(weights,weights)*side_counts
    return contrast,covariance
