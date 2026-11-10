"""Projection of fitted reference-centre offsets into diffraction calibration.

This models the weighted polynomial coefficient observable after peaks have
already been fitted and selected. It does not extract Gaussian centres, select
native calibration parameters, establish true pixel geometry or supply physical
source timing. Peak-height weights are not automatically error precisions.
"""
from dataclasses import dataclass
import numpy as np

VERSION='powgen-reference-centre-clock-projection/1'
_TERMS=('difc','tzero','difa')


def _design(d,terms):
    d=np.asarray(d,float)
    if d.ndim!=1 or not 1<=len(d)<=1000000 or not np.isfinite(d).all() or np.any(d<=0):
        raise ValueError('Finite positive one-dimensional reference spacings are required')
    with np.errstate(over='ignore'):
        columns={'difc':lambda:d,'tzero':lambda:np.ones_like(d),'difa':lambda:d*d}
        matrix=np.column_stack([columns[t]() for t in terms])
    if not np.isfinite(matrix).all():raise ValueError('Calibration basis exceeds its finite numerical domain')
    return matrix


@dataclass(frozen=True)
class CalibrationClockProjection:
    terms: tuple
    coefficients: np.ndarray
    reference_d_a: np.ndarray
    reference_residual_us: np.ndarray
    normalized_weights: np.ndarray
    metadata: dict

    def projected_offset_us(self,d_a):
        """Offset absorbed into the supplied calibration basis, without a mean-time claim."""
        result=_design(d_a,self.terms)@self.coefficients
        if not np.isfinite(result).all():raise ValueError('Projected clock exceeds its finite numerical domain')
        return result

    def residual_offset_us(self,d_a,unprojected_offset_us):
        """Remaining offset after the reference polynomial has been absorbed."""
        projected=self.projected_offset_us(d_a);offset=np.asarray(unprojected_offset_us,float)
        if offset.shape!=projected.shape or not np.isfinite(offset).all():
            raise ValueError('Matching finite unprojected offsets are required')
        with np.errstate(over='ignore'):
            result=offset-projected
        if not np.isfinite(result).all():raise ValueError('Residual clock exceeds its finite numerical domain')
        return result


def project_reference_centres(reference_d_a,centroid_offsets_us,weights,*,terms,provenance_label):
    """Fit an explicitly chosen DIFC/TZERO/DIFA offset basis to reference centres.

    For DIFC only, delta_DIFC=sum(w*d*offset)/sum(w*d²). The remaining event
    offset is delta_event(d)-d*delta_DIFC. Thus an offset proportional to d is
    absorbed completely. If lambda is proportional to d at a pixel, a common
    wavelength-linear delay is absorbed too. A constant electronics zero is
    generally absorbed partly by a DIFC-only fit, depending on reference peaks
    and their weights. Do not add either offset to calibrated coefficients
    again without modelling what their reference extraction has absorbed.

    The archived Mantid Gaussian/DIFC/default-UseChiSq=false pathway uses
    height² weights; UseChiSq=true uses inverse fitted-centre variance. Callers
    supply the actual selected weights and centroid offsets. This routine
    implements the selected linear projection, not the full Mantid algorithm.
    No posterior or interval coverage follows from the projection weights.
    """
    if (not isinstance(terms,tuple) or terms not in (('difc',),('difc','tzero'),_TERMS)
            or type(provenance_label) is not str or not provenance_label.strip() or len(provenance_label)>512):
        raise ValueError('An explicit supported calibration basis and bounded provenance label are required')
    X=_design(reference_d_a,terms);n,p=X.shape
    y=np.asarray(centroid_offsets_us,float);w=np.asarray(weights,float)
    if (n>4096 or y.shape!=(n,) or w.shape!=(n,) or not np.isfinite(y).all() or not np.isfinite(w).all()
            or np.any(w<0) or np.count_nonzero(w)>4096 or np.count_nonzero(w)<p+1):
        raise ValueError('Matching finite offsets and nonnegative weights need at least parameters+1 positive-weight references')
    w=w/float(w.max())
    original=np.asarray(weights,float)
    if np.any((original>0)&(w==0)):raise ValueError('Reference weight dynamic range underflows the projection domain')
    scales=np.max(abs(X),axis=0);B=np.sqrt(w)[:,None]*(X/scales)
    value,_,rank,singular=np.linalg.lstsq(B,np.sqrt(w)*y,rcond=None)
    if rank!=p:raise ValueError('The selected reference basis is numerically unidentified')
    coefficients=value/scales;residual=y-X@coefficients
    if not np.isfinite(coefficients).all() or not np.isfinite(residual).all():
        raise ValueError('Reference calibration projection exceeds its finite numerical domain')
    arrays=[np.array(v,copy=True) for v in (coefficients,np.asarray(reference_d_a,float),residual,w)]
    for a in arrays:a.setflags(write=False)
    return CalibrationClockProjection(terms,*arrays,dict(version=VERSION,provenance_label=provenance_label,
        selected_basis=list(terms),reference_peaks=n,positive_weight_reference_peaks=int(np.count_nonzero(w)),
        common_weight_scale_removed=True,scaled_design_condition_number=float(singular[0]/singular[-1]),
        coefficient_units={'difc':'us/A','tzero':'us','difa':'us/A^2'},
        weights_are_not_assumed_statistical_precisions=True,peak_extraction_or_native_parameter_selection_reproduced=False,
        geometry_source_clock_or_covariance_independently_identified=False,experimental_accuracy_certified=False))
