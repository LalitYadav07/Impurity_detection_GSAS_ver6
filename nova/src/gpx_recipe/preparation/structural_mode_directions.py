"""Named native displacement axes in the complete physical quotient space.

Random directions normalized by a Gram eigensystem need not isolate named
ISODISTORT modes. These additional directions retain that identity, modulo
only the native preparation's demonstrated origin gauges. They do not replace
mixed-mode samples, select a space group, or change any fitting coordinate.
"""
from dataclasses import dataclass

import numpy as np

from ..structural_live.optimization.physical_prior_geometry import orthonormal_lift


@dataclass(frozen=True)
class NamedModeDirections:
    labels: tuple
    physical: np.ndarray
    gauge: np.ndarray
    native_scale: np.ndarray
    gauge_only: np.ndarray
    maximum_reconstruction_error: float


def _owned(value):
    value=np.asarray(value)
    return np.frombuffer(value.tobytes(),dtype=value.dtype).reshape(value.shape)


def from_basis(native_map,gauge_basis,physical_gram,labels):
    """Resolve every named axis without mixing it with other native modes.

    ``native_map`` maps reviewed physical coordinates to native amplitudes.
    Together with the already-proven gauge basis it must span every native
    coordinate. Physical directions have unit occupied-site Cartesian RMS;
    a demonstrated pure gauge remains an explicit null control. Neither a
    scattering threshold nor a weak-mode truncation is used.
    """
    if not isinstance(labels,(list,tuple)):
        raise ValueError('Native mode labels must retain their indexed inventory')
    mapping=np.asarray(native_map,float);gauges=np.asarray(gauge_basis,float)
    gram=np.asarray(physical_gram,float);labels=tuple(labels)
    if (mapping.ndim!=2 or gauges.ndim!=2 or gram.ndim!=2
            or not np.isfinite(mapping).all() or not np.isfinite(gauges).all()
            or not np.isfinite(gram).all()):
        raise ValueError('Complete finite native, gauge and physical metric matrices required')
    native,physical=mapping.shape
    if (not 0<=native<=2048 or gauges.shape!=(native,native-physical)
            or gram.shape!=(physical,physical) or len(labels)!=native
            or any(not isinstance(label,str) or not 1<=len(label)<=1000 for label in labels)):
        raise ValueError('Every named native coordinate and only its demonstrated gauges must remain')
    # Use the existing full-rank physical-metric audit, not its rotated axes.
    orthonormal_lift(gram)
    if not native:
        return NamedModeDirections(labels,_owned(np.empty((0,0))),_owned(np.empty((0,0))),
            _owned(np.empty(0)),_owned(np.empty(0,bool)),0.)
    basis=np.concatenate((mapping,gauges),axis=1)
    scales=np.linalg.norm(basis,axis=0)
    if not np.isfinite(scales).all() or np.any(scales<=0):
        raise ValueError('Native physical/gauge basis contains an unresolved coordinate')
    normalized=basis/scales
    tolerance=128*np.finfo(float).eps*native
    singular=np.linalg.svd(normalized,compute_uv=False)
    if singular[-1]<=tolerance:
        raise ValueError('Native physical/gauge coordinates do not form a complete resolved basis')
    solved=np.linalg.solve(normalized,np.eye(native))/scales[:,None]
    error=float(np.max(abs(basis@solved-np.eye(native))))
    if not np.isfinite(error) or error>1e-10:
        raise ValueError('Named native axes could not be reconstructed without changing coordinates')
    raw=solved[:physical].T
    squared=np.einsum('ni,ij,nj->n',raw,gram,raw)
    gauge_only=np.all(raw==0.,axis=1)
    if (not np.isfinite(squared).all() or np.any(squared[~gauge_only]<=0)
            or np.any(squared[gauge_only]!=0.)):
        raise ValueError('An unresolved physical norm cannot become a missing or relabelled mode')
    # Do not turn tiny physical directions into gauges by applying a floor.
    factors=np.ones(native)
    factors[~gauge_only]=1/np.sqrt(squared[~gauge_only])
    directions=raw*factors[:,None];gauge=solved[physical:].T*factors[:,None]
    native_scale=factors
    if (not np.isfinite(directions).all() or not np.isfinite(gauge).all()
            or not np.isfinite(native_scale).all()):
        raise ValueError('Named-mode normalization exceeded finite native coordinates')
    reconstructed=mapping@directions.T+gauges@gauge.T
    relative=float(np.max(abs(reconstructed-np.diag(native_scale))
        /np.maximum(abs(native_scale),np.finfo(float).tiny)[None,:]))
    if relative>1e-10:
        raise ValueError('Normalized named-mode reconstruction lost its native identity')
    return NamedModeDirections(labels,_owned(directions),_owned(gauge),_owned(native_scale),
        _owned(gauge_only),max(error,relative))


def from_source(source):
    """Use the same complete native geometry and physical metric as fitting."""
    from .structural_sampling import physical_grams
    nm=source.mode_dimensions
    if (source.fast.nm!=nm or source.frozen_modes
            or source.native_mode_map.shape[1]!=nm):
        raise ValueError('Named coverage requires every physical native displacement coordinate')
    gram,_=physical_grams(source)
    return from_basis(source.native_mode_map,source.fast.basis['gauge_basis'],gram,
        source.fast.meta['inventory']['native_mode_labels'])
