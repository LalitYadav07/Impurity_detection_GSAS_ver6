"""Apply captured POWGEN reference processing to simulated count histograms.

Reference observations and their processed estimates are fixed, shared inputs.
These operators never convert weighted vanadium observations into Poisson data.
"""
import numpy as np
from scipy import sparse


def overlap_map(reference_edges, target_edges):
    """Finite-bin conservation for a piecewise constant reference rate density."""
    ref, target = np.asarray(reference_edges, float), np.asarray(target_edges, float)
    if any(e.ndim != 1 or len(e) < 2 or not np.isfinite(e).all() or np.any(np.diff(e) <= 0) for e in (ref, target)):
        raise ValueError('Reference and target edges must increase')
    if target[0] < ref[0] or target[-1] > ref[-1]:
        raise ValueError(f'Target bins must lie inside measured reference edges {ref[0]:g}–{ref[-1]:g}')
    edges = np.unique(np.r_[target, ref[(ref > target[0]) & (ref < target[-1])]])
    mid = (edges[1:] + edges[:-1])/2
    column = np.searchsorted(ref, mid, side='right') - 1
    row = np.searchsorted(target, mid, side='right') - 1
    return sparse.csr_matrix((np.diff(edges)/np.diff(ref)[column], (row, column)),
                             shape=(len(target)-1, len(ref)-1))


def project_reduction_reference(native_tof_edges, processed_vanadium, smoothed_empty,
                                target_tof_edges, *, scale=100.):
    """Rebin observed rates, then subtract/divide; do not rebin the ratio.

On the native grid this is exactly ScaleData * (sample/Q - empty) / V.
Other grids use explicit piecewise constant TOF densities for both references.
The input V already contains the native absorption/multiple-scattering,
event compression, peak stripping and smoothing operations.
"""
    van, empty = np.asarray(processed_vanadium, float), np.asarray(smoothed_empty, float)
    if (van.ndim != 1 or empty.shape != van.shape or len(van)+1 != len(native_tof_edges)
            or not np.isfinite(van+empty).all() or np.any(van <= 0)
            or isinstance(scale, (bool, np.bool_)) or not np.isfinite(scale) or scale <= 0):
        raise ValueError('Finite reference rates, positive vanadium and scale required')
    projection = overlap_map(native_tof_edges, target_tof_edges)
    return np.asarray(projection @ van)/scale, np.asarray(projection @ empty)
