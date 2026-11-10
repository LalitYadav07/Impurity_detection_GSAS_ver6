"""Explicit histogram quadrature for forward density predictions.

Experimental bin edges should come from the producer workspace. Reconstruction
requires a declared coordinate convention. Regularly spaced log points alone
cannot distinguish arithmetic centers from geometric centers or bin boundaries.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss


LOG_CENTER_CONVENTION = 'arithmetic-centers-of-constant-log-bins'


def log_edges(points, *, convention, tolerance=1e-8):
    if convention != LOG_CENTER_CONVENTION:
        raise ValueError('Declare arithmetic centers of constant-log bins or supply producer edges')
    if not np.isfinite(tolerance) or not 0 < tolerance <= 1e-6:
        raise ValueError('A positive reconstruction tolerance at most 1e-6 is required')
    points=np.asarray(points,float)
    if points.ndim!=1 or len(points)<3 or not np.isfinite(points).all() or np.any(points<=0) or np.any(np.diff(points)<=0):
        raise ValueError('Positive finite increasing point coordinates required')
    ratio=float(np.exp(np.mean(np.diff(np.log(points)))))
    edges=2*points[0]/(1+ratio)*ratio**np.arange(len(points)+1)
    error=float(np.max(abs((edges[:-1]+edges[1:])/2/points-1)))
    if error>tolerance:
        raise ValueError('Point coordinates do not establish constant-log histogram edges; supply producer edges')
    return edges,{'convention':'Arithmetic centers of constant-log bins',
                  'maximum_relative_center_reconstruction_error':error,'ratio':ratio,
                  'producer_edges_verified':False,
                  'scope':'Compatibility with the declared convention; point spacing alone does not establish producer semantics'}


def quadrature(edges, order):
    edges=np.asarray(edges,float)
    if edges.ndim!=1 or len(edges)<2 or not np.isfinite(edges).all() or np.any(edges<=0) or np.any(np.diff(edges)<=0):
        raise ValueError('Positive finite increasing histogram edges required')
    if type(order) is not int or not 1<=order<=32:
        raise ValueError('Choose an integer quadrature order from 1 through 32')
    nodes,weights=leggauss(order)
    x=(edges[:-1,None]+edges[1:,None])/2+np.diff(edges)[:,None]/2*nodes
    return x,weights/2


def average(values, weights):
    values=np.asarray(values,float);weights=np.asarray(weights,float)
    if weights.ndim!=1 or not len(weights) or not np.isfinite(weights).all() or np.any(weights<=0) or not np.isclose(weights.sum(),1,rtol=0,atol=1e-12):
        raise ValueError('Positive normalized quadrature weights required')
    if not values.ndim or values.shape[-1]%len(weights) or not np.isfinite(values).all():
        raise ValueError('Finite density values on complete quadrature bins required')
    return np.tensordot(values.reshape(*values.shape[:-1],-1,len(weights)),weights,axes=(-1,0))
