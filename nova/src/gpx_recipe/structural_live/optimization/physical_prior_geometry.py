"""Physical prior coordinates independent of native mode/cell normalization.

These are coordinate/metric transformations, not scattering calculations.
Isotropic Gaussian or radial priors in the resulting coordinates are invariant
to orthogonal basis choices. Independent box priors are not invariant and must
not be described that way. The current legacy sampler still uses its original
box priors; this module does not silently change existing inference.
"""
import numpy as np


def metric_matrix(value):
    value=np.array(value,float,copy=True)
    if value.shape!=(3,3) or not np.isfinite(value).all() or not np.allclose(value,value.T,rtol=1e-12,atol=0):
        raise ValueError('Finite symmetric crystallographic metric required')
    if np.linalg.eigvalsh(value).min()<=0:raise ValueError('Positive definite crystallographic metric required')
    return value


def displacement_gram(fractional_mapping,direct_metric,site_weights):
    """Mean squared Cartesian displacement over occupied physical sites.

    Weights are verified child-orbit multiplicity times occupancy. Symmetry-
    related atoms have identical displacement lengths when their rotations
    preserve the child metric. Site/neighbor identities must stay unchanged.
    A zero-weight site cannot silently remove a purported physical coordinate.
    """
    b=np.asarray(fractional_mapping,float);g=metric_matrix(direct_metric);w=np.asarray(site_weights,float)
    if (b.ndim!=3 or b.shape[1]!=3 or not len(b) or not np.isfinite(b).all()
            or w.shape!=(len(b),) or not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0):
        raise ValueError('Finite atom-by-axis-by-mode mapping and physical site weights required')
    return np.einsum('a,aim,ij,ajn->mn',w/w.sum(),b,g,b)


def reciprocal_metric_gram(reciprocal_metric,metric_directions):
    """Dimensionless relative reciprocal-metric tensor norm.

    G(q)=G0+sum(q_j D_j). The squared norm is tr[(G0^-1 dG)^2],
    invariant under a simultaneous change of fractional cell basis. It is
    approximately four times the infinitesimal real-strain Frobenius norm;
    it must not be mislabeled as the exact finite real-space strain.
    """
    g=metric_matrix(reciprocal_metric);d=np.asarray(metric_directions,float)
    if d.ndim!=3 or d.shape[1:]!=(3,3) or not np.isfinite(d).all() or not np.allclose(d,d.transpose(0,2,1),rtol=1e-12,atol=0):
        raise ValueError('Symmetric native reciprocal-metric directions required')
    transformed=np.array([np.linalg.solve(g,v) for v in d])
    return np.einsum('mij,nji->mn',transformed,transformed) if len(d) else np.empty((0,0))


def orthonormal_lift(gram):
    """q = lift @ u gives a physical squared norm u.T @ u.

    A rank failure is an error, never permission to remove a weakly scattering
    direction. Correlation scaling avoids dropping dimensions with small units.
    """
    gram=np.array(gram,float,copy=True)
    if gram.ndim!=2 or gram.shape[0]!=gram.shape[1] or not np.isfinite(gram).all():
        raise ValueError('Finite square physical-coordinate Gram matrix required')
    if not len(gram):return gram
    if np.any(np.diag(gram)<=0):raise ValueError('Physical coordinate has zero or negative norm')
    sd=np.sqrt(np.diag(gram));corr=gram/sd[:,None]/sd[None,:]
    tolerance=128*np.finfo(float).eps*len(gram)
    if not np.allclose(corr,corr.T,rtol=0,atol=tolerance):raise ValueError('Asymmetric physical-coordinate Gram matrix')
    eigen,vectors=np.linalg.eigh((corr+corr.T)/2)
    if eigen.min()<=tolerance:raise ValueError('Unresolved physical-coordinate rank deficiency')
    lift=(vectors/np.sqrt(eigen))/sd[:,None]
    if not np.allclose(lift.T@gram@lift,np.eye(len(gram)),rtol=1e-8,atol=1e-8):
        raise ValueError('Physical-coordinate normalization is numerically unstable')
    return lift


def native_reciprocal_matrix(a):
    a=np.asarray(a,float)
    if a.shape!=(6,) or not np.isfinite(a).all():raise ValueError('Six finite native reciprocal coefficients required')
    return np.array([[a[0],a[3]/2,a[4]/2],[a[3]/2,a[1],a[5]/2],[a[4]/2,a[5]/2,a[2]]])
