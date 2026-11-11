"""ISODISTORT lattice-strain coordinates, separate from atomic displacements.

The native TOPAS export defines B_strained = B_parent S, where S is symmetric
in parent lattice coordinates. Its engineering shear entries are E4/2, E5/2,
E6/2. This is neither Cartesian Green-Lagrange strain nor a displacement in A.
Only the positive-stretch branch is recoverable here. No scattering model,
mode-presence decision, trained head or experimental confidence is implied.
"""
import numpy as np

from ..structural_live.optimization.physical_prior_geometry import metric_matrix
from ..symmetry.displacements import metric as cell_metric

SCHEMA='radar-isodistort-parent-lattice-strain/v1'
NORMALIZATION='isodistort-normalized-unitless-parent-lattice-strain/v1'


class UnresolvedStrainProjection(ValueError):
    """A valid metric cannot be labeled in the supplied ISO strain basis.

    This is distinct from malformed metrics, input maps and failed numerical
    reconstruction. Training may retain the native curve with unknown strain
    labels; it must never reinterpret this condition as a zero amplitude.
    """

    def __init__(self,residual):
        self.residual=float(residual)
        super().__init__('Child metric contains strain outside the declared complete mode basis')


def _metric(value):
    """Only average antisymmetry within the scale of matrix-product roundoff."""
    a=np.asarray(value,float)
    if (a.shape!=(3,3) or not np.isfinite(a).all()
            or np.max(abs(a-a.T))>64*np.finfo(float).eps*np.max(abs(a))):
        raise ValueError('Finite symmetric lattice metric required')
    return metric_matrix((a+a.T)/2)


def _root(metric):
    metric=_metric(metric);values,vectors=np.linalg.eigh(metric)
    if values[-1]/values[0]>1e10:
        raise ValueError('Parent lattice-strain metric is numerically unresolved')
    return (vectors*np.sqrt(values))@vectors.T


def _stretch(strain):
    e=np.asarray(strain,float)
    if e.shape!=(6,) or not np.isfinite(e).all():
        raise ValueError('Six finite parent lattice-strain components required')
    return metric_matrix([[1+e[0],e[5]/2,e[4]/2],
                          [e[5]/2,1+e[1],e[3]/2],
                          [e[4]/2,e[3]/2,1+e[2]]])


class StrainProjection:
    """Complete declared strain-mode basis and an explicit child direct basis.

    `matrix` is the CIF's six-row strain-mode matrix. Its columns are multiplied
    by the corresponding CIF normalization factors, exactly as in TOPAS output.
    Missing modes are never invented and linearly dependent columns are rejected.
    The caller must bind these data to the exact reviewed CIF/candidate.
    """
    def __init__(self,parent_metric,basis,matrix,normalizations):
        parent=metric_matrix(parent_metric);_root(parent)
        basis=np.asarray(basis,float);matrix=np.asarray(matrix,float);norm=np.asarray(normalizations,float)
        if (basis.shape!=(3,3) or not np.isfinite(basis).all() or np.linalg.cond(basis)>1e8
                or matrix.ndim!=2 or matrix.shape[0]!=6 or not 1<=matrix.shape[1]<=6
                or not np.isfinite(matrix).all() or norm.shape!=(matrix.shape[1],)
                or not np.isfinite(norm).all() or np.any(norm<=0)):
            raise ValueError('Complete finite parent/child basis and normalized strain columns required')
        normalized=matrix*norm;scales=np.linalg.norm(normalized,axis=0)
        if np.any(scales<=0) or not np.isfinite(scales).all():
            raise ValueError('Every strain mode needs a finite nonzero normalized column')
        unit=normalized/scales;left,singular,right=np.linalg.svd(unit,full_matrices=False)
        if singular[-1]<=128*np.finfo(float).eps*max(unit.shape)*singular[0]:
            raise ValueError('Strain-mode columns are not independently resolved')
        self.parent=np.frombuffer(parent.tobytes(),dtype=float).reshape(3,3)
        self.basis=np.frombuffer(basis.tobytes(),dtype=float).reshape(3,3)
        self.matrix=np.frombuffer(normalized.tobytes(),dtype=float).reshape(matrix.shape)
        self._inverse_basis=np.linalg.inv(basis)
        self._inverse=(right.T/singular)@left.T/scales[:,None]
        self.dimensions=matrix.shape[1]

    def forward(self,amplitudes,*,parent_metric=None):
        """Render the exact lattice metric implied by normalized ISO amplitudes."""
        m=np.asarray(amplitudes,float)
        if m.shape!=(self.dimensions,) or not np.isfinite(m).all():
            raise ValueError('One finite coefficient per declared strain mode required')
        parent=self.parent if parent_metric is None else metric_matrix(parent_metric)
        s=_stretch(self.matrix@m)
        return _metric(self.basis@s@parent@s@self.basis.T)

    def project(self,child_metric,*,parent_metric=None):
        """Recover signed ISO coefficients; reject deformation outside the basis.

        Supplying the current parent-compatible metric removes that explicit
        thermal reference only. No isotropic lattice change is silently removed.
        Measured intensity arrays and GSAS-II cell parameters are never edited.
        """
        child=_metric(child_metric)
        parent=self.parent if parent_metric is None else metric_matrix(parent_metric)
        half=_root(parent);inverse=np.linalg.inv(half)
        strained=self._inverse_basis@child@self._inverse_basis.T
        stretch=inverse@_root(half@strained@half)@inverse
        stretch=(stretch+stretch.T)/2
        e=np.array([stretch[0,0]-1,stretch[1,1]-1,stretch[2,2]-1,
                    2*stretch[1,2],2*stretch[0,2],2*stretch[0,1]])
        amplitudes=self._inverse@e;reconstructed=self.matrix@amplitudes
        error=float(np.linalg.norm(reconstructed-e))
        if error>1e-10*max(1.,float(np.linalg.norm(e))):
            raise UnresolvedStrainProjection(error)
        check=self.forward(amplitudes,parent_metric=parent)
        relative=float(np.linalg.norm(check-child)/np.linalg.norm(child))
        if not np.isfinite(relative) or relative>1e-10:
            raise ValueError('Strain-mode reconstruction did not preserve the complete child metric')
        return dict(schema=SCHEMA,normalization=NORMALIZATION,units='dimensionless',amplitudes=amplitudes,
            parent_lattice_strain=e,reconstruction_relative_error=relative,
            experimental_mode_presence_qualified=False,experimental_mode_absence_qualified=False)
