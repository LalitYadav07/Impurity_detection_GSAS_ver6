"""All physical child displacements and symmetry-allowed cell coordinates.

Only the unobservable origin gauge is removed by the native preparation.
No irrep, atom, secondary mode, or low-symmetry family is frozen here.
"""
import numpy as np
from ..defaults import DEFAULT_CELL_RADIUS
from .bounded_chemical_prior import BoundedChemicalGeometry
from .chemical_geometry import direct_metric_from_reciprocal
from .physical_prior_geometry import native_reciprocal_matrix, displacement_gram


class FullModeSlice:
    def __init__(self, fast, mode_radius=.05, cell_radius=DEFAULT_CELL_RADIUS):
        self.fast = fast
        self.mode_dimensions = fast.nm
        self.cell_dimensions = fast.nc
        self.radii = np.r_[np.full(fast.nm, mode_radius), np.full(fast.nc, cell_radius)]
        if not np.isfinite(self.radii).all() or np.any(self.radii <= 0):
            raise ValueError('Positive finite mode and cell bounds required')
        g = fast.empirical_constraint.geometry
        metric = direct_metric_from_reciprocal(native_reciprocal_matrix(g.a0))
        gram = displacement_gram(g.mapping, metric, g.occupancy)
        if gram.shape != (fast.nm, fast.nm) or np.any(np.diag(gram) <= 0):
            raise ValueError('Physical native basis lost a displacement direction')
        self.lift = np.eye(fast.nm + fast.nc)
        self.lift[:fast.nm, :fast.nm] = np.diag(1 / np.sqrt(np.diag(gram)))
        self.native_mode_map = fast.basis['physical_basis'] @ self.lift[:fast.nm, :fast.nm]
        if fast.nm and np.linalg.matrix_rank(self.native_mode_map) != fast.nm:
            raise ValueError('All independent native displacements must remain represented')
        self.mode_name = [f'physical-{i}' for i in range(fast.nm)]
        self.mode_label = self.mode_name.copy()
        self.native_mode_labels = fast.meta['inventory']['native_mode_labels']
        self.frozen_modes = []
        self.cell_index = list(range(fast.nc))
        self.chemistry = BoundedChemicalGeometry(g.base, g.mapping, g.a0, g.cell_mapping,
            g.species, g.occupancy, self.lift, [1]*len(self.radii), self.radii.tolist(),
            fast.empirical_constraint.catalog)
        self.check(np.zeros(len(self.radii)))
        fast.native._reflection_cell_signature = None

    def check_domain(self, point):
        p=np.asarray(point,float)
        if p.shape!=self.radii.shape or not np.isfinite(p).all() or np.any(abs(p)>self.radii):
            raise ValueError("Point outside the reviewed structural domain")
        return p

    def check(self, point):
        p = self.check_domain(point)
        if not np.isfinite(self.chemistry.log_kernel(p[None, :])[0]):
            from .materials_project_chemistry import EmpiricalConfigurationRejected
            raise EmpiricalConfigurationRejected(dict(feasible=False,source='certified_bounded_contact_geometry',
                point=p.tolist(),catalog_fingerprint=self.chemistry.catalog_fingerprint))
        return p

    def theta(self, point):
        p = self.check(point)
        return np.r_[self.lift@p, 0., 0.]

    def native(self, point):
        p = self.check(point)
        theta = self.theta(p)
        return self.fast.evaluate(theta, chemistry_domain=(self.chemistry, p))

    def snapshot(self, point):
        profile = self.native(point).copy()
        n = self.fast.native
        h = n.histo_phases[0][n.parts[0][0]]
        phase_name = self.fast.slow.phase.name
        dictionary = h['Reflection Lists'][phase_name]
        phase = n.histo_phases[1][phase_name]
        if (phase['General'].get('Modulated') or phase['General'].get('Type') != 'nuclear'
                or phase['General'].get('doPawley')):
            raise ValueError('Commensurate nuclear Rietveld phase required')
        r = dictionary['RefList'].copy()
        r = r[np.lexsort((r[:,2],r[:,1],r[:,0]))]
        background = np.zeros_like(profile)
        background[self.fast.slice] = h['Data'][4][self.fast.slice]
        return profile, r, background, phase, h

    def native_mode_jacobian(self, point):
        p=self.check(point)
        y,j=self.fast.evaluate_mode_jacobian(self.theta(p),chemistry_domain=(self.chemistry,p))
        return y,self.lift[:self.fast.nm,:self.mode_dimensions].T@j
