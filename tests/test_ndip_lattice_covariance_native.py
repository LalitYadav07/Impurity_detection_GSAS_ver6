"""Native GSAS-II covariance regression, also run inside the release image."""
import unittest
from types import SimpleNamespace

import numpy as np
try:
    from GSASII import GSASIIlattice as lattice
except ImportError:
    lattice = None

from scripts.ndip_lattice import cell_fit_information


CASES = [
    ('triclinic P','-1','',[4,5,6,80,95,110],[0,1,2,3,4,5]),
    *[(f'monoclinic {c}','2/m','b',[4,5,6,90,105,90],[0,1,2,4]) for c in ('P','C')],
    *[(f'orthorhombic {c}','mmm','',[4,5,6,90,90,90],[0,1,2]) for c in ('P','C','I','F')],
    *[(f'tetragonal {c}','4/mmm','',[4,4,6,90,90,90],[0,2]) for c in ('P','I')],
    ('trigonal R, hexagonal axes','3m1','',[4.76,4.76,13,90,90,120],[0,2]),
    ('hexagonal P','6/mmm','',[4,4,6,90,90,120],[0,2]),
    *[(f'cubic {c}','m3m','',[4.2,4.2,4.2,90,90,90],[0]) for c in ('P','I','F')],
    ('monoclinic unique a','2/m','a',[4,5,6,105,90,90],[0,1,2,5]),
    ('monoclinic unique c','2/m','c',[4,5,6,90,90,105],[0,1,2,3]),
    ('rhombohedral axes','3mR','',[5,5,5,75,75,75],[0,3]),
]


def analytic_jacobian(A, basis):
    """Independent inverse-metric calculus; no GSAS-II cell-error API."""
    G = np.array([[A[0],A[3]/2,A[4]/2],[A[3]/2,A[1],A[5]/2],[A[4]/2,A[5]/2,A[2]]])
    g = np.linalg.inv(G)
    lengths = np.sqrt(np.diag(g))
    V = np.sqrt(np.linalg.det(g))
    J = np.zeros((7, basis.shape[1]))
    for j in range(basis.shape[1]):
        d = basis[:,j]
        dG = np.array([[d[0],d[3]/2,d[4]/2],[d[3]/2,d[1],d[5]/2],[d[4]/2,d[5]/2,d[2]]])
        dg = -g @ dG @ g
        dl = np.diag(dg)/(2*lengths)
        J[:3,j] = dl
        for k,(u,v) in enumerate(((1,2),(0,2),(0,1))):
            cosine = g[u,v]/(lengths[u]*lengths[v])
            dc = dg[u,v]/(lengths[u]*lengths[v])-cosine*(dl[u]/lengths[u]+dl[v]/lengths[v])
            J[3+k,j] = -180/np.pi*dc/np.sqrt(1-cosine*cosine)
        J[6,j] = -V*np.trace(g@dG)/2
    J[np.abs(J)<1e-7] = 0
    return J


@unittest.skipIf(lattice is None, 'Requires the native GSAS-II scientific runtime')
class CellCovarianceTests(unittest.TestCase):
    def test_symmetry_propagation_against_independent_metric_derivatives(self):
        rng = np.random.default_rng(20261027)
        for identifier,(label,laue,unique,cell,independent) in enumerate(CASES):
            with self.subTest(symmetry=label):
                A = np.asarray(lattice.cell2A(cell))
                basis = np.eye(6)[:,independent]
                if laue == 'm3m': basis[:3,0] = 1
                elif laue == '4/mmm': basis[1,0] = 1
                elif laue in {'3m1','6/mmm'}: basis[1,0] = basis[3,0] = 1
                elif laue == '3mR': basis[:3,0] = 1;basis[3:,1] = 1
                phase = SimpleNamespace(id=identifier, data={'General':{
                    'Cell':[True,*cell,lattice.calc_V(A)],'SGData':{'SGLaue':laue,'SGUniq':unique}}})
                names = [f'{identifier}::A{k}' for k in independent]
                noise = rng.normal(size=(len(names),len(names)))
                C = (noise@noise.T+np.eye(len(names)))*1e-13
                covariance = {'freshCOV':True,'varyList':names,'variables':A[independent].tolist(),'covMatrix':C}
                result = cell_fit_information(phase,covariance)
                J = analytic_jacobian(A,basis)
                np.testing.assert_allclose(result['cell_covariance'],J@C@J.T,rtol=2e-5,atol=1e-13)
                for invalid in (-C, np.full_like(C, np.nan)):
                    self.assertIsNone(cell_fit_information(phase,{**covariance,'covMatrix':invalid})['cell_covariance'])
                partial = {**covariance,'varyList':names[:-1],'variables':A[independent[:-1]].tolist(),'covMatrix':C[:-1,:-1]}
                self.assertIsNone(cell_fit_information(phase,partial)['cell_covariance'])
                self.assertIsNone(cell_fit_information(phase,{**covariance,'freshCOV':False})['cell_covariance'])


if __name__ == '__main__':
    unittest.main()
