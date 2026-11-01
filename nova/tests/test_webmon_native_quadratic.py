import unittest
import numpy as np
from radar_pd_nova.webmon.native_quadratic import compile_native_magnetic_jacobian


class QuadraticObjective:
    def __init__(self,bad=False):
        self.numerical_indices=[0];self.bad=bad
        self.q=np.array([[[2.,3.],[.3,.4]],[[.3,.4],[1.,2.]]])
    def profile(self,x,jacobian=False):
        p=np.asarray(x)[1:];mean=np.array([100.,200.])+np.einsum('i,ijb,j->b',p,self.q,p)
        if not jacobian:return mean
        jac=np.vstack([np.zeros(2),2*np.einsum('ijb,j->ib',self.q,p)])
        if self.bad:jac[1]*=2
        return mean,jac


class NativeQuadraticTests(unittest.TestCase):
    def test_compilation_exact_and_policy_restored(self):
        objective=QuadraticObjective();r=compile_native_magnetic_jacobian(objective,[1.,2.,3.],[1,2])
        np.testing.assert_allclose(r['quadratic'],objective.q,rtol=0,atol=1e-14)
        self.assertEqual(objective.numerical_indices,[0])
        self.assertEqual(r['diagnostics']['derivative_calls'],4)
    def test_bad_derivatives_fail_and_restore_policy(self):
        objective=QuadraticObjective(bad=True)
        with self.assertRaises(ValueError):compile_native_magnetic_jacobian(objective,[1.,2.,3.],[1,2])
        self.assertEqual(objective.numerical_indices,[0])


if __name__=='__main__':unittest.main()
