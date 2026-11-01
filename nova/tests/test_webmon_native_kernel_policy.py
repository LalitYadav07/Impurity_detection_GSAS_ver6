import unittest
import numpy as np
from radar_pd_nova.webmon.native_kernel_policy import compile_native_magnetic_kernel


class Objective:
    def __init__(self, dimension=6, bad_derivative=False, nonquadratic=False):
        self.numerical_indices=[0]
        self.bad_derivative=bad_derivative
        self.nonquadratic=nonquadratic
        rng=np.random.default_rng(142)
        basis=rng.normal(size=(dimension, 4))
        self.q=np.einsum('ib,jb->ijb', basis, basis)

    def profile(self, x, jacobian=False):
        self.last=np.asarray(x).copy()
        p=self.last[1:]
        mean=np.full(4, 100.)+np.einsum('i,ijb,j->b', p, self.q, p)
        if self.nonquadratic:mean+=p[0]**4
        if not jacobian:return mean
        jac=np.vstack([np.zeros(4), 2*np.einsum('ijb,j->ib', self.q, p)])
        if self.bad_derivative:jac[1]*=2
        return mean, jac


class KernelPolicyTests(unittest.TestCase):
    def test_large_space_uses_derivatives_and_restores_state(self):
        obj=Objective();x=np.arange(7, dtype=float)
        result=compile_native_magnetic_kernel(obj, x, list(range(1,7)))
        self.assertEqual(result['diagnostics']['selected_method'], 'derivative')
        np.testing.assert_allclose(result['quadratic'], obj.q, atol=1e-12)
        np.testing.assert_array_equal(obj.last, x)
        self.assertEqual(obj.numerical_indices, [0])

    def test_small_space_uses_central_profiles(self):
        obj=Objective(3)
        result=compile_native_magnetic_kernel(obj, np.ones(4), [1,2,3])
        self.assertEqual(result['diagnostics']['selected_method'], 'central')

    def test_bad_derivatives_fall_back_to_independent_forward_model(self):
        obj=Objective(bad_derivative=True)
        result=compile_native_magnetic_kernel(obj, np.ones(7), list(range(1,7)))
        self.assertEqual(result['diagnostics']['selected_method'], 'central')
        self.assertIsNotNone(result['diagnostics']['fallback_reason'])
        np.testing.assert_allclose(result['quadratic'], obj.q, atol=1e-12)
        self.assertEqual(obj.numerical_indices, [0])

    def test_nonquadratic_forward_model_is_rejected(self):
        obj=Objective(nonquadratic=True);x=np.ones(7)
        with self.assertRaises(ValueError):
            compile_native_magnetic_kernel(obj, x, list(range(1,7)))
        np.testing.assert_array_equal(obj.last, x)


if __name__=='__main__':unittest.main()
