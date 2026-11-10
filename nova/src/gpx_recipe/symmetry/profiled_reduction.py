"""Stable row reduction of one complete weighted magnetic objective.

This is a numerical representation of the current observation, not prepared
physics or a saved likelihood. No profile column or singular direction is
discarded. The caller retains original arrays for final independent checks.
"""
import time

import numpy as np


class MagneticResidualReduction:
    def __init__(self, offset, nuclear, quadratic):
        self.dimension = quadratic.shape[1]
        self.first, self.second = np.triu_indices(self.dimension)
        self.multipliers = np.where(self.first == self.second, 1., 2.)
        matrix = np.column_stack((offset, nuclear,
            quadratic[:, self.first, self.second] * self.multipliers))
        basis, root = np.linalg.qr(matrix, mode='reduced')
        if not np.isfinite(root).all():
            raise ValueError('Nonfinite reduced magnetic residual')
        # Preserve a measured reconstruction/orthogonality error estimate.
        # In particular, never use expanded Gram polynomials to evaluate a
        # small residual as the difference of large squared intensities.
        errors = np.linalg.norm(matrix - basis @ root, axis=0)
        errors += 16 * np.finfo(float).eps * np.linalg.norm(matrix, axis=0)
        self.orthogonality = float(np.linalg.norm(basis.T @ basis - np.eye(root.shape[0])))
        self.column_errors = np.frombuffer(errors.tobytes(), dtype=float)
        self.root = np.frombuffer(root.tobytes(), dtype=float).reshape(root.shape)
        self.offset_norm = float(np.linalg.norm(self.root[:, 0]))

    @classmethod
    def build(cls, offset, nuclear, quadratic):
        began = time.monotonic()
        columns = 2 + quadratic.shape[1] * (quadratic.shape[1] + 1) // 2
        report = dict(method='full_pattern', bins=len(offset), columns=columns,
                      rows=len(offset), setup_seconds=0., reason='Small, inexpensive or wide objective')
        # QR setup is not worthwhile when the complete column space is
        # comparable to the number of observations. This is an internal
        # implementation choice; no scientific parameter or start is removed.
        # Native-array multistart measurements put the three-component
        # break-even between 701 and 1402 bins. Include the complete symmetric
        # contraction size so a more expensive basis can benefit sooner.
        if (len(offset) < max(128, 4 * columns)
                or len(offset) * (columns - 2) < 8192):
            return None, report
        try:
            reduced = cls(offset, nuclear, quadratic)
        except (np.linalg.LinAlgError, ValueError, MemoryError) as error:
            report['reason'] = 'Reduction unavailable: ' + type(error).__name__
            report['setup_seconds'] = time.monotonic() - began
            return None, report
        report.update(method='stable_qr', rows=reduced.root.shape[0],
                      setup_seconds=time.monotonic() - began,
                      reason='Complete weighted column space; no rank truncation')
        return reduced, report

    def evaluate(self, point, fixed_scale):
        """Return the same cost/gradient/scale, or request full evaluation.

        Floating-point error estimates control use of this representation.
        They are not a statistical uncertainty or a native-physics proof.
        All retained optimizer states are checked on the original arrays.
        """
        terms = np.r_[1., point[self.first] * point[self.second]]
        derivative = np.zeros((len(terms), self.dimension))
        rows = np.arange(1, len(terms))
        derivative[rows, self.first] += point[self.second]
        derivative[rows, self.second] += point[self.first]
        offset = self.root[:, 0]
        curve = self.root[:, 1:] @ terms
        curve_norm = float(np.linalg.norm(curve))
        denominator = curve_norm**2
        curve_error = float(self.column_errors[1:] @ np.abs(terms))
        offset_error = float(self.column_errors[0])
        delta = self.orthogonality
        scale_error = 0.
        if fixed_scale is None:
            numerator = -float(curve @ offset)
            denominator_error = (2 * curve_norm * curve_error + curve_error**2
                                 + delta * denominator)
            # Near an unidentifiable scale or its nonnegative boundary, use
            # the original objective instead of guessing a derivative branch.
            if numerator <= 0 or denominator <= max(denominator_error, np.finfo(float).tiny):
                return None
            scale = numerator / denominator
            numerator_error = (curve_norm * offset_error + self.offset_norm * curve_error
                               + curve_error * offset_error + delta * curve_norm * self.offset_norm)
            scale_error = (numerator_error + scale * denominator_error) / (denominator - denominator_error)
        else:
            scale = fixed_scale
        residual = offset + scale * curve
        jacobian = self.root[:, 1:] @ derivative
        gradient = scale * jacobian.T @ residual
        residual_norm = float(np.linalg.norm(residual))
        jacobian_norm = float(np.linalg.norm(jacobian))
        jacobian_error = float(np.linalg.norm(self.column_errors[1:] @ np.abs(derivative)))
        residual_error = offset_error + abs(scale) * curve_error + scale_error * (curve_norm + curve_error)
        gradient_error = (abs(scale) * (jacobian_norm * residual_error
            + jacobian_error * (residual_norm + residual_error)
            + delta * jacobian_norm * residual_norm)
            + scale_error * (jacobian_norm + jacobian_error) * (residual_norm + residual_error))
        if (not np.isfinite([residual_norm, residual_error, gradient_error, scale]).all()
                or residual_error > 1e-8 * max(1., residual_norm)
                or gradient_error > 1e-8 * max(1., float(np.linalg.norm(gradient)))):
            return None
        return .5 * float(residual @ residual), gradient, float(scale)
