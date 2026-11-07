"""Training-only geometric envelope for advisory structural ranking."""
import math
import numpy as np

SCHEMA='radar-structural-ranking-training-domain/v1'


def validate(value):
    if (not isinstance(value,dict) or set(value)!={'schema','maximum_relative_reciprocal_metric_norm','source'}
            or value['schema']!=SCHEMA or value['source']!='native_training_examples_only'
            or type(value['maximum_relative_reciprocal_metric_norm']) not in (int,float)
            or not math.isfinite(value['maximum_relative_reciprocal_metric_norm'])
            or not 0<=value['maximum_relative_reciprocal_metric_norm']<=10):
        raise ValueError('Structural ranking needs its finite training-only metric envelope')
    return value


def envelope(maximum):
    return validate(dict(schema=SCHEMA,maximum_relative_reciprocal_metric_norm=float(maximum),
                         source='native_training_examples_only'))


def relative_metric_norm(reference,current):
    """Same basis-invariant reciprocal-metric norm used by the sampler.

    This is sqrt(tr[(G0^-1 dG)^2]), not an isotropic lattice shift or an exact
    finite real-space strain. Both cells must use the same checked setting.
    """
    from ..symmetry.displacements import metric
    g0=np.linalg.inv(metric(reference));delta=np.linalg.inv(metric(current))-g0
    value=np.linalg.solve(g0,delta)
    squared=float(np.trace(value@value))
    if not math.isfinite(squared) or squared < -1e-12:
        raise ValueError('Invalid current reciprocal-metric displacement')
    return math.sqrt(max(0.,squared))
