"""Training-only geometric envelope for advisory structural ranking."""
import math
import numpy as np

SCHEMA='radar-structural-ranking-training-domain/v1'
PARENT_SCHEMA='radar-structural-ranking-parent-reference-domain/v1'


def validate(value):
    if (not isinstance(value,dict) or set(value)!={'schema','maximum_relative_reciprocal_metric_norm','source'}
            or (value['schema'],value['source']) not in {
                (SCHEMA,'native_training_examples_only'),
                (PARENT_SCHEMA,'native_training_parent_references_only')}
            or type(value['maximum_relative_reciprocal_metric_norm']) not in (int,float)
            or not math.isfinite(value['maximum_relative_reciprocal_metric_norm'])
            or not 0<=value['maximum_relative_reciprocal_metric_norm']<=10):
        raise ValueError('Structural ranking needs its finite training-only metric envelope')
    return value


def envelope(maximum,*,parent_references=False):
    if type(parent_references) is not bool:raise ValueError('Explicit training reference scope required')
    return validate(dict(schema=PARENT_SCHEMA if parent_references else SCHEMA,
        maximum_relative_reciprocal_metric_norm=float(maximum),
        source='native_training_parent_references_only' if parent_references else 'native_training_examples_only'))


def parent_reference_norms(coefficients):
    """Actual parent metrics, excluding the child's symmetry-breaking strain."""
    a=np.asarray(coefficients)
    if (a.ndim!=2 or a.shape[1]!=6 or not 1<=len(a)<=32768
            or a.dtype.kind not in 'fi' or not np.isfinite(a).all()):
        raise ValueError('Complete finite parent-reference reciprocal metrics required')
    g=np.zeros((len(a),3,3));g[:,range(3),range(3)]=a[:,:3]
    for index,(i,j) in enumerate(((0,1),(0,2),(1,2))):g[:,i,j]=g[:,j,i]=a[:,index+3]/2
    if np.any(np.linalg.eigvalsh(g)<=0):raise ValueError('Parent-reference metrics must be positive definite')
    delta=np.linalg.solve(g[0],g-g[0]);square=np.einsum('nij,nji->n',delta,delta)
    if not np.isfinite(square).all() or np.any(square< -1e-12):
        raise ValueError('Invalid sampled parent-reference displacement')
    return np.sqrt(np.maximum(square,0.))


def maximum_parent_metric(metadata):
    domain=metadata.get('training_domain')
    if domain is None:return 0.
    validate(domain)
    from .structural_attention_weights import SCHEMA as THREE_CURVE_SCHEMA
    if metadata.get('schema')==THREE_CURVE_SCHEMA and domain['schema']!=PARENT_SCHEMA:
        # Older paired models mixed parent shifts and child strain in this
        # scalar. Keep their exact-parent use, without inventing parent-cell
        # coverage from symmetry-breaking distortions.
        return 0.
    return domain['maximum_relative_reciprocal_metric_norm']


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
