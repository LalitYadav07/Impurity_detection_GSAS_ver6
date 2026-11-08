"""Float64 mode-amplitude transforms without overflow or false zero labels."""
import numpy as np


def _inputs(values,scale):
    values,scale=np.broadcast_arrays(np.asarray(values,dtype=np.float64),np.asarray(scale,dtype=np.float64))
    if (not np.isfinite(values).all() or np.any(values<0)
            or not np.isfinite(scale).all() or np.any(scale<=0)):
        raise ValueError('Finite nonnegative amplitudes and positive scales required')
    return values,scale


def transform(values,scale):
    """asinh(amplitude/scale), retaining sub-float32 native amplitudes."""
    values,scale=_inputs(values,scale)
    result=np.zeros(values.shape,np.float64);nonzero=values>0
    logs=np.log(values[nonzero])-np.log(scale[nonzero]);large=logs>300.
    actual=np.empty(len(logs),np.float64)
    actual[large]=logs[large]+np.log(2.)
    actual[~large]=np.arcsinh(values[nonzero][~large]/scale[nonzero][~large])
    if np.any(actual==0):raise ValueError('Mode normalization would erase a nonzero native target')
    result[nonzero]=actual
    return result


def physical(values,scale):
    """scale*sinh(value), allowing a small scale to cancel a large exponent."""
    values,scale=_inputs(values,scale)
    result=np.empty(values.shape,np.float64);small=values<1.
    result[small]=scale[small]*np.sinh(values[small])
    large=~small
    with np.errstate(over='ignore',under='ignore'):
        result[large]=np.exp(np.log(scale[large])+values[large]-np.log(2.))*(-np.expm1(-2*values[large]))
    if not np.isfinite(result).all() or np.any((values>0)&(result==0)):
        raise ValueError('Predicted physical mode amplitude exceeds finite float64 range')
    return result
