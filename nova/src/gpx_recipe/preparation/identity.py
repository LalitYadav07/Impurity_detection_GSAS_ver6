"""Canonical data and reviewed native identity for reusable calculations."""
import hashlib
import inspect
from pathlib import Path
import sys
import numpy as np

# Saved-file locations/users and refinement limits are not forward physics.
# The physical bounds, implementation and parameter basis are pinned by the
# requesting model separately. Do not remove actual calibration data here.
BOOKKEEPING_CONTROLS = {'Seq Data', 'parmMinDict', 'parmMaxDict', 'parmFrozen',
                        'LastSavedAs', 'LastSavedBy'}

def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        keys = [str(k) for k in value]
        if len(set(keys)) != len(keys):
            raise ValueError('Ambiguous native identity dictionary')
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, complex):
        return {'complex': [value.real, value.imag]}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError('Unsupported native physics identity value: '+type(value).__name__)


def native_identity(native):
    """Pin reviewed native sources, active functions and loaded GSAS binaries."""
    from ..live.validated_backend import (active_identity, backend_identity,
                                          PINNED_PRODUCTION_STRMATH)
    source_hash = hashlib.sha256(Path(native.math.__file__).read_bytes()).hexdigest()
    if source_hash != PINNED_PRODUCTION_STRMATH and active_identity() is None:
        raise ValueError('Preparation requires the reviewed native scientific backend')
    identity = backend_identity()
    identity['active_backend'] = active_identity()
    identity['powder_source_sha256'] = hashlib.sha256(Path(native.math.G2pwd.__file__).read_bytes()).hexdigest()
    identity['active_functions'] = {name: hashlib.sha256(inspect.getsource(getattr(native.math, name)).encode()).hexdigest()
        for name in ('StructureFactor2', 'MagStructureFactor2', 'MagStructureFactorDerv', 'getPowderProfile')}
    native_root = Path(native.math.__file__).parent.resolve()
    binaries = {}
    for name, module in list(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        path = Path(filename)
        if path.suffix in {'.so', '.pyd'} and path.resolve().is_relative_to(native_root):
            binaries[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    identity['native_binaries'] = binaries
    return identity
