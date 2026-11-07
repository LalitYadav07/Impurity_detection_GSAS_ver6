"""Canonical data and reviewed native identity for reusable calculations."""
import hashlib
import inspect
from pathlib import Path
import platform
import sys
import numpy as np
import scipy

# Saved-file locations/users and refinement limits are not forward physics.
# The physical bounds, implementation and parameter basis are pinned by the
# requesting model separately. Do not remove actual calibration data here.
BOOKKEEPING_CONTROLS = {'Seq Data', 'parmMinDict', 'parmMaxDict', 'parmFrozen',
                        'LastSavedAs', 'LastSavedBy'}
# These are recorded sample conditions in the reviewed fixed PNC/PNT forward
# calculation. They do not parameterize a constitutive temperature/field law.
# Actual cells, coordinates, ADPs, moments and instrument parameters remain
# dependencies even when they were obtained by fitting those conditions.
ACQUISITION_PARAMETERS = frozenset({'Temperature','Pressure','FreePrm1','FreePrm2','FreePrm3'})


def without_recorded_conditions(values, *, prefix=''):
    """Copy fixed-forward dependencies without changing acquisition metadata.

    Only the five exact bank/sample keys are omitted. This helper belongs to
    the reviewed PNC/PNT preparation identities, never to experiment lineage
    or an arbitrary user recipe. Unknown and phase-local parameters remain.
    """
    omitted = {prefix+name for name in ACQUISITION_PARAMETERS}
    return {key:value for key,value in values.items() if key not in omitted}

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


def numerical_identity():
    """Portable qualification also depends on the numerical runtime build.

    Record reported BLAS/LAPACK build metadata without installation paths or
    process-local thread counts. This is a compatibility boundary, not a claim
    of bitwise reproducibility: consumers still run independent native checks.
    """
    builds = {}
    for name, module in (('numpy', np), ('scipy', scipy)):
        config = getattr(module.__config__, 'CONFIG', {})
        dependencies = config.get('Build Dependencies', {}) if isinstance(config, dict) else {}
        builds[name] = {kind: {key: value for key, value in dependencies.get(kind, {}).items()
            if key in {'name', 'version', 'openblas configuration', 'configuration'}}
            for kind in ('blas', 'lapack')}
    return dict(schema='radar-numerical-runtime/v1', numpy=np.__version__, scipy=scipy.__version__,
        python=dict(implementation=sys.implementation.name, version=list(sys.version_info[:3]),
                    cache_tag=sys.implementation.cache_tag),
        system=platform.system(), machine=platform.machine(), byteorder=sys.byteorder,
        reported_builds=builds)


def native_identity(native):
    """Pin reviewed native sources, active functions and loaded GSAS binaries."""
    from ..live.validated_backend import (active_identity, backend_identity,
                                          PINNED_PRODUCTION_STRMATH)
    source_hash = hashlib.sha256(Path(native.math.__file__).read_bytes()).hexdigest()
    if source_hash != PINNED_PRODUCTION_STRMATH and active_identity() is None:
        raise ValueError('Preparation requires the reviewed native scientific backend')
    identity = backend_identity()
    identity['numerical_runtime'] = numerical_identity()
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
