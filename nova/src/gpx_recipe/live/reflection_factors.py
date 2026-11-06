"""Bounded, observation-free magnetic scattering factors.

The pinned native magnetic function reads atomic parameters, its symmetry,
metric, hkl/d inventory and magnetic form-factor tables. Instrument peak widths,
the rendering grid, counts and weights are not inputs to those amplitudes.
Native peak rendering and full-profile checks remain owned by the caller.
"""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
import re
import uuid

import numpy as np

from ..preparation.identity import _plain

SCHEMA = 'radar-native-magnetic-amplitudes/v1'
MAX_BYTES = 64 * 1024 * 1024


def digest_array(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(json.dumps([a.dtype.str, a.shape]).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def scattering_identity(*, math, parameters, controls, pfx, hfx, symmetry,
                        metric, reflections, basis):
    """Exact read dependencies of the reviewed MagStructureFactor2 path.

    Phase/histogram display IDs are replaced by roles. All fixed phase
    parameters are retained conservatively; only the homogeneous moments are
    represented by their complete basis instead of their current values.
    """
    moments = {k for _, _, keys in basis.atoms for k in keys}
    atom_roles = []
    for _, index, keys in basis.atoms:
        if any(not key.startswith(pfx) for key in keys):
            raise ValueError('Scattering factors require one complete phase moment basis')
        atom_roles.append([index, [key[len(pfx):] for key in keys]])
    record = dict(schema=SCHEMA,
        native_source_sha256=hashlib.sha256(Path(math.__file__).read_bytes()).hexdigest(),
        native_dependency_sources={Path(m.__file__).name:
            hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
            for m in (math.G2lat, math.G2el)},
        active_geometry_functions={name: hashlib.sha256(inspect.getsource(
            getattr(math.G2lat, name)).encode()).hexdigest() for name in ('Gmat2AB','U6toUij')},
        active_atomic_function_sha256=hashlib.sha256(inspect.getsource(math.GetAtomFXU).encode()).hexdigest(),
        form_factor_function_sha256=hashlib.sha256(inspect.getsource(math.G2el.MagScatFac).encode()).hexdigest(),
        numpy_version=np.__version__,
        phase_parameters={k[len(pfx):]: v for k, v in parameters.items()
                          if k.startswith(pfx) and k not in moments},
        phase_controls={k[len(pfx):]: v for k, v in controls.items() if k.startswith(pfx)},
        atoms=controls['Natoms'][pfx], form_factors=controls['MFtables'],
        histogram_kind=controls[hfx+'histType'], symmetry=symmetry,
        metric=digest_array(np.asarray(metric)),
        reflections=digest_array(np.asarray(reflections)[:, :5]),
        moment_atoms=atom_roles, components=digest_array(basis.components),
        cartesian=digest_array(basis.cartesian))
    encoded = json.dumps(_plain(record), sort_keys=True, allow_nan=False, separators=(',', ':'))
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class AmplitudeBlock:
    identity: str
    factors: np.ndarray
    proof: dict

    def __post_init__(self):
        a = np.asarray(self.factors)
        if (not isinstance(self.identity, str) or not re.fullmatch('[a-f0-9]{64}', self.identity)
                or a.ndim != 3 or a.shape[1] != 6 or not 1 <= a.shape[2] <= 24
                or not len(a) or a.dtype.kind != 'f' or a.dtype.itemsize != 8
                or a.nbytes > MAX_BYTES or not np.isfinite(a).all()):
            raise ValueError('A bounded finite native amplitude block is required')
        # A read-only owning ndarray can be made writable again. Back this
        # immutable physical block with bytes so that ordinary array aliases
        # cannot change a qualified resident. Hash once at this boundary.
        storage = np.asarray(a, dtype='<f8').tobytes(order='C')
        a = np.frombuffer(storage, dtype='<f8').reshape(a.shape)
        proof = json.dumps(self.proof, sort_keys=True, allow_nan=False)
        if len(proof) > 65536:
            raise ValueError('Amplitude qualification metadata exceeds its budget')
        object.__setattr__(self, 'factors', a)
        object.__setattr__(self, 'proof', json.loads(proof))
        object.__setattr__(self, '_digest', digest_array(a))
        object.__setattr__(self, '_shape', a.shape)
        object.__setattr__(self, '_strides', a.strides)
        object.__setattr__(self, '_owner', a.base)
        object.__setattr__(self, '_storage', storage)
        object.__setattr__(self, '_address', a.ctypes.data)


class ScatteringFactorCache:
    """One worker's LRU; imported files never carry native qualification.

    Only the native compiler admits freshly checked blocks. A checksum guards
    integrity, not provenance: a loaded block must be checked against the
    current native function before becoming a qualified resident entry.
    """
    def __init__(self, directory=None, *, maximum_bytes=MAX_BYTES):
        if type(maximum_bytes) is not int or not 1 <= maximum_bytes <= MAX_BYTES:
            raise ValueError('A bounded positive amplitude-cache budget is required')
        self.directory = Path(directory).resolve() if directory is not None else None
        self.maximum_bytes = maximum_bytes
        self._entries = OrderedDict()
        self._seals = {}
        self._sources = {}

    def attach_source(self, path, *, identity):
        """Register one immutable library hint; load it only if physics match.

        Never write to this source or inherit its qualification. Current
        native checks admit the resulting block into this worker's cache.
        """
        if (not isinstance(identity,str) or not re.fullmatch('[a-f0-9]{64}',identity)
                or len(self._sources)>=5000 and identity not in self._sources):
            raise ValueError('A bounded inventory of native scattering identities is required')
        self._sources[identity] = Path(path).resolve()

    def lookup(self, identity):
        if not isinstance(identity, str) or not re.fullmatch('[a-f0-9]{64}', identity):
            raise ValueError('A native scattering identity is required')
        block = self._entries.get(identity)
        if block is not None:
            if (block.factors.flags.writeable or block.factors.dtype.str != '<f8'
                    or block.factors.shape != block._shape or block.factors.base is not block._owner
                    or block.factors.strides != block._strides
                    or block._owner.base is not block._storage
                    or block.factors.ctypes.data != block._address):
                del self._entries[identity]
                self._seals.pop(identity, None)
                raise ValueError('Resident scattering amplitudes changed')
            self._entries.move_to_end(identity)
            return block, self._seals.get(identity) == block._digest
        path = self.directory / (identity+'.npz') if self.directory is not None else None
        if path is None or not path.exists():
            path = self._sources.get(identity)
        if path is None:
            return None, False
        from .array_limits import validate_npz
        validate_npz(path, maximum_bytes=MAX_BYTES+262144)
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {'manifest', 'factors'}:
                raise ValueError('Unexpected scattering-factor file inventory')
            entry = data['manifest']
            if entry.shape != () or entry.dtype.kind != 'U' or entry.nbytes > 262144:
                raise ValueError('Invalid scattering-factor manifest')
            manifest = json.loads(str(entry))
            if (set(manifest) != {'schema', 'identity', 'sha256', 'proof', 'observations_included'}
                    or manifest['schema'] != SCHEMA or manifest['identity'] != identity
                    or manifest['observations_included'] is not False
                    or digest_array(data['factors']) != manifest['sha256']):
                raise ValueError('Scattering-factor identity or checksum differs')
            block = AmplitudeBlock(identity, data['factors'], manifest['proof'])
        return block, False

    def _admit_native(self, identity, factors, proof):
        if (proof.get('all_coefficient_directions_checked') is not True
                or proof.get('dimensions') != np.shape(factors)[-1]
                or proof.get('reflections') != len(factors)):
            raise ValueError('Complete native scattering qualification is required')
        block = AmplitudeBlock(identity, factors, proof)
        if block.factors.nbytes > self.maximum_bytes:
            return block
        self._entries.pop(identity, None)
        self._seals.pop(identity, None)
        while sum(b.factors.nbytes for b in self._entries.values()) + block.factors.nbytes > self.maximum_bytes:
            oldest, _ = self._entries.popitem(last=False)
            self._seals.pop(oldest, None)
        self._entries[identity] = block
        self._seals[identity] = block._digest
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / (identity+'.npz')
            temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
            manifest = dict(schema=SCHEMA, identity=identity, sha256=block._digest,
                proof=block.proof, observations_included=False)
            try:
                with temporary.open('xb') as stream:
                    np.savez_compressed(stream, manifest=np.array(json.dumps(manifest, allow_nan=False)),
                        factors=block.factors)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return block
