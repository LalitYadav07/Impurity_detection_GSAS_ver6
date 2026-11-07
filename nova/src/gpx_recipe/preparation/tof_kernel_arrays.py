"""Bounded data-only native TOF kernels; integrity does not grant authority."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
import zipfile
import zlib

import numpy as np

from ..live.array_limits import validate_npz_bytes
from ..symmetry.contracts import digest

SCHEMA = 'radar-native-tof-kernel/v1'
MAX_BYTES = 64 * 1024 * 1024
MAX_METADATA = 65536
MAX_REFLECTIONS = 100000
MAX_BINS = 1000000
DTYPES = dict(hkls='<i4', parameters='<f8', bounds='<i8', pointers='<i8',
              values='<f4', axis='<f8')


def checksum(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def array_identity(arrays):
    return digest({name: dict(shape=list(value.shape), dtype=value.dtype.str,
        sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())
        for name, value in sorted(arrays.items())})


def validate_arrays(arrays):
    if (not isinstance(arrays, dict) or set(arrays) != set(DTYPES)
            or any(not isinstance(a, np.ndarray) or a.dtype != np.dtype(DTYPES[k])
                   or not a.flags.c_contiguous for k, a in arrays.items())
            or sum(a.nbytes for a in arrays.values()) > MAX_BYTES):
        raise ValueError('Invalid bounded native kernel array inventory or dtype')
    h, p, b, ptr, values, axis = (arrays[k] for k in DTYPES)
    n = len(h) if h.ndim else 0
    if (h.shape != (n, 3) or not 1 <= n <= MAX_REFLECTIONS
            or p.shape != (n, 5) or b.shape != (n, 2) or ptr.shape != (n+1,)
            or values.ndim != 1 or axis.ndim != 1 or not 2 <= len(axis) <= MAX_BINS
            or not all(np.isfinite(a).all() for a in (p, values, axis))
            or np.any(np.diff(axis) <= 0) or np.max(abs(h.astype(np.int64))) > 1000000
            or len({tuple(row) for row in h}) != n
            or np.any(b[:, 0] < 0) or np.any(b[:, 1] > len(axis))
            or np.any(b[:, 0] >= b[:, 1]) or ptr[0] != 0 or ptr[-1] != len(values)
            or not np.array_equal(np.diff(ptr), b[:, 1]-b[:, 0])):
        raise ValueError('Native kernel dimensions, supports or identities are invalid')
    # Bound inputs before an uploaded chunk may reach a native numerical call.
    # These are computational admission limits, never a physical-fit rejection.
    if (np.any(p[:, 0] <= 0) or np.any(p[:, 0] > 2000000)
            or np.any(p[:, 1:3] < 1e-12) or np.any(p[:, 1:3] > 1e6)
            or np.any(p[:, 3:] < 0) or np.any(p[:, 3] > 1e12)
            or np.any(p[:, 4] > 1e6) or np.any(p[:, 3]+p[:, 4] <= 0)):
        raise ValueError('Native kernel parameters exceed the reviewed computational domain')
    return arrays


def validate_metadata(value):
    keys = {'schema', 'kernel_sha256', 'binary_sha256', 'arrays_sha256',
            'reflection_count', 'numeric_bytes', 'observations_included'}
    if (not isinstance(value, dict) or set(value) != keys or value['schema'] != SCHEMA
            or any(not checksum(value[k]) for k in ('kernel_sha256', 'binary_sha256', 'arrays_sha256'))
            or type(value['reflection_count']) is not int or not 1 <= value['reflection_count'] <= MAX_REFLECTIONS
            or type(value['numeric_bytes']) is not int or not 1 <= value['numeric_bytes'] <= MAX_BYTES
            or value['observations_included'] is not False):
        raise ValueError('Invalid native kernel metadata')
    return value


class KernelData:
    """Immutable numeric data; a reader's checksum is never native qualification.

    A chunk permits at most 64 MiB of expanded numeric arrays. Reading can use
    three bounded buffers while compressed input and immutable storage coexist;
    complete native-process memory is a separate allocation.
    """
    def __init__(self, arrays, metadata):
        validate_arrays(arrays); validate_metadata(metadata)
        if (metadata['arrays_sha256'] != array_identity(arrays)
                or metadata['reflection_count'] != len(arrays['hkls'])
                or metadata['numeric_bytes'] != sum(a.nbytes for a in arrays.values())):
            raise ValueError('Native kernel metadata differs from its array content')
        # A bytes-backed ndarray cannot regain write permission. This avoids
        # rehashing a large array for every peak, or trusting mutable flags.
        self._arrays = MappingProxyType({k: np.frombuffer(v.tobytes(order='C'), dtype=v.dtype).reshape(v.shape)
                                       for k, v in arrays.items()})
        self._metadata = MappingProxyType(dict(metadata))

    @property
    def metadata(self):
        return dict(self._metadata)

    @property
    def numeric_bytes(self):
        return self._metadata['numeric_bytes']


def encode(data):
    if not isinstance(data, KernelData):
        raise TypeError('Validated immutable native kernel data required')
    stream = io.BytesIO()
    np.savez_compressed(stream, **data._arrays)
    raw = stream.getvalue()
    validate_npz_bytes(raw, maximum_bytes=MAX_BYTES)
    return raw


def decode(raw, metadata):
    if not isinstance(raw, bytes):
        raise TypeError('Immutable native kernel bytes required')
    validate_metadata(metadata)
    try:
        validate_npz_bytes(raw, maximum_bytes=MAX_BYTES)
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if set(archive.files) != set(DTYPES):
                raise ValueError('Unexpected native kernel array inventory')
            arrays = {name: archive[name] for name in DTYPES}
    except (zipfile.BadZipFile, zlib.error, NotImplementedError, EOFError, OSError) as error:
        raise ValueError('Malformed native kernel arrays') from error
    return KernelData(arrays, metadata)


def read(path, metadata_path):
    path, metadata_path = Path(path), Path(metadata_path)
    return decode(read_bounded(path, MAX_BYTES), json.loads(read_bounded(metadata_path, MAX_METADATA)))


def read_bounded(path, maximum):
    """Bound reads on an opened regular file, including concurrent replacement."""
    if type(maximum) is not int or maximum < 1:
        raise ValueError('A positive file bound is required')
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
                raise ValueError('Native kernel files exceed their bounds')
            raw = stream.read(maximum+1)
            if len(raw) > maximum:
                raise ValueError('Native kernel file grew beyond its bound')
            return raw
    except OSError as error:
        raise ValueError('Native kernel files must be regular owned files') from error
