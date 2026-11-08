"""Bounded batches for dense or worker-local disk-backed ranking features."""
import hashlib
import os
import stat

import numpy as np

MAX_DENSE_BYTES = 384*1024**2
MAX_STORE_BYTES = 8*1024**3
MAX_ROWS = 131072
MAX_BATCH_ROWS = 256
_PERMIT = object()


class FeatureRows:
    """A feature matrix that cannot accidentally materialize as one ndarray."""
    def __init__(self, array, rows, *, storage, descriptor=None, offset=0, bins=None, channels=8, _permit=None):
        if _permit is not _PERMIT:
            raise TypeError('Use a checked dense array or owned feature-cache reader')
        self._array = array; self._descriptor = descriptor; self._offset = offset
        self.shape = (rows, *(array.shape[1:] if array is not None else (channels, bins)))
        self.dtype = np.dtype('float32'); self.storage = storage; self.ndim = 3
        self._file_stat = os.fstat(descriptor) if descriptor is not None else None

    def __len__(self):
        return self.shape[0]

    def take(self, indices):
        indices = np.asarray(indices)
        if (indices.ndim != 1 or indices.dtype.kind not in 'iu'
                or len(indices) > MAX_BATCH_ROWS or np.any(indices < 0) or np.any(indices >= self.shape[0])):
            raise ValueError('Read a bounded batch of valid structural feature rows')
        if self._descriptor is None:
            if self._array is None:
                raise ValueError('The structural feature cache is closed')
            value = np.array(self._array[indices], dtype=np.float32, copy=True, order='C')
        else:
            current = os.fstat(self._descriptor)
            if (current.st_size, current.st_mtime_ns) != (self._file_stat.st_size, self._file_stat.st_mtime_ns):
                raise ValueError('The completed structural feature cache changed during training')
            value = np.empty((len(indices), *self.shape[1:]), np.float32)
            size = self.shape[1]*self.shape[2]*4
            for i, row in enumerate(indices):
                raw = os.pread(self._descriptor, size, self._offset+int(row)*size)
                if len(raw) != size:
                    raise ValueError('Incomplete structural feature-cache read')
                value[i] = np.frombuffer(raw, dtype='<f4').reshape(self.shape[1:])
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite structural features cannot enter training')
        return value

    def close(self):
        if self._descriptor is not None:
            os.close(self._descriptor); self._descriptor = None
        self._array = None


def feature_rows(value):
    if isinstance(value, FeatureRows):
        return value
    if (not isinstance(value, np.ndarray) or value.ndim != 3 or value.dtype != np.float32
            or value.shape[1] not in (3,8) or not 64 <= value.shape[2] <= 8192
            or not 1 <= len(value) <= MAX_ROWS or value.nbytes > MAX_DENSE_BYTES
            or not np.isfinite(value).all()):
        raise ValueError('Use bounded finite structural features or a checked disk-backed cache')
    return FeatureRows(value, len(value), storage='in_memory', _permit=_PERMIT)


def open_rows(path, *, sha256, capacity, rows, bins, channels=8):
    """Inspect and hash the same regular file descriptor that supplies bounded row reads.

    These NPY files are internal working state, not uploaded model programs
    or portable experiment-library assets. Completed cache files are read in bounded batches; no pickle or object dtype is accepted.
    """
    if (type(capacity) is not int or type(rows) is not int or type(bins) is not int
            or type(channels) is not int or channels not in (3,8)
            or not 1 <= rows <= capacity <= MAX_ROWS or not 64 <= bins <= 8192
            or capacity*channels*bins*4 > MAX_STORE_BYTES):
        raise ValueError('Invalid bounded structural feature-cache dimensions')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_STORE_BYTES+10240:
            raise ValueError('The structural feature cache must be a bounded regular file')
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=10240)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=10240)
        else:
            raise ValueError('Unsupported structural feature-cache header')
        offset = stream.tell()
        if (shape != (capacity, channels, bins) or fortran or dtype != np.dtype('<f4')
                or info.st_size != offset+capacity*channels*bins*4):
            raise ValueError('Feature-cache content differs from its declared dimensions')
        stream.seek(0); digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
        if digest.hexdigest() != sha256:
            raise ValueError('The completed structural feature cache changed')
        owned = os.dup(stream.fileno())
    return FeatureRows(None, rows, storage='file_backed', descriptor=owned,
                       offset=offset, bins=bins, channels=channels, _permit=_PERMIT)


def batches(rows, indices, *, size=64):
    if type(size) is not int or not 1 <= size <= MAX_BATCH_ROWS:
        raise ValueError('Use a bounded structural training batch size')
    for start in range(0, len(indices), size):
        selected = indices[start:start+size]
        yield selected, rows.take(selected)


def feature_digest(rows):
    """Bind the same complete feature stream for every storage representation.

    Owned replay rows captured this digest while the shared assembler actually
    computed every row. Their source/bank mutation checks still run before it
    is reused. Other representations hash current bounded row reads.
    """
    rows=feature_rows(rows)
    from .structural_replay import ReplayRows
    if isinstance(rows, ReplayRows):return rows.feature_sha256()
    digest=hashlib.sha256()
    for _,values in batches(rows,np.arange(len(rows)),size=MAX_BATCH_ROWS):
        digest.update(values.tobytes(order='C'))
    return digest.hexdigest()


def channel_statistics(rows, indices, *, minimum_scale=.02, center=True):
    """Two-pass float64 statistics, with bounded feature working memory."""
    if (type(minimum_scale) not in (int,float) or not np.isfinite(minimum_scale)
            or not 0<minimum_scale<=1 or type(center) is not bool):
        raise ValueError('Finite bounded channel normalization controls required')
    if not len(indices):
        raise ValueError('Normalization needs independent training rows')
    channels=rows.shape[1]
    total = np.zeros(channels, np.float64); count = 0
    for _, values in batches(rows, indices):
        total += values.sum(axis=(0, 2), dtype=np.float64)
        count += len(values)*values.shape[2]
    mean = total/count; square = np.zeros(channels, np.float64)
    for _, values in batches(rows, indices):
        delta = values.astype(np.float64)-mean[:, None]
        square += np.einsum('ijk,ijk->j', delta, delta)
    scale = np.maximum(np.sqrt(square/count), minimum_scale)
    mean = mean.astype(np.float32)[:, None]; scale = scale.astype(np.float32)[:, None]
    if not center:mean[:]=0.
    if channels==8:
        # Only the legacy representation has mask/coordinate channels.
        # Channel 2 in the new representation is a physical signed residual.
        mean[[2, 3]] = 0.; scale[[2, 3]] = 1.
    return mean, scale
