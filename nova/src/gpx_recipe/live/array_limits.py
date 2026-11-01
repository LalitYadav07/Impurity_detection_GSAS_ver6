"""Inspect NPZ/NPY headers before NumPy can allocate untrusted arrays."""
import math
from pathlib import Path
import zipfile

import numpy as np

MAX_ARRAY_BYTES = 384 * 1024 * 1024
MAX_ARRAY_ELEMENTS = 48_000_000
MAX_OBSERVATION_BINS = 1_000_000


def validate_npz(path, *, observation=False):
    path = Path(path)
    limit = 32 * 1024 * 1024 if observation else MAX_ARRAY_BYTES
    if path.stat().st_size > limit:
        raise ValueError('NPZ exceeds the array byte budget')
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if not entries or len(entries) > 16 or len({v.filename for v in entries}) != len(entries):
            raise ValueError('NPZ has an invalid array inventory')
        if observation and {v.filename for v in entries} != {'x.npy','y.npy','sigma.npy','mask.npy'}:
            raise ValueError('Invalid observation arrays')
        total_bytes = total_elements = 0
        dimensions = {}
        for entry in entries:
            if (not entry.filename.endswith('.npy') or '/' in entry.filename or '\\' in entry.filename
                    or entry.is_dir() or entry.flag_bits & 1 or entry.file_size > limit + 10240):
                raise ValueError('NPZ contains an unsupported or oversized array')
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1,0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=10240)
                elif version == (2,0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=10240)
                else:
                    raise ValueError('Unsupported NPY header version')
                if dtype.hasobject or dtype.fields is not None or len(shape)>3 or any(v<0 for v in shape):
                    raise ValueError('Unsupported NPZ array shape or dtype')
                count = math.prod(shape)
                size = count * dtype.itemsize
                total_bytes += size; total_elements += count
                if size > limit or total_bytes > limit or total_elements > MAX_ARRAY_ELEMENTS:
                    raise ValueError('NPZ expands beyond the array budget')
                if entry.file_size != stream.tell() + size:
                    raise ValueError('NPY payload size differs from its declared shape')
                if observation:
                    if len(shape)!=1 or not 2 <= shape[0] <= MAX_OBSERVATION_BINS:
                        raise ValueError('Invalid observation dimensions')
                    expected = 'b' if entry.filename == 'mask.npy' else 'fiu'
                    if dtype.kind not in expected or dtype.itemsize > 8:
                        raise ValueError('Unsupported observation dtype')
                    dimensions[entry.filename] = shape
        if observation and len(set(dimensions.values())) != 1:
            raise ValueError('Invalid observation dimensions')
    return True
