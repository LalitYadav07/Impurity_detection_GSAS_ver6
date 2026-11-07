"""Worker-local feature caches for large structural preparation catalogs.

These are disposable training working files, not portable experiment library
assets. Only a complete checksum-bound cache may be reopened for training.
"""
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np

from ..execution import write_json
from ..live.array_limits import validate_npz_bytes
from .structural_cohort import assemble_into, read_block
from .structural_training_data import MAX_ROWS, MAX_STORE_BYTES, open_rows
from .tof_kernel_arrays import read_bounded

SCHEMA = 'radar-structural-feature-cache/v1'
MAX_METADATA_BYTES = 64*1024**2
MAX_TARGET_BYTES = 64*1024**2
ARRAYS = {'compatible_labels', 'classes', 'split', 'q_edges', 'reference_features'}


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


class DiskFeatures:
    def __init__(self, destination):
        self.path = Path(destination)/'features.npy'
        self.stream = None; self.count = 0

    def reserve(self, count, bins):
        if not 1 <= count <= MAX_ROWS or count*8*bins*4 > MAX_STORE_BYTES:
            raise ValueError('Structural feature working files exceed their declared disk budget')
        self.capacity = count; self.bins = bins
        self.stream = self.path.open('xb')
        np.lib.format.write_array_header_1_0(self.stream,
            dict(descr='<f4', fortran_order=False, shape=(count, 8, bins)))
        self.offset = self.stream.tell()
        self.stream.truncate(self.offset+count*8*bins*4)

    def append(self, feature):
        if self.count >= self.capacity:
            raise ValueError('Structural feature count exceeded its reserved capacity')
        self.stream.write(feature.astype('<f4', copy=False).tobytes(order='C')); self.count += 1

    def finish(self, rows):
        if rows != self.count:
            raise ValueError('Structural feature cache has an incomplete row inventory')
        self.stream.flush(); os.fsync(self.stream.fileno())
        return None

    def close(self):
        if self.stream is not None:
            self.stream.close(); self.stream = None


def build(directories, destination, *, bins=2048, split_seed=3917, augmenter=None):
    """Read source blocks twice, with bounded resident feature memory."""
    directories = list(directories)
    if not 1 <= len(directories) <= 5000:
        raise ValueError('A bounded structural preparation catalog is required')
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=False)
    sink = DiskFeatures(destination)
    try:
        cohort = assemble_into(lambda: (read_block(p) for p in directories), sink,
                               bins=bins, split_seed=split_seed, augmenter=augmenter)
    finally:
        sink.close()
    stream = io.BytesIO()
    np.savez_compressed(stream, **{k: cohort[k] for k in ARRAYS})
    raw = stream.getvalue(); validate_npz_bytes(raw, maximum_bytes=MAX_TARGET_BYTES)
    (destination/'targets.npz').write_bytes(raw)
    metadata = {k: v for k, v in cohort.items() if k not in ARRAYS|{'features'}}
    metadata.update(schema=SCHEMA, rows=cohort['unique_examples'], capacity=sink.capacity,
                    bins=bins, features_sha256=digest_file(sink.path),
                    targets_sha256=hashlib.sha256(raw).hexdigest(), split_seed=split_seed)
    # Admission must match write_json's actual representation; its indentation
    # can make a large inventory exceed the reader's bound after publication.
    if len((json.dumps(metadata, indent=2, allow_nan=False)+'\n').encode()) > MAX_METADATA_BYTES:
        raise ValueError('Structural feature-cache metadata exceeds its bounded inventory')
    write_json(destination/'store.json', metadata)
    return load(destination)


def load(directory):
    """Reopen completed numerical state; no Python object/pickle is accepted."""
    directory = Path(directory)
    d = json.loads(read_bounded(directory/'store.json', MAX_METADATA_BYTES))
    if (not isinstance(d, dict) or d.get('schema') != SCHEMA
            or type(d.get('rows')) is not int or not 1 <= d['rows'] <= MAX_ROWS
            or d.get('unique_examples') != d['rows']
            or not isinstance(d.get('groups'), list) or len(d['groups']) != d['rows']
            or not isinstance(d.get('references'), list) or len(d['references']) != d['rows']):
        raise ValueError('A complete structural feature-cache inventory is required')
    raw = read_bounded(directory/'targets.npz', MAX_TARGET_BYTES)
    if hashlib.sha256(raw).hexdigest() != d.get('targets_sha256'):
        raise ValueError('Structural feature-cache targets changed')
    validate_npz_bytes(raw, maximum_bytes=MAX_TARGET_BYTES)
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if set(archive.files) != ARRAYS:
            raise ValueError('Unexpected structural feature-cache target inventory')
        arrays = {k: archive[k] for k in ARRAYS}
    n = d['rows']; classes = arrays['classes']; targets = arrays['compatible_labels']
    if (classes.dtype != np.int32 or classes.ndim != 1 or not 1 <= len(classes) <= 230
            or np.any(classes < 1) or np.any(classes > 230) or np.any(np.diff(classes) <= 0)
            or targets.dtype != bool or targets.shape != (n, len(classes)) or not targets.any(1).all()
            or arrays['split'].dtype != np.int8 or arrays['split'].shape != (n,)
            or set(arrays['split']) != {0, 1, 2}):
        raise ValueError('Invalid structural feature-cache target dimensions')
    features = open_rows(directory/'features.npy', sha256=d.get('features_sha256'),
                         capacity=d.get('capacity'), rows=n, bins=d.get('bins'))
    return dict(d, **arrays, features=features)
