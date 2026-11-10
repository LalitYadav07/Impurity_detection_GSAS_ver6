"""Three-curve working caches with bounded resident memory and sealed bytes."""
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np

from ..live.array_limits import validate_npz_bytes
from .structural_cohort import assemble_into
from .structural_paired_blocks import read,checksum
from .structural_store import DiskFeatures,digest_file,MAX_METADATA_BYTES,MAX_TARGET_BYTES
from .structural_training_data import MAX_ROWS,open_rows
from .tof_kernel_arrays import read_bounded

SCHEMA='radar-three-curve-structural-feature-cache/v1'
ARRAYS={'compatible_labels','classes','split','q_edges','reference_features'}


def build(directories,destination,*,bins=4096,split_seed=69271,augmenter=None,names='pairs',reference_kind=None):
    directories=list(directories)
    if not 1<=len(directories)<=5000:raise ValueError('A bounded paired example catalog is required')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    sink=DiskFeatures(destination,channels=3)
    def blocks(indices=None):
        return (read(directories[i],names=names) for i in (
            range(len(directories)) if indices is None else indices))
    try:
        cohort=assemble_into(blocks,sink,bins=bins,
            split_seed=split_seed,augmenter=augmenter,paired=True,subset_factory=blocks,reference_kind=reference_kind)
    finally:sink.close()
    stream=io.BytesIO();np.savez_compressed(stream,**{k:cohort[k] for k in ARRAYS})
    raw=stream.getvalue();validate_npz_bytes(raw,maximum_bytes=MAX_TARGET_BYTES)
    (destination/'targets.npz').write_bytes(raw)
    metadata={k:v for k,v in cohort.items() if k not in ARRAYS|{'features'}}
    metadata.update(schema=SCHEMA,rows=cohort['unique_examples'],capacity=sink.capacity,bins=bins,
        features_sha256=digest_file(sink.path),targets_sha256=hashlib.sha256(raw).hexdigest(),split_seed=split_seed)
    # This is large disposable worker state. Compact JSON retains all source
    # references and simulation provenance without indentation amplification.
    encoded=(json.dumps(metadata,separators=(',',':'),allow_nan=False)+'\n').encode()
    if len(encoded)>MAX_METADATA_BYTES:raise ValueError('Paired working-cache metadata exceeds its bound')
    with (destination/'store.partial').open('xb') as out:
        out.write(encoded);out.flush();os.fsync(out.fileno())
    os.replace(destination/'store.partial',destination/'store.json')
    return load(destination)


def load(directory):
    from .structural_attention_weights import REFERENCE_KINDS
    directory=Path(directory);d=json.loads(read_bounded(directory/'store.json',MAX_METADATA_BYTES))
    if (not isinstance(d,dict) or d.get('schema')!=SCHEMA or d.get('channels')!=3
            or d.get('representation')!='reference-current-signed-difference/shared-Q-asinh/v1'
            or not isinstance(d.get('reference_kind'),str) or d['reference_kind'] not in REFERENCE_KINDS
            or d.get('partition_method')!='class-supported-whole-trajectories/v1'
            or type(d.get('rows')) is not int or not 1<=d['rows']<=MAX_ROWS
            or d.get('unique_examples')!=d['rows']
            or not isinstance(d.get('groups'),list) or len(d['groups'])!=d['rows']
            or any(not isinstance(g,str) or not 1<=len(g)<=500 for g in d['groups'])
            or not isinstance(d.get('references'),list) or len(d['references'])!=d['rows']
            or not checksum(d.get('parent_sha256')) or not checksum(d.get('instrument_sha256'))):
        raise ValueError('A complete three-curve paired working cache is required')
    raw=read_bounded(directory/'targets.npz',MAX_TARGET_BYTES)
    if hashlib.sha256(raw).hexdigest()!=d.get('targets_sha256'):
        raise ValueError('Paired cache targets changed')
    validate_npz_bytes(raw,maximum_bytes=MAX_TARGET_BYTES)
    with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
        if set(archive.files)!=ARRAYS:raise ValueError('Unexpected paired cache target inventory')
        a={k:archive[k] for k in ARRAYS}
    classes=a['classes'];targets=a['compatible_labels'];split=a['split'];edges=a['q_edges'];reference=a['reference_features']
    n=d['rows'];bins=d.get('bins')
    if (type(bins) is not int or not 64<=bins<=8192
            or classes.dtype!=np.int32 or classes.ndim!=1 or not 2<=len(classes)<=230
            or np.any(classes<1) or np.any(classes>230) or np.any(np.diff(classes)<=0)
            or targets.dtype!=bool or targets.shape!=(n,len(classes)) or not targets.any(1).all()
            or split.dtype!=np.int8 or split.shape!=(n,) or set(split)!={0,1,2}
            or not all(targets[split==i].any(0).all() for i in range(3))
            or edges.dtype!=np.float64 or edges.shape!=(bins+1,) or not np.isfinite(edges).all()
            or np.any(edges<=0) or np.any(np.diff(edges)<=0)
            or reference.dtype!=np.float32 or reference.shape!=(3,bins) or not np.isfinite(reference).all()
            or hashlib.sha256(reference.tobytes()).hexdigest()!=d.get('reference_feature_sha256')):
        raise ValueError('Invalid three-curve cache dimensions or incomplete class holdouts')
    assignments={}
    for group,code in zip(d['groups'],split):
        assignments.setdefault(group,set()).add(int(code))
    if any(len(codes)!=1 for codes in assignments.values()):
        raise ValueError('A physical trajectory crossed the paired cache holdouts')
    features=open_rows(directory/'features.npy',sha256=d.get('features_sha256'),capacity=d.get('capacity'),
        rows=n,bins=bins,channels=3)
    return dict(d,**a,features=features)
