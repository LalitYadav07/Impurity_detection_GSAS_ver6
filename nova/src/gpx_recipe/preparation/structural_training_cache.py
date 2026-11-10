"""Choose one bounded worker-local representation for unchanged training rows.

Cache a cohort once when its complete upper bound fits the existing row/disk
limits and leaves filesystem headroom. Otherwise retain native-pair replay.
Both paths use the same assembler, features, aliases, folds and augmentation.
No scientific sample, amplitude, epoch or candidate is reduced by this choice.
"""
from pathlib import Path
import json
import shutil
import stat

from .structural_training_data import MAX_ROWS,MAX_STORE_BYTES
from .structural_store import MAX_METADATA_BYTES,MAX_TARGET_BYTES
from .structural_training_blocks import representation,context,REPRESENTATION
from .tof_kernel_arrays import read_bounded

MINIMUM_HEADROOM=1024**3
WORKING_METADATA_ALLOWANCE=MAX_METADATA_BYTES+MAX_TARGET_BYTES


def discard_completed(destination):
    """Remove only the closed, complete disposable cache from a finished worker.

    Replay owns an unlinked temporary file and leaves this directory empty.
    A file-backed cache has exactly these three files. Preserve unexpected or
    partial state for inspection instead of recursively deleting it.
    """
    destination=Path(destination)
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError('Completed cache cleanup requires an owned regular directory')
    paths=list(destination.iterdir())
    if paths and ({p.name for p in paths}!={'features.npy','targets.npz','store.json'}
            or any(not stat.S_ISREG(p.lstat().st_mode) for p in paths)):
        raise ValueError('Preserve unexpected or partial training-cache state')
    for path in paths:path.unlink()


def placement(capacity,bins,free_bytes):
    if (type(capacity)is not int or capacity<1 or type(bins)is not int or not 64<=bins<=8192
            or type(free_bytes)is not int or free_bytes<0):
        raise ValueError('Complete bounded training dimensions and observed disk space required')
    required=capacity*3*bins*4+10240+WORKING_METADATA_ALLOWANCE
    reserve=max(MINIMUM_HEADROOM,(free_bytes+9)//10)
    fits=(capacity<=MAX_ROWS and capacity*3*bins*4<=MAX_STORE_BYTES and required+reserve<=free_bytes)
    return dict(schema='radar-training-feature-placement/v1',storage='file_backed' if fits else 'native_pair_replay',
        complete_upper_bound_rows=capacity,expanded_feature_bytes=capacity*3*bins*4,
        required_working_bytes=required,observed_free_bytes=free_bytes,reserved_free_bytes=reserve,
        scientific_scope_changed=False)


def build(directories,destination,*,bins=4096,split_seed=69271,augmenter=None,names='examples',reference_kind=None):
    directories=tuple(Path(p) for p in directories);destination=Path(destination)
    if not 1<=len(directories)<=5000:raise ValueError('A complete bounded native block inventory is required')
    if destination.is_symlink() or destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError('Training feature placement requires fresh owned working state')
    if not isinstance(names,str) or not names or len(names)>64 or Path(names).name!=names or names in ('.','..'):
        raise ValueError('A closed native block basename is required')
    count=0
    for directory in directories:
        document=json.loads(read_bounded(directory/(names+'.json'),8*1024**2))
        if representation(document)!=REPRESENTATION:raise ValueError('Training cache requires paired native examples')
        context(document);rows=document.get('examples')
        if type(rows)is not int or rows<1:raise ValueError('Every native block needs its complete example count')
        count+=rows
    variants=0 if augmenter is None else augmenter.controls.variants
    capacity=count*(1+variants)
    policy=placement(capacity,bins,shutil.disk_usage(destination.parent).free)
    options=dict(bins=bins,split_seed=split_seed,augmenter=augmenter,names=names,reference_kind=reference_kind)
    if policy['storage']=='file_backed':
        from .structural_paired_store import build as cached
        # Only an empty worker-owned directory may be replaced. A partial or
        # changed previous cache is rejected above, never silently reused.
        if destination.exists():destination.rmdir()
        result=cached(directories,destination,**options)
    else:
        from .structural_replay import build as replay
        destination.mkdir(exist_ok=True)
        result=replay(directories,destination=destination,**options)
    result['feature_placement']=policy
    return result
