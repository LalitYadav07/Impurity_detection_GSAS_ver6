"""Bounded numerical continuation between owned training workers.

This is transport, not another optimizer or a trained model. The ordinary
checkpoint restorer must still validate every tensor against the installed
network, optimizer, scheduler and recomputed complete training identity.
Creating-job ownership belongs to the queue; uploaded checksums grant none.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from ..execution import write_json
from ..live.array_limits import validate_npz_bytes
from ..symmetry.compiler import package_signature
from . import structural_training_checkpoint as checkpoint
from .structural_model_files import MAX_BYTES,MAX_METADATA_BYTES
from .structural_training_data import MAX_ROWS
from .tof_kernel_arrays import array_identity,checksum,read_bounded

SCHEMA='radar-attention-training-continuation/v1'
STATE='training-state.npz'
RECEIPT='training-state.json'


def inspect(path,*,identity=None,epochs=None):
    """Check the data container; never replace the canonical tensor audit."""
    raw=read_bounded(path,MAX_BYTES);validate_npz_bytes(raw,maximum_bytes=MAX_BYTES)
    with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
        if set(archive.files)!=checkpoint.ARRAYS|{'metadata'}:
            raise ValueError('Invalid training continuation array inventory')
        arrays={key:archive[key] for key in archive.files}
    encoded=arrays.pop('metadata')
    if encoded.dtype!=np.uint8 or encoded.ndim!=1 or len(encoded)>MAX_METADATA_BYTES:
        raise ValueError('Invalid training continuation metadata')
    metadata=json.loads(encoded.tobytes())
    if (not isinstance(metadata,dict) or metadata.get('schema')!=checkpoint.SCHEMA
            or not checksum(metadata.get('identity')) or not checksum(metadata.get('layout'))
            or (identity is not None and metadata['identity']!=identity)
            or metadata.get('arrays_sha256')!=array_identity(arrays)):
        raise ValueError('Training continuation identity or numerical bytes changed')
    parameters=arrays['parameters']
    if (parameters.dtype!=np.float32 or parameters.ndim!=1 or not len(parameters)
            or any(arrays[key].dtype!=np.float32 or arrays[key].shape!=parameters.shape
                   for key in ('first_moment','second_moment'))
            or arrays['best_parameters'].dtype!=np.float32
            or arrays['best_parameters'].shape not in [(0,),parameters.shape]
            or any(not np.isfinite(arrays[key]).all() for key in
                   ('parameters','best_parameters','first_moment','second_moment','steps'))
            or np.any(arrays['second_moment']<0)
            or arrays['steps'].dtype!=np.float32 or arrays['steps'].ndim!=1
            or np.any(arrays['steps']<0) or np.any(arrays['steps']>10**7)
            or np.any(arrays['steps']!=np.rint(arrays['steps']))
            or arrays['initialized'].dtype!=bool or arrays['initialized'].shape!=arrays['steps'].shape
            or not np.array_equal(arrays['initialized'],arrays['steps']>0)
            or arrays['visited'].dtype!=bool or arrays['visited'].ndim!=1
            or not 1<=len(arrays['visited'])<=MAX_ROWS
            or arrays['torch_rng'].dtype!=np.uint8 or arrays['torch_rng'].ndim!=1):
        raise ValueError('Invalid bounded training continuation arrays')
    done=metadata.get('epochs_completed');warmup=metadata.get('warmup_completed')
    elapsed=metadata.get('elapsed_seconds');history=metadata.get('history')
    if (type(done)is not int or not 0<=done<=1000 or (epochs is not None and done>epochs)
            or type(warmup)is not int or not 0<=warmup<=200
            or (done and warmup!=200) or not isinstance(history,list) or len(history)!=done
            or any(not isinstance(row,dict) or row.get('epoch')!=i+1 for i,row in enumerate(history))
            or type(elapsed) not in (int,float) or not np.isfinite(elapsed) or elapsed<0):
        raise ValueError('Training continuation must end at a complete warmup or epoch boundary')
    return raw,metadata


def snapshot(directory,destination,*,identity,epochs):
    """Export only a complete numerical checkpoint, without a feature cache."""
    if not checksum(identity) or type(epochs)is not int or not 1<=epochs<=1000:
        raise ValueError('A complete fixed training identity and epoch target are required')
    directory=Path(directory)
    owner=json.loads(read_bounded(directory/'training-identity.json',1024))
    if owner!={'schema':'radar-owned-attention-training-run/v1','identity':identity}:
        raise ValueError('Training continuation has no matching owned run')
    raw,metadata=inspect(directory/'training-checkpoint.npz',identity=identity,epochs=epochs)
    receipt=dict(schema=SCHEMA,identity=identity,implementation_sha256=package_signature(),
        checkpoint_sha256=hashlib.sha256(raw).hexdigest(),epochs=epochs,
        epochs_completed=metadata['epochs_completed'],warmup_completed=metadata['warmup_completed'],
        elapsed_seconds=metadata['elapsed_seconds'],role='optimizer_continuation_only',
        trained_model=False,scientific_result=False,candidate_pruning_authorized=False)
    destination=Path(destination)
    if destination.exists() or destination.is_symlink():
        existing,data=read(destination,identity=identity,epochs=epochs)
        if existing!=receipt or data!=raw:
            raise ValueError('A different training continuation already occupies its immutable destination')
        return existing
    destination.parent.mkdir(parents=True,exist_ok=True)
    # Publish the pair together. A process dying during serialization leaves
    # no half-written checkpoint at the acknowledged destination.
    with tempfile.TemporaryDirectory(prefix='training-export-',dir=destination.parent) as temporary:
        staged=Path(temporary)/'complete';staged.mkdir()
        with (staged/STATE).open('wb') as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        write_json(staged/RECEIPT,receipt)
        os.rename(staged,destination)
    return receipt


def read(directory,*,identity=None,epochs=None):
    """A typed transfer does not make the contained optimizer authoritative."""
    directory=Path(directory);receipt=json.loads(read_bounded(directory/RECEIPT,8192))
    expected={'schema','identity','implementation_sha256','checkpoint_sha256','epochs',
        'epochs_completed','warmup_completed','elapsed_seconds','role','trained_model',
        'scientific_result','candidate_pruning_authorized'}
    if (not isinstance(receipt,dict) or set(receipt)!=expected or receipt['schema']!=SCHEMA
            or not checksum(receipt['identity']) or not checksum(receipt['checkpoint_sha256'])
            or receipt['implementation_sha256']!=package_signature()
            or type(receipt['epochs'])is not int or not 1<=receipt['epochs']<=1000
            or (identity is not None and receipt['identity']!=identity)
            or (epochs is not None and receipt['epochs']!=epochs)
            or receipt['role']!='optimizer_continuation_only' or receipt['trained_model'] is not False
            or receipt['scientific_result'] is not False or receipt['candidate_pruning_authorized'] is not False):
        raise ValueError('Invalid bound training continuation receipt')
    raw,metadata=inspect(directory/STATE,identity=receipt['identity'],epochs=receipt['epochs'])
    if (hashlib.sha256(raw).hexdigest()!=receipt['checkpoint_sha256']
            or any(receipt[key]!=metadata[key] for key in
                   ('epochs_completed','warmup_completed','elapsed_seconds'))):
        raise ValueError('Training continuation receipt differs from its numerical state')
    return receipt,raw


def restore(directory,destination,*,identity,epochs):
    """Install under exclusive ownership, before the optimizer acquires it."""
    if not checksum(identity) or type(epochs)is not int or not 1<=epochs<=1000:
        raise ValueError('A complete fixed training identity and epoch target are required')
    read(directory,identity=identity,epochs=epochs)
    with checkpoint.owned_run(destination,identity):
        return _restore_owned(directory,destination,identity=identity,epochs=epochs)


def _restore_owned(directory,destination,*,identity,epochs):
    """Never roll back newer work or replace an alternate optimizer branch."""
    receipt,raw=read(directory,identity=identity,epochs=epochs)
    destination=Path(destination)
    owner=json.loads(read_bounded(destination/'training-identity.json',1024))
    if owner!={'schema':'radar-owned-attention-training-run/v1','identity':identity}:
        raise ValueError('Restore training state only inside its matching owned run')
    target=destination/'training-checkpoint.npz'
    if target.exists() or target.is_symlink():
        existing,metadata=inspect(target,identity=identity,epochs=epochs)
        if existing==raw:return dict(receipt,disposition='identical_owned_checkpoint_retained')
        before=(receipt['epochs_completed'],receipt['warmup_completed'])
        after=(metadata['epochs_completed'],metadata['warmup_completed'])
        if after>before:return dict(receipt,disposition='newer_owned_checkpoint_retained')
        raise ValueError('Training continuation cannot overwrite a different owned checkpoint')
    descriptor,name=tempfile.mkstemp(prefix='training-transfer-',suffix='.partial',dir=destination)
    try:
        with os.fdopen(descriptor,'wb') as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        os.replace(name,target)
    finally:Path(name).unlink(missing_ok=True)
    return dict(receipt,disposition='copied_for_complete_native_checkpoint_audit')
