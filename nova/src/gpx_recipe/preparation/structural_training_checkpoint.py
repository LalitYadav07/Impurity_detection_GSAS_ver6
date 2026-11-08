"""Owned, data-only training recovery; never an uploaded model or certificate.

The installed CPU attention/AdamW code determines every tensor and optimizer.
One atomic NPZ retains model, optimizer and random state at warmup/epoch
boundaries. Recovery binds actual feature bytes, labels, groups and code.
These disposable working files are excluded from portable library artifacts.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tempfile

import numpy as np

from ..execution import write_json
from ..live.array_limits import validate_npz_bytes
from ..symmetry.compiler import package_signature
from . import structural_attention_weights as weights
from .structural_model_files import MAX_BYTES,MAX_METADATA_BYTES
from .structural_training_data import feature_digest
from .tof_kernel_arrays import array_identity,checksum,read_bounded

SCHEMA='radar-owned-attention-training-checkpoint/v1'
REPLAY_SCHEMA='radar-owned-attention-training-checkpoint/v2'
ARRAYS={'parameters','best_parameters','first_moment','second_moment','steps',
        'initialized','visited','torch_rng'}


def checked_row_layout(metadata, visited):
    """Legacy checkpoints keep their limit; large rows need the new layout."""
    from .structural_training_data import MAX_ROWS
    from . import structural_replay_limits as limits
    if visited.dtype!=bool or visited.ndim!=1 or not len(visited):
        raise ValueError('Invalid checkpoint row visitation inventory')
    if metadata.get('schema')==SCHEMA:
        if len(visited)>MAX_ROWS or 'row_layout' in metadata:
            raise ValueError('Legacy checkpoint exceeds its bounded row inventory')
        return None
    if metadata.get('schema')!=REPLAY_SCHEMA:
        raise ValueError('Unsupported training checkpoint row schema')
    return limits.validate(metadata.get('row_layout'),rows=len(visited))


def identity(cohort,rows,targets,split,clean,mean,scale,*,epochs,seed,workers):
    import torch
    digest=hashlib.sha256()
    metadata={key:cohort.get(key) for key in ('parent_sha256','instrument_sha256','input_sha256',
        'groups','example_simulations','references','candidates','candidate_groups','class_candidate_ids',
        'reference_feature_sha256','training_domain','augmentation','augmentation_sha256','original_examples',
        'augmented_examples','ambiguous_examples','channels','representation','reference_kind','partition_method','partition_anchor',
        'mode_projection_inventory','mode_projection_references','mode_projection_coverage','calibration_partition',
        'training_objective','checkpoint_selection','mode_family_target_identity','mode_family_supervision','additive_pair_inventory','mode_readout')}
    metadata.update(implementation=package_signature(),architecture=(weights.MODE_ARCHITECTURE if cohort.get('mode_readout') is not None else weights.ARCHITECTURE),
        torch_version=str(torch.__version__),numpy_version=np.__version__,epochs=epochs,seed=seed,workers=workers)
    from .structural_training_data import MAX_ROWS
    if len(rows)>MAX_ROWS:metadata['row_layout']=rows.row_layout
    for chunk in json.JSONEncoder(sort_keys=True,separators=(',',':'),allow_nan=False).iterencode(metadata):
        digest.update(chunk.encode())
    for value in (targets,split,clean,mean,scale,cohort['classes'],cohort['q_edges'],cohort['reference_features']):
        digest.update(value.dtype.str.encode());digest.update(str(value.shape).encode());digest.update(value.tobytes())
    digest.update(b'complete-feature-stream-sha256/v1\0')
    digest.update(rows.dtype.str.encode());digest.update(str(rows.shape).encode())
    digest.update(bytes.fromhex(feature_digest(rows)))
    return digest.hexdigest()


@contextmanager
def owned_run(directory,run_identity):
    """One writer, with exact identity even after an interrupted publication."""
    if not checksum(run_identity):raise ValueError('A complete training recovery identity is required')
    directory=Path(directory)
    if directory.is_symlink():raise ValueError('Training recovery needs an owned regular directory')
    directory.mkdir(parents=True,exist_ok=True)
    descriptor=os.open(directory/'.training.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW|os.O_NONBLOCK,0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):raise ValueError('Training lock must be a regular owned file')
        try:fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise ValueError('This training destination already has an active owner') from error
        anchor=directory/'training-identity.json'
        expected=dict(schema='radar-owned-attention-training-run/v1',identity=run_identity)
        if anchor.exists() or anchor.is_symlink():
            if json.loads(read_bounded(anchor,1024))!=expected:
                raise ValueError('Training recovery inputs, settings or implementation changed')
        else:
            if any(path.name!='.training.lock' for path in directory.iterdir()):
                raise ValueError('Existing training work has no matching recovery identity')
            write_json(anchor,expected)
        yield directory
    finally:os.close(descriptor)


def _plane(state):
    return np.concatenate([value.detach().cpu().numpy().ravel() for value in state.values()])


def _state(model,plane):
    import torch
    template=model.state_dict();offset=0;result={}
    for name,tensor in template.items():
        size=tensor.numel()
        result[name]=torch.from_numpy(plane[offset:offset+size].copy().reshape(tensor.shape));offset+=size
    return result


def save(directory,run_identity,model,optimizer,scheduler,rng,visited,*,warmup,epochs,
         history,best,selected_epoch,best_key,elapsed_seconds,row_layout=None):
    import torch
    parameters,layout=weights.pack(model);named=list(model.named_parameters())
    if list(model.state_dict())!=[name for name,_ in named]:
        raise ValueError('Checkpoint optimizer and model must share the installed complete tensor layout')
    first=[];second=[];steps=[];initialized=[]
    for _,parameter in named:
        state=optimizer.state.get(parameter,{})
        initialized.append(bool(state));steps.append(float(state['step']) if state else 0.)
        first.append(state['exp_avg'].detach().numpy().ravel() if state else np.zeros(parameter.numel(),np.float32))
        second.append(state['exp_avg_sq'].detach().numpy().ravel() if state else np.zeros(parameter.numel(),np.float32))
    arrays=dict(parameters=parameters,best_parameters=np.empty(0,np.float32) if best is None else _plane(best),
        first_moment=np.concatenate(first),second_moment=np.concatenate(second),steps=np.asarray(steps,np.float32),
        initialized=np.asarray(initialized,bool),visited=visited.copy(),torch_rng=torch.get_rng_state().numpy().copy())
    metadata=dict(schema=SCHEMA,identity=run_identity,layout=layout,arrays_sha256=array_identity(arrays),
        warmup_completed=warmup,epochs_completed=epochs,history=history,selected_epoch=selected_epoch,
        best_key=best_key,numpy_rng=rng.bit_generator.state,optimizer_groups=optimizer.state_dict()['param_groups'],
        scheduler=scheduler.state_dict(),elapsed_seconds=elapsed_seconds)
    if row_layout is not None:metadata.update(schema=REPLAY_SCHEMA,row_layout=row_layout.copy())
    checked_row_layout(metadata,arrays['visited'])
    encoded=json.dumps(metadata,separators=(',',':'),allow_nan=False).encode()
    if len(encoded)>MAX_METADATA_BYTES:raise ValueError('Training checkpoint metadata exceeds its bound')
    arrays['metadata']=np.frombuffer(encoded,np.uint8)
    descriptor,name=tempfile.mkstemp(prefix='training-checkpoint-',suffix='.partial',dir=directory)
    try:
        with os.fdopen(descriptor,'wb') as stream:
            # Worker-local state is overwritten atomically and never shipped.
            # Avoid spending fitting CPU on repeated float-plane compression.
            np.savez(stream,**arrays);stream.flush();os.fsync(stream.fileno())
        validate_npz_bytes(read_bounded(name,MAX_BYTES),maximum_bytes=MAX_BYTES)
        os.replace(name,Path(directory)/'training-checkpoint.npz')
    finally:Path(name).unlink(missing_ok=True)


def restore(directory,run_identity,model,optimizer,scheduler,rng,visited,*,epochs,warmup_steps,row_layout=None):
    import torch
    path=Path(directory)/'training-checkpoint.npz'
    if not path.exists() and not path.is_symlink():return None
    raw=read_bounded(path,MAX_BYTES);validate_npz_bytes(raw,maximum_bytes=MAX_BYTES)
    with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
        if set(archive.files)!=ARRAYS|{'metadata'}:raise ValueError('Invalid training checkpoint inventory')
        arrays={key:archive[key] for key in archive.files}
    encoded=arrays.pop('metadata')
    if encoded.dtype!=np.uint8 or encoded.ndim!=1 or len(encoded)>MAX_METADATA_BYTES:
        raise ValueError('Invalid training checkpoint metadata')
    metadata=json.loads(encoded.tobytes())
    if not isinstance(metadata,dict) or checked_row_layout(metadata,arrays['visited'])!=row_layout:
        raise ValueError('Training checkpoint differs from the current row layout')
    current,layout=weights.pack(model);named=list(model.named_parameters());count=len(named)
    if (not isinstance(metadata,dict) or metadata.get('identity')!=run_identity
            or metadata.get('layout')!=layout or metadata.get('arrays_sha256')!=array_identity(arrays)
            or any(arrays[k].dtype!=np.float32 or arrays[k].shape!=current.shape or not np.isfinite(arrays[k]).all()
                   for k in ('parameters','first_moment','second_moment'))
            or arrays['best_parameters'].dtype!=np.float32 or arrays['best_parameters'].shape not in [(0,),current.shape]
            or not np.isfinite(arrays['best_parameters']).all() or np.any(arrays['second_moment']<0)
            or arrays['initialized'].dtype!=bool or arrays['initialized'].shape!=(count,)
            or arrays['steps'].dtype!=np.float32 or arrays['steps'].shape!=(count,)
            or not np.isfinite(arrays['steps']).all() or np.any(arrays['steps']<0) or np.any(arrays['steps']>10**7)
            or np.any(arrays['steps']!=np.rint(arrays['steps'])) or not np.array_equal(arrays['initialized'],arrays['steps']>0)
            or arrays['visited'].dtype!=bool or arrays['visited'].shape!=visited.shape
            or arrays['torch_rng'].dtype!=np.uint8 or arrays['torch_rng'].shape!=torch.get_rng_state().numpy().shape):
        raise ValueError('Training checkpoint data, identity or tensor layout changed')
    done=metadata.get('epochs_completed');warmup=metadata.get('warmup_completed');history=metadata.get('history')
    if (type(done) is not int or not 0<=done<=epochs or type(warmup) is not int or not 0<=warmup<=warmup_steps
            or done>0 and warmup!=warmup_steps or not isinstance(history,list) or len(history)!=done
            or any(not isinstance(row,dict) or row.get('epoch')!=i+1 for i,row in enumerate(history))
            or type(metadata.get('elapsed_seconds')) not in (float,int) or not np.isfinite(metadata['elapsed_seconds'])
            or metadata['elapsed_seconds']<0):
        raise ValueError('Training checkpoint does not describe a complete warmup or epoch boundary')
    eligible=[row for row in history if row.get('checkpoint_eligible') is True]
    from .structural_compatible_objective import checkpoint_key
    methods={row['validation'].get('loss_method') for row in eligible}
    if len(methods)>1:raise ValueError('Training checkpoint mixes incompatible validation objectives')
    selected=min(eligible,key=lambda row:checkpoint_key(row['validation'])) if eligible else None
    expected_key=None if selected is None else list(checkpoint_key(selected['validation']))
    if (metadata.get('selected_epoch')!=(None if selected is None else selected['epoch'])
            or metadata.get('best_key')!=expected_key or bool(arrays['best_parameters'].size)!=bool(selected)):
        raise ValueError('Training checkpoint lost its validation-selected incumbent')
    original_groups=optimizer.state_dict()['param_groups'];groups=metadata.get('optimizer_groups')
    if not isinstance(groups,list) or len(groups)!=1 or len(original_groups)!=1:
        raise ValueError('Only the installed attention AdamW optimizer can be recovered')
    # JSON represents Adam's beta tuple as a list. Compare the fixed installed
    # settings in the same data representation, without relaxing their values.
    initial=json.loads(json.dumps(original_groups[0],allow_nan=False));group=groups[0]
    if (not isinstance(group,dict) or set(group)!=set(initial)
            or any(group[k]!=v for k,v in initial.items() if k!='lr')
            or type(group['lr']) is not float or not 0<group['lr']<=initial['lr']):
        raise ValueError('Training checkpoint changed the declared AdamW optimizer')
    initial_scheduler=scheduler.state_dict();schedule=metadata.get('scheduler')
    dynamic={'last_epoch','_step_count','_last_lr','_is_initial'}
    if (not isinstance(schedule,dict) or set(schedule)!=set(initial_scheduler)
            or any(schedule[k]!=v for k,v in initial_scheduler.items() if k not in dynamic)
            or schedule.get('last_epoch')!=initial_scheduler['last_epoch']+done
            or schedule.get('_step_count')!=initial_scheduler['_step_count']+done
            or schedule.get('_last_lr')!=[group['lr']]):
        raise ValueError('Training checkpoint changed the declared learning-rate schedule')
    # All bytes and fixed code identities pass before any training state is
    # replaced. Random-state decoders are installed numerical APIs, not pickle.
    test_rng=np.random.default_rng();test_rng.bit_generator.state=metadata['numpy_rng']
    with torch.random.fork_rng(devices=[]):torch.set_rng_state(torch.from_numpy(arrays['torch_rng']))
    state={};offset=0
    for index,(_,parameter) in enumerate(named):
        size=parameter.numel()
        if arrays['initialized'][index]:
            state[index]=dict(step=torch.tensor(float(arrays['steps'][index]),dtype=torch.float32),
                exp_avg=torch.from_numpy(arrays['first_moment'][offset:offset+size].copy().reshape(parameter.shape)),
                exp_avg_sq=torch.from_numpy(arrays['second_moment'][offset:offset+size].copy().reshape(parameter.shape)))
        offset+=size
    model.load_state_dict(_state(model,arrays['parameters']),strict=True)
    optimizer.load_state_dict(dict(state=state,param_groups=groups));scheduler.load_state_dict(schedule)
    rng.bit_generator.state=test_rng.bit_generator.state;torch.set_rng_state(torch.from_numpy(arrays['torch_rng']))
    visited[:]=arrays['visited']
    return dict(warmup=warmup,epochs=done,history=history,selected_epoch=metadata['selected_epoch'],
        best_key=None if expected_key is None else tuple(expected_key),
        best=None if selected is None else _state(model,arrays['best_parameters']),elapsed_seconds=metadata['elapsed_seconds'])
