"""One preparation workflow with fixed CPU and CUDA job producers.

Routing is captured in new training handles, including ambiguous submissions.
Older handles retain their original tool and dispatch identity. This coordinator
does not infer a GPU allocation. New runs use the existing CPU producer unless
trusted deployment policy explicitly selects CUDA. Both run the same trainer.
"""
import copy
import json
from pathlib import Path

from .model_runtime import GalaxyModelRuntime
from .native_operations import SCHEMA,BOUND_SCHEMA
from .preparation_limits import (
    CPU_TRAINING_TOOL_ID, CUDA_TRAINING_TOOL_ID, TRAINING_JOBS,
    TRAINING_OPERATION, training_device,
)
from .refinement_runtime import RuntimeUnavailable

# Kept as the default producer name for clients that display the tool identity.
TRAINING_TOOL_ID=CPU_TRAINING_TOOL_ID


class TrainingCapacityPending(RuntimeUnavailable):
    pass


class TrainingContractUnsupported(ValueError):
    pass


class CpuTrainingRuntime(GalaxyModelRuntime):
    tool_id=CPU_TRAINING_TOOL_ID
    native_handle_tool_id=CPU_TRAINING_TOOL_ID
    device='cpu'

    def validate_native_request(self,operation,payload):
        if operation!=TRAINING_OPERATION:
            raise ValueError('The preparation training runtime accepts only structural training')

    def available_capacity(self):
        from .native_job_admission import HistoryJobAdmission
        return HistoryJobAdmission(self.service).available_capacity(
            self.tool_id,family_limit=TRAINING_JOBS,operation=TRAINING_OPERATION)

    def wait_capacity(self,cancel=None):
        if cancel and cancel.is_set():raise RuntimeUnavailable('Training submission paused before dispatch')
        if not self.available_capacity():
            raise TrainingCapacityPending('Waiting for the current training job or shared scientific-job capacity; completed results can still be collected.')

    def _submit(self,action,source,request,asset_id,key,*,cancel=None):
        if action!='native' or request.get('operation')!=TRAINING_OPERATION:
            raise ValueError('The preparation training runtime accepts only reviewed structural training')
        self.validate_native_request(request['operation'],request.get('payload'))
        return super()._submit(action,source,request,asset_id,key,cancel=cancel)


class CudaTrainingRuntime(CpuTrainingRuntime):
    tool_id=CUDA_TRAINING_TOOL_ID
    native_handle_tool_id=CUDA_TRAINING_TOOL_ID
    device='cuda'

    def validate_native_request(self,operation,payload):
        from gpx_recipe.preparation.structural_training_blocks import REPRESENTATION
        settings=payload.get('settings') if isinstance(payload,dict) else None
        if (operation!=TRAINING_OPERATION or not isinstance(settings,dict)
                or settings.get('representation')!=REPRESENTATION):
            raise TrainingContractUnsupported('GPU training requires the current three-curve preparation. '
                'This older training plan cannot run on the GPU trainer; create a new preparation. '
                'Its saved models and original jobs remain available.')

class PreparationRuntime:
    def __init__(self,root,service,*,max_active_jobs=8,poll_seconds=3,timeout=7200):
        device=training_device()
        self.model=GalaxyModelRuntime(root,service,max_active_jobs=max_active_jobs,poll_seconds=poll_seconds,timeout=timeout)
        self.cpu_training=CpuTrainingRuntime(Path(root)/'cpu-training',service,
            max_active_jobs=TRAINING_JOBS,poll_seconds=poll_seconds,timeout=timeout)
        self.cuda_training=CudaTrainingRuntime(Path(root)/'cuda-training',service,
            max_active_jobs=TRAINING_JOBS,poll_seconds=poll_seconds,timeout=timeout)
        self.training=self.cuda_training if device=='cuda' else self.cpu_training

    def __getattr__(self,name):return getattr(self.model,name)

    @property
    def max_active_jobs(self):return self.model.max_active_jobs

    @max_active_jobs.setter
    def max_active_jobs(self,value):self.model.max_active_jobs=value

    @property
    def preparation_provider(self):return self.model.preparation_provider

    @preparation_provider.setter
    def preparation_provider(self,value):
        for runtime in (self.model,self.cpu_training,self.cuda_training):runtime.preparation_provider=value

    def native_runtime_for_operation(self,operation,*,payload=None):
        if operation!=TRAINING_OPERATION:return self.model
        if payload is not None:
            from gpx_recipe.preparation.structural_jobs import read_plan,resume_directory
            from gpx_recipe.preparation.structural_training_transfer import read,CUDA_SCHEMA
            path=Path(payload['plan']);resume=resume_directory(path,read_plan(path))
            if resume is not None:
                receipt,_=read(resume)
                return self.cuda_training if receipt['schema']==CUDA_SCHEMA else self.cpu_training
        return self.training

    def preparation_operation_ready(self,operation,jobs,library):
        if operation!=TRAINING_OPERATION:return True
        for job in jobs:
            if job['phase']=='collected':continue
            handle=job.get('handle')
            if handle is not None:
                # A recovered v1 CPU trainer also owns this training slot.
                if handle.get('request',{}).get('operation')==TRAINING_OPERATION:
                    self.native_runtime_for_handle(handle)
                    return False
            elif any(library.row(m['entry'])['model']['kind']=='structural-training' for m in job['members']):
                return False
        return True

    def native_runtime_for_handle(self,handle):
        if not isinstance(handle,dict):return self.model
        if handle.get('schema')==BOUND_SCHEMA:
            if (handle.get('tool_id') not in {CPU_TRAINING_TOOL_ID,CUDA_TRAINING_TOOL_ID}
                    or not isinstance(handle.get('request'),dict)
                    or handle['request'].get('operation')!=TRAINING_OPERATION):
                raise ValueError('Saved preparation names an unknown training producer')
            return self.cpu_training if handle['tool_id']==CPU_TRAINING_TOOL_ID else self.cuda_training
        # The v1 schema predates CUDA routing. Never relabel it from its
        # operation name, including an unacknowledged CPU submission.
        if handle.get('schema')==SCHEMA:return self.model
        raise ValueError('Unknown saved preparation handle schema')

    def retain_native_producer(self,handle,root):
        """A policy change must not duplicate an owned or uncertain submission.

        Derive the three historical keys from the already-hashed request. This
        never searches unrelated jobs and does not re-read large training arrays.
        """
        if handle['request']['operation']!=TRAINING_OPERATION:return handle
        from gpx_recipe.symmetry.contracts import digest
        from .native_operations import handle_path,validate_handle
        request=copy.deepcopy(handle['request']);request.pop('dispatch_key')
        found=[]
        for tool in (None,CPU_TRAINING_TOOL_ID,CUDA_TRAINING_TOOL_ID):
            candidate=copy.deepcopy(handle)
            dispatch=[handle['history_id'],handle['source']['id'],request]
            if tool is not None:dispatch.append(tool)
            candidate['key']=digest(dispatch)[:24]
            candidate['request']['dispatch_key']=candidate['key']
            candidate['schema']=BOUND_SCHEMA if tool is not None else SCHEMA
            candidate.pop('tool_id',None)
            context=dict(history_id=candidate['history_id'],source=candidate['source'],request=candidate['request'])
            if tool is not None:candidate['tool_id']=tool;context['tool_id']=tool
            candidate['identity']=digest(context)
            path=handle_path(root,candidate['key'])
            if not path.exists():continue
            if path.is_symlink() or path.stat().st_size>8*1024**2:
                raise ValueError('Saved training intent is not bounded owned metadata')
            saved=validate_handle(self,json.loads(path.read_text()))
            if saved['identity']!=candidate['identity']:
                raise ValueError('Saved training intent changed its captured input scope')
            found.append(candidate)
        if len(found)>1:
            raise RuntimeUnavailable('Several producers own this training request; recover the saved jobs before starting another')
        return found[0] if found else handle

    def submit_native(self,operation,payload,root,*,source,timeout=120,**kwargs):
        # Capture once through this coordinator, then submit through the fixed
        # producer selected by that handle. The event belongs to all routes.
        return GalaxyModelRuntime.submit_native(self,operation,payload,root,
            source=source,timeout=timeout,**kwargs)

    def native(self,operation,payload,root,*,source,timeout=120,**kwargs):
        return GalaxyModelRuntime.native(self,operation,payload,root,
            source=source,timeout=timeout,**kwargs)

    def wait_job(self,job,*args,**kwargs):
        runtime=self.cuda_training if job.tool_id==CUDA_TRAINING_TOOL_ID else self.model
        return runtime.wait_job(job,*args,**kwargs)

    def poll_native(self,handle):return self.native_runtime_for_handle(handle).poll_native(handle)
    def collect_native(self,handle,*args,**kwargs):return self.native_runtime_for_handle(handle).collect_native(handle,*args,**kwargs)
    def restore_native_inputs(self,handle,*args,**kwargs):return self.native_runtime_for_handle(handle).restore_native_inputs(handle,*args,**kwargs)
    def cancel_native(self,handle):return self.native_runtime_for_handle(handle).cancel_native(handle)

    def stop_preparations(self):
        for runtime in (self.model,self.cpu_training,self.cuda_training):runtime.stop_preparations()

    def allow_preparations(self,root):
        for runtime in (self.model,self.cpu_training,self.cuda_training):runtime.allow_preparations(root)
