"""One preparation workflow with fixed CPU and CUDA job producers.

Routing is captured in new training handles, including ambiguous submissions.
Older handles retain their original tool and dispatch identity. This coordinator
does not infer a GPU allocation or change the shared scientific-job admission.
"""
from pathlib import Path

from .model_runtime import GalaxyModelRuntime
from .native_operations import SCHEMA,BOUND_SCHEMA
from .preparation_limits import CUDA_TRAINING_TOOL_ID
from .refinement_runtime import RuntimeUnavailable

TRAINING_TOOL_ID=CUDA_TRAINING_TOOL_ID
TRAINING_OPERATION='preparation-structural-training'


class TrainingCapacityPending(RuntimeUnavailable):
    pass


class TrainingContractUnsupported(ValueError):
    pass


class CudaTrainingRuntime(GalaxyModelRuntime):
    tool_id=TRAINING_TOOL_ID
    native_handle_tool_id=TRAINING_TOOL_ID

    def validate_native_request(self,operation,payload):
        from gpx_recipe.preparation.structural_training_blocks import REPRESENTATION
        settings=payload.get('settings') if isinstance(payload,dict) else None
        if (operation!=TRAINING_OPERATION or not isinstance(settings,dict)
                or settings.get('representation')!=REPRESENTATION):
            raise TrainingContractUnsupported('GPU training requires the current three-curve preparation. '
                'This older training plan cannot run on the GPU trainer; create a new preparation. '
                'Its saved models and original jobs remain available.')

    def wait_capacity(self,cancel=None):
        if cancel and cancel.is_set():raise RuntimeUnavailable('Training submission paused before dispatch')
        if not self.available_capacity():
            raise TrainingCapacityPending('Waiting for the current GPU training job to finish; completed results can still be collected.')

    def _submit(self,action,source,request,asset_id,key,*,cancel=None):
        if action!='native' or request.get('operation')!=TRAINING_OPERATION:
            raise ValueError('The GPU preparation runtime accepts only reviewed structural training')
        self.validate_native_request(request['operation'],request.get('payload'))
        return super()._submit(action,source,request,asset_id,key,cancel=cancel)


class PreparationRuntime:
    def __init__(self,root,service,*,max_active_jobs=8,poll_seconds=3,timeout=7200):
        self.model=GalaxyModelRuntime(root,service,max_active_jobs=max_active_jobs,poll_seconds=poll_seconds,timeout=timeout)
        self.training=CudaTrainingRuntime(Path(root)/'cuda-training',service,
            max_active_jobs=1,poll_seconds=poll_seconds,timeout=timeout)

    def __getattr__(self,name):return getattr(self.model,name)

    @property
    def max_active_jobs(self):return self.model.max_active_jobs

    @max_active_jobs.setter
    def max_active_jobs(self,value):self.model.max_active_jobs=value

    @property
    def preparation_provider(self):return self.model.preparation_provider

    @preparation_provider.setter
    def preparation_provider(self,value):
        self.model.preparation_provider=value;self.training.preparation_provider=value

    def native_runtime_for_operation(self,operation):
        return self.training if operation==TRAINING_OPERATION else self.model

    def preparation_operation_ready(self,operation,jobs,library):
        if operation!=TRAINING_OPERATION:return True
        for job in jobs:
            if job['phase']=='collected':continue
            handle=job.get('handle')
            if handle is not None:
                if self.native_runtime_for_handle(handle) is self.training:return False
            elif any(library.row(m['entry'])['model']['kind']=='structural-training' for m in job['members']):
                return False
        return True

    def native_runtime_for_handle(self,handle):
        if not isinstance(handle,dict):return self.model
        if handle.get('schema')==BOUND_SCHEMA:
            if (handle.get('tool_id')!=TRAINING_TOOL_ID
                    or not isinstance(handle.get('request'),dict)
                    or handle['request'].get('operation')!=TRAINING_OPERATION):
                raise ValueError('Saved preparation names an unknown training producer')
            return self.training
        # The v1 schema predates CUDA routing. Never relabel it from its
        # operation name, including an unacknowledged CPU submission.
        if handle.get('schema')==SCHEMA:return self.model
        raise ValueError('Unknown saved preparation handle schema')

    def submit_native(self,operation,*args,**kwargs):
        return self.native_runtime_for_operation(operation).submit_native(operation,*args,**kwargs)

    def native(self,operation,*args,**kwargs):
        return self.native_runtime_for_operation(operation).native(operation,*args,**kwargs)

    def poll_native(self,handle):return self.native_runtime_for_handle(handle).poll_native(handle)
    def collect_native(self,handle,*args,**kwargs):return self.native_runtime_for_handle(handle).collect_native(handle,*args,**kwargs)
    def restore_native_inputs(self,handle,*args,**kwargs):return self.native_runtime_for_handle(handle).restore_native_inputs(handle,*args,**kwargs)
    def cancel_native(self,handle):return self.native_runtime_for_handle(handle).cancel_native(handle)

    def stop_preparations(self):
        self.model.stop_preparations();self.training.stop_preparations()

    def allow_preparations(self,root):
        self.model.allow_preparations(root);self.training.allow_preparations(root)
