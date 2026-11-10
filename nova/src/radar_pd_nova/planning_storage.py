"""Owned, portable planner snapshots with authenticated Galaxy History recovery."""
import asyncio
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import uuid

from gpx_recipe.execution import write_json
from . import planning_bundle as bundle


FILE_FIELDS = ('plan_cif', 'plan_mcif', 'plan_comparison_cif', 'plan_custom_profile',
               'plan_phase_1_cif', 'plan_phase_2_cif', 'plan_phase_3_cif')
EXTRA_FIELDS = ('plan_name', 'plan_step_ui', 'plan_transition', 'plan_cycle', 'plan_k',
    'plan_sites', 'plan_site_options', 'plan_site_rows', 'plan_form_factor_options',
    'plan_form_factors', 'plan_site_policy', 'plan_candidate', 'plan_candidates',
    'plan_parameters', 'plan_parameter_rows', 'plan_structural_candidate', 'plan_structural_candidates',
    'plan_structural_parameters', 'plan_structural_parameter_rows', 'plan_structural_cell', 'plan_cell_values',
    'plan_coordinate_note', 'plan_magnetic_message', 'plan_structural_message', 'plan_parent_note',
    'plan_promoted_parent', 'plan_can_promote_parent', 'plan_current', 'plan_result_ready',
    'plan_look', 'plan_look_options', 'plan_pattern_view', 'plan_qualification', 'plan_summary',
    'plan_warnings', 'plan_isotope_options', 'plan_plot_mode', 'plan_error_model', 'plan_error_note',
    'plan_moment_sites', 'plan_displacement_sites', 'plan_structural_note', 'plan_child_cell_note')
BACKEND_FIELDS = ('_planning_output', '_planning_receipt_sha256', '_planning_parent_output',
    '_planning_parent_receipt_sha256', '_planning_parent_state', '_planning_catalog',
    '_planning_structural_catalog', '_planning_prepared', '_planning_prepared_kind',
    '_planning_magnetic_request', '_planning_input_errors')


def implementation_signature():
    from gpx_recipe.symmetry.compiler import package_signature
    return package_signature()


def _planning_history_label(row, identifier):
    """Recover only our bounded bundle label, not another tool's filename policy."""
    name=row.get('name');display=row.get('display_name')
    fallback=identifier
    for value in (display,name):
        if isinstance(value,str) and (label:=' '.join(value[:512].split())[:256]):
            fallback=label;break
    match=(re.fullmatch(r'RADAR-PD Planner bundle ([^\x00-\x1f\x7f]{1,100}) [0-9a-f]{64} \| plan\.zip',name)
           if isinstance(name,str) and len(name)<=256 else None)
    if not match or not match[1].strip():return fallback
    updated=row.get('update_time')
    timestamp=(updated[:16].replace('T',' ') if isinstance(updated,str) and len(updated)<=64
               and re.fullmatch(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}',updated[:16]) else '')
    suffix=' · '.join(value for value in (timestamp,identifier[:8]) if value)
    return match[1].strip()+' · '+suffix


def _planning_restore_settings(settings, declared_types):
    """Normalize numeric text controls against their declared, stable schema."""
    positive={'plan_mass','plan_radius','plan_packing','plan_beam_width','plan_beam_height',
              'plan_current_ua','plan_power','plan_energy','plan_yield_ratio',
              'plan_d_min','plan_d_max','plan_step'}
    bounds={'plan_profile_bank':(1,100),'plan_seed':(0,2**32-1),
            'plan_step_ui':(0,3),'plan_look':(0,23),'plan_packing':(0,1),
            'plan_moment_multiplier':(0,10)}
    optional_text=set()
    for index in range(1,4):
        bounds[f'plan_phase_{index}_percent']=(0,100)
        size=f'plan_phase_{index}_size';strain=f'plan_phase_{index}_microstrain'
        optional_text.update((size,strain));positive.add(size);bounds[strain]=(0,None)
    normalized=copy.deepcopy(settings)
    for key,value in settings.items():
        kind=declared_types.get(key)
        try:
            numeric=kind in (int,float) or key=='plan_yield_ratio' or key in optional_text
            if not numeric:
                if kind is None or type(value) is not kind:raise ValueError('Unexpected setting type')
                continue
            if key=='plan_yield_ratio' and (value is None or type(value) is str and not value.strip()):
                normalized[key]=None;continue
            if key in optional_text and type(value) is str and not value.strip():
                normalized[key]='';continue
            if type(value) not in (int,float,str) or type(value) is str and len(value)>128:
                raise ValueError('A bounded numeric value is required')
            number=float(value)
            if not math.isfinite(number) or key in positive and number<=0:
                raise ValueError('A finite numeric value in range is required')
            lower,upper=bounds.get(key,(None,None))
            if lower is not None and number<lower or upper is not None and number>upper:
                raise ValueError('Numeric value is out of range')
            if kind is int:
                if not number.is_integer():raise ValueError('An integer is required')
                normalized[key]=int(number)
            elif key in optional_text:normalized[key]=str(value).strip()
            else:normalized[key]=number
        except (ValueError,TypeError,OverflowError) as exc:
            raise ValueError('Invalid saved planner setting: '+key) from exc
    return normalized


class PlanningStorageMixin:
    def _planning_begin_calculation(self):
        event=threading.Event()
        self._planning_cancel=event
        s=self.server.state
        s.plan_can_cancel=True
        s.plan_save_message=''
        s.flush()
        return event

    def _planning_schedule(self, coroutine, name):
        # Progress callbacks run in worker threads; only this loop owns Trame
        # state publication, including during saved-operation recovery.
        try:self._planning_progress_loop=asyncio.get_running_loop()
        except RuntimeError:pass
        return self._refinement_schedule(coroutine,name)

    def _planning_on_owner(self, action):
        """Commit state and run its callbacks on the captured interactive loop."""
        loop=getattr(self,'_planning_progress_loop',None)
        try:current=asyncio.get_running_loop()
        except RuntimeError:current=None
        if loop is None or current is loop:return action()
        if not loop.is_running():raise RuntimeError('The planner viewer closed before its state could be updated')
        result=Future()
        def deliver():
            if not result.set_running_or_notify_cancel():return
            try:result.set_result(action())
            except BaseException as exc:result.set_exception(exc)
        loop.call_soon_threadsafe(deliver)
        while True:
            try:return result.result(timeout=.5)
            except FutureTimeoutError:
                if result.done():raise
                if not loop.is_running():
                    result.cancel()
                    raise RuntimeError('The planner viewer closed before its state could be updated')

    def _planning_root(self):
        if not hasattr(self, '_planning_owned_root'):
            root=getattr(getattr(self, 'service', None), 'output_root', None)
            configured=os.environ.get('RADAR_PD_PLANNING_ROOT')
            base=(Path(configured) if configured else Path(root)/'powgen-planning' if root else
                  Path(tempfile.mkdtemp(prefix='radar-planning-'))).resolve()
            history=getattr(getattr(self,'service',None),'history_id',None)
            self._planning_owned_root=base/hashlib.sha256(str(history).encode()).hexdigest()[:20] if history else base
            self._planning_owned_root.mkdir(parents=True, exist_ok=True)
        return self._planning_owned_root

    def _planning_input(self, value):
        """Check ownership at the job/export boundary; never trust browser paths."""
        from .uploads import read_browser_upload
        from gpx_recipe.planning import __file__ as package_file
        if Path(value).is_symlink():raise ValueError('Use the uploaded input, not a symbolic link.')
        path=Path(value).resolve(strict=True)
        example=Path(package_file).parent/'data/examples/silicon.cif'
        if path==example.resolve():return path
        digest=getattr(self, '_planning_owned_inputs', {}).get(str(path))
        if digest and not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest()==digest:
            return path
        read_browser_upload(str(value), max_bytes=20*1024*1024, suffix=path.suffix.lower())
        return path

    def _planning_saved_list(self):
        rows=[]
        for path in (self._planning_root()/'saved').glob('*/plan.json'):
            try:
                if path.is_symlink() or path.stat().st_size>bundle.MAX_MANIFEST_BYTES:continue
                data=bundle.read_json(path.read_bytes())
                if data.get('schema')!=bundle.SCHEMA:continue
                rows.append({'title':data['name']+' · '+data['saved_at'][:16].replace('T',' ')+' UTC',
                             'value':path.parent.name})
            except (ValueError, OSError, KeyError):continue
        rows.extend(getattr(self,'_planning_history_options',[]))
        self.server.state.plan_saved_options=sorted(rows,key=lambda row:row['title'],reverse=True)

    def _planning_history_service(self):
        service=getattr(self,'service',None)
        return service if service and all(getattr(service,k,None) for k in ('galaxy_url','galaxy_key','history_id')) else None

    def _planning_storage_action(self, action, message, *, completed=None, failed=None):
        """Do storage work in a thread; deliver UI effects on the owner loop."""
        s=self.server.state
        if s.plan_busy:return
        async def run():
            try:
                value=await asyncio.to_thread(action)
                if completed is not None:completed(value)
            except Exception as exc:
                if failed is not None:failed()
                s.plan_error=str(exc)
            finally:s.plan_busy=False;s.plan_can_cancel=False;s.flush()
        if hasattr(self,'_refinement_schedule'):
            s.plan_busy=True;s.plan_can_cancel=False;s.plan_error='';s.plan_save_message=message;s.flush()
            self._planning_schedule(run(),'radar-planner-storage')
        else:
            try:
                value=action()
                if completed is not None:completed(value)
            except Exception as exc:
                if failed is not None:failed()
                s.plan_error=str(exc)
            s.flush()

    def _planning_build_bundle(self):
        from .planning_controller import FIELDS
        s=self.server.state
        if not s.plan_cif:raise ValueError('Choose a nuclear CIF before saving the plan.')
        settings={key:copy.deepcopy(getattr(s,key)) for key in (*FIELDS,*EXTRA_FIELDS)}
        builder=bundle.Builder()
        for key in FILE_FIELDS:
            if settings.get(key):
                source=self._planning_input(settings[key])
                settings[key]=builder.add_file(source,'inputs/'+key+'/'+source.name)
        backend={key:copy.deepcopy(getattr(self,key,None)) for key in BACKEND_FIELDS}
        backend['_planning_input_errors']={}
        for key in ('_planning_output','_planning_parent_output'):
            if backend[key]:
                root=Path(backend[key]).resolve(strict=True)
                if not root.is_relative_to(self._planning_root()):raise ValueError('Saved result is outside this workspace')
                receipt=self._planning_receipt(root,backend['_planning_receipt_sha256' if key=='_planning_output' else '_planning_parent_receipt_sha256'])
                builder.add_tree(root,'results/'+key.removeprefix('_planning_'),receipt=receipt)
        for key in ('_planning_catalog','_planning_structural_catalog'):
            if backend[key]:
                root=Path(backend[key][-1]).resolve(strict=True)
                if not root.is_relative_to(self._planning_root()):raise ValueError('Saved catalog is outside this workspace')
                builder.add_tree(root,'catalogs/'+key.removeprefix('_planning_'),
                    relative_files=['catalog.json'] if key=='_planning_catalog' else None)
        prepared=backend.get('_planning_prepared')
        if isinstance(prepared,dict) and prepared.get('gpx'):
            path=self._planning_native_input(prepared['gpx'])
            builder.add_file(path,'prepared/model.gpx')
        return builder.finish(name=str(s.plan_name).strip() or 'Experiment plan',
            saved_at=datetime.now(timezone.utc).isoformat(),implementation=implementation_signature(),
            settings=settings,backend=backend,native_artifacts=getattr(self,'_planning_native_artifacts',{}),
            history_id=getattr(self._planning_history_service(),'history_id',None))

    def _planning_write_snapshot(self, *, history=True, label='Planner bundle'):
        document,files=self._planning_build_bundle()
        uid=uuid.uuid4().hex;root=self._planning_root()/'saved'/uid
        bundle.write_directory(root,document,files)
        raw=bundle.archive(document,files);digest=hashlib.sha256(raw).hexdigest()
        service=self._planning_history_service() if history else None
        dataset=None
        if service:
            path=root/'plan.zip';path.write_bytes(raw)
            dataset=service.upload_document(path,label=label+' '+document['name']+' '+digest,wait_ready=True)
            write_json(root/'history.json',dict(history_id=service.history_id,dataset_id=dataset,sha256=digest))
        return uid,dataset,digest

    def save_planning_plan(self, **_):
        def save():
            uid,dataset,_=self._planning_write_snapshot()
            s=self.server.state
            if dataset:
                self._planning_refresh_history()
                s.plan_saved_id='history:'+dataset
            else:s.plan_saved_id=uid
            self._planning_saved_list()
            s.plan_saved_id='history:'+dataset if dataset else uid
            s.plan_save_message='Saved to Galaxy History.' if dataset else 'Saved locally. Export the portable plan to keep it outside this session.'
            s.plan_error=''
        self._planning_storage_action(save,'Saving the plan and verified results…')

    def _planning_refresh_history(self):
        service=self._planning_history_service()
        if not service:return
        rows=[];ids=set()
        for offset in range(0,1000,200):
            page=service.search_history_datasets(query='Planner bundle ',limit=200,offset=offset,include_generated=True)
            for row in page:
                identifier=str(row['id'])
                if not re.fullmatch('[A-Za-z0-9_-]{1,128}',identifier):continue
                rows.append(dict(title='Galaxy · '+_planning_history_label(row,identifier),value='history:'+identifier));ids.add(identifier)
            if len(page)<200:break
        self._planning_history_ids=ids;self._planning_history_options=rows;self._planning_saved_list()

    def refresh_planning_plans(self, **_):
        self._planning_storage_action(self._planning_refresh_history,'Reading saved plans from Galaxy History…')

    def _planning_read_history(self, identifier, *, expected=None):
        service=self._planning_history_service()
        if not service or not re.fullmatch('[A-Za-z0-9_-]{1,128}',identifier):raise ValueError('Select a saved plan in the current Galaxy History')
        metadata=service._dataset_metadata(identifier)
        if metadata.get('deleted') or metadata.get('purged') or metadata.get('state')!='ok':raise ValueError('Saved Galaxy plan is unavailable')
        if metadata.get('history_id') not in (None,service.history_id):raise ValueError('The saved plan belongs to another History')
        path=self._planning_root()/('download-'+uuid.uuid4().hex+'.zip')
        try:
            service._download_dataset(identifier,path,maximum_bytes=bundle.MAX_BYTES)
            raw=path.read_bytes()
            if expected and hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('The saved Galaxy bundle changed')
            document,files=bundle.read_archive(raw)
            if document['history_id']!=service.history_id:raise ValueError('The saved plan belongs to another History')
            return document,files
        finally:path.unlink(missing_ok=True)

    def restore_planning_plan(self, **_):
        def restore():
            try:
                uid=str(self.server.state.plan_saved_id or '')
                trusted=False
                if uid.startswith('history:'):
                    identifier=uid.removeprefix('history:')
                    if identifier not in getattr(self,'_planning_history_ids',set()):raise ValueError('Select a saved plan from the current History list')
                    document,files=self._planning_read_history(identifier);trusted=True
                else:
                    if not re.fullmatch('[0-9a-f]{32}',uid):raise ValueError('Select a saved plan from this workspace.')
                    root=self._planning_root()/'saved'/uid
                    if root.is_symlink() or not root.resolve().is_relative_to(self._planning_root()):
                        raise ValueError('The saved plan is outside this workspace')
                    document,files=bundle.read_directory(root)
                self._planning_apply_bundle(document,files,trusted=trusted)
            except Exception:
                self._planning_on_owner(self._planning_edited)
                raise
        self._planning_storage_action(restore,'Opening and validating the saved plan…')

    def _planning_apply_bundle(self, document, files, *, trusted=False):
        from .planning_controller import FIELDS
        s=self.server.state
        bundle.validate(document,files)
        allowed=set(FIELDS)|set(EXTRA_FIELDS)
        if set(document['settings'])-allowed or set(document['backend'])-set(BACKEND_FIELDS):raise ValueError('Unknown saved planner fields')
        settings=_planning_restore_settings(document['settings'],self._planning_setting_types)
        for key in FILE_FIELDS:
            value=document['settings'].get(key)
            if value and (not isinstance(value,str) or value not in files or not value.startswith('inputs/'+key+'/')):
                raise ValueError('A saved input is not part of this plan')
        root=self._planning_root()/'restored'/uuid.uuid4().hex
        bundle.write_directory(root,document,files)
        owned={}
        for key in FILE_FIELDS:
            if settings.get(key):
                path=bundle.confined(root,settings[key]);settings[key]=str(path)
                owned[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        backend=bundle.decode(document['backend'],root)
        native={str(bundle.confined(root,row['file'])):{k:v for k,v in row.items() if k!='file'}
                for row in document['native_artifacts']}
        verified={};native_message=''
        if trusted and native:
            try:verified=self._planning_runtime().verify_artifacts(native,root/'verification')
            except Exception as exc:native_message=' Native model reuse needs regeneration: '+str(exc)
        original_scenario=None;original_root=None
        if verified and backend.get('_planning_output'):
            try:
                original_scenario,original_root=self._planning_runtime().verify_result(backend['_planning_output'],verified)
            except Exception as exc:
                verified={}
                native_message+=' Preview provenance is unverified: '+str(exc)
        same_engine=document['implementation']==implementation_signature()
        if not same_engine or not verified:
            for key in ('_planning_parent_output','_planning_parent_receipt_sha256','_planning_parent_state',
                        '_planning_catalog','_planning_structural_catalog','_planning_prepared','_planning_prepared_kind','_planning_magnetic_request'):
                backend[key]=None
            settings['plan_can_promote_parent']=False;settings['plan_promoted_parent']=False
        if not same_engine:
            settings['plan_current']=False;settings['plan_result_ready']=False
            native_message=' The calculation engine changed. Prepare the transition and recalculate.'
        if not backend.get('_planning_output'):
            settings['plan_current']=False;settings['plan_result_ready']=False
        for key in ('_planning_catalog','_planning_structural_catalog'):
            if backend.get(key):backend[key]=(*backend[key][:-1],Path(backend[key][-1]))
        def apply():
            self._planning_generation+=1
            self._planning_restoring=True
            try:
                for key in (*FIELDS,*EXTRA_FIELDS):
                    if key in settings:setattr(s,key,settings[key])
                self._planning_owned_inputs.update(owned)
                self._planning_native_artifacts.update(verified)
                for key in BACKEND_FIELDS:
                    value=backend.get(key)
                    if key in {'_planning_output','_planning_parent_output'} and value:value=Path(value)
                    setattr(self,key,value)
                self._planning_profile_context=(s.plan_configuration,s.plan_custom_profile)
                self._planning_isotope_source=self._planning_structure_sources()
                self._planning_profile_options()
                s.plan_error='';s.plan_save_message='Opened '+document['name']+'.'+native_message
                s.plan_message='Saved inputs and results opened.'+native_message
                if s.plan_current and original_scenario is None:
                    s.plan_qualification='Imported saved preview · scientific provenance unverified'
                    s.plan_warnings=list(s.plan_warnings)+['Recalculate this saved preview to verify it against the current inputs and worker.']
                elif s.plan_current:
                    if not self._planning_matches_original(original_scenario,original_root):
                        s.plan_current=False;s.plan_result_ready=False
                        s.plan_message='Saved settings differ from the original calculation. Recalculate this plan before using its prediction.'
                if not s.plan_current:s.plan_figure={'data':[],'layout':{}}
                if same_engine:
                    self._planning_plot()
            finally:
                try:s.flush()
                finally:self._planning_restoring=False
        self._planning_on_owner(apply)

    def _planning_matches_original(self, original, root):
        """Do not present an authentic result as the prediction of edited inputs."""
        from gpx_recipe.planning.managed import file_references
        def identity(value, relative_root=None):
            request=copy.deepcopy(value)
            request.pop('prepared_magnetic',None);request.pop('prepared_structural',None)
            for row,key,_ in file_references('planning-simulate',request):
                path=Path(row[key])
                if relative_root is not None and not path.is_absolute():path=bundle.confined(relative_root,str(path))
                row[key]=hashlib.sha256(path.read_bytes()).hexdigest()
            return request
        try:
            if identity(original,root)!=identity(self._planning_request()):return False
            s=self.server.state
            if original.get('prepared_magnetic'):
                state=original['prepared_magnetic']
                return (s.plan_transition=='magnetic' and s.plan_candidate==state['candidate']['id']
                    and json.loads(s.plan_parameters)=={r['name']:r['value'] for r in state['parameters']})
            if original.get('prepared_structural') and s.plan_transition=='structural':
                state=original['prepared_structural']
                # The final native cell is resolved even when the UI inherits
                # it. Bind both adding and removing a cell override; old
                # receipts without that intent cannot certify a current plan.
                requested=json.loads(s.plan_structural_cell) if str(s.plan_structural_cell).strip() else None
                return ('requested_cell' in state and requested==state['requested_cell']
                    and s.plan_structural_candidate==state['candidate']['id']
                    and json.loads(s.plan_structural_parameters)==state['modes'])
            return s.plan_transition not in {'magnetic','structural'} or bool(original.get('prepared_structural'))
        except (KeyError,ValueError,TypeError,OSError):return False

    def export_planning_plan(self, **_):
        def export():
            document,files=self._planning_build_bundle()
            return bundle.archive(document,files)
        def deliver(raw):
            self.download_file('powgen-experiment-plan.zip','application/zip',raw)
            self.server.state.plan_save_message='Portable plan download requested.'
        self._planning_storage_action(export,'Preparing the portable plan and result bundle…',
                                      completed=deliver,failed=self._planning_edited)

    def import_planning_plan(self, **_):
        def imported():
            from .uploads import read_browser_upload
            raw=read_browser_upload(self.server.state.plan_bundle_upload,max_bytes=bundle.MAX_BYTES,suffix='.zip')
            document,files=bundle.read_archive(raw)
            self._planning_apply_bundle(document,files,trusted=False)
            self.server.state.plan_save_message+=' Imported GPX files are retained only as data; prepare transitions again before simulation.'
        self._planning_storage_action(imported,'Validating the portable plan…')

    def _planning_checkpoint_operation(self, handle):
        service=self._planning_history_service()
        if not service:return
        _,dataset,digest=self._planning_write_snapshot(label='Planner checkpoint')
        checkpoint=dict(schema='radar-planner-operation/v1',history_id=service.history_id,
            bundle_id=dataset,bundle_sha256=digest,handle=handle)
        service.upload_json_document(checkpoint,name='planner-operation.json',
            label='Planner operation '+handle['key'],wait_ready=True)

    def _planning_refresh_jobs(self):
        service=self._planning_history_service()
        if not service:return
        found={}
        for row in service.search_history_datasets(query='Planner operation ',limit=200,include_generated=True):
            identifier=str(row['id'])
            if not re.fullmatch('[A-Za-z0-9_-]{1,128}',identifier):continue
            target=self._planning_root()/('job-'+uuid.uuid4().hex+'.json')
            try:
                service._download_dataset(identifier,target,maximum_bytes=8*1024*1024)
                document=bundle.read_json(target.read_bytes())
                if (document.get('schema')!='radar-planner-operation/v1'
                        or document.get('history_id')!=service.history_id):continue
                from .native_operations import validate_handle
                handle=validate_handle(self._planning_runtime().native,document['handle'])
                if not handle['request']['operation'].startswith('planning-'):continue
                found.setdefault(handle['key'],dict(identifier=identifier,document=document))
            finally:target.unlink(missing_ok=True)
        self._planning_recoverable=found
        self.server.state.plan_recoverable_jobs=[dict(value=key,
            title=row['document']['handle']['request']['operation'].removeprefix('planning-')+' · '+key)
            for key,row in found.items()]

    def refresh_planning_jobs(self, **_):
        self._planning_storage_action(self._planning_refresh_jobs,'Finding recoverable planner jobs in Galaxy History…')

    def recover_planning_job(self, **_):
        async def recover():
            s=self.server.state
            if s.plan_busy:return
            s.plan_busy=True;s.plan_error=''
            event=self._planning_begin_calculation()
            try:
                row=getattr(self,'_planning_recoverable',{}).get(s.plan_recover_job_id)
                if row is None:raise ValueError('Select a planner operation from the current History list')
                checkpoint=row['document'];handle=checkpoint['handle']
                document,files=await asyncio.to_thread(self._planning_read_history,checkpoint['bundle_id'],expected=checkpoint['bundle_sha256'])
                await asyncio.to_thread(self._planning_apply_bundle,document,files,trusted=True)
                generation=self._planning_generation
                root=self._planning_root()/('recovered-job-'+uuid.uuid4().hex)
                runtime=self._planning_runtime()
                result=await asyncio.to_thread(runtime.recover,handle,root,event)
                if event.is_set() or generation!=self._planning_generation:return
                self._planning_native_artifacts.update(runtime.artifacts)
                operation=handle['request']['operation']
                if operation in {'planning-parent','planning-simulate'}:
                    output=root/'simulation'
                    digest=hashlib.sha256((output/'result.json').read_bytes()).hexdigest()
                    if operation=='planning-parent':
                        self._planning_parent_output=output;self._planning_parent_receipt_sha256=digest
                    else:
                        self._planning_output=output;self._planning_receipt_sha256=digest
                        s.plan_current=True;s.plan_result_ready=True
                        if not result.get('magnetic_state') and not result.get('structural_state'):
                            self._planning_parent_output=output;self._planning_parent_receipt_sha256=digest
                        s.plan_qualification={'profile_only':'Peak-shape preview · counting-time response unavailable',
                            'historical_estimate':'Historical response estimate · experimental accuracy unqualified',
                            'current_cycle':'2026B measured response · conditional prediction, validation pending'}[result['mode']]
                        s.plan_warnings=result['warnings'];s.plan_can_promote_parent=bool(result.get('structural_state') and not result.get('magnetic_state'))
                        s.plan_look_options=[{'title':str(look['cumulative_charge_uah'])+' μAh','value':i}
                            for i,look in enumerate(result['looks'])] or [{'title':'Expected profile','value':0}]
                        s.plan_look=0;self._planning_plot()
                    self._planning_isotope_rows(result)
                s.plan_message='Recovered the existing Galaxy '+operation.removeprefix('planning-')+' result. No new calculation was submitted.'
            except Exception as exc:
                if not event.is_set():s.plan_error=str(exc)
            finally:self._planning_finish_job(event)
        self._planning_schedule(recover(),'radar-planner-recover')
