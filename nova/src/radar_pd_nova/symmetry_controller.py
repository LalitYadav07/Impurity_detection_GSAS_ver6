"""Model exploration inside Refinement, sharing published RADAR-PD sources."""
from __future__ import annotations
import asyncio
import copy
from collections import Counter,OrderedDict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from fractions import Fraction

from gpx_recipe.jobs import scientific_environment
from gpx_recipe.process import run_scientific
from gpx_recipe.symmetry.campaign import Campaign,verify_receipt
from gpx_recipe.symmetry.compiler import confined
from gpx_recipe.symmetry.compiler import file_sha,package_signature,verify_inputs
from gpx_recipe.symmetry.contracts import digest,validate_protocol
from gpx_recipe.symmetry.discovery import discover_magnetic
from gpx_recipe.symmetry.local_transport import LocalTransport
from gpx_recipe.symmetry.worker import require_runtime
from gpx_recipe.symmetry.chemistry import validate_sites,sites_from_inputs
from .model_planner import make_intent,planning_failure_receipt
from .magnetic_optimizer import selected_optimizer_settings, optimizer_budget_label
from gpx_recipe.usage import summarize_calls,usage_labels
from gpx_recipe.symmetry.isodistort_discovery import discover_structural
from gpx_recipe.symmetry.plans import build_plan
from gpx_recipe.symmetry.contracts import iter_trials
from .symmetry_workspace import campaign_rows,group_models,display_groups,view_for,choose_group,choose_member,disposition_reason,coverage_rows,scope_rows,parameter_rows,moment_precision_note,campaign_can_resume,display_status,embedding_evidence,fit_caption,fit_diagnostics,pin_label
from . import symmetry_plots as plots
from .parent_moments import parent_frame_moments
from .symmetry_sources import start_choices,resolve_start,acquisition_details,acquisition_notices,acquisition_temperature,display_acquisition_notices
from .k_search_controller import KSearchMixin
from .model_runtime import GalaxyModelRuntime,GalaxyChunkTransport,CollectOnlyTransport,PreparationSlotTransport,active_batch_count,MODEL_BATCH_LIMITS
from .checkpoint_uploads import CheckpointUploadPending, CheckpointUploadInterrupted
from .model_review_recovery import SelectionRecovery, ReviewSelectionChanged, campaign_recency,acquisition_index,AcquisitionReview
from .model_navigation import scan_options
from .model_campaigns import choices as model_choices,origin as model_origin,queue_sources,enroll_source,restore_history,portable_source,restore_captured_sources,scope_progress,next_acquisition,prioritize_acquisition,acquisition_rows
from .model_campaigns import START_BARRIER_SCHEMA,START_BARRIER_MESSAGE,start_barrier_ready,store_start_barrier
from gpx_recipe.symmetry.compiler import prepare_shared_inputs
from gpx_recipe.symmetry.templates import make_template,trial_count

MODEL_FIELDS=('sx_kind','sx_moment_model','sx_magnetic_optimizer','sx_optimizer_iterations','sx_refine_scale','sx_refine_background','sx_parent','sx_histogram','sx_k','sx_sites','sx_form_factors','sx_lande',
    'sx_directions','sx_adequate','sx_max_moment','sx_mag_d_min','sx_max_mode','sx_start_mode','sx_cycles',
    'sx_sign_starts','sx_candidates','sx_modes','sx_existing','sx_follow_new','sx_max_scans','sx_request',
    'sx_opd_policy','sx_construct_limit')


class SymmetryMixin(KSearchMixin):
    def _init_symmetry(self):
        self._model_runtime_cache=None;self._models_restored=False;self._model_restored_scopes=set();self._model_restore_requests=set();self._model_following_active=False;self._model_stopping=False
        self._model_active_campaigns={};self._model_owned_campaigns={};self._model_review_campaigns=OrderedDict();self._model_checkpoint_errors={};self._model_preparations={};self._model_enrollment_task=None;self._model_tick_task=None;self._model_pause_task=None;self._model_start_cancel=None
        self._symmetry_rows=[];self._symmetry_all_rows=[];self._symmetry_groups=[];self._symmetry_draft=None;self._symmetry_approved='';self._symmetry_task=None;self._symmetry_loading=False;self._symmetry_refresh_pending=False;self._symmetry_explicit_selection=False;self._symmetry_display_identity=None;self._symmetry_loading_identity=None;self._symmetry_setup_details=None
        s=self.server.state
        defaults=dict(sx_campaign='',sx_campaigns=[],sx_source='',sx_source_options=[],sx_busy=False,sx_error='',sx_message='',sx_progress='',sx_groups=[],sx_group='',sx_members=[],sx_selected='',sx_search='',sx_status='all',
            sx_overview=plots.empty('Open a campaign to inspect its coverage'),sx_fit=plots.empty('Select an audited checkpoint'),sx_fit_title='',sx_expanded_title='',sx_source_notices=[],sx_source_display_notices=[],sx_axis='d',sx_reset=0,sx_pins=[],sx_pin_options=[],sx_parameters=[],sx_parent_moments=[],sx_parent_moment_note='',sx_details=[],sx_fit_diagnostics=[],sx_moment_note='',sx_embedding_evidence=[],sx_embedding_summary='',sx_gsas_model='',sx_intensity_scale='linear',
            sx_parent='',sx_parents=[],sx_histogram='',sx_histograms=[],sx_k='',sx_sites=[],sx_site_options=[],sx_form_factors='',sx_lande='',sx_directions=['x','y','z'],sx_adequate=None,sx_max_moment='',sx_cycles=20,sx_slots=2,sx_moment_model='collinear',sx_refine_scale=False,sx_refine_background=False,
            sx_kind='magnetic',sx_magnetic_optimizer='gsas_hessian',sx_optimizer_iterations=60,sx_mag_d_min=0.,sx_max_mode=.5,sx_start_mode=.1,sx_candidate_options=[],sx_candidates=[],sx_mode_options=[],sx_modes=[],sx_discovered=False,sx_discovery_scope='',
            sx_plan_ready=False,sx_approved=False,sx_plan_summary='',sx_plan_candidates=[],sx_setup=False,sx_expanded=False,sx_expanded_layout={},sx_has_context=False,sx_has_gpx=False,sx_has_receipt=False,sx_resumable=False,
            sx_request='',sx_planning_summary='',sx_planning_notes=[],sx_planning_usage='',sx_planning_cost='',sx_has_setup_details=False,sx_sign_starts=True,sx_coverage=[],sx_group_limit=12,sx_result_scope='experiment',sx_start_source='',sx_start_sources=[],sx_results_current=False,sx_scope_progress='')
        for key,value in defaults.items():setattr(s,key,value)
        extra=dict(sx_existing='selected',sx_follow_new=False,sx_max_scans=200,sx_enrollments=[],sx_priority_acquisition='',sx_priority_options=[],sx_priority_notice='',sx_following=False,sx_campaign_running=False,sx_pause_pending=False,sx_parallel_jobs=1,
            sx_queue_notice='',sx_backend='',sx_plan_token='',sx_cancel_notice='Cancelling a Galaxy chunk stops all starts in that job.',
            sx_opd_policy='highest-symmetry-per-irrep',sx_construct_limit=100)
        for key,value in extra.items():setattr(s,key,value)
        self._symmetry_defaults={**defaults,**extra}
        s.change('rf_mode')(self._symmetry_mode_changed)
        s.change('rf_experiment','sx_result_scope')(self._symmetry_mode_changed)
        s.change('sx_campaign')(self._symmetry_campaign_changed)
        s.change('sx_group')(self._symmetry_group_changed)
        s.change('sx_source')(self._symmetry_source_changed)
        s.change('sx_selected','sx_axis','sx_pins','sx_reset','sx_intensity_scale')(self._symmetry_plot_changed)
        s.change('sx_kind','sx_parent','sx_histogram','sx_k','sx_sites','sx_form_factors','sx_lande','sx_directions','sx_adequate','sx_max_moment','sx_mag_d_min','sx_max_mode','sx_start_mode','sx_cycles','sx_request','sx_sign_starts','rf_source','sx_start_source')(self._symmetry_unapprove)
        s.change('sx_parent')(self._symmetry_parent_changed)
        s.change('sx_existing','sx_follow_new','sx_max_scans')(self._symmetry_unapprove)
        s.change('sx_moment_model','sx_refine_scale','sx_refine_background')(self._symmetry_unapprove)
        s.change('sx_magnetic_optimizer','sx_optimizer_iterations')(self._symmetry_unapprove)
        s.change('sx_candidates','sx_modes')(self._symmetry_unapprove)
        s.change('sx_opd_policy','sx_construct_limit')(self._symmetry_unapprove)
        s.change('sx_start_source')(self._symmetry_start_changed)
        s.change('rf_experiment','sx_start_source','sx_request','sx_magnetic_optimizer','sx_optimizer_iterations')(self._clear_symmetry_setup_details)
        s.change('sx_parallel_jobs')(self._symmetry_resource_changed)
        self._init_k_search()

    def _symmetry_root(self):
        root=Path(os.environ.get('RADAR_PD_SYMMETRY_ROOT') or (Path(getattr(self.service,'output_root',tempfile.gettempdir()))/'symmetry'))
        root=root/digest([str(getattr(self.service,'history_id',None) or 'local')])[:16]
        root.mkdir(parents=True,exist_ok=True);return root.resolve()

    def _model_remote(self):
        if os.environ.get('RADAR_PD_SYMMETRY_PYTHON','').strip():return None
        from .galaxy_service import GalaxyService
        if not isinstance(self.service,GalaxyService):return None
        if not (self.service.galaxy_url and self.service.galaxy_key and self.service.history_id):return None
        if self._model_runtime_cache is None:self._model_runtime_cache=GalaxyModelRuntime(self._symmetry_root()/'runtime',self.service)
        self._model_runtime_cache.max_active_jobs=self.server.state.sx_parallel_jobs
        return self._model_runtime_cache

    def _symmetry_resource_changed(self,**_):
        try:
            value=self.server.state.sx_parallel_jobs
            if type(value) is not int or value not in MODEL_BATCH_LIMITS:raise ValueError('Choose 1, 2, 4 or 8 concurrent Galaxy model batches')
            if self._model_runtime_cache:self._model_runtime_cache.max_active_jobs=value
            for campaign,transport in self._model_active_campaigns.values():
                if isinstance(transport,GalaxyChunkTransport):transport.max_active_jobs=value;transport.runtime.max_active_jobs=value
                if campaign is not None:campaign.set_setting('viewer_parallel_jobs',value)
        except ValueError as exc:self.server.state.sx_error=str(exc)

    def _symmetry_transport(self,campaign=None):
        python=os.environ.get('RADAR_PD_SYMMETRY_PYTHON','')
        if not python:
            remote=self._model_remote()
            if remote and campaign:
                owned=getattr(self,'_model_owned_campaigns',{}).get(campaign.root.name)
                if owned and owned[0].root==campaign.root:
                    transport=owned[1];transport.max_active_jobs=remote.max_active_jobs
                    return transport
                return GalaxyChunkTransport(remote,campaign)
            raise ValueError('Configure a local model worker or the Galaxy model-campaign tool before starting exploration.')
        environment={}
        for name in ['GSASII_ROOT','RADAR_GPX_DOCKER_IMAGE']:
            if os.environ.get(name):environment[name]=os.environ[name]
        return LocalTransport([python],environment=environment)

    def _symmetry_campaign(self,root):
        """Retain stamp-checked receipt caches across ordinary UI refreshes."""
        root=Path(root).resolve()
        owned=getattr(self,'_model_owned_campaigns',{}).get(root.name)
        if owned and owned[0].root==root:return owned[0]
        if not hasattr(self,'_model_review_campaigns'):self._model_review_campaigns=OrderedDict()
        cache=self._model_review_campaigns
        if root not in cache:cache[root]=Campaign(root)
        cache.move_to_end(root)
        while len(cache)>2:cache.popitem(last=False)
        return cache[root]

    def _symmetry_mode_changed(self,**_):
        if self.server.state.rf_mode=='models':self.refresh_symmetry()

    def _symmetry_unapprove(self,**_):
        self._symmetry_approved='';self.server.state.sx_approved=False;self.server.state.sx_plan_ready=False
        d=getattr(self,'_symmetry_draft',None)
        if d and d.get('discovery_identity') and d['discovery_identity']!=self._discovery_identity():
            s=self.server.state;s.sx_discovered=False;s.sx_candidates=[];s.sx_candidate_options=[];s.sx_modes=[];s.sx_mode_options=[]

    def _symmetry_parent_changed(self,**_):
        d=self._symmetry_draft;s=self.server.state
        if d:s.sx_site_options=[dict(title=a['label']+' ('+a['type']+')',value=a['label']) for p in d['context']['phases'] if p['name']==s.sx_parent for a in p['atoms']]
        if not getattr(self,'_symmetry_applying_intent',False):
            s.sx_sites=[];s.sx_form_factors='';s.sx_lande=''

    def _symmetry_start_changed(self,**_):
        self._clear_symmetry_setup_details()
        if self._symmetry_draft:
            event=getattr(self,'_model_preparations',{}).get(str(self._symmetry_draft['root']))
            if event:event.set()
        self._symmetry_draft=None
        s=self.server.state;s.sx_has_context=False;s.sx_discovered=False;s.sx_planning_summary='';s.sx_planning_notes=[];s.sx_planning_usage='';s.sx_planning_cost=''
        s.sx_sites=[];s.sx_form_factors='';s.sx_lande=''

    def _symmetry_setup_identity(self):
        s=self.server.state;d=getattr(self,'_symmetry_draft',None) or {}
        return (getattr(s,'rf_experiment',''),s.sx_start_source,s.sx_request,d.get('context',{}).get('sha256'),
                getattr(s,'sx_magnetic_optimizer','gsas_hessian'),getattr(s,'sx_optimizer_iterations',60))

    def _clear_symmetry_setup_details(self,**_):
        self._symmetry_setup_details=None;self.server.state.sx_has_setup_details=False

    def _retain_symmetry_setup_details(self,receipt,identity):
        if identity!=self._symmetry_setup_identity():return
        try:data=(json.dumps(receipt,indent=2,allow_nan=False)+'\n').encode()
        except (ValueError,TypeError):return
        if len(data)>512*1024:return
        self._symmetry_setup_details=dict(identity=identity,data=data)
        self.server.state.sx_has_setup_details=True

    def download_symmetry_setup_details(self,**_):
        s=self.server.state;details=getattr(self,'_symmetry_setup_details',None)
        if not details or details['identity']!=self._symmetry_setup_identity():
            self._clear_symmetry_setup_details()
            s.sx_error='Draft search settings for the current source and request before downloading setup details.';s.flush();return
        self.download_file('magnetic-setup-details.json','application/json',details['data'])

    def new_symmetry_search(self):
        self._symmetry_start_changed()
        self._symmetry_unapprove()
        s=self.server.state
        selected=getattr(s,'sx_source','')
        if selected and selected in {entry['value'] for entry in getattr(s,'sx_start_sources',[])}:
            s.sx_start_source=selected
        s.sx_setup=True;s.sx_plan_token='';s.sx_plan_summary='';s.sx_plan_candidates=[]
        s.sx_candidates=[];s.sx_candidate_options=[];s.sx_modes=[];s.sx_mode_options=[]
        s.sx_message='Select a starting GPX and inspect its exact parent to prepare a new search.'

    def symmetry_adjacent_scan(self,step):
        if type(step) is not int or step not in (-1,1):return
        s=self.server.state;ids=[row['value'] for row in s.sx_source_options]
        if s.sx_source in ids:
            index=ids.index(s.sx_source)+step
            if 0<=index<len(ids):s.sx_source=ids[index]

    def _symmetry_native(self,action,payload,root):
        import threading
        event=self._model_preparations.setdefault(str(root),threading.Event())
        if event.is_set():raise ValueError('This model preparation was cancelled')
        remote=self._model_remote()
        if remote:
            if not self._symmetry_draft:raise ValueError('Inspect the source before native discovery')
            result,_=remote.native(action,payload,root,source=self._symmetry_draft['source'],cancel=event)
            return result
        transport=self._symmetry_transport();uid=uuid.uuid4().hex;request=root/(uid+'-request.json');output=root/(uid+'-response.json')
        request.write_text(json.dumps(payload)+'\n');env={**scientific_environment(),**transport.environment,'RADAR_GPX_DOCKER_WORKDIR':str(root)}
        with (root/(uid+'-native.log')).open('w') as log:
            result=run_scientific(transport.command+['-m','gpx_recipe.symmetry.isodistort_native',action,str(request),str(output)],timeout=120,cancel_event=event,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode or not output.is_file():raise ValueError('Native preparation failed; retained log '+uid+'-native.log')
        return json.loads(output.read_text())

    def _inspect_model_input(self,source,root):
        import threading
        root=Path(root);root.mkdir(parents=True,exist_ok=True)
        target=root/'source.gpx'
        if not target.exists():shutil.copyfile(source['source_path'],target)
        if file_sha(target)!=source['source_sha256']:raise ValueError('Model source changed while being captured')
        event=self._model_preparations.setdefault(str(root),threading.Event())
        if event.is_set():raise ValueError('Model inspection cancelled')
        remote=self._model_remote()
        if remote:
            context,_=remote.native('inspect',dict(source=str(target)),root,source=source,cancel=event)
        else:
            transport=self._symmetry_transport()
            env={**scientific_environment(),**transport.environment,'RADAR_GPX_DOCKER_WORKDIR':str(root)}
            with (root/'inspect.log').open('w') as log:
                result=run_scientific(transport.command+['-m','gpx_recipe.symmetry.inspect',str(target),str(root/'context.json')],
                    timeout=120,cancel_event=event,env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise ValueError('Native inspection stopped; inspect its captured log')
            context=json.loads((root/'context.json').read_text())
        require_runtime(context['runtime'],context['runtime']['magnetic_function_sha256'])
        if context['sha256']!=source['source_sha256']:raise ValueError('The source GPX revision changed during native inspection')
        return context

    def _model_following(self):return getattr(self,'_model_following_active',False)

    def _symmetry_campaign_changed(self,**_):
        if not self._symmetry_loading:
            self._symmetry_explicit_selection=False
        self.refresh_symmetry()

    def restore_saved_symmetry_campaigns(self):
        """Restore prior work only after an explicit request, never on tab entry."""
        s=self.server.state
        if self._symmetry_loading or s.sx_busy or getattr(s,'sx_pause_pending',False):return
        scope=s.rf_experiment if s.sx_result_scope=='experiment' else None
        requests=getattr(self,'_model_restore_requests',set())
        self._model_restore_requests=requests|{scope}
        # Also discover campaigns completed by another viewer since our last
        # explicit load. Existing local ledgers are not overwritten by restore.
        self._model_restored_scopes=getattr(self,'_model_restored_scopes',set())-{scope,None}
        self.refresh_symmetry()

    def refresh_symmetry(self):
        if self._symmetry_loading:
            s=self.server.state
            if (s.sx_campaign,s.rf_experiment,s.sx_result_scope)!=self._symmetry_loading_identity:
                self._symmetry_refresh_pending=True;s.sx_results_current=False
                s.sx_source_options=[]
                event=getattr(self,'_symmetry_review_cancel',None)
                if event:event.set()
            return
        # Claim the refresh before scheduling it. Several ticks/UI events can
        # arrive before the coroutine starts; only one may audit this ledger.
        self._symmetry_loading=True
        s=self.server.state
        self._symmetry_recovery_source=getattr(s,'sx_source','')
        self._symmetry_loading_identity=(s.sx_campaign,s.rf_experiment,s.sx_result_scope)
        if getattr(self,'_symmetry_navigation_identity',None)!=self._symmetry_loading_identity:
            s.sx_source_options=[]
            self._symmetry_navigation_identity=self._symmetry_loading_identity
        async def load():
            s=self.server.state;s.sx_busy=True;s.sx_error=''
            self._symmetry_loading_identity=(s.sx_campaign,s.rf_experiment,s.sx_result_scope)
            s.sx_results_current=getattr(self,'_symmetry_display_identity',None)==self._symmetry_loading_identity
            s.flush()
            try:
                remote=self._model_remote()
                backend='Galaxy jobs · allocated CPU/RAM' if remote else 'Configured local scientific worker'
                s.sx_backend=backend
                scope=s.rf_experiment if s.sx_result_scope=='experiment' else None
                restored=getattr(self,'_model_restored_scopes',set())
                requested=getattr(self,'_model_restore_requests',set())
                # Consume this user action even if the load fails or selection
                # changed before the coroutine started. Polling must not retry
                # an expensive historical restore without another request.
                self._model_restore_requests=set()
                if remote and scope in requested and scope not in restored and None not in restored:
                    loop=asyncio.get_running_loop()
                    identity=self._symmetry_loading_identity
                    def publish_progress(message):
                        if identity==(s.sx_campaign,s.rf_experiment,s.sx_result_scope):
                            s.sx_message=message;s.flush()
                    def progress(message):loop.call_soon_threadsafe(publish_progress,message)
                    await asyncio.to_thread(restore_history,self._symmetry_root(),remote,experiment=scope,progress=progress)
                    self._model_restored_scopes=restored|{scope}
                    self._models_restored=True
                    if identity!=(s.sx_campaign,s.rf_experiment,s.sx_result_scope):
                        self._symmetry_refresh_pending=True;return
                choices=start_choices(self._refinement_store(),s.rf_experiment)
                s.sx_start_sources=[dict(title=entry['title'],value=entry['id']) for entry in choices]
                if s.sx_start_source not in {entry['id'] for entry in choices}:
                    s.sx_start_source=s.rf_source if s.rf_source in {entry['id'] for entry in choices} else (choices[0]['id'] if choices else '')
                roots=list(self._symmetry_root().glob('*/campaign.sqlite'))
                if scope is not None:
                    roots=[p for p in roots if (Campaign(p.parent).setting('template',{}) or {}).get('scope',{}).get('experiment')==scope]
                roots=sorted(roots,key=campaign_recency)
                s.sx_campaigns=[dict(title=p.parent.name,value=p.parent.name) for p in roots]
                if s.sx_campaign not in {p.parent.name for p in roots}:s.sx_campaign=roots[-1].parent.name if roots else ''
                if not s.sx_campaign:
                    self._symmetry_rows=[];self._symmetry_all_rows=[];self._symmetry_groups=[]
                    s.sx_source='';s.sx_source_options=[];s.sx_enrollments=[];s.sx_coverage=[]
                    s.sx_progress='';s.sx_scope_progress='';s.sx_resumable=False;s.sx_following=False
                    self._symmetry_publish_source()
                    s.sx_message='Select a completed RADAR-PD GPX above, then prepare model exploration.';return
                root=next(p.parent for p in roots if p.parent.name==s.sx_campaign)
                campaign=self._symmetry_campaign(root)
                review_source=None;indexed_options=None
                if remote and campaign.setting('restored',False):
                    # The active driver already polls and verifies these same
                    # batches under the campaign lock. A second recovery pass
                    # can block dispatch while rereading every completed start.
                    # Review-only/restored viewers still collect existing jobs;
                    # campaign_rows independently verifies displayed artifacts.
                    coordinating=campaign.root.name in getattr(self,'_model_active_campaigns',{})
                    s.sx_message=('Reading verified results while the campaign collects accepted Galaxy jobs.' if coordinating
                        else 'Recovering saved Galaxy results and checking their artifacts. No new refinements are being submitted.')
                    s.flush()
                    if not coordinating:
                        indexed_options,protocols_by_source=acquisition_index(campaign,experiment=scope)
                        if indexed_options:
                            review_source=s.sx_source if s.sx_source in indexed_options else next(iter(indexed_options))
                            self._symmetry_recovery_source=review_source
                            s.sx_source_options=[dict(title=label,value=key) for key,label in indexed_options.items()]
                            s.sx_source=review_source
                            s.sx_results_current=False;s.flush()
                        else:indexed_options=None
                        self._symmetry_loading_identity=(s.sx_campaign,s.rf_experiment,s.sx_result_scope)
                        self._symmetry_review_cancel=threading.Event()
                        cancelled=self._symmetry_review_cancel
                        identity=self._symmetry_loading_identity
                        loop=asyncio.get_running_loop()
                        def publish_result_progress(count, request_identity=identity, request_cancelled=cancelled):
                            if not request_cancelled.is_set() and request_identity==(s.sx_campaign,s.rf_experiment,s.sx_result_scope):
                                s.sx_message=f'Verifying saved native result group {count}. No new refinements are being submitted.'
                                s.flush()
                        def result_progress(count):loop.call_soon_threadsafe(publish_result_progress,count)
                        recovery=SelectionRecovery(self._symmetry_transport(campaign),cancelled,progress=result_progress)
                        if review_source:
                            await asyncio.to_thread(AcquisitionReview(campaign,protocols_by_source[review_source]).recover,recovery)
                        else:
                            await asyncio.to_thread(campaign.recover,recovery)
                allocated=campaign.setting('allocation')
                if allocated:backend+=f" · last chunk: {allocated['workers']} native workers · {allocated.get('memory_mb') or 'unreported'} MiB memory"
                s.sx_backend=backend
                template=campaign.setting('template') or {}
                visible=s.sx_result_scope!='experiment' or template.get('scope',{}).get('experiment')==s.rf_experiment
                s.sx_enrollments=acquisition_rows(template,campaign.setting('enrollments',{})) if visible else []
                s.sx_priority_options=[dict(title=entry['scan'],value=entry['id']) for entry in s.sx_enrollments if entry['status']=='queued']
                eligible_priorities={option['value'] for option in s.sx_priority_options}
                if getattr(s,'sx_priority_acquisition','') not in eligible_priorities:s.sx_priority_acquisition=''
                priority=campaign.setting('priority_acquisition')
                queued_priority=next((entry for entry in s.sx_enrollments if entry['id']==priority and entry['status'] in {'queued','preparing'}),None)
                s.sx_priority_notice=('Priority: '+queued_priority['scan']+'. Already accepted work finishes normally; the remaining captured scans stay queued.' if queued_priority else '')
                s.sx_scope_progress=scope_progress(template,s.sx_enrollments)
                s.sx_following=bool(campaign.setting('following',False)) and not campaign.setting('paused',False)
                s.sx_queue_notice=campaign.setting('checkpoint_notice','') or campaign.setting('dispatch_notice','') or campaign.setting('preparation_message','') or campaign.setting('budget_message','')
                if campaign.setting('paused',False):s.sx_queue_notice='Campaign paused. Submitted Galaxy jobs continue independently; Continue recovers them before new starts.'
                if getattr(s,'sx_pause_pending',False):s.sx_queue_notice='Pausing new submissions and saving recovery state. Already accepted native jobs are not cancelled.'
                identity=(s.sx_campaign,s.rf_experiment,s.sx_result_scope)
                self._symmetry_loading_identity=identity
                review_args=dict(experiment=identity[1] if identity[2]=='experiment' else None)
                if review_source is not None:review_args['acquisition']=review_source
                rows,summaries=await asyncio.to_thread(campaign_rows,root,**review_args)
                if identity!=(s.sx_campaign,s.rf_experiment,s.sx_result_scope):
                    self._symmetry_refresh_pending=True;return
                rows=scope_rows(rows,s.rf_experiment,s.sx_result_scope)
                s.sx_coverage=coverage_rows(rows)
                s.sx_resumable=campaign_can_resume(summaries,list(campaign.setting('enrollments',{}).values()),
                    following=bool(template and campaign.setting('following',False)))
                counts=Counter(r['status'] for r in rows)
                current=sum(r['execution_state']=='terminal' for r in rows)
                s.sx_progress=f'{current} / {len(rows)} terminal receipts in this view · '+', '.join(f'{n} {display_status(k)}' for k,n in sorted(counts.items()))
                if s.sx_result_scope=='experiment':s.sx_progress+=f' · whole campaign verified receipts: {sum(v["received_count"] for v in summaries)} / {sum(v["declared_count"] for v in summaries)}'
                if rows:s.sx_message='Audited fits remain hypotheses. Matching calculated curves do not establish a unique space group.'
                elif s.sx_enrollments:
                    pending=sum(e['status'] in {'queued','preparing'} for e in s.sx_enrollments)
                    s.sx_message=(f'{pending} acquisition(s) awaiting native preparation. Trial results appear after their Galaxy jobs finish.' if pending
                        else 'No native starts are available yet. Open the acquisition queue to inspect compatibility or review reasons.')
                elif not visible:s.sx_message='This campaign belongs to another experiment. Choose that experiment or view the whole campaign.'
                else:s.sx_message='No native trials have been declared yet. Check the acquisition queue and reviewed campaign scope.'
                options=indexed_options if indexed_options is not None else scan_options(rows)
                source_order={entry['id']:index for index,entry in enumerate(s.sx_enrollments)}
                options=dict(sorted(options.items(),key=lambda item:source_order.get(item[0],len(source_order))))
                selected=s.sx_source if s.sx_source in options else next(iter(options),'')
                # Prepare a new view without mutating the displayed snapshot.
                # The user may inspect its verified scans while this audit waits
                # on Galaxy. A late result must not restore an old selection.
                source_rows,groups=await asyncio.to_thread(self._symmetry_source_snapshot,rows,selected)
                if identity!=(s.sx_campaign,s.rf_experiment,s.sx_result_scope):
                    self._symmetry_refresh_pending=True;return
                latest=s.sx_source if s.sx_source in options else next(iter(options),'')
                if selected!=latest:
                    self._symmetry_refresh_pending=True;return
                self._symmetry_all_rows=rows
                self._symmetry_review_acquisition=review_source
                self._symmetry_recovery_source=selected
                self._symmetry_rows=source_rows;self._symmetry_groups=groups
                s.sx_source_options=[dict(title=label,value=key) for key,label in options.items()]
                s.sx_source=selected
                self._symmetry_publish_source()
                self._symmetry_display_identity=identity;s.sx_results_current=True
            except ReviewSelectionChanged:self._symmetry_refresh_pending=True
            except Exception as exc:s.sx_error=str(exc)
            finally:
                self._symmetry_review_cancel=None
                s.sx_results_current=getattr(self,'_symmetry_display_identity',None)==(s.sx_campaign,s.rf_experiment,s.sx_result_scope)
                self._symmetry_loading=False;s.sx_busy=False;s.flush()
                if self._symmetry_refresh_pending:
                    self._symmetry_refresh_pending=False;self.refresh_symmetry()
        request=load()
        try:self._refinement_schedule(request,'radar-symmetry-refresh')
        except Exception:
            self._symmetry_loading=False;request.close()
            raise

    @staticmethod
    def _symmetry_source_snapshot(rows,source):
        selected=[r for r in rows if r['source']==source]
        return selected,display_groups(group_models(selected))

    def _symmetry_source_data(self,source):
        self._symmetry_rows,self._symmetry_groups=self._symmetry_source_snapshot(self._symmetry_all_rows,source)

    def _symmetry_publish_source(self):
        s=self.server.state;groups=self._symmetry_groups;rows=self._symmetry_rows
        s.sx_groups=[dict(id=g['id'],title=g['representative']['name'] if g['representative']['has_view'] else display_status(g['representative']['status']).capitalize(),count=len(g['members']),rwp=(f"{g['representative']['rwp']:.3f}%" if g['representative']['rwp'] is not None else '—'),status=display_status(g['representative']['status'])) for g in groups]
        s.sx_overview=plots.overview(rows,s.sx_selected)
        s.sx_pin_options=[dict(title=pin_label(r,include_start=True),value=r['id']) for r in rows if r['has_view']]
        pinnable={r['id'] for r in rows if r['has_view']}
        s.sx_pins=[uid for uid in s.sx_pins if uid in pinnable][-2:]
        s.sx_group=choose_group(groups,s.sx_selected,explicit=self._symmetry_explicit_selection)
        self._symmetry_group_changed()

    def _symmetry_source_changed(self,**_):
        # Inspection consumes only the last verified snapshot. It neither
        # resumes work nor changes the reviewed campaign's input/plan scope.
        s=self.server.state
        loaded=getattr(self,'_symmetry_review_acquisition',None)
        trusted_full_snapshot=(loaded is None and s.sx_results_current and
            getattr(self,'_symmetry_display_identity',None)==
            (getattr(s,'sx_campaign',''),getattr(s,'rf_experiment',''),getattr(s,'sx_result_scope','')))
        if getattr(self,'_symmetry_loading',False) and not trusted_full_snapshot:
            if getattr(self,'_symmetry_recovery_source',None)!=s.sx_source:
                self._symmetry_refresh_pending=True;s.sx_results_current=False
                event=getattr(self,'_symmetry_review_cancel',None)
                if event:event.set()
            return
        if loaded is not None and s.sx_source!=loaded:
            self._symmetry_explicit_selection=False;s.sx_results_current=False
            self.refresh_symmetry();return
        if not s.sx_results_current:return
        if getattr(self,'_symmetry_display_identity',None)!=(s.sx_campaign,s.rf_experiment,s.sx_result_scope):return
        self._symmetry_explicit_selection=False
        s.sx_group_limit=12
        self._symmetry_source_data(s.sx_source);self._symmetry_publish_source()

    def select_symmetry_group(self,uid):
        self._symmetry_explicit_selection=True;self.server.state.sx_group=uid

    def _symmetry_user_selected(self,**_):self._symmetry_explicit_selection=True

    def _symmetry_group_changed(self,**_):
        s=self.server.state;group=next((g for g in self._symmetry_groups if g['id']==s.sx_group),None)
        members=group['members'] if group else []
        s.sx_members=[dict(title=r['name']+' · '+r['candidate'][-6:]+' · '+r['seed']+' · '+display_status(r['status']),value=r['id']) for r in members]
        s.sx_selected=choose_member(members,s.sx_selected,explicit=self._symmetry_explicit_selection)
        self._symmetry_plot_changed()

    def _symmetry_plot_changed(self,**_):
        s=self.server.state
        # A newly selected view may fail before plotting or receipt reads finish.
        # Never leave another trial's derived direction under that selection.
        s.sx_parent_moments=[];s.sx_parent_moment_note=''
        row=next((r for r in self._symmetry_rows if r['id']==s.sx_selected),None)
        s.sx_fit_title=fit_caption(row)
        s.sx_expanded_title=s.sx_fit_title
        s.sx_source_notices=[];s.sx_source_display_notices=[]
        if len(s.sx_pins)>2:s.sx_pins=s.sx_pins[-2:]
        pins=[r for r in self._symmetry_rows if r['id'] in s.sx_pins]
        try:
            s.sx_overview=plots.overview(self._symmetry_rows,s.sx_selected)
            s.sx_fit=plots.fit(row,pins,axis=s.sx_axis,reset=s.sx_reset,intensity_scale=s.sx_intensity_scale)
            s.sx_expanded_layout={**{k:v for k,v in s.sx_fit['layout'].items() if k!='height'},'autosize':True}
            s.sx_details=[];s.sx_parameters=[];s.sx_moment_note='';s.sx_has_gpx=bool(row and row.get('has_gpx') and row['has_view'])
            s.sx_fit_diagnostics=fit_diagnostics(row)
            s.sx_embedding_evidence,s.sx_embedding_summary=embedding_evidence(self._symmetry_rows,row)
            s.sx_has_receipt=bool(row and row.get('receipt_sha256'))
            if row:
                workspace=self._refinement_store()
                s.sx_source_notices=acquisition_notices(workspace,s.rf_experiment,row['source'])
                s.sx_source_display_notices=display_acquisition_notices(s.sx_source_notices)
                s.sx_expanded_title=fit_caption(row,temperature=acquisition_temperature(
                    workspace,s.rf_experiment,row['source'],row.get('source_sha256')))
                reason=disposition_reason(row)
                s.sx_details=[dict(label='Status',value=display_status(row['status'])),dict(label='Reason',value=reason),
                    dict(label='Candidate',value=row['name']+(' · BNS '+row['bns'] if row['bns'] else '')),dict(label='Embedding',value=json.dumps(row['embedding'])),dict(label='Start',value=row['seed']),
                    dict(label='Worker',value='GSAS-II '+row.get('native_version','pending')+' · implementation '+row['worker_revision'][:12])]
                if row['kind']=='magnetic':s.sx_details.append(dict(label='Complete BNS operator audit',value='Passed' if row.get('bns_audited') else 'Not present in this archived or unfinished trial'))
                s.sx_details.append(dict(label='Parent-basis k',value='('+', '.join(row['k'])+')'))
                s.sx_details.append(dict(label='Starting GPX revision',value=row['source_sha256'][:16]))
                s.sx_details.extend(acquisition_details(self._refinement_store(),s.rf_experiment,row['source']))
                usage=row.get('resources') or {}
                if isinstance(usage.get('wall_seconds'),(int,float)):
                    detail=f"{usage['wall_seconds']:.2f} s native wall time"
                    if isinstance(usage.get('cpu_seconds'),(int,float)):detail+=f" · {usage['cpu_seconds']:.2f} s CPU"
                    if isinstance(usage.get('peak_rss_bytes'),(int,float)):detail+=f" · {usage['peak_rss_bytes']/1024**2:.0f} MiB peak RAM"
                    s.sx_details.append(dict(label='Measured resources',value=detail+' · excludes Galaxy queue and transfer'))
                if row.get('sites'):s.sx_details.append(dict(label='Magnetic sites',value='; '.join(site['label']+'='+site['form_factor']+' (g='+str(site['lande_g'])+')' for site in row['sites'])))
                if row.get('baseline_rwp') is not None and row.get('rwp') is not None:
                    baseline=row['baseline_rwp']
                    s.sx_details.append(dict(label='Nuclear baseline Rwp',value=f'{baseline:.6f}% · selected change {row["rwp"]-baseline:+.6f} percentage points'))
                if row['has_view']:
                    view=view_for(row)
                    s.sx_parameters=parameter_rows(row,view)
                    s.sx_parent_moments,s.sx_parent_moment_note=parent_frame_moments(row,view)
                    s.sx_moment_note=moment_precision_note(row)
        except Exception as exc:s.sx_error=str(exc)

    def symmetry_reset_zoom(self):self.server.state.sx_reset+=1

    def download_symmetry_artifact(self,kind):
        s=self.server.state
        try:
            row=next(r for r in self._symmetry_rows if r['id']==s.sx_selected)
            receipt=verify_receipt(row['directory'],dict(id=row['id'],protocol_sha256=row['protocol']))
            if digest(receipt)!=row.get('receipt_sha256'):raise ValueError('The receipt changed; refresh evidence before downloading')
            if kind=='gpx':
                if not row.get('has_gpx') or not row['has_view']:raise ValueError('No independently audited native checkpoint is available')
                path=confined(row['directory'],row['gpx_relative']);self.download_file(row['id']+'-'+path.name,'application/octet-stream',path.read_bytes())
            elif kind=='receipt':self.download_file(row['id']+'-receipt.json','application/json',(json.dumps(receipt,indent=2)+'\n').encode())
            else:raise ValueError('Unknown native artifact')
        except Exception as exc:s.sx_error=str(exc);s.flush()

    def open_symmetry_in_gsasii(self):
        """Open the exact audited trial selected at click time, never a collection preference."""
        s=self.server.state
        if s.gsasii_session_status in {'starting','ready'}:return
        row=next((dict(r) for r in self._symmetry_rows if r['id']==s.sx_selected),None)
        if not row or not row.get('has_gpx') or not row.get('has_view'):
            s.sx_error='Select an independently audited native checkpoint first.';s.flush();return
        s.gsasii_session_status='starting';s.gsasii_launch_url='';s.sx_error=''
        s.sx_gsas_model=row['name']+' · '+row['id']
        s.gsasii_status_message='Publishing this exact audited trial and waiting for a GSAS-II desktop.';s.flush()
        async def launch():
            try:
                def capture():
                    receipt=verify_receipt(row['directory'],dict(id=row['id'],protocol_sha256=row['protocol']))
                    if digest(receipt)!=row.get('receipt_sha256'):raise ValueError('The receipt changed; refresh evidence before opening')
                    source=confined(row['directory'],row['gpx_relative'])
                    expected=receipt.get('files',{}).get(row['gpx_relative'])
                    if not expected:raise ValueError('This native GPX is absent from the audited receipt')
                    root=self._symmetry_root()/'desktop-inputs';root.mkdir(parents=True,exist_ok=True)
                    target=root/(digest(receipt)+'.gpx');shutil.copyfile(source,target)
                    if file_sha(target)!=expected:raise ValueError('The selected native GPX changed during capture')
                    return target
                path=await asyncio.to_thread(capture)
                dataset=await asyncio.to_thread(self.service.upload_document,path,label='Audited model '+row['id'])
                from .galaxy_service import GSASII_INTERACTIVE_TOOL_ID
                action=await self._submit_utility_action(tool_id=GSASII_INTERACTIVE_TOOL_ID,
                    name='Inspect audited model '+row['name']+' · '+row['id'],
                    inputs={'project_source|source_kind':'single','project_source|gpx_project':{'dataset_id':dataset}})
                if action is None:raise ValueError('Galaxy did not accept the selected-model desktop; review its activity error')
            except Exception as exc:
                s.sx_error=str(exc);s.gsasii_session_status='error';s.gsasii_status_message=str(exc);s.flush()
        self._schedule_utility(launch(),'radar-model-gsasii')

    def symmetry_overview_click(self,event):
        points=event.get('points',[]) if isinstance(event,dict) else []
        uid=points[0].get('customdata') if points else None
        for group in self._symmetry_groups:
            if any(r['id']==uid for r in group['members']):self._symmetry_explicit_selection=True;self.server.state.sx_group=group['id'];self.server.state.sx_selected=uid;return

    def inspect_symmetry_source(self):
        async def inspect():
            self._clear_symmetry_setup_details()
            s=self.server.state;s.sx_busy=True;s.sx_error='';s.sx_has_context=False
            s.sx_message='Preparing the exact parent: capturing the selected GPX, then staging a native inspection. Galaxy transfer and queue time are included in this wait.';s.flush()
            try:
                w=self._refinement_store()
                identity=(s.rf_experiment,s.sx_start_source)
                source=await asyncio.to_thread(resolve_start,w,s.rf_experiment,s.sx_start_source)
                captured=await asyncio.to_thread(w.runtime.resolve,source)
                root=self._symmetry_root()/'drafts'/uuid.uuid4().hex;root.mkdir(parents=True)
                context=await asyncio.to_thread(self._inspect_model_input,captured,root)
                if identity!=(s.rf_experiment,s.sx_start_source):raise ValueError('The starting checkpoint changed during inspection; inspect the current selection')
                if context['sha256']!=captured['source_sha256']:raise ValueError('The captured GPX revision changed during native inspection')
                self._symmetry_draft=dict(root=root,context=context,source=source,source_id=identity[1],experiment=identity[0])
                s.sx_discovered=False;s.sx_planning_summary='';s.sx_planning_notes=[]
                s.sx_parents=[dict(title=p['name']+' · '+p['space_group'],value=p['name']) for p in context['phases'] if p['type']=='nuclear']
                s.sx_histograms=[dict(title=h['name'],value=h['name']) for h in context['histograms']]
                s.sx_parent=s.sx_parents[0]['value'] if s.sx_parents else '';s.sx_histogram=s.sx_histograms[0]['value'] if s.sx_histograms else ''
                s.sx_site_options=[dict(title=p['name']+' / '+a['label']+' ('+a['type']+')',value=a['label']) for p in context['phases'] if p['name']==s.sx_parent for a in p['atoms']]
                s.sx_has_context=True;s.sx_setup=True;s.sx_message='Parent setting, operations and instrument coordinates captured from this exact GPX.'
            except Exception as exc:s.sx_error=str(exc);s.sx_message='Parent inspection stopped; review the error above.'
            finally:s.sx_busy=False;s.flush()
        self._refinement_schedule(inspect(),'radar-symmetry-inspect')

    def plan_symmetry_request(self):
        """A setup-only call; discovery and native dispatch still need review."""
        async def plan():
            self._clear_symmetry_setup_details()
            s=self.server.state;s.sx_busy=True;s.sx_error='';s.sx_planning_usage='';s.sx_planning_cost='';s.sx_planning_summary='';s.sx_planning_notes=[]
            s.sx_message='Asking Illumine for a setup-only draft. No model refinement starts from this request.';s.flush()
            d=None;identity=None;request=s.sx_request
            try:
                d=self._symmetry_draft
                if not d or d['source_id']!=s.sx_start_source:raise ValueError('Inspect this exact source GPX first')
                identity=self._symmetry_setup_identity()
                declared=selected_optimizer_settings(getattr(s,'sx_kind','magnetic'),
                    getattr(s,'sx_magnetic_optimizer','gsas_hessian'),
                    getattr(s,'sx_optimizer_iterations',60))
                if str(s.sx_k or '').strip():
                    k=[str(Fraction(v.strip())) for v in s.sx_k.split(',')]
                    if len(k)!=3:raise ValueError('Review three rational k components, or clear k to prepare a data-based survey')
                    declared['k']=k
                result=await asyncio.to_thread(make_intent,request,d['context'],declared=declared)
                if identity!=self._symmetry_setup_identity():raise ValueError('The source or request changed during planning; plan again')
                (d['root']/'planning.json').write_text(json.dumps(result,indent=2)+'\n')
                d['planning']=result
                receipt={key:result[key] for key in ['schema','status','intent','payload_sha256','intent_sha256','payload_bytes'] if key in result}
                metadata=('provider','label','model','reasoning_effort','purpose','output_format','finish_reason','output_allowance','elapsed_seconds','usage')
                receipt['calls']=[]
                for call in result.get('calls',[])[:2]:
                    kept={key:call[key] for key in metadata if key in call}
                    if isinstance(call.get('response'),str):
                        kept.update(response=call['response'][:65536],response_truncated=len(call['response'])>65536)
                    receipt['calls'].append(kept)
                self._retain_symmetry_setup_details(receipt,identity)
                s.sx_planning_usage,s.sx_planning_cost=usage_labels(summarize_calls(result['calls']))
                intent=result['intent'];s.sx_planning_notes=intent['assumptions']+intent['questions']+intent['unsupported']
                s.sx_planning_summary=f"{len(result['calls'])} setup call(s) · "+('needs clarification' if result['status']=='needs_input' else 'ready to review and discover candidates')
                if result['status']=='ready_for_k_survey':
                    s.sx_planning_summary=f"{len(result['calls'])} setup call(s) · review the site/ion settings, then choose Find candidate k-vectors. Review and use a measured candidate before discovery."
                self._symmetry_unapprove();s.sx_discovered=False
                if intent['settings'] is not None:
                    settings=intent['settings'];self._symmetry_applying_intent=True
                    try:
                        s.sx_kind=settings['kind'];s.sx_parent=settings['parent'];s.sx_histogram=settings['histogram']
                        s.sx_k=', '.join(settings['k']) if settings['k'] is not None else '';s.sx_sites=[site['label'] for site in settings['sites']]
                        s.sx_moment_model=settings.get('moment_model','collinear')
                        s.sx_magnetic_optimizer=settings.get('magnetic_optimizer','gsas_hessian')
                        s.sx_optimizer_iterations=settings.get('optimizer_iterations',60)
                        s.sx_refine_scale=settings.get('refine_histogram_scale',False)
                        s.sx_refine_background=settings.get('refine_background',False)
                        s.sx_form_factors='; '.join(site['label']+'='+site['form_factor'] for site in settings['sites'])
                        gs={site['lande_g'] for site in settings['sites']}
                        s.sx_lande=str(next(iter(gs))) if len(gs)==1 else '; '.join(site['label']+'='+str(site['lande_g']) for site in settings['sites'])
                        s.sx_directions=settings['directions'];s.sx_adequate=settings['adequate_rwp'];s.sx_max_moment=settings['maximum_moment']
                        s.sx_mag_d_min=settings['magnetic_d_min'];s.sx_cycles=settings['cycles'];s.sx_sign_starts=settings['relative_sign_starts']
                        enrollment=intent.get('enrollment')
                        if enrollment is not None:
                            s.sx_existing=enrollment['existing'];s.sx_max_scans=enrollment['maximum_scans'];s.sx_follow_new=enrollment['follow_new']
                            scope='published matching scans for this sample and bank' if s.sx_existing=='sample' else 'the selected scan'
                            s.sx_planning_notes=list(s.sx_planning_notes)+[f"Enrollment controls set to {scope}, maximum {s.sx_max_scans}; following new scans {'on' if s.sx_follow_new else 'off'}. Build the finite plan to check the actual eligible count."]
                        else:
                            s.sx_planning_notes=list(s.sx_planning_notes)+['Scan enrollment controls are unchanged. Review the acquisition scope and follow-new-scans setting before building the finite plan.']
                        s.flush()
                    finally:self._symmetry_applying_intent=False
                s.sx_message=s.sx_planning_summary
            except Exception as exc:
                calls=getattr(exc,'planning_calls',[])
                if calls:
                    s.sx_planning_usage,s.sx_planning_cost=usage_labels(summarize_calls(calls))
                    s.sx_planning_summary=f'{len(calls)} setup call(s) · no settings accepted'
                    receipt=planning_failure_receipt(exc,request=request,source_sha256=(d or {}).get('context',{}).get('sha256'))
                    if identity is not None:self._retain_symmetry_setup_details(receipt,identity)
                    if d and d.get('root'):
                        target=d['root']/('planning-failure-'+uuid.uuid4().hex+'.json')
                        try:
                            target.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
                            s.sx_planning_notes=['No settings were accepted. Download setup details to inspect the returned values and validation error.'] if s.sx_has_setup_details else []
                        except (OSError,TypeError,ValueError):
                            s.sx_planning_notes=['Failed setup replies could not be retained; no settings were accepted.']
                s.sx_error=str(exc);s.sx_message='Setup planning stopped; review the error above.'
            finally:s.sx_busy=False;s.flush()
        self._refinement_schedule(plan(),'radar-symmetry-plan')

    def discover_symmetry_candidates(self):
        async def discover():
            s=self.server.state;s.sx_busy=True;s.sx_error=''
            s.sx_message='Discovering finite candidates for the reviewed parent and k. Refinement starts only after you build and approve the plan.';s.flush()
            try:
                d=self._symmetry_draft
                if not d or d['source_id']!=s.sx_start_source:raise ValueError('Inspect the selected source GPX first')
                parent=next(p for p in d['context']['phases'] if p['name']==s.sx_parent);k=[str(Fraction(v.strip())) for v in s.sx_k.split(',')]
                identity=self._discovery_identity();root=d['root']
                if s.sx_kind=='magnetic':catalog=await asyncio.to_thread(discover_magnetic,parent,k,api_key=os.environ.get('BCS_API_KEY',''))
                else:
                    parent_cif=root/'parent.cif'
                    await asyncio.to_thread(self._symmetry_native,'export',dict(source=str(root/'source.gpx'),phase=s.sx_parent,destination=str(parent_cif)),root)
                    discovery=root/('isodistort-'+uuid.uuid4().hex)
                    def inventory_many(children):
                        rows=[]
                        for offset in range(0,len(children),32):
                            rows.extend(self._symmetry_native('modes-many',dict(children=[str(p) for p in children[offset:offset+32]],workers=int(s.sx_slots)),root)['items'])
                        return rows
                    def prepare():
                        return discover_structural(parent_cif,parent,k,discovery,matcher=lambda cif,vector,options:self._symmetry_native('match',dict(cif=cif,k=vector,options=options),root),
                            inventory=lambda child:self._symmetry_native('modes',dict(child=str(child),temporary=str(root/(child.stem+'.gpx'))),root),
                            opd_policy=s.sx_opd_policy,maximum=int(s.sx_construct_limit),
                            inventory_many=inventory_many if self._model_remote() else None)
                    catalog=await asyncio.to_thread(prepare)
                    for candidate in catalog['candidates']:candidate['child']['path']=str((discovery/candidate['child']['path']).relative_to(root))
                if identity!=self._discovery_identity():raise ValueError('Inputs changed during discovery; discover again for the current parent and k')
                if d.get('planning'):
                    catalog.setdefault('provenance',{})['setup_planning_sha256']=digest(d['planning'])
                (root/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n');d.update(catalog=catalog,discovery_identity=identity)
                s.sx_candidate_options=[dict(title=(c['magnetic_space_group'] or c['irrep'])+' · '+c['id'][-6:],value=c['id']) for c in catalog['candidates']]
                s.sx_candidates=[c['id'] for c in catalog['candidates']]
                s.sx_mode_options=[dict(title=(c['irrep'] or c['id'])+' / '+mode,value=c['id']+'|'+mode) for c in catalog['candidates'] for mode in c['modes']]
                s.sx_modes=[m['value'] for m in s.sx_mode_options]
                coverage=catalog.get('coverage',{});s.sx_discovery_scope=(f"{coverage.get('discovered_count',len(catalog['candidates']))} OPDs discovered · {len(catalog['candidates'])} constructed in the declared tier · deferred/failed branches retained in the discovery record" if s.sx_kind=='structural' else f"{len(catalog['candidates'])} magnetic embeddings discovered in this exact parent setting")
                s.sx_discovered=True;s.sx_plan_ready=False;s.sx_approved=False
                s.sx_message=s.sx_discovery_scope+'. Review the finite plan before starting refinement.'
            except Exception as exc:s.sx_error=str(exc);s.sx_message='Candidate discovery stopped; review the error above.'
            finally:s.sx_busy=False;s.flush()
        self._refinement_schedule(discover(),'radar-symmetry-discovery')

    def _discovery_identity(self):
        s=self.server.state
        identity=(s.sx_kind,s.sx_parent,s.sx_k,s.sx_start_source)
        return identity+(s.sx_opd_policy,int(s.sx_construct_limit)) if s.sx_kind=='structural' else identity

    def _model_form(self,payload=None):
        s=self.server.state
        if isinstance(payload,dict):
            if set(payload)-set(MODEL_FIELDS)-{'sx_start_source','rf_experiment','sx_plan_token'}:raise ValueError('Unexpected model setup fields')
            if payload.get('sx_start_source',s.sx_start_source)!=s.sx_start_source or payload.get('rf_experiment',s.rf_experiment)!=s.rf_experiment:
                raise ValueError('The model source changed while the form was captured')
            for name in MODEL_FIELDS:
                if name in payload and name in {'sx_follow_new','sx_sign_starts','sx_refine_scale','sx_refine_background'} and type(payload[name]) is not bool:raise ValueError('Model switches must be true or false')
                if name in payload:setattr(s,name,payload[name])
            s.flush()
        return digest({name:getattr(s,name) for name in MODEL_FIELDS})

    def build_symmetry_plan(self,payload=None):
        s=self.server.state;s.sx_error=''
        try:
            form=self._model_form(payload)
            d=self._symmetry_draft
            if not d or d.get('discovery_identity')!=self._discovery_identity():raise ValueError('Discover candidates for the current parent and k first')
            if not s.sx_adequate or float(s.sx_adequate)<=0:raise ValueError('Enter an experiment-specific adequate Rwp (%) before building the plan')
            sites=[]
            if s.sx_kind=='magnetic':
                sites=sites_from_inputs(s.sx_sites,s.sx_form_factors,s.sx_lande)
            settings=dict(acquisition_id=d['source_id'],kind=s.sx_kind,parent=s.sx_parent,histogram=s.sx_histogram,k=[str(Fraction(v.strip())) for v in s.sx_k.split(',')],candidates=list(s.sx_candidates),modes=list(s.sx_modes),sites=sites,directions=list(s.sx_directions),adequate_rwp=float(s.sx_adequate),maximum_moment=float(s.sx_max_moment),magnetic_d_min=float(s.sx_mag_d_min),maximum_mode=float(s.sx_max_mode),start_mode=float(s.sx_start_mode),cycles=int(s.sx_cycles),relative_sign_starts=s.sx_sign_starts)
            if s.sx_kind=='magnetic':settings.update(moment_model=s.sx_moment_model,refine_histogram_scale=s.sx_refine_scale,refine_background=s.sx_refine_background)
            settings.update(selected_optimizer_settings(s.sx_kind,s.sx_magnetic_optimizer,s.sx_optimizer_iterations))
            p=build_plan(d['context'],d['catalog'],d['root'],settings);verify_inputs(p,d['root']);d.update(protocol=p,settings=settings)
            w=self._refinement_store()
            template=make_template(p,d['context'],d['source'],model_choices(w,s.rf_experiment),
                origin=model_origin(d['source'],w),existing=s.sx_existing,future=bool(s.sx_follow_new),maximum_scans=int(s.sx_max_scans))
            d.update(template=template,form_signature=form);s.sx_plan_token=digest(template)
            d['scope_sources']={v['id']:v for v,_ in model_choices(w,s.rf_experiment)}
            d['scope_origins']={v['id']:kind for v,kind in model_choices(w,s.rf_experiment)}
            magnetic_scope='Site-symmetry-allowed moment components' if s.sx_moment_model=='symmetry_allowed' else 'Selected collinear moments'
            s.sx_plan_candidates=[dict(name=c['magnetic_space_group'] or c['irrep'],bns=c['bns_number'] or '—',basis=str(c['transformation_matrix']),origin=str(c['origin_shift']),modes=', '.join(c['modes']) or magnetic_scope) for c in p['candidates']]
            free='Selected modes, cell, background and global scale' if s.sx_kind=='structural' else magnetic_scope+('; shared nuclear/magnetic histogram scale varied' if s.sx_refine_scale else '; all scales fixed')+('; existing background coefficients varied' if s.sx_refine_background else '; background fixed')+'; nuclear geometry fixed'
            if s.sx_kind=='magnetic':free+=('; all fitted d spacings' if settings['magnetic_d_min']==0 else f"; magnetic d ≥ {settings['magnetic_d_min']} Å")
            s.sx_plan_summary=f"{len(p['candidates'])} embeddings · {sum(1 for _ in iter_trials(p))} finite trials. {optimizer_budget_label(p['policy'])} {free}. Instrument, occupancy and displacement parameters fixed. Adequacy: Rwp ≤ {s.sx_adequate}%."
            s.sx_plan_summary+=f" {len(template['existing_ids'])} existing acquisitions · {trial_count(template)} starts per acquisition · maximum {template['maximum_scans']} acquisitions. Following new scans: {'on' if template['future'] else 'off'}."
            s.sx_plan_summary+=f" Existing-cohort budget: {len(template['existing_ids'])*trial_count(template):,} starts before per-scan compatibility checks."
            s.sx_plan_ready=True;s.sx_approved=False
        except Exception as exc:s.sx_error=str(exc)

    def approve_symmetry(self,payload=None):
        d=self._symmetry_draft;s=self.server.state
        if not d or not s.sx_plan_ready:return
        try:
            if isinstance(payload,dict) and payload.get('sx_plan_token')!=digest(d['template']):raise ValueError('Review the plan currently shown before approving')
            if self._model_form(payload)!=d.get('form_signature'):raise ValueError('Model settings changed; build and review the current plan again')
            self._symmetry_approved=digest(d['template']);s.sx_approved=True
        except Exception as exc:s.sx_error=str(exc);self._symmetry_unapprove()

    def start_symmetry_campaign(self,payload=None):
        if self._symmetry_task and not self._symmetry_task.done():return
        self._model_stopping=False
        cancelled=threading.Event()
        self._model_start_cancel=cancelled
        async def run():
            s=self.server.state;s.sx_error='';s.sx_busy=True
            s.sx_message='Preparing the approved campaign: checking the GPX revision and publishing shared inputs to Galaxy.';s.flush()
            staged=None;driver_entered=False
            try:
                draft=self._symmetry_draft
                if isinstance(payload,dict) and draft and payload.get('sx_plan_token')!=digest(draft['template']):raise ValueError('The visible plan changed before Start')
                if not draft or not s.sx_plan_ready or not s.sx_approved or self._symmetry_approved!=digest(draft['template']) or self._model_form(payload)!=draft.get('form_signature'):raise ValueError('Approve the current finite plan first')
                d=copy.deepcopy(draft)
                if d['source_id']!=s.sx_start_source:raise ValueError('The source changed after approval')
                workspace=self._refinement_store();service=self.service
                history=getattr(service,'history_id',None);experiment=s.rf_experiment
                approved=self._symmetry_approved;root=self._symmetry_root()
                def require_current():
                    if self._model_stopping or cancelled.is_set():raise ValueError('Session paused or stopped before the model campaign could start')
                    if (self._model_start_cancel is not cancelled or self.service is not service
                            or getattr(service,'history_id',None)!=history or s.rf_experiment!=experiment
                            or s.sx_start_source!=d['source_id'] or self._refinement_store() is not workspace
                            or self._symmetry_draft is not draft or not s.sx_plan_ready or not s.sx_approved
                            or self._symmetry_approved!=approved or digest(draft['template'])!=approved
                            or draft.get('protocol')!=d['protocol']
                            or self._model_form(payload)!=d.get('form_signature')):
                        raise ValueError('The approved model scope or source context changed during preparation; review it before starting')
                current=await asyncio.to_thread(resolve_start,workspace,experiment,d['source_id'])
                current=copy.deepcopy(await asyncio.to_thread(workspace.runtime.resolve,current))
                require_current()
                if current.get('source_sha256') and current['source_sha256']!=d['protocol']['source']['sha256']:raise ValueError('The published GPX revision changed after approval; inspect it again')
                c=Campaign(root/('campaign-'+approved[:12]),shared_inputs=True)
                if c.trials():raise ValueError('This campaign already exists. Use Continue to recover its work.')
                marker=dict(schema=START_BARRIER_SCHEMA,attempt=uuid.uuid4().hex,
                    template_sha256=approved,source_sha256=current.get('source_sha256'))
                store_start_barrier(c,marker)
                staged=(c,marker)
                transport=self._symmetry_transport(c)
                async def stage():
                    ref=await asyncio.to_thread(prepare_shared_inputs,d['protocol'],d['root'],c.root)
                    require_current()
                    c.set_setting('template',d['template']);c.set_setting('reference_input_root',str(ref))
                    c.set_setting('enrollments',{key:dict(id=key,status='queued',
                        scan=d['scope_sources'][key]['scan_label'],sample=d['scope_sources'][key]['sample_label'],
                        bank=d['scope_sources'][key].get('bank'),source=portable_source(d['scope_sources'][key]),
                        origin=d['scope_origins'][key]) for key in d['template']['existing_ids']})
                    if isinstance(transport,GalaxyChunkTransport):
                        # Preserve distinct reference/rebound packets and their
                        # existing acknowledged-HDA/dependency readiness rules.
                        await asyncio.to_thread(transport.prepare_source,d['protocol'],ref,current)
                # Staging cannot admit trials or submit science. It may overlap
                # the source backup, but both existing operations must succeed
                # before the original enrollment/claim/dispatch sequence starts.
                await self._model_start_barrier(asyncio.to_thread(workspace.checkpoint),stage(),on_cancel=cancelled.set)
                require_current()
                # Keep the captured entries unarmed through the final await.
                # The first active driver tick may then discover future scans.
                await asyncio.to_thread(queue_sources,c,workspace)
                require_current()
                if not store_start_barrier(c,marker,complete=True,following=d['template']['future'],current_only=True):
                    raise ValueError('Another Start replaced this preparation; review the current campaign')
                s.sx_campaign=c.root.name;s.sx_setup=False
                driver_entered=True
                await self._drive_symmetry_campaign(c,transport)
            except Exception as exc:s.sx_error=str(exc);s.sx_message='Campaign stopped; review the error above.'
            finally:
                try:
                    if staged and not driver_entered:
                        store_start_barrier(*staged,current_only=True)
                finally:
                    if self._model_start_cancel is cancelled:self._model_start_cancel=None
                    s.sx_busy=False;s.flush()
        work=run()
        try:self._symmetry_task=self._refinement_schedule(work,'radar-symmetry-campaign')
        except BaseException:
            cancelled.set();work.close()
            if self._model_start_cancel is cancelled:self._model_start_cancel=None
            raise
        if self._symmetry_task is None:
            cancelled.set();work.close()
            if self._model_start_cancel is cancelled:self._model_start_cancel=None

    async def _model_start_barrier(self,checkpoint,staging,*,on_cancel):
        """Settle both owned operations, even when their caller is cancelled."""
        pending=asyncio.gather(checkpoint,staging,return_exceptions=True)
        cancelled=False
        while True:
            try:
                results=await asyncio.shield(pending)
                break
            except asyncio.CancelledError:
                cancelled=True;on_cancel()
        if cancelled:raise asyncio.CancelledError
        for result in results:
            if isinstance(result,BaseException):raise result

    async def _await_model_checkpoint_work(self,task,campaign,transport):
        """Keep collecting accepted work while a durable upload/POST settles.

        This pass cannot dispatch. A slow recovery checkpoint must hold new
        admission, rather than hiding already finished scientific results.
        The shielded owner still settles before cancellation can abort a wave.
        """
        while True:
            done,_=await asyncio.wait({task},timeout=getattr(self,'_model_checkpoint_poll_seconds',3.))
            if done:return await asyncio.shield(task)
            await asyncio.to_thread(campaign.tick,CollectOnlyTransport(transport),slots=int(self.server.state.sx_slots))
            self.refresh_symmetry()

    def _model_checkpoint_pending(self,campaign,error):
        self._model_checkpoint_errors[campaign.root.name]=error
        notice=(f'Galaxy restart interrupted recovery checkpoint {error.dataset_id}. '
            'Its failed zero-byte backup will be retried. Accepted jobs continue; new batches wait for a ready checkpoint.'
            if isinstance(error,CheckpointUploadInterrupted) else
            f'Recovery checkpoint {error.dataset_id} is still {error.state}. '
            'Accepted jobs continue and their results are collected. New batches wait for the existing upload; no replacement is posted while it is pending.')
        campaign.set_setting('checkpoint_notice',notice)
        self.refresh_symmetry()

    async def _flush_model_dispatch_wave(self,campaign,transport):
        """Retry one frozen intent, without another scheduler admission pass."""
        def allowed():
            return not campaign.setting('paused',False) and transport.dispatch_guard()
        while True:
            if not allowed():
                campaign.set_setting('paused',True)
                transport.pause()
                transport.abort_dispatch_wave()
                return
            self._model_tick_task=asyncio.create_task(asyncio.to_thread(transport.flush_dispatch_wave))
            try:
                await self._await_model_checkpoint_work(self._model_tick_task,campaign,transport)
                return
            except CheckpointUploadPending as exc:
                self._model_checkpoint_pending(campaign,exc)
                if not allowed():
                    campaign.set_setting('paused',True)
                    transport.pause()
                    transport.abort_dispatch_wave()
                    return
                await asyncio.sleep(3)

    async def _drive_symmetry_campaign(self,campaign,transport):
        if not start_barrier_ready(campaign):raise ValueError(START_BARRIER_MESSAGE)
        preparation=None
        campaign.set_setting('viewer_parallel_jobs',self.server.state.sx_parallel_jobs)
        owner=(campaign,transport)
        self._model_active_campaigns[campaign.root.name]=owner
        if isinstance(transport,GalaxyChunkTransport):
            approved_template=digest(campaign.setting('template'))
            transport.dispatch_guard=lambda: (not self._model_stopping
                and self._model_active_campaigns.get(campaign.root.name) is owner
                and digest(campaign.setting('template'))==approved_template)
            if not hasattr(self,'_model_owned_campaigns'):self._model_owned_campaigns={}
            self._model_owned_campaigns[campaign.root.name]=(campaign,transport)
            getattr(self,'_model_review_campaigns',{}).pop(campaign.root,None)
        if not hasattr(self,'_model_checkpoint_errors'):self._model_checkpoint_errors={}
        self._model_checkpoint_errors.pop(campaign.root.name,None)
        self.server.state.sx_busy=False;self.server.state.sx_campaign_running=True
        self.server.state.sx_message='Campaign running. Each acquisition is checked before its finite starts enter the Galaxy resource queue.'
        self.server.state.flush()
        try:
            while True:
                if self._model_stopping:
                    campaign.set_setting('paused',True)
                    if isinstance(transport,GalaxyChunkTransport):transport.pause()
                following=bool(campaign.setting('following',False)) and not campaign.setting('paused',False)
                self._model_following_active=following
                tick_transport=transport
                if campaign.setting('template'):
                    if preparation and preparation.done():
                        await asyncio.shield(preparation)
                        preparation=None;self._model_enrollment_task=None
                        campaign.set_setting('preparation_message','')
                    entries=await asyncio.to_thread(queue_sources,campaign,self._refinement_store())
                    pending=next_acquisition(campaign,entries)
                    if pending and not preparation and not campaign.setting('paused',False):
                        # Collect first. A normal tick can refill a newly freed
                        # slot before the next acquisition ever gets a turn.
                        await asyncio.to_thread(campaign.tick,CollectOnlyTransport(transport),slots=int(self.server.state.sx_slots))
                    active=active_batch_count(campaign.trials())
                    if not preparation and active<getattr(transport,'max_active_jobs',1) and not campaign.setting('paused',False):
                        if pending:
                            message='Preparing '+pending['scan']+': checking its GPX and declaring the reviewed finite model starts.'
                            campaign.set_setting('preparation_message',message)
                            self.server.state.sx_queue_notice=message
                            self.server.state.flush()
                            preparation=asyncio.create_task(asyncio.to_thread(enroll_source,campaign,self._refinement_store(),pending['id'],
                                self._inspect_model_input,transport=transport if isinstance(transport,GalaxyChunkTransport) else None))
                            self._model_enrollment_task=preparation
                            if not hasattr(transport,'submit_many'):
                                # Preserve the local worker's serial preparation
                                # policy; Galaxy has a separate bounded job slot.
                                await asyncio.shield(preparation)
                                preparation=None;self._model_enrollment_task=None
                                campaign.set_setting('preparation_message','')
                    elif pending and not preparation and not campaign.setting('paused',False):
                        tick_transport=CollectOnlyTransport(transport)
                        campaign.set_setting('preparation_message','Waiting for an accepted Galaxy batch to finish before preparing '+pending['scan']+'.')
                    if preparation:
                        # Native inspection can wait on Galaxy while existing
                        # scans keep refining. Reserve its admission slot so
                        # refilling old scans cannot starve the next one.
                        tick_transport=PreparationSlotTransport(transport)
                # One core scheduler tick admits one finite chunk. Collect a
                # bounded wave, then persist all finite member lists together
                # before its first native POST. Each POST keeps its stable key.
                wave=max(1,getattr(tick_transport,'max_active_jobs',1))
                buffered=isinstance(transport,GalaxyChunkTransport)
                if buffered:transport.begin_dispatch_wave()
                checkpoint_pending=False
                try:
                    for _ in range(wave):
                        self._model_tick_task=asyncio.create_task(asyncio.to_thread(campaign.tick,tick_transport,slots=int(self.server.state.sx_slots)))
                        trials=await asyncio.shield(self._model_tick_task)
                        self.refresh_symmetry()
                        if (campaign.setting('paused',False)
                                or active_batch_count(trials)>=getattr(tick_transport,'max_active_jobs',1)
                                or not any(r['state']=='prepared'for r in trials)):
                            break
                    if buffered:
                        campaign.set_setting('dispatch_notice','Saving queued batch recovery state before submitting new Galaxy jobs.')
                        self.refresh_symmetry()
                        await self._flush_model_dispatch_wave(campaign,transport)
                except (CheckpointUploadPending,CheckpointUploadInterrupted) as exc:
                    # The inner flush retries its frozen pending intent. A
                    # confirmed interrupted upload, or pending work encountered
                    # outside that flush, still leaves this wave unposted.
                    if buffered:transport.abort_dispatch_wave()
                    self._model_checkpoint_pending(campaign,exc)
                    checkpoint_pending=True
                except BaseException:
                    if buffered:
                        # Shielded thread work can outlive cancellation of its
                        # caller. Stop admission and let that exact operation
                        # settle before marking any remaining intents unposted.
                        transport.pause()
                        task=self._model_tick_task
                        if task and not task.done():
                            try:await asyncio.shield(task)
                            except BaseException:pass
                        if not task or task.done():transport.abort_dispatch_wave()
                    raise
                finally:
                    if buffered:campaign.set_setting('dispatch_notice','')
                if checkpoint_pending:
                    if campaign.setting('paused',False):break
                    await asyncio.sleep(3)
                    continue
                if isinstance(transport,GalaxyChunkTransport):
                    try:
                        self._model_tick_task=asyncio.create_task(asyncio.to_thread(transport.checkpoint))
                        await self._await_model_checkpoint_work(self._model_tick_task,campaign,transport)
                    except (CheckpointUploadPending,CheckpointUploadInterrupted) as exc:self._model_checkpoint_pending(campaign,exc)
                    else:
                        self._model_checkpoint_errors.pop(campaign.root.name,None)
                        campaign.set_setting('checkpoint_notice','')
                self.refresh_symmetry()
                queued=any(r['status'] in {'queued','preparing'} for r in campaign.setting('enrollments',{}).values())
                pending=any(r['state'] not in {'terminal','cancelled','needs_recovery'} for r in trials)
                active=any(r['state'] in {'running','submitting','cancelling'} for r in trials)
                if campaign.setting('paused',False):break
                # An inspection can commit new trials after this tick's local
                # snapshot. Consume that task and poll the new ledger on the
                # next pass before declaring the finite campaign complete.
                if not pending and not queued and not following and preparation is None:break
                await asyncio.sleep(3)
        except Exception as exc:
            self._model_checkpoint_errors[campaign.root.name]=exc
            raise
        finally:
            try:
                if preparation:
                    for key,event in getattr(self,'_model_preparations',{}).items():
                        if Path(key).is_relative_to(campaign.root):event.set()
                    await asyncio.shield(preparation)
            finally:
                self._model_enrollment_task=None
                self._model_following_active=False
                self.server.state.sx_campaign_running=False
                if self._model_active_campaigns.get(campaign.root.name) is owner:
                    self._model_active_campaigns.pop(campaign.root.name,None)

    def continue_symmetry_campaign(self):
        if self._symmetry_task and not self._symmetry_task.done():return
        self._model_stopping=False
        async def resume():
            s=self.server.state;s.sx_error='';s.sx_busy=True
            s.sx_message='Recovering accepted Galaxy jobs and verifying saved results before dispatching any new starts.'
            s.sx_queue_notice=''
            s.flush()
            try:
                root=(self._symmetry_root()/s.sx_campaign).resolve()
                if root.parent!=self._symmetry_root() or not (root/'campaign.sqlite').is_file():raise ValueError('Select an existing campaign')
                campaign=Campaign(root,shared_inputs=True)
                if not start_barrier_ready(campaign):raise ValueError(START_BARRIER_MESSAGE)
                saved_budget=campaign.setting('viewer_parallel_jobs')
                if type(saved_budget) is int and saved_budget in MODEL_BATCH_LIMITS:
                    s.sx_parallel_jobs=saved_budget
                await asyncio.to_thread(restore_captured_sources,campaign,self._refinement_store())
                transport=self._symmetry_transport(campaign)
                await asyncio.to_thread(campaign.recover,transport)
                pending=any(r['state'] not in {'terminal','cancelled'} for r in campaign.trials())
                if pending:
                    from .model_ledger_recovery import verify_continuation
                    loop=asyncio.get_running_loop()
                    identity=(s.sx_campaign,s.rf_experiment)
                    def publish_verification(done,total, request_identity=identity):
                        if request_identity==(s.sx_campaign,s.rf_experiment):
                            s.sx_message=f'Verifying sealed starts: {done:,} of {total:,}. No new refinements are being submitted.'
                            s.flush()
                    def verification_progress(done,total):loop.call_soon_threadsafe(publish_verification,done,total)
                    await asyncio.to_thread(verify_continuation,campaign,progress=verification_progress)
                template=campaign.setting('template')
                if template and template['protocol']['implementation_sha256']!=package_signature():
                    raise ValueError('Use the original worker release to continue this template, or review a new campaign. Saved results remain viewable.')
                campaign.set_setting('paused',False)
                for key in list(self._model_preparations):
                    if Path(key).is_relative_to(campaign.root):self._model_preparations.pop(key)
                remote=self._model_remote()
                if remote:remote.allow_preparations(campaign.root)
                if isinstance(transport,GalaxyChunkTransport):transport.resume()
                await self._drive_symmetry_campaign(campaign,transport)
            except Exception as exc:s.sx_error=str(exc)
            finally:s.sx_busy=False;s.flush()
        self._symmetry_task=self._refinement_schedule(resume(),'radar-symmetry-continue')

    def prioritize_symmetry_acquisition(self):
        s=self.server.state
        try:
            root=(self._symmetry_root()/s.sx_campaign).resolve()
            if root.parent!=self._symmetry_root() or not (root/'campaign.sqlite').is_file():
                raise ValueError('Select an existing campaign')
            campaign=self._symmetry_campaign(root)
            scope=(campaign.setting('template') or {}).get('scope',{})
            if scope.get('experiment')!=s.rf_experiment:
                raise ValueError('Choose the experiment belonging to this campaign')
            entry=prioritize_acquisition(campaign,s.sx_priority_acquisition)
            s.sx_error=''
            s.sx_priority_notice='Priority: '+entry['scan']+'. Already accepted work finishes normally; the remaining captured scans stay queued.'
        except (OSError,ValueError,TypeError) as exc:s.sx_error=str(exc)

    def cancel_symmetry_selected(self):
        async def cancel():
            try:
                s=self.server.state;c=Campaign(self._symmetry_root()/s.sx_campaign)
                await asyncio.to_thread(c.cancel,s.sx_selected,self._symmetry_transport(c));self.refresh_symmetry()
            except Exception as exc:self.server.state.sx_error=str(exc)
        self._refinement_schedule(cancel(),'radar-symmetry-cancel')

    def pause_model_campaign(self):
        pending=getattr(self,'_model_pause_task',None)
        if pending and not pending.done():return
        starting=getattr(self,'_model_start_cancel',None)
        if starting is not None:starting.set()
        s=self.server.state
        s.sx_pause_pending=True
        s.sx_queue_notice='Pausing new submissions and saving recovery state. Already accepted native jobs are not cancelled.'
        s.flush()
        async def pause():
            try:
                for campaign,transport in list(self._model_active_campaigns.values()):
                    campaign.set_setting('paused',True)
                    if isinstance(transport,GalaxyChunkTransport):transport.pause()
                    if isinstance(transport,GalaxyChunkTransport):await asyncio.to_thread(transport.checkpoint,force=True)
            except Exception as exc:
                if not hasattr(self,'_model_checkpoint_errors'):self._model_checkpoint_errors={}
                self._model_checkpoint_errors[campaign.root.name]=exc
                s.sx_error=str(exc)
            finally:
                s.sx_pause_pending=False
                self.refresh_symmetry()
                s.flush()
        work=pause()
        try:
            self._model_pause_task=self._refinement_schedule(work,'radar-model-pause')
            if self._model_pause_task is None:
                work.close();s.sx_pause_pending=False;s.flush()
        except Exception:
            work.close();s.sx_pause_pending=False;s.flush()
            raise

    async def _pause_model_campaigns(self):
        """Stop new dispatch, preserve Galaxy jobs and await captured enrollment."""
        self._model_stopping=True
        for event in self._model_preparations.values():event.set()
        if self._k_cancel:self._k_cancel.set()
        remote=self._model_remote()
        if remote:remote.stop_preparations()
        known={**getattr(self,'_model_owned_campaigns',{}),**self._model_active_campaigns}
        active=list(known.values())
        prior_errors=dict(getattr(self,'_model_checkpoint_errors',{}))
        for campaign,transport in active:
            campaign.set_setting('paused',True)
            if isinstance(transport,GalaxyChunkTransport):transport.pause()
        if self._model_enrollment_task and not self._model_enrollment_task.done():await asyncio.shield(self._model_enrollment_task)
        if self._model_tick_task and not self._model_tick_task.done():await asyncio.shield(self._model_tick_task)
        pause_task=getattr(self,'_model_pause_task',None)
        if pause_task and not pause_task.done():await asyncio.shield(pause_task)
        if self._symmetry_task and not self._symmetry_task.done() and self._symmetry_task is not asyncio.current_task():
            await asyncio.shield(self._symmetry_task)
        for campaign,transport in active:
            if isinstance(transport,GalaxyChunkTransport):
                from .checkpoint_uploads import CheckpointUploadPending
                error=getattr(self,'_model_checkpoint_errors',{}).get(campaign.root.name)
                if isinstance(error,CheckpointUploadPending) and error is not prior_errors.get(campaign.root.name):
                    raise error
                await asyncio.to_thread(transport.checkpoint,force=True)
                getattr(self,'_model_checkpoint_errors',{}).pop(campaign.root.name,None)
            else:
                for row in campaign.trials():
                    if row['state'] in {'running','submitting','cancelling'}:await asyncio.to_thread(campaign.cancel,row['id'],transport)
        self._model_following_active=False

    def _reset_model_view(self):
        from copy import deepcopy
        for name,value in self._symmetry_defaults.items():setattr(self.server.state,name,deepcopy(value))
        self._symmetry_rows=[];self._symmetry_all_rows=[];self._symmetry_groups=[];self._symmetry_draft=None;self._symmetry_approved=''
        self._model_runtime_cache=None;self._models_restored=False;self._model_restored_scopes=set();self._model_restore_requests=set();self._model_active_campaigns={};self._model_owned_campaigns={};self._model_review_campaigns=OrderedDict();self._model_checkpoint_errors={};self._model_pause_task=None;self._model_start_cancel=None
        self._k_result=None;self._k_result_identity=None
        s=self.server.state;s.sx_k_rows=[];s.sx_k_choices=[];s.sx_k_notes=[];s.sx_k_has_result=False;s.sx_k_busy=False
        s.sx_k_plot=plots.empty('Run a k survey to compare magnetic templates')
        from .symmetry_workspace import clear_evidence_cache
        clear_evidence_cache()
