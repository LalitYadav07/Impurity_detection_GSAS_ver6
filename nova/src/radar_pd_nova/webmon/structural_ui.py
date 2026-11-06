"""Structural point-fit controls in the existing Live acquisition workspace."""
import asyncio
import copy
import json
from pathlib import Path
import uuid

from trame.widgets import html,plotly
from trame.widgets import vuetify3 as v
import plotly.graph_objects as go

from gpx_recipe.execution import write_json
from gpx_recipe.symmetry.compiler import compile_shared_job,prepare_shared_inputs,file_sha
from gpx_recipe.symmetry.contracts import digest,iter_trials,trial_id
from ..symmetry_workspace import candidate_display_name
from .. import symmetry_plots
from .scope import ScopeCancellation
from .stream import LatestQueue,scope_of
from .structural_stream import StructuralObservationStream,analyse_snapshot
from .structural_runtime import StructuralSnapshotRuntime


class StructuralLiveMixin:
    def _init_structural_live(self):
        self._webmon_structural_plan=None;self._webmon_structural_results={}
        self._webmon_display_kind='magnetic';self._webmon_kind_headers={}
        self.server.state.update(dict(webmon_analysis_kind='magnetic',webmon_structural_ready=False,
            webmon_structural_recovery=False,
            webmon_structural_summary='',webmon_structural_message='Use a reviewed structural plan from Model exploration.',
            webmon_structural_rows=[],webmon_structural_choices=[],webmon_structural_selected='',
            webmon_structural_axis='d',webmon_structural_scale='linear',webmon_structural_has_result=False,
            webmon_structural_parameters=[],webmon_structural_reset=0))

    def _clear_structural_live(self):
        self._webmon_structural_plan=None;self._webmon_structural_results={}
        s=self.server.state
        for name,value in dict(webmon_structural_ready=False,webmon_structural_summary='',webmon_structural_rows=[],
                webmon_structural_recovery=False,
                webmon_structural_choices=[],webmon_structural_selected='',webmon_structural_has_result=False,
                webmon_structural_parameters=[],
                webmon_structural_message='Review a structural plan for this acquisition.').items():setattr(s,name,value)
        if getattr(self,'_webmon_structural_fit',None):self._webmon_structural_fit.update(go.Figure(symmetry_plots.empty('No structural fit selected')))

    def webmon_analysis_kind_changed(self,value,**_):
        if value not in {'magnetic','structural'}:return
        s=self.server.state;old=getattr(self,'_webmon_display_kind','magnetic')
        if s.webmon_busy:s.webmon_analysis_kind=old;return
        self.webmon_pause_stream()
        fields=('webmon_message','webmon_calculation_status','webmon_count_summary','webmon_validity',
            'webmon_target','webmon_analysis_revision','webmon_stale')
        self._webmon_kind_headers[old]={key:getattr(s,key,None) for key in fields}
        defaults=dict(webmon_message='',webmon_calculation_status='Not started',webmon_count_summary='No models analyzed',
            webmon_validity='Structural point fits · review required' if value=='structural' else 'No fit assessed',
            webmon_target='',webmon_analysis_revision='',webmon_stale=False)
        for key,field in self._webmon_kind_headers.get(value,defaults).items():
            if field is not None:setattr(s,key,field)
        self._webmon_display_kind=value;s.webmon_analysis_kind=value;s.flush()

    def webmon_open_structural_setup(self,**_):
        s=self.server.state
        try:
            source=getattr(self,'_webmon_published_catalog',{}).get(s.webmon_published_source)
            if not source:raise ValueError('Select a published nuclear GPX first')
            self._webmon_require_reference(source)
            self._webmon_handoff_source(source,'Prepare a structural distortion model search for live snapshots. Review this nuclear parent, optional parent reference, k and complete displacement-mode children. Build and approve the finite plan, then return to Live acquisition; do not start a completed-scan campaign.')
            s.rf_mode='models';s.sx_kind='structural';s.sx_start_source=source['id'];s.sx_source=source['id']
            s.sx_follow_new=False;s.sx_existing='selected';self.inspect_symmetry_source()
        except Exception as exc:s.webmon_structural_message=str(exc)
        s.flush()

    async def webmon_adopt_structural_plan(self,**_):
        s=self.server.state
        if s.webmon_busy:return
        if getattr(s,'webmon_structural_recovery',False):
            s.webmon_structural_message='Recover the saved structural jobs before adopting another plan';s.flush();return
        token=object();self._webmon_structural_adoption=token
        owner=None
        try:
            owner=self._webmon_scope_snapshot();self._webmon_require_target()
            source=getattr(self,'_webmon_published_catalog',{}).get(s.webmon_published_source)
            draft=getattr(self,'_symmetry_draft',None);mapping=getattr(self,'_webmon_reviewed_mapping',None)
            if (not source or not draft or draft.get('source_id')!=source['id'] or s.sx_kind!='structural'
                    or draft.get('discovery_identity')!=self._discovery_identity()
                    or not s.sx_approved or self._symmetry_approved!=digest(draft['template'])
                    or self._model_form()!=draft.get('form_signature')):
                raise ValueError('Build and approve the current structural plan for this exact GPX in Model exploration first')
            p=copy.deepcopy(draft['protocol']);form=draft['form_signature'];source=copy.deepcopy(source)
            pairs=list(iter_trials(p))
            if len(pairs)!=len(p['candidates']) or len({c['id'] for c,_ in pairs})!=len(pairs):
                raise ValueError('Live structural models require the complete-mode plan with its internal starting portfolio')
            row,receipt,_=self._webmon.capture(self._webmon_effective_ipts(),s.webmon_run)
            self._webmon_require_reference(source,row)
            if (not mapping or mapping['source_id']!=source['id'] or mapping['snapshot_sha256']!=receipt['content_sha256']
                    or mapping['source_gpx_sha256']!=p['source']['sha256']):
                raise ValueError('Save the reviewed producer mapping for this source and current snapshot first')
            runtime=self._model_remote()
            if runtime is None:raise ValueError('Structural live fits need the configured Galaxy Model Campaign worker')
            root=Path(self._webmon.config['jobs'])/'structural'/uuid.uuid4().hex;root.mkdir(parents=True)
            anchor=root/'anchor.dat';anchor.write_bytes(self._webmon.archive.read(receipt))
            labels={c['id']:candidate_display_name(draft['catalog'],c) for c in p['candidates']}
            refine_transfer=s.webmon_refine_transfer
            if type(refine_transfer) is not bool:raise ValueError('Snapshot scale/background transfer must be explicitly on or off')
            s.webmon_busy=True;s.webmon_structural_message='Capturing the reviewed structural portfolio; no refinement has started.';s.flush()
            def capture():
                shared=prepare_shared_inputs(p,draft['root'],root/'models')
                self._capture_prepared_model_inputs(p,shared)
                templates=[]
                for candidate,seed in pairs:
                    directory=root/'models/trials'/trial_id(p,candidate['id'],seed['id'])
                    compile_shared_job(p,directory,candidate_id=candidate['id'],seed_id=seed['id'],approved_hash=digest(p))
                    templates.append(dict(source=str(shared/p['source']['path']),directory=str(directory),
                        anchor_snapshot=str(anchor),snapshot=str(anchor),receipt=copy.deepcopy(receipt),mapping=copy.deepcopy(mapping),
                        source_scope=scope_of(row),revision=0,source_revision='PG3_'+str(row['run'])+':'+receipt['content_sha256'],
                        reference_revision='GPX:'+p['source']['sha256'],refine_transfer=refine_transfer,
                        destination=str(root/'unstarted')))
                return templates
            templates=await asyncio.to_thread(capture)
            material=await asyncio.to_thread(runtime.resolve,source)
            if (not self._webmon_scope_current(owner) or self._model_form()!=form
                    or s.webmon_published_source!=source['id'] or s.webmon_run!=str(row['run'])
                    or s.webmon_refine_transfer!=refine_transfer
                    or getattr(self,'_webmon_reviewed_mapping',None)!=mapping):
                raise ValueError('Live source or model review changed during capture; adopt the current plan again')
            if material['source_sha256']!=p['source']['sha256']:raise ValueError('The published source revision changed')
            self._webmon_structural_plan=dict(root=root,templates=templates,source=material,labels=labels,
                run=str(row['run']),owner=owner,protocol_sha256=digest(p),source_record=source,runtime=runtime)
            write_json(root/'review.json',dict(protocol_sha256=digest(p),templates=templates,
                source={k:material[k] for k in ('id','source_sha256')},labels=labels,
                reviewed_scope=scope_of(row),forecast_enabled=False))
            s.webmon_structural_ready=True
            s.webmon_structural_summary=(f"{len(p['candidates'])} reviewed models · all physical modes and independent cell coordinates · "
                f"{p['policy']['structural_starts']} starts per model · {p['policy']['structural_seconds']} s optimization per model. "
                'The instrument, occupancies and ADPs stay fixed. Every snapshot replaces the likelihood.')
            s.webmon_structural_message='Ready. Fit the current snapshot once or follow new snapshots for this run. Original RADAR-PD results remain in Results.'
        except Exception as exc:
            if owner is None or self._webmon_scope_current(owner):
                self._webmon_structural_plan=None;s.webmon_structural_ready=False;s.webmon_structural_message=str(exc)
        finally:
            if getattr(self,'_webmon_structural_adoption',None) is token and (owner is None or self._webmon_scope_current(owner)):
                s.webmon_busy=False;s.flush()

    def _webmon_reference_controls(self):
        v.VBtn('Load published GPX sources',click=self.webmon_load_published_sources,disabled=('webmon_busy',),variant='outlined')
        v.VSelect(label='Published source GPX',v_model=('webmon_published_source',),items=('webmon_published_sources',),update_modelValue=(self.webmon_published_source_changed,'[$event]'),disabled=('webmon_busy',),density='compact')
        html.P('{{ webmon_source_notice }}')
        v.VBtn('Inspect selected GPX in Refinement',click=self.webmon_open_published_source,disabled=('!webmon_published_source || webmon_busy',),variant='text')
        with html.Details():
            html.Summary('Reviewed producer mapping')
            html.P('Check the physical bank, export conversion and normalization against the producer configuration. The GPX instrument DIFC does not by itself establish the live export conversion.')
            for label,name in [('Physical bank','webmon_mapping_bank'),('Export TOF/d factor (µs/Å)','webmon_mapping_factor'),
                    ('Normalization convention','webmon_mapping_normalization'),('Producer configuration SHA-256','webmon_mapping_producer'),
                    ('Where this mapping was verified','webmon_mapping_evidence')]:
                v.VTextField(label=label,v_model=(name,),disabled=('webmon_busy',),density='compact')
            v.VCheckbox(label='I checked this bank, conversion and normalization against the producer configuration',v_model=('webmon_mapping_accepted',),disabled=('webmon_busy',),density='compact')
            v.VBtn('Save reviewed mapping',click=self.webmon_save_mapping,disabled=('!webmon_published_source || !webmon_mapping_accepted || webmon_busy',),variant='outlined')

    def _structural_live_ui(self):
        with html.Div(v_if="webmon_analysis_kind === 'structural'",classes='wm-card mt-4'):
            html.H3('Structural live fitting')
            self._webmon_reference_controls()
            with html.Div(classes='wm-toolbar'):
                v.VBtn('Set structural models in Refinement',click=self.webmon_open_structural_setup,disabled=('webmon_busy || !webmon_published_source',),variant='outlined')
                v.VBtn('Use approved structural plan',click=self.webmon_adopt_structural_plan,disabled=('webmon_busy || webmon_structural_recovery || !webmon_published_source',),variant='outlined')
            html.P('In Model exploration: inspect the parent, discover structural children, build and approve the finite plan. Return here without starting a completed-scan campaign.')
            v.VCheckbox(label='Refit existing scales/background during snapshot preparation',v_model=('webmon_refine_transfer',),
                update_modelValue=(self.webmon_structural_transfer_changed,'[$event]'),disabled=('webmon_busy',),density='compact')
            html.P('{{ webmon_structural_summary }}')
            with html.Div(classes='wm-toolbar'):
                v.VBtn('Fit current structural snapshot',click=self.webmon_start_structural,disabled=('webmon_busy || !webmon_structural_ready',),color='#15543c')
                v.VBtn('Follow structural updates for this run',click=self.webmon_follow_structural,disabled=('webmon_busy || webmon_stream_active || !webmon_structural_ready',),variant='outlined')
                v.VBtn('Pause automatic analysis',click=self.webmon_pause_stream,disabled=('!webmon_stream_active',),variant='text')
                v.VBtn('Stop waiting for jobs',click=self.webmon_cancel,disabled=('!webmon_busy',),variant='text')
                v.VBtn('Recover structural jobs',click=self.webmon_recover_structural,disabled=('webmon_busy || !webmon_structural_recovery',),variant='outlined')
            html.P('{{ webmon_structural_message }}')
            html.P('{{ webmon_stream_message }}',v_if='webmon_stream_active')
            html.P('Point fits require optimization and scientific review. A current Rwp leader is not a uniquely identified structure; no structural posterior or counting-stop recommendation is produced.')
            v.VDataTable(headers=([dict(title='Model',key='name'),dict(title='State',key='state'),dict(title='Rwp (%)',key='rwp'),dict(title='Revision',key='revision')],),items=('webmon_structural_rows',),items_per_page=8,density='compact')
            with html.Div(v_if='webmon_structural_has_result'):
                with html.Div(classes='wm-toolbar'):
                    v.VSelect(label='Inspect structural fit',v_model=('webmon_structural_selected',),items=('webmon_structural_choices',),update_modelValue=(self.webmon_render_structural,'[$event]'),density='compact')
                    v.VSelect(label='Coordinate',v_model=('webmon_structural_axis',),items=([dict(title='d (Å)',value='d'),dict(title='Q (Å⁻¹)',value='q'),dict(title='TOF (µs)',value='native')],),update_modelValue=(self.webmon_structural_control,"['webmon_structural_axis', $event]"),density='compact')
                    v.VSelect(label='Intensity',v_model=('webmon_structural_scale',),items=(['linear','log'],),update_modelValue=(self.webmon_structural_control,"['webmon_structural_scale', $event]"),density='compact')
                    v.VBtn('Reset zoom',click=self.webmon_reset_structural,variant='text')
                    v.VBtn('Download continuation GPX',click=self.webmon_download_structural,variant='outlined')
                self._webmon_structural_fit=plotly.Figure(display_mode_bar=True)
                with html.Details():
                    html.Summary('Fitted cell and mode amplitudes')
                    html.P('Point estimates follow the captured GSAS-II/ISODISTORT mode definitions. No structural covariance or posterior uncertainty is claimed.')
                    v.VDataTable(headers=([dict(title='Phase',key='phase'),dict(title='Parameter',key='parameter'),dict(title='Value',key='value'),dict(title='Convention',key='unit')],),items=('webmon_structural_parameters',),density='compact',items_per_page=10)

    def _new_structural_stream(self):
        plan=self._webmon_structural_plan
        if not plan or not self._webmon_scope_current(plan['owner']):raise ValueError('Review the structural plan for the current saved Live scope')
        if str(self.server.state.webmon_run)!=plan['run']:raise ValueError('Return to the reviewed live run before starting structural fits')
        if (self.server.state.webmon_refine_transfer!=plan['templates'][0]['refine_transfer']
                or getattr(self,'_webmon_reviewed_mapping',None)!=plan['templates'][0]['mapping']):
            self.server.state.webmon_structural_ready=False
            self.webmon_pause_stream()
            raise ValueError('Snapshot transfer settings or producer mapping changed; use the approved structural plan again before fitting')
        # Starting again is an explicit review action. Inputs stay shared and
        # previous native dispatch receipts/results remain immutable.
        root=plan['root']
        if self.server.state.webmon_structural_recovery:raise ValueError('Recover the saved structural jobs before starting another snapshot')
        if 'stream' not in plan:
            plan['stream']=StructuralObservationStream(self._webmon,self._webmon_effective_ipts(),plan['run'],plan['templates'],root)
        return plan['stream']

    def webmon_structural_transfer_changed(self,value,**_):
        if type(value) is not bool:return
        s=self.server.state;s.webmon_refine_transfer=value
        plan=self._webmon_structural_plan
        if plan and value!=plan['templates'][0]['refine_transfer']:
            self.webmon_pause_stream();s.webmon_structural_ready=False
            s.webmon_structural_message='Snapshot transfer settings changed. Use the approved structural plan again before fitting; existing jobs and results retain their original settings.'
        s.flush()

    def _launch_structural(self,item,stream,plan):
        # Reserve the UI owner before yielding to the event loop. Two clicks
        # must not capture two revisions or overwrite the recoverable task.
        s=self.server.state
        task=getattr(self,'_webmon_task',None)
        if s.webmon_busy or (task and not task.done()):return
        self._webmon_cancelled=False;s.webmon_busy=True;s.flush()
        self._webmon_task=asyncio.create_task(self._webmon_execute_structural(item,stream,plan))

    def webmon_start_structural(self,**_):
        s=self.server.state
        task=getattr(self,'_webmon_task',None)
        if s.webmon_busy or (task and not task.done()):return
        try:
            self.webmon_pause_stream();stream=self._new_structural_stream();item=stream.capture()
            pending=self._webmon_structural_plan.pop('paused_pending',None)
            if item is None:item=pending
            if item is None:
                s.webmon_structural_message='No new snapshot. The last completed structural fit is retained.';s.flush();return
            self._launch_structural(item,stream,self._webmon_structural_plan)
        except Exception as exc:s.webmon_structural_message=str(exc);s.flush()

    def webmon_follow_structural(self,**_):
        s=self.server.state
        if s.webmon_busy:return
        try:
            if self._webmon.config.get('allow_conditional_stream') is not True:raise ValueError('This server has not enabled live snapshot analysis')
            self.webmon_pause_stream();stream=self._new_structural_stream()
            self._webmon_stream=stream;self._webmon_stream_scope=self._webmon_structural_plan['owner']
            self._webmon_queue=LatestQueue(stream.root/'queue.json');s.webmon_stream_active=True
            pending=self._webmon_structural_plan.pop('paused_pending',None)
            if pending:self._webmon_queue.offer(pending)
            s.webmon_stream_message='Following this run. One snapshot is active; only the latest pending cumulative revision is kept.'
            self.webmon_stream_tick()
        except Exception as exc:s.webmon_structural_message=str(exc)
        s.flush()

    async def _webmon_execute_structural(self,item,stream,plan):
        s=self.server.state;owner=plan['owner']
        def current():return not self._webmon_cancelled and self._webmon_scope_current(owner) and self._webmon_structural_plan is plan
        self._webmon_remote_cancel=ScopeCancellation(current)
        if not current():return
        plan['last_item']=item
        s.webmon_busy=True;s.webmon_stale=bool(s.webmon_structural_has_result);s.webmon_calculation_status='Submitting structural models';s.flush()
        def progress(rows):
            if not current():return
            s.webmon_structural_rows=[{k:v for k,v in row.items() if k!='result'} for row in rows]
            complete=sum(row['state']=='verified_point_fit' for row in rows)
            failed=sum(row['state'] in {'failed','cancelled'} for row in rows)
            s.webmon_count_summary=f'{complete} / {len(rows)} point fits · {failed} need attention'
            s.webmon_structural_message=f"Run {stream.run} · revision {item['revision']} · {s.webmon_count_summary}. Waiting for remaining models."
            for row in rows:
                if row['state']=='verified_point_fit':
                    row['view_sha256']=file_sha(row['result']['view'])
                    self._webmon_structural_results[item['id']+':'+row['candidate']]=row
            self._structural_result_choices(item['id']);self.webmon_render_structural();s.flush()
        try:
            rows=await analyse_snapshot(StructuralSnapshotRuntime(plan['runtime']),item,plan['root'],source=plan['source'],
                labels=plan['labels'],cancel=self._webmon_remote_cancel,on_progress=progress)
            if not current():return
            s.webmon_structural_recovery=False
            good=[row for row in rows if row['state']=='verified_point_fit']
            s.webmon_target=stream.run;s.webmon_analysis_revision=item['receipt']['content_sha256'][:12]
            s.webmon_validity='Structural point fits · review required'
            s.webmon_calculation_status='Completed' if good else 'Failed'
            s.webmon_stale=str(s.webmon_run)!=stream.run or s.webmon_revision!=s.webmon_analysis_revision
            s.webmon_structural_message=f"Run {stream.run} · revision {item['revision']} completed: {len(good)} / {len(rows)} checked point fits. All model outcomes remain visible; review optimization and physical interpretation."
            self.webmon_render_structural()
        except Exception as exc:
            if current():
                s.webmon_calculation_status='Failed';s.webmon_structural_message=str(exc)
                from .stream import ReviewRequired
                s.webmon_structural_recovery=isinstance(exc,ReviewRequired)
        finally:
            if self._webmon_scope_current(owner):
                if self._webmon_cancelled:s.webmon_structural_recovery=True
                s.webmon_busy=False;s.flush()

    def webmon_recover_structural(self,**_):
        s=self.server.state;plan=self._webmon_structural_plan
        if s.webmon_busy:return
        try:
            if not plan or 'last_item' not in plan or 'stream' not in plan:raise ValueError('No saved structural dispatch is available in this workspace')
            self.webmon_pause_stream()
            self._launch_structural(plan['last_item'],plan['stream'],plan)
        except Exception as exc:s.webmon_structural_message=str(exc);s.flush()

    def _structural_result_choices(self,prefix):
        s=self.server.state
        rows=[(key,row) for key,row in self._webmon_structural_results.items() if key.startswith(prefix+':')]
        if not rows:return  # Keep the last checked plot visibly stale while new work is pending.
        rows.sort(key=lambda pair:(pair[1]['rwp'],pair[0]))
        s.webmon_structural_choices=[dict(title=f"{row['name']} · Rwp {row['rwp']:.3f}% · revision {row['revision']} · review required",value=key) for key,row in rows]
        s.webmon_structural_has_result=bool(rows)
        if s.webmon_structural_selected not in {key for key,_ in rows}:
            s.webmon_structural_selected=rows[0][0] if rows else ''

    def webmon_render_structural(self,value=None,**_):
        s=self.server.state
        if isinstance(value,str) and value in self._webmon_structural_results:s.webmon_structural_selected=value
        row=self._webmon_structural_results.get(s.webmon_structural_selected)
        if row is None or not getattr(self,'_webmon_structural_fit',None):return
        try:
            result=row['result'];view=Path(result['view'])
            if file_sha(view)!=row['view_sha256']:raise ValueError('The structural plot artifact changed; recover its checked job result')
            selected=dict(id=s.webmon_structural_selected,has_view=True,view_path=str(view),status='optimization_unresolved',
                source=result['snapshot_sha256'],name=row['name'])
            self._webmon_structural_fit.update(go.Figure(symmetry_plots.fit(selected,axis=s.webmon_structural_axis,
                reset=s.webmon_structural_reset,intensity_scale=s.webmon_structural_scale)))
            data=json.loads(view.read_text());parameters=[]
            for phase in data.get('phases',[]):
                for label,number,unit in zip(('a','b','c','alpha','beta','gamma'),phase.get('cell',[]),('Å','Å','Å','°','°','°')):
                    parameters.append(dict(phase=phase['name'],parameter=label,value=f'{number:.7g}',unit=unit))
                for mode in phase.get('modes',[]):
                    parameters.append(dict(phase=phase['name'],parameter=mode['label'],value=f"{mode['value']:.7g}",unit='native mode amplitude'))
            s.webmon_structural_parameters=parameters
        except (OSError,ValueError,KeyError,TypeError) as exc:s.webmon_structural_message='Plot unavailable: '+str(exc)

    def webmon_structural_control(self,name,value,**_):
        choices={'webmon_structural_axis':{'d','q','native'},'webmon_structural_scale':{'linear','log'}}
        if name not in choices or value not in choices[name]:return
        setattr(self.server.state,name,value);self.webmon_render_structural()

    def webmon_reset_structural(self,**_):
        self.server.state.webmon_structural_reset+=1;self.webmon_render_structural()

    def webmon_download_structural(self,**_):
        s=self.server.state
        try:
            row=self._webmon_structural_results[s.webmon_structural_selected]
            artifact=row['result']['result']['continuation'];path=Path(artifact['path'])
            if file_sha(path)!=artifact['sha256']:raise ValueError('The structural publication changed')
            self.download_file('Structural continuation GPX','structural-'+digest(s.webmon_structural_selected)[:12]+'.gpx',path.read_bytes(),'application/octet-stream')
        except Exception as exc:s.webmon_structural_message=str(exc);s.flush()
