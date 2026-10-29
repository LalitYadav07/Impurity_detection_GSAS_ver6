"""Application integration for reviewed GPX recipes and incremental scan queues."""
from __future__ import annotations

import asyncio
import copy
import json
import os
from pathlib import Path
import re
import tempfile

from gpx_recipe.contracts import stage_table
from gpx_recipe.usage import usage_labels
from .refinement_plots import empty_figure, fit_figure, trend_figure
from .refinement_runtime import configured_runtime, verify_result_files
from .refinement_sources import RefinementScope, digest, source_from_record
from .refinement_workspace import RefinementWorkspace
from .results import experiment_axis_options


def figure_json(figure):
    return json.loads(figure.to_json())


ATOMIC_REFINEMENT_EXAMPLE = (
    'Use only the phases and histograms already in this GPX. Keep their published '
    'starting model and preserve existing constraints, restraints, background '
    'function, coefficient count and support points. Fit the full measured range '
    'of each histogram, retaining its existing invalid/masked bins and exclusions; '
    'show the changed fit limits in the plan. Use at most six cycles per stage. '
    'Stage 1: refine the existing background and phase scales. Stage 2: keep '
    'those terms free and refine the unit cells. Stage 3: keep those terms free '
    'and refine only symmetry-allowed independent atomic positions for these '
    'sites: [enter phase and atom labels]. Then add isotropic Uiso one named '
    'site per stage for: [enter phase and atom labels]. Bound each Uiso to '
    '0–0.05 Å². Hold occupancies, instrument/profile, sample parameters and '
    'texture fixed throughout. Every stage must list its complete free set. '
    'If an atom, displacement model, measured range or requested bound is '
    'ambiguous, ask for review. Stop on unphysical values; do not silently '
    'clip, drop a phase or claim the full-range Rwp is comparable to the original.'
)


def refinement_rwp_summary(notice):
    def number(value):return f'{value:.3f}%' if value is not None else '—'
    original=number(notice.get('original_rwp'))
    selected=number(notice.get('polished_rwp'))
    if notice.get('rwp_comparable'):
        return f'Original Rwp {original} → selected {selected}'
    return f'Original Rwp {original} (original fit range) · selected {selected} (different fit range; compare the curves, not these scores)'


def refinement_status_message(result, baseline):
    message=str(result.get('error') or 'Verified output ready for scientific review')
    if result.get('status')!='needs_review' or 'violated physical/requested bounds' not in message:
        return message
    matches=re.findall(r"\('(\d+)::AUiso:(\d+)',\s*([-+\d.eE]+)\)",message)
    phases=(baseline or {}).get('phases') or []
    if not matches:return message
    labels=[]
    for phase_index,atom_index,value in matches:
        try:
            phase=phases[int(phase_index)];atom=phase['atoms'][int(atom_index)]
            labels.append(f"{phase['name']} · {atom['label']} Uiso = {float(value):.5g} Å²")
        except (IndexError,KeyError,TypeError,ValueError):return message
    completed=len(result.get('stages') or [])
    return (f"Stage {completed+1} stopped: {'; '.join(labels)} fell outside the requested physical bounds. "
            f"{completed} earlier verified stage{'s' if completed!=1 else ''} remain available; no final GPX was accepted.")


def selected_refinement_dataset(result, stage):
    if stage=='final':return result.get('final_dataset_id')
    return (result.get('stage_dataset_ids') or {}).get(stage)


def refinement_output_provenance(result, stage):
    if stage=='final' and result.get('final_gpx'):
        location=(f"Galaxy History → Verified refinement-stage GPXs → final (dataset {result['final_dataset_id']})"
                  if result.get('final_dataset_id') else 'Download below')
        return f'Final refined GPX: {location}. The starting RADAR-PD GPX is unchanged.'
    stages=result.get('stages') or []
    selected=next((v for v in stages if v['id']==stage),None)
    if not selected:return ''
    dataset=selected_refinement_dataset(result,stage)
    location=(f'Galaxy History → Verified refinement-stage GPXs → {stage} (dataset {dataset})'
              if dataset else 'Download below')
    return (f'Verified intermediate GPX: {location}. This is stage {stages.index(selected)+1} of the recipe; '
            'no final GPX was accepted.' if not result.get('final_gpx') else
            f'Verified intermediate GPX: {location}. The final refined GPX is also available in Galaxy History.')


def published_refinement_source(service, workspace, record, scan, sha_cache):
    root=record.local_output_dir
    has_index=root and any(p.is_file() for p in [root/'gpx_index.json',root/'ndip/gpx_index.json'])
    if not has_index and record.output_dataset_ids.get('summary') and record.output_dataset_ids.get('gpx_projects'):
        # Modern Live publications carry their project index in the summary.
        # Keep the initial catalog small; resolve a GPX only when selected.
        root=workspace.root/'catalog'/digest([record.uid,record.output_dataset_ids])[:24];root.mkdir(parents=True,exist_ok=True)
        for name in ['summary','gpx_index']:
            dataset=record.output_dataset_ids.get(name);target=root/(name+'.json')
            if dataset and not target.exists():
                temporary=target.with_suffix('.partial')
                service._download_dataset(dataset,temporary)
                json.loads(temporary.read_text());temporary.replace(target)
    elif not has_index and record.output_dataset_ids:
        record=service.collect_results(record);root=record.local_output_dir
    if root is None:raise ValueError('RADAR-PD result files are not available yet')
    payload={}
    for name in ['summary','gpx_index']:
        p=next((p for p in [root/(name+'.json'),root/'ndip'/(name+'.json')] if p.is_file()),None)
        if p:payload[name]=json.loads(p.read_text())
    return source_from_record(record,payload,scan,workspace.history_id,sha_cache=sha_cache,root_override=root)


def recover_completed_refinement_records(service, controller):
    """Hydrate restored completed jobs without collecting their large archives."""
    warnings=[]
    for run_id,run in controller.state.completed.items():
        record=controller.records.get(run_id)
        if record is None and run.galaxy_job_id:
            record=controller._minimal_record(run)
        if record is None:
            continue
        has_publication=record.output_dataset_ids.get('summary') and record.output_dataset_ids.get('gpx_projects')
        if str(getattr(record.status,'value',record.status)).lower()=='ok' and has_publication:
            continue
        try:
            controller.records[run_id]=service.refresh(record)
        except Exception as exc:
            warnings.append(dict(scan=record.name,reason=f'Could not refresh saved Galaxy result: {exc}'))
    return warnings


def refinement_experiment_options(workspace):
    """Keep older completed recipes accessible after source metadata improves."""
    labels={v['experiment']:v['experiment_label'] for v in workspace.state['sources'].values()}
    for job in workspace.state['jobs'].values():
        source=job['source'];experiment=source['experiment']
        if experiment not in labels:
            labels[experiment]='Earlier refinement results · '+str(source.get('experiment_label') or source.get('scan_label') or experiment)
    return labels


def recipe_cards(plan, context):
    phases={p['key']:p['name'] for p in context['phases']}
    histograms={h['key']:h['name'] for h in context['histograms']}
    flags={'X':'positions','U':'Uiso','F':'occupancy'}
    def label(target):
        kind=target['kind'];p=phases.get(target.get('phase'),'');h=histograms.get(target.get('histogram'),'')
        if kind=='atom':return f"{p} / {target['atom']}: "+', '.join(flags[f] for f in target['flags'])
        name={'cell':'cell','phase_scale':'phase scale','histogram_scale':'histogram scale','background':'background','preferred_orientation':'preferred orientation'}[kind]
        return ' / '.join(x for x in [p,h,name] if x)
    def held(value):
        parts=value.split(':')
        if parts[0] in phases:return f"{phases[parts[0]]} / {':'.join(parts[1:-1])}: {flags.get(parts[-1],parts[-1])}"
        return ' / '.join(phases.get(x,histograms.get(x,{'background':'background','phase_scale':'phase scale','histogram_scale':'histogram scale','cell':'cell'}.get(x,x))) for x in parts if x)
    cards=[]
    for number,(stage,row) in enumerate(zip(plan['stages'],stage_table(plan,context)),1):
        changes=[]
        if stage.get('background'):changes.append('Background: '+stage['background']['function']+f" · {stage['background']['coefficients']} coefficients")
        for key in ['phase_normalization','constraints','bounds','exclusions','fit_ranges','texture']:
            if stage.get(key):changes.append(key.replace('_',' ').capitalize()+': '+json.dumps(stage[key],ensure_ascii=False))
        cards.append(dict(id=stage['id'],number=number,title=stage['title'],cycles=stage['cycles'],free=[label(t) for t in stage['free']],
            held=[held(t) for t in row['newly_fixed']],edits=changes))
    return cards


class RefinementMixin:
    def _init_refinement(self):
        self._refinement_workspace=None;self._refinement_tasks=set();self._refinement_refresh_task=None;self._refinement_queue_task=None
        self._refinement_sha_cache={};self._refinement_view_cache={};self._refinement_selection_token=''
        self._refinement_refresh_again=False;self._refinement_internal=False
        self._refinement_checkpoint_task=None;self._refinement_checkpoint_dirty=False
        self._refinement_approving=False
        self._refinement_inspecting=False
        s=self.server.state
        for key,value in self._refinement_state_defaults().items():setattr(s,key,value)
        self.server.state.change('experiment_result_view','workspace_view')(self._refinement_view_changed)
        self.server.state.change('rf_experiment')(self._refinement_experiment_changed)
        self.server.state.change('rf_scope','rf_samples','rf_scans','rf_banks','rf_existing','rf_future')(self._refinement_scope_changed)
        self.server.state.change('rf_source')(self._refinement_source_changed)
        self.server.state.change('rf_request')(self._refinement_request_changed)
        self.server.state.change('rf_stage','rf_histogram','rf_axis','rf_display','rf_zoom')(self._refinement_plot_changed)
        self.server.state.change('rf_trend_metric','rf_trend_axis','rf_trend_display','rf_recipe_filter')(self._refinement_trend_changed)

    @staticmethod
    def _refinement_state_defaults():
        return dict(rf_mode='recipe',rf_busy=False,rf_loading=False,rf_message='',rf_error='',rf_transport='',rf_provider='Illumine Luna · lowest reasoning',
            rf_experiment='',rf_experiments=[],rf_source='',rf_sources=[],rf_source_details=[],rf_context_atoms=[],rf_context_search='',rf_scope='all',rf_samples=[],rf_sample_options=[],rf_scans=[],rf_scan_options=[],rf_banks=[],rf_bank_options=[],
            rf_existing=True,rf_future=False,rf_scope_summary='No published GPXs available',rf_scope_token='',rf_request='',rf_draft_status='',rf_cards=[],rf_questions=[],rf_assumptions=[],rf_plan_calls='',rf_model_tokens='',rf_model_cost='',rf_model_has_luna_price=False,rf_plan_ready=False,
            rf_approved=False,rf_recipe_id='',rf_recipe_summary='',rf_active_summary='',rf_saved_recipe='',rf_recipe_versions=[],rf_recipe_filter='all',rf_metrics=[],rf_source_warnings=[],rf_future_armed=False,
            rf_jobs=[],rf_job_search='',rf_job_status='all',rf_progress_text='No refinements started',rf_progress_percent=0,rf_selected_job='',rf_selected_title='',rf_selected_message='',rf_has_output=False,rf_output_provenance='',
            rf_result_tab='fit',rf_stage='final',rf_stages=[],rf_histogram='',rf_histograms=[],rf_axis='d',rf_display='compare',rf_fit=figure_json(empty_figure()),rf_fit_notice='',rf_rwp_summary='',rf_zoom=0,
            rf_trend=figure_json(empty_figure('Completed refinements will populate this trend')),rf_trend_notice='',rf_trend_metric='rwp',rf_trend_metrics=[dict(title='Rwp · original and polished',value='rwp')],rf_trend_axis='run_number',rf_trend_axes=[dict(title='Scan number',value='run_number')],rf_trend_display='absolute',
            rf_parameters=[],rf_parameter_search='',rf_downloads=[],rf_download='',rf_can_open_gsas=False,rf_open_gsas_label='Open selected verified GPX in GSAS-II',rf_queue_running=False,rf_follow=True,rf_expanded=False)

    def _refinement_schedule(self,coroutine,name):
        try:loop=asyncio.get_running_loop()
        except RuntimeError:coroutine.close();return None
        task=loop.create_task(coroutine,name=name);self._refinement_tasks.add(task);task.add_done_callback(self._refinement_tasks.discard)
        return task

    def _refinement_visible(self):
        s=self.server.state
        return getattr(s,'workspace_view',None)=='refinement' or (getattr(s,'workspace_view',None)=='experiment' and getattr(s,'experiment_result_view',None)=='refinement')

    def _refinement_view_changed(self,**_):
        if self._refinement_visible():self.refresh_refinement_sources()

    def _refinement_maybe_refresh(self):
        workspace=getattr(self,'_refinement_workspace',None)
        if self._refinement_visible() or (workspace and any(r.get('armed') for r in workspace.state['recipes'].values())):
            self.refresh_refinement_sources()

    def _refinement_store(self):
        if self._refinement_workspace:return self._refinement_workspace
        history=str(getattr(self.service,'history_id',None) or 'local')
        base=Path(os.environ.get('RADAR_PD_REFINEMENT_ROOT') or (Path(getattr(self.service,'output_root',tempfile.gettempdir()))/'refinement'))
        root=base/digest([history])[:16]
        runtime=configured_runtime(root,self.service)
        if runtime is None:raise ValueError('Configure the native refinement worker or a signed-in Galaxy history to use Refinement')
        workspace=RefinementWorkspace(root,history_id=history,runtime=runtime)
        workspace.restore_remote();self._refinement_workspace=workspace
        return workspace

    def refresh_refinement_sources(self,**_):
        if self._refinement_refresh_task and not self._refinement_refresh_task.done():
            self._refinement_refresh_again=True;return
        self._refinement_refresh_task=self._refinement_schedule(self._refresh_refinement_sources(),'radar-refinement-catalog')

    async def _refresh_refinement_sources(self):
        s=self.server.state;s.rf_loading=True;s.rf_error='';s.flush()
        try:
            workspace=await asyncio.to_thread(self._refinement_store)
            s.rf_transport='Local scientific worker' if workspace.runtime.kind=='local' else 'Galaxy scientific jobs'
            controller=getattr(self,'_powgen_controller',None)
            recovery_warnings=await asyncio.to_thread(recover_completed_refinement_records,self.service,controller) if controller else []
            records=dict(self.records)
            if controller:records.update({r.uid:r for r in controller.records.values()})
            rows={r.get('record_uid'):r for r in (s.powgen_all_scientific_rows or [])}
            def catalog():
                sources=[];warnings=[]
                for record in records.values():
                    if str(getattr(record.status,'value',record.status)).lower()!='ok':continue
                    try:
                        sources.append(published_refinement_source(self.service,workspace,record,rows.get(record.uid),self._refinement_sha_cache))
                    except Exception as exc:warnings.append(dict(scan=record.name,reason=str(exc)))
                return sources,warnings
            sources,warnings=await asyncio.to_thread(catalog)
            warnings=recovery_warnings+warnings
            await asyncio.to_thread(workspace.observe,sources)
            # Legacy publications need a checksum captured from actual bytes.
            # Download only newly enrolled future matches here; backlog choices
            # remain browsable before materializing the user's selected scope.
            for source in sources:
                enrolled=any(r.get('armed') and RefinementScope(**r['scope']).matches(source) and workspace.state['sources'][source['id']]['observed_utc']>r['approved_utc'] for r in workspace.state['recipes'].values())
                if not source.get('source_sha256') and enrolled:
                    try:
                        material=await asyncio.to_thread(workspace.runtime.resolve,source)
                        workspace.observe([material])
                    except Exception as exc:warnings.append(dict(scan=source['scan_label'],reason=str(exc)))
            s.rf_source_warnings=warnings
            experiments=refinement_experiment_options(workspace)
            s.rf_experiments=[dict(title=v,value=k) for k,v in sorted(experiments.items(),key=lambda x:x[1])]
            if s.rf_experiment not in experiments:
                selected=next((v['experiment'] for v in sources if v['record_uid']==s.selected_run_uid),None)
                s.rf_experiment=selected or (next(iter(experiments)) if experiments else '')
                self._refinement_experiment_changed()
            self._refinement_sync();self._refinement_start_pump()
            if not s.rf_selected_job:
                completed=[j for j in workspace.state['jobs'].values() if j['source']['experiment']==s.rf_experiment and (j.get('result') or {}).get('stages')]
                if completed:await self._select_refinement_job(max(completed,key=lambda j:(j['source'].get('run_number') or 0,j['queued_utc']))['id'])
        except Exception as exc:s.rf_error=str(exc)
        finally:
            s.rf_loading=False;s.flush()
            if self._refinement_refresh_again:
                self._refinement_refresh_again=False
                self._refinement_refresh_task=None;self.refresh_refinement_sources()

    def _refinement_experiment_changed(self,**_):
        if self._refinement_internal:return
        s=self.server.state;w=self._refinement_workspace
        draft=(w.state['drafts'].get(s.rf_experiment) if w else None) or {}
        recipe=(w.state['recipes'].get(draft.get('approved_recipe')) if w else None) or {}
        scope=recipe.get('scope') or {}
        s.rf_source=draft.get('source_id','');s.rf_request=draft.get('request','')
        s.rf_scope=scope.get('mode','all');s.rf_samples=list(scope.get('samples',[]));s.rf_scans=list(scope.get('scans',[]));s.rf_banks=list(scope.get('banks',[]))
        s.rf_existing=recipe.get('existing',True);s.rf_future=recipe.get('future',False);s.rf_selected_job='';s.rf_recipe_filter='all'
        if w and any(j['source']['experiment']==s.rf_experiment for j in w.state['jobs'].values()):s.rf_mode='results'
        self._refinement_sync()

    def _refinement_scope(self):
        s=self.server.state
        return RefinementScope(s.rf_experiment,s.rf_scope,tuple(s.rf_samples or []),tuple(s.rf_scans or []),tuple(s.rf_banks or []))

    def _refinement_scope_changed(self,**_):
        if not self._refinement_internal:
            self._refinement_sync()

    def refinement_policy_edited(self,*_,**__):
        self._refinement_hold_enrollment();self._refinement_sync_approval()

    def load_atomic_refinement_example(self,**_):
        self._refinement_hold_enrollment()
        self.server.state.rf_request=ATOMIC_REFINEMENT_EXAMPLE
        self.server.state.rf_message='Example loaded. Replace both bracketed atom lists with exact names from Inspect starting model before generating.'
        self._refinement_sync_approval();self.server.state.flush()

    def _refinement_hold_enrollment(self):
        w=self._refinement_workspace
        if w:
            with w.lock:
                changed=False
                for recipe in w.state['recipes'].values():
                    if recipe['experiment']==self.server.state.rf_experiment and recipe.get('armed'):
                        recipe['armed']=False;changed=True
                if changed:
                    w.save();self.server.state.rf_future_armed=False;self.server.state.rf_active_summary='';self._refinement_checkpoint()

    def _refinement_request_changed(self,**_):
        if self._refinement_internal:return
        self._refinement_sync_approval()

    def _refinement_source_changed(self,**_):
        self._refinement_sync_source()
        self._refinement_sync_approval()

    def _refinement_sync_source(self):
        s=self.server.state;w=self._refinement_workspace
        source=(w.state['sources'].get(s.rf_source) if w else None) or {}
        s.rf_source_details=[dict(label='Sample',value=source.get('sample_label','—')),dict(label='Bank',value=source.get('bank') or '—'),dict(label='Published GPX',value=source.get('collection_name','—'))]
        s.rf_context_atoms=[dict(phase=p['name'],atom=a['label'],element=a['element'],adp=a['adp_type'],position_dof=a.get('position_dof'),occupancy=a.get('occupancy')) for p in source.get('context',{}).get('phases',[]) for a in p['atoms']]

    def inspect_refinement_project(self,**_):
        s=self.server.state
        if s.rf_busy or not s.rf_source:return
        uid=s.rf_source;self._refinement_inspecting=True;s.rf_busy=True;s.rf_error='';s.rf_message='Inspecting the starting model without a model call…';s.flush()
        async def inspect():
            try:
                await asyncio.to_thread(self._refinement_workspace.inspect,uid)
                s.rf_message='Starting model inspected. Atom labels and displacement models are available below the source GPX.'
            except Exception as exc:s.rf_error=str(exc)
            finally:self._refinement_inspecting=False;self._refinement_sync();s.flush()
        self._refinement_schedule(inspect(),'radar-refinement-inspect')

    def _refinement_sync_approval(self):
        s=self.server.state;w=self._refinement_workspace
        draft=(w.state['drafts'].get(s.rf_experiment) if w else None) or {}
        recipe=w.state['recipes'].get(draft.get('approved_recipe')) if w else None
        s.rf_recipe_id=recipe['id'] if recipe else ''
        same_request=draft.get('request')==s.rf_request
        s.rf_approved=bool(recipe and same_request and recipe['selection_token']==self._refinement_selection_token and draft.get('source_id')==s.rf_source)
        s.rf_plan_ready=bool(same_request and draft.get('status')=='ready' and draft.get('source_id')==s.rf_source)
        s.rf_recipe_summary=(f"Recipe {recipe['version']} · {len(recipe['baseline'])} captured results"+(' · future matches included' if recipe['future'] else '')) if recipe else ''

    def _refinement_sync(self):
        w=self._refinement_workspace;s=self.server.state
        if not w or not s.rf_experiment:return
        w.lock.acquire()
        self._refinement_internal=True
        try:
            sources=w.sources(s.rf_experiment)
            s.rf_sample_options=[dict(title=label,value=key) for key,label in sorted({v['sample_key']:v['sample_label'] for v in sources}.items(),key=lambda x:x[1])]
            s.rf_scan_options=[dict(title=f"{v['scan_label']} · {v['sample_label']}",value=v['scan_key']) for v in sources]
            s.rf_bank_options=[dict(title='Bank '+b,value=b) for b in sorted({v['bank'] for v in sources if v['bank']})]
            try:
                scope=self._refinement_scope();snapshot,token=w.selection(scope,existing=s.rf_existing,future=s.rf_future)
                selected=[v for v in sources if scope.matches(v)];self._refinement_selection_token=token;s.rf_scope_token=token
                s.rf_scope_summary=f"{len(selected)} available run results"+(' · matching future arrivals' if s.rf_future else ' · existing scans only')
            except ValueError as exc:selected=[];self._refinement_selection_token='';s.rf_scope_token='';s.rf_scope_summary=str(exc)
            s.rf_sources=[dict(title=f"{v['scan_label']} · {v['sample_label']}",value=v['id']) for v in selected]
            if s.rf_source not in {v['id'] for v in selected}:s.rf_source=selected[0]['id'] if selected else ''
            self._refinement_sync_source()
            draft=w.state['drafts'].get(s.rf_experiment) or {}
            s.rf_draft_status=draft.get('status','');s.rf_busy=draft.get('status')=='busy' or self._refinement_approving or self._refinement_inspecting
            s.rf_cards=recipe_cards(draft['plan'],draft['context']) if draft.get('plan') else []
            s.rf_questions=(draft.get('plan') or {}).get('questions',[])+(draft.get('plan') or {}).get('unsupported',[])
            s.rf_assumptions=(draft.get('plan') or {}).get('assumptions',[])
            s.rf_plan_calls=f"{draft.get('planning_calls',0)} model call(s) · native preflight passed" if draft.get('preview') else ''
            s.rf_model_tokens,s.rf_model_cost=usage_labels(draft.get('model_usage'))
            s.rf_model_has_luna_price=bool((draft.get('model_usage') or {}).get('price_basis'))
            if draft.get('error'):s.rf_error=draft['error']
            recipes=[r for r in w.state['recipes'].values() if r['experiment']==s.rf_experiment]
            s.rf_recipe_versions=[dict(title=f"Recipe {r['version']} · {r['plan']['request'][:65]}",value=r['id']) for r in recipes]
            s.rf_future_armed=any(r.get('armed') for r in recipes)
            active=[r for r in recipes if r.get('armed')]
            s.rf_active_summary=('Active recipe '+', '.join(str(r['version']) for r in active)+' continues to enroll matching future results. Approving the current form captures its available scans separately.') if active else ''
            jobs=[j for j in w.state['jobs'].values() if j['source']['experiment']==s.rf_experiment]
            jobs.sort(key=lambda j:(j['source'].get('run_number') or 0,j['queued_utc']),reverse=True)
            version={r['id']:r['version'] for r in recipes}
            s.rf_jobs=[dict(id=j['id'],scan=j['source']['scan_label'],sample=j['source']['sample_label'],bank=j['source'].get('bank'),status=j['status'],message=j['message'],recipe=', '.join(str(version.get(rid,'?')) for rid in j['recipes']),recipe_id=j['recipe_id'],recipe_ids=j['recipes'],has_output=bool((j.get('result') or {}).get('final_view') or (j.get('result') or {}).get('stages'))) for j in jobs]
            completed=sum(j['status']=='completed' for j in jobs)
            attention=sum(j['status'] in {'failed','interrupted','needs_review','needs_recipe'} for j in jobs)
            processed=completed+attention
            s.rf_progress_text=(f'{processed} of {len(jobs)} processed · {completed} completed · {attention} need attention' if jobs else 'No refinements started')
            s.rf_progress_percent=round(100*processed/len(jobs)) if jobs else 0
            s.rf_metrics=[dict(label='Published GPXs',value=len(sources)),dict(label='In progress',value=sum(j['status'] in {'queued','running','recovering'} for j in jobs)),dict(label='Completed',value=sum(j['status']=='completed' for j in jobs)),dict(label='Needs attention',value=sum(j['status'] in {'failed','interrupted','needs_review','needs_recipe'} for j in jobs))]
            self._refinement_sync_approval()
            s.flush()
        finally:self._refinement_internal=False;w.lock.release()

    def generate_refinement_recipe(self,**_):
        s=self.server.state
        if s.rf_busy:return
        experiment,source,request=s.rf_experiment,s.rf_source,s.rf_request
        if not source or not request.strip():s.rf_error='Choose a representative GPX and describe the refinement';return
        self._refinement_hold_enrollment()
        s.rf_busy=True;s.rf_error='';s.rf_message='Inspecting the project, planning with Luna and checking native parameter releases…';s.flush()
        async def work():
            try:
                s.rf_message='Planning the recipe and checking native parameter releases…';s.flush()
                await asyncio.to_thread(self._refinement_workspace.prepare,experiment,source,request)
                s.rf_message='Review every stage and the execution scope before approving.'
            except Exception as exc:s.rf_error=str(exc);s.rf_message='Planning did not complete; no new approval was issued.'
            finally:self._refinement_sync();self._refinement_checkpoint();s.flush()
        self._refinement_schedule(work(),'radar-refinement-plan')

    def reuse_refinement_recipe(self,**_):
        s=self.server.state;w=self._refinement_workspace
        if s.rf_busy or not s.rf_source or not s.rf_saved_recipe:return
        saved=w.state['recipes'].get(s.rf_saved_recipe)
        if not saved or saved['experiment']!=s.rf_experiment:return
        experiment,source,recipe_id=s.rf_experiment,s.rf_source,s.rf_saved_recipe
        s.rf_request=saved['plan']['request'];s.rf_busy=True;s.rf_error='';self._refinement_hold_enrollment();s.rf_message='Checking the saved recipe against this source GPX without a model call…';s.flush()
        async def work():
            try:
                await asyncio.to_thread(w.prepare,experiment,source,saved['plan']['request'],reuse_recipe=recipe_id)
                s.rf_message='Saved recipe preflight passed. Review and approve the current scan scope.'
            except Exception as exc:s.rf_error=str(exc)
            finally:self._refinement_sync();self._refinement_checkpoint();s.flush()
        self._refinement_schedule(work(),'radar-refinement-reuse')

    def approve_refinement_recipe(self,selection_token=None,**_):
        s=self.server.state;w=self._refinement_workspace
        if s.rf_busy:return
        try:
            experiment=s.rf_experiment;draft=copy.deepcopy(w.state['drafts'][experiment]);scope=self._refinement_scope()
            existing,future=bool(s.rf_existing),bool(s.rf_future)
            before,before_token=w.selection(scope,existing=existing,future=future)
            if selection_token is not None and selection_token!=before_token:raise ValueError('The selected scans changed. Review the updated scope before approving.')
            if draft['source_id']!=s.rf_source or draft['request']!=s.rf_request:raise ValueError('Regenerate the recipe for this representative GPX and request')
        except Exception as exc:s.rf_error=str(exc)
        else:
            self._refinement_approving=True;s.rf_busy=True;s.rf_error='';s.flush()
            async def capture_and_approve():
                try:
                    selected=[v for v in w.sources(experiment) if scope.matches(v) and not v.get('source_sha256')] if existing else []
                    for index,value in enumerate(selected,1):
                        s.rf_message=f'Capturing selected published GPXs · {index} of {len(selected)}';s.flush()
                        material=await asyncio.to_thread(w.runtime.resolve,value);w.observe([material])
                    after,token=w.selection(scope,existing=existing,future=future)
                    initial={v['id']:v['sha256'] for v in before['baseline']};current={v['id']:v['sha256'] for v in after['baseline']}
                    if set(initial)!=set(current) or any(sha and sha!=current[key] for key,sha in initial.items()):
                        raise ValueError('The selected scans changed while their GPXs were captured. Review the updated scope and approve again; no new model call is needed.')
                    if s.rf_experiment!=experiment or self._refinement_scope()!=scope or s.rf_existing!=existing or s.rf_future!=future or s.rf_source!=draft['source_id'] or s.rf_request!=draft['request']:
                        raise ValueError('The request or execution scope changed; review it before approving')
                    w.approve(experiment,draft['generation'],scope,existing=existing,future=future,selection_token=token,approver='RADAR-PD interactive approval')
                    s.rf_message='Recipe and scan scope approved. Start the queue when ready.'
                except Exception as exc:s.rf_error=str(exc)
                finally:self._refinement_approving=False;self._refinement_sync();self._refinement_checkpoint();s.flush()
            self._refinement_schedule(capture_and_approve(),'radar-refinement-approve')
        s.flush()

    def start_refinement_queue(self,**_):
        s=self.server.state
        try:
            self._refinement_sync_approval()
            if not s.rf_approved:raise ValueError('Review and approve the current recipe and execution scope')
            self._refinement_workspace.start(s.rf_recipe_id,self._refinement_selection_token)
            s.rf_mode='results';s.rf_error='';s.rf_message='Running the approved recipe. Original RADAR-PD results are preserved.'
            self._refinement_sync();self._refinement_checkpoint();self._refinement_start_pump()
        except Exception as exc:s.rf_error=str(exc)
        s.flush()

    def _refinement_start_pump(self):
        if self._refinement_queue_task and not self._refinement_queue_task.done():return
        self._refinement_queue_task=self._refinement_schedule(self._refinement_pump(),'radar-refinement-queue')

    async def _refinement_pump(self):
        w=self._refinement_workspace;s=self.server.state
        if not w:return
        s.rf_queue_running=True;s.flush()
        try:
            while True:
                task=asyncio.create_task(asyncio.to_thread(w.run_next))
                await asyncio.sleep(0)
                self._refinement_sync()
                job=await task
                if job is None:break
                self._refinement_sync();self._refinement_checkpoint()
                if job['source']['experiment']==s.rf_experiment and (s.rf_follow or not s.rf_selected_job):
                    await self._select_refinement_job(job['id'])
        except Exception as exc:s.rf_error=str(exc)
        finally:s.rf_queue_running=False;s.flush()

    def _refinement_checkpoint(self):
        self._refinement_checkpoint_dirty=True
        if self._refinement_checkpoint_task and not self._refinement_checkpoint_task.done():return
        async def work():
            try:
                while self._refinement_checkpoint_dirty:
                    self._refinement_checkpoint_dirty=False
                    await asyncio.to_thread(self._refinement_workspace.checkpoint)
            except Exception as exc:self.server.state.rf_error='The local receipt is saved, but Galaxy session backup failed: '+str(exc);self.server.state.flush()
        self._refinement_checkpoint_task=self._refinement_schedule(work(),'radar-refinement-checkpoint')

    def pause_refinement_queue(self,**_):
        self._refinement_workspace.pause(self.server.state.rf_experiment);self.server.state.rf_message='Queued work paused and future enrollment stopped. The active native job will finish.';self._refinement_sync();self._refinement_checkpoint()

    def resume_refinement_queue(self,**_):
        self._refinement_workspace.resume_queue(self.server.state.rf_experiment);self._refinement_sync();self._refinement_start_pump()

    def retry_refinement_job(self,**_):
        try:self._refinement_workspace.retry(self.server.state.rf_selected_job);self._refinement_sync();self._refinement_start_pump()
        except Exception as exc:self.server.state.rf_error=str(exc);self.server.state.flush()

    def select_refinement_job(self,uid,**_):
        self.server.state.rf_follow=False
        self._refinement_schedule(self._select_refinement_job(str(uid)),'radar-refinement-select')

    def _read_refinement_view(self,job,name):
        root=Path(job['result']['directory']).resolve();path=(root/name).resolve()
        expected=job['result']['files'].get(name)
        if not expected or not path.is_relative_to(root):raise ValueError('This view is not in the verified refinement receipt')
        from .refinement_sources import file_sha
        stamp=(str(path),path.stat().st_mtime_ns,path.stat().st_size,expected)
        if stamp not in self._refinement_view_cache:
            if file_sha(path)!=expected:raise ValueError('The saved refinement view changed; recover its original output')
            self._refinement_view_cache[stamp]=json.loads(path.read_text())
            if len(self._refinement_view_cache)>30:self._refinement_view_cache.pop(next(iter(self._refinement_view_cache)))
        return self._refinement_view_cache[stamp]

    async def _select_refinement_job(self,uid):
        s=self.server.state;w=self._refinement_workspace
        job=w.state['jobs'].get(uid)
        if not job or job['source']['experiment']!=s.rf_experiment:return
        s.rf_selected_job=uid;s.rf_selected_title=job['source']['scan_label']+f" · {job['source']['sample_label']}"
        s.rf_selected_message=job['message'];s.rf_error='';s.rf_stage='final';s.rf_has_output=False;s.rf_output_provenance=''
        result=job.get('result') or {};s.rf_can_open_gsas=False;s.rf_open_gsas_label='Open selected verified GPX in GSAS-II'
        if result and not Path(result.get('directory','')).is_dir() and job.get('remote_action'):
            try:await asyncio.to_thread(w.recover_job_artifacts,uid);result=job['result']
            except Exception as exc:s.rf_error=str(exc)
        stages=result.get('stages') or []
        s.rf_stages=([dict(title='Final polished result',value='final')] if result.get('final_view') else [])+[dict(title=f"{i+1}. {v['title']}",value=v['id']) for i,v in enumerate(stages)]
        baseline=None
        if s.rf_stages:
            s.rf_has_output=True
            if not result.get('final_view'):s.rf_stage=s.rf_stages[-1]['value']
            try:
                baseline=await asyncio.to_thread(self._read_refinement_view,job,result['baseline_view'])
                s.rf_histograms=[dict(title=h['name'],value=h['name']) for h in baseline['histograms']]
                s.rf_histogram=s.rf_histograms[0]['value']
            except Exception as exc:s.rf_error=str(exc);s.rf_has_output=False
        stage_labels={v['gpx']:f"Stage {i+1} GPX · {v['title']}" for i,v in enumerate(stages)}
        s.rf_downloads=[dict(title=('Final refined GPX' if name==result.get('final_gpx') else stage_labels.get(name,name)),value=name) for name in sorted(result.get('files',{})) if name.endswith('.gpx') or name in ['refine.py','polished.execution.json','plan.json','approval.json']]
        s.rf_download=(result.get('final_gpx') or (stages[-1]['gpx'] if stages else (s.rf_downloads[0]['value'] if s.rf_downloads else '')))
        if result:s.rf_selected_message=refinement_status_message(result,baseline) if result.get('error') else job['message']
        self._refinement_plot_changed();self._refinement_trend_changed();s.flush()

    def _refinement_plot_changed(self,**_):
        s=self.server.state;w=self._refinement_workspace
        if not w or not s.rf_has_output:return
        job=w.state['jobs'].get(s.rf_selected_job)
        if not job:return
        try:
            result=job['result'];name=result.get('final_view') if s.rf_stage=='final' else next((v['view'] for v in result.get('stages',[]) if v['id']==s.rf_stage),None)
            if not name:return
            baseline=self._read_refinement_view(job,result['baseline_view']);view=self._read_refinement_view(job,name)
            fig,notice=fit_figure(baseline,view,histogram=s.rf_histogram,axis=s.rf_axis,display=s.rf_display,reset=s.rf_zoom)
            s.rf_fit=figure_json(fig);s.rf_fit_notice=notice.get('notice','')
            s.rf_rwp_summary=refinement_rwp_summary(notice)
            s.rf_output_provenance=refinement_output_provenance(result,s.rf_stage)
            s.rf_can_open_gsas=bool(selected_refinement_dataset(result,s.rf_stage))
            s.rf_open_gsas_label=('Open final refined GPX in GSAS-II' if s.rf_stage=='final' else 'Open selected verified stage in GSAS-II')
            labels={'refined':'Refined','refined_in_prior_stage':'Refined earlier; held now','fixed':'Held','fixed_by_model':'Fixed by model','derived':'Derived','configured':'Released; not estimated','unknown':'Not estimated'}
            s.rf_parameters=[dict(**p,value_display=f"{p['value']:.8g}",sigma_display=f"{p['sigma']:.3g}" if p.get('sigma') is not None else '—',status_display=labels.get(p['status'],p['status'].replace('_',' ')),stage=p.get('refined_stage') or s.rf_stage) for p in view.get('parameters',[])]
        except Exception as exc:s.rf_error=str(exc);s.rf_fit=figure_json(empty_figure('The verified fit view is unavailable'))

    def _refinement_trend_changed(self,**_):
        s=self.server.state;w=self._refinement_workspace
        if not w:return
        try:
            rows=[];eligible=0
            for job in w.state['jobs'].values():
                if job['source']['experiment']!=s.rf_experiment or job['status']!='completed':continue
                if s.rf_recipe_filter!='all' and s.rf_recipe_filter not in job['recipes']:continue
                eligible+=1
                result=job['result']
                rid=s.rf_recipe_filter if s.rf_recipe_filter!='all' else job['recipe_id']
                try:
                    compact=result.get('trend_views') or {}
                    baseline=compact.get('baseline') or self._read_refinement_view(job,result['baseline_view'])
                    final=compact.get('final') or self._read_refinement_view(job,result['final_view'])
                    rows.append(dict(id=job['id'],source=job['source'],recipe_id=rid,recipe_label='Recipe '+str(w.state['recipes'][rid]['version']),baseline=baseline,final=final))
                except (OSError,ValueError):continue
            s.rf_trend_notice=f'{len(rows)} of {eligible} completed results shown.'+(' Recover missing saved artifacts to complete this trend.' if len(rows)<eligible else '')
            options={'rwp':'Rwp · original and polished'}
            for row in rows:
                for p in row['final'].get('parameters',[]):
                    if p['kind']=='weight':options['phase_fractions']='Modeled crystalline phase fractions'
                    options[p['key']]=p['label']+(f" ({p['unit']})" if p.get('unit') else '')
            s.rf_trend_metrics=[dict(title=v,value=k) for k,v in options.items()]
            s.rf_trend_axes=experiment_axis_options([dict(run_number=r['source'].get('run_number'),metadata=r['source']['metadata'],axis_overrides=r['source'].get('axis_overrides') or {}) for r in rows])
            if s.rf_trend_metric not in options:s.rf_trend_metric='rwp'
            if s.rf_trend_metric in {'rwp','phase_fractions'}:s.rf_trend_display='absolute'
            if s.rf_trend_axis not in {v['value'] for v in s.rf_trend_axes}:s.rf_trend_axis='run_number'
            s.rf_trend=figure_json(trend_figure(rows,metric=s.rf_trend_metric,x_key=s.rf_trend_axis,display=s.rf_trend_display))
        except Exception as exc:s.rf_error=str(exc);s.rf_trend=figure_json(empty_figure(str(exc)))

    def download_refinement_artifact(self,**_):
        s=self.server.state
        try:
            job=self._refinement_workspace.state['jobs'][s.rf_selected_job];result=job['result'];name=s.rf_download
            if name not in {v['value'] for v in s.rf_downloads}:raise ValueError('Choose a verified output artifact')
            verify_result_files(result['directory'],result);path=Path(result['directory'])/name
            self.download_file(path.name,'application/octet-stream',path.read_bytes())
        except Exception as exc:s.rf_error=str(exc);s.flush()

    def open_refinement_in_gsasii(self,**_):
        s=self.server.state;job=self._refinement_workspace.state['jobs'].get(s.rf_selected_job)
        if not job or not job.get('result'):return
        dataset=selected_refinement_dataset(job['result'],s.rf_stage)
        if not dataset:return
        from .galaxy_service import GSASII_INTERACTIVE_TOOL_ID
        self._schedule_utility(self._submit_utility_action(tool_id=GSASII_INTERACTIVE_TOOL_ID,name='Verified refinement '+s.rf_stage+': '+job['source']['scan_label'],
            inputs={'project_source|source_kind':'single','project_source|gpx_project':{'dataset_id':dataset}},associated_run_uid=job['source']['record_uid']),'radar-polished-gsasii')
