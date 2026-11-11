"""Persistent recipe revisions, immutable backlog snapshots and live enrollment."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time
import uuid

from gpx_recipe.compiler import approve as approve_plan
from gpx_recipe.contracts import plan_hash, stage_table
from gpx_recipe.execution import write_json
from .refinement_planner import make_plan
from gpx_recipe.providers import configure_provider, provider_info, make_client
from gpx_recipe.usage import summarize_calls
from .refinement_runtime import RuntimeUnavailable
from .checkpoint_uploads import CheckpointUploadPending
from .refinement_sources import RefinementScope, digest, retained_fallback_group
from .refinement_messages import galaxy_refinement_activity


def utc():
    return datetime.now(timezone.utc).isoformat()


def refinement_job_key(source, plan):
    return digest([source['experiment'],source['scan_key'],source['source_sha256'],plan_hash(plan)])[:24]


class RefinementWorkspace:
    def __init__(self,root,*,history_id,runtime,planner=None):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.manifest=self.root/'workspace.json';self.runtime=runtime;self.history_id=str(history_id)
        self._planner=planner;self.lock=threading.RLock();self._checkpoint_lock=threading.Lock();self._inflight=set()
        self._submission_cancels={}
        self._checkpoint_hash=None;self._checkpoint_dataset=None
        self._inspection_locks={}
        if self.manifest.exists():
            self.state=json.loads(self.manifest.read_text())
            if self.state.get('schema')!='radar-refinement-workspace/v1' or self.state.get('history_id')!=self.history_id:
                raise ValueError('Refinement workspace belongs to a different history')
        else:self.state=dict(schema='radar-refinement-workspace/v1',history_id=self.history_id,sources={},drafts={},recipes={},jobs={},paused=False,parallel_jobs=2 if runtime.kind=='galaxy' else 1)
        self._validate_parallel_jobs(self.state.get('parallel_jobs',1))
        self._recover_interrupted()
        self.save()

    def _recover_interrupted(self):
        for draft in self.state['drafts'].values():
            if draft.get('status')=='busy':draft.update(status='interrupted',error='Planning/preflight was interrupted; review or regenerate this draft')
        for job in self.state['jobs'].values():
            if job['status']=='running':
                job['status']='recovering' if job.get('remote_action') else 'interrupted'
                job['message']='Recovering the existing Galaxy job' if job.get('remote_action') else 'The local worker was interrupted; explicitly retry after review'

    def save(self):
        with self.lock:
            self.state['updated_utc']=utc();write_json(self.manifest,self.state)

    def restore_remote(self):
        if self.state['recipes'] or self.state['jobs']:return False
        document=self.runtime.recover_workspace()
        if not document:return False
        if document.get('schema')!='radar-refinement-workspace/v1' or document.get('history_id')!=self.history_id:
            raise ValueError('Saved refinement session belongs to another schema or Galaxy history')
        self._validate_parallel_jobs(document.get('parallel_jobs',1))
        with self.lock:
            self.state=document
            self._recover_interrupted()
            self.save()
        return True

    def checkpoint(self):
        # Capture after acquiring the upload lock: an older pending upload
        # cannot overwrite the recovery order of a newer scientific receipt.
        with self._checkpoint_lock:
            with self.lock:document=copy.deepcopy(self.state)
            stable={k:v for k,v in document.items() if k!='updated_utc'}
            fingerprint=digest(stable)
            if fingerprint==self._checkpoint_hash:return self._checkpoint_dataset
            dataset=self.runtime.save_workspace(document)
            # Publish the cache only after Galaxy acknowledged the backup.
            self._checkpoint_hash=fingerprint;self._checkpoint_dataset=dataset
            return dataset

    def canonical_source(self,source):
        """Resolve only proven pending-handle drift, preserving saved scopes."""
        with self.lock:
            source=copy.deepcopy(source)
            prior=[self.state['sources'].get(source['id'])]
            prior.extend(job.get('source') for job in self.state['jobs'].values())
            prior.extend(draft.get('source') for draft in self.state['drafts'].values())
            groups={}
            for candidate in prior:
                group=retained_fallback_group(source,candidate,self.history_id) if isinstance(candidate,dict) else None
                if group:
                    identity=tuple(group[key] for key in ('experiment','scan_key','sample_key'))
                    groups.setdefault(identity,group)
            # Conflicting saved scopes stay separate; do not guess which owns
            # the current publication or alter immutable recipe approvals.
            if len(groups)==1:source.update(next(iter(groups.values())))
            return source

    def observe(self,sources,*,active_live_ids=None,enroll_future=True,manual=False):
        with self.lock:
            changed=False
            for value in sources:
                source=self.canonical_source(value);old=self.state['sources'].get(source['id'])
                # Opening an otherwise unobserved checkpoint for inspection
                # does not make it a new acquisition for an armed recipe.
                # A later real catalog observation may explicitly promote it.
                if manual and (old is None or old.get('manual_recipe_input')
                               or old.get('source_sha256')!=source.get('source_sha256')):
                    source['manual_recipe_input']=True
                else:
                    source.pop('manual_recipe_input',None)
                # A summary-only recovery lacks the reviewed Live condition
                # channels and experiment/sample mapping. It must not erase
                # that provenance for the same exact published checkpoint.
                same_publication=(old and source.get('collection_id') and all(
                    old.get(key)==source.get(key) for key in
                    ['collection_id','collection_name','internal_source_sha256','published_sha256']))
                if same_publication and old.get('live_watch') and not source.get('live_watch'):
                    for key in ['experiment','experiment_label','scan_key','sample_key','sample_label','scan_label',
                                'run_id','run_number','bank','metadata','axis_overrides','live_watch']:
                        if key in old:source[key]=copy.deepcopy(old[key])
                for key,prior in list(self.state['sources'].items()):
                    if key!=source['id'] and prior.get('record_uid')==source.get('record_uid') and prior.get('collection_name')==source.get('collection_name') and source.get('record_uid'):
                        self.state.setdefault('superseded_sources',{})[key]=self.state['sources'].pop(key)
                        changed=True
                # Some older Galaxy publications omit the *published* checksum.
                # Inspecting the immutable collection captures it from bytes;
                # a later summary-only catalog refresh must not erase that
                # verified checksum and the atom table derived from it.
                if (old and not source.get('source_sha256') and old.get('source_sha256')
                        and source.get('collection_id') and source.get('collection_id')==old.get('collection_id')
                        and source.get('collection_name')==old.get('collection_name')
                        and source.get('internal_source_sha256')==old.get('internal_source_sha256')):
                    source['source_sha256']=old['source_sha256']
                if old and old.get('source_sha256')==source.get('source_sha256'):
                    source['observed_utc']=old['observed_utc']
                else:source['observed_utc']=utc();changed=True
                # Keep native context/view caches only for the same exact bytes.
                if old and old.get('source_sha256')==source.get('source_sha256'):
                    for key in ['context','inspection','inspection_revision','inspection_error']:
                        if key in old:source[key]=old[key]
                if old != source:changed=True
                self.state['sources'][source['id']]=source
            # Only a complete Live catalog can retire earlier watch flags.
            # Incremental GPX inspection/recovery must retain its provenance.
            if active_live_ids is not None:
                active=set(active_live_ids)
                for source in self.state['sources'].values():
                    watched=source['id'] in active
                    if bool(source.get('live_watch'))!=watched:
                        source['live_watch']=watched;changed=True
            if changed:
                for recipe in self.state['recipes'].values():
                    if not manual and enroll_future and recipe.get('armed'):
                        self._enroll_future(recipe)
                self.save()
        return changed

    def sources(self,experiment):
        with self.lock:
            return sorted([copy.deepcopy(s) for s in self._available_sources() if s['experiment']==experiment],
                          key=lambda s:(str(s.get('sample_label') or ''),s.get('run_number') or 0,str(s.get('bank') or ''),s['scan_key']))

    def _available_sources(self):
        """Prefer the monitored publication when an earlier test job used the same scan/bank."""
        values=list(self.state['sources'].values())
        def identity(source):
            run=source.get('run_number')
            if run is None:return None
            metadata=source.get('metadata') or {}
            if metadata.get('instrument')=='HB2A':
                # SPICE scan numbers restart for every experiment, including
                # experiments in one IPTS. A monitored scan must not hide an
                # unrelated recovered experiment's same-numbered scan.
                experiment=metadata.get('experiment_number')
                return (source['experiment'],'HB2A',experiment or source['id'],str(run),str(source.get('bank') or ''))
            return (source['experiment'],str(run),str(source.get('bank') or ''))
        monitored={identity(source) for source in values if source.get('live_watch') and not source.get('manual_recipe_input') and identity(source) is not None}
        superseded={source['model_parent_id'] for source in values if source.get('model_parent_id')}
        return [source for source in values if source['id'] not in superseded and
                (source.get('manual_recipe_input') or source.get('model_parent_id') or source.get('live_watch') or identity(source) not in monitored)]

    def selection(self,scope,*,existing=True,future=False):
        with self.lock:
            sources=[copy.deepcopy(s) for s in self._available_sources() if scope.matches(s)]
        snapshot=dict(scope=scope.to_dict(),existing=bool(existing),future=bool(future),
            baseline=[dict(id=s['id'],sha256=s.get('source_sha256')) for s in sorted(sources,key=lambda s:s['id'])])
        return snapshot,digest(snapshot)

    def inspect(self,source_id):
        with self.lock:inspection_lock=self._inspection_locks.setdefault(source_id,threading.Lock())
        with inspection_lock:return self._inspect_source(source_id)

    def _inspect_source(self,source_id):
        from gpx_recipe.symmetry.compiler import package_signature
        revision=package_signature()
        with self.lock:
            source=copy.deepcopy(self.state['sources'][source_id])
            if (source.get('context') and source.get('inspection_revision')==revision
                    and source.get('source_sha256')==source['context']['source']['sha256']):
                return source
        material=self.runtime.resolve(source)
        directory=self.root/'inspections'/material['source_sha256']/revision[:16]
        response_path=directory/'response.json'
        if response_path.exists():
            from .refinement_runtime import verify_result_files
            result=json.loads(response_path.read_text());verify_result_files(directory,result);result['directory']=str(directory)
        else:result=self.runtime.call('inspect',material,directory,dispatch_key=digest(['inspect',material['source_sha256'],revision])[:24])
        if result['status']!='inspected':raise ValueError(result.get('error') or 'This GPX cannot be used for recipe polishing')
        if result['context']['source']['sha256']!=material['source_sha256']:raise ValueError('Inspection belongs to another GPX revision')
        material.update(context=result['context'],inspection=str(directory),inspection_revision=revision,inspection_error=None)
        with self.lock:
            current=self.state['sources'].get(source_id)
            if current is None or (current.get('source_sha256') and current['source_sha256']!=material['source_sha256']):
                raise ValueError('The original result changed while its project was being inspected')
            material['observed_utc']=current['observed_utc'];self.state['sources'][source_id]=material;self.save()
        return copy.deepcopy(material)

    def edit_source(self,source_id,request):
        from gpx_recipe.model_edits import validate_edit
        source=self.inspect(source_id)
        request=validate_edit(source['context'],request)
        revision=uuid.uuid4().hex
        directory=self.root/'model_edits'/revision
        result=self.runtime.call('edit',source,directory,plan=request,dispatch_key=digest(['edit',revision])[:24])
        if result['status']!='model_edited':raise ValueError(result.get('error') or 'Phase correction failed')
        from .refinement_runtime import verify_result_files
        verify_result_files(directory,result)
        edited=copy.deepcopy(source)
        # Do not let catalog refresh coalesce a derivative with its publication.
        for key in ('dataset_id','collection_id','record_uid','published_sha256','internal_source_sha256','live_watch'):
            edited.pop(key,None)
        edited.update(id='model-'+revision,model_parent_id=source_id,model_parent_sha256=source['source_sha256'],
            source_path=str(directory/result['final_gpx']),source_sha256=result['output_sha256'],
            scan_key=source['scan_key']+':model:'+revision,scan_label=source['scan_label']+' · corrected model',
            collection_name=result['final_gpx'],context=result['context'],inspection=str(directory),
            project=dict(stage='edited_unqualified'),model_edit=str(directory/'model_edit.json'),observed_utc=utc())
        if result.get('model_dataset_id'):edited['dataset_id']=result['model_dataset_id']
        edited['metadata']=copy.deepcopy(source.get('metadata') or {})
        # Profile evidence remains valid for unchanged observations/instrument,
        # but the parent receipt no longer certifies the phase list or GPX hash.
        if edited['metadata'].get('preparation'):
            edited['metadata']['parent_preparation']=edited['metadata'].pop('preparation')
        with self.lock:
            if self.state['sources'][source_id]['source_sha256']!=source['source_sha256']:
                raise ValueError('The source changed while phase correction was running; edited model was not selected.')
            self.state['sources'][edited['id']]=edited
            self.state['drafts'].pop(source['experiment'],None)
            for recipe in self.state['recipes'].values():
                if recipe['experiment']==source['experiment']:
                    recipe['armed']=False
                    if recipe['source_id']==source_id or any(s['id']==source_id for s in recipe.get('baseline',[])):
                        recipe['invalidated_by_model']=edited['id']
            for job in self.state['jobs'].values():
                if self.state['recipes'][job['recipe_id']].get('invalidated_by_model') and job['status'] in {'queued','paused','failed','interrupted'} and not job.get('remote_action'):
                    job.update(status='needs_recipe',message='Starting phase model was corrected. Build and approve a recipe for the new revision.')
            self.save()
        return copy.deepcopy(edited)

    def prepare(self,experiment,source_id,request,*,previous=None,reuse_recipe=None,template=None,guided=None,progress=None,failure_policy=None):
        if not request.strip():raise ValueError('Describe the refinement stages first')
        if template not in {None,'battery_baseline'} or sum(x is not None for x in (template,reuse_recipe,guided))>1:
            raise ValueError('Choose one supported recipe source')
        generation=uuid.uuid4().hex
        with self.lock:
            source=self.state['sources'].get(source_id)
            if not source or source['experiment']!=experiment:raise ValueError('Choose a source GPX from this experiment')
            prior=copy.deepcopy(self.state['drafts'].get(experiment))
            previous=previous or ((prior or {}).get('plan'))
            self.state['drafts'][experiment]=dict(generation=generation,source_id=source_id,request=request,status='busy',plan=None,preview=None,error=None)
            # Editing cannot silently keep enrolling new scans under the older recipe.
            for recipe in self.state['recipes'].values():
                if recipe['experiment']==experiment:recipe['armed']=False
            self.save()
        started=time.monotonic();phase_started=started;phase='starting';durations=[]
        def report(next_phase=None,*,status='running'):
            nonlocal phase,phase_started
            now=time.monotonic()
            if next_phase and next_phase!=phase:
                durations.append(dict(phase=phase,seconds=max(0.,now-phase_started)))
                phase=next_phase;phase_started=now
            event=dict(generation=generation,phase=phase,status=status,
                source_sha256=source.get('source_sha256'),
                elapsed_seconds=max(0.,now-started),phase_seconds=max(0.,now-phase_started),
                phases=copy.deepcopy(durations))
            with self.lock:
                draft=self.state['drafts'].get(experiment)
                if not draft or draft['generation']!=generation:return
                draft['preparation']=copy.deepcopy(event);self.save()
            if progress:
                # A closed or superseded UI must not change the native work.
                try:progress(dict(event,sampled_monotonic=now))
                except Exception:pass
        planning=None
        try:
            report()
            client=None
            if not reuse_recipe and not template and guided is None and not self._planner:
                # Fail before dispatching an inspection job when this session
                # was launched without the selected Galaxy credential context.
                configure_provider(os.getenv('RADAR_GPX_PLANNER_PROVIDER','illumine-luna'),os.getenv('RADAR_PD_REFINEMENT_KEY_FILE') or None)
                client=make_client()
            report('inspect')
            source=self.inspect(source_id);context=source['context']
            planning_context=copy.deepcopy(context)
            planning_context['host_execution']={
                'starting_result_review':{
                    'status':source.get('scientific_status') or 'not_assessed',
                    'warnings':list(source.get('scientific_warnings') or []),
                    'scope':'These are qualifications of the starting data/model. Generating or executing a recipe does not resolve or remove them.'},
                'recipe_unit':'One supplied GPX, using only its actual histogram keys',
                'histogram_keys':[h['key']for h in context['histograms']],
                'scan_enrollment':'NOVA already manages scan, sample and bank selection outside this project. '
                    'Apply this per-project recipe independently after compatibility checks. A request for '
                    'several scans does not imply several histograms in this GPX. Do not invent extra '
                    'histograms or ask for a cross-scan mapping that this host already handles.',
                'future_enrollment':{'supported':True,'requires_approval':True,
                    'scope':'NOVA can enroll future matching scans after explicit recipe/scope approval and start. '
                        'Compatibility checks and the host selection controls remain authoritative.'},
                'output_artifacts':{'stage_gpx':True,'final_gpx':True,'source_plan_lineage':True,
                    'scope':'The native worker and NOVA retain per-stage and final GPXs with source/plan lineage. '
                        'These outputs are host-managed, not additional refinement targets.'},
                'condition_mapping':{'preserve_reviewed_temperature_axis':True,
                    'scope':'NOVA preserves available reviewed temperature/condition mappings in result metadata. '
                        'Do not infer measured conditions from GPX defaults or invent a missing mapping; '
                        'this capability does not change the diffraction fit.'}}
            if previous and previous.get('model_signature')!=context['model_signature']:previous=None
            report('guided' if guided is not None else 'baseline' if template else 'saved' if reuse_recipe else 'planning')
            if guided is not None:
                from gpx_recipe.guided import build_plan
                plan=build_plan(context,guided)
                if request!=plan['request']:raise ValueError('The guided request changed; rebuild from the current answers')
                planning=dict(plan=plan,calls=copy.deepcopy(guided.get('model_calls',[])),origin='Guided rules · deterministic recipe',guided=copy.deepcopy(guided))
            elif template:
                from gpx_recipe.battery.recipes import BASELINE_REQUEST,baseline_recipe
                if source.get('project',{}).get('stage')!='prepared_unqualified' or request!=BASELINE_REQUEST:
                    raise ValueError('The battery baseline requires an explicit prepared model and its displayed request')
                planning=dict(plan=baseline_recipe(context),calls=[],origin='Battery baseline template · no model call')
            elif reuse_recipe:
                from gpx_recipe.contracts import validate_plan
                with self.lock:saved=copy.deepcopy(self.state['recipes'][reuse_recipe])
                if saved['experiment']!=experiment:raise ValueError('Choose a saved recipe from this experiment')
                if saved.get('guided') and saved['plan'].get('source_sha256'):
                    from gpx_recipe.guided import check_binding
                    check_binding(context,saved['guided'])
                plan=validate_plan(saved['plan'],context)
                if request!=plan['request']:raise ValueError('A changed request needs a revised plan')
                planning=dict(plan=plan,calls=[],origin='Saved recipe '+str(saved['version']),guided=saved.get('guided'))
            elif self._planner:planning=self._planner(request,planning_context,previous=previous)
            else:
                planning=make_plan(request,planning_context,previous=previous,client=client)
            planning=dict(planning)
            planning['usage_summary']=summarize_calls(planning.get('calls',[]))
            plan=planning['plan']
            if failure_policy is not None:
                if failure_policy not in {'stop','continue_independent'}:raise ValueError('Invalid stage failure policy')
                plan=copy.deepcopy(plan);plan['failure_policy']=failure_policy;planning['plan']=plan
            directory=self.root/'drafts'/generation;directory.mkdir(parents=True,exist_ok=True)
            write_json(directory/'planning.json',planning)
            preview=None
            if not plan['questions'] and not plan['unsupported']:
                report('preflight')
                response=self.runtime.call('preview',source,directory/'preview',plan=plan,
                    dispatch_key=digest(['preview',source['source_sha256'],plan_hash(plan),source['inspection_revision']])[:24])
                if response['status']!='preview_passed':raise ValueError(response.get('error') or 'Native parameter preflight failed')
                preview=response['preview']
            with self.lock:
                draft=self.state['drafts'].get(experiment)
                if not draft or draft['generation']!=generation:return None
                draft.update(status='needs_input' if plan['questions'] or plan['unsupported'] else 'ready',plan=plan,
                    context=context,source=source,preview=preview,directory=str(directory),planning_calls=len(planning.get('calls',[])),
                    model_usage=planning['usage_summary'],guided=planning.get('guided'),
                    origin=(planning.get('origin') or (planning['calls'][0]['label'] if planning.get('calls') else 'Deterministic clarification')))
                report(status=draft['status'])
                return copy.deepcopy(draft)
        except Exception as exc:
            with self.lock:
                draft=self.state['drafts'].get(experiment)
                if draft and draft['generation']==generation:
                    calls=planning.get('calls',[]) if planning is not None else getattr(exc,'planning_calls',None)
                    draft.update(status='error',error=str(exc),preview=None)
                    if isinstance(getattr(exc,'planning_provenance',None),dict):
                        draft['planning_provenance']=copy.deepcopy(exc.planning_provenance)
                    if calls is not None:
                        draft.update(planning_calls=len(calls),model_usage=summarize_calls(calls))
                    report(status='error')
            raise

    def discard_draft(self,experiment):
        with self.lock:
            self.state['drafts'].pop(experiment,None);self.save()

    def approve(self,experiment,generation,scope,*,existing,future,selection_token,approver):
        with self.lock:
            draft=self.state['drafts'].get(experiment)
            if not draft or draft['generation']!=generation or draft['status']!='ready':
                raise ValueError('The recipe draft changed; review the current plan')
            snapshot,token=self.selection(scope,existing=existing,future=future)
            if token!=selection_token:raise ValueError('The selected scan set changed; review its updated scope before approving')
            if draft['plan'].get('source_sha256') and (future or not existing or {row['id'] for row in snapshot['baseline']}!={draft['source_id']}):
                raise ValueError('This recipe is approved for one exact GPX only. Select only its scan and turn off future enrollment, or rebuild with series reference applicability.')
            if existing and any(not row['sha256'] for row in snapshot['baseline']):raise ValueError('Some selected GPXs have not downloaded yet; refresh the results before approving')
            if not existing and not future:raise ValueError('Select existing scans, future matching arrivals, or both')
            if not scope.matches(draft['source']):raise ValueError('The representative source GPX is outside the selected scope')
            current=self.state['sources'].get(draft['source_id'])
            if not current or current.get('source_sha256')!=draft['context']['source']['sha256']:
                raise ValueError('The representative GPX changed after planning; regenerate and review the recipe')
            if existing:
                from gpx_recipe.compatibility import preview
                compatibility=preview([self.state['sources'][row['id']] for row in snapshot['baseline']],draft['context'])
                incompatible=[row for row in compatibility['rows'] if row['status']=='incompatible']
                if incompatible:
                    first=incompatible[0]
                    raise ValueError(f"{len(incompatible)} selected scans need a separate recipe. {first['scan']}: {first['reason']} Choose only compatible saved results, or prepare another recipe for the different model.")
            receipt=approve_plan(draft['plan'],draft['context'],draft.get('preview') or {},approver=approver)
            receipt['execution_policy_sha256']=token
            recipe_id=digest([plan_hash(draft['plan']),token])[:24]
            if recipe_id in self.state['recipes']:
                saved=self.state['recipes'][recipe_id]
                if saved.get('invalidated_by_model'):
                    raise ValueError('This recipe was invalidated by a phase correction. Review the corrected model.')
                draft['approved_recipe']=recipe_id;self.save()
                return copy.deepcopy(saved)
            recipe=dict(id=recipe_id,experiment=experiment,version=1+sum(r['experiment']==experiment for r in self.state['recipes'].values()),
                plan=copy.deepcopy(draft['plan']),approval=receipt,context=copy.deepcopy(draft['context']),source_id=draft['source_id'],
                scope=scope.to_dict(),selection_token=token,existing=existing,future=future,approved_utc=utc(),armed=False,started=False,
                baseline=[copy.deepcopy(self.state['sources'][row['id']]) for row in snapshot['baseline']] if existing else [],
                generation=generation,guided=copy.deepcopy(draft.get('guided')))
            self.state['recipes'][recipe_id]=recipe;draft['approved_recipe']=recipe_id;self.save();return copy.deepcopy(recipe)

    def _enqueue(self,recipe,source):
        if not source.get('source_sha256'):
            return None
        key=refinement_job_key(source,recipe['plan'])
        if key in self.state['jobs']:
            job=self.state['jobs'][key]
            if recipe['id'] not in job['recipes']:job['recipes'].append(recipe['id'])
            return job
        job=dict(id=key,recipes=[recipe['id']],recipe_id=recipe['id'],source=copy.deepcopy(source),status='queued',
                 queued_utc=utc(),attempts=0,message='Waiting for the refinement worker',remote_action=None)
        from gpx_recipe.compatibility import assess
        compatibility=assess(source,recipe.get('context') or {'model_signature':recipe['plan'].get('model_signature')})
        if compatibility['status']=='incompatible':
            job.update(status='needs_recipe',message=compatibility['reason']+' Prepare a separate recipe for this model.',
                       finished_utc=utc())
        self.state['jobs'][key]=job;return job

    def _enroll_future(self,recipe):
        scope=RefinementScope(**recipe['scope'])
        for source in self._available_sources():
            if not source.get('manual_recipe_input') and scope.matches(source) and source.get('observed_utc','')>recipe['approved_utc']:
                self._enqueue(recipe,source)

    def start(self,recipe_id,selection_token):
        with self.lock:
            recipe=self.state['recipes'][recipe_id]
            if recipe.get('invalidated_by_model'):raise ValueError('This recipe was invalidated by a phase correction. Build and approve a recipe for the corrected model.')
            if recipe['selection_token']!=selection_token:raise ValueError('Approve this execution scope before starting')
            for source in recipe['baseline']:
                job=self._enqueue(recipe,source)
                if job and job['status']=='paused':job.update(status='queued',message='Waiting for the refinement worker')
            recipe.update(started=True,armed=bool(recipe['future']))
            if recipe['armed']:self._enroll_future(recipe)
            self.state['paused']=False;self.save()
            return [copy.deepcopy(j) for j in self.state['jobs'].values() if recipe_id in j['recipes']]

    def pause(self,experiment,paused=True):
        with self.lock:
            if paused and self.runtime.kind=='galaxy':
                for uid,event in self._submission_cancels.items():
                    job=self.state['jobs'][uid]
                    if job['source']['experiment']==experiment and not job.get('remote_action'):
                        event.set()
            for recipe in self.state['recipes'].values():
                if recipe['experiment']==experiment:recipe['armed']=False
            for job in self.state['jobs'].values():
                if job['source']['experiment']==experiment and job['status']=='queued':job['status']='paused' if paused else 'queued'
            self.save()

    def resume_queue(self,experiment):
        with self.lock:
            for job in self.state['jobs'].values():
                if job['source']['experiment']==experiment and job['status']=='paused':job['status']='queued'
            self.save()

    def retry(self,job_id):
        with self.lock:
            job=self.state['jobs'][job_id]
            if self.state['recipes'][job['recipe_id']].get('invalidated_by_model'):
                raise ValueError('This recipe was invalidated by a phase correction. Use the corrected model.')
            if job['status'] not in {'interrupted','failed'}:raise ValueError('Only interrupted or failed transport jobs can be retried; changed models need a new recipe')
            job.setdefault('history',[]).append({k:copy.deepcopy(job.get(k)) for k in ['status','message','attempts','directory','remote_action']})
            job.update(status='recovering' if job.get('remote_action') else 'queued',message='Explicit retry requested');self.save()

    @property
    def parallel_jobs(self):
        return self.state.get('parallel_jobs',1) if self.runtime.kind=='galaxy' else 1

    @staticmethod
    def _validate_parallel_jobs(value):
        if type(value) is not int or value not in {1,2,4}:
            raise ValueError('Choose 1, 2 or 4 concurrent polishing jobs')

    def set_parallel_jobs(self,value):
        self._validate_parallel_jobs(value)
        if self.runtime.kind!='galaxy' and value!=1:
            raise ValueError('Parallel polishing requires Galaxy-managed jobs')
        with self.lock:
            if self._inflight:
                raise ValueError('Let active polishing jobs finish before changing their concurrency')
            self.state['parallel_jobs']=value;self.save()

    def run_next(self):
        with self.lock:
            if len(self._inflight)>=self.parallel_jobs:return None
            job=next((j for j in self.state['jobs'].values() if j['status'] in {'queued','recovering'}),None)
            if job is None:return None
            self._inflight.add(job['id']);job.update(status='running',attempts=job['attempts']+1,
                message='Saving the dispatch checkpoint before submission' if self.runtime.kind=='galaxy' else 'Refining the approved stages')
            cancel_event=threading.Event();self._submission_cancels[job['id']]=cancel_event
            recipe=copy.deepcopy(self.state['recipes'][job['recipe_id']]);source=copy.deepcopy(job['source'])
            directory=self.root/'jobs'/job['id']/('attempt-'+str(job['attempts']));job['directory']=str(directory);self.save()
        def submitted(action):
            with self.lock:
                job['remote_action']=action
                job['message']='Submitted to Galaxy; waiting for the verified result'
                self.save()
            if self.runtime.kind=='galaxy':self.checkpoint()
        def progress(action):
            # Polling already occurs in the scientific runtime. Persist the
            # observed phase locally without another History upload/poll.
            with self.lock:
                if not job.get('remote_action') or action.get('galaxy_job_id')!=job['remote_action'].get('galaxy_job_id'):
                    raise ValueError('Refinement progress belongs to a different Galaxy job')
                job['remote_action']=action
                job['message']=galaxy_refinement_activity(action)
                self.save()
        try:
            # Persist the claimed stable dispatch before contacting Galaxy.
            # A restored interrupted claim searches/reuses that dispatch.
            if self.runtime.kind=='galaxy':self.checkpoint()
            with self.lock:
                job['message']='Preparing the approved refinement for Galaxy' if self.runtime.kind=='galaxy' else 'Refining the approved stages'
                self.save()
            result=self.runtime.call('run',source,directory,plan=recipe['plan'],approval=recipe['approval'],
                dispatch_key=job['id'],on_submitted=submitted,resume=job.get('remote_action'),recover=job['attempts']>1,
                cancel_event=cancel_event,on_progress=progress)
            with self.lock:
                status=result['status']
                if status=='needs_review' and 'match' in result.get('error','').lower() and 'model' in result.get('error','').lower():status='needs_recipe'
                job.update(status=status,result=result,message=result.get('error') or 'Verified output ready for scientific review',finished_utc=utc())
                self.save()
        except CheckpointUploadPending as exc:
            with self.lock:
                # The upload is already acknowledged and must not be posted
                # again. Pause the scope before another lane claims untouched
                # work; accepted scientific jobs remain independently alive.
                self.pause(source['experiment'])
                job.update(status='interrupted' if job.get('remote_action') else 'paused',
                    message=(str(exc) + ' New submissions for this experiment are paused. '
                             + ('Recover this accepted Galaxy job before resuming.' if job.get('remote_action')
                                else 'No scientific job was submitted for this scan; resume the queue after the checkpoint is ready.')),
                    finished_utc=utc())
                self.save()
        except Exception as exc:
            with self.lock:
                status=('paused' if cancel_event.is_set() and not job.get('remote_action')
                        else 'interrupted' if isinstance(exc,RuntimeUnavailable) or job.get('remote_action') else 'failed')
                job.update(status=status,message=str(exc),finished_utc=utc());self.save()
        finally:
            with self.lock:
                self._inflight.discard(job['id']);self._submission_cancels.pop(job['id'],None)
        with self.lock:return copy.deepcopy(job)

    def recover_job_artifacts(self,job_id):
        with self.lock:
            job=copy.deepcopy(self.state['jobs'][job_id]);recipe=copy.deepcopy(self.state['recipes'][job['recipe_id']])
        if not job.get('remote_action'):raise ValueError('The local refinement output is unavailable; restore its saved workspace directory')
        directory=self.root/'jobs'/job_id/('recovered-'+uuid.uuid4().hex[:8])
        result=self.runtime.call('run',job['source'],directory,plan=recipe['plan'],approval=recipe['approval'],
            dispatch_key=job_id,resume=job['remote_action'])
        if result['status'] not in {'completed','needs_review'}:
            raise ValueError(result.get('error') or 'The saved Galaxy result has no recoverable refinement stages')
        with self.lock:
            current=self.state['jobs'][job_id];current.update(result=result,directory=str(directory))
            current['artifact_recoveries']=current.get('artifact_recoveries',0)+1;self.save()
        return result

    def public(self,experiment):
        with self.lock:
            sources=self.sources(experiment);draft=copy.deepcopy(self.state['drafts'].get(experiment))
            recipes=[copy.deepcopy(r) for r in self.state['recipes'].values() if r['experiment']==experiment]
            jobs=[copy.deepcopy(j) for j in self.state['jobs'].values() if j['source']['experiment']==experiment]
        if draft and draft.get('plan'):draft['table']=stage_table(draft['plan'],draft.get('context'))
        return dict(sources=sources,draft=draft,recipes=recipes,jobs=jobs,provider=provider_info(),runtime=self.runtime.kind)
