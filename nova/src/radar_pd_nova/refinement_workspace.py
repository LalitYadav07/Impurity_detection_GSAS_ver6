"""Persistent recipe revisions, immutable backlog snapshots and live enrollment."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import uuid

from gpx_recipe.compiler import approve as approve_plan
from gpx_recipe.contracts import plan_hash, stage_table
from gpx_recipe.execution import write_json
from gpx_recipe.planner import make_plan
from gpx_recipe.providers import configure_provider, provider_info, make_client
from gpx_recipe.usage import summarize_calls
from .refinement_runtime import RuntimeUnavailable
from .refinement_sources import RefinementScope, digest


def utc():
    return datetime.now(timezone.utc).isoformat()


class RefinementWorkspace:
    def __init__(self,root,*,history_id,runtime,planner=None):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.manifest=self.root/'workspace.json';self.runtime=runtime;self.history_id=str(history_id)
        self._planner=planner;self.lock=threading.RLock();self._checkpoint_lock=threading.Lock();self._inflight=set()
        self._checkpoint_hash=None;self._checkpoint_dataset=None
        self._inspection_locks={}
        if self.manifest.exists():
            self.state=json.loads(self.manifest.read_text())
            if self.state.get('schema')!='radar-refinement-workspace/v1' or self.state.get('history_id')!=self.history_id:
                raise ValueError('Refinement workspace belongs to a different history')
        else:self.state=dict(schema='radar-refinement-workspace/v1',history_id=self.history_id,sources={},drafts={},recipes={},jobs={},paused=False)
        self._validate_parallel_jobs(self.state.get('parallel_jobs',1))
        for draft in self.state['drafts'].values():
            if draft.get('status')=='busy':draft.update(status='interrupted',error='Planning/preflight was interrupted; review or regenerate this draft')
        for job in self.state['jobs'].values():
            if job['status']=='running':
                job['status']='recovering' if job.get('remote_action') else 'interrupted'
                job['message']='Recovering the existing Galaxy job' if job.get('remote_action') else 'The local worker was interrupted; explicitly retry after review'
        self.save()

    def save(self):
        self.state['updated_utc']=utc();write_json(self.manifest,self.state)

    def restore_remote(self):
        if self.state['recipes'] or self.state['jobs']:return False
        document=self.runtime.recover_workspace()
        if not document:return False
        if document.get('history_id')!=self.history_id:raise ValueError('Saved refinement session belongs to another Galaxy history')
        self._validate_parallel_jobs(document.get('parallel_jobs',1))
        with self.lock:
            self.state=document
            for job in self.state['jobs'].values():
                if job['status']=='running':job['status']='recovering' if job.get('remote_action') else 'interrupted'
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

    def observe(self,sources,*,active_live_ids=None):
        with self.lock:
            changed=False
            for value in sources:
                source=copy.deepcopy(value);old=self.state['sources'].get(source['id'])
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
                    if recipe.get('armed'):
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
            return (source['experiment'],str(run),str(source.get('bank') or '')) if run is not None else None
        monitored={identity(source) for source in values if source.get('live_watch') and identity(source) is not None}
        return [source for source in values if source.get('live_watch') or identity(source) not in monitored]

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
        if response_path.exists():result=json.loads(response_path.read_text());result['directory']=str(directory)
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

    def prepare(self,experiment,source_id,request,*,previous=None,reuse_recipe=None):
        if not request.strip():raise ValueError('Describe the refinement stages first')
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
        planning=None
        try:
            client=None
            if not reuse_recipe and not self._planner:
                # Fail before dispatching an inspection job when this session
                # was launched without the selected Galaxy credential context.
                configure_provider(os.getenv('RADAR_GPX_PLANNER_PROVIDER','illumine-luna'),os.getenv('RADAR_PD_REFINEMENT_KEY_FILE') or None)
                client=make_client()
            source=self.inspect(source_id);context=source['context']
            if previous and previous.get('model_signature')!=context['model_signature']:previous=None
            if reuse_recipe:
                from gpx_recipe.contracts import validate_plan
                with self.lock:saved=copy.deepcopy(self.state['recipes'][reuse_recipe])
                if saved['experiment']!=experiment:raise ValueError('Choose a saved recipe from this experiment')
                plan=validate_plan(saved['plan'],context)
                if request!=plan['request']:raise ValueError('A changed request needs a revised plan')
                planning=dict(plan=plan,calls=[],origin='Saved recipe '+str(saved['version']))
            elif self._planner:planning=self._planner(request,context,previous=previous)
            else:
                planning=make_plan(request,context,previous=previous,client=client)
            planning=dict(planning)
            planning['usage_summary']=summarize_calls(planning.get('calls',[]))
            plan=planning['plan'];directory=self.root/'drafts'/generation;directory.mkdir(parents=True,exist_ok=True)
            write_json(directory/'planning.json',planning)
            preview=None
            if not plan['questions'] and not plan['unsupported']:
                response=self.runtime.call('preview',source,directory/'preview',plan=plan,
                    dispatch_key=digest(['preview',source['source_sha256'],plan_hash(plan),source['inspection_revision']])[:24])
                if response['status']!='preview_passed':raise ValueError(response.get('error') or 'Native parameter preflight failed')
                preview=response['preview']
            with self.lock:
                draft=self.state['drafts'][experiment]
                if draft['generation']!=generation:return None
                draft.update(status='needs_input' if plan['questions'] or plan['unsupported'] else 'ready',plan=plan,
                    context=context,source=source,preview=preview,directory=str(directory),planning_calls=len(planning.get('calls',[])),
                    model_usage=planning['usage_summary'],
                    origin=(planning['calls'][0]['label'] if planning.get('calls') else planning.get('origin','Deterministic clarification')))
                self.save();return copy.deepcopy(draft)
        except Exception as exc:
            with self.lock:
                draft=self.state['drafts'].get(experiment)
                if draft and draft['generation']==generation:
                    calls=planning.get('calls',[]) if planning is not None else getattr(exc,'planning_calls',None)
                    draft.update(status='error',error=str(exc),preview=None)
                    if calls is not None:
                        draft.update(planning_calls=len(calls),model_usage=summarize_calls(calls))
                    self.save()
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
            if existing and any(not row['sha256'] for row in snapshot['baseline']):raise ValueError('Some selected GPXs have not downloaded yet; refresh the results before approving')
            if not existing and not future:raise ValueError('Select existing scans, future matching arrivals, or both')
            if not scope.matches(draft['source']):raise ValueError('The representative source GPX is outside the selected scope')
            current=self.state['sources'].get(draft['source_id'])
            if not current or current.get('source_sha256')!=draft['context']['source']['sha256']:
                raise ValueError('The representative GPX changed after planning; regenerate and review the recipe')
            receipt=approve_plan(draft['plan'],draft['context'],draft.get('preview') or {},approver=approver)
            receipt['execution_policy_sha256']=token
            recipe_id=digest([plan_hash(draft['plan']),token])[:24]
            if recipe_id in self.state['recipes']:return copy.deepcopy(self.state['recipes'][recipe_id])
            recipe=dict(id=recipe_id,experiment=experiment,version=1+sum(r['experiment']==experiment for r in self.state['recipes'].values()),
                plan=copy.deepcopy(draft['plan']),approval=receipt,context=copy.deepcopy(draft['context']),source_id=draft['source_id'],
                scope=scope.to_dict(),selection_token=token,existing=existing,future=future,approved_utc=utc(),armed=False,started=False,
                baseline=[copy.deepcopy(self.state['sources'][row['id']]) for row in snapshot['baseline']] if existing else [],
                generation=generation)
            self.state['recipes'][recipe_id]=recipe;draft['approved_recipe']=recipe_id;self.save();return copy.deepcopy(recipe)

    def _enqueue(self,recipe,source):
        if not source.get('source_sha256'):
            return None
        key=digest([source['experiment'],source['scan_key'],source['source_sha256'],plan_hash(recipe['plan'])])[:24]
        if key in self.state['jobs']:
            job=self.state['jobs'][key]
            if recipe['id'] not in job['recipes']:job['recipes'].append(recipe['id'])
            return job
        job=dict(id=key,recipes=[recipe['id']],recipe_id=recipe['id'],source=copy.deepcopy(source),status='queued',
                 queued_utc=utc(),attempts=0,message='Waiting for the refinement worker',remote_action=None)
        self.state['jobs'][key]=job;return job

    def _enroll_future(self,recipe):
        scope=RefinementScope(**recipe['scope'])
        for source in self._available_sources():
            if scope.matches(source) and source.get('observed_utc','')>recipe['approved_utc']:
                self._enqueue(recipe,source)

    def start(self,recipe_id,selection_token):
        with self.lock:
            recipe=self.state['recipes'][recipe_id]
            if recipe['selection_token']!=selection_token:raise ValueError('Approve this execution scope before starting')
            for source in recipe['baseline']:self._enqueue(recipe,source)
            recipe.update(started=True,armed=bool(recipe['future']))
            if recipe['armed']:self._enroll_future(recipe)
            self.state['paused']=False;self.save()
            return [copy.deepcopy(j) for j in self.state['jobs'].values() if recipe_id in j['recipes']]

    def pause(self,experiment,paused=True):
        with self.lock:
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
            self._inflight.add(job['id']);job.update(status='running',attempts=job['attempts']+1,message='Refining the approved stages')
            recipe=copy.deepcopy(self.state['recipes'][job['recipe_id']]);source=copy.deepcopy(job['source'])
            directory=self.root/'jobs'/job['id']/('attempt-'+str(job['attempts']));job['directory']=str(directory);self.save()
        def submitted(action):
            with self.lock:job['remote_action']=action;self.save()
            if self.runtime.kind=='galaxy':self.checkpoint()
        try:
            # Persist the claimed stable dispatch before contacting Galaxy.
            # A restored interrupted claim searches/reuses that dispatch.
            if self.runtime.kind=='galaxy':self.checkpoint()
            result=self.runtime.call('run',source,directory,plan=recipe['plan'],approval=recipe['approval'],
                dispatch_key=job['id'],on_submitted=submitted,resume=job.get('remote_action'),recover=job['attempts']>1)
            with self.lock:
                status=result['status']
                if status=='needs_review' and 'match' in result.get('error','').lower() and 'model' in result.get('error','').lower():status='needs_recipe'
                job.update(status=status,result=result,message=result.get('error') or 'Verified output ready for scientific review',finished_utc=utc())
                self.save()
        except Exception as exc:
            with self.lock:
                job.update(status='interrupted' if isinstance(exc,RuntimeUnavailable) or job.get('remote_action') else 'failed',message=str(exc),finished_utc=utc());self.save()
        finally:
            with self.lock:self._inflight.discard(job['id'])
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
