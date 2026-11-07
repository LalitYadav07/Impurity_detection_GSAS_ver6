"""Durable finite preparation: one Galaxy dispatch per bounded model batch.

Members retain independent scientific outcomes. A worker-proven unstarted
member may continue in a later batch with the same logical attempt; failures
require explicit retry. Counts, weights and experiment inference stay absent.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import threading
import uuid
import zipfile

from gpx_recipe.execution import write_json
from gpx_recipe.preparation import batch
from gpx_recipe.preparation.operations import pending_entries, stage_entry, publish_entry
from gpx_recipe.symmetry.compiler import confined, file_sha
from .native_operations import capture, handle_path, intent_lock, validate_handle
from .preparation_queue_state import SCHEMA, MAX_METADATA, TERMINAL, validate, upgrade
from .preparation_queue_results import requests_for, stable_digest, outcomes


class PreparationQueue:
    def __init__(self, root, library, runtime, *, source, checkpoint, _batch_size=16):
        if not callable(checkpoint):
            raise ValueError('Preparation needs a durable checkpoint callback before dispatch')
        if type(_batch_size) is not int or not 1<=_batch_size<=64:
            raise ValueError('A preparation batch contains 1–64 models')
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        if self.root!=library.root/'queue':
            raise ValueError('Use the single preparation queue inside this library revision')
        self.path=self.root/'queue.json'
        self.library,self.runtime,self.checkpoint=library,runtime,checkpoint
        self._lock=threading.RLock();self._paused=threading.Event();self._verified=set()
        self.source=dict(source);self.source.pop('record_uid',None)
        self.source['id']='preparation-'+library.document['id']
        expected=next(a['sha256'] for a in library.document['inputs']
                      if a['path']==library.document['specification']['source'])
        resolved=runtime.resolve(self.source)
        if resolved['source_sha256']!=expected:
            raise ValueError('Preparation source differs from the captured parent CIF')
        self.source=resolved;self.source['dataset_id']=runtime.dataset_id(resolved)
        self.identity={k:self.source[k] for k in ('id','dataset_id','source_sha256')}
        with self._lock,intent_lock(self.path):
            if not self.path.exists():
                write_json(self.path,dict(schema=SCHEMA,history_id=runtime.service.history_id,
                    library_id=library.document['id'],source=self.identity,
                    base_attempts={r['id']:r['attempt'] for r in library.document['entries']},
                    paused=True,maximum_jobs=1,batch_size=_batch_size,jobs=[]))
            self._read()

    def _validate(self,value):
        # Also used by History validation without constructing a live queue.
        return validate(value,self.library,self.runtime,self.identity)

    def _read(self):
        if self.path.is_symlink() or self.path.stat().st_size>MAX_METADATA:
            raise ValueError('Preparation queue is absent or oversized')
        original=json.loads(self.path.read_text());state=self._validate(upgrade(original))
        if state!=original:write_json(self.path,state)
        return state

    def _save(self,state,*,remote=False):
        self._validate(state);write_json(self.path,state)
        if remote:self.checkpoint(copy.deepcopy(state))

    def snapshot(self):
        return copy.deepcopy(self._read())

    def restore(self,snapshot):
        state=copy.deepcopy(self._validate(upgrade(snapshot)));state['paused']=True
        with self._lock,intent_lock(self.path):
            existing=self._read()
            if existing['jobs'] and existing!=state:
                raise ValueError('Restore into a fresh preparation queue; preserve existing progress')
            self._save(state);self._verified.clear()

    def resume(self,*,maximum_jobs=None):
        with self._lock,intent_lock(self.path):
            state=self._read()
            if maximum_jobs is not None:state['maximum_jobs']=maximum_jobs
            state['paused']=False;self._paused.clear();self._save(state,remote=True)

    def pause(self):
        self._paused.set()
        with self._lock,intent_lock(self.path):
            state=self._read();state['paused']=True;self._save(state,remote=True)

    def _ensure_attempt(self,member):
        self.library.refresh();row=self.library.row(member['entry']);attempt=member['attempt']
        if row['attempt']>attempt:return False
        if row['attempt']<attempt:
            if row['attempt']!=attempt-1:raise ValueError('Preparation attempt lineage has a gap')
            if row['state'] in TERMINAL-{'prepared'}:self.library.retry(member['entry'])
            actual=self.library.start(member['entry'])
            if actual!=attempt:raise ValueError('Preparation ownership changed during recovery')
            return True
        # An explicit review may already have put a terminal attempt back in
        # pending. Replaying its old failure must not undo that review.
        return row['state']!='pending'

    def _stage(self,job):
        if job['mode']=='batch':
            return batch.stage(self.library,[(m['entry'],m['attempt']) for m in job['members']],
                self.root/'batches'/job['id'],workers=4,seconds=480)
        member=job['members'][0]
        return stage_entry(self.library,member['entry'],member['attempt'],
            self.root/'attempts'/member['entry']/('attempt-'+str(member['attempt'])))

    def _submit(self,state,job):
        payload=self._stage(job)
        operation='preparation-batch' if job['mode']=='batch' else (
            'preparation-'+self.library.row(job['members'][0]['entry'])['model']['kind'])
        def before_post(handle):
            job.update(handle=handle,phase='awaiting_submission',reason='')
            self._save(state,remote=True)
        try:
            handle=self.runtime.submit_native(operation,payload,self.root,source=self.source,
                timeout=600,cancel=self._paused,on_intent=before_post)
        except Exception as exc:
            try:
                captured,_=capture(self.runtime,operation,payload,self.root,self.source,600)
                path=handle_path(self.root,captured['key'])
                if path.exists():job['handle']=validate_handle(self.runtime,json.loads(path.read_text()))
            except (OSError,ValueError):pass
            uncertain=bool(job['handle'] and job['handle']['submission_started'])
            job.update(phase='needs_recovery' if uncertain else 'awaiting_submission',
                reason=('Recovering the existing Galaxy dispatch; no duplicate will be submitted.' if uncertain
                    else 'Submission is waiting for inputs or its durable History checkpoint ('+type(exc).__name__+').'))
            self._save(state);return
        job.update(handle=handle,phase='submitted' if handle['job'] else 'needs_recovery',reason='')
        self._save(state,remote=True)

    def _inputs(self,job):
        hashes=job['handle']['request']['input_hashes'];missing=False
        for name,checksum in hashes.items():
            path=confined(self.root,name,exists=False)
            if not path.exists():missing=True
            elif file_sha(path)!=checksum:raise ValueError('Captured preparation input changed')
        if missing:self.runtime.restore_native_inputs(job['handle'],self.root)
        return requests_for(job,self.root)

    def _finish(self,member,state,reason):
        self.library.refresh();row=self.library.row(member['entry'])
        if row['attempt']!=member['attempt'] or row['state']=='pending':return
        if row['state']=='preparing':
            self.library.finish(member['entry'],member['attempt'],state=state,reason=reason)
        elif row['state']!=state:
            raise ValueError('A late receipt cannot replace a different completed preparation')

    def _publish(self,member,request,result):
        self.library.refresh();row=self.library.row(member['entry'])
        if row['attempt']!=member['attempt'] or row['state']=='pending':return
        if row['state']=='prepared':
            paths=self.library.prepared_assets(member['entry'])
            entry=next(p for p in paths.values() if p.name=='entry.json')
            if file_sha(entry)!=result.get('entry_sha256'):
                raise ValueError('Recovered preparation differs from its published worker receipt')
        else:publish_entry(self.library,member['entry'],member['attempt'],request,result)

    def _collect(self,state,job):
        requests,plan=self._inputs(job)
        result,_=self.runtime.collect_native(job['handle'],self.root)
        rows=outcomes(job,result,self.root,plan);checksum=stable_digest(result,self.root)
        if job['result_sha256'] is not None and job['result_sha256']!=checksum:
            raise ValueError('Recovered preparation receipt changed')
        job['result_sha256']=checksum;self._save(state)
        for row in rows:
            member={k:row[k] for k in ('entry','attempt')};identifier=member['entry']
            previous=job['outcomes'].get(identifier)
            # Preserve a locally rejected artifact on recovery. Its native
            # worker completion was never permission to accept it anyway.
            if previous and previous['source']=='verification':
                self._finish(member,previous['state'],previous['reason']);continue
            outcome=dict(state=row['state'],reason=row['reason'],source='worker')
            if row['state']=='prepared':
                try:self._publish(member,requests[identifier],row['result'])
                except (ValueError,KeyError,TypeError) as exc:
                    self.library.refresh()
                    current=self.library.row(identifier)
                    if current['attempt']==member['attempt'] and current['state']=='prepared':raise
                    reason='Preparation result failed verification ('+type(exc).__name__+'); no artifact was accepted.'
                    self._finish(member,'failed',reason)
                    outcome=dict(state='failed',reason=reason,source='verification')
            elif row['state']!='pending':self._finish(member,row['state'],row['reason'])
            # Each library publication is already atomic and independently
            # receipt-checked on replay. Do not rewrite the growing queue for
            # every sibling; persist its outcome table once per whole batch.
            job['outcomes'][identifier]=outcome
        cancelled=any(row['state']=='cancelled' or row['reason']=='cancelled_before_start' for row in rows)
        # Origin metadata is an optional acceleration aid. Failure to fetch a
        # receipt cannot invalidate already checked scientific artifacts; a
        # fresh recovery can discover it again without rerunning native work.
        from .preparation_evidence import remember
        remember(self.library,self.runtime,job['handle'],self.root)
        if cancelled:
            state['paused']=True;self._paused.set()
        job.update(phase='collected',reason='Native work was cancelled; preparation is paused for review.' if cancelled else '')
        self._save(state,remote=True)
        self._verified.add(job['id'])

    def _reject(self,state,job,error):
        reason='Preparation result failed verification ('+type(error).__name__+'); no unchecked artifact was accepted.'
        for member in job['members']:
            self.library.refresh();row=self.library.row(member['entry'])
            if row['attempt']==member['attempt'] and row['state']=='prepared':
                if job['outcomes'].get(member['entry'],{}).get('state')!='prepared':raise error
                continue
            self._finish(member,'failed',reason)
            job['outcomes'][member['entry']]=dict(state='failed',reason=reason,source='verification')
        job.update(phase='collected',reason=reason);self._save(state,remote=True)
        self._verified.add(job['id'])

    def _replay(self,state):
        for job in state['jobs']:
            for member in job['members']:self._ensure_attempt(member)
            self.library.refresh()
            # A locally reviewed retry still needs terminal remote ownership
            # proof after restart; its pending row is not permission to submit
            # a second job while an earlier one might still be running.
            needed=any(self.library.row(m['entry'])['attempt']<=m['attempt'] for m in job['members'])
            if job['phase']!='collected' or job['id'] in self._verified or not needed:continue
            if job['result_sha256'] is not None or any(
                    o['state']=='prepared' for o in job['outcomes'].values()):
                # Authenticate the original complete receipt. Claimed status
                # flags never reconstruct missing numeric arrays by themselves.
                self._collect(state,job)
            else:
                status=self.runtime.poll_native(job['handle']);job['handle']=status['handle']
                if status['state'] not in {'finished','failed','cancelled'}:
                    raise ValueError('A recovered terminal preparation job is not terminal')
                for member in job['members']:
                    outcome=job['outcomes'][member['entry']]
                    if outcome['source']=='job' and outcome['state']!=status['state']:
                        raise ValueError('Recovered Galaxy failure differs from its saved receipt')
                    self._finish(member,outcome['state'],outcome['reason'])
                self._verified.add(job['id'])

    def _deferred(self,state):
        latest={m['entry']:(job,m) for job in state['jobs'] for m in job['members']}
        self.library.refresh();result=[]
        for identifier,(job,member) in latest.items():
            row=self.library.row(identifier)
            if (job['phase']=='collected' and job['id'] in self._verified
                    and job['outcomes'][identifier]['state']=='pending'
                    and row['state']=='preparing' and row['attempt']==member['attempt']):
                result.append(identifier)
        return result

    def tick(self):
        """Collect each batch once, then admit a bounded fair finite scope."""
        with self._lock,intent_lock(self.path):
            state=self._read();self._replay(state)
            if state['paused']:self._paused.set()
            for job in state['jobs']:
                if job['phase']=='collected':continue
                handle=job['handle']
                if not handle or not handle['submission_started']:
                    if not state['paused'] and not self._paused.is_set():self._submit(state,job)
                    continue
                status=self.runtime.poll_native(handle);job['handle']=status['handle']
                if status['state']=='finished':
                    if not job['handle']['job']['outputs'].get('result_bundle'):
                        job['reason']='Calculation completed; waiting for Galaxy to publish its result bundle.'
                        continue
                    try:self._collect(state,job)
                    except (ValueError,KeyError,TypeError,zipfile.BadZipFile) as exc:self._reject(state,job,exc)
                elif status['state'] in {'failed','cancelled'}:
                    reason='Galaxy preparation job '+status['state']+'; inspect its console before retrying.'
                    for member in job['members']:
                        self._finish(member,status['state'],reason)
                        job['outcomes'][member['entry']]=dict(state=status['state'],reason=reason,source='job')
                    job.update(phase='collected',reason=reason);self._save(state,remote=True)
                    self._verified.add(job['id'])
                elif status['state']=='needs_recovery':job.update(phase='needs_recovery',reason=status['message'])
                else:job.update(phase='submitted',reason='')
            limit=min(state['maximum_jobs'],self.runtime.max_active_jobs)
            active=sum(j['phase']!='collected' for j in state['jobs'])
            available=max(0,limit-active)
            for slot in range(available):
                if state['paused'] or self._paused.is_set():break
                deferred=self._deferred(state)
                pending=pending_entries(self.library,deferred=deferred)
                # A small catalog must not fill one large batch and leave the
                # other explicitly allowed Galaxy job slots idle. Spread only
                # unowned work; existing batch membership and recovery stay
                # immutable. Keep the per-batch artifact and CPU guards intact.
                remaining_slots=available-slot
                members_per_job=min(state['batch_size'],
                    (len(pending)+remaining_slots-1)//remaining_slots)
                identifiers=pending[:members_per_job]
                if not identifiers:break
                members=[dict(entry=i,attempt=self.library.row(i)['attempt']+(i not in deferred)) for i in identifiers]
                job=dict(id=uuid.uuid4().hex,mode='batch',phase='awaiting_submission',handle=None,
                    members=members,outcomes={},reason='',result_sha256=None)
                state['jobs'].append(job);self._save(state)
                for member in members:self._ensure_attempt(member)
                self._submit(state,job)
            self._save(state)
            return dict(queue=copy.deepcopy(state),coverage=self.library.coverage(),
                active=sum(j['phase']!='collected' for j in state['jobs']),
                deferred=len(self._deferred(state)),effective_job_limit=limit)
