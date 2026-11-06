"""Finite preparation scheduling over the shared durable native transport.

Large arrays stay in library assets or immutable Galaxy result bundles. The
checkpoint callback persists only this queue's data-only task/handle records.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import threading
import zipfile

from gpx_recipe.execution import write_json
from gpx_recipe.preparation.operations import pending_entries, stage_entry, publish_entry
from gpx_recipe.symmetry.compiler import file_sha
from .native_operations import capture, handle_path, intent_lock, validate_handle

SCHEMA = 'radar-preparation-queue/v1'
TERMINAL = {'prepared', 'failed', 'cancelled', 'unsupported'}
PHASES = TERMINAL | {'awaiting_submission', 'submitted', 'needs_recovery'}
MAX_METADATA = 64*1024*1024


class PreparationQueue:
    def __init__(self, root, library, runtime, *, source, checkpoint):
        if not callable(checkpoint):
            raise ValueError('Preparation needs a durable checkpoint callback before dispatch')
        self.root = Path(root).resolve();self.root.mkdir(parents=True, exist_ok=True)
        if self.root != library.root/'queue':
            raise ValueError('Use the single preparation queue inside this library revision')
        self.path = self.root/'queue.json'
        self.library, self.runtime, self.checkpoint = library, runtime, checkpoint
        self._lock = threading.RLock();self._paused = threading.Event()
        self.source = dict(source)
        self.source.pop('record_uid',None)
        self.source['id'] = 'preparation-'+library.document['id']
        expected = next(a['sha256'] for a in library.document['inputs']
                        if a['path'] == library.document['specification']['source'])
        resolved = runtime.resolve(self.source)
        if resolved['source_sha256'] != expected:
            raise ValueError('Preparation source differs from the captured parent CIF')
        self.source = resolved
        self.source['dataset_id'] = runtime.dataset_id(resolved)
        self.identity = {k:self.source[k] for k in ('id', 'dataset_id', 'source_sha256')}
        with self._lock, intent_lock(self.path):
            if not self.path.exists():
                write_json(self.path, dict(schema=SCHEMA, history_id=runtime.service.history_id,
                    library_id=library.document['id'], source=self.identity,
                    base_attempts={r['id']:r['attempt'] for r in library.document['entries']},
                    paused=True, maximum_jobs=1, tasks=[]))
            self._read()

    def _validate(self, value):
        if (not isinstance(value, dict) or set(value) != {'schema', 'history_id', 'library_id',
                'source', 'base_attempts', 'paused', 'maximum_jobs', 'tasks'} or value['schema'] != SCHEMA
                or value['history_id'] != self.runtime.service.history_id
                or value['library_id'] != self.library.document['id'] or value['source'] != self.identity
                or type(value['paused']) is not bool or type(value['maximum_jobs']) is not int
                or value['maximum_jobs'] not in {1,2,4,8} or not isinstance(value['tasks'], list)
                or len(value['tasks']) > 20000):
            raise ValueError('Preparation queue belongs to another library, History or schema')
        base=value['base_attempts']
        if (not isinstance(base,dict) or set(base)!={r['id'] for r in self.library.document['entries']}
                or any(type(v) is not int or not 0<=v<=1000000 for v in base.values())):
            raise ValueError('Preparation baseline attempt inventory changed')
        seen=set();attempts=dict(base)
        for task in value['tasks']:
            if (not isinstance(task, dict) or set(task) != {'entry', 'attempt', 'phase', 'handle', 'reason'}
                    or not isinstance(task['entry'], str) or not re.fullmatch('[a-f0-9]{24}',task['entry'])
                    or type(task['attempt']) is not int or task['attempt'] < 1
                    or task['phase'] not in PHASES or not isinstance(task['reason'],str)
                    or len(task['reason']) > 2000 or (task['entry'],task['attempt']) in seen
                    or task['attempt'] != attempts.get(task['entry'], 0)+1):
                raise ValueError('Invalid or reordered preparation attempt record')
            self.library.row(task['entry'])
            seen.add((task['entry'],task['attempt']));attempts[task['entry']]=task['attempt']
            if task['handle'] is not None:
                validate_handle(self.runtime,task['handle'])
                if task['handle']['source'] != self.identity:
                    raise ValueError('Preparation handle names another source')
            if task['phase'] == 'prepared' and (not task['handle'] or not task['handle']['job']):
                raise ValueError('Prepared work needs its acknowledged native job')
        if len(json.dumps(value,allow_nan=False).encode()) > MAX_METADATA:
            raise ValueError('Preparation queue exceeds its bounded metadata scope')
        return value

    def _read(self):
        if self.path.is_symlink() or self.path.stat().st_size > MAX_METADATA:
            raise ValueError('Preparation queue is absent or oversized')
        return self._validate(json.loads(self.path.read_text()))

    def _save(self, state, *, remote=False):
        self._validate(state);write_json(self.path,state)
        if remote:
            self.checkpoint(copy.deepcopy(state))

    def snapshot(self):
        return copy.deepcopy(self._read())

    def restore(self, snapshot):
        """Restore beside the same captured base library, paused for review.

        A tick replays acknowledged results into the new local library. It
        never accepts missing local arrays from a prepared status alone.
        """
        state=copy.deepcopy(self._validate(snapshot));state['paused']=True
        with self._lock, intent_lock(self.path):
            existing=self._read()
            if existing['tasks'] and existing != state:
                raise ValueError('Restore into a fresh preparation queue; preserve existing progress')
            self._save(state)

    def resume(self, *, maximum_jobs=None):
        with self._lock, intent_lock(self.path):
            state=self._read()
            if maximum_jobs is not None:state['maximum_jobs']=maximum_jobs
            state['paused']=False;self._paused.clear();self._save(state,remote=True)

    def pause(self):
        # Stop new admission promptly; active Galaxy work remains collectible.
        self._paused.set()
        with self._lock, intent_lock(self.path):
            state=self._read();state['paused']=True;self._save(state,remote=True)

    def _directory(self, task):
        return self.root/'attempts'/task['entry']/('attempt-'+str(task['attempt']))

    def _stage(self, task, *, recover_completed=False):
        return stage_entry(self.library,task['entry'],task['attempt'],self._directory(task),
                           recover_completed=recover_completed)

    def _submit(self, state, task):
        request=self._stage(task);kind=self.library.row(task['entry'])['model']['kind']
        def before_post(handle):
            task.update(handle=handle,phase='awaiting_submission',reason='')
            self._save(state,remote=True)
        try:
            handle=self.runtime.submit_native('preparation-'+kind,request,self.root,
                source=self.source,timeout=600,cancel=self._paused,on_intent=before_post)
        except Exception as exc:
            # Capture uses no POST. Its exact key locates the retained intent
            # even when the dispatcher lost its acknowledgment.
            try:
                captured,_=capture(self.runtime,'preparation-'+kind,request,self.root,self.source,600)
                path=handle_path(self.root,captured['key'])
                if path.exists():task['handle']=validate_handle(self.runtime,json.loads(path.read_text()))
            except (OSError,ValueError):
                pass
            uncertain=bool(task['handle'] and task['handle']['submission_started'])
            task.update(phase='needs_recovery' if uncertain else 'awaiting_submission',
                reason=('Recovering the existing Galaxy dispatch; no duplicate will be submitted.' if uncertain
                        else 'Submission is waiting for its inputs or durable History checkpoint ('+type(exc).__name__+').'))
            self._save(state)
            return
        task.update(handle=handle,phase='submitted' if handle['job'] else 'needs_recovery',reason='')
        self._save(state,remote=True)

    def _publish(self, state, task, request, result):
        self.library.refresh();row=self.library.row(task['entry'])
        if row['state']=='prepared' and row['attempt']==task['attempt']:
            paths=self.library.prepared_assets(task['entry'])
            entry=next(p for p in paths.values() if p.name=='entry.json')
            if file_sha(entry)!=result['entry_sha256']:
                raise ValueError('Recovered preparation differs from the published worker receipt')
        else:
            publish_entry(self.library,task['entry'],task['attempt'],request,result)
        task.update(phase='prepared',reason='');self._save(state,remote=True)

    def _reject_result(self, state, task, error):
        reason='Preparation result failed verification ('+type(error).__name__+'); no artifact was accepted.'
        self.library.finish(task['entry'],task['attempt'],state='failed',reason=reason)
        task.update(phase='failed',reason=reason);self._save(state,remote=True)

    def _replay(self, state):
        """Reconstruct progress after base inputs and queue state are restored."""
        for task in state['tasks']:
            self.library.refresh();row=self.library.row(task['entry'])
            if row['attempt']>task['attempt']:continue
            if row['attempt']<task['attempt']:
                if row['state'] in {'failed','cancelled','unsupported'}:self.library.retry(task['entry'])
                attempt=self.library.start(task['entry'])
                if attempt!=task['attempt']:raise ValueError('Preparation attempt lineage changed during recovery')
                row=self.library.row(task['entry'])
            if task['phase']=='prepared' and row['state']!='prepared':
                request=self._stage(task)
                result,_=self.runtime.collect_native(task['handle'],self.root)
                self._publish(state,task,request,result)
            elif task['phase'] in TERMINAL-{'prepared'} and row['state']=='preparing':
                self.library.finish(task['entry'],task['attempt'],state=task['phase'],reason=task['reason'])

    def tick(self):
        """Collect existing jobs and admit only the bounded fair pending scope."""
        with self._lock, intent_lock(self.path):
            state=self._read();self._replay(state)
            if state['paused']:self._paused.set()
            for task in state['tasks']:
                if task['phase'] in TERMINAL:continue
                handle=task['handle']
                if not handle or not handle['submission_started']:
                    if not state['paused'] and not self._paused.is_set():self._submit(state,task)
                    continue
                status=self.runtime.poll_native(handle)
                task['handle']=status['handle']
                if status['state']=='finished':
                    if not task['handle']['job']['outputs'].get('result_bundle'):
                        task['reason']='Calculation completed; waiting for Galaxy to publish its result bundle.'
                        continue
                    request=self._stage(task,recover_completed=True)
                    try:
                        result,_=self.runtime.collect_native(task['handle'],self.root)
                    except (ValueError,zipfile.BadZipFile) as exc:
                        self._reject_result(state,task,exc)
                        continue
                    try:
                        self._publish(state,task,request,result)
                    except ValueError as exc:
                        self.library.refresh()
                        if self.library.row(task['entry'])['state']=='prepared':raise
                        self._reject_result(state,task,exc)
                elif status['state'] in {'failed','cancelled'}:
                    reason='Galaxy preparation job '+status['state']+'; inspect its console before retrying.'
                    self.library.finish(task['entry'],task['attempt'],state=status['state'],reason=reason)
                    task.update(phase=status['state'],reason=reason);self._save(state,remote=True)
                elif status['state']=='needs_recovery':
                    task.update(phase='needs_recovery',reason=status['message'])
                else:task.update(phase='submitted',reason='')
            active=sum(t['phase'] not in TERMINAL for t in state['tasks'])
            slots=min(state['maximum_jobs'],self.runtime.max_active_jobs)-active
            if not state['paused'] and not self._paused.is_set():
                for identifier in pending_entries(self.library)[:max(0,slots)]:
                    attempt=self.library.row(identifier)['attempt']+1
                    task=dict(entry=identifier,attempt=attempt,phase='awaiting_submission',handle=None,reason='')
                    state['tasks'].append(task);self._save(state)
                    actual=self.library.start(identifier)
                    if actual!=attempt:raise ValueError('Preparation ownership changed before admission')
                    self._submit(state,task)
                    if self._paused.is_set():break
            self._save(state)
            return dict(queue=copy.deepcopy(state),coverage=self.library.coverage(),
                active=sum(t['phase'] not in TERMINAL for t in state['tasks']),
                effective_job_limit=min(state['maximum_jobs'],self.runtime.max_active_jobs))
