"""Shared submission admission for scientific jobs in one Galaxy history.

This controls requests made by this NOVA process. Galaxy remains responsible
for admission across viewers and for actual CPU and memory allocation. A slot
is held only across the capacity check and submission, never during a fit.
"""
from contextlib import contextmanager
import base64
import binascii
import json
import os
import re
import threading
import time

import requests

ACTIVE_STATES = frozenset({'new', 'upload', 'waiting', 'queued', 'running',
                           'resubmitted', 'paused', 'setting_metadata'})
_LOCKS = {}
_WAITERS = {}
_LOCKS_LOCK = threading.Lock()
MAX_REQUEST_BYTES = 4*1024*1024


class AdmissionUnavailable(RuntimeError):
    pass


def operation_from_inputs(service, inputs, *, fetch=None):
    """Read the actual native request, not a caller-supplied admission label.

    Current wrappers carry inline JSON. Legacy request datasets are read with
    the same bounded, authenticated origin; no uploaded object is executed.
    """
    from .galaxy_service import _parameter_value, _scalar, _dataset_id
    def parameter(name):
        return inputs[name] if isinstance(inputs,dict) and name in inputs else _parameter_value(inputs,name)
    if _scalar(parameter('action')) != 'native':return None
    encoded = _scalar(parameter('model_request_b64'))
    reference = parameter('model_request')
    dataset = (str(reference['dataset_id']) if isinstance(reference, dict) and reference.get('dataset_id')
               else _dataset_id(reference))
    if encoded:
        if dataset:raise AdmissionUnavailable('Native admission found conflicting request transports')
        if not isinstance(encoded,str) or len(encoded)>4*((MAX_REQUEST_BYTES+2)//3):
            raise AdmissionUnavailable('Native admission request exceeds its metadata budget')
        try:raw=base64.b64decode(encoded,validate=True)
        except (ValueError,binascii.Error):raise AdmissionUnavailable('Native admission request is invalid') from None
    else:
        if not dataset or not re.fullmatch('[A-Za-z0-9_-]{1,128}',dataset):
            raise AdmissionUnavailable('Native admission could not identify its captured request')
        response=None
        try:
            response=(fetch or requests.get)(service.galaxy_url.rstrip('/')+'/api/datasets/'+dataset+'/display',
                headers=service._headers,params={'raw':'true'},timeout=30,stream=True,allow_redirects=False)
            response.raise_for_status();chunks=[];size=0
            for chunk in response.iter_content(chunk_size=65536):
                size+=len(chunk)
                if size>MAX_REQUEST_BYTES:raise AdmissionUnavailable('Native admission request exceeds its metadata budget')
                chunks.append(chunk)
            raw=b''.join(chunks)
        except requests.RequestException as exc:
            raise AdmissionUnavailable('Could not verify the captured native request; nothing submitted') from exc
        finally:
            close=getattr(response,'close',None)
            if close:close()
    try:
        if len(raw)>MAX_REQUEST_BYTES:raise ValueError('oversized')
        request=json.loads(raw)
        dispatch=_scalar(parameter('dispatch_key'))
        if (not isinstance(request,dict) or request.get('schema')!='radar-model-native/v1'
                or not isinstance(request.get('operation'),str) or not isinstance(request.get('payload'),dict)
                or dispatch is not None and request.get('dispatch_key')!=dispatch):
            raise ValueError('request identity')
    except (ValueError,UnicodeError):
        raise AdmissionUnavailable('Native admission request is invalid or changed') from None
    return request['operation']


def configured_limit(environ=None):
    value = (os.environ if environ is None else environ).get('RADAR_PD_MAX_ACTIVE_JOBS', '8')
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError('RADAR_PD_MAX_ACTIVE_JOBS must be an integer from 1 to 32') from None
    if str(result) != str(value).strip() or not 1 <= result <= 32:
        raise ValueError('RADAR_PD_MAX_ACTIVE_JOBS must be an integer from 1 to 32')
    return result


def _history_lock(service):
    key = (service.galaxy_url.rstrip('/'), service.history_id)
    with _LOCKS_LOCK:
        return _LOCKS.setdefault(key, threading.Lock())


class HistoryJobAdmission:
    def __init__(self, service, *, total_limit=None, timeout=7200, poll_seconds=3,
                 fetch=None, clock=time.monotonic, wait=None):
        self.service = service
        self.total_limit = configured_limit() if total_limit is None else total_limit
        if type(self.total_limit) is not int or not 1 <= self.total_limit <= 32:
            raise ValueError('Use a total scientific-job limit from 1 to 32')
        self.timeout, self.poll_seconds = timeout, poll_seconds
        self.fetch, self.clock, self.wait = fetch or requests.get, clock, wait or time.sleep
        self._operations = {}
        self.lock = _history_lock(service)
        with _LOCKS_LOCK:
            self.waiters = _WAITERS.setdefault((service.galaxy_url.rstrip('/'), service.history_id), [])

    def active_jobs(self):
        try:
            response = self.fetch(self.service.galaxy_url.rstrip('/') + '/api/jobs',
                headers=self.service._headers,
                params={'history_id': self.service.history_id,
                        'state': sorted(ACTIVE_STATES), 'limit': 1001}, timeout=30,
                allow_redirects=False)
            response.raise_for_status()
            rows = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise AdmissionUnavailable('Could not verify scientific-job capacity; nothing submitted') from exc
        if not isinstance(rows, list) or len(rows) > 1000:
            raise AdmissionUnavailable('Galaxy did not return a complete scientific-job queue')
        jobs = {}
        for row in rows:
            if not isinstance(row, dict):
                raise AdmissionUnavailable('Galaxy returned an invalid scientific-job queue')
            # Do not charge other tools or terminal jobs against this limit.
            tool = row.get('tool_id', '')
            if (not isinstance(tool, str) or not tool.startswith('neutrons_radar_pd_')
                    or row.get('state') not in ACTIVE_STATES):
                continue
            history = row.get('history_id')
            if history is not None and str(history) != str(self.service.history_id):
                raise AdmissionUnavailable('Galaxy returned a job from another history')
            uid = row.get('id')
            if not isinstance(uid, str) or not uid:
                raise AdmissionUnavailable('A queued scientific job has no identity')
            # The viewer is not a scientific batch, and must not consume its own
            # only slot. Hosted GSAS-II desktops are likewise separate sessions.
            if tool in {'neutrons_radar_pd_interactive_prototype',
                        'neutrons_radar_pd_gsasii_interactive_prototype'}:
                continue
            jobs[uid] = row
        return list(jobs.values())

    def _operation(self, job):
        from .preparation_limits import CPU_TRAINING_TOOL_ID,CUDA_TRAINING_TOOL_ID,TRAINING_OPERATION
        if job['tool_id']==CUDA_TRAINING_TOOL_ID:return TRAINING_OPERATION
        if job['tool_id']!=CPU_TRAINING_TOOL_ID:return None
        uid=job['id']
        if uid not in self._operations:
            try:
                response=self.fetch(self.service.galaxy_url.rstrip('/')+'/api/jobs/'+uid,
                    headers=self.service._headers,timeout=30,allow_redirects=False)
                response.raise_for_status();details=response.json()
                if (not isinstance(details,dict) or details.get('id')!=uid
                        or details.get('history_id')!=self.service.history_id
                        or details.get('tool_id')!=job['tool_id']):
                    raise ValueError('job identity')
                inputs=details.get('params') or details.get('parameters') or {}
                from .galaxy_service import _parameter_value,_scalar
                if _scalar(_parameter_value(inputs,'action')) is None:
                    raise AdmissionUnavailable('The active model job has no verifiable operation yet')
                self._operations[uid]=operation_from_inputs(self.service,inputs,fetch=self.fetch)
            except (requests.RequestException,ValueError) as exc:
                raise AdmissionUnavailable('Could not verify the active training queue; nothing submitted') from exc
        return self._operations[uid]

    def _family_count(self,jobs,tool_id,operation):
        from .preparation_limits import TRAINING_OPERATION
        if operation==TRAINING_OPERATION:
            return sum(self._operation(row)==TRAINING_OPERATION for row in jobs)
        return sum(row['tool_id']==tool_id for row in jobs)

    def available_capacity(self,tool_id,*,family_limit,operation=None):
        jobs=self.active_jobs()
        return max(0,min(self.total_limit-len(jobs),
            family_limit-self._family_count(jobs,tool_id,operation)))

    @contextmanager
    def slot(self, tool_id, *, family_limit, cancel=None, operation=None):
        if type(family_limit) is not int or family_limit < 1:
            raise ValueError('A positive per-workflow job limit is required')
        started = self.clock()
        ticket = dict(tool=tool_id, family_limit=family_limit, total_limit=self.total_limit, cancel=cancel, operation=operation)
        enqueued = False
        try:
            while True:
                if cancel is not None and cancel.is_set():
                    raise AdmissionUnavailable('Scientific submission paused before dispatch')
                acquired = self.lock.acquire(timeout=min(self.poll_seconds, .25))
                if acquired:
                    try:
                        if not enqueued:
                            self.waiters.append(ticket); enqueued = True
                        jobs = self.active_jobs()
                        # Oldest eligible request wins. A family already at its
                        # own limit must not block another workflow's free slot.
                        eligible=None
                        for waiter in self.waiters:
                            if (waiter['cancel'] is not None and waiter['cancel'].is_set()
                                    or len(jobs)>=waiter['total_limit']):continue
                            try:count=self._family_count(jobs,waiter['tool'],waiter['operation'])
                            except AdmissionUnavailable:
                                if waiter is ticket:raise
                                # An unverifiable trainer must not stop an
                                # unrelated request using known shared capacity.
                                continue
                            if count<waiter['family_limit']:
                                eligible=waiter;break
                        if eligible is ticket:
                            if cancel is not None and cancel.is_set():
                                raise AdmissionUnavailable('Scientific submission paused before dispatch')
                            self.waiters[:] = [w for w in self.waiters if w is not ticket]
                            enqueued = False
                            # The caller performs the POST before releasing the lock.
                            yield
                            return
                    finally:
                        self.lock.release()
                if self.clock() - started >= self.timeout:
                    raise AdmissionUnavailable('Waiting for an existing scientific job to release capacity')
                if cancel is not None:
                    cancel.wait(self.poll_seconds)
                else:
                    self.wait(self.poll_seconds)
        finally:
            if enqueued:
                with self.lock:
                    # Identity, not dict equality: two requests can have the
                    # same tool/settings and still be separate queue entries.
                    self.waiters[:] = [w for w in self.waiters if w is not ticket]
