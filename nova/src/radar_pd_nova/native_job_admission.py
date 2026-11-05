"""Shared submission admission for scientific jobs in one Galaxy history.

This controls requests made by this NOVA process. Galaxy remains responsible
for admission across viewers and for actual CPU and memory allocation. A slot
is held only across the capacity check and submission, never during a fit.
"""
from contextlib import contextmanager
import os
import threading
import time

import requests

ACTIVE_STATES = frozenset({'new', 'upload', 'waiting', 'queued', 'running',
                           'resubmitted', 'paused', 'setting_metadata'})
_LOCKS = {}
_WAITERS = {}
_LOCKS_LOCK = threading.Lock()


class AdmissionUnavailable(RuntimeError):
    pass


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

    @contextmanager
    def slot(self, tool_id, *, family_limit, cancel=None):
        if type(family_limit) is not int or family_limit < 1:
            raise ValueError('A positive per-workflow job limit is required')
        started = self.clock()
        ticket = dict(tool=tool_id, family_limit=family_limit, total_limit=self.total_limit, cancel=cancel)
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
                        eligible = next((w for w in self.waiters if
                            not (w['cancel'] is not None and w['cancel'].is_set()) and
                            len(jobs) < w['total_limit'] and
                            sum(r['tool_id'] == w['tool'] for r in jobs) < w['family_limit']), None)
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
