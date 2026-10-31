"""Bound acknowledged checkpoint waits without reposting a queued upload."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import threading
import time


class CheckpointUploadPending(RuntimeError):
    def __init__(self, dataset_id, state):
        self.dataset_id, self.state = dataset_id, state
        super().__init__(
            f'Galaxy checkpoint {dataset_id} is still {state}. No replacement upload was submitted. '
            'The latest state remains in this session; retry after the upload queue recovers.'
        )


class CheckpointUploadFailed(RuntimeError):
    pass


class CheckpointUploads:
    """Serialize each checkpoint stream, retaining its acknowledged HDA.

    Different streams can upload concurrently. An older pending snapshot must
    settle before its newer replacement is sent. Identical retry attempts poll
    the existing HDA, including after a timeout; a changed state gets a new HDA
    only after the preceding upload has reached a verified terminal state.
    """
    def __init__(self, upload, status, *, clock=time.monotonic, sleep=time.sleep):
        self.upload, self.status = upload, status
        self.clock, self.sleep = clock, sleep
        self._guard = threading.Lock()
        self._locks, self._latest = {}, {}

    def _wait(self, row, deadline):
        state = 'awaiting status'
        while True:
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise CheckpointUploadPending(row['id'], state)
            try:
                metadata = self.status(row['id'], min(15., remaining))
            except Exception:
                # An acknowledged upload remains ambiguous after a status-read
                # failure. Retain it; never turn that failure into another POST.
                raise CheckpointUploadPending(row['id'], 'awaiting status') from None
            if not isinstance(metadata, dict) or not isinstance(metadata.get('state'), str):
                raise CheckpointUploadPending(row['id'], 'awaiting status')
            if metadata.get('id') is not None and metadata['id'] != row['id']:
                raise CheckpointUploadPending(row['id'], 'awaiting matching status')
            state = metadata['state']
            if metadata.get('deleted') or metadata.get('purged'):
                state = 'deleted'
            if state == 'ok':
                return row['id']
            if state in {'error', 'discarded', 'deleted', 'empty', 'failed_metadata'}:
                raise CheckpointUploadFailed(f'Galaxy checkpoint {row["id"]} ended in state {state}')
            if state not in {'new', 'upload', 'uploading', 'queued', 'running', 'waiting',
                             'setting_metadata', 'paused', 'deferred', 'resubmitted'}:
                state = 'awaiting status'
            self.sleep(min(2., max(0., deadline - self.clock())))

    def save(self, path, *, label, semantic_sha256=None, timeout=180.):
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 600:
            raise ValueError('Choose a bounded checkpoint wait of at most 600 seconds')
        if semantic_sha256 is not None and (not isinstance(semantic_sha256, str) or not re.fullmatch('[a-f0-9]{64}', semantic_sha256)):
            raise ValueError('Invalid checkpoint content identity')
        checksum = semantic_sha256 or hashlib.sha256(Path(path).read_bytes()).hexdigest()
        with self._guard:
            lock = self._locks.setdefault(label, threading.Lock())
        with lock:
            deadline = self.clock() + timeout
            prior = self._latest.get(label)
            if prior:
                try:
                    self._wait(prior, deadline)
                except CheckpointUploadFailed:
                    self._latest.pop(label, None)
                    raise
                if prior['sha256'] == checksum:
                    return prior['id']
            if self.clock() >= deadline:
                raise CheckpointUploadPending(prior['id'], 'ready; a newer snapshot is retained locally')
            identifier = str(self.upload(path, label=label))
            if not re.fullmatch('[A-Za-z0-9_-]{1,128}', identifier) or identifier == 'None':
                raise ValueError('Galaxy did not return a checkpoint dataset identity')
            row = dict(id=identifier, sha256=checksum)
            self._latest[label] = row
            try:
                return self._wait(row, deadline)
            except CheckpointUploadFailed:
                self._latest.pop(label, None)
                raise
