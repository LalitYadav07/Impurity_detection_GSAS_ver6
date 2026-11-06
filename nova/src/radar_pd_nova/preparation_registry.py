"""One bounded preparation collection for setup, model and live workflows.

The index stores scientific library identities and relative local locations.
It contains no service credentials, executable state or fitted observations.
Portable input remains the preparation library's checked data-only archive.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import threading
import uuid

from gpx_recipe.execution import write_json
from gpx_recipe.preparation.library import PreparationLibrary, import_library, validate_manifest
from gpx_recipe.preparation.operations import create_library

SCHEMA = 'radar-preparation-registry/v1'
MAX_LIBRARIES = 128


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch('[a-f0-9]{32}', value):
        raise ValueError('Select a registered preparation identity')
    return value


class PreparationRegistry:
    """Durable references; an imported revision never overwrites prior work."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root/'registry.json'
        self._lock = threading.RLock()
        self._loaded = OrderedDict()
        self._summaries = {}
        with self._transaction():
            if not self.path.exists():
                write_json(self.path, dict(schema=SCHEMA, selected='', items=[]))
            self._read()

    def _read(self):
        if self.path.is_symlink() or self.path.stat().st_size > 1024*1024:
            raise ValueError('Preparation index is absent or oversized')
        value = json.loads(self.path.read_text())
        if (not isinstance(value, dict) or set(value) != {'schema', 'selected', 'items'}
                or value['schema'] != SCHEMA or not isinstance(value['items'], list)
                or not isinstance(value['selected'], str)
                or len(value['items']) > MAX_LIBRARIES):
            raise ValueError('Invalid preparation index')
        seen = set()
        for row in value['items']:
            if (not isinstance(row, dict) or set(row) != {'id', 'library_id', 'origin', 'created'}
                    or not isinstance(row['origin'], str) or row['origin'] not in {'created', 'imported'}
                    or not isinstance(row['library_id'], str)
                    or not re.fullmatch('[a-f0-9]{24}', row['library_id'])
                    or not isinstance(row['created'], str) or len(row['created']) > 60):
                raise ValueError('Invalid registered preparation')
            key = identifier(row['id'])
            if key in seen:
                raise ValueError('Duplicate registered preparation')
            seen.add(key)
        if value['selected'] != '' and value['selected'] not in seen:
            raise ValueError('The selected preparation is outside this index')
        return value

    @contextmanager
    def _transaction(self):
        import fcntl
        with self._lock, (self.root/'.registry.lock').open('a+b') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _directory(self, key):
        directory = self.root/'libraries'/identifier(key)
        if directory.is_symlink() or not directory.resolve().is_relative_to(self.root):
            raise ValueError('Registered preparation escaped its workspace')
        return directory

    def _row(self, key):
        identifier(key)
        rows = [r for r in self._read()['items'] if r['id'] == key]
        if len(rows) != 1:
            raise ValueError('Preparation is not registered in this workspace')
        return rows[0]

    def _register(self, key, library, origin):
        with self._transaction():
            state = self._read()
            if len(state['items']) >= MAX_LIBRARIES:
                raise ValueError('This workspace has reached its preparation-library limit')
            if any(r['id'] == key for r in state['items']):
                raise ValueError('Preparation is already registered')
            # Factories already checked the complete created/imported library.
            # Registration publishes only its validated metadata identity.
            # Consumption and export retain their scientific integrity checks.
            library.refresh()
            state['items'].append(dict(id=key, library_id=library.document['id'], origin=origin,
                created=datetime.now(timezone.utc).isoformat()))
            state['selected'] = key
            write_json(self.path, state)
            self._loaded[key] = library
            self._trim()
        return key

    def _trim(self):
        # Only metadata objects are cached. Numeric arrays are loaded by the
        # scientific evaluator under its separate allocation budget.
        while len(self._loaded) > 4:
            self._loaded.popitem(last=False)

    def _capacity(self):
        if len(self._read()['items']) >= MAX_LIBRARIES:
            raise ValueError('This workspace has reached its preparation-library limit')

    def _discard_unregistered(self, key, target):
        # A failed write acknowledgment can follow an atomic index publication.
        # Keep the completed library if it is already indexed, or if the index
        # itself cannot be read well enough to establish safe cleanup.
        try:
            registered = any(r['id'] == key for r in self._read()['items'])
        except (OSError, ValueError):
            return
        if not registered and target.exists():
            shutil.rmtree(target)

    def create(self, *, name, source, banks, models, scope=None):
        self._capacity()
        key = uuid.uuid4().hex
        target = self._directory(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            library = create_library(target, name=name, source=source, banks=banks, models=models,
                **({'scope':scope} if scope is not None else {}))
            return self._register(key, library, 'created')
        except Exception:
            self._discard_unregistered(key, target)
            raise

    def import_archive(self, archive):
        self._capacity()
        key = uuid.uuid4().hex
        target = self._directory(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            library = import_library(archive, target)
            return self._register(key, library, 'imported')
        except Exception:
            self._discard_unregistered(key, target)
            raise

    def get(self, key=None):
        with self._lock:
            key = key or self._read()['selected']
            row = self._row(key)
            library = self._loaded.get(key)
            if library is None:
                library = PreparationLibrary(self._directory(key))
                self._loaded[key] = library
            library.refresh()
            if library.document['id'] != row['library_id']:
                raise ValueError('The registered preparation changed its captured scientific scope')
            self._loaded.move_to_end(key);self._trim()
            return library

    def select(self, key):
        self.get(key)
        with self._transaction():
            state = self._read();self._row(key)
            state['selected'] = key
            write_json(self.path, state)

    def summaries(self):
        """Inspect state metadata without rereading every numerical artifact."""
        rows = []
        for row in self._read()['items']:
            try:
                path = self._directory(row['id'])/'library.json'
                if path.is_symlink() or path.stat().st_size > 8*1024*1024:
                    raise ValueError('Preparation state is absent or oversized')
                stat=path.stat();stamp=(stat.st_ino,stat.st_size,stat.st_mtime_ns,row['library_id'])
                cached=self._summaries.get(row['id'])
                if cached and cached[0]==stamp:
                    rows.append(copy.deepcopy(cached[1]));continue
                document = validate_manifest(json.loads(path.read_text()))
                if document['id'] != row['library_id']:
                    raise ValueError('A preparation changed its registered scientific scope')
                counts = dict(Counter(e['state'] for e in document['entries']))
                summary=dict(row,name=document['name'],total=len(document['entries']),
                    prepared=counts.get('prepared',0),counts=counts,available=True,message='',
                    kinds=sorted({e['model']['kind'] for e in document['entries']}))
                self._summaries[row['id']]=(stamp,copy.deepcopy(summary));rows.append(summary)
            except (OSError, ValueError):
                rows.append(dict(row, name='Preparation needs review', total=None,
                    prepared=None, counts={}, kinds=[], available=False,
                    message='The saved preparation is missing or changed. Import a verified copy to recover it.'))
        return rows

    def export(self, key, destination):
        self.get(key).export(destination)
        return Path(destination)
