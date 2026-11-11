"""Resumable, data-only experiment libraries with explicit finite coverage."""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from functools import wraps
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import uuid
import zipfile

from ..execution import write_json
from ..locking import file_lock, process_lock
from ..symmetry.contracts import digest

SCHEMA = 'radar-experiment-preparation-library/v1'
STATES = {'pending', 'preparing', 'prepared', 'unsupported', 'failed', 'cancelled'}
TERMINAL = STATES-{'pending', 'preparing'}
SUFFIXES = {'.json', '.npz', '.cif', '.instprm', '.prm', '.ins', '.inst', '.txt'}
# The measured complete paired corpus occupies 3.19 GB before model/native
# assets. Working data and a portable analysis copy have different lifetimes.
# Keep the untrusted archive/extraction budget at 2 GiB; bound the complete
# working corpus separately rather than thinning its physical trajectories.
MAX_BYTES = 8*1024**3
MAX_ARCHIVE_BYTES = 2*1024**3
MAX_FILES = 20000
MAX_MODELS = 5000
MAX_METADATA = 8*1024**2


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def relative(value):
    if (not isinstance(value, str) or not value or len(value) > 500 or '\\' in value
            or '\x00' in value or ':' in value):
        raise ValueError('Use a confined preparation asset path')
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or str(p) != value or p.suffix.lower() not in SUFFIXES:
        raise ValueError('Only normalized, data-only preparation paths are allowed')
    return value


def confined(root, value, *, exists=True):
    root = Path(root).resolve()
    target = root/relative(value)
    if target.is_symlink() or not target.resolve(strict=exists).is_relative_to(root):
        raise ValueError('Preparation asset escaped its library')
    return target


def _asset(value):
    if (not isinstance(value, dict) or set(value) != {'path', 'sha256'}
            or not isinstance(value['sha256'], str) or not re.fullmatch('[a-f0-9]{64}', value['sha256'])):
        raise ValueError('Preparation assets require an exact file checksum')
    relative(value['path'])
    return dict(value)


def validate_manifest(value):
    required={'schema','id','name','specification','specification_sha256','inputs','entries','observations_included','scientific_result'}
    if (not isinstance(value, dict) or set(value)-required-{'evidence', 'storage'} or required-set(value)
            or value['schema'] != SCHEMA or value['observations_included'] is not False
            or value['scientific_result'] is not False
            or not isinstance(value['name'], str) or not 1 <= len(value['name']) <= 200
            or not isinstance(value['specification'], dict)
            or digest(value['specification']) != value['specification_sha256']
            or not isinstance(value['id'], str)):
        raise ValueError('Invalid experiment preparation manifest')
    if 'evidence' in value:
        from .evidence import validate_references
        validate_references(value['evidence'])
    entries = value['entries']
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_MODELS:
        raise ValueError('Declare a bounded, nonempty preparation scope')
    seen = set()
    for row in entries:
        if (not isinstance(row, dict) or set(row) != {'id', 'model', 'state', 'attempt', 'reason', 'files'}
                or row['id'] != digest(row['model'])[:24] or row['id'] in seen
                or row['state'] not in STATES or type(row['attempt']) is not int or row['attempt'] < 0
                or not isinstance(row['reason'], str) or len(row['reason']) > 2000
                or not isinstance(row['model'], dict)
                or row['model'].get('kind') not in {'reference', 'magnetic', 'structural',
                                                 'structural-examples','structural-training','structural-ranking-model'}
                or not isinstance(row['model'].get('bank'), str) or not row['model']['bank']
                or not isinstance(row['files'], list) or len(row['files']) > 200):
            raise ValueError('Invalid or duplicate preparation model record')
        seen.add(row['id'])
        paths = set()
        for asset in row['files']:
            asset = _asset(asset)
            if not asset['path'].startswith('entries/'+row['id']+'/') or asset['path'] in paths:
                raise ValueError('Prepared model assets must belong to that exact entry')
            paths.add(asset['path'])
        if (row['state'] == 'prepared') != bool(row['files']):
            raise ValueError('Only fully prepared models can publish their data artifacts')
        if row['model']['kind']=='structural-ranking-model':
            from .compiled_ranker import validate_entry
            validate_entry(row)
    if not isinstance(value['inputs'], list) or not value['inputs'] or len(value['inputs']) > 10000:
        raise ValueError('Preparation inputs are missing or oversized')
    inputs = [_asset(a) for a in value['inputs']]
    paths = [a['path'] for a in inputs]
    if len(paths) != len(set(p.casefold() for p in paths)) or any(not p.startswith('inputs/') for p in paths):
        raise ValueError('Preparation inputs must have unique captured paths')
    identity = digest(dict(specification=value['specification'], inputs=inputs,
                           models=[r['model'] for r in entries]))
    if value['id'] != identity[:24]:
        raise ValueError('Preparation identity differs from its exact inputs and model scope')
    if any(row['model']['kind'] in {'structural-examples','structural-training'} for row in entries):
        from .structural_dependencies import validate_entries
        validate_entries(entries)
    from .compiled_storage import validate
    validate(value)
    # JSON values, including nested model/specification fields, remain data.
    # Match the actual stored bytes, including the newline. Pretty-printing
    # repeated native mode inventories could pass admission and then exceed
    # the reader limit, making a newly created multi-bank library unreadable.
    if len((json.dumps(value,separators=(',', ':'),allow_nan=False)+'\n').encode('utf-8')) > MAX_METADATA:
        raise ValueError('Preparation manifest exceeds its metadata budget')
    return value


def _locked_read(method):
    @wraps(method)
    def read(self, *args, **kwargs):
        with process_lock(self.root/'.library.lock'):
            return method(self, *args, **kwargs)
    return read


class PreparationLibrary:
    """One durable model/bank denominator; retries never erase past coverage."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.path = self.root/'library.json'
        if not self.path.is_file() or self.path.stat().st_size > MAX_METADATA:
            raise ValueError('Preparation library manifest is absent or oversized')
        self.document = validate_manifest(json.loads(self.path.read_text()))
        self.verify()

    @classmethod
    def create(cls, root, *, name, specification, models, inputs):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        captured = []
        try:
            for destination, source in inputs.items():
                destination = 'inputs/'+relative(destination)
                target = confined(root, destination, exists=False)
                source = Path(source)
                if not source.is_file() or source.is_symlink():
                    raise ValueError('Capture a regular scientific input file')
                before = sha(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                if sha(source) != before or sha(target) != before:
                    raise ValueError('Preparation input changed during capture')
                captured.append(dict(path=destination, sha256=before))
            specification_sha256 = digest(specification)
            identity = digest(dict(specification=specification, inputs=captured, models=models))
            rows = [dict(id=digest(m)[:24], model=m, state='pending', attempt=0, reason='', files=[])
                    for m in models]
            document = dict(schema=SCHEMA, id=identity[:24], name=name,
                specification=specification, specification_sha256=specification_sha256,
                inputs=captured, entries=rows, observations_included=False, scientific_result=False)
            validate_manifest(document)
            write_json(root/'library.json', document, compact=True)
            return cls(root)
        except Exception:
            shutil.rmtree(root)
            raise

    @_locked_read
    def verify(self):
        validate_manifest(self.document)
        if digest(self._read_document()) != digest(self.document):
            raise ValueError('Preparation state changed; refresh its durable manifest')
        self._verify_artifacts()
        return True

    def _read_document(self):
        if (self.path.is_symlink() or not self.path.is_file()
                or self.path.stat().st_size > MAX_METADATA):
            raise ValueError('Preparation library manifest is absent or oversized')
        return validate_manifest(json.loads(self.path.read_text()))

    @_locked_read
    def refresh(self):
        """Read durable state; state inspection does not consume scientific arrays.

        Opening/importing/exporting a library checks its full inventory. Work
        dispatch and consumption additionally verify their exact dependencies
        through ``verify_assets``. State-only transitions must not repeatedly
        hash and decompress every unrelated completed model.
        """
        self.document = self._read_document()
        return self.document

    def _assets(self):
        return self.document['inputs']+[a for r in self.document['entries'] for a in r['files']]

    def _check_budget(self, extra_paths=()):
        # Retain the aggregate size and confinement checks at every publication.
        # This inspects metadata; it never reads/decompresses unrelated arrays.
        total = len((json.dumps(self.document, separators=(',', ':'), allow_nan=False)+'\n').encode())
        from .compiled_storage import deferred
        archived = deferred(self.document)
        names = list(dict.fromkeys([a['path'] for a in self._assets()]+list(extra_paths)))
        if len(names)+1 > MAX_FILES:
            raise ValueError('Preparation exceeds its portable file-count budget')
        for name in names:
            if name in archived:
                continue
            path = confined(self.root, name)
            if not path.is_file():
                raise ValueError('A preparation artifact changed: '+name)
            total += path.stat().st_size
            if total > (MAX_ARCHIVE_BYTES if archived else MAX_BYTES):
                raise ValueError('Preparation exceeds the library byte budget; split its declared scope')

    def _verify_artifacts(self, assets=None):
        self._check_budget()
        from .compiled_storage import deferred
        archived = deferred(self.document)
        if assets is not None and any(a['path'] in archived for a in assets):
            raise ValueError('Training arrays are archived in the original preparation History; '
                             'this compiled copy contains the model and native fitting assets')
        for asset in self._assets() if assets is None else assets:
            if asset['path'] in archived:
                continue
            path = confined(self.root, asset['path'])
            if not path.is_file() or sha(path) != asset['sha256']:
                raise ValueError('A preparation artifact changed: '+asset['path'])
            if path.suffix == '.npz':
                from ..live.array_limits import validate_npz
                validate_npz(path)

    @_locked_read
    def verify_assets(self, names):
        """Check the exact published inputs/outputs a calculation will consume."""
        self.refresh()
        inventory = {a['path']: a for a in self._assets()}
        names = list(names)
        if (any(not isinstance(n, str) for n in names) or len(names) != len(set(names))
                or any(n not in inventory for n in names)):
            raise ValueError('Scientific dependencies must belong to this preparation inventory')
        self._verify_artifacts([inventory[n] for n in names])
        return {n: confined(self.root, n) for n in names}

    @_locked_read
    def prepared_assets(self, identifier):
        """Verify a completed entry before handing its arrays to an evaluator."""
        self.refresh()
        row = self.row(identifier)
        if row['state'] != 'prepared':
            raise ValueError('This preparation entry has no completed scientific artifacts')
        outputs = [a['path'] for a in row['files']]
        dependencies = [self.document['specification'].get('source')]
        dependencies += [row['model'].get(k) for k in ('instrument', 'child')]
        dependencies = [p for p in dependencies if p is not None]
        if not dependencies:
            dependencies = [a['path'] for a in self.document['inputs']]
        verified = self.verify_assets(list(dict.fromkeys(dependencies+outputs)))
        return {name: verified[name] for name in outputs}

    @contextmanager
    def _transaction(self):
        with file_lock(self.root/'.library.lock'):
            self.refresh()
            yield

    def row(self, identifier):
        matches = [r for r in self.document['entries'] if r['id'] == identifier]
        if len(matches) != 1:
            raise ValueError('Model is outside this preparation scope')
        return matches[0]

    def start(self, identifier):
        with self._transaction():
            from .compiled_storage import require_working_copy
            row = self.row(identifier)
            require_working_copy(self.document, row['model'])
            if row['state'] != 'pending':
                raise ValueError('Only a pending preparation entry can start')
            row.update(state='preparing', attempt=row['attempt']+1, reason='', files=[])
            self._save()
            return row['attempt']

    def finish(self, identifier, attempt, *, state, files=(), reason=''):
        with self._transaction():
            row = self.row(identifier)
            if row['state'] != 'preparing' or row['attempt'] != attempt:
                raise ValueError('Late preparation result belongs to a different entry attempt')
            if state not in TERMINAL:
                raise ValueError('A preparation result needs an explicit terminal state')
            files = list(files)
            for name in files:
                relative(name)
                if not name.startswith('entries/'+identifier+'/'):
                    raise ValueError('Preparation output belongs to another model')
            self._check_budget(files)
            copied = []
            for name in files:
                target = confined(self.root, name)
                copied.append(dict(path=name, sha256=sha(target)))
            self._verify_artifacts(copied)
            old = dict(row)
            row.update(state=state, files=copied, reason=reason)
            try:
                self._save()
            except Exception:
                row.clear(); row.update(old)
                raise

    def retry(self, identifier):
        with self._transaction():
            from .compiled_storage import require_working_copy
            row = self.row(identifier)
            require_working_copy(self.document, row['model'])
            if row['state'] not in {'failed', 'cancelled', 'unsupported'}:
                raise ValueError('Only a reviewed terminal failure can be retried')
            row.update(state='pending', reason='', files=[])
            self._save()

    def add_evidence(self, reference):
        """Persist a routing hint; the native launcher verifies actual origin."""
        import copy
        from .evidence import validate_references
        validate_references([reference])
        with self._transaction():
            values=copy.deepcopy(self.document.get('evidence',[]))
            previous=[r for r in values if (r['site'],r['dataset_id'])==(reference['site'],reference['dataset_id'])]
            if previous:
                if previous[0]['sha256']!=reference['sha256']:raise ValueError('An origin receipt dataset changed')
                previous[0]['entries']=sorted(set(previous[0]['entries'])|set(reference['entries']))
            else:values.append(copy.deepcopy(reference))
            validate_references(values)
            previous_document=copy.deepcopy(self.document)
            self.document['evidence']=values
            try:self._save()
            except Exception:
                self.document=previous_document
                raise

    def interrupt(self, identifier, attempt):
        # The caller must first establish that the old job/process is terminal.
        self.finish(identifier, attempt, state='cancelled', reason='Preparation interrupted; retry is explicit')

    def detach_imported_work(self):
        """Branch an imported copy without claiming ownership of its old jobs.

        Call only on a fresh portable import, before attaching a local queue.
        This does not cancel, retry or change any originating Galaxy job. A
        later explicit start creates new attempts for this independent copy.
        Same-History recovery must instead retain the original task handles.
        """
        if (self.root/'queue/queue.json').exists():
            raise ValueError('Detach only a fresh portable copy; recover owned jobs through their History checkpoint')
        with self._transaction():
            count=0
            for row in self.document['entries']:
                if row['state']=='preparing':
                    row.update(state='pending',reason='Imported branch; original job ownership was not transferred.')
                    count+=1
            if count:self._save()
            return count

    @_locked_read
    def coverage(self):
        self.refresh()
        counts = {state: 0 for state in sorted(STATES)}
        banks = {}; families = {}
        for row in self.document['entries']:
            counts[row['state']] += 1
            for table, key in ((banks, row['model']['bank']),
                    (families, json.dumps([row['model']['kind'], row['model'].get('k'), row['model'].get('irrep')]))):
                totals = table.setdefault(key, {state: 0 for state in sorted(STATES)})
                totals[row['state']] += 1
        return dict(total=len(self.document['entries']), counts=counts, banks=banks, families=families,
            complete=not (counts['pending'] or counts['preparing']),
            all_prepared=counts['prepared'] == len(self.document['entries']))

    def _save(self):
        validate_manifest(self.document)
        self._check_budget()
        write_json(self.path, self.document, compact=True)

    def export(self, destination):
        with self._transaction():
            return self._export_snapshot(destination)

    def _export_snapshot(self, destination):
        self.verify()
        from .compiled_storage import snapshot, deferred
        document = snapshot(self.document)
        validate_manifest(document)
        archived = deferred(document)
        paths = [a['path'] for a in self._assets() if a['path'] not in archived]
        raw = (json.dumps(document, separators=(',', ':'), allow_nan=False)+'\n').encode()
        if len(raw)+sum(confined(self.root, name).stat().st_size for name in paths) > MAX_ARCHIVE_BYTES:
            raise ValueError('Preparation exceeds the portable byte budget. Complete training '
                             'to export its compiled model, or split the declared native scope.')
        temporary = Path(destination).with_name(Path(destination).name+'.'+uuid.uuid4().hex+'.partial')
        try:
            with zipfile.ZipFile(temporary, 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                archive.writestr('library.json', raw)
                for name in paths:
                    archive.write(confined(self.root, name), name)
            if temporary.stat().st_size > MAX_ARCHIVE_BYTES:
                raise ValueError('Preparation archive exceeds its portable byte budget')
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def import_library(archive, destination):
    """Validate the entire inventory before extracting any scientific input."""
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Import into a fresh preparation directory')
    if Path(archive).stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError('Preparation archive exceeds its byte budget')
    temporary = destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.partial')
    temporary.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive) as packet:
            members = packet.infolist()
            names = [m.filename for m in members]
            if (not members or len(members) > MAX_FILES or len(set(n.casefold() for n in names)) != len(names)
                    or sum(m.file_size for m in members) > MAX_ARCHIVE_BYTES):
                raise ValueError('Preparation archive exceeds its declared data budget')
            for member in members:
                relative(member.filename)
                if (member.is_dir() or stat.S_ISLNK(member.external_attr >> 16)
                        or member.flag_bits & 1):
                    raise ValueError('Preparation archive contains an unsupported entry')
            manifest_info = packet.getinfo('library.json')
            if manifest_info.file_size > MAX_METADATA:
                raise ValueError('Preparation manifest exceeds its metadata budget')
            document = validate_manifest(json.loads(packet.read('library.json')))
            expected = {'library.json'} | {a['path'] for a in document['inputs']}
            expected.update(a['path'] for r in document['entries'] for a in r['files'])
            from .compiled_storage import deferred
            expected -= deferred(document)
            if expected != set(names):
                raise ValueError('Preparation archive differs from its exact artifact inventory')
            for name in names:
                target = confined(temporary, name, exists=False)
                target.parent.mkdir(parents=True, exist_ok=True)
                with packet.open(name) as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output, 1024*1024)
        checked = PreparationLibrary(temporary)
        temporary.replace(destination)
        # Atomic directory relocation changes no scientific bytes. Reuse the
        # completed full inventory check instead of hashing/decompressing the
        # entire imported library a second time at its new local pathname.
        checked.root = destination.resolve()
        checked.path = checked.root/'library.json'
        return checked
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
