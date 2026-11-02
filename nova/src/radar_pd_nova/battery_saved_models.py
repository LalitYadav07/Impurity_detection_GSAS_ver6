"""Read-only discovery and explicit recovery of published battery preparations.

The catalog reads bounded job metadata. Scientific files are downloaded only
after Open, from the captured History and original successful preparation job.
No method here submits, retries, cancels, or invokes a scientific worker.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from urllib.parse import urlsplit
import zipfile

import requests

from .refinement_sources import digest, file_sha
from .refinement_runtime import REFINEMENT_TOOL_ID, WORKSPACE_LABEL

MAX_JOBS = 200
PAGE_SIZE = 50
MAX_JSON = 8 * 1024 * 1024
MAX_REQUEST = 2 * 1024 * 1024
MAX_PROJECT = 128 * 1024 * 1024
MAX_BUNDLE = 768 * 1024 * 1024
MAX_EXPANDED = 1536 * 1024 * 1024


class StaleSavedModel(RuntimeError):
    """The explicit selection no longer owns this read."""


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', value):
        raise ValueError('Published preparation has an invalid Galaxy identifier')
    return value


def service_identity(service):
    return (str(getattr(service, 'galaxy_url', '') or '').rstrip('/'),
            str(getattr(service, 'history_id', '') or ''),
            dict(getattr(service, '_headers', {}) or {}))


class SavedModelReader:
    """All requests retain the URL, actor and History captured at the click."""

    def __init__(self, service, *, current=lambda: True):
        self.service = service
        self.galaxy_url, self.history_id, self.headers = service_identity(service)
        parsed = urlsplit(self.galaxy_url)
        if (parsed.scheme not in {'http', 'https'} or not parsed.netloc or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError('Configure a valid Galaxy connection to find saved models')
        _identifier(self.history_id)
        self.current = current

    def check(self):
        if not self.current() or service_identity(self.service) != (self.galaxy_url, self.history_id, self.headers):
            raise StaleSavedModel('Saved-model selection changed; choose the current model again')

    def _read(self, path, *, params=None, maximum=MAX_JSON, destination=None):
        self.check()
        try:
            response = requests.get(self.galaxy_url + path, headers=self.headers,
                                    params=params, allow_redirects=False, stream=True, timeout=30)
            try:
                if response.status_code != 200:
                    raise ValueError('Galaxy could not read this saved-model record')
                self.check()
                data = bytearray() if destination is None else None
                total = 0
                handle = Path(destination).open('xb') if destination is not None else None
                try:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        self.check()
                        total += len(chunk)
                        if total > maximum:
                            raise ValueError('Saved-model file exceeds its explicit recovery size limit')
                        if handle is not None: handle.write(chunk)
                        else: data.extend(chunk)
                finally:
                    if handle is not None: handle.close()
                self.check()
                return bytes(data) if data is not None else total
            finally:
                response.close()
        except requests.RequestException as exc:
            raise ValueError('Galaxy saved-model read failed; the current model was not replaced') from exc

    def json(self, path, *, params=None, maximum=MAX_JSON):
        try:
            return json.loads(self._read(path, params=params, maximum=maximum))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError('Galaxy returned an unreadable saved-model record') from exc

    def dataset(self, ident, maximum):
        ident = _identifier(ident)
        row = self.json('/api/datasets/' + ident)
        size = row.get('file_size') if isinstance(row, dict) else None
        if (not isinstance(row, dict) or row.get('id') != ident or row.get('history_id') != self.history_id
                or row.get('state') != 'ok' or row.get('deleted') or row.get('purged')
                or type(size) is not int or not 0 < size <= maximum):
            raise ValueError('Saved-model dataset is missing, unavailable, oversized, or belongs to another History')
        return row

    def download(self, ident, destination, maximum):
        row = self.dataset(ident, maximum)
        actual = self._read('/api/datasets/' + ident + '/display', params={'raw': 'true'},
                            maximum=maximum, destination=destination)
        if actual != row['file_size']:
            raise ValueError('Saved-model dataset size changed during recovery')
        return file_sha(Path(destination))

    def _job(self, ident):
        from .galaxy_service import _parameter_value
        ident = _identifier(ident)
        row = self.json('/api/jobs/' + ident, params={'full': 'true'})
        if (not isinstance(row, dict) or row.get('id') != ident or row.get('history_id') != self.history_id
                or row.get('tool_id') != REFINEMENT_TOOL_ID or row.get('state') != 'ok'
                or row.get('deleted') or row.get('copied_from_job_id')):
            raise ValueError('Choose an original, completed preparation in this History')
        action = _parameter_value(row.get('params') or row.get('parameters') or {}, 'action')
        if action != 'prepare': return None
        dispatch = _parameter_value(row.get('params') or row.get('parameters') or {}, 'dispatch_key')
        if not isinstance(dispatch, str) or not re.fullmatch(r'[a-f0-9]{24}', dispatch):
            raise ValueError('Preparation has no exact dispatch identity')
        outputs = self.service._job_output_ids(row)
        inputs = row.get('inputs') or {}
        def input_id(name):
            value = inputs.get(name) or {}
            if value.get('src') != 'hda': raise ValueError('Preparation has no original captured input')
            return _identifier(value.get('id'))
        publication = dict(job_id=ident, history_id=self.history_id,
            dispatch_key=dispatch,
            bundle_id=_identifier(outputs.get('result_bundle')),
            collection_id=_identifier(outputs.get('stage_projects')),
            packet_id=input_id('source_gpx'), request_id=input_id('recipe_request'))
        publication['identity'] = digest(publication)
        publication['created'] = str(row.get('create_time') or '')
        return publication

    def list_models(self):
        jobs = []
        truncated = False
        for offset in range(0, MAX_JOBS, PAGE_SIZE):
            page = self.json('/api/jobs', params=dict(history_id=self.history_id,
                tool_id=REFINEMENT_TOOL_ID, order='create_time-dsc', limit=PAGE_SIZE, offset=offset))
            if not isinstance(page, list): raise ValueError('Galaxy returned an invalid preparation catalog')
            if len(page) > PAGE_SIZE: raise ValueError('Galaxy ignored the bounded preparation catalog request')
            jobs.extend(row for row in page if isinstance(row, dict)
                        and row.get('tool_id') == REFINEMENT_TOOL_ID and row.get('state') == 'ok')
            if len(page) < PAGE_SIZE: break
            truncated = offset + PAGE_SIZE == MAX_JOBS
        ids = [_identifier(row.get('id')) for row in jobs]
        if len(set(ids)) != len(ids): raise ValueError('Galaxy returned overlapping preparation pages; refresh the catalog')
        def one(ident):
            try: return self._job(ident), None
            except StaleSavedModel: raise
            except ValueError: return None, ident
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix='battery-saved') as pool:
            outcomes = list(pool.map(one, ids))
        self.check()
        return ([row for row, _ in outcomes if row], sum(bool(error) for _, error in outcomes), truncated)

    def model_dataset(self, publication):
        collection = self.json('/api/dataset_collections/' + publication['collection_id'], params={'view': 'element'})
        if not isinstance(collection, dict) or collection.get('id') != publication['collection_id']:
            raise ValueError('Preparation collection identity changed')
        if collection.get('history_id') not in (None, self.history_id) or collection.get('deleted'):
            raise ValueError('Preparation collection belongs to another History or was deleted')
        elements = collection.get('elements') or []
        if not isinstance(elements, list) or len(elements) > 256:
            raise ValueError('Preparation collection is invalid or too large')
        matches = [v.get('object') or {} for v in elements
                   if isinstance(v, dict) and v.get('element_identifier') == 'model']
        if len(matches) != 1:
            raise ValueError('Preparation has no unique published model')
        return _identifier(matches[0].get('id'))

    # Minimal read-only service surface used by the existing checkpoint parser.
    def search_history_datasets(self, *, query='', limit=5, include_generated=False):
        if query != WORKSPACE_LABEL or not 1 <= limit <= 5:
            raise ValueError('Saved-model recovery only reads bounded refinement checkpoints')
        rows = self.json('/api/histories/' + self.history_id + '/contents', params={
            'v': 'dev', 'limit': limit, 'offset': 0, 'order': 'update_time-dsc',
            'q': 'name-contains', 'qv': query})
        if not isinstance(rows, list) or len(rows) > limit:
            raise ValueError('Galaxy returned an invalid checkpoint catalog')
        return [r for r in rows if isinstance(r, dict) and r.get('history_content_type') == 'dataset'
                and r.get('state') == 'ok' and not r.get('deleted') and not r.get('purged')]

    def _download_dataset(self, ident, destination):
        self.download(ident, destination, MAX_PROJECT)


def _checked_archive(archive):
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if len(members) > 1024 or sum(m.file_size for m in members) > MAX_EXPANDED:
            raise ValueError('Prepared-model bundle exceeds its recovery budget')
        seen = set()
        for member in members:
            name = member.filename
            path = PurePosixPath(name)
            canonical = path.as_posix() + ('/' if member.is_dir() else '')
            mode = stat.S_IFMT(member.external_attr >> 16)
            if (not name or name != canonical or canonical in seen or path.is_absolute() or '..' in path.parts or '\\' in name
                    or ':' in name or member.flag_bits & 1 or mode not in {0, stat.S_IFREG, stat.S_IFDIR}
                    or member.file_size > MAX_BUNDLE):
                raise ValueError('Prepared-model bundle contains an unsafe or duplicate member')
            seen.add(canonical)


def recover_preparation(reader, publication, root):
    """Recover the original packet and publication; never create a replacement."""
    from gpx_recipe.battery.transport import extract_preparation_packet, MAX_PACKET_BYTES
    from .battery_preparation import verify_prepared, check_prepared_profiles
    from .galaxy_service import _extract_results_archive
    from .refinement_runtime import verify_result_files
    current = reader._job(publication['job_id'])
    if current is None or current['identity'] != publication['identity']:
        raise ValueError('Published preparation changed since listing; refresh the saved models')
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    job = Path(tempfile.mkdtemp(prefix='recovered-' + publication['job_id'] + '-', dir=root))
    try:
        bundle_sha = reader.download(publication['bundle_id'], job/'result.zip', MAX_BUNDLE)
        _checked_archive(job/'result.zip')
        _extract_results_archive(job/'result.zip', job)
        response_path = job/'response.json'
        if not response_path.is_file() or response_path.stat().st_size > MAX_REQUEST:
            raise ValueError('Preparation response is missing or oversized')
        response = json.loads(response_path.read_text())
        verify_result_files(job, response, action='prepare')
        if response.get('status') != 'model_prepared': raise ValueError('Preparation did not produce a usable model')
        request_sha = reader.download(publication['request_id'], job/'original-request.json', MAX_REQUEST)
        request = json.loads((job/'original-request.json').read_text())
        packet_sha = reader.download(publication['packet_id'], job/'original-packet.zip', MAX_PACKET_BYTES)
        if (request.get('expected_source_sha256') != packet_sha or response.get('source_sha256') != packet_sha
                or request.get('dispatch_key') != publication['dispatch_key']
                or not re.fullmatch(r'[a-f0-9]{64}', str(response.get('implementation_sha256') or ''))
                or request.get('plan', {}).get('implementation_sha256') != response.get('implementation_sha256')):
            raise ValueError('Preparation response does not match its original request and source packet')
        spec = extract_preparation_packet(job/'original-packet.zip', job/'original-inputs')
        original_files = {p.name: file_sha(p) for p in (job/'original-inputs').iterdir()}
        captured_files = {p.name: file_sha(p) for p in (job/'inputs').iterdir() if p.is_file()}
        if original_files != captured_files:
            raise ValueError('Prepared-model captured inputs differ from the original packet')
        model_id = reader.model_dataset(publication)
        model_sha = reader.download(model_id, job/'published-model.gpx', MAX_PROJECT)
        if model_sha != response.get('output_sha256'):
            raise ValueError('Published model differs from the verified preparation response')
        result = verify_prepared(job)
        if (result != response.get('preparation') or result.get('specification') != spec
                or result.get('project_sha256') != model_sha):
            raise ValueError('Prepared model, source specification and native receipt disagree')
        check_prepared_profiles(job, result)
        record = dict(dataset_id=model_id, source_sha256=model_sha, galaxy_job_id=publication['job_id'],
                      history_id=reader.history_id, bundle_id=publication['bundle_id'], bundle_sha256=bundle_sha,
                      request_sha256=request_sha, packet_sha256=packet_sha, response=response)
        (job/'preparation-publication.json').write_text(json.dumps(record, indent=2)+'\n')
        reader.check()
        return job, result
    except Exception:
        shutil.rmtree(job, ignore_errors=True)
        raise


def recover_review_workspace(reader, root):
    """Use existing checkpoint validation and hold work before UI publication."""
    from .refinement_runtime import GalaxyRefinementRuntime
    from .refinement_workspace import RefinementWorkspace
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='saved-model-review-', dir=root))
    try:
        runtime = GalaxyRefinementRuntime(directory, reader)
        workspace = RefinementWorkspace(directory, history_id=reader.history_id, runtime=runtime)
        workspace.restore_remote()
        # This is a local review copy. Immutable History checkpoints are never
        # rewritten, and external jobs are neither cancelled nor resubmitted.
        with workspace.lock:
            for recipe in workspace.state['recipes'].values():
                recipe['armed'] = False
            for job in workspace.state['jobs'].values():
                if job['status'] in {'queued', 'running', 'recovering'}:
                    job.setdefault('recovered_status', job['status'])
                    job.update(status='paused', message='Recovered for review; resume explicitly to continue existing work')
            workspace.state['paused'] = True
            workspace.save()
        reader.check()
        return workspace
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
