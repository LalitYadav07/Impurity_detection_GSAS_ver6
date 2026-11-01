"""Bounded ONCat context, separate from verified acquisition identity.

No catalog value selects a phase, calibration, cell, clock or exposure. Only
allowlisted scientific fields enter the UI or exported evidence.
"""
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import quote

from ..facility_metadata import context, nexus_candidates, parse_nexus_timestamp, resolve_identity

SCHEMA = 'radar-battery-catalog/v1'
ENDPOINTS = {'https://oncat.ornl.gov', 'https://calvera.ornl.gov/oncat'}
MAX_BYTES = 2 * 1024 * 1024
MAX_REQUESTS = 64
MAX_SECONDS = 30
FIELDS = {
    'run_number': 'run_number', 'ipts': 'experiment_identifier',
    'run_title': 'title', 'proposal_title': 'experiment_title',
    'start_time': 'start_time', 'end_time': 'end_time', 'duration': 'duration',
    'sample_name': 'sample.name', 'sample_id': 'sample.id',
    'sample_identifier': 'sample.identifier', 'registered_formula': 'sample.chemical_formula',
    'sample_description': 'sample.description', 'sample_comments': 'sample.comments',
}


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _value(value, maximum=4096):
    if isinstance(value, str):
        return value if len(value) <= maximum else None
    if type(value) in (int, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list) and len(value) <= 32:
        values = [_value(v, maximum) for v in value]
        return values if all(v is not None and not isinstance(v, list) for v in values) else None
    return None


def _nested(value, path):
    for part in path.split('.'):
        if not isinstance(value, dict): return None
        value = value.get(part)
    return value


def _envelope(scope, instrument, ipts, *, run=None, status='unavailable', message='', **values):
    return dict(schema=SCHEMA, scope=scope, instrument=instrument, ipts=ipts,
        **({'run': run} if run is not None else {}), status=status, message=message, **values)


def unavailable_context(scope, instrument, ipts, *, run=None):
    return _envelope(scope, instrument, ipts, run=run,
        message='Catalog context could not be read; file metadata remains usable. Retry catalog discovery or inspection separately.')


def _context(instrument, ipts, run=None):
    instrument, ipts = context(instrument, ipts, run)
    if instrument not in {'NOM', 'PG3'}:
        raise ValueError('Battery catalog context supports NOMAD and POWGEN')
    return instrument, ipts


def parse_proposal(data, instrument, ipts):
    if not isinstance(data, dict) or data.get('id') != ipts:
        raise ValueError('Catalog proposal identity does not match the selected IPTS')
    fields = {key: value for key in ('title', 'abstract')
              if isinstance(value := _value(data.get(key), 20000), str)}
    instruments = sorted({row['id'] for row in data.get('instruments') or []
        if isinstance(row, dict) and isinstance(row.get('id'), str) and len(row['id']) <= 32})
    issues = []
    if instruments and instrument not in instruments:
        issues.append('Selected beamline is not in the catalog proposal instrument list; review the experiment records.')
    return dict(fields=fields, instruments=instruments, issues=issues,
        status='available' if fields.get('abstract') else 'partial',
        message='Proposal purpose is context; it does not establish the contents of an individual exposure.'
            if fields.get('abstract') else 'Proposal abstract is unavailable with the current catalog access.')


def parse_run(data, instrument, ipts, run, location):
    if not isinstance(data, dict) or data.get('location') != location:
        raise ValueError('Catalog location does not match the exact requested acquisition')
    for key, expected in (('facility','SNS'), ('instrument',instrument), ('experiment',ipts)):
        if data.get(key) not in (None, expected):
            raise ValueError('Catalog acquisition scope disagrees with the selected experiment')
    entry = _nested(data, 'metadata.entry') or {}
    fields = {}
    for name, path in FIELDS.items():
        value = _value(_nested(entry, path))
        if value is not None:
            fields[name] = dict(value=value, field='/entry/'+path.replace('.', '/'))
    for value in (_nested(data, 'indexed.run_number'), _nested(entry, 'run_number')):
        if value is not None and str(value).strip() != str(run):
            raise ValueError('Catalog run number disagrees with the requested acquisition')
    if entry.get('experiment_identifier') not in (None, '', ipts):
        raise ValueError('Catalog stored proposal disagrees with the requested acquisition')
    aliases = {'NOM': {'NOM', 'NOMAD'}, 'PG3': {'PG3', 'POWGEN', 'POWGEN3'}}
    name = _nested(entry, 'instrument.name')
    if name and str(name).upper() not in aliases[instrument]:
        raise ValueError('Catalog stored instrument disagrees with the requested beamline')
    identity = []
    for key in ('sample_id', 'sample_identifier'):
        if key not in fields: continue
        values = fields[key]['value']
        for value in values if isinstance(values, list) else [values]:
            identity.append(dict(value=value, path=location, kind='oncat_catalog', field=fields[key]['field']))
    return dict(location=location, fields=fields, identity=resolve_identity(identity),
        indexed_run=_value(_nested(data, 'indexed.run_number')),
        modified=_value(data.get('modified')), checksum=_value(data.get('checksum')),
        status='available' if fields else 'partial',
        message='Catalog fields are recorded context; verify acquisition identity and conditions from the experiment files.'
            if fields else 'Catalog entry exists, but its acquisition fields are unavailable with the current access.')


class CatalogClient:
    """One app session, bounded exact-resource GETs, no login or credential export."""
    def __init__(self, *, base_url='https://oncat.ornl.gov', token_loader=None, enabled=True,
                 configuration_error='', session=None):
        self.base_url = base_url.rstrip('/')
        if self.base_url not in ENDPOINTS:
            raise ValueError('Choose the documented direct ONCat or NDIP proxy endpoint')
        self.token_loader = token_loader
        self.enabled = enabled
        self.configuration_error = configuration_error
        self.session = session
        self.rejected_credential = None
        self.begin_batch()

    def begin_batch(self):
        self.started = time.monotonic()
        self.requests = 0
        self.network_failed = False

    def _get(self, route, instrument, ipts, projection):
        import requests
        if self.configuration_error: return None, 'configuration_error', self.configuration_error, {}
        if not self.enabled: return None, 'disabled', 'Catalog retrieval is disabled in this workspace.', {}
        try:
            token = self.token_loader() if self.token_loader else None
            if token is not None and (not isinstance(token, str) or not token or any(c.isspace() for c in token)):
                raise ValueError('Invalid configured credential')
        except (OSError, ValueError):
            return None, 'access_unavailable', 'Configured catalog access is unavailable; check the server credential configuration.', {}
        credential = sha256(token.encode()).hexdigest() if token else 'public'
        if token and credential == self.rejected_credential:
            return None, 'access_unavailable', 'Catalog credential was rejected; replace or renew it before retrying.', {}
        if self.network_failed:
            return None, 'unavailable', 'Catalog connection failed for this inspection; file metadata remains available.', {}
        remaining = MAX_SECONDS - (time.monotonic()-self.started)
        if self.requests >= MAX_REQUESTS or remaining <= 0:
            return None, 'partial', 'Catalog request/time budget reached; inspect fewer runs for additional context.', {}
        if self.session is None:
            self.session = requests.Session()
            self.session.trust_env = False  # Public mode must not pick up unrelated netrc credentials.
        url = self.base_url+'/api/'+route
        params = [('facility','SNS'), ('instrument',instrument), ('experiment',ipts)]
        params.extend(('projection', value) for value in projection)
        headers = {'Authorization':'Bearer '+token} if token else {}
        provenance = dict(source_url=url, retrieved_utc=datetime.now(timezone.utc).isoformat(),
                          access='configured' if token else 'public',
                          request_scope=dict(facility='SNS', instrument=instrument, experiment=ipts))
        self.requests += 1
        try:
            with self.session.get(url, params=params, headers=headers, stream=True,
                    allow_redirects=False, timeout=(min(3,remaining),min(10,remaining))) as response:
                if response.status_code in (401,403):
                    if response.status_code == 401 and token: self.rejected_credential = credential
                    return None, 'access_unavailable', 'Catalog fields are not accessible with the current credentials or public access.', provenance
                if response.status_code == 404:
                    return None, 'not_found', 'This exact resource is not indexed or not accessible in the catalog.', provenance
                if response.status_code != 200 or response.headers.get('Link'):
                    return None, 'unavailable', 'Catalog did not return one complete exact-resource response.', provenance
                content = bytearray()
                for chunk in response.iter_content(65536):
                    content.extend(chunk)
                    if len(content) > MAX_BYTES or time.monotonic()-self.started > MAX_SECONDS:
                        return None, 'partial', 'Catalog response exceeded its bounded metadata budget.', provenance
                data = json.loads(content)
                if not isinstance(data, dict): raise ValueError('Expected one resource')
                # Only the parser's allowlisted result leaves this provider;
                # raw contacts and unrelated response fields are not retained.
                return data, 'received', '', provenance
        except requests.RequestException:
            self.network_failed = True
            return None, 'unavailable', 'Catalog connection is unavailable; file metadata remains usable.', provenance
        except (ValueError, TypeError):
            return None, 'unavailable', 'Catalog returned malformed metadata; file metadata remains usable.', provenance

    def _fetch(self, scope, instrument, ipts, route, projection, parser, *, run=None):
        data, status, message, provenance = self._get(route, instrument, ipts, projection)
        result = _envelope(scope, instrument, ipts, run=run, status=status, message=message, **provenance)
        if data is not None:
            try:
                result.update(parser(data))
                result['content_sha256'] = fingerprint({k:v for k,v in result.items() if k not in {'retrieved_utc','source_url','access'}})
            except (ValueError, TypeError, AttributeError):
                result.update(status='conflict', message='Catalog response has conflicting or unsupported acquisition identity; its fields were not adopted.')
        return result

    def proposal(self, instrument, ipts):
        instrument, ipts = _context(instrument, ipts)
        return self._fetch('proposal', instrument, ipts, 'proposals/'+ipts,
            ['id','title','abstract','instruments.id'], lambda data:parse_proposal(data,instrument,ipts))

    def acquisition(self, instrument, ipts, run, *, location=None):
        instrument, ipts = _context(instrument, ipts, run)
        candidates = [str(p) for p in nexus_candidates(instrument,ipts,run)]
        if location is not None and location not in candidates:
            return _envelope('run',instrument,ipts,run=run,status='conflict',
                message='File metadata did not supply a supported exact catalog path; no catalog request was made.')
        projection = ['location','facility','instrument','experiment','modified','checksum','indexed.run_number',
            'metadata.entry.instrument.name'] + ['metadata.entry.'+path for path in FIELDS.values()]
        for path in [location] if location else candidates:
            result = self._fetch('run',instrument,ipts,'datafiles/'+quote(path.lstrip('/'),safe='/'),
                projection, lambda data:parse_run(data,instrument,ipts,run,path), run=run)
            if result['status'] != 'not_found': return result
        return result


def catalog_client():
    enabled = os.getenv('RADAR_PD_ONCAT_ENABLED','true').lower() not in {'0','false','off'}
    token = os.getenv('RADAR_PD_ONCAT_TOKEN')
    token_file = os.getenv('RADAR_PD_ONCAT_TOKEN_FILE')
    def load():
        if token and token_file: raise ValueError('Choose one credential source')
        if token_file:
            path = Path(token_file).expanduser()
            with path.open() as stream: value = stream.read(16385)
            if len(value)>16384: raise ValueError('Credential file exceeds its bounded read')
            return value.strip()
        return token
    try:
        return CatalogClient(base_url=os.getenv('RADAR_PD_ONCAT_URL') or 'https://oncat.ornl.gov',
            enabled=enabled, token_loader=load if token or token_file else None)
    except ValueError:
        return CatalogClient(configuration_error='Catalog endpoint configuration is unsupported; file metadata remains usable.')


def _normal(value, key):
    if isinstance(value, list):
        if len(value) != 1: return None
        value = value[0]
    if value in (None, ''): return None
    if key in {'start_time','end_time'}:
        try:
            dt = parse_nexus_timestamp(value)
            return dt.astimezone(timezone.utc).isoformat() if dt.tzinfo else None
        except (ValueError, TypeError): return None
    if key in {'run_number','duration'}:
        try:
            number = Decimal(str(value))
            return number if number.is_finite() else None
        except InvalidOperation: return None
    return ' '.join(str(value).split())


def compare_acquisition(record, catalog, proposal=None):
    """Retain disagreement; never merge catalog fields into acquisition truth."""
    rows = []
    local, remote = record.get('fields', {}), catalog.get('fields', {})
    for key in sorted(set(local) | set(remote)):
        if key in {'sample_id','sample_identifier'}: continue
        a, b = local.get(key,{}).get('value'), remote.get(key,{}).get('value')
        if b is None: continue
        x, y = _normal(a,key), _normal(b,key)
        status = 'catalog_only' if a is None else 'unresolved' if x is None or y is None else 'agrees' if x == y else 'conflict'
        rows.append(dict(field=key, archive_value=a, catalog_value=b, status=status))
    a = record.get('identity',{}).get('sample_id')
    identity = catalog.get('identity',{})
    b = identity.get('sample_id')
    if b or identity.get('identity_status') == 'conflict':
        rows.append(dict(field='sample_identity',archive_value=a,catalog_value=b or 'Conflicting catalog IDs',
            status='conflict' if identity.get('identity_status')=='conflict' or (a and a!=b) else 'agrees' if a else 'catalog_only'))
    title = (proposal or {}).get('fields',{}).get('title')
    if title and local.get('proposal_title'):
        a = local['proposal_title']['value']
        rows.append(dict(field='proposal_title_vs_proposal', archive_value=a, catalog_value=title,
                         status='agrees' if _normal(a,'title')==_normal(title,'title') else 'conflict'))
    conflicts = [r['field'] for r in rows if r['status']=='conflict']
    return dict(status='conflict' if conflicts or catalog.get('status')=='conflict' else
                'compared' if any(r['status']=='agrees' for r in rows) else 'context_only' if rows else 'unavailable',
        conflicts=conflicts, fields=rows,
        message='Catalog disagrees on: '+', '.join(conflicts)+'. File-derived binding is retained; review the sources.' if conflicts else
            catalog['message']+' File-derived binding is retained.' if catalog.get('status')=='conflict' else
            'Catalog context does not replace verified file metadata or establish phases, temperature or calibration.')
