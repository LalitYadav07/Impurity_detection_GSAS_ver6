"""Immutable catalog context for a preparation; no network or inferred inputs."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

SCHEMA = 'radar-battery-catalog-evidence/v1'
MAX_BYTES = 512 * 1024
IDENTITY = ('instrument','ipts','run','sample_id','metadata_sha256')


def validate_payload(value, acquisition):
    from radar_pd_nova.battery.catalog import SCHEMA as CATALOG_SCHEMA, compare_acquisition, fingerprint
    expected = {'schema','role','acquisition','archive_fields','archive_identity','proposal','run','comparison'}
    if not isinstance(value, dict) or set(value)!=expected or value['schema']!=SCHEMA or value['role']!='review_only':
        raise ValueError('Unsupported catalog evidence bundle')
    if value['acquisition']!={key:acquisition[key] for key in IDENTITY}:
        raise ValueError('Catalog context belongs to another reviewed acquisition')
    if (not isinstance(value['archive_fields'],dict) or not isinstance(value['archive_identity'],dict)
            or value['archive_identity'].get('sample_id')!=acquisition['sample_id']):
        raise ValueError('Catalog evidence lacks the reviewed file-derived identity')
    for scope in ('proposal','run'):
        item=value[scope]
        if item is None and scope=='proposal':continue
        if (not isinstance(item,dict) or item.get('schema')!=CATALOG_SCHEMA or
                item.get('scope')!=scope or any(item.get(k)!=acquisition[k] for k in ('instrument','ipts')) or
                (scope=='run' and item.get('run')!=acquisition['run'])):
            raise ValueError('Catalog scope differs from the selected acquisition')
        if item.get('content_sha256') and item['content_sha256']!=fingerprint({k:v for k,v in item.items()
                if k not in {'content_sha256','retrieved_utc','source_url','access'}}):
            raise ValueError('Captured catalog content digest changed')
    record=dict(fields=value['archive_fields'],identity=value['archive_identity'])
    if compare_acquisition(record,value['run'],value['proposal'])!=value['comparison']:
        raise ValueError('Captured catalog comparison cannot be reproduced')
    return value


def capture_catalog(proposal, record, acquisition, directory):
    if not record.get('catalog'):return None
    value=dict(schema=SCHEMA,role='review_only',acquisition={key:acquisition[key] for key in IDENTITY},
        archive_fields=deepcopy(record.get('fields',{})),archive_identity=deepcopy(record.get('identity',{})),
        proposal=deepcopy(proposal),run=deepcopy(record['catalog']),comparison=deepcopy(record['catalog_comparison']))
    validate_payload(value,acquisition)
    raw=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    if len(raw)>MAX_BYTES:raise ValueError('Catalog context exceeds its bounded preparation capture')
    path=Path(directory)/'catalog-context.json';path.write_bytes(raw)
    return dict(path=path.name,sha256=sha256(raw).hexdigest())


def read_catalog(reference, root, acquisition):
    from .preparation import input_path
    path=input_path(root,reference)
    with path.open('rb') as stream:raw=stream.read(MAX_BYTES+1)
    if len(raw)>MAX_BYTES:raise ValueError('Catalog context exceeds its bounded preparation capture')
    return validate_payload(json.loads(raw),acquisition)


def catalog_summary(value, reference):
    return dict(role='review_only',sha256=reference['sha256'],
        comparison_status=value['comparison']['status'],conflicts=value['comparison']['conflicts'],
        proposal_title=(value['proposal'] or {}).get('fields',{}).get('title'),
        message=value['comparison']['message'])
