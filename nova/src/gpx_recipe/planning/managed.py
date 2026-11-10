"""Closed, data-only planner requests for the shared native job transport.

This module imports no native engine. NOVA and the worker use the same path and
inventory checks before the worker calls the existing planning implementation.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

SCHEMA = 'radar-planning-job/v1'
RESULT_SCHEMA = 'radar-planning-operation-result/v1'
OPERATIONS = frozenset({'planning-parent', 'planning-simulate', 'planning-context',
    'planning-export-parent', 'planning-magnetic-prepare', 'planning-structural-match',
    'planning-structural-modes', 'planning-structural-prepare'})
NATIVE = {'planning-context':'context', 'planning-export-parent':'export-parent',
    'planning-magnetic-prepare':'prepare', 'planning-structural-match':'structural-match',
    'planning-structural-modes':'structural-modes', 'planning-structural-prepare':'structural-prepare'}
TEXT_SUFFIXES = frozenset({'.cif', '.mcif', '.instprm', '.prm', '.ins', '.inst', '.irf'})
MAX_INPUT_BYTES = 20 * 1024 * 1024
MAX_PLAN_BYTES = 4 * 1024 * 1024


def relative_path(value):
    if not isinstance(value, str) or not value or len(value)>1024:
        raise ValueError('A bounded relative planning path is required')
    path = PurePosixPath(value)
    if (path.is_absolute() or '..' in path.parts or '\\' in value or ':' in value
            or str(path)!=value or value=='.'):
        raise ValueError('Planning paths must be relative and confined')
    return value


def confined(root, value, *, exists=True):
    value = relative_path(value)
    root = Path(root).resolve()
    candidate = root/value
    if any(p.is_symlink() for p in [candidate, *candidate.parents] if p!=root and p.is_relative_to(root)):
        raise ValueError('Planning inputs cannot use symbolic links')
    path = candidate.resolve(strict=exists)
    if not path.is_relative_to(root) or (exists and not path.is_file()):
        raise ValueError('A planning file escaped its captured inputs')
    return path


def file_references(operation, request):
    """Return mutable containers for the only executable file references."""
    if operation not in OPERATIONS or not isinstance(request, dict):
        raise ValueError('Unsupported planning operation')
    refs=[]
    if operation in {'planning-parent','planning-simulate'}:
        for key in ('cif','mcif','comparison_cif','custom_profile'):
            if request.get(key):refs.append((request,key,'text'))
        for phase in request.get('additional_phases',[]):
            if not isinstance(phase,dict) or not phase.get('cif'):
                raise ValueError('Each additional phase requires a captured CIF')
            refs.append((phase,'cif','text'))
        for key in ('prepared_magnetic','prepared_structural'):
            if request.get(key):refs.append((request[key],'gpx','native'))
    else:
        refs.append((request,'source_gpx','native'))
        if operation in {'planning-structural-modes','planning-structural-prepare'}:
            refs.append((request,'child_cif','text'))
    return refs


def validate_payload(operation, payload):
    if operation not in OPERATIONS or not isinstance(payload,dict) or set(payload)!={'source','plan','children','destination'}:
        raise ValueError('Unexpected managed planning operation fields')
    for key in ('source','plan','destination'):
        if not isinstance(payload[key],str) or not payload[key] or len(payload[key])>4096:
            raise ValueError('A managed planning file path is required')
    if (not isinstance(payload['children'],list) or len(payload['children'])>12
            or any(not isinstance(p,str) or not p for p in payload['children'])):
        raise ValueError('Invalid managed planning input inventory')


def read_plan(path, operation):
    path = Path(path)
    if path.stat().st_size>MAX_PLAN_BYTES:raise ValueError('Planning request exceeds its metadata budget')
    document=json.loads(path.read_text(),parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite planning value')))
    if (not isinstance(document,dict) or set(document)!={'schema','operation','request','files'}
            or document['schema']!=SCHEMA or document['operation']!=operation
            or not isinstance(document['files'],dict) or not 1<=len(document['files'])<=12):
        raise ValueError('Invalid managed planning request')
    request=copy.deepcopy(document['request'])
    references=file_references(operation,request)
    names={relative_path(container[key]) for container,key,_ in references}
    if names!=set(document['files']):raise ValueError('Planning request differs from its captured file inventory')
    for name,expected in document['files'].items():
        if not isinstance(expected,str) or not re.fullmatch('[a-f0-9]{64}',expected):
            raise ValueError('Invalid captured planning input hash')
        source=confined(path.parent,name)
        if not 0<source.stat().st_size<=MAX_INPUT_BYTES or hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
            raise ValueError('A captured planning input is missing or changed')
    for container,key,kind in references:
        source=confined(path.parent,container[key])
        if source.suffix.lower() not in (TEXT_SUFFIXES if kind=='text' else {'.gpx'}):
            raise ValueError('Unexpected planning input type')
        container[key]=str(source)
    # Closed native request fields keep paths hidden inside arbitrary objects
    # from becoming a second native input channel.
    if operation in NATIVE:
        base={'source_gpx','source_gpx_sha256','nuclear_reference'}
        extra={'planning-context':set(),'planning-export-parent':set(),
            'planning-magnetic-prepare':{'parent_setting_sha256','candidate','sites','site_policy','k','parameters'},
            'planning-structural-match':{'cif_text','k','options'},
            'planning-structural-modes':{'child_cif'},
            'planning-structural-prepare':{'parent_setting_sha256','candidate','child_cif','k','mode_values','cell'}}[operation]
        if set(request)-base-extra:raise ValueError('Unknown native planning request fields')
        source=Path(request['source_gpx'])
        if hashlib.sha256(source.read_bytes()).hexdigest()!=request.get('source_gpx_sha256'):
            raise ValueError('Planning parent changed after capture')
    if operation=='planning-parent' and any(request.get(key) for key in
            ('mcif','comparison_cif','prepared_magnetic','prepared_structural')):
        raise ValueError('Parent inspection requires the nuclear-only scenario')
    return request
