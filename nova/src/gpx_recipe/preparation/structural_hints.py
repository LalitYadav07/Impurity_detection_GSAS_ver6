"""Portable structural starts; imports never attest their native calculation.

The complete scientific protocol and candidate inventory stay unchanged.
Current native geometry and a complete start-bank check decide actual reuse.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re

from ..execution import write_json
from ..symmetry.compiler import file_sha
from ..symmetry.contracts import digest
from .library import confined
from .operations import portable_entry_artifacts

SCHEMA='radar-prepared-structural-input/v1'
MANIFEST='_prepared_structural/manifest.json'
MAX_BYTES=64*1024*1024
MAX_METADATA=4*1024*1024
MAX_ENTRIES=5000


def candidate_identity(candidate):
    value=copy.deepcopy(candidate)
    if isinstance(value.get('child'),dict):
        # File placement is transport; content and all symmetry/mode choices
        # still belong to the immutable scientific model.
        value['child']={'sha256':value['child']['sha256']}
    return digest(value)


def validate(document,protocol):
    from .hints import _hash
    keys={'schema','protocol_sha256','library_id','library_name','entries',
        'eligible_entries','omitted_entries','snapshot_sha256','observations_included'}
    if (not isinstance(document,dict) or set(document)-keys-{'evidence','profile_tables'} or keys-set(document) or document['schema']!=SCHEMA
            or protocol['kind']!='structural' or document['protocol_sha256']!=digest(protocol)
            or not isinstance(document['library_id'],str) or not re.fullmatch('[a-f0-9]{24}',document['library_id'])
            or not isinstance(document['library_name'],str) or len(document['library_name'])>200
            or document['observations_included'] is not False
            or not isinstance(document['entries'],list) or len(document['entries'])>MAX_ENTRIES
            or any(type(document[k]) is not int or not 0<=document[k]<=MAX_ENTRIES
                for k in ('eligible_entries','omitted_entries'))
            or document['eligible_entries']!=len(document['entries'])+document['omitted_entries']
            or len(json.dumps(document,allow_nan=False).encode())>MAX_METADATA
            or not _hash(document['snapshot_sha256']) or document['snapshot_sha256']!=
                digest({k:v for k,v in document.items() if k!='snapshot_sha256'})):
        raise ValueError('Structural preparation needs its exact bounded protocol and library snapshot')
    if 'evidence' in document:
        from .evidence import validate_references
        validate_references(document['evidence'])
    if 'profile_tables' in document:
        from .tof_profile_hints import validate as validate_profiles
        validate_profiles(document['profile_tables'])
    candidates={c['id'] for c in protocol['candidates']};seen=set()
    fields={'entry','candidate','bank','identity','path','sha256','receipt_path','receipt_sha256','entry_sha256'}
    for row in document['entries']:
        if (not isinstance(row,dict) or set(row)!=fields
                or not isinstance(row['entry'],str) or not re.fullmatch('[a-f0-9]{24}',row['entry'])
                or not isinstance(row['candidate'],str) or row['candidate'] not in candidates
                or not isinstance(row['bank'],str) or not row['bank'] or len(row['bank'])>100
                or any(not _hash(row[k]) for k in ('identity','sha256','receipt_sha256','entry_sha256'))
                or row['path']!='_prepared_structural/'+row['sha256']+'.npz'
                or row['receipt_path']!='_prepared_structural/'+row['receipt_sha256']+'.json'
                or row['entry'] in seen):
            raise ValueError('Invalid structural preparation entry')
        seen.add(row['entry'])
    return document


def read(root,protocol):
    path=confined(root,MANIFEST,exists=False)
    if not path.exists():return None
    if path.is_symlink() or path.stat().st_size>MAX_METADATA:
        raise ValueError('Structural preparation snapshot is oversized or changed')
    return validate(json.loads(path.read_text()),protocol)


def paths(document):
    from .tof_profile_hints import paths as profile_paths
    return sorted({row[k] for row in document['entries'] for k in ('path','receipt_path')}|set(profile_paths(document)))


def capture(library,protocol,root):
    from .hints import _copy
    root=Path(root).resolve();existing=read(root,protocol)
    if existing is not None:return existing
    library.refresh();document=copy.deepcopy(library.document)
    candidates={c['id']:c for c in protocol['candidates']};rows=[];eligible=omitted=total=0;copied=set()
    hashes={a['path']:a['sha256'] for a in document['inputs']}
    for row in sorted(document['entries'],key=lambda r:r['id']):
        model=row['model'];candidate=model.get('settings',{}).get('candidate',{})
        expected=candidates.get(candidate.get('id'))
        if model['kind']!='structural' or row['state']!='prepared' or expected is None:continue
        if candidate_identity(candidate)!=candidate_identity(expected):continue
        eligible+=1
        inventory={Path(a['path']).name:a for a in row['files']}
        if not {'entry.json','starts.npz','starts.json'}<=set(inventory):omitted+=1;continue
        dependencies=[document['specification']['source'],model['instrument'],model['child'],
            *[inventory[n]['path'] for n in ('entry.json','starts.npz','starts.json')]]
        assets=library.verify_assets(list(dict.fromkeys(dependencies)))
        entry=assets[inventory['entry.json']['path']]
        if entry.stat().st_size>MAX_METADATA:raise ValueError('Structural entry metadata exceeds its budget')
        manifest=json.loads(entry.read_text())
        if (not isinstance(manifest,dict) or not isinstance(manifest.get('detail'),dict)
                or not isinstance(manifest.get('artifacts'),dict)):
            raise ValueError('Structural preparation metadata requires typed model and artifact records')
        if (manifest.get('schema')!='radar-experiment-preparation-entry/v1' or manifest.get('kind')!='structural'
                or manifest.get('settings')!=model['settings'] or manifest.get('grid')!=model['grid']
                or manifest.get('observations_included') is not False or manifest.get('scientific_result') is not False
                or manifest.get('input_sha256')!=[hashes[name] for name in dependencies[:3]]
                or expected['child']['sha256']!=hashes[model['child']]):
            raise ValueError('Structural preparation differs from its declared child, parent or profile')
        portable_entry_artifacts('structural',manifest)
        receipt=assets[inventory['starts.json']['path']]
        if receipt.stat().st_size>MAX_METADATA:raise ValueError('Structural start receipt exceeds its budget')
        bank=json.loads(receipt.read_text())
        if not isinstance(bank,dict) or not isinstance(bank.get('context'),dict):
            raise ValueError('Structural preparation receipt requires a typed physical context')
        if (bank.get('sha256')!=inventory['starts.npz']['sha256']
                or bank.get('context',{}).get('physics_sha256')!=manifest.get('detail',{}).get('physics_sha256')
                or any(manifest['artifacts'].get(n)!=inventory[n]['sha256'] for n in ('starts.npz','starts.json'))):
            raise ValueError('Structural bank differs from its published native receipt')
        additions=[(n,inventory[n]['sha256'],assets[inventory[n]['path']]) for n in ('starts.npz','starts.json')]
        required=sum(p.stat().st_size for _,sha,p in additions if sha not in copied)
        if total+required>MAX_BYTES:omitted+=1;continue
        locations={}
        for name,checksum,source in additions:
            relative='_prepared_structural/'+checksum+Path(name).suffix
            _copy(source,confined(root,relative,exists=False),checksum);locations[name]=relative
            if checksum not in copied:total+=source.stat().st_size;copied.add(checksum)
        rows.append(dict(entry=row['id'],candidate=candidate['id'],bank=model['bank'],
            identity=manifest['detail']['physics_sha256'],path=locations['starts.npz'],
            sha256=inventory['starts.npz']['sha256'],receipt_path=locations['starts.json'],
            receipt_sha256=inventory['starts.json']['sha256'],entry_sha256=inventory['entry.json']['sha256']))
    result=dict(schema=SCHEMA,protocol_sha256=digest(protocol),library_id=document['id'],library_name=document['name'],
        entries=rows,eligible_entries=eligible,omitted_entries=omitted,observations_included=False)
    from .tof_profile_hints import capture as capture_profiles,entry_hashes
    profiles=capture_profiles(library,document,root)
    if profiles['eligible_entries']:result['profile_tables']=profiles
    if document.get('evidence'):
        from .evidence import select
        result['evidence']=select(document['evidence'],[r['entry_sha256'] for r in rows]+entry_hashes(result))
    result['snapshot_sha256']=digest(result);validate(result,protocol)
    write_json(confined(root,MANIFEST,exists=False),result)
    return result


def packet_paths(root,protocol):
    document=read(root,protocol)
    if document is None:return []
    names=[MANIFEST]
    for name in paths(document):
        try:
            path=confined(root,name,exists=False)
            if path.is_file() and not path.is_symlink() and path.stat().st_size<=MAX_BYTES:names.append(name)
        except (OSError,ValueError):continue
    return names


def checked_files(root,row):
    files={}
    for name,checksum,maximum in [('path',row['sha256'],MAX_BYTES),('receipt_path',row['receipt_sha256'],MAX_METADATA)]:
        path=confined(root,row[name])
        if path.is_symlink() or path.stat().st_size>maximum or file_sha(path)!=checksum:
            raise ValueError('Structural preparation bytes changed or exceed their budget')
        files[name]=path
    from ..structural_live.optimization.offline_native_start_bank import read_bank_receipt
    receipt=read_bank_receipt(files['path'],files['receipt_path'],maximum_bytes=MAX_BYTES)
    if receipt.get('sha256')!=row['sha256'] or receipt.get('context',{}).get('physics_sha256')!=row['identity']:
        raise ValueError('Structural preparation receipt belongs to another numeric bank')
    return files


def restore(document,protocol,source_root,target_root):
    from .hints import _copy
    validate(document,protocol);accepted=[];discarded=[];total=0;copied=set()
    previous=read(target_root,protocol)
    if previous is not None and previous!=document:raise ValueError('These model inputs captured another preparation snapshot')
    for row in document['entries']:
        try:
            files=checked_files(source_root,row)
            required=sum(path.stat().st_size for k,path in files.items() if row['sha256' if k=='path' else 'receipt_sha256'] not in copied)
            if total+required>MAX_BYTES:raise ValueError('Structural preparation byte budget exceeded')
            for key,path in files.items():
                checksum=row['sha256' if key=='path' else 'receipt_sha256']
                target=confined(target_root,row[key],exists=False)
                if target!=path:_copy(path,target,checksum)
                if checksum not in copied:total+=path.stat().st_size;copied.add(checksum)
            accepted.append(row['entry'])
        except (OSError,ValueError,EOFError,KeyError,TypeError):discarded.append(row['entry'])
    from .tof_profile_hints import restore as restore_profiles
    profiles=restore_profiles(document,source_root,target_root)
    write_json(confined(target_root,MANIFEST,exists=False),document)
    return dict(snapshot_sha256=document['snapshot_sha256'],attached=len(accepted),discarded_entries=discarded,
        native_qualification_required=True,kind='structural',profile_tables=profiles)


def candidates(root,protocol,candidate):
    """Return only bounded checked files; current native context decides use."""
    try:document=read(root,protocol)
    except (OSError,ValueError,EOFError) as error:
        return [],dict(status='unavailable',attached=0,error_type=type(error).__name__,native_qualification_required=True)
    if document is None:return [],dict(attached=0,native_qualification_required=True)
    found=[];discarded=[];total=0
    for row in document['entries']:
        if row['candidate']!=candidate:continue
        try:
            files=checked_files(root,row);size=sum(p.stat().st_size for p in files.values())
            if total+size>MAX_BYTES:raise ValueError('Structural preparation byte budget exceeded')
            total+=size;found.append(dict(row,files=files))
        except (OSError,ValueError,EOFError,KeyError,TypeError):discarded.append(row['entry'])
    return found,dict(library_id=document['library_id'],snapshot_sha256=document['snapshot_sha256'],
        attached=len(found),discarded_entries=discarded,native_qualification_required=True)
