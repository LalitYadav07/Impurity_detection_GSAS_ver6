"""Immutable, data-only scattering inputs for actual model calculations.

The scientific protocol is unchanged. These files contain no observations or
saved likelihood, and an imported file never carries native qualification.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import shutil
import uuid

from ..execution import write_json
from ..live.array_limits import validate_npz
from ..live.reflection_factors import ScatteringFactorCache
from ..symmetry.compiler import file_sha
from ..symmetry.contracts import digest
from .library import confined
from .operations import portable_entry_artifacts

SCHEMA='radar-prepared-scattering-input/v1'
MANIFEST='_prepared_scattering/manifest.json'
MAX_BYTES=64*1024*1024
MAX_METADATA=4*1024*1024
MAX_ENTRIES=5000


def entry_assets(row):
    """The same optional dependency closure in Model and Live packets."""
    return [row]+([row['response']] if 'response' in row else [])


def live_context(request,catalog):
    """Bind hints to reviewed live model choices, independent of path/counts.

    The scientific request and packet separately bind the exact observation,
    source GPX and grid. Current native scattering identities still decide
    whether any theory asset is compatible with that actual source.
    """
    return dict(schema='radar-live-scattering-context/v1',kind='magnetic',
        k=copy.deepcopy(catalog['k']),candidates=copy.deepcopy(catalog['candidates']),
        sites=copy.deepcopy(request['sites']))


def _hash(value):
    return isinstance(value,str) and re.fullmatch('[a-f0-9]{64}',value) is not None


def validate(document,protocol):
    if protocol['kind']=='structural':
        from .structural_hints import validate as structural_validate
        return structural_validate(document,protocol)
    keys={'schema','protocol_sha256','library_id','library_name','entries',
          'eligible_entries','omitted_entries','snapshot_sha256','observations_included'}
    if (not isinstance(document,dict) or set(document)-keys-{'evidence'} or keys-set(document) or document['schema']!=SCHEMA
            or protocol['kind']!='magnetic' or document['protocol_sha256']!=digest(protocol)
            or not isinstance(document['library_id'],str)
            or not re.fullmatch('[a-f0-9]{24}',document['library_id'])
            or not isinstance(document['library_name'],str) or len(document['library_name'])>200
            or document['observations_included'] is not False
            or not isinstance(document['entries'],list) or len(document['entries'])>MAX_ENTRIES
            or any(type(document[k]) is not int or not 0<=document[k]<=MAX_ENTRIES
                   for k in ('eligible_entries','omitted_entries'))
            or document['eligible_entries']!=len(document['entries'])+document['omitted_entries']
            or len(json.dumps(document,allow_nan=False).encode())>MAX_METADATA):
        raise ValueError('Prepared scattering inputs need their exact bounded protocol and library snapshot')
    if 'evidence' in document:
        from .evidence import validate_references
        validate_references(document['evidence'])
    if (not _hash(document['snapshot_sha256']) or document['snapshot_sha256']!=
            digest({k:v for k,v in document.items() if k!='snapshot_sha256'})):
        raise ValueError('Prepared scattering snapshot changed')
    candidates={c['id'] for c in protocol['candidates']};seen=set()
    for row in document['entries']:
        if (not isinstance(row,dict) or set(row)-{'response'}!={'entry','candidate','bank','identity','path','sha256','entry_sha256'}
                or not isinstance(row['entry'],str) or not re.fullmatch('[a-f0-9]{24}',row['entry'])
                or not isinstance(row['candidate'],str) or row['candidate'] not in candidates
                or not isinstance(row['bank'],str) or not row['bank'] or len(row['bank'])>100
                or any(not _hash(row[k]) for k in ('identity','sha256','entry_sha256'))
                or row['path']!='_prepared_scattering/'+row['sha256']+'.npz'
                or row['entry'] in seen):
            raise ValueError('Invalid prepared scattering input entry')
        if 'response' in row:
            response=row['response']
            if (not isinstance(response,dict) or set(response)!={'identity','physics_sha256','path','sha256'}
                    or any(not _hash(response[k]) for k in ('identity','physics_sha256','sha256'))
                    or response['path']!='_prepared_scattering/'+response['sha256']+'.response.npz'):
                raise ValueError('Invalid prepared rendered-response input')
        seen.add(row['entry'])
    return document


def read(root,protocol):
    if protocol.get('kind')=='structural':
        from .structural_hints import read as structural_read
        return structural_read(root,protocol)
    path=confined(root,MANIFEST,exists=False)
    if not path.exists():return None
    if path.is_symlink() or path.stat().st_size>MAX_METADATA:
        raise ValueError('Prepared scattering snapshot is oversized or changed')
    return validate(json.loads(path.read_text()),protocol)


def _copy(source,target,checksum):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if target.is_symlink() or file_sha(target)!=checksum:
            raise ValueError('An immutable scattering input changed')
        return
    temporary=target.with_name(target.name+'.'+uuid.uuid4().hex+'.partial')
    try:
        shutil.copyfile(source,temporary)
        if file_sha(temporary)!=checksum:raise ValueError('Scattering input changed during capture')
        temporary.replace(target)
    finally:temporary.unlink(missing_ok=True)


def capture(library,protocol,root):
    """Capture completed matching assets once; never narrow the model search.

    This checks only the dependency closure that is actually transferred.
    New preparations may supply a certified rendered response as well as
    geometry-adaptable factors. Older factor-only packets remain readable.
    """
    if protocol['kind']=='structural':
        from .structural_hints import capture as structural_capture
        return structural_capture(library,protocol,root)
    root=Path(root).resolve()
    existing=read(root,protocol)
    if existing is not None:return existing
    if protocol['kind']!='magnetic':return None
    library.refresh();document=copy.deepcopy(library.document)
    candidates={c['id'] for c in protocol['candidates']};rows=[];eligible=omitted=total=0;copied=set()
    for row in sorted(document['entries'],key=lambda r:r['id']):
        model=row['model']
        if (model['kind']!='magnetic' or row['state']!='prepared'
                or model['settings']['candidate']['id'] not in candidates
                or model['settings']['k']!=protocol['k']):continue
        eligible+=1
        inventory={Path(a['path']).name:a for a in row['files']}
        if 'scattering.npz' not in inventory:
            omitted+=1;continue
        dependencies=[document['specification']['source'],model['instrument'],
                      inventory['entry.json']['path'],inventory['scattering.npz']['path']]
        assets=library.verify_assets(list(dict.fromkeys(dependencies)))
        entry=assets[inventory['entry.json']['path']]
        if entry.stat().st_size>MAX_METADATA:raise ValueError('Prepared entry metadata exceeds its budget')
        manifest=json.loads(entry.read_text())
        input_hashes={a['path']:a['sha256'] for a in document['inputs']}
        if (manifest.get('schema')!='radar-experiment-preparation-entry/v1'
                or manifest.get('kind')!='magnetic' or manifest.get('settings')!=model['settings']
                or manifest.get('grid')!=model['grid'] or manifest.get('observations_included') is not False
                or manifest.get('scientific_result') is not False
                or manifest.get('input_sha256')!=[input_hashes[name] for name in dependencies[:2]]):
            raise ValueError('Prepared scattering entry differs from its declared model')
        portable_entry_artifacts('magnetic',manifest)
        artifact=inventory['scattering.npz'];source=assets[artifact['path']]
        if manifest['artifacts']['scattering.npz']!=artifact['sha256']:
            raise ValueError('Prepared scattering artifact differs from its published entry')
        checksum=artifact['sha256'];size=source.stat().st_size
        if checksum not in copied and total+size>MAX_BYTES:
            omitted+=1;continue
        relative='_prepared_scattering/'+checksum+'.npz'
        _copy(source,confined(root,relative,exists=False),checksum)
        if checksum not in copied:total+=size;copied.add(checksum)
        captured=dict(entry=row['id'],candidate=model['settings']['candidate']['id'],bank=model['bank'],
            identity=manifest['detail']['scattering']['identity'],path=relative,sha256=checksum,
            entry_sha256=inventory['entry.json']['sha256'])
        from .rendered_magnetic import metadata as rendered_metadata
        rendered=rendered_metadata(manifest['detail'])
        if rendered is not None and 'response.npz' in inventory:
            response=inventory['response.npz'];response_sha=response['sha256']
            path=confined(library.root,response['path']);response_size=path.stat().st_size
            if response_sha in copied or total+response_size<=MAX_BYTES:
                assets=library.verify_assets([response['path']]);path=assets[response['path']]
                if manifest['artifacts']['response.npz']!=response_sha:
                    raise ValueError('Prepared response differs from its published entry')
                relative='_prepared_scattering/'+response_sha+'.response.npz'
                _copy(path,confined(root,relative,exists=False),response_sha)
                if response_sha not in copied:total+=response_size;copied.add(response_sha)
                captured['response']=dict(rendered,path=relative,sha256=response_sha)
        rows.append(captured)
    result=dict(schema=SCHEMA,protocol_sha256=digest(protocol),library_id=document['id'],
        library_name=document['name'],entries=rows,eligible_entries=eligible,
        omitted_entries=omitted,observations_included=False)
    if document.get('evidence'):
        from .evidence import select
        result['evidence']=select(document['evidence'],[r['entry_sha256'] for r in rows])
    result['snapshot_sha256']=digest(result);validate(result,protocol)
    write_json(confined(root,MANIFEST,exists=False),result)
    return result


def packet_paths(root,protocol):
    """Missing optional hints do not prevent the scientific inputs dispatching."""
    if protocol['kind']=='structural':
        from .structural_hints import packet_paths as structural_paths
        return structural_paths(root,protocol)
    document=read(root,protocol)
    if document is None:return []
    names=[MANIFEST]
    for name in sorted({asset['path'] for row in document['entries']
            for asset in entry_assets(row)}):
        try:
            path=confined(root,name,exists=False)
            if path.is_file() and not path.is_symlink() and path.stat().st_size<=MAX_BYTES:names.append(name)
        except (OSError,ValueError):continue
    return names


def subset(document,protocol,candidates):
    """Select acceleration assets without changing scientific candidates."""
    validate(document,protocol)
    requested=set(candidates)
    if not requested.issubset({c['id'] for c in protocol['candidates']}):
        raise ValueError('Prepared subset requested an undeclared scientific candidate')
    result=copy.deepcopy(document)
    result['entries']=[row for row in result['entries'] if row['candidate'] in requested]
    if result.get('evidence'):
        from .evidence import select
        result['evidence']=select(result['evidence'],[r['entry_sha256'] for r in result['entries']])
    result['omitted_entries']=result['eligible_entries']-len(result['entries'])
    result['snapshot_sha256']=digest({k:v for k,v in result.items() if k!='snapshot_sha256'})
    return validate(result,protocol)


def restore(document,protocol,source_root,target_root):
    if protocol['kind']=='structural':
        from .structural_hints import restore as structural_restore
        return structural_restore(document,protocol,source_root,target_root)
    validate(document,protocol);accepted=[];discarded=[];total=0;copied=set()
    previous=read(target_root,protocol)
    if previous is not None and previous!=document:
        raise ValueError('These model inputs already captured another preparation snapshot')
    for row in document['entries']:
        attached=False
        for asset in entry_assets(row):
            try:
                source=confined(source_root,asset['path']);size=source.stat().st_size
                if asset['sha256'] not in copied and total+size>MAX_BYTES:
                    raise ValueError('Prepared scattering byte budget exceeded')
                if source.is_symlink() or file_sha(source)!=asset['sha256']:
                    raise ValueError('Prepared scattering bytes changed')
                validate_npz(source,maximum_bytes=MAX_BYTES+262144)
                target=confined(target_root,asset['path'],exists=False)
                if source!=target:_copy(source,target,asset['sha256'])
                if asset['sha256'] not in copied:total+=size;copied.add(asset['sha256'])
                attached=True
            except (OSError,ValueError,EOFError):pass
        (accepted if attached else discarded).append(row['entry'])
    write_json(confined(target_root,MANIFEST,exists=False),document)
    return dict(snapshot_sha256=document['snapshot_sha256'],attached=len(accepted),
                discarded_entries=discarded,native_qualification_required=True)


def cache_for(root,protocol,*,candidate=None):
    if protocol['kind']!='magnetic':return None,dict(attached=0,native_qualification_required=True)
    try:document=read(root,protocol)
    except (OSError,ValueError,EOFError) as error:
        return None,dict(attached=0,status='unavailable',error_type=type(error).__name__,
            native_qualification_required=True)
    if document is None:return None,dict(attached=0,native_qualification_required=True)
    cache=ScatteringFactorCache();accepted=[];discarded=[];total=0;copied=set();responses=0
    for row in document['entries']:
        if candidate is not None and row['candidate']!=candidate:continue
        attached=False
        for asset in entry_assets(row):
            try:
                path=confined(root,asset['path']);size=path.stat().st_size
                if asset['sha256'] not in copied and total+size>MAX_BYTES:
                    raise ValueError('Prepared scattering byte budget exceeded')
                if file_sha(path)!=asset['sha256']:raise ValueError('Prepared scattering bytes changed')
                if asset['sha256'] not in copied:total+=size;copied.add(asset['sha256'])
                if asset is not row:
                    cache.attach_response_source(path,identity=asset['identity'],sha256=asset['sha256'],
                        physics_sha256=asset['physics_sha256']);responses+=1
                else:cache.attach_source(path,identity=asset['identity'])
                attached=True
            except (OSError,ValueError):pass
        (accepted if attached else discarded).append(row['entry'])
    return (cache if accepted else None),dict(library_id=document['library_id'],
        snapshot_sha256=document['snapshot_sha256'],attached=len(accepted),
        discarded_entries=discarded,rendered_responses_attached=responses,native_qualification_required=True)
