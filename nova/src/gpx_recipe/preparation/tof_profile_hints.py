"""Bounded shared profile dependencies for structural Model/Live packets."""
import json
from pathlib import Path
import re

from .library import confined
from .operations import portable_entry_artifacts
from .tof_profile_arrays import MAX_BYTES,MAX_METADATA,validate_metadata,origin_identity
from .tof_kernel_arrays import checksum
from ..symmetry.compiler import file_sha

SCHEMA='radar-prepared-tof-profile-input/v1'
# This supplements the existing 64 MiB start-bank allowance. The unchanged
# outer 512 MiB packet limit still covers source/child inputs and both kinds.
MAX_TOTAL=128*1024*1024
MAX_ENTRIES=100


def validate(value):
    if (not isinstance(value,dict) or set(value)!={'schema','entries','eligible_entries','omitted_entries'}
            or value['schema']!=SCHEMA or not isinstance(value['entries'],list)
            or len(value['entries'])>MAX_ENTRIES
            or any(type(value[k]) is not int or not 0<=value[k]<=MAX_ENTRIES
                   for k in ('eligible_entries','omitted_entries'))
            or value['eligible_entries']!=len(value['entries'])+value['omitted_entries']):
        raise ValueError('Invalid bounded shared TOF preparation selection')
    seen=set()
    for row in value['entries']:
        if (not isinstance(row,dict) or set(row)!={'entry','bank','identity','path','sha256',
                'receipt_path','receipt_sha256','entry_sha256'}
                or not isinstance(row['entry'],str) or re.fullmatch('[a-f0-9]{24}',row['entry']) is None
                or row['entry'] in seen or not isinstance(row['bank'],str) or not 1<=len(row['bank'])<=100
                or any(not checksum(row[k]) for k in ('identity','sha256','receipt_sha256','entry_sha256'))
                or row['path']!='_prepared_tof/'+row['sha256']+'.npz'
                or row['receipt_path']!='_prepared_tof/'+row['receipt_sha256']+'.json'):
            raise ValueError('Invalid shared TOF proposal dependency')
        seen.add(row['entry'])
    return value


def paths(document):
    return sorted({row[k] for row in document.get('profile_tables',{}).get('entries',[])
                   for k in ('path','receipt_path')})


def entry_hashes(document):
    return [r['entry_sha256'] for r in document.get('profile_tables',{}).get('entries',[])]


def capture(library,document,root):
    from .hints import _copy
    rows=[];eligible=omitted=total=0;copied=set()
    hashes={a['path']:a['sha256'] for a in document['inputs']}
    for row in sorted(document['entries'],key=lambda r:r['id']):
        model=row['model']
        if model['kind']!='reference' or row['state']!='prepared':continue
        inventory={Path(a['path']).name:a for a in row['files']}
        if not {'entry.json','profile.npz','profile.json'}<=inventory.keys():continue
        eligible+=1
        if eligible>MAX_ENTRIES:raise ValueError('Too many shared TOF profile entries')
        dependencies=[document['specification']['source'],model['instrument'],
            *[inventory[n]['path'] for n in ('entry.json','profile.npz','profile.json')]]
        assets=library.verify_assets(list(dict.fromkeys(dependencies)))
        entry=assets[inventory['entry.json']['path']]
        if entry.stat().st_size>4*1024*1024:raise ValueError('Reference preparation metadata exceeds its bound')
        manifest=json.loads(entry.read_text());portable_entry_artifacts('reference',manifest)
        if (manifest.get('schema')!='radar-experiment-preparation-entry/v1' or manifest.get('kind')!='reference'
                or manifest.get('settings')!=model['settings'] or manifest.get('grid')!=model['grid']
                or manifest.get('input_sha256')!=[hashes[k] for k in dependencies[:2]]
                or manifest.get('observations_included') is not False or manifest.get('scientific_result') is not False
                or any(manifest['artifacts'].get(k)!=inventory[k]['sha256'] for k in ('profile.npz','profile.json'))):
            raise ValueError('Shared TOF table differs from its declared reference/profile')
        receipt=assets[inventory['profile.json']['path']]
        if receipt.stat().st_size>MAX_METADATA:raise ValueError('TOF metadata exceeds its bound')
        meta=validate_metadata(json.loads(receipt.read_text()))
        if origin_identity(meta)!=manifest['detail']['physics_sha256']:
            raise ValueError('Shared TOF physical identity differs from its reference entry')
        additions=[(k,inventory[k]['sha256'],assets[inventory[k]['path']]) for k in ('profile.npz','profile.json')]
        if any(p.stat().st_size>(MAX_BYTES if k.endswith('.npz') else MAX_METADATA) for k,_,p in additions):
            raise ValueError('Shared TOF dependency exceeds its bound')
        required=sum(p.stat().st_size for _,sha,p in additions if sha not in copied)
        if total+required>MAX_TOTAL:omitted+=1;continue
        locations={}
        for name,sha,path in additions:
            relative='_prepared_tof/'+sha+Path(name).suffix
            _copy(path,confined(root,relative,exists=False),sha);locations[name]=relative
            if sha not in copied:copied.add(sha);total+=path.stat().st_size
        rows.append(dict(entry=row['id'],bank=model['bank'],identity=origin_identity(meta),
            path=locations['profile.npz'],sha256=inventory['profile.npz']['sha256'],
            receipt_path=locations['profile.json'],receipt_sha256=inventory['profile.json']['sha256'],
            entry_sha256=inventory['entry.json']['sha256']))
    return validate(dict(schema=SCHEMA,entries=rows,eligible_entries=eligible,omitted_entries=omitted))


def checked_files(root,row):
    files={}
    for key,sha,limit in [('path',row['sha256'],MAX_BYTES),
                          ('receipt_path',row['receipt_sha256'],MAX_METADATA)]:
        path=confined(root,row[key])
        if path.is_symlink() or path.stat().st_size>limit or file_sha(path)!=sha:
            raise ValueError('Shared TOF artifact changed or exceeds its bound')
        files[key]=path
    meta=validate_metadata(json.loads(files['receipt_path'].read_text()))
    if origin_identity(meta)!=row['identity']:raise ValueError('Shared TOF metadata belongs to another field')
    return files


def restore(document,source_root,target_root):
    from .hints import _copy
    value=document.get('profile_tables')
    if value is None:return dict(attached=0,discarded_entries=[])
    validate(value);total=0;copied=set();accepted=[];discarded=[]
    for row in value['entries']:
        try:
            files=checked_files(source_root,row)
            required=sum(p.stat().st_size for k,p in files.items()
                         if row['sha256' if k=='path' else 'receipt_sha256'] not in copied)
            if total+required>MAX_TOTAL:raise ValueError('Shared TOF transport budget exceeded')
            for key,path in files.items():
                sha=row['sha256' if key=='path' else 'receipt_sha256'];target=confined(target_root,row[key],exists=False)
                if target!=path:_copy(path,target,sha)
                if sha not in copied:copied.add(sha);total+=path.stat().st_size
            accepted.append(row['entry'])
        except (OSError,ValueError,EOFError,KeyError,TypeError):discarded.append(row['entry'])
    return dict(attached=len(accepted),discarded_entries=discarded)


def candidates(root,protocol):
    from .structural_hints import read
    try:document=read(root,protocol)
    except (OSError,ValueError,EOFError) as error:
        return [],dict(status='unavailable',attached=0,error_type=type(error).__name__)
    if document is None:return [],dict(status='not_selected',attached=0)
    value=document.get('profile_tables')
    if value is None:return [],dict(status='not_selected',attached=0)
    validate(value);rows=[];total=0;copied=set();discarded=[]
    for row in value['entries']:
        try:
            files=checked_files(root,row)
            required=sum(p.stat().st_size for k,p in files.items()
                         if row['sha256' if k=='path' else 'receipt_sha256'] not in copied)
            if total+required>MAX_TOTAL:raise ValueError('Shared TOF transport budget exceeded')
            for key,path in files.items():copied.add(row['sha256' if key=='path' else 'receipt_sha256'])
            total+=required;rows.append(dict(row,files=files))
        except (OSError,ValueError,EOFError,KeyError,TypeError) as error:
            discarded.append(dict(entry=row['entry'],reason=str(error)[:300]))
    return rows,dict(status='attached' if rows else 'rebuild_current_context',attached=len(rows),
        discarded_entries=discarded,omitted_entries=value['omitted_entries'],native_admission_required=True)


def bind(source,choices):
    """Bind before the shared service creates its current starting portfolio."""
    import time
    from .tof_profile_arrays import read,read_metadata,admit
    began=time.monotonic();discarded=[]
    if not isinstance(choices,(list,tuple)) or len(choices)>MAX_ENTRIES:
        raise ValueError('Bounded shared TOF preparation choices required')
    for row in choices:
        try:
            from ..structural_live.optimization.tof_shape import IsotropicTOFShape
            shape=IsotropicTOFShape(source.fast.native,source.fast.slow.hist.name,source.fast.slow.phase.name)
            metadata,receipt_sha=read_metadata(row['files']['receipt_path'])
            if receipt_sha!=row['receipt_sha256'] or metadata['shape_sha256']!=shape.identity:
                raise ValueError('Prepared TOF metadata differs from the actual bank/profile; numeric arrays were not loaded')
            data,sha,receipt_sha=read(row['files']['path'],row['files']['receipt_path'])
            if (sha!=row['sha256'] or receipt_sha!=row['receipt_sha256']
                    or origin_identity(data.metadata)!=row['identity']):
                raise ValueError('Shared TOF table changed before native admission')
            table=admit(data,source.fast.native,source.fast.slow.hist.name,source.fast.slow.phase.name,
                artifact_sha256=sha,receipt_sha256=receipt_sha)
            return dict(status='bound_certified_proposal_table',entry=row['entry'],bank=row['bank'],
                numeric_bytes=table.numeric_bytes,current_native_probes=table.portable_probes,
                native_likelihood_authority=False,seconds=time.monotonic()-began,discarded_entries=discarded)
        except (OSError,ValueError,EOFError,KeyError,TypeError) as error:
            discarded.append(dict(entry=row.get('entry') if isinstance(row,dict) else None,reason=str(error)[:300]))
    return dict(status='rebuild_current_context' if choices else 'not_selected',
        native_likelihood_authority=False,seconds=time.monotonic()-began,discarded_entries=discarded)
