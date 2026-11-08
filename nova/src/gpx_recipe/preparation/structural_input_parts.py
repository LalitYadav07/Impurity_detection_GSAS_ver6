"""Data-only multipart staging for one exact three-curve training inventory.

Each ZIP retains the normal model-packet limit. Only a closed structural
training request can combine parts, whose complete inventory is bound to its
parent, profile, plan and per-file request hashes. No credentials or executable
content are needed on a scientific worker.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import stat
import tempfile
import zipfile

from ..symmetry.bundle import MAX_BYTES, unpack
from ..symmetry.compiler import confined
from ..symmetry.contracts import digest
from .batch import relative

SCHEMA='radar-structural-training-input-parts/v1'
REFERENCES='radar-model-input-packets/v1'
INDEX='structural-input-parts.json'
MAX_METADATA=4*1024**2
MAX_PART_BYTES=480*1024**2
MAX_PARTS=32
MAX_FILES=10000
OPERATION='preparation-structural-training'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(1024**2):h.update(chunk)
    return h.hexdigest()


def references(value):
    """Keep legacy single-HDA handles; validate new ordered multipart HDAs."""
    import re
    def identifier(key):
        if not isinstance(key,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',key):
            raise ValueError('Invalid native scientific input packet identity')
        return key
    if isinstance(value,str):return identifier(value),[]
    if (not isinstance(value,dict) or set(value)!={'schema','primary','parts'}
            or value['schema']!=REFERENCES or not isinstance(value['parts'],list)
            or not 1<=len(value['parts'])<=MAX_PARTS):
        raise ValueError('Invalid multipart scientific input references')
    ids=[identifier(key) for key in [value['primary'],*value['parts']]]
    if len(set(ids))!=len(ids):raise ValueError('Multipart inputs require distinct dataset identities')
    return ids[0],ids[1:]


def _pack(root,path,names,inventory):
    # Write canonical ZIP metadata: restoring an identical input with another
    # filesystem timestamp must not change a cached part's archive checksum.
    # Hash the bytes actually written, not a second read of the source later.
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        for name in sorted(names):
            h=hashlib.sha256();size=0
            with confined(root,name).open('rb') as source,archive.open(name,'w',force_zip64=True) as target:
                while data:=source.read(1024**2):h.update(data);size+=len(data);target.write(data)
            if dict(sha256=h.hexdigest(),bytes=size)!=inventory[name]:
                raise ValueError('Training input changed during multipart capture')


def _inventory(root,request):
    from .structural_jobs import read_plan,plan_inputs,MAX_PAIRED_INPUT_BYTES
    from .structural_training_blocks import REPRESENTATION
    if (request.get('schema')!='radar-model-native/v1' or request.get('operation')!=OPERATION
            or not isinstance(request.get('payload'),dict)):
        raise ValueError('Multipart inputs only support typed structural training')
    payload=request['payload']
    if payload.get('settings',{}).get('representation')!=REPRESENTATION:
        raise ValueError('Multipart inputs require the three-curve training representation')
    control=[relative(payload[key]) for key in ('source','instrument','plan')]
    if len(set(control))!=3:raise ValueError('Training controls require distinct paths')
    plan_path=confined(root,control[2]);plan=read_plan(plan_path)
    if (plan.get('representation')!=REPRESENTATION
            or plan['parent_sha256']!=request.get('expected_source_sha256')
            or sha256(confined(root,control[0]))!=plan['parent_sha256']
            or sha256(confined(root,control[1]))!=plan['instrument_sha256']):
        raise ValueError('Multipart training parent, profile or representation differs from its plan')
    data=[str(Path(control[2]).parent/name) for name in plan_inputs(plan_path)]
    names=control+data
    if (not 3<len(names)<=MAX_FILES or len({name.casefold() for name in names})!=len(names)
            or INDEX.casefold() in {name.casefold() for name in names}):
        raise ValueError('Multipart training inventory has colliding or excessive paths')
    hashes=request.get('input_hashes')
    if not isinstance(hashes,dict) or set(hashes)!=set(names):
        raise ValueError('Multipart training inventory differs from its captured request')
    from .tof_kernel_arrays import checksum
    if any(not checksum(hashes[name]) for name in names):
        raise ValueError('Multipart training requires exact file hashes')
    # The data budget is separate from the tiny control packet, and covers a
    # continuation's numerical checkpoint as well as all original examples.
    return control,data,MAX_PAIRED_INPUT_BYTES


def prepare(root,request,workspace,publish):
    """Stream one archive at a time to publish(signature, path) -> dataset id.

    A content-addressed uploader can reuse acknowledged part HDAs on retry.
    The control ZIP carries archive hashes, sizes and the complete file census;
    transport metadata is never part of the scientific input directory.
    """
    root=Path(root).resolve();workspace=Path(workspace)
    control,data,limit=_inventory(root,request)
    from .structural_jobs import checked_blocks,read_plan
    plan_path=root/request['payload']['plan']
    checked_blocks(plan_path,read_plan(plan_path),validate_arrays=False,grid=request['payload']['grid'])
    inventory={};groups=[];current=[];size=0;total=0
    for name in control+data:
        path=confined(root,relative(name));length=path.stat().st_size
        if not 0<length<=MAX_PART_BYTES:
            raise ValueError('A training file exceeds one bounded part; retain smaller native generation shards')
        checksum=sha256(path)
        if checksum!=request['input_hashes'][name]:raise ValueError('A captured training input changed')
        inventory[name]=dict(sha256=checksum,bytes=length)
        if name in control:continue
        total+=length
        if total>limit:raise ValueError('Training input parts exceed their typed total byte budget')
        if current and size+length>MAX_PART_BYTES:
            groups.append(current);current=[];size=0
        current.append(name);size+=length
    if current:groups.append(current)
    if (not 1<=len(groups)<=MAX_PARTS
            or sum(inventory[name]['bytes'] for name in control)+MAX_METADATA>MAX_PART_BYTES):
        raise ValueError('Training control packet or part count exceeds its bounded inventory')
    workspace.mkdir(parents=True,exist_ok=True)
    parts=[];ids=[]
    for number,names in enumerate(groups):
        path=workspace/f'part-{number:03d}.zip'
        try:
            _pack(root,path,names,inventory)
            if path.stat().st_size>MAX_BYTES:raise ValueError('Training archive exceeds the normal ZIP byte budget')
            parts.append(dict(sha256=sha256(path),bytes=path.stat().st_size,
                expanded_bytes=sum(inventory[name]['bytes'] for name in names),files=names))
            ids.append(publish(digest(dict(schema=SCHEMA,archive_sha256=parts[-1]['sha256'])),path))
        finally:path.unlink(missing_ok=True)
    index=dict(schema=SCHEMA,operation=OPERATION,source_sha256=request['expected_source_sha256'],
        plan=dict(path=request['payload']['plan'],sha256=request['input_hashes'][request['payload']['plan']]),
        files=inventory,parts=parts)
    raw=(json.dumps(index,separators=(',',':'),sort_keys=True)+'\n').encode()
    if len(raw)>MAX_METADATA:raise ValueError('Training part index exceeds its metadata budget')
    path=workspace/'controls.zip'
    try:
        _pack(root,path,control,inventory)
        with zipfile.ZipFile(path,'a',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
            archive.writestr(INDEX,raw)
        if path.stat().st_size>MAX_BYTES:raise ValueError('Training control archive exceeds its ZIP byte budget')
        signature=digest(dict(files={name:inventory[name]['sha256'] for name in control},index=hashlib.sha256(raw).hexdigest()))
        primary=publish(signature,path)
    finally:path.unlink(missing_ok=True)
    value=dict(schema=REFERENCES,primary=primary,parts=ids);references(value)
    return value


def _index(primary,request,parts):
    path=primary/INDEX
    if path.is_symlink() or not path.is_file() or not 0<path.stat().st_size<=MAX_METADATA:
        raise ValueError('Missing bounded structural input part index')
    index=json.loads(path.read_bytes())
    if (not isinstance(index,dict) or set(index)!={'schema','operation','source_sha256','plan','files','parts'}
            or index['schema']!=SCHEMA or index['operation']!=OPERATION
            or index['source_sha256']!=request.get('expected_source_sha256')
            or not isinstance(index['parts'],list) or len(index['parts'])!=len(parts)
            or not 1<=len(parts)<=MAX_PARTS or not isinstance(index['files'],dict)):
        raise ValueError('Multipart input index differs from the captured training scope')
    control,data,limit=_inventory(primary,request)
    expected=request['input_hashes'];files=index['files']
    if (set(files)!=set(expected) or index['plan']!=dict(path=request['payload']['plan'],
            sha256=expected[request['payload']['plan']])):
        raise ValueError('Multipart input index differs from the exact plan inventory')
    for name,info in files.items():
        if (not isinstance(info,dict) or set(info)!={'sha256','bytes'} or info['sha256']!=expected[name]
                or type(info['bytes']) is not int or not 0<info['bytes']<=MAX_PART_BYTES):
            raise ValueError('Invalid multipart file checksum or byte count')
    if sum(files[name]['bytes'] for name in data)>limit:
        raise ValueError('Training input parts exceed their typed total byte budget')
    seen=[];data=set(data)
    from .tof_kernel_arrays import checksum
    for info in index['parts']:
        if (not isinstance(info,dict) or set(info)!={'sha256','bytes','expanded_bytes','files'}
                or not checksum(info['sha256']) or type(info['bytes']) is not int
                or not 0<info['bytes']<=MAX_BYTES or type(info['expanded_bytes']) is not int
                or not 0<info['expanded_bytes']<=MAX_PART_BYTES or not isinstance(info['files'],list)
                or not 1<=len(info['files'])<=MAX_FILES
                or any(not isinstance(name,str) or name not in data for name in info['files'])):
            raise ValueError('Invalid bounded training archive inventory')
        if sum(files[name]['bytes'] for name in info['files'])!=info['expanded_bytes']:
            raise ValueError('Training archive expanded byte count differs from its index')
        seen.extend(info['files'])
    if len(seen)!=len(data) or set(seen)!=set(data):
        raise ValueError('Multipart archives omit, duplicate or collide with captured training files')
    if sum(info['bytes'] for info in index['parts'])>limit+MAX_BYTES:
        raise ValueError('Training archives exceed their typed aggregate transport budget')
    actual={str(path.relative_to(primary)) for path in primary.rglob('*') if path.is_file()}
    if actual!=set(control)|{INDEX}:raise ValueError('Unexpected training control files')
    return index


def inspect_parts(archive,request,count,workspace):
    """Bound downloads by the checked control index before fetching a part."""
    if type(count) is not int or not 1<=count<=MAX_PARTS or Path(archive).stat().st_size>MAX_BYTES:
        raise ValueError('Invalid bounded multipart control archive or part count')
    with tempfile.TemporaryDirectory(prefix='training-index-',dir=workspace) as temporary:
        return _index(unpack(archive,Path(temporary)/'controls'),request,[None]*count)


def stage(archive,parts,request,destination):
    """Validate every part before merging; publish only a complete exact input."""
    if not parts:return unpack(archive,destination)
    if Path(archive).stat().st_size>MAX_BYTES:raise ValueError('Training control archive exceeds its ZIP byte budget')
    destination=Path(destination).absolute()
    if destination.exists() or destination.is_symlink():raise ValueError('Multipart staging requires a fresh owned directory')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='structural-staging-',dir=destination.parent) as temporary:
        root=Path(temporary);captured=unpack(archive,root/'inputs')
        index=_index(captured,request,parts)
        # Inspect all headers and archive hashes first. Cross-part collisions
        # must fail before any of their data is extracted.
        for path,info in zip(parts,index['parts']):
            path=Path(path)
            if path.stat().st_size!=info['bytes'] or sha256(path)!=info['sha256']:
                raise ValueError('A captured training archive changed or is out of order')
            with zipfile.ZipFile(path) as packet:
                entries=packet.infolist()
                if len(entries)!=len(info['files']) or {e.filename for e in entries}!=set(info['files']):
                    raise ValueError('Training archive members differ from the captured part')
                for entry in entries:
                    name=relative(entry.filename)
                    if (entry.is_dir() or stat.S_ISLNK(entry.external_attr>>16)
                            or Path(name).suffix.lower() not in {'.json','.npz'}
                            or entry.file_size!=index['files'][name]['bytes']):
                        raise ValueError('Unsupported or changed training part member')
        for path,info in zip(parts,index['parts']):
            # Reuse the normal per-archive extraction bound and traversal
            # protection, into an isolated directory, then merge disjoint data.
            isolated=unpack(path,root/'part')
            for name in info['files']:
                source=confined(isolated,name);target=captured/name
                if sha256(source)!=index['files'][name]['sha256']:
                    raise ValueError('A training archive file changed')
                target.parent.mkdir(parents=True,exist_ok=True)
                source.replace(target)
            shutil.rmtree(isolated)
        (captured/INDEX).unlink()
        actual={str(path.relative_to(captured)):dict(sha256=sha256(path),bytes=path.stat().st_size)
                for path in captured.rglob('*') if path.is_file()}
        if actual!=index['files']:raise ValueError('Staged training files differ from the exact multipart inventory')
        from .structural_jobs import checked_blocks,read_plan
        plan=captured/request['payload']['plan']
        checked_blocks(plan,read_plan(plan),validate_arrays=False,grid=request['payload']['grid'])
        captured.replace(destination)
    return destination
