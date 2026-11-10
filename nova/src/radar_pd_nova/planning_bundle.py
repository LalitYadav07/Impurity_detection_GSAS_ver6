"""Portable planner data, with bounded extraction and no executable imports."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

from gpx_recipe.planning.managed import relative_path, confined

SCHEMA='radar-experiment-plan/v2'
MAX_BYTES=512*1024*1024
MAX_FILES=10000
MAX_MANIFEST_BYTES=16*1024*1024
SUFFIXES=frozenset({'.json','.npz','.gpx','.gsa','.cif','.mcif','.instprm','.prm',
    '.ins','.inst','.irf','.h5','.txt','.log','.lst','.dat'})


def checksum(raw):return hashlib.sha256(raw).hexdigest()


def json_bytes(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode('utf8')


def read_json(raw):
    if len(raw)>MAX_MANIFEST_BYTES:raise ValueError('Planner metadata exceeds its size limit')
    def pairs(values):
        result={}
        for key,value in values:
            if key in result:raise ValueError('Duplicate planner metadata key')
            result[key]=value
        return result
    def constant(_):raise ValueError('Planner metadata must contain finite numbers')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)


def validate_arrays(raw):
    """Bound nested NPZ members and headers before NumPy can allocate arrays."""
    import math
    import numpy as np
    total=0
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries=archive.infolist()
        if len(entries)>128:raise ValueError('Planner numeric archive has too many arrays')
        for entry in entries:
            if (not re.fullmatch(r'[A-Za-z0-9_-]+\.npy',entry.filename)
                    or stat.S_ISLNK(entry.external_attr>>16) or entry.file_size>128*1024*1024):
                raise ValueError('Unsupported or oversized planner numeric array')
            total+=entry.file_size
            if total>MAX_BYTES:raise ValueError('Planner numeric arrays exceed their expanded-size limit')
            with archive.open(entry) as stream:
                version=np.lib.format.read_magic(stream)
                if version==(1,0):shape,_,dtype=np.lib.format.read_array_header_1_0(stream)
                elif version==(2,0):shape,_,dtype=np.lib.format.read_array_header_2_0(stream)
                else:raise ValueError('Unsupported planner numeric array format')
                if dtype.hasobject or dtype.fields or dtype.kind not in 'biufcUS' or len(shape)>4:
                    raise ValueError('Planner arrays must be numeric or fixed text data without pickles')
                expected=math.prod(shape)*dtype.itemsize
                if expected>128*1024*1024 or expected!=entry.file_size-stream.tell():
                    raise ValueError('Planner numeric array shape exceeds its captured bytes')
        with np.load(io.BytesIO(raw),allow_pickle=False) as values:
            if any(not np.isfinite(values[key]).all() for key in values.files if values[key].dtype.kind in 'biufc'):
                raise ValueError('Planner numeric arrays must be finite')


class Builder:
    def __init__(self):
        self.files={};self.paths={};self.directories={};self.total=0

    def add_file(self, source, name):
        name=relative_path(name);source=Path(source)
        if source.is_symlink():raise ValueError('Planner bundles cannot contain symbolic links')
        path=source.resolve(strict=True)
        if not path.is_file() or path.suffix.lower() not in SUFFIXES:
            raise ValueError('Unsupported planner bundle file')
        if str(path) in self.paths:return self.paths[str(path)]
        raw=path.read_bytes();self.total+=len(raw)
        if len(self.files)>=MAX_FILES or self.total>MAX_BYTES:raise ValueError('Planner bundle exceeds its file/size limit')
        if name in self.files:raise ValueError('Duplicate planner bundle destination')
        self.paths[str(path)]=name;self.files[name]=raw
        return name

    def add_tree(self, source, name, *, receipt=None, relative_files=None):
        source=Path(source)
        if source.is_symlink():raise ValueError('Planner result directory cannot be a symbolic link')
        source=source.resolve(strict=True);name=relative_path(name)
        if str(source) in self.directories:return self.directories[str(source)]
        self.directories[str(source)]=name
        if receipt is not None:
            manifest=receipt.get('artifacts',{})
            if not isinstance(manifest,dict):raise ValueError('Invalid planner result inventory')
            for relative,expected in manifest.items():
                path=confined(source,relative)
                if checksum(path.read_bytes())!=expected:raise ValueError('A simulation artifact is missing or changed')
            paths=[confined(source,p) for p in manifest]+[source/'result.json']
        elif relative_files is not None:
            paths=[confined(source,p) for p in relative_files]
        else:
            paths=sorted(source.rglob('*'))
        for path in paths:
            if path.is_symlink():raise ValueError('Planner bundles cannot contain symbolic links')
            if path.is_file():self.add_file(path,name+'/'+str(path.relative_to(source)))
        return name

    def encode(self, value):
        if isinstance(value,Path):value=str(value)
        if isinstance(value,dict):return {str(k):self.encode(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [self.encode(v) for v in value]
        if isinstance(value,str) and (value.startswith('/') or re.match(r'^[A-Za-z]:[\\/]',value)):
            if value in self.paths:return {'$path':self.paths[value]}
            if value in self.directories:return {'$path':self.directories[value]}
            # A worker scratch filename is provenance, not portable authority.
            return {'$unavailable_path':Path(value).name}
        return value

    def finish(self, *, name, saved_at, implementation, settings, backend, native_artifacts, history_id=None):
        # Normalize JSON provenance paths. Numeric arrays and GPX bytes are
        # copied opaquely; this code never executes them.
        def plain(value):
            encoded=self.encode(value)
            def visit(v):
                if isinstance(v,dict):
                    if set(v)=={'$path'}:return v['$path']
                    if set(v)=={'$unavailable_path'}:return 'unavailable/'+v['$unavailable_path']
                    return {k:visit(x) for k,x in v.items()}
                if isinstance(v,list):return [visit(x) for x in v]
                return v
            return visit(encoded)
        for path,raw in list(self.files.items()):
            if path.endswith('.json'):self.files[path]=json_bytes(plain(read_json(raw)))
        for path,raw in list(self.files.items()):
            if not path.endswith('/result.json'):continue
            receipt=read_json(raw)
            if not isinstance(receipt,dict) or 'artifacts' not in receipt:continue
            base=str(PurePosixPath(path).parent)
            for relative in receipt['artifacts']:
                target=base+'/'+relative_path(relative)
                if target not in self.files:raise ValueError('A result references a file outside its bundle')
                receipt['artifacts'][relative]=checksum(self.files[target])
            self.files[path]=json_bytes(receipt)
        backend=self.encode(backend)
        for key,root_key in [('_planning_receipt_sha256','_planning_output'),
                             ('_planning_parent_receipt_sha256','_planning_parent_output')]:
            root=backend.get(root_key)
            if isinstance(root,dict) and set(root)=={'$path'}:
                backend[key]=checksum(self.files[root['$path']+'/result.json'])
        document=dict(schema=SCHEMA,name=str(name)[:100],saved_at=saved_at,
            implementation=implementation,history_id=history_id,settings=settings,backend=backend,
            directories=sorted(set(self.directories.values())),
            native_artifacts=[dict(file=self.paths[path],**record) for path,record in native_artifacts.items()
                              if path in self.paths and record.get('handle')],
            files={p:dict(sha256=checksum(raw),bytes=len(raw)) for p,raw in self.files.items()})
        validate(document,self.files)
        return document,self.files


def validate(document, files):
    required={'schema','name','saved_at','implementation','history_id','settings','backend',
              'directories','native_artifacts','files'}
    if (not isinstance(document,dict) or set(document)!=required or document['schema']!=SCHEMA
            or not isinstance(document['settings'],dict) or not isinstance(document['backend'],dict)
            or not isinstance(document['files'],dict) or set(files)!=set(document['files'])
            or not isinstance(document['directories'],list) or not isinstance(document['native_artifacts'],list)):
        raise ValueError('Unsupported or incomplete saved-plan format')
    if len(json_bytes(document))>MAX_MANIFEST_BYTES or len(files)>MAX_FILES or sum(map(len,files.values()))>MAX_BYTES:
        raise ValueError('Planner bundle exceeds its file/size limit')
    for path,raw in files.items():
        relative_path(path)
        if PurePosixPath(path).suffix.lower() not in SUFFIXES:raise ValueError('Unsupported planner bundle file')
        row=document['files'][path]
        if not isinstance(row,dict) or set(row)!={'sha256','bytes'} or row!=dict(sha256=checksum(raw),bytes=len(raw)):
            raise ValueError('A saved input or result is missing or changed')
        if path.endswith('.npz'):validate_arrays(raw)
    for name in document['directories']:
        relative_path(name)
        if name in files or not any(p.startswith(name+'/') for p in files):raise ValueError('Invalid planner directory reference')
    for row in document['native_artifacts']:
        if (not isinstance(row,dict) or set(row)!={'file','sha256','handle','relative'}
                or row['file'] not in files or PurePosixPath(row['file']).suffix.lower() not in {'.gpx','.npz'}
                or row['sha256']!=checksum(files[row['file']])):
            raise ValueError('Invalid native planner provenance')
        relative_path(row['relative'])
    # Validate tagged paths recursively before changing any open-plan state.
    def visit(value,depth=0):
        if depth>80:raise ValueError('Planner metadata nesting exceeds its limit')
        if isinstance(value,dict):
            if '$path' in value:
                if set(value)!={'$path'} or value['$path'] not in set(files)|set(document['directories']):
                    raise ValueError('Saved path is outside this workspace bundle')
            else:
                for v in value.values():visit(v,depth+1)
        elif isinstance(value,list):
            for v in value:visit(v,depth+1)
        elif isinstance(value,str) and (value.startswith('/') or re.match(r'^[A-Za-z]:[\\/]',value)):
            raise ValueError('Saved path is outside this workspace bundle')
    visit(document['backend'])
    return document


def decode(value, root):
    if isinstance(value,dict):
        if set(value)=={'$path'}:return str((Path(root)/relative_path(value['$path'])).resolve())
        if set(value)=={'$unavailable_path'}:return None
        return {k:decode(v,root) for k,v in value.items()}
    if isinstance(value,list):return [decode(v,root) for v in value]
    return value


def archive(document, files):
    validate(document,files);buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as output:
        output.writestr('plan.json',json_bytes(document))
        for path,raw in sorted(files.items()):output.writestr(path,raw)
    return buffer.getvalue()


def read_archive(raw):
    if len(raw)>MAX_BYTES:raise ValueError('Planner archive exceeds its compressed-size limit')
    with zipfile.ZipFile(io.BytesIO(raw)) as source:
        entries=source.infolist();seen=set();total=0
        if len(entries)>MAX_FILES+1:raise ValueError('Planner archive has too many files')
        for entry in entries:
            name=relative_path(entry.filename)
            if entry.is_dir() or stat.S_ISLNK(entry.external_attr>>16) or name.casefold() in seen:
                raise ValueError('Planner archive contains a duplicate, link or unsupported entry')
            seen.add(name.casefold());total+=entry.file_size
            if total>MAX_BYTES or (name=='plan.json' and entry.file_size>MAX_MANIFEST_BYTES):
                raise ValueError('Planner archive exceeds its expanded-size limit')
        if 'plan.json' not in {e.filename for e in entries}:raise ValueError('Planner archive has no plan manifest')
        document=read_json(source.read('plan.json'))
        files={entry.filename:source.read(entry) for entry in entries if entry.filename!='plan.json'}
    validate(document,files)
    return document,files


def write_directory(root, document, files):
    validate(document,files);root=Path(root)
    if root.exists() and any(root.iterdir()):raise ValueError('Choose a new immutable saved-plan directory')
    root.mkdir(parents=True,exist_ok=True)
    for name,raw in files.items():
        path=confined(root,name,exists=False);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    (root/'plan.json').write_bytes(json_bytes(document))
    return root


def read_directory(root):
    root=Path(root);manifest=root/'plan.json'
    if manifest.is_symlink() or manifest.stat().st_size>MAX_MANIFEST_BYTES:
        raise ValueError('Invalid saved planner manifest')
    document=read_json(manifest.read_bytes())
    files={name:confined(root,name).read_bytes() for name in document.get('files',{})}
    validate(document,files)
    return document,files
