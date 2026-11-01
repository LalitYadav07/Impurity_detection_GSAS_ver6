"""Bounded, portable requests and response caches for isolated live workers."""
from __future__ import annotations
import hashlib,json,re,stat,zipfile
from pathlib import Path,PurePosixPath
from .contracts import file_hash,validate_request

SCHEMA='radar-live-packet/v1'
INPUTS={'source_gpx':'inputs/source.gpx','catalog':'inputs/catalog.json',
        'preflight':'inputs/preflight.json','observation':'inputs/observation.npz',
        'observation_source':'inputs/source.dat'}
CACHE_FILES={'kernel.json','kernel.npz','magnetic-refinement.gpx'}
MAX_BYTES=1024*1024*1024
MAX_FILES=4096


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def _member(name):
    p=PurePosixPath(name)
    if not name or '\\' in name or p.is_absolute() or any(v in {'','..','.'} for v in p.parts) or str(p)!=name:
        raise ValueError('Unsafe live-packet member')
    return name


def _entries(archive):
    entries=archive.infolist()
    if len(entries)>MAX_FILES or sum(v.file_size for v in entries)>MAX_BYTES:
        raise ValueError('Live packet exceeds archive limits')
    names=set()
    for item in entries:
        _member(item.filename)
        if item.filename in names or item.is_dir() or stat.S_ISLNK(item.external_attr>>16):
            raise ValueError('Repeated or nonregular live-packet member')
        names.add(item.filename)
    return entries


def _zip(path,payload):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(payload.items()):
            info=zipfile.ZipInfo(_member(name),date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=(stat.S_IFREG|0o600)<<16
            z.writestr(info,data)


def create_packet(request,destination):
    validate_request(request)
    portable=dict(request);payload={}
    total=sum(Path(request[field]).stat().st_size for field in INPUTS if request.get(field))
    if total>MAX_BYTES:raise ValueError('Live request exceeds packet limits')
    for field,name in INPUTS.items():
        if request.get(field):
            payload[name]=Path(request[field]).read_bytes();portable[field]=name
    payload['request.json']=canonical(portable)
    manifest={'schema':SCHEMA,'files':{name:hashlib.sha256(data).hexdigest() for name,data in payload.items()}}
    packet_id=hashlib.sha256(canonical(manifest)).hexdigest()
    payload['manifest.json']=canonical(manifest);_zip(destination,payload)
    return packet_id


def unpack_packet(archive,directory):
    root=Path(directory).resolve();root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):raise ValueError('Live request requires a fresh working directory')
    with zipfile.ZipFile(archive) as z:
        entries=_entries(z)
        manifest=json.loads(z.read('manifest.json'))
        if manifest.get('schema')!=SCHEMA:raise ValueError('Unsupported live-packet schema')
        files=manifest.get('files')
        if not isinstance(files,dict) or set(files)|{'manifest.json'}!={v.filename for v in entries}:
            raise ValueError('Live packet is not completely enumerated')
        if set(files)-({'request.json'}|set(INPUTS.values())):raise ValueError('Unexpected live-packet input')
        payload={}
        for name,expected in files.items():
            data=z.read(name)
            if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Live-packet input checksum mismatch')
            payload[name]=data
    portable=json.loads(payload['request.json'])
    for field,name in INPUTS.items():
        if portable.get(field) and (portable[field]!=name or name not in payload):
            raise ValueError('Live packet contains an unbound input path')
    for name,data in payload.items():
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    request=dict(portable)
    for field,name in INPUTS.items():
        if request.get(field):request[field]=str((root/name).resolve())
    validate_request(request)
    (root/'request.json').write_bytes(canonical(request))
    return request,hashlib.sha256(canonical(manifest)).hexdigest()


def pack_cache(directory,destination):
    root=Path(directory);payload={}
    for path in sorted((root/'models').glob('*/*')):
        if path.name not in CACHE_FILES:continue
        if path.is_symlink() or not path.is_file():raise ValueError('Invalid response cache source')
        payload[str(path.relative_to(root))]=path.read_bytes()
    manifest={'schema':'radar-live-cache/v1','files':{n:hashlib.sha256(v).hexdigest() for n,v in payload.items()}}
    payload['manifest.json']=canonical(manifest);_zip(destination,payload)


def unpack_cache(archive,directory):
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        entries=_entries(z);manifest=json.loads(z.read('manifest.json'));files=manifest.get('files')
        if manifest.get('schema')!='radar-live-cache/v1' or not isinstance(files,dict):raise ValueError('Unsupported live-response cache')
        if set(files)|{'manifest.json'}!={v.filename for v in entries}:raise ValueError('Incomplete live-response cache')
        for name,checksum in files.items():
            parts=PurePosixPath(name).parts
            if len(parts)!=3 or parts[0]!='models' or parts[-1] not in CACHE_FILES or not re.fullmatch('[A-Za-z0-9_-]{1,160}',parts[1]):
                raise ValueError('Unexpected response-cache file')
            data=z.read(name)
            if hashlib.sha256(data).hexdigest()!=checksum:raise ValueError('Live-response cache checksum mismatch')
            path=root/name
            if not path.resolve().is_relative_to(root.resolve()):raise ValueError('Response cache escapes work root')
            if path.exists():raise ValueError('Response cache may not replace existing work')
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    # The native worker still verifies candidate/source/kernel/code/runtime/mask
    # compatibility. A valid archive receipt alone never makes a cache reusable.
