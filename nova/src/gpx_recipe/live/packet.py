"""Bounded, portable requests and response caches for isolated live workers."""
from __future__ import annotations
import hashlib,json,re,stat,zipfile,uuid
from pathlib import Path,PurePosixPath
from .contracts import file_hash,validate_request

SCHEMA='radar-live-packet/v1'
INPUTS={'source_gpx':'inputs/source.gpx','catalog':'inputs/catalog.json',
        'preflight':'inputs/preflight.json','observation':'inputs/observation.npz',
        'observation_source':'inputs/source.dat'}
INPUT_LIMITS={'source_gpx':128*1024*1024,'catalog':4*1024*1024,'preflight':4*1024*1024,
              'observation':32*1024*1024,'observation_source':64*1024*1024}
CACHE_FILES={'kernel.json','kernel.npz','magnetic-refinement.gpx'}
MAX_BYTES=1024*1024*1024
MAX_FILES=4096
MAX_CACHE_BYTES=384*1024*1024
MAX_JSON_BYTES=4*1024*1024


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def _member(name):
    p=PurePosixPath(name)
    if not name or '\\' in name or p.is_absolute() or any(v in {'','..','.'} for v in p.parts) or str(p)!=name:
        raise ValueError('Unsafe live-packet member')
    return name


def _entries(archive,*,maximum=MAX_BYTES):
    entries=archive.infolist()
    if len(entries)>MAX_FILES or sum(v.file_size for v in entries)>maximum:
        raise ValueError('Live packet exceeds archive limits')
    names=set()
    for item in entries:
        _member(item.filename)
        if item.filename.endswith('.json') and item.file_size>MAX_JSON_BYTES:
            raise ValueError('Live packet JSON exceeds metadata limit')
        if (item.filename in names or item.is_dir() or item.flag_bits & 1
                or stat.S_IFMT(item.external_attr>>16) not in {0,stat.S_IFREG}):
            raise ValueError('Repeated or nonregular live-packet member')
        names.add(item.filename)
    return entries


def _read_json(archive,name):
    if archive.getinfo(name).file_size>MAX_JSON_BYTES:
        raise ValueError('Live packet JSON exceeds metadata limit')
    return json.loads(archive.read(name))


def _zip(path,payload,expected=None):
    """Write files incrementally and publish only the completed archive."""
    path=Path(path);temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    try:
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
            for name,value in sorted(payload.items()):
                info=zipfile.ZipInfo(_member(name),date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
                info.create_system=3;info.external_attr=(stat.S_IFREG|0o600)<<16
                if isinstance(value,Path):
                    digest=hashlib.sha256();count=0
                    with value.open('rb') as source,z.open(info,'w') as target:
                        for block in iter(lambda:source.read(1024*1024),b''):
                            count+=len(block)
                            if count>MAX_BYTES:raise ValueError('Live input grew beyond archive budget')
                            digest.update(block);target.write(block)
                    if expected and digest.hexdigest()!=expected[name]:raise ValueError('Live input changed during packing')
                else:z.writestr(info,value)
        temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)


def _validate(request):
    if request.get('action')=='prepare':
        from .prepare_source import validate_preparation
        validate_preparation(request)
    else:validate_request(request)
    if request.get('prepared_scattering') is not None:
        from ..preparation.hints import validate,live_context
        catalog=json.loads(Path(request['catalog']).read_text())
        validate(request['prepared_scattering'],live_context(request,catalog))


def create_packet(request,destination,*,prepared_root=None):
    portable=dict(request);payload={};hashes={}
    for field,name in INPUTS.items():
        if request.get(field):
            path=Path(request[field])
            if path.is_symlink() or not path.is_file() or path.stat().st_size>INPUT_LIMITS[field]:
                raise ValueError('Live input exceeds its file budget: '+field)
            payload[name]=path;hashes[name]=file_hash(path);portable[field]=name
    _validate(request)
    if request.get('prepared_scattering') is not None:
        from ..preparation.hints import MANIFEST,MAX_BYTES as PREPARED_LIMIT,entry_assets
        from ..preparation.library import confined
        document=request['prepared_scattering'];payload[MANIFEST]=canonical(document)
        hashes[MANIFEST]=hashlib.sha256(payload[MANIFEST]).hexdigest()
        total=0
        # Optional hints never crowd scientific inputs out of the packet.
        for row in (asset for row in document['entries'] for asset in entry_assets(row)):
            name=row['path']
            if name in payload or prepared_root is None or len(payload)>=MAX_FILES-2:continue
            try:
                path=confined(prepared_root,name);size=path.stat().st_size
                if total+size>PREPARED_LIMIT or file_hash(path)!=row['sha256']:continue
                payload[name]=path;hashes[name]=row['sha256'];total+=size
            except (OSError,ValueError):continue
    metadata=canonical(portable)
    if len(metadata)>MAX_JSON_BYTES:raise ValueError('Live request exceeds metadata limit')
    payload['request.json']=metadata;hashes['request.json']=hashlib.sha256(metadata).hexdigest()
    manifest={'schema':SCHEMA,'files':hashes}
    packet_id=hashlib.sha256(canonical(manifest)).hexdigest()
    payload['manifest.json']=canonical(manifest);_zip(destination,payload,hashes)
    return packet_id


def scientific_packet_identity(archive):
    """Exact scientific binding, independent of optional acceleration bytes.

    Recovery still verifies the original complete packet ID on the returned
    job. This identity permits recovering that job after disposable hints are
    lost, without treating a changed observation or recipe as equivalent.
    """
    with zipfile.ZipFile(archive) as z:
        _entries(z);manifest=_read_json(z,'manifest.json');request=_read_json(z,'request.json')
    request.pop('prepared_scattering',None)
    files={name:value for name,value in manifest['files'].items()
        if name in INPUTS.values()}
    files['request.json']=hashlib.sha256(canonical(request)).hexdigest()
    return hashlib.sha256(canonical(dict(schema=SCHEMA,files=files))).hexdigest()


def _copy_entry(archive,name,target,checksum):
    if not isinstance(checksum,str) or not re.fullmatch('[a-f0-9]{64}',checksum):
        raise ValueError('Invalid live-packet checksum')
    target.parent.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256()
    with archive.open(name) as source,target.open('xb') as output:
        for block in iter(lambda:source.read(1024*1024),b''):
            digest.update(block);output.write(block)
    if digest.hexdigest()!=checksum:
        target.unlink(missing_ok=True);raise ValueError('Live-packet input checksum mismatch')


def unpack_packet(archive,directory):
    root=Path(directory).resolve();root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):raise ValueError('Live request requires a fresh working directory')
    with zipfile.ZipFile(archive) as z:
        entries=_entries(z)
        manifest=_read_json(z,'manifest.json')
        if manifest.get('schema')!=SCHEMA:raise ValueError('Unsupported live-packet schema')
        files=manifest.get('files')
        if not isinstance(files,dict) or set(files)|{'manifest.json'}!={v.filename for v in entries}:
            raise ValueError('Live packet is not completely enumerated')
        portable=_read_json(z,'request.json')
        extras=set()
        if portable.get('prepared_scattering') is not None:
            from ..preparation.hints import MANIFEST,MAX_BYTES as PREPARED_LIMIT,validate,live_context,entry_assets
            document=portable['prepared_scattering']
            validate(document,live_context(portable,_read_json(z,INPUTS['catalog'])))
            extras={MANIFEST}|{asset['path'] for row in document['entries'] for asset in entry_assets(row)}
            if MANIFEST not in files or _read_json(z,MANIFEST)!=document:
                raise ValueError('Live packet preparation differs from its captured selection')
            if sum(v.file_size for v in entries if v.filename in extras-{MANIFEST})>PREPARED_LIMIT:
                raise ValueError('Live preparation exceeds its byte budget')
        if set(files)-({'request.json'}|set(INPUTS.values())|extras):raise ValueError('Unexpected live-packet input')
        for field,name in INPUTS.items():
            if portable.get(field) and (portable[field]!=name or name not in files):
                raise ValueError('Live packet contains an unbound input path')
            if name in files and z.getinfo(name).file_size>INPUT_LIMITS[field]:
                raise ValueError('Live input exceeds its file budget: '+field)
        for name,expected in files.items():_copy_entry(z,name,root/name,expected)
    request=dict(portable)
    for field,name in INPUTS.items():
        if request.get(field):request[field]=str((root/name).resolve())
    _validate(request)
    (root/'request.json').write_bytes(canonical(request))
    return request,hashlib.sha256(canonical(manifest)).hexdigest()


def pack_cache(directory,destination):
    """Cache whole model groups within budget; omitting cache never drops a fit."""
    root=Path(directory);payload={};hashes={};omitted=[];total=0
    for folder in sorted((root/'models').glob('*')):
        if folder.is_symlink() or not folder.is_dir():raise ValueError('Invalid response cache source')
        paths=[folder/name for name in sorted(CACHE_FILES)]
        if any(not p.is_file() or p.is_symlink() for p in paths):
            omitted.append(folder.name);continue
        size=sum(p.stat().st_size for p in paths)
        if (total+size>MAX_CACHE_BYTES or len(payload)+len(paths)>MAX_FILES-1
                or (folder/'kernel.json').stat().st_size>MAX_JSON_BYTES):
            omitted.append(folder.name);continue
        try:
            from .array_limits import validate_npz
            validate_npz(folder/'kernel.npz')
        except (ValueError, OSError, zipfile.BadZipFile):
            omitted.append(folder.name);continue
        for path in paths:
            name=str(path.relative_to(root));payload[name]=path;hashes[name]=file_hash(path)
        total+=size
    manifest={'schema':'radar-live-cache/v1','files':hashes,'omitted_models':omitted}
    payload['manifest.json']=canonical(manifest);_zip(destination,payload,hashes)


def unpack_cache(archive,directory):
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        entries=_entries(z,maximum=MAX_CACHE_BYTES+MAX_JSON_BYTES);manifest=_read_json(z,'manifest.json');files=manifest.get('files')
        if manifest.get('schema')!='radar-live-cache/v1' or not isinstance(files,dict):raise ValueError('Unsupported live-response cache')
        if set(files)|{'manifest.json'}!={v.filename for v in entries}:raise ValueError('Incomplete live-response cache')
        for name,checksum in files.items():
            parts=PurePosixPath(name).parts
            if len(parts)!=3 or parts[0]!='models' or parts[-1] not in CACHE_FILES or not re.fullmatch('[A-Za-z0-9_-]{1,160}',parts[1]):
                raise ValueError('Unexpected response-cache file')
            path=root/name
            if not path.resolve().is_relative_to(root.resolve()):raise ValueError('Response cache escapes work root')
            if path.exists():raise ValueError('Response cache may not replace existing work')
            _copy_entry(z,name,path,checksum)
            if path.suffix=='.npz':
                from .array_limits import validate_npz
                validate_npz(path)


def extract_result_archive(archive,directory):
    """Bound result extraction before any scientific arrays are loaded."""
    root=Path(directory).resolve();root.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        entries=_entries(z)
        for item in entries:
            target=root/item.filename
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(item) as source,target.open('xb') as output:
                for block in iter(lambda:source.read(1024*1024),b''):output.write(block)
            if target.suffix=='.npz':
                from .array_limits import validate_npz
                validate_npz(target)
    return root
