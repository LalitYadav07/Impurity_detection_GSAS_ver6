"""Atomic resumable kernel chunks and bounded multipart data-only transport.

Stored/imported counts describe byte integrity, never scientific qualification.
The native verifier supplies an ephemeral permission separately on consumption.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import threading
import zipfile
import zlib

from ..symmetry.contracts import digest
from .tof_kernel_arrays import (KernelData, checksum, decode, encode, read_bounded,
                               MAX_BYTES, MAX_METADATA)

SCHEMA = 'radar-tof-kernel-store/v1'
GEOMETRY_SCHEMA = 'radar-tof-kernel-store/v2'
PART_SCHEMA = 'radar-tof-kernel-part/v1'
MAX_CELLS = 512
MAX_STORE_BYTES = 2 * 1024**3
MAX_INDEX = 1024 * 1024
MAX_PART_BYTES = 128 * 1024**2
_THREAD_LOCK = threading.RLock()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _spec(value):
    required={'model_sha256', 'bank', 'cell_count'}
    if (not isinstance(value, dict) or set(value)-{'geometry'} != required
            or not checksum(value['model_sha256']) or not isinstance(value['bank'], str)
            or not 1 <= len(value['bank']) <= 100 or type(value['cell_count']) is not int
            or not 1 <= value['cell_count'] <= MAX_CELLS):
        raise ValueError('Declare a bounded model, physical bank and kernel-cell count')
    if 'geometry' in value:
        from .kernel_geometry import validate_geometry
        validate_geometry(value['geometry'],value['cell_count'])
    return value


def _schema(specification):
    return GEOMETRY_SCHEMA if 'geometry' in specification else SCHEMA


def _manifest(value):
    if (not isinstance(value, dict) or set(value) != {'schema', 'specification', 'id', 'chunks'}
            or value['schema'] not in {SCHEMA,GEOMETRY_SCHEMA} or not isinstance(value['chunks'], dict)
            or len(value['chunks']) > MAX_CELLS or len(_json(value)) > MAX_INDEX):
        raise ValueError('Invalid bounded kernel-store manifest')
    _spec(value['specification'])
    if value['schema'] != _schema(value['specification']):
        raise ValueError('Kernel geometry requires its declared store format')
    if value['id'] != digest(value['specification']):
        raise ValueError('Kernel-store identity differs from its specification')
    total = 0
    for index, row in value['chunks'].items():
        if (not isinstance(index, str) or not index.isdecimal() or str(int(index)) != index
                or not 0 <= int(index) < value['specification']['cell_count']
                or not isinstance(row, dict) or set(row) != {'data_sha256', 'metadata_sha256', 'data_bytes', 'metadata_bytes'}
                or not all(checksum(row[k]) for k in ('data_sha256', 'metadata_sha256'))
                or any(type(row[k]) is not int or not 1 <= row[k] <= limit
                       for k, limit in [('data_bytes', MAX_BYTES), ('metadata_bytes', MAX_METADATA)])):
            raise ValueError('Invalid kernel-store chunk descriptor')
        total += row['data_bytes'] + row['metadata_bytes']
    if total > MAX_STORE_BYTES:
        raise ValueError('Kernel store exceeds its total byte budget')
    return value


def _atomic(path, raw):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Kernel-store files must not be symbolic links')
    descriptor, temporary = tempfile.mkstemp(prefix='.kernel-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _directory(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Kernel-store directories must not be symbolic links')
    path.mkdir(exist_ok=True)
    if not path.is_dir():
        raise ValueError('A kernel-store directory is required')
    return path


@contextmanager
def _archive(raw):
    import io
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            yield archive
    except (zipfile.BadZipFile, zlib.error, NotImplementedError) as error:
        raise ValueError('Malformed bounded kernel transport part') from error


class KernelStore:
    def __init__(self, root):
        if Path(root).is_symlink():
            raise ValueError('An owned kernel-store root is required')
        self.root = Path(root).resolve()
        self.document = self._read()

    @classmethod
    def create(cls, root, specification):
        _spec(specification); root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        value = dict(schema=_schema(specification), specification=dict(specification), id=digest(specification), chunks={})
        _atomic(root/'index.json', _json(value))
        return cls(root)

    @classmethod
    def open_or_create(cls, root, specification):
        """Coordinate simultaneous part arrivals before the index exists."""
        _spec(specification); root = Path(root)
        if root.is_symlink():
            raise ValueError('An owned kernel-store root is required')
        root.mkdir(parents=True, exist_ok=True)
        pending = object.__new__(cls); pending.root = root.resolve()
        with pending._lock():
            index = pending.root/'index.json'
            if not index.exists():
                if any(p.name != '.lock' and not p.name.startswith('.kernel-')
                       for p in pending.root.iterdir()):
                    raise ValueError('Cannot initialize a kernel store over unrelated files')
                _atomic(index, _json(dict(schema=_schema(specification), specification=dict(specification),
                    id=digest(specification), chunks={})))
            document = pending._read()
            if document['specification'] != specification:
                raise ValueError('Kernel-store specification differs from the imported preparation')
        return cls(root)

    def _read(self):
        return _manifest(json.loads(read_bounded(self.root/'index.json', MAX_INDEX)))

    @contextmanager
    def _lock(self):
        import fcntl
        with _THREAD_LOCK:
            descriptor = os.open(self.root/'.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
            with os.fdopen(descriptor, 'a+b') as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError('Kernel-store lock must be an owned regular file')
                fcntl.flock(stream, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(stream, fcntl.LOCK_UN)

    def _paths(self, index, descriptor, *, create=False):
        directory = self.root/'chunks'
        if create:
            _directory(directory)
        if directory.is_symlink():
            raise ValueError('Kernel-store chunks escaped their owned directory')
        directory = directory/f'{index:04d}'
        if create:
            _directory(directory)
        if directory.is_symlink():
            raise ValueError('Kernel-store chunk directory must be owned')
        return (directory/(descriptor['data_sha256']+'.npz'),
                directory/(descriptor['metadata_sha256']+'.json'))

    def put(self, index, data):
        if not isinstance(data, KernelData):
            raise TypeError('Immutable checked kernel data required')
        return self._put(index, data, encode(data), _json(data.metadata))

    def _put(self, index, data, raw, metadata):
        if (type(index) is not int or not 0 <= index < self.document['specification']['cell_count']
                or len(raw) > MAX_BYTES or len(metadata) > MAX_METADATA
                or json.loads(metadata) != data.metadata):
            raise ValueError('Kernel chunk differs from the declared store')
        descriptor = dict(data_sha256=_sha(raw), metadata_sha256=_sha(metadata),
                          data_bytes=len(raw), metadata_bytes=len(metadata))
        with self._lock():
            current = self._read()
            if current['id'] != self.document['id']:
                raise ValueError('Kernel-store specification changed')
            prior = current['chunks'].get(str(index))
            if prior is not None and prior != descriptor:
                raise ValueError('A stored kernel chunk is immutable; create a new preparation revision')
            proposed = dict(current, chunks=dict(current['chunks'], **{str(index): descriptor}))
            _manifest(proposed)
            paths = self._paths(index, descriptor, create=True)
            changed = False
            for path, content, maximum in zip(paths, (raw, metadata), (MAX_BYTES, MAX_METADATA)):
                try:
                    identical = read_bounded(path, maximum) == content
                except ValueError:
                    identical = False
                if not identical:
                    _atomic(path, content); changed = True
            _atomic(self.root/'index.json', _json(proposed))
            self.document = proposed
        return dict(index=index, state='stored' if prior is None else 'repaired' if changed else 'unchanged',
                    scientific_qualification=False)

    def _bytes(self, index, descriptor):
        paths = self._paths(index, descriptor)
        raw, metadata = (read_bounded(p, maximum) for p, maximum in zip(paths, (MAX_BYTES, MAX_METADATA)))
        if (len(raw) != descriptor['data_bytes'] or len(metadata) != descriptor['metadata_bytes']
                or _sha(raw) != descriptor['data_sha256'] or _sha(metadata) != descriptor['metadata_sha256']):
            raise ValueError('Stored kernel bytes differ from their immutable descriptor')
        return raw, metadata

    def get(self, index):
        self.document = self._read()
        if type(index) is not int or str(index) not in self.document['chunks']:
            raise ValueError('The requested kernel chunk has not been stored')
        raw, metadata = self._bytes(index, self.document['chunks'][str(index)])
        return decode(raw, json.loads(metadata))

    def coverage(self):
        self.document = self._read()
        total = self.document['specification']['cell_count']
        return dict(total=total, stored=len(self.document['chunks']),
            missing=[i for i in range(total) if str(i) not in self.document['chunks']],
            bytes=sum(r['data_bytes']+r['metadata_bytes'] for r in self.document['chunks'].values()),
            scientific_qualification=False, selected_chunk_integrity_checked_on_read=True)

    def export_parts(self, directory, *, check_cancel=None):
        """Stream deterministic parts below the existing native transport cap."""
        document = self._read(); catalog = _json(document); snapshot = digest(document)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        _directory(directory)
        prior = directory/'export.json'
        if prior.exists() and json.loads(read_bounded(prior, MAX_INDEX)).get('snapshot_sha256') != snapshot:
            raise ValueError('Use a new export directory for a changed preparation snapshot')
        groups = []; group = []; size = 0
        for index, row in sorted(document['chunks'].items(), key=lambda pair: int(pair[0])):
            required = row['data_bytes']+row['metadata_bytes']
            if group and size+required > MAX_PART_BYTES-4*MAX_INDEX:
                groups.append(group); group=[]; size=0
            group.append(int(index)); size += required
        if group:
            groups.append(group)
        if not groups:
            raise ValueError('There are no stored kernel chunks to export')
        records=[]
        for number, indices in enumerate(groups):
            if check_cancel is not None:
                check_cancel()
            part = dict(schema=PART_SCHEMA, snapshot_sha256=snapshot, index=number,
                        count=len(groups), cells=indices)
            descriptor, temporary = tempfile.mkstemp(prefix='.kernel-part-', dir=directory)
            os.close(descriptor)
            try:
                with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED) as archive:
                    def write(name, raw):
                        info=zipfile.ZipInfo(name, date_time=(1980,1,1,0,0,0))
                        info.external_attr=0o100644 << 16
                        archive.writestr(info, raw)
                    write('part.json', _json(part)); write('catalog.json', catalog)
                    for index in indices:
                        if check_cancel is not None:
                            check_cancel()
                        row=document['chunks'][str(index)]; raw, metadata=self._bytes(index,row)
                        write(f'chunks/{index:04d}/data.npz', raw)
                        write(f'chunks/{index:04d}/metadata.json', metadata)
                path=Path(temporary)
                if path.stat().st_size > MAX_PART_BYTES:
                    raise ValueError('Kernel part exceeds its bounded transport size')
                if check_cancel is not None:
                    check_cancel()
                final=directory/f'part-{number+1:04d}-of-{len(groups):04d}.zip'
                if final.is_symlink():
                    raise ValueError('Kernel part output must be owned')
                os.replace(path,final)
                raw=read_bounded(final,MAX_PART_BYTES)
                records.append(dict(path=final.name,sha256=_sha(raw),bytes=len(raw),cells=indices))
            finally:
                Path(temporary).unlink(missing_ok=True)
        result=dict(schema='radar-tof-kernel-export/v1',snapshot_sha256=snapshot,
                    store_id=document['id'],parts=records,scientific_qualification=False)
        _atomic(prior,_json(result))
        return result

    @classmethod
    def import_part(cls, path, root, *, check_cancel=None):
        if check_cancel is not None:
            check_cancel()
        raw=read_bounded(path,MAX_PART_BYTES)
        with _archive(raw) as archive:
            members=archive.infolist(); names=[m.filename for m in members]
            if (len(names)!=len(set(names)) or len(names)>2+2*MAX_CELLS
                    or any(m.compress_type!=zipfile.ZIP_STORED or m.flag_bits & 1 for m in members)
                    or not {'part.json','catalog.json'}<=set(names)
                    or any(archive.getinfo(n).file_size>MAX_INDEX for n in ('part.json','catalog.json'))):
                raise ValueError('Invalid bounded kernel part inventory')
            part=json.loads(archive.read('part.json')); document=_manifest(json.loads(archive.read('catalog.json')))
            if (not isinstance(part,dict) or set(part)!={'schema','snapshot_sha256','index','count','cells'}
                    or part['schema']!=PART_SCHEMA or part['snapshot_sha256']!=digest(document)
                    or type(part['index']) is not int or type(part['count']) is not int
                    or not 0<=part['index']<part['count']<=MAX_CELLS
                    or not isinstance(part['cells'],list) or not part['cells']
                    or any(type(i)is not int or str(i)not in document['chunks'] for i in part['cells'])
                    or len(set(part['cells']))!=len(part['cells'])):
                raise ValueError('Kernel part does not bind its complete store snapshot')
            expected={'part.json','catalog.json'}
            for index in part['cells']:
                expected.update({f'chunks/{index:04d}/data.npz',f'chunks/{index:04d}/metadata.json'})
            if set(names)!=expected:
                raise ValueError('Kernel part contains unrelated or missing files')
            store=cls.open_or_create(root,document['specification'])
            if store.document['id']!=document['id']:
                raise ValueError('Imported kernels belong to another store')
            for index in part['cells']:
                if check_cancel is not None:
                    check_cancel()
                descriptor=document['chunks'][str(index)]
                data_name=f'chunks/{index:04d}/data.npz'; meta_name=f'chunks/{index:04d}/metadata.json'
                if (archive.getinfo(data_name).file_size!=descriptor['data_bytes']
                        or archive.getinfo(meta_name).file_size!=descriptor['metadata_bytes']):
                    raise ValueError('Kernel part asset sizes differ from the declared snapshot')
                data_raw,metadata=archive.read(data_name),archive.read(meta_name)
                if _sha(data_raw)!=descriptor['data_sha256'] or _sha(metadata)!=descriptor['metadata_sha256']:
                    raise ValueError('Kernel part asset checksums differ from the declared snapshot')
                data=decode(data_raw,json.loads(metadata))
                store._put(index,data,data_raw,metadata)
        return store
