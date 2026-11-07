"""Conservative numerical identity for native TOF cross-worker qualification.

Unavailable evidence disables origin-based reuse, never the native calculation.
Addresses and local installation paths are not serialized. Native library
bytes, CPU classes and the active floating-point control state are retained.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import struct
import subprocess
import sys
import threading

from .identity import numerical_identity
from ..symmetry.contracts import digest

SCHEMA='radar-native-tof-runtime/v1'
MAX_LIBRARY_BYTES=128*1024*1024
_local=threading.local()
_cache=None
_libc=platform.libc_ver()
_supported=(sys.platform=='linux' and platform.machine()=='x86_64'
            and sys.byteorder=='little' and _libc[0]=='glibc')


class RuntimeUnavailable(ValueError):
    """Complete native qualification remains required in this environment."""


def floating_state():
    """Read only x87/SSE controls on the reviewed Linux/glibc x86-64 ABI.

    Sticky exception-status bits are not arithmetic inputs. Control masks,
    rounding, precision, denormal-input and flush-to-zero controls are. The
    oversized buffer safely contains the native fenv_t (32 bytes on this ABI).
    No control or status flag is modified by this reader.
    """
    if not _supported:return None
    try:
        if getattr(_local,'pid',None)!=os.getpid():
            library=ctypes.CDLL('libm.so.6')
            function=library.fegetenv;function.argtypes=[ctypes.c_void_p];function.restype=ctypes.c_int
            _local.pid=os.getpid();_local.library=library;_local.function=function
            _local.buffer=ctypes.create_string_buffer(256)
        if _local.function(_local.buffer)!=0:return None
        raw=_local.buffer.raw
        control=struct.unpack_from('<H',raw,0)[0]
        mxcsr=struct.unpack_from('<I',raw,28)[0]
        if control & ~0x1f7f or mxcsr & ~0xffff:return None
        return control,mxcsr & ~0x3f
    except (OSError,AttributeError,ValueError):
        return None


def _maps():
    try:
        raw=Path('/proc/self/maps').read_bytes()
        if len(raw)>4*1024*1024:raise RuntimeUnavailable('Native map inventory exceeds its bound')
        rows=[];paths={}
        for line in raw.decode().splitlines():
            fields=line.split(None,5)
            if len(fields)!=6 or not fields[5].startswith('/'):continue
            if fields[5].endswith(' (deleted)'):
                raise RuntimeUnavailable('A mapped numerical library was deleted')
            low,high=(int(v,16) for v in fields[0].split('-'))
            # One library has several mapped segments. Resolve its path once
            # within THIS fresh maps snapshot, not again for every segment.
            # Nothing is cached across calls; _stamp still matches the actual
            # file's current device and inode against these kernel records.
            name=fields[5]
            if name not in paths:paths[name]=Path(name).resolve()
            rows.append((paths[name],low,high,fields[3],int(fields[4])))
        return rows
    except (OSError,UnicodeError,ValueError) as error:
        raise RuntimeUnavailable('Native map evidence is unavailable') from error


def _stamp(path,maps):
    try:
        path=Path(path).resolve();info=path.stat()
        device=f'{os.major(info.st_dev):x}:{os.minor(info.st_dev):x}'
        matched=[r for r in maps if r[0]==path and r[4]==info.st_ino
                 and ':'.join(f'{int(v,16):x}' for v in r[3].split(':'))==device]
        if not matched or not 0<info.st_size<=MAX_LIBRARY_BYTES:
            raise RuntimeUnavailable('Linked numerical library does not match its mapped file')
        return str(path),info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns
    except OSError as error:
        raise RuntimeUnavailable('Linked numerical library is unavailable') from error


def _sha(path):
    with Path(path).open('rb') as stream:
        h=hashlib.sha256();total=0
        while block:=stream.read(1024*1024):
            total+=len(block)
            if total>MAX_LIBRARY_BYTES:raise RuntimeUnavailable('Numerical library grew beyond its bound')
            h.update(block)
    return h.hexdigest()


def _cpu():
    path=Path('/proc/cpuinfo')
    try:
        with path.open('rb') as stream:raw=stream.read(4*1024*1024+1)
        if len(raw)>4*1024*1024:raise RuntimeUnavailable('CPU evidence exceeds its bound')
        keys={'vendor_id','cpu family','model','stepping','microcode','flags'};classes=set()
        for block in raw.decode().split('\n\n'):
            record={k.strip():v.strip() for line in block.splitlines() if ':' in line
                    for k,v in [line.split(':',1)] if k.strip() in keys}
            if not record:continue
            if not {'vendor_id','cpu family','model','stepping','flags'}<=record.keys():
                raise RuntimeUnavailable('Incomplete x86 CPU evidence')
            record['flags']=' '.join(sorted(set(record['flags'].split())))
            classes.add(tuple(sorted(record.items())))
        if not classes or len(classes)>64:raise RuntimeUnavailable('Unsupported CPU class inventory')
        return [dict(row) for row in sorted(classes)]
    except (OSError,UnicodeError) as error:
        raise RuntimeUnavailable('CPU evidence is unavailable') from error


def _linked(binary,maps):
    if os.environ.get('LD_PRELOAD') or os.environ.get('LD_AUDIT'):
        raise RuntimeUnavailable('Preloaded numerical overrides require complete native qualification')
    try:
        result=subprocess.run(['ldd',str(binary)],capture_output=True,text=True,timeout=5,check=True)
    except (OSError,subprocess.SubprocessError) as error:
        raise RuntimeUnavailable('Native link evidence is unavailable') from error
    if len(result.stdout)>65536:raise RuntimeUnavailable('Native link evidence exceeds its bound')
    libraries={Path(binary).resolve()}
    for line in result.stdout.splitlines():
        line=line.strip()
        if not line or line.startswith('linux-vdso.so.'):continue
        match=re.fullmatch(r'(?:\S+\s+=>\s+)?(/\S+)\s+\(0x[0-9a-f]+\)',line)
        if match is None:raise RuntimeUnavailable('Unrecognized native linked-library record')
        libraries.add(Path(match.group(1)).resolve())
    if len(libraries)>32:raise RuntimeUnavailable('Native library inventory exceeds its bound')
    return sorted(_stamp(p,maps) for p in libraries)


def _fingerprint(source):
    """Exact-runtime evidence for an origin receipt, not native authorization."""
    global _cache
    from .native_tof_kernels import context
    _,_,function,_,native=context(source)
    state=floating_state()
    if state is None:raise RuntimeUnavailable('Floating-point ABI is not qualified for origin reuse')
    if os.environ.get('LD_PRELOAD') or os.environ.get('LD_AUDIT'):
        raise RuntimeUnavailable('Preloaded numerical overrides require complete native qualification')
    module=function.__globals__.get('pyd');binary=Path(module.__file__).resolve()
    maps=_maps();stamp=_stamp(binary,maps)
    if _cache is None or _cache['pid']!=os.getpid() or _cache['binary']!=stamp:
        links=_linked(binary,maps)
        hashes={Path(row[0]).name:_sha(row[0]) for row in links}
        if len(hashes)!=len(links):raise RuntimeUnavailable('Ambiguous numerical library names')
        _cache=dict(pid=os.getpid(),binary=stamp,links=links,libraries=hashes,cpu=_cpu())
    if [_stamp(row[0],maps) for row in _cache['links']]!=_cache['links']:
        _cache=None;raise RuntimeUnavailable('Mapped numerical dependencies changed')
    capsule=module.pyepsvoigt._cpointer
    pointer=ctypes.pythonapi.PyCapsule_GetPointer
    pointer.argtypes=[ctypes.py_object,ctypes.c_char_p];pointer.restype=ctypes.c_void_p
    address=pointer(capsule,None)
    regions=[r for r in maps if r[0]==binary]
    if not address or not any(r[1]<=address<r[2] for r in regions):
        raise RuntimeUnavailable('Native entry point escaped the reviewed binary')
    content=dict(schema=SCHEMA,native=native,numerical=numerical_identity(),
        libc=list(_libc),libraries=dict(_cache['libraries']),cpu=[dict(row) for row in _cache['cpu']],
        floating=list(state),native_entry_offset=address-min(r[1] for r in regions))
    validate_record(content)
    return content,digest(content)


def fingerprint(source):
    try:return _fingerprint(source)
    except RuntimeUnavailable:raise
    except (OSError,AttributeError,TypeError,ValueError,ctypes.ArgumentError) as error:
        raise RuntimeUnavailable('Current numerical-runtime evidence is unavailable') from error


def validate_record(value):
    """Bound the data-only runtime record before a wrapper may attest it."""
    sha=lambda v:isinstance(v,str) and re.fullmatch('[a-f0-9]{64}',v) is not None
    keys={'schema','native','numerical','libc','libraries','cpu','floating','native_entry_offset'}
    if (not isinstance(value,dict) or set(value)!=keys or value['schema']!=SCHEMA
            or not isinstance(value['native'],dict) or set(value['native'])!={'kernel_sha256','binary_sha256'}
            or any(not sha(v) for v in value['native'].values())
            or not isinstance(value['libc'],list) or len(value['libc'])!=2 or value['libc'][0]!='glibc'
            or any(not isinstance(v,str) or not 1<=len(v)<=64 for v in value['libc'])
            or not isinstance(value['libraries'],dict) or not 1<=len(value['libraries'])<=32
            or any(not isinstance(k,str) or re.fullmatch('[a-zA-Z0-9_.+-]{1,128}',k) is None
                   or not sha(v) for k,v in value['libraries'].items())
            or value['native']['binary_sha256'] not in value['libraries'].values()
            or type(value['native_entry_offset']) is not int
            or not 0<=value['native_entry_offset']<MAX_LIBRARY_BYTES
            or not isinstance(value['floating'],list) or len(value['floating'])!=2
            or any(type(v)is not int or v<0 for v in value['floating'])
            or value['floating'][0]&~0x1f7f or value['floating'][1]&~0xffc0
            or not isinstance(value['cpu'],list) or not 1<=len(value['cpu'])<=64):
        raise ValueError('Invalid bounded native kernel runtime')
    required={'vendor_id','cpu family','model','stepping','flags'}
    for row in value['cpu']:
        if (not isinstance(row,dict) or not required<=row.keys() or set(row)-required-{'microcode'}
                or any(not isinstance(v,str) or not 1<=len(v)<=8192 or '\x00' in v or '\n' in v
                       for v in row.values())):
            raise ValueError('Invalid native CPU class')
    numerical=value['numerical']
    if (not isinstance(numerical,dict) or numerical.get('schema')!='radar-numerical-runtime/v1'
            or numerical.get('system')!='Linux' or numerical.get('machine')!='x86_64'
            or numerical.get('byteorder')!='little'
            or len(json.dumps(value,allow_nan=False).encode())>128*1024):
        raise ValueError('Invalid native numerical runtime')
    return value


def origin_identity(arrays_sha256,runtime_sha256):
    """Bind semantic immutable array bytes to their qualified native runtime."""
    if any(not isinstance(v,str) or re.fullmatch('[a-f0-9]{64}',v) is None
           for v in (arrays_sha256,runtime_sha256)):
        raise ValueError('Native kernel origin requires array and runtime checksums')
    return digest(dict(schema='radar-native-tof-kernel-origin/v1',
        arrays_sha256=arrays_sha256,runtime_sha256=runtime_sha256))
