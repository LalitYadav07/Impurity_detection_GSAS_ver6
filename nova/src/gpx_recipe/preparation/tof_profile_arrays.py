"""Portable data for native-compiled TOF proposals, never a likelihood.

Integrity, producer origin and current-runtime admission are separate checks.
No uploaded flag, callable or Python object can create an owned table.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
import zipfile
import zlib

import numpy as np

from ..live.array_limits import validate_npz_bytes
from ..symmetry.contracts import digest
from .tof_kernel_arrays import array_identity,checksum
from .kernel_runtime import validate_record

SCHEMA='radar-native-tof-proposal-table/v1'
ALGORITHM='native-midpoint-cubic-halo33/v1'
MAX_BYTES=64*1024*1024
MAX_METADATA=256*1024
SHAPES=dict(coefficients=(513,8193),halo=(513,8193),logd=(513,),eta=(8193,))
DTYPES=dict(coefficients='<f8',halo='u1',logd='<f8',eta='<f8')
_ADMISSION=object()


def validate_metadata(value):
    from ..structural_live.optimization.tof_shape import validate as validate_shape
    keys={'schema','algorithm','shape','shape_sha256','runtime','runtime_sha256',
          'arrays_sha256','numeric_bytes','midpoint_checks','rejected_cells',
          'native_only_stripes','observations_included','role'}
    if (not isinstance(value,dict) or set(value)!=keys or value['schema']!=SCHEMA
            or value['algorithm']!=ALGORITHM or value['role']!='starting_proposals'
            or value['observations_included'] is not False
            or any(not checksum(value[k]) for k in ('shape_sha256','runtime_sha256','arrays_sha256'))
            or type(value['numeric_bytes']) is not int or not 0<value['numeric_bytes']<MAX_BYTES
            or any(type(value[k]) is not int or not 0<=value[k]<=512*8192
                   for k in ('midpoint_checks','rejected_cells'))
            or value['rejected_cells']>value['midpoint_checks']
            or type(value['native_only_stripes']) is not int or not 0<=value['native_only_stripes']<=512
            or len(json.dumps(value,allow_nan=False).encode())>MAX_METADATA):
        raise ValueError('Invalid bounded TOF proposal-table metadata')
    validate_shape(value['shape']);validate_record(value['runtime'])
    if digest(value['shape'])!=value['shape_sha256'] or digest(value['runtime'])!=value['runtime_sha256']:
        raise ValueError('TOF proposal identities differ from their numeric specification')
    return value


def validate_arrays(arrays):
    if (not isinstance(arrays,dict) or set(arrays)!=set(SHAPES)
            or any(not isinstance(v,np.ndarray) or v.shape!=SHAPES[k]
                   or v.dtype!=np.dtype(DTYPES[k]) or not v.flags.c_contiguous
                   for k,v in arrays.items())
            or sum(v.nbytes for v in arrays.values())>=MAX_BYTES
            or any(not np.isfinite(v).all() for v in arrays.values())
            or np.any(arrays['halo']>1)):
        raise ValueError('Invalid bounded TOF proposal-table arrays')
    for name in ('logd','eta'):
        axis=arrays[name];steps=np.diff(axis)
        if (np.any(steps<=0) or not np.allclose(steps,steps[0],rtol=1e-10,atol=1e-14)
                or np.max(abs(axis))>100):
            raise ValueError('TOF proposal coordinates need finite regular bounded axes')
    return arrays


def origin_identity(metadata):
    """Exact physical, numerical and interpolation scope, without file paths."""
    validate_metadata(metadata)
    return digest(dict(schema=SCHEMA,algorithm=metadata['algorithm'],
        shape_sha256=metadata['shape_sha256'],runtime_sha256=metadata['runtime_sha256'],
        arrays_sha256=metadata['arrays_sha256']))


class ProfileTableData:
    def __init__(self,arrays,metadata):
        validate_arrays(arrays);validate_metadata(metadata)
        if (array_identity(arrays)!=metadata['arrays_sha256']
                or sum(v.nbytes for v in arrays.values())!=metadata['numeric_bytes']):
            raise ValueError('TOF proposal metadata differs from its arrays')
        self._arrays=MappingProxyType({k:np.frombuffer(v.tobytes(),dtype=v.dtype).reshape(v.shape)
                                      for k,v in arrays.items()})
        self._metadata=copy.deepcopy(metadata)

    @property
    def metadata(self):return copy.deepcopy(self._metadata)

    def arrays(self):return dict(self._arrays)


def capture(table):
    from ..structural_live.optimization.tof_profile_table import CompiledTOFTable
    from ..structural_live.optimization.native_cell_proposals import ProposalUnavailable
    from .kernel_runtime import powder_fingerprint
    if not isinstance(table,CompiledTOFTable):raise TypeError('A currently owned TOF table is required')
    table.check_runtime()
    if not np.any(table.halo):raise ProposalUnavailable('A portable TOF proposal table needs usable interpolation cells')
    runtime,runtime_sha=powder_fingerprint(table._powder)
    arrays=table.arrays()
    metadata=dict(schema=SCHEMA,algorithm=ALGORITHM,shape=table.shape_specification,
        shape_sha256=table.shape_identity,runtime=runtime,runtime_sha256=runtime_sha,
        arrays_sha256=array_identity(arrays),numeric_bytes=table.numeric_bytes,
        midpoint_checks=table.midpoint_checks,rejected_cells=table.rejected_cells,
        native_only_stripes=table.native_only_stripes,observations_included=False,role='starting_proposals')
    table.check_runtime()
    return ProfileTableData(arrays,metadata)


def encode(data):
    if not isinstance(data,ProfileTableData):raise TypeError('Immutable validated TOF table data is required')
    stream=io.BytesIO();np.savez_compressed(stream,**data.arrays());raw=stream.getvalue()
    validate_npz_bytes(raw,maximum_bytes=MAX_BYTES)
    return raw


def decode(raw,metadata):
    if not isinstance(raw,bytes):raise TypeError('Immutable numeric bytes are required')
    validate_metadata(metadata);validate_npz_bytes(raw,maximum_bytes=MAX_BYTES)
    try:
        with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
            if len(archive.files)!=len(SHAPES) or set(archive.files)!=set(SHAPES):
                raise ValueError('Unexpected TOF table array inventory')
            arrays={k:archive[k] for k in SHAPES}
    except (zipfile.BadZipFile,zlib.error,NotImplementedError,EOFError,OSError) as error:
        raise ValueError('Malformed TOF proposal arrays') from error
    return ProfileTableData(arrays,metadata)


def _read(path,maximum):
    path=Path(path)
    fd=os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
    with os.fdopen(fd,'rb') as stream:
        before=os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0<before.st_size<=maximum:
            raise ValueError('TOF proposal artifact is absent or oversized')
        raw=stream.read(maximum+1);after=os.fstat(stream.fileno())
        if (len(raw)!=before.st_size or len(raw)>maximum
                or (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns)):
            raise ValueError('TOF proposal artifact changed while reading')
    return raw


def read(path,receipt_path):
    metadata,metadata_sha=read_metadata(receipt_path)
    raw=_read(path,MAX_BYTES);data=decode(raw,metadata)
    return data,hashlib.sha256(raw).hexdigest(),metadata_sha


def read_metadata(path):
    raw=_read(path,MAX_METADATA)
    return validate_metadata(json.loads(raw)),hashlib.sha256(raw).hexdigest()


def admit(data,native,histogram,phase,*,artifact_sha256,receipt_sha256):
    """Admit certified proposal bytes at matching actual shape and runtime.

    The table remains an initial-proposal approximation. Native likelihood,
    optimizer derivatives and accepted-GPX checks retain their original paths.
    """
    from .authority import profile_table_origin
    from .kernel_runtime import powder_fingerprint
    from ..structural_live.optimization.tof_shape import IsotropicTOFShape
    from ..structural_live.optimization.tof_profile_table import CompiledTOFTable,_remember
    from ..structural_live.optimization.native_cell_proposals import ProposalUnavailable
    if not isinstance(data,ProfileTableData):raise TypeError('Immutable TOF proposal data is required')
    native._check();metadata=data.metadata
    origin=profile_table_origin(artifact_sha256,receipt_sha256,origin_identity(metadata))
    if origin is None:raise ProposalUnavailable('TOF proposal table lacks a verified creating-job origin')
    shape=IsotropicTOFShape(native,histogram,phase)
    if shape.identity!=metadata['shape_sha256']:
        raise ProposalUnavailable('Prepared TOF shape differs from the current instrument or sample broadening')
    powder=native.math.G2pwd;_,runtime_sha=powder_fingerprint(powder)
    if runtime_sha!=metadata['runtime_sha256']:
        raise ProposalUnavailable('TOF proposal table needs reconstruction in this numerical runtime')
    table=CompiledTOFTable._from_portable(data,shape,powder,origin,_permit=_ADMISSION)
    # Recheck current reader/kernel behavior at interior qualified points.
    # These probes do not establish a global error bound or native authority.
    from scipy.ndimage import map_coordinates
    indices=np.argwhere(table.halo[1:-1:32,1:-1:128])
    indices=indices[np.linspace(0,len(indices)-1,min(17,len(indices)),dtype=int)] if len(indices) else indices
    probes=0
    for row,col in indices:
        row=int(row)*32+1;col=int(col)*128+1
        d=np.exp(table.logd[row]);parameters=shape.at(np.asarray([d]))[0]
        width=np.sqrt(parameters[2]);offset=np.sinh(table.eta[col])*width
        # The original wrapper also integrates its returned values and needs
        # two x positions. Only its first, unnormalized native value is used.
        offsets=np.asarray([offset,np.nextafter(offset,np.inf)])
        actual=float(powder.getEpsVoigt(0.,*parameters,offsets)[0][0])*width
        predicted=float(map_coordinates(table.coefficients,np.asarray([[float(row)],[float(col)]]),
            order=3,prefilter=False,mode='nearest')[0])
        if not np.isfinite(actual) or abs(actual-predicted)>max(abs(actual),1e-30)*1e-7+1e-12:
            raise ProposalUnavailable('Current native TOF probe differs from the prepared proposal table')
        probes+=1
    native._check();table.check_runtime();table.portable_probes=probes
    _remember(table)
    return table
