"""Portable native displacement coordinates, never an absence certificate.

Every map retains its complete native/gauge basis and physical Gram matrix.
The fixed data filenames carry no executable objects. Input hashes and linear
algebra checks bind coordinates; only the creating native job establishes
their experimental/preparation provenance.
"""
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import shutil

import numpy as np

from ..execution import write_json
from ..live.array_limits import validate_npz_bytes
from ..symmetry.compiler import file_sha,package_signature
from ..symmetry.contracts import fields,validate_candidate,digest
from .tof_kernel_arrays import checksum,read_bounded

SCHEMA='radar-native-structural-mode-projection/v1'
STRAIN_SCHEMA='radar-native-structural-mode-projection/v2'
SCHEMAS={SCHEMA,STRAIN_SCHEMA}
RAW_SCHEMA='radar-structural-ranking-paired-pilot/v3'
PACKED_SCHEMA='radar-structural-ranking-pairs/v3'
from .structural_supplement_pairs import SCHEMAS as ADDITIVE_SCHEMAS
PAIR_SCHEMAS={RAW_SCHEMA,PACKED_SCHEMA}|ADDITIVE_SCHEMAS
METADATA='mode-projection.json'
ARRAY_FILE='mode-projection.npz'
FILES=(METADATA,ARRAY_FILE)
ARRAYS={'physical_to_native','gauge_basis','physical_gram'}
NORMALIZATION='complete-occupied-site-cartesian-rms/native-physical-quotient/v1'
MAX_METADATA_BYTES=2*1024**2
MAX_BYTES=64*1024**2
MAX_NATIVE=2048


@dataclass(frozen=True)
class Projection:
    metadata:dict
    arrays:dict
    stamp:dict


def marker(value):
    d=fields(value,['schema','metadata_sha256','arrays_sha256'])
    if d['schema'] not in SCHEMAS or not all(checksum(d[k]) for k in ['metadata_sha256','arrays_sha256']):
        raise ValueError('A native mode projection needs its complete typed file identities')
    return d


def _metadata(value):
    d=fields(value,['schema','normalization','source_sha256','instrument_sha256','child_sha256',
        'candidate','native_dimensions','physical_dimensions','native_mode_labels','native_normalization',
        'native_module_sha256','producer_implementation_sha256','geometry_sha256','arrays_sha256',
        'observations_included','scientific_result','candidate_pruning_authorized'],{'strain':None})
    if 'strain' not in value:d.pop('strain')
    n=d['native_dimensions'];m=d['physical_dimensions']
    if (d['schema'] not in SCHEMAS or (d['schema']==STRAIN_SCHEMA)!=('strain' in d) or d['normalization']!=NORMALIZATION
            or type(n)is not int or not 0<=n<=MAX_NATIVE or type(m)is not int or not 0<=m<=n
            or any(not checksum(d[k]) for k in ['source_sha256','instrument_sha256','child_sha256',
                'producer_implementation_sha256','geometry_sha256','arrays_sha256'])
            or any(d[k] is not False for k in ['observations_included','scientific_result','candidate_pruning_authorized'])
            or not isinstance(d['native_mode_labels'],list) or len(d['native_mode_labels'])!=n
            or any(not isinstance(s,str) or not 1<=len(s)<=1000 or any(ord(c)<32 for c in s)
                   for s in d['native_mode_labels'])
            or not isinstance(d['native_normalization'],list) or len(d['native_normalization'])!=n
            or any(type(v) not in (int,float) or not np.isfinite(v) or v<=0 for v in d['native_normalization'])
            or not isinstance(d['native_module_sha256'],dict) or not 1<=len(d['native_module_sha256'])<=32
            or any(not isinstance(k,str) or not 1<=len(k)<=200 or any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.' for c in k)
                   or not checksum(v) for k,v in d['native_module_sha256'].items())):
        raise ValueError('Invalid complete native mode projection metadata')
    candidate=validate_candidate(d['candidate'])
    if (candidate!=d['candidate'] or not candidate['child'] or candidate['magnetic_space_group'] is not None
            or candidate['child']['sha256']!=d['child_sha256']):
        raise ValueError('Mode coordinates need their exact normalized nuclear child and embedding')
    if 'strain' in d:
        from .structural_strain_cif import from_metadata
        from ..structural_live.optimization.physical_prior_geometry import native_reciprocal_matrix,metric_matrix
        s=fields(d['strain'],['map','baseline_a','cell_mapping'])
        from_metadata(s['map'],parent_sha256=d['source_sha256'],candidate=candidate)
        baseline=np.asarray(s['baseline_a']);mapping=np.asarray(s['cell_mapping'])
        if (baseline.shape!=(6,) or baseline.dtype.kind not in 'fi' or not np.isfinite(baseline).all()
                or mapping.ndim!=2 or mapping.shape[0]!=6 or not 1<=mapping.shape[1]<=6
                or mapping.dtype.kind not in 'fi' or not np.isfinite(mapping).all()
                or np.linalg.matrix_rank(mapping)!=mapping.shape[1]):
            raise ValueError('Every native reciprocal-cell coordinate must join the strain map')
        metric_matrix(native_reciprocal_matrix(baseline))
    return d


def _arrays(document,arrays):
    n=document['native_dimensions'];m=document['physical_dimensions']
    shapes=dict(physical_to_native=(n,m),gauge_basis=(n,n-m),physical_gram=(m,m))
    if (set(arrays)!=ARRAYS or any(type(v)is not np.ndarray for v in arrays.values())
            or sum(v.nbytes for v in arrays.values())>MAX_BYTES
            or any(v.dtype!=np.float64 or v.shape!=shapes[k]
                   or not np.isfinite(v).all() for k,v in arrays.items())):
        raise ValueError('Native mode projection arrays have invalid dimensions, types or values')
    gram=arrays['physical_gram']
    if not np.allclose(gram,gram.T,rtol=1e-12,atol=1e-12):
        raise ValueError('The complete physical mode Gram matrix must be symmetric')
    # Reuse the same full-rank/gauge reconstruction checks as native sampling.
    # Never truncate a weak direction to obtain a numerically easier map.
    from .structural_mode_directions import from_basis
    from_basis(arrays['physical_to_native'],arrays['gauge_basis'],gram,document['native_mode_labels'])
    return {k:np.frombuffer(v.tobytes(),dtype=v.dtype).reshape(v.shape) for k,v in arrays.items()}


def read(directory,*,expected=None,stamp=None):
    directory=Path(directory)
    for name in FILES:
        if (directory/name).is_symlink():raise ValueError('Native mode projection files must be regular captured data')
    raw=read_bounded(directory/METADATA,MAX_METADATA_BYTES)
    d=_metadata(json.loads(raw));values=read_bounded(directory/ARRAY_FILE,MAX_BYTES)
    actual=dict(schema=d['schema'],metadata_sha256=hashlib.sha256(raw).hexdigest(),
        arrays_sha256=hashlib.sha256(values).hexdigest())
    if d['arrays_sha256']!=actual['arrays_sha256'] or stamp is not None and marker(stamp)!=actual:
        raise ValueError('Captured native mode projection bytes changed')
    if expected is not None:
        for key in ['source_sha256','instrument_sha256','child_sha256','candidate']:
            if d[key]!=expected[key]:raise ValueError('Mode coordinates belong to a different parent, profile, child or embedding')
        if d['physical_dimensions']!=expected['mode_dimensions']:
            raise ValueError('Mode projection does not retain every paired physical coordinate')
        if 'strain' in d and np.asarray(d['strain']['cell_mapping']).shape[1]!=expected.get('cell_dimensions'):
            raise ValueError('Strain projection does not retain every paired cell coordinate')
    validate_npz_bytes(values,maximum_bytes=MAX_BYTES)
    with np.load(io.BytesIO(values),allow_pickle=False) as archive:
        if set(archive.files)!=ARRAYS:raise ValueError('Unexpected native mode projection array inventory')
        arrays=_arrays(d,{k:archive[k] for k in archive.files})
    return Projection(d,arrays,actual)


def write(source,request,directory):
    """Capture the same geometry/normalization used by the actual fit engine."""
    from .structural_sampling import physical_grams
    directory=Path(directory)
    if any((directory/name).exists() or (directory/name).is_symlink() for name in FILES):
        raise ValueError('Native mode projection outputs are immutable')
    source_meta=source.fast.meta
    if source_meta.get('observations_included') is not False:
        raise ValueError('Only observation-free native preparation may produce these mode coordinates')
    nm=source.mode_dimensions
    if source.frozen_modes or source.fast.nm!=nm:
        raise ValueError('Mode projection must retain the complete physical native basis')
    gram,_=physical_grams(source)
    arrays=dict(physical_to_native=np.asarray(source.native_mode_map,dtype=np.float64),
        gauge_basis=np.asarray(source.fast.basis['gauge_basis'],dtype=np.float64),physical_gram=gram)
    inventory=source_meta['inventory']
    d=dict(schema=SCHEMA,normalization=NORMALIZATION,
        source_sha256=file_sha(request['source']),instrument_sha256=file_sha(request['instrument']),
        child_sha256=file_sha(request['child']),candidate=validate_candidate(request['settings']['candidate']),
        native_dimensions=len(inventory['native_mode_labels']),physical_dimensions=nm,
        native_mode_labels=inventory['native_mode_labels'],native_normalization=inventory['normalization'],
        native_module_sha256=source_meta['native_source_sha256'],producer_implementation_sha256=package_signature(),
        geometry_sha256=source.fast.empirical_constraint.geometry.fingerprint,arrays_sha256='0'*64,
        observations_included=False,scientific_result=False,candidate_pruning_authorized=False)
    from .structural_strain_cif import read as read_strain
    strain=read_strain(request['child'],parent_sha256=d['source_sha256'],candidate=d['candidate'])
    if strain is not None:
        geometry=source.fast.empirical_constraint.geometry
        d.update(schema=STRAIN_SCHEMA,strain=dict(map=strain.metadata,baseline_a=geometry.a0.tolist(),
            cell_mapping=(geometry.cell_mapping@source.lift[nm:,nm:]).tolist()))
    _metadata(d);_arrays(d,arrays)
    stream=io.BytesIO();np.savez_compressed(stream,**arrays);raw=stream.getvalue()
    validate_npz_bytes(raw,maximum_bytes=MAX_BYTES)
    d['arrays_sha256']=hashlib.sha256(raw).hexdigest();directory.mkdir(parents=True,exist_ok=True)
    (directory/ARRAY_FILE).write_bytes(raw);write_json(directory/METADATA,d)
    return read(directory).stamp


def copy(source,destination,*,expected=None,stamp=None):
    """Keep exact numerical sidecars during lossless curve packing."""
    value=read(source,expected=expected,stamp=stamp);destination=Path(destination)
    for name in FILES:
        target=destination/name
        if target.is_symlink() or target.exists():raise ValueError('Mode projection copies are immutable')
        shutil.copyfile(Path(source)/name,target)
    return read(destination,expected=expected,stamp=value.stamp)


def descriptor(projection):
    """Small JSON identity for training provenance; no projected mode labels."""
    d=projection.metadata
    return dict(candidate_id=d['candidate']['id'],identity=digest(d),stamp=projection.stamp,
        native_dimensions=d['native_dimensions'],physical_dimensions=d['physical_dimensions'],
        native_mode_labels=list(d['native_mode_labels']))


def bind_block(source,destination,projection_directory,*,names='examples',stamp):
    """Bind a recovered native map without regenerating a single saved curve.

    This is a format adapter, not native provenance qualification. Its caller
    must establish that the recovered map uses the original fitting basis.
    All parent/profile/child/embedding identities and physical dimensions are
    checked here; publication still requires its normal creating-job evidence.
    """
    from .structural_paired_blocks import read as read_block
    block=read_block(source,names=names)
    if block.mode_projection is not None:
        raise ValueError('An already mapped paired block cannot acquire a replacement projection')
    return _bind_copy(source,destination,projection_directory,names=names,stamp=stamp)


def bind_strain(source,destination,projection_directory,*,names='examples',stamp):
    """Append native strain coordinates while preserving all prior geometry.

    The caller must establish provenance of the creating native context. This
    format operation alone grants no scientific authority. A prior strain map
    cannot be replaced, and original displacement maps must match bit for bit.
    """
    from .structural_paired_blocks import read as read_block
    block=read_block(source,names=names);old=block.mode_projection
    value=read(projection_directory,expected=block.manifest,stamp=stamp)
    if old is None or 'strain' in old.metadata or value.metadata['schema']!=STRAIN_SCHEMA:
        raise ValueError('Append strain once to an existing complete displacement projection')
    compared=['source_sha256','instrument_sha256','child_sha256','candidate','normalization',
        'native_dimensions','physical_dimensions','native_mode_labels','native_normalization',
        'native_module_sha256','geometry_sha256']
    if (any(old.metadata[k]!=value.metadata[k] for k in compared)
            or any(old.arrays[k].dtype!=v.dtype or old.arrays[k].tobytes()!=v.tobytes() for k,v in value.arrays.items())):
        raise ValueError('A strain addition cannot replace original native displacement geometry')
    return _bind_copy(source,destination,projection_directory,names=names,stamp=stamp)


def _bind_copy(source,destination,projection_directory,*,names,stamp):
    from .structural_paired_blocks import read as read_block,filenames
    source=Path(source);destination=Path(destination);primary,arrays=filenames(names)
    original_sha=file_sha(source/primary)
    block=read_block(source,names=names)
    value=read(projection_directory,expected=block.manifest,stamp=stamp)
    if ('mode_coverage' in block.manifest
            and block.manifest['mode_coverage']['labels']!=value.metadata['native_mode_labels']):
        raise ValueError('Recovered mode coordinates differ from the original named axes')
    if file_sha(source/primary)!=original_sha:
        raise ValueError('The original paired manifest changed during mode-map binding')
    destination.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(source/arrays,destination/arrays)
    if file_sha(destination/arrays)!=block.manifest['pairs_sha256']:
        raise ValueError('The original paired arrays changed during mode-map binding')
    copy(projection_directory,destination,expected=block.manifest,stamp=stamp)
    document=dict(block.manifest,schema=PACKED_SCHEMA if 'array_encoding' in block.manifest else RAW_SCHEMA,
        mode_projection=value.stamp,mode_projection_source_manifest_sha256=original_sha)
    write_json(destination/primary,document)
    restored=read_block(destination,names=names)
    if (file_sha(source/primary)!=original_sha
            or file_sha(source/arrays)!=block.manifest['pairs_sha256']
            or any(restored.arrays[k].dtype!=v.dtype or restored.arrays[k].tobytes()!=v.tobytes()
                   for k,v in block.arrays.items())):
        raise ValueError('Mode-map binding changed an original paired value or input')
    return restored
