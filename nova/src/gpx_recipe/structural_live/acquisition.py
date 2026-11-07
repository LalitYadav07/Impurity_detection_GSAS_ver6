"""Structural snapshot handoff under an existing acquisition owner.

This adapter has no polling loop and performs no symmetry discovery. An owner
supplies an already captured snapshot, its receipt and a monotonic revision.
Each update replaces the supplied-variance objective in StructuralSession.
"""
from dataclasses import asdict
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import re
import threading

import numpy as np

from ..execution import write_json
from ..live.snapshot import parse_live_ascii, stable_read, uniform_density_to_tof
from ..symmetry.contracts import digest
from .native import sha
from .reduced import from_snapshot
from .service import SearchSettings
from .session import StructuralSession, PublicationUnavailable

SCHEMA='radar-structural-acquisition/v1'
MAX_JSON=4*1024*1024


def _mapping(value):
    """Keep the reviewed physical mapping, independently of snapshot bytes."""
    if not isinstance(value,dict) or value.get('mapping_kind')!='user_reviewed':
        raise ValueError('A reviewed producer mapping is required')
    keys=('source_gpx_sha256','physical_bank','export_geometry','normalization',
          'producer_config_sha256','identity')
    try:result=copy.deepcopy({key:value[key] for key in keys})
    except KeyError as exc:raise ValueError('The structural producer mapping is incomplete') from exc
    for key in ('source_gpx_sha256','producer_config_sha256'):
        if not re.fullmatch('[a-f0-9]{64}',str(result[key])):
            raise ValueError('The producer and reference need exact checksums')
    if type(result['physical_bank']) is not int or result['physical_bank']<1:
        raise ValueError('A physical bank is required')
    geometry=result['export_geometry']
    if not isinstance(geometry,dict) or set(geometry)!={'kind','microseconds_per_angstrom'} or geometry['kind']!='linear_tof_from_d':
        raise ValueError('Use the reviewed linear TOF exporter geometry')
    factor=geometry['microseconds_per_angstrom']
    if type(factor) not in (int,float) or not np.isfinite(factor) or factor<=0:
        raise ValueError('A finite positive producer conversion is required')
    identity=result['identity']
    if (not isinstance(identity,dict) or set(identity)!={'facility','instrument','ipts','run','epoch'}
            or identity['facility']!='SNS' or identity['instrument']!='PG3'
            or any(type(identity[key]) is not int or identity[key]<1 for key in ('ipts','run'))
            or not isinstance(identity['epoch'],str) or not 1<=len(identity['epoch'])<=200):
        raise ValueError('An exact POWGEN acquisition and producer epoch are required')
    if not isinstance(result['normalization'],str) or not 5<=len(result['normalization'])<=1000:
        raise ValueError('Review the producer normalization')
    return result


def _scope(value,identity):
    if (not isinstance(value,dict) or set(value)!={'run','sample_id','sample','wavelength'}
            or any(not isinstance(v,str) or not v.strip() or len(v)>1000 for v in value.values())
            or value['run']!=str(identity['run'])):
        raise ValueError('The acquisition owner must supply the reviewed run/sample/configuration scope')
    return copy.deepcopy(value)


def _json(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>MAX_JSON:
        raise ValueError('The structural acquisition receipt is missing or oversized')
    return json.loads(path.read_text())


def prepare_acquisition(request,child,transform,origin,directory,*,source_scope,
                        settings=None,prepared_banks=(),prepared_profiles=(),reference_gpx=None):
    """Transfer the nuclear reference once, then use the shared mode engine.

    Experiment snapshots remain outside portable theory-only libraries. The
    same snapshot conversion is used by the magnetic and structural callers.
    """
    from ..live.nuclear_transfer import prepare_nuclear_snapshot
    from .service import prepare_model
    from .native import modules
    mapping=_mapping(request.get('mapping'))
    source_scope=_scope(source_scope,mapping['identity'])
    settings=settings or SearchSettings()
    if not isinstance(settings,SearchSettings):raise TypeError('Explicit SearchSettings required')
    root=Path(directory);root.mkdir(parents=True,exist_ok=False)
    transfer=prepare_nuclear_snapshot(request,root)
    write_json(root/'transfer.json',transfer)
    metadata=prepare_model(root/transfer['source_gpx'],child,transform,origin,root/'model',
        phase_name=request['parent_phase'],settings=settings,prepared_banks=prepared_banks,prepared_profiles=prepared_profiles,
        reference_gpx=reference_gpx)
    G,*_=modules();project=G.G2Project(gpxfile=str(root/'model'/'prepared.gpx'))
    histogram=project.histogram(metadata['histogram'])
    raw=stable_read(request['observation_source'])
    if hashlib.sha256(raw).hexdigest()!=request['observation_source_sha256']:
        raise ValueError('The captured snapshot changed during structural preparation')
    pattern=parse_live_ascii(raw)
    transformed=uniform_density_to_tof(pattern,export_difc=mapping['export_geometry']['microseconds_per_angstrom'])
    x=np.asarray(np.ma.getdata(histogram.data['data'][1][0]),float)
    if not np.array_equal(x,transformed['x']):
        raise ValueError('The structural native context changed the producer grid')
    np.savez_compressed(root/'grid.npz',x=x,initial_invalid_mask=pattern.mask)
    contract=dict(schema=SCHEMA,prepared_sha256=sha(root/'model'/'prepared.json'),
        transfer_sha256=sha(root/'transfer.json'),grid_sha256=sha(root/'grid.npz'),
        mapping=mapping,source_scope=source_scope,spectrum_marker=pattern.spectrum_marker,
        limits=copy.deepcopy(histogram.data['Limits']),settings=asdict(settings),
        acquisition=digest([mapping,source_scope]),forecast_enabled=False,
        statistical_policy_qualified=False,kind='structural_point_fit')
    write_json(root/'acquisition.json',contract)
    return contract


class StructuralAcquisition:
    """One candidate and acquisition, invoked by the shared source owner."""
    def __init__(self,prepared,directory,*,runner=None):
        from .service import fit_snapshot
        from ..live.array_limits import validate_npz
        self.prepared=Path(prepared).resolve();self.root=Path(directory).resolve()
        self.contract=_json(self.prepared/'acquisition.json')
        c=self.contract
        if (set(c)!={'schema','prepared_sha256','transfer_sha256','grid_sha256','mapping',
                    'source_scope','spectrum_marker','limits','settings','acquisition','forecast_enabled',
                    'statistical_policy_qualified','kind'} or c['schema']!=SCHEMA
                or c['kind']!='structural_point_fit' or c['forecast_enabled'] is not False
                or c['statistical_policy_qualified'] is not False
                or type(c['spectrum_marker']) is not int or c['spectrum_marker']<0):
            raise ValueError('Unsupported structural acquisition contract')
        mapping=_mapping(dict(c['mapping'],mapping_kind='user_reviewed'))
        scope=_scope(c['source_scope'],mapping['identity'])
        if c['acquisition']!=digest([mapping,scope]):raise ValueError('Structural acquisition identity changed')
        for name,key in [('model/prepared.json','prepared_sha256'),('transfer.json','transfer_sha256'),('grid.npz','grid_sha256')]:
            if sha(self.prepared/name)!=c[key]:raise ValueError('Prepared structural acquisition changed')
        validate_npz(self.prepared/'grid.npz',maximum_bytes=32*1024*1024)
        with np.load(self.prepared/'grid.npz',allow_pickle=False) as data:
            if set(data.files)!={'x','initial_invalid_mask'}:raise ValueError('Unexpected structural snapshot grid inventory')
            self.x=data['x'].copy();self.initial_invalid=data['initial_invalid_mask'].copy()
        if (self.x.ndim!=1 or self.x.dtype.kind!='f' or self.x.dtype.itemsize!=8
                or self.initial_invalid.shape!=self.x.shape or self.initial_invalid.dtype!=bool
                or len(self.x)<2 or not np.isfinite(self.x).all() or not np.all(np.diff(self.x)>0)):
            raise ValueError('Invalid structural snapshot grid')
        limits=c['limits']
        if (not isinstance(limits,list) or len(limits)<2 or any(
                not isinstance(pair,list) or len(pair)!=2 or any(type(v) not in (int,float)
                or not np.isfinite(v) for v in pair) or pair[0]>pair[1] for pair in limits)
                or limits[0][0]>=limits[0][1] or limits[1][0]>=limits[1][1]):
            raise ValueError('Review the native structural fit limits and exclusions')
        self.root.mkdir(parents=True,exist_ok=True);self.lock=threading.Lock()
        self.runner=runner or fit_snapshot
        self._reload_session()

    def _reload_session(self):
        c=self.contract
        if _json(self.prepared/'acquisition.json')!=c:
            raise ValueError('The structural acquisition contract changed')
        for name,key in [('model/prepared.json','prepared_sha256'),('transfer.json','transfer_sha256'),('grid.npz','grid_sha256')]:
            if sha(self.prepared/name)!=c[key]:raise ValueError('Prepared structural acquisition changed')
        transfer=_json(self.prepared/'transfer.json')
        for field,path,checksum in [('source_gpx','prepared/source.gpx','source_sha256'),
                                    ('source_dat','inputs/source.dat','source_dat_sha256')]:
            if transfer.get(field)!=path or sha(self.prepared/path)!=transfer.get(checksum):
                raise ValueError('The original structural snapshot transfer artifact changed')
        self.session=StructuralSession(self.prepared/'model',self.root/'fits',
            settings=SearchSettings(**c['settings']),runner=self.runner)
        if self.session.record:
            prior=self.session.record
            path=self.root/'snapshots'/('revision-'+str(prior['revision']))
            binding=_json(path/'binding.json');raw=stable_read(path/'source.dat')
            observation=self._observation(raw,binding['receipt'],revision=prior['revision'],
                current_scope=binding['scope'],current_mapping=binding['mapping'])
            if observation.observation_sha256!=prior['observation_sha256']:
                raise PublicationUnavailable('Published structural fit does not bind its captured snapshot')

    @contextmanager
    def _writer(self):
        import fcntl
        path=self.root/'.acquisition.lock'
        if path.is_symlink():raise ValueError('The acquisition writer lock is not owned')
        with path.open('a+b') as stream:
            try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('Another acquisition owner is updating this structural series') from exc
            try:yield
            finally:fcntl.flock(stream,fcntl.LOCK_UN)

    def _observation(self,raw,receipt,*,revision,current_scope,current_mapping):
        c=self.contract
        if not isinstance(raw,bytes) or not 1<=len(raw)<=20_000_000:
            raise ValueError('A bounded immutable captured snapshot is required')
        if (_mapping(current_mapping)!=c['mapping'] or _scope(current_scope,c['mapping']['identity'])!=c['source_scope']):
            raise ValueError('Producer mapping, sample or acquisition scope changed; review a new structural context')
        if (not isinstance(receipt,dict) or receipt.get('identity')!=c['mapping']['identity']
                or receipt.get('content_sha256')!=hashlib.sha256(raw).hexdigest()
                or not isinstance(receipt.get('metadata',{}),dict)):
            raise ValueError('The structural snapshot receipt does not bind its source bytes or acquisition')
        metadata=receipt.get('metadata',{})
        for key,expected in [('sample_id',c['source_scope']['sample_id']),
                             ('physical_bank',c['mapping']['physical_bank']),
                             ('producer_config_sha256',c['mapping']['producer_config_sha256'])]:
            if key in metadata and str(metadata[key])!=str(expected):
                raise ValueError('Captured sample, bank or producer metadata changed')
        pattern=parse_live_ascii(raw)
        transformed=uniform_density_to_tof(pattern,export_difc=c['mapping']['export_geometry']['microseconds_per_angstrom'])
        if pattern.spectrum_marker!=c['spectrum_marker'] or not np.array_equal(transformed['x'],self.x):
            raise ValueError('Snapshot spectrum or grid changed; review native structural preparation again')
        # Zero-weight reduced bins can acquire valid measurements as counts
        # arrive. Their mask belongs to this observation, not the prepared
        # physics. The shared service separately enforces native fit limits
        # and any genuinely masked native grid coordinates.
        return from_snapshot(transformed['x'],transformed['y'],transformed['sigma'],pattern.mask,c['limits'],
            acquisition=c['acquisition'],protocol_sha256=c['prepared_sha256'],revision=revision,
            exposure=receipt.get('metadata',{}).get('cumulative_exposure'))

    def update(self,raw,receipt,*,revision,current_scope,current_mapping):
        if not self.lock.acquire(blocking=False):raise RuntimeError('One acquisition owner must serialize structural snapshots')
        try:
            with self._writer():
                # A different recovered owner may have advanced the ledger
                # since this object was created. Validate that publication
                # before accepting an update or returning a duplicate.
                self._reload_session()
                return self._update(raw,copy.deepcopy(receipt),revision=revision,
                    current_scope=copy.deepcopy(current_scope),current_mapping=copy.deepcopy(current_mapping))
        finally:self.lock.release()

    def _update(self,raw,receipt,*,revision,current_scope,current_mapping):
        observation=self._observation(raw,receipt,revision=revision,
            current_scope=current_scope,current_mapping=current_mapping)
        path=self.root/'snapshots'/('revision-'+str(revision))
        binding=dict(receipt=receipt,scope=current_scope,mapping=current_mapping,
            observation_sha256=observation.observation_sha256)
        # Seal input lineage before the fit can publish. An interrupted fit
        # can retry the identical input without losing the prior result.
        if path.exists():
            if (_json(path/'binding.json')!=binding or stable_read(path/'source.dat')!=raw):
                raise ValueError('A structural snapshot changed without a new owner revision')
        else:
            import tempfile
            temporary=Path(tempfile.mkdtemp(prefix='.snapshot-',dir=self.root))
            try:
                (temporary/'source.dat').write_bytes(raw);write_json(temporary/'binding.json',binding)
                path.parent.mkdir(exist_ok=True);temporary.rename(path)
            finally:
                import shutil
                shutil.rmtree(temporary,ignore_errors=True)
        result=self.session.update(observation)
        return dict(result,kind='structural_point_fit',snapshot_sha256=receipt['content_sha256'],
            snapshot_revision=revision,forecast_enabled=False,model_exclusion_authorized=False)
