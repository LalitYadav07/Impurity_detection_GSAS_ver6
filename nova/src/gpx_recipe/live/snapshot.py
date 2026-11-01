"""POWGEN saved-live ASCII bridge. No WebMon scraping or instrument control.

The parser preserves reduced observations; it never creates Poisson counts.
Geometry/profile/exposure must come from captured producer metadata. A stable
read is a consumer heuristic, not a substitute for atomic producer publication.
"""
from dataclasses import dataclass
from datetime import datetime,timezone
import hashlib,io,json,os,re,tempfile
from pathlib import Path
import numpy as np

VERSION='powgen-reduced-snapshot/1'

@dataclass
class ReducedPattern:
    x: np.ndarray
    y: np.ndarray
    sigma: np.ndarray
    mask: np.ndarray
    spectrum_marker: int


def parse_live_ascii(raw,*,expected_points=None):
    if not isinstance(raw,bytes) or not raw or len(raw)>20_000_000:
        raise ValueError('Expected bounded nonempty ASCII bytes')
    if not raw.endswith(b'\n'):raise ValueError('Incomplete final line; retry capture')
    lines=raw.decode('utf-8').splitlines()
    if len(lines)<4 or not re.fullmatch(r'#\s*X\s*,\s*Y\s*,\s*E\s+Distribution=true\s*',lines[0]):
        raise ValueError('Unqualified header or distribution convention')
    if not re.fullmatch(r'\d+',lines[1].strip()):raise ValueError('Missing spectrum marker')
    values=np.loadtxt(io.StringIO('\n'.join(lines[2:])),delimiter=',',ndmin=2)
    if values.ndim!=2 or values.shape[1]!=3 or len(values)<2 or not np.isfinite(values).all():
        raise ValueError('Expected finite d,Y,sigma rows')
    d,y,e=values.T
    if np.any(d<=0) or np.any(np.diff(d)<=0) or np.any(e<0):raise ValueError('Invalid d-grid or uncertainty')
    mask=(e==0)&(y==0)
    if np.any((e==0)&~mask):raise ValueError('Missing uncertainty on a nonzero observation')
    if expected_points is not None and len(d)!=expected_points:raise ValueError('Snapshot shape differs from producer contract')
    return ReducedPattern(d.copy(),y.copy(),e.copy(),mask,int(lines[1]))


def density_to_tof(pattern,*,d_edges,export_difc):
    """Exact bin-integral preserving map using the EXPORT geometry, not profile DIFC."""
    edges=np.asarray(d_edges,float)
    if (edges.shape!=(len(pattern.x)+1,) or not np.isfinite(edges).all() or np.any(edges<=0)
            or np.any(np.diff(edges)<=0) or not np.allclose((edges[:-1]+edges[1:])/2,pattern.x,rtol=0,atol=1e-8)
            or not np.isfinite(export_difc) or export_difc<=0):
        raise ValueError('Exact producer edges and positive focused export DIFC required')
    target=edges*export_difc;factor=np.diff(edges)/np.diff(target)
    y=pattern.y*factor;sigma=pattern.sigma*factor;weights=np.zeros_like(sigma)
    np.divide(1.,sigma*sigma,out=weights,where=~pattern.mask)
    return dict(x=pattern.x*export_difc,edges=target,y=y,sigma=sigma,weights=weights,mask=pattern.mask.copy())


def _signature(stat):return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)

def stable_read(path,*,limit=20_000_000):
    """Two matching bounded reads; never accepts a renamed or truncated source."""
    path=Path(path);before=path.stat()
    if before.st_size>limit:raise ValueError('Snapshot exceeds capture budget')
    with path.open('rb') as stream:
        if _signature(os.fstat(stream.fileno()))!=_signature(before):raise RuntimeError('Source changed before read')
        first=stream.read(limit+1)
        if len(first)>limit:raise ValueError('Snapshot exceeds capture budget')
        middle=os.fstat(stream.fileno());stream.seek(0);second=stream.read(limit+1);after=os.fstat(stream.fileno())
    if len(first)!=before.st_size or first!=second or any(_signature(s)!=_signature(before) for s in (middle,after,path.stat())):
        raise RuntimeError('Source changed during capture; keep last good revision')
    return first


def _hash_json(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def admission(metadata):
    required=('physical_bank','profile_sha256','profile_original_bank','producer_config_sha256',
              'normalization','export_geometry','grid_contract','sample_id')
    missing=[key for key in required if not metadata.get(key)]
    return dict(preview_allowed=True,missing_metadata=missing,
        numerical_metadata_complete=not missing,
        metadata_verified_before_final=metadata.get('metadata_verified_before_final',False) is True,
        cumulative_exposure_known=metadata.get('cumulative_exposure') is not None,
        native_import_qualified=False,statistical_forecast_enabled=False,
        reason='Requires native-import/profile qualification and an empirically validated reduced-data likelihood')


class SnapshotArchive:
    """Single-writer local archive; commit a revision directory atomically."""
    def __init__(self,root):self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
    def read(self,receipt):
        if not all(re.fullmatch(r'[0-9a-f]{64}',str(receipt.get(k,''))) for k in ('scope_id','revision_id','content_sha256')):raise ValueError('Invalid archived revision')
        path=self.root/receipt['scope_id']/receipt['revision_id']/'source.dat'
        if not path.resolve().is_relative_to(self.root.resolve()):raise ValueError('Archive path escapes root')
        raw=stable_read(path)
        if hashlib.sha256(raw).hexdigest()!=receipt['content_sha256']:raise ValueError('Archive checksum mismatch')
        return raw
    def capture(self,path,*,ipts,run,epoch,metadata):
        if not str(ipts).isdigit() or not str(run).isdigit() or not str(epoch).strip():raise ValueError('IPTS/run and producer epoch required')
        path=Path(path)
        if path.name!=f'PG3_{run}.dat':raise ValueError('Filename/run mismatch')
        raw=stable_read(path);pattern=parse_live_ascii(raw,expected_points=metadata.get('expected_points'))
        identity=dict(facility='SNS',instrument='PG3',ipts=int(ipts),run=int(run),epoch=str(epoch))
        # Exposure and observation time are revision properties, not scientific configuration.
        scientific={k:v for k,v in metadata.items() if k not in ('cumulative_exposure','captured_at')}
        scope=_hash_json(dict(identity=identity,scientific=scientific))
        content=hashlib.sha256(raw).hexdigest()
        revision=_hash_json(dict(scope=scope,content=content,exposure=metadata.get('cumulative_exposure')))
        folder=self.root/scope/revision
        if (folder/'receipt.json').exists():
            receipt=json.loads((folder/'receipt.json').read_text())
            if receipt.get('revision_id')!=revision or receipt.get('scope_id')!=scope or receipt.get('content_sha256')!=content:raise ValueError('Archive receipt mismatch')
            self.read(receipt)
            return dict(receipt,duplicate=True)
        folder.parent.mkdir(parents=True,exist_ok=True)
        prior=[]
        for p in folder.parent.glob('*/receipt.json'):
            if not re.fullmatch(r'[0-9a-f]{64}',p.parent.name):continue  # ignore uncommitted crash remnants
            r=json.loads(p.read_text());exposure=r['metadata'].get('cumulative_exposure')
            if exposure is not None:prior.append(float(exposure))
        exposure=metadata.get('cumulative_exposure')
        if exposure is not None:
            if not isinstance(exposure,(int,float)) or not np.isfinite(exposure) or exposure<=0:raise ValueError('Positive exposure required when known')
            if prior and exposure<max(prior):raise ValueError('Exposure reset: a new acquisition epoch is required')
        receipt=dict(schema=VERSION,identity=identity,scope_id=scope,revision_id=revision,content_sha256=content,
            source_path=str(path),captured_at=datetime.now(timezone.utc).isoformat(),stage='live_provisional',
            metadata=metadata,spectrum_marker=pattern.spectrum_marker,physical_bank=metadata.get('physical_bank'),
            points=len(pattern.x),signed_points=int(np.sum(pattern.y<0)),masked_points=int(pattern.mask.sum()),
            exposure_inferred_from_clock=False,admission=admission(metadata),duplicate=False)
        temporary=Path(tempfile.mkdtemp(prefix='.capture-',dir=folder.parent))
        try:
            (temporary/'source.dat').write_bytes(raw)
            np.savez_compressed(temporary/'observations.npz',d=pattern.x,y=pattern.y,sigma=pattern.sigma,mask=pattern.mask)
            (temporary/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False))
            try:temporary.rename(folder)
            except OSError:
                # Another reader may have committed the identical revision.
                if not (folder/'receipt.json').is_file():raise
                existing=json.loads((folder/'receipt.json').read_text())
                if existing.get('revision_id')!=revision or self.read(existing)!=raw:raise ValueError('Concurrent archive revision differs')
                receipt=dict(existing,duplicate=True)
        except Exception:
            import shutil
            shutil.rmtree(temporary,ignore_errors=True);raise
        finally:
            import shutil
            shutil.rmtree(temporary,ignore_errors=True)
        return receipt
