"""Reviewed cumulative updates: one active job and one latest pending revision.
Every revision replaces the likelihood. The nuclear response stays fixed.
"""
import asyncio, copy, json, os, time, uuid
from pathlib import Path
import numpy as np
from .contracts import file_hash, validate_request
from .snapshot import density_to_tof


from gpx_recipe.live.io import atomic_json


def scope_of(row):
    return {k:str(row.get(k,'')) for k in ('run','sample_id','sample','wavelength')}


class ReviewRequired(ValueError):pass


class ObservationStream:
    def __init__(self, session, ipts, run, binding, request, directory):
        self.session=session;self.ipts=str(ipts);self.run=str(run)
        self.binding=copy.deepcopy(binding);self.request=copy.deepcopy(request)
        validate_request(self.request,self.binding)
        row,receipt,pattern=session.capture(ipts,run)
        if receipt['content_sha256']!=binding['source_dat_sha256']:
            raise ValueError('Review the current snapshot before starting updates')
        self.scope=scope_of(row);self.grid=pattern.x.copy();self.mask=pattern.mask.copy()
        self.root=Path(directory);self.root.mkdir(parents=True,exist_ok=False)
        source=self.root/'reviewed-source.dat';source.write_bytes(session.archive.read(receipt))
        self.binding['source_dat']=str(source);self.last_content=None
        atomic_json(self.root/'review.json',dict(ipts=self.ipts,run=self.run,scope=self.scope,binding=self.binding,request=self.request,
            conditional_fixed_response=True,resume_requires_review=True,forecast_enabled=False))

    def capture(self):
        row,receipt,pattern=self.session.capture(self.ipts,self.run)
        if scope_of(row)!=self.scope:raise ReviewRequired('Sample or configuration changed; review a new response')
        if not np.array_equal(pattern.x,self.grid) or not np.array_equal(pattern.mask,self.mask):
            raise ReviewRequired('Observation grid or mask changed; review native preparation again')
        if receipt['content_sha256']==self.last_content:return None
        validate_request(self.request,self.binding)
        difc=self.binding['display_export_difc'];dx=np.diff(pattern.x)
        if not np.allclose(dx,dx[0],rtol=1e-6,atol=1e-10):raise ValueError('Uniform exported bins required')
        a=density_to_tof(pattern,d_edges=np.r_[pattern.x[0]-dx[0]/2,pattern.x+dx[0]/2],export_difc=difc)
        folder=self.root/'revisions'/receipt['revision_id'];folder.mkdir(parents=True,exist_ok=True)
        source=folder/'source.dat';source.write_bytes(self.session.archive.read(receipt))
        observation=folder/'observation.npz';np.savez_compressed(observation,x=a['x'],y=a['y'],sigma=a['sigma'],mask=pattern.mask)
        req=copy.deepcopy(self.request)
        req.update(source_revision=f"PG3_{self.run}:{receipt['content_sha256']}",observation=str(observation),observation_sha256=file_hash(observation),
                   observation_source=str(source),observation_source_sha256=receipt['content_sha256'],cumulative_policy='replace',stream_scope=self.scope)
        b=copy.deepcopy(self.binding);b.update(source_dat=str(source),source_dat_sha256=receipt['content_sha256'],observation_sha256=req['observation_sha256'])
        validate_request(req,b);atomic_json(folder/'receipt.json',receipt)
        self.last_content=receipt['content_sha256']
        return dict(id=receipt['revision_id'],captured_monotonic=time.monotonic(),binding=b,request=req,receipt=receipt)


class LatestQueue:
    """Persistent receipts; restart always requires a fresh review."""
    def __init__(self,path):
        self.path=Path(path);self.pending=None;self.active=None;self.closed=False
        self.state=dict(schema='webmon-stream-queue/1',status='reviewed',offered=0,coalesced=0,completed=[],failed=[],interrupted_previous=False)
        if self.path.exists():
            old=json.loads(self.path.read_text());self.state['interrupted_previous']=old.get('status') in ('running','queued')
            self.state['previous_receipt']=old
        self.save()
    def save(self):
        atomic_json(self.path,dict(self.state,active=self.active,pending=self.pending['id'] if self.pending else None))
    def offer(self,item):
        if self.closed:raise RuntimeError('Stream queue is closed')
        if item is None:return False
        known=set(self.state['completed'])|{self.active,self.pending['id'] if self.pending else None}
        if item['id'] in known:return False
        if self.pending:self.state['coalesced']+=1
        self.pending=item;self.state['offered']+=1;self.state['status']='queued';self.save();return True
    async def drain(self,execute):
        if self.active or self.closed:return
        while self.pending and not self.closed:
            item=self.pending;self.pending=None;self.active=item['id'];self.state['status']='running';self.save()
            try:
                await execute(item);self.state['completed'].append(item['id'])
            except asyncio.CancelledError:
                self.state['status']='cancelled';raise
            except Exception as exc:
                self.state['failed'].append(dict(id=item['id'],reason=str(exc)));self.state['status']='failed'
                self.closed=True;self.pending=None;raise
            finally:
                self.active=None
                if self.state['status']=='running':self.state['status']='idle'
                self.save()
    def close(self):
        self.closed=True;self.pending=None;self.state['status']='stopped';self.save()
