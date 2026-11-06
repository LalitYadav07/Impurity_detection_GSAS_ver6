"""Structural revisions under the existing acquisition owner and latest queue.

Capture never starts work. The caller's existing WebMon tick and LatestQueue
decide admission. Native dispatch/recovery belongs to Model Campaign runtime.
"""
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from gpx_recipe.execution import write_json
from gpx_recipe.live.snapshot import parse_live_ascii,uniform_density_to_tof
from gpx_recipe.structural_live.acquisition import _mapping,_scope
from gpx_recipe.structural_live.job import validate_payload,reviewed_job
from gpx_recipe.symmetry.compiler import file_sha
from gpx_recipe.symmetry.contracts import digest
from .stream import ReviewRequired,scope_of


class StructuralObservationStream:
    kind='structural'

    def __init__(self,session,ipts,run,templates,root):
        if not isinstance(templates,list) or not 1<=len(templates)<=500:
            raise ValueError('Review a finite nonempty structural model portfolio')
        self.session=session;self.ipts=str(ipts);self.run=str(run)
        self.root=Path(root).resolve();self.templates=copy.deepcopy(templates)
        self.root.mkdir(parents=True,exist_ok=True)
        if (self.root/'structural-stream.json').exists():
            raise ValueError('Review a new structural stream; saved jobs remain recoverable')
        identities=[];candidates=[];self.protocol_sha=None
        for template in self.templates:
            validate_payload(template)
            for key in ('directory','source','anchor_snapshot'):
                path=Path(template[key]).resolve()
                if not path.is_relative_to(self.root):raise ValueError('Capture structural inputs inside their owned workspace')
            job,protocol,candidate=reviewed_job(template['directory'])
            if file_sha(template['source'])!=protocol['source']['sha256']:
                raise ValueError('Structural source differs from its reviewed model')
            identities.append(digest([_mapping(template['mapping']),template['source_scope'],
                protocol,template['refine_transfer'],template['source_revision'],template['reference_revision']]))
            candidates.append(candidate['id']);self.protocol_sha=digest(protocol)
        if len(set(identities))!=1 or len(set(candidates))!=len(candidates):
            raise ValueError('One structural acquisition needs one reviewed protocol and unique complete model jobs')
        protocol=reviewed_job(self.templates[0]['directory'])[1]
        if set(candidates)!={c['id'] for c in protocol['candidates']}:
            raise ValueError('Capture every model in the reviewed structural portfolio')
        first=self.templates[0];self.scope=_scope(first['source_scope'],first['mapping']['identity'])
        if self.run!=self.scope['run'] or str(first['mapping']['identity']['ipts'])!=self.ipts.removeprefix('IPTS-'):
            raise ValueError('The structural stream belongs to another experiment/run')
        self.mapping=_mapping(first['mapping'])
        anchor=Path(first['anchor_snapshot']).read_bytes()
        if hashlib.sha256(anchor).hexdigest()!=first['mapping']['snapshot_sha256']:
            raise ValueError('Structural anchor changed after review')
        pattern=parse_live_ascii(anchor);self.grid=pattern.x.copy();self.spectrum=pattern.spectrum_marker
        self.last_content=None;self.revision=0;self.exposure=None;self.exposure_seen=False
        self.identity=digest([identities[0],candidates])
        write_json(self.root/'structural-stream.json',dict(schema='radar-structural-stream/v1',
            identity=self.identity,protocol_sha256=self.protocol_sha,templates=self.templates,
            cumulative_policy='replace',resume_requires_review=True,forecast_enabled=False))

    def capture(self):
        row,receipt,_=self.session.capture(self.ipts,self.run)
        if scope_of(row)!=self.scope or receipt.get('identity')!=self.mapping['identity']:
            raise ReviewRequired('Structural sample, producer epoch or acquisition changed; review a new context')
        metadata=receipt.get('metadata',{})
        for key,expected in [('sample_id',self.scope['sample_id']),('physical_bank',self.mapping['physical_bank']),
                             ('producer_config_sha256',self.mapping['producer_config_sha256'])]:
            if key in metadata and str(metadata[key])!=str(expected):
                raise ReviewRequired('Structural sample, bank or producer configuration changed')
        raw=self.session.archive.read(receipt)
        if hashlib.sha256(raw).hexdigest()!=receipt.get('content_sha256'):
            raise ValueError('Archived structural snapshot does not match its capture receipt')
        pattern=parse_live_ascii(raw)
        if pattern.spectrum_marker!=self.spectrum or not np.array_equal(pattern.x,self.grid):
            raise ReviewRequired('Structural spectrum or grid changed; review native preparation again')
        # Validate the actual variance conversion before admitting a job. A
        # newly measured bin may become valid; it is not fixed geometry.
        uniform_density_to_tof(pattern,export_difc=self.mapping['export_geometry']['microseconds_per_angstrom'])
        exposure=metadata.get('cumulative_exposure')
        if exposure is not None and (type(exposure) not in (int,float) or not np.isfinite(exposure) or exposure<=0):
            raise ReviewRequired('Structural cumulative exposure must be unknown or finite and positive')
        if self.exposure_seen and ((exposure is None)!=(self.exposure is None)
                or (exposure is not None and exposure<self.exposure)):
            raise ReviewRequired('Structural exposure semantics changed or cumulative acquisition reset')
        content=receipt['content_sha256']
        if content==self.last_content:
            self.exposure=exposure;self.exposure_seen=True
            return None
        revision=self.revision+1;key=digest([self.identity,revision,content])[:24]
        folder=self.root/'revisions'/key;folder.mkdir(parents=True,exist_ok=False)
        captured=folder/'source.dat';captured.write_bytes(raw)
        payloads=[]
        for template in self.templates:
            payload=copy.deepcopy(template)
            model=reviewed_job(template['directory'])[0]['candidate_id']
            payload.update(snapshot=str(captured),receipt=copy.deepcopy(receipt),revision=revision,
                destination=str(self.root/'results'/key/digest(model)[:24]))
            payloads.append(payload)
        item=dict(id=key,kind='structural',captured_monotonic=time.monotonic(),revision=revision,
            binding=dict(source_dat=str(captured)),payloads=payloads,receipt=copy.deepcopy(receipt),
            protocol_sha256=self.protocol_sha,folder=str(folder))
        write_json(folder/'capture.json',item)
        self.last_content=content;self.revision=revision;self.exposure=exposure;self.exposure_seen=True
        return item


async def analyse_snapshot(adapter,item,root,*,source,labels=None,cancel=None,on_progress=None,poll_seconds=3.):
    """Use the existing bounded model jobs; retain every candidate outcome."""
    root=Path(root).resolve();labels=labels or {};rows=[];unresolved=[]
    for payload in item['payloads']:
        job,_,candidate=reviewed_job(payload['directory'])
        rows.append(dict(id=job['id'],candidate=candidate['id'],name=labels.get(candidate['id'],candidate['id']),
            state='pending',revision=item['revision'],snapshot_sha256=item['receipt']['content_sha256']))
    def publish():
        write_json(Path(item['folder'])/'progress.json',dict(schema='radar-structural-stream-progress/v1',
            snapshot_sha256=item['receipt']['content_sha256'],revision=item['revision'],rows=rows,
            total=len(rows),terminal=sum(r['state'] in {'verified_point_fit','failed','cancelled'} for r in rows)))
        if on_progress:on_progress(copy.deepcopy(rows))
    publish()
    maximum=getattr(adapter.runtime,'max_active_jobs',1)
    slots=asyncio.Semaphore(maximum)
    def stopped():return cancel is not None and cancel.is_set()
    async def one(index,payload):
        row=rows[index]
        async with slots:
            if stopped():
                row.update(state='not_started',reason='Monitoring paused before dispatch');publish();return
            try:
                row['state']='submitting';publish()
                def retain(handle):
                    write_json(Path(item['folder'])/('handle-'+digest(row['candidate'])[:24]+'.json'),handle)
                handle=await asyncio.to_thread(adapter.submit,payload,root,source=source,cancel=cancel,on_intent=retain)
                retain(handle);row['dispatch_key']=handle['key']
                started=time.monotonic()
                while True:
                    if stopped():
                        row.update(state='waiting_stopped',reason='Existing Galaxy job retained; recover its saved handle')
                        unresolved.append(row['candidate']);publish();return
                    status=await asyncio.to_thread(adapter.poll,handle)
                    handle=status.get('handle',handle);retain(handle)
                    state=status['state'];row['state']=state;publish()
                    if state=='finished':
                        result,seconds=await asyncio.to_thread(adapter.collect,handle,root)
                        row.update(state='verified_point_fit',result=result,native_seconds=seconds,
                            rwp=result['result']['metrics']['rwp_native'],
                            seconds_from_capture=time.monotonic()-item['captured_monotonic'])
                        publish();return
                    if state in {'failed','cancelled'}:
                        row['reason']=status.get('message') or 'Galaxy job '+state;publish();return
                    if state in {'needs_recovery','not_submitted'}:
                        unresolved.append(row['candidate']);row['reason']='Recover the saved dispatch before admitting another snapshot';publish();return
                    if time.monotonic()-started>getattr(adapter.runtime,'timeout',7200):
                        unresolved.append(row['candidate']);row.update(state='waiting_stopped',reason='Galaxy job is still pending; saved handle retained');publish();return
                    await asyncio.sleep(poll_seconds)
            except Exception as exc:
                # An uncertain POST must not be treated as a terminal native
                # failure that allows the next acquisition to start.
                saved=Path(item['folder'])/('handle-'+digest(row['candidate'])[:24]+'.json')
                h=json.loads(saved.read_text()) if saved.is_file() else None
                pending=h is not None and h.get('submission_started') and row.get('state') not in {'finished','failed','cancelled'}
                if pending:unresolved.append(row['candidate'])
                row.update(state='needs_recovery' if pending else 'failed',reason=str(exc));publish()
    await asyncio.gather(*(one(index,payload) for index,payload in enumerate(item['payloads'])))
    if unresolved:raise ReviewRequired('Structural dispatch is unresolved. Existing job handles are saved; recover them before continuing snapshots.')
    return rows
