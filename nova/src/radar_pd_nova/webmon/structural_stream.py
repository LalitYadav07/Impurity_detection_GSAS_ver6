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


async def _native_step(adapter,payload,item,root,*,source,cancel,row,handle_name,submit,collect,accept,publish,poll_seconds):
    """One durable lifecycle for optional ranking and scientific point fits."""
    def stopped():return cancel is not None and cancel.is_set()
    if stopped():
        row.update(state='not_started',reason='Monitoring paused before dispatch');publish();return False
    saved=Path(item['folder'])/handle_name
    def retain(handle):write_json(saved,handle)
    try:
        row['state']='submitting';publish()
        handle=await asyncio.to_thread(submit,payload,root,source=source,cancel=cancel,on_intent=retain)
        retain(handle);row['dispatch_key']=handle['key'];started=time.monotonic()
        while True:
            if stopped():
                row.update(state='waiting_stopped',reason='Existing Galaxy job retained; recover its saved handle');publish();return True
            status=await asyncio.to_thread(adapter.poll,handle)
            handle=status.get('handle',handle);retain(handle);row['state']=status['state'];publish()
            if row['state']=='finished':
                result,seconds=await asyncio.to_thread(collect,handle,root)
                row.update(accept(result,seconds));publish();return False
            if row['state'] in {'failed','cancelled'}:
                row['reason']=status.get('message') or 'Galaxy job '+row['state'];publish();return False
            if row['state'] in {'needs_recovery','not_submitted'}:
                row['reason']='Recover the saved dispatch before admitting another snapshot';publish();return True
            if time.monotonic()-started>getattr(adapter.runtime,'timeout',7200):
                row.update(state='waiting_stopped',reason='Galaxy job is still pending; saved handle retained');publish();return True
            await asyncio.sleep(poll_seconds)
    except Exception as exc:
        handle=json.loads(saved.read_text()) if saved.is_file() else None
        pending=handle is not None and handle.get('submission_started') and row.get('state') not in {'failed','cancelled'}
        reason=str(exc)
        if pending and row.get('state') in {'finished','verified_point_fit','ranked'}:
            reason='Galaxy job finished; result collection needs recovery: '+reason
        row.update(state='needs_recovery' if pending else 'failed',reason=reason);publish();return bool(pending)


async def analyse_snapshot(adapter,item,root,*,source,labels=None,cancel=None,on_progress=None,on_ranking_progress=None,poll_seconds=3.):
    """Rank inside an existing fit; reprioritize only still-pending work."""
    root=Path(root).resolve();labels=labels or {};rows=[];protocol=None;ranking_row=None;payloads={}
    for index,payload in enumerate(item['payloads']):
        job,current,candidate=reviewed_job(payload['directory']);protocol=protocol or current
        if digest(current)!=digest(protocol):raise ValueError('Structural snapshot contains conflicting reviewed protocols')
        if candidate['id'] in payloads:raise ValueError('A structural snapshot cannot duplicate a reviewed model')
        payloads[candidate['id']]=copy.deepcopy(payload)
        # Ranking ownership belongs to this source owner, not a restored
        # per-model template. Exactly one existing point fit may own it.
        payloads[candidate['id']].pop('rank_structural',None)
        rows.append(dict(id=job['id'],candidate=candidate['id'],name=labels.get(candidate['id'],candidate['id']),
            state='pending',priority=index+1,revision=item['revision'],snapshot_sha256=item['receipt']['content_sha256']))
    if not rows:raise ValueError('A structural snapshot needs its complete reviewed portfolio')
    if set(payloads)!={c['id'] for c in protocol['candidates']}:
        raise ValueError('A structural snapshot must retain every reviewed model')
    maximum=getattr(adapter.runtime,'max_active_jobs',1)
    if type(maximum) is not int or maximum<1:raise ValueError('A positive structural job allocation is required')
    def publish():
        value=dict(schema='radar-structural-stream-progress/v1',snapshot_sha256=item['receipt']['content_sha256'],
            revision=item['revision'],rows=rows,total=len(rows),
            terminal=sum(r['state'] in {'verified_point_fit','failed','cancelled'} for r in rows))
        if ranking_row is not None:value['ranking']=ranking_row
        write_json(Path(item['folder'])/'progress.json',value)
        if on_progress:on_progress(copy.deepcopy(rows))
    def rank_progress():
        publish()
        if on_ranking_progress:on_ranking_progress(copy.deepcopy(ranking_row))
    def stopped():return cancel is not None and cancel.is_set()
    eligible=getattr(adapter,'has_prepared_ranking',None)
    ranking_owner=None
    if not stopped() and eligible is not None and await asyncio.to_thread(eligible,item['payloads'][0]):
        ranking_owner=rows[0]['candidate']
        payloads[ranking_owner]=dict(copy.deepcopy(payloads[ranking_owner]),rank_structural=True)
        ranking_row=dict(state='pending',candidate=ranking_owner)
    else:
        ranking_row=dict(state='not_started' if stopped() else 'not_available',
            reason='Monitoring paused' if stopped() else 'No prepared structural ranking model is attached; using the reviewed model order.')
    rank_progress()
    pending_rows=list(rows);unresolved=[]
    def accept_ranking(result):
        from gpx_recipe.preparation.structural_ranking_result import validate,observation_binding
        payload=payloads[ranking_owner];observation=observation_binding(result['observation'])
        if (result.get('schema')!='radar-structural-snapshot-ranking-result/v1'
                or result.get('protocol_sha256')!=digest(protocol) or result.get('source_sha256')!=protocol['source']['sha256']
                or observation['snapshot_sha256']!=item['receipt']['content_sha256'] or observation['revision']!=item['revision']
                or observation['acquisition_sha256']!=digest([_mapping(payload['mapping']),payload['source_scope']])):
            raise ValueError('Prepared ranking belongs to another structural snapshot')
        evidence=validate(result['ranking'],protocol,observation=observation)
        write_json(Path(item['folder'])/'ranking-evidence.json',evidence)
        if evidence['ranking_available']:
            positions={key:i+1 for i,key in enumerate(evidence['candidate_order'])}
            for row in rows:row['priority']=positions[row['candidate']]
            rows.sort(key=lambda row:row['priority']);pending_rows.sort(key=lambda row:row['priority'])
        ranking_row.update(state='ranked',result=result,native_seconds=result['seconds'])
    async def worker():
        while pending_rows and not unresolved:
            # No await separates taking a row and assigning its native work.
            # Other allocated fits run while the first fit supplies advice.
            row=pending_rows.pop(0)
            def accept(result,seconds):
                if (row['candidate']==ranking_owner)!=('structural_ranking' in result):
                    raise ValueError('A structural fit changed its ranking ownership')
                if row['candidate']==ranking_owner:accept_ranking(result['structural_ranking'])
                return dict(state='verified_point_fit',result=result,native_seconds=seconds,
                    rwp=result['result']['metrics']['rwp_native'],seconds_from_capture=time.monotonic()-item['captured_monotonic'])
            def point_progress():
                if row['candidate']==ranking_owner:
                    if ranking_row['state']!='ranked':
                        ranking_row.update(state=row['state'])
                        if row.get('reason'):ranking_row['reason']=row['reason']
                    rank_progress()
                else:publish()
            pending=await _native_step(adapter,payloads[row['candidate']],item,root,source=source,cancel=cancel,row=row,
                handle_name='handle-'+digest(row['candidate'])[:24]+'.json',submit=adapter.submit,collect=adapter.collect,
                accept=accept,publish=point_progress,poll_seconds=poll_seconds)
            if pending:unresolved.append(row['candidate'])
    await asyncio.gather(*(worker() for _ in range(min(maximum,len(rows)))))
    if unresolved:
        for row in pending_rows:row.update(state='not_started',reason='Recover existing dispatches before admitting more work')
        publish()
    if unresolved:raise ReviewRequired('Structural dispatch is unresolved. Existing job handles are saved; recover them before continuing snapshots.')
    return rows
