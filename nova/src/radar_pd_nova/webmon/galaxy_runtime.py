"""Durable Galaxy dispatch for provisional scan jobs; no native code in NOVA."""
from __future__ import annotations
import hashlib,json,re,time,uuid
from pathlib import Path
from gpx_recipe.live.packet import create_packet,canonical,extract_result_archive,scientific_packet_identity
from gpx_recipe.live.contracts import file_hash
from ..refinement_runtime import GalaxyRefinementRuntime,RuntimeUnavailable

LIVE_TOOL_ID='neutrons_radar_pd_live_snapshot_prototype'


class GalaxyLiveRuntime(GalaxyRefinementRuntime):
    tool_id=LIVE_TOOL_ID
    workspace_label='RADAR-PD live snapshot workspace'
    workspace_schema='radar-live-workspace/v1'

    def _prepared_request(self,request,key):
        """Freeze optional preparation independently of a lost job reply."""
        if request.get('action')=='prepare':return request,None
        from gpx_recipe.preparation.hints import live_context,read
        from gpx_recipe.execution import write_json
        from gpx_recipe.symmetry.contracts import digest
        from ..native_operations import intent_lock
        catalog=Path(request['catalog'])
        if catalog.stat().st_size>4*1024*1024:raise ValueError('Live catalog exceeds its metadata budget')
        context=live_context(request,json.loads(catalog.read_text()))
        root=self.root/'prepared-inputs'/key;root.mkdir(parents=True,exist_ok=True)
        choice=root/'choice.json'
        with intent_lock(root/'choice-intent.json'):
            if choice.exists():
                saved=json.loads(choice.read_text())
                if saved.get('context_sha256')!=digest(context):
                    raise ValueError('Live dispatch already captured different preparation choices')
            else:
                try:
                    provider=getattr(self,'preparation_provider',None)
                    if callable(provider):provider(context,root)
                    document=read(root,context)
                    saved=dict(context_sha256=digest(context),selected=document is not None)
                except (OSError,ValueError,EOFError) as error:
                    saved=dict(context_sha256=digest(context),selected=False,status='unavailable',
                        error_type=type(error).__name__)
                write_json(choice,saved)
            result=dict(request)
            if saved.get('selected'):
                try:document=read(root,context)
                except (OSError,ValueError,EOFError):document=None
                if document is not None:result['prepared_scattering']=document
            return result,root

    def call(self,request,directory,*,dispatch_key,on_submitted=None,resume=None,
             cache_dataset_id=None,recover=False,cancel_event=None):
        from ..models import UtilityActionRecord,RunStatus
        if not re.fullmatch('[a-f0-9]{24}',dispatch_key):raise ValueError('A stable live dispatch key is required')
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeUnavailable('Live request is no longer active; existing dispatch receipts are retained')
        directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
        request,prepared_root=self._prepared_request(request,dispatch_key)
        packet=directory/'request-packet.zip';packet_id=create_packet(request,packet,prepared_root=prepared_root)
        scientific_id=scientific_packet_identity(packet)
        ledger=self.root/'dispatches';ledger.mkdir(exist_ok=True)
        acknowledged=ledger/(dispatch_key+'.json');pending=ledger/(dispatch_key+'.pending')
        def original_packet(saved,message):
            if saved.get('scientific_packet_id'):
                if saved['scientific_packet_id']!=scientific_id:raise ValueError(message)
            elif saved['packet_id']!=packet_id:raise ValueError(message)
            return saved['packet_id']
        if pending.exists():
            text=pending.read_text().strip()
            saved=json.loads(text) if text.startswith('{') else dict(packet_id=text)
            packet_id=original_packet(saved,'Pending dispatch key already belongs to different live inputs')
        if acknowledged.exists():
            saved=json.loads(acknowledged.read_text())
            packet_id=original_packet(saved,'Dispatch key already belongs to different live inputs')
            job=UtilityActionRecord.model_validate(saved['job'])
        elif resume:
            job=UtilityActionRecord.model_validate(resume)
        else:
            job=self.find_dispatch(dispatch_key) if recover or pending.exists() else None
            if job is None and pending.exists():
                raise RuntimeUnavailable('Live submission acknowledgement is missing; recover before retrying. No duplicate was submitted.')
            if job is None:
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeUnavailable('Live request stopped before upload; no new job was submitted')
                packet_dataset=self.service.upload_document(packet,label='RADAR-PD live packet '+dispatch_key,wait_ready=False)
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeUnavailable('Live request stopped after upload; no new job was submitted')
                inputs={'dispatch_key':dispatch_key,'packet':{'dataset_id':packet_dataset}}
                if cache_dataset_id:inputs['response_cache']={'dataset_id':cache_dataset_id}
                try:
                    with pending.open('x') as handle:json.dump(dict(packet_id=packet_id,scientific_packet_id=scientific_id),handle)
                except FileExistsError:raise RuntimeUnavailable('Another caller owns this live dispatch; recover it') from None
                try:
                    job=self.service.submit_utility(tool_id=self.tool_id,name='RADAR-PD live snapshot '+dispatch_key,
                        associated_run_uid=request.get('source_revision'),inputs=inputs,cancel_event=cancel_event)
                except Exception as exc:
                    from ..native_job_admission import AdmissionUnavailable
                    if isinstance(exc,AdmissionUnavailable):
                        pending.unlink(missing_ok=True)
                        raise RuntimeUnavailable(str(exc)) from exc
                    raise RuntimeUnavailable('Galaxy did not acknowledge the live submission. The dispatch key is retained for recovery.') from exc
        from gpx_recipe.live.io import atomic_json
        atomic_json(acknowledged,{'packet_id':packet_id,'scientific_packet_id':scientific_id,'job':job.model_dump(mode='json')})
        if on_submitted:on_submitted(job.model_dump(mode='json'))
        start=time.monotonic()
        while job.status not in {RunStatus.OK,RunStatus.ERROR,RunStatus.CANCELLED}:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeUnavailable('Stopped waiting for the live job; its immutable Galaxy dispatch remains recoverable')
            if time.monotonic()-start>self.timeout:raise RuntimeUnavailable('The live job is still queued/running; recover it without resubmitting')
            time.sleep(self.poll_seconds);self.service.refresh_utility(job)
        if job.status==RunStatus.ERROR:raise RuntimeError(job.message or 'The live snapshot worker failed')
        if job.status==RunStatus.CANCELLED:
            raise RuntimeUnavailable('The live snapshot job was cancelled; its immutable dispatch receipt is retained and no result was accepted')
        if not job.outputs.get('result_bundle'):raise RuntimeError('The live snapshot worker has no result bundle')
        archive=directory/'result.zip';self.service._download_dataset(job.outputs['result_bundle'],archive)
        output=extract_result_archive(archive,directory/('result-'+uuid.uuid4().hex))
        transport=json.loads((output/'transport.json').read_text())
        if transport.get('schema')!='radar-live-result/v1' or transport.get('packet_id')!=packet_id:
            raise ValueError('Live worker returned a different packet')
        actual={str(p.relative_to(output)) for p in output.rglob('*') if p.is_file() and p.name!='transport.json'}
        if set(transport.get('files',{}))!=actual:raise ValueError('Live result manifest does not cover all artifacts')
        for name,expected in transport['files'].items():
            path=(output/name).resolve()
            if not path.is_relative_to(output) or not path.is_file() or file_hash(path)!=expected:
                raise ValueError('Live result artifact differs from its receipt')
        result=json.loads((output/'result.json').read_text())
        if result.get('packet_id')!=packet_id:raise ValueError('Live result belongs to a different request')
        for field in ('source_revision','reference_revision'):
            if result.get(field)!=request[field]:raise ValueError('Live result scientific revision mismatch')
        published=[]
        if job.outputs.get('verified_projects'):
            published=self.service.collection_elements(job.outputs['verified_projects'])
        def dataset_for(name):
            matches=[v['id'] for v in published if str(v['name']).removesuffix('.gpx')==name]
            return matches[0] if len(matches)==1 else None
        for row in result['rows']:
            if row.get('published_project'):
                row['fitted_dataset_id']=dataset_for(row['published_project'])
            for field in ('posterior','fitted_gpx'):
                if row.get(field):
                    p=(output/row[field]).resolve()
                    if not p.is_relative_to(output) or not p.is_file():raise ValueError('Live result path escapes verified output')
                    row[field]=str(p)
        if result.get('preparation'):
            result['preparation']=dict(result['preparation'])
            result['preparation']['dataset_id']=dataset_for(result['preparation'].get('published_project'))
            for field in ('request','reference_view','source_gpx','source_dat'):
                p=(output/result['preparation'][field]).resolve()
                if not p.is_relative_to(output) or not p.is_file():raise ValueError('Prepared live source escapes verified result')
                result['preparation'][field]=str(p)
        result.update(directory=str(output),galaxy_job_id=job.galaxy_job_id,
                      response_cache_dataset_id=job.outputs.get('response_cache_output'),transport_sha256=file_hash(output/'transport.json'))
        return result
