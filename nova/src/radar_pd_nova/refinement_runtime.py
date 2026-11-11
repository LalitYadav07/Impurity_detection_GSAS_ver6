"""Native refinement transport; NOVA never loads GSAS-II or unpickles GPXs."""
from __future__ import annotations
from .tool_ids import tool_id

import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Callable

from gpx_recipe.jobs import scientific_environment
from gpx_recipe.process import run_scientific
from gpx_recipe.execution import write_json
from .refinement_sources import file_sha
from .refinement_messages import retain_review_evidence, final_stage_criterion

REFINEMENT_TOOL_ID = tool_id('refine_recipe')
WORKSPACE_LABEL = 'RADAR-PD Refinement workspace'


def portable_refinement_workspace(manifest):
    """Reference repeated native diagnostic tables in their verified artifact.

    The full execution receipt remains in the immutable result bundle. Retain
    the inline tables for older outputs that do not have that recoverable file.
    Do not mutate live job state or any scientific result receipt.
    """
    document=dict(manifest)
    if 'jobs' not in manifest:return document
    document['jobs']=dict(manifest.get('jobs') or {})
    for uid,job in document['jobs'].items():
        result=job.get('result') or {}
        execution=result.get('execution');sha=(result.get('files') or {}).get(execution)
        if not isinstance(execution,str) or not isinstance(sha,str) or len(sha)!=64:continue
        rows=result.get('stage_outcomes')
        if not isinstance(rows,list):continue
        compact=[]
        for row in rows:
            if not isinstance(row,dict) or 'metric_rows' not in row:
                compact.append(row);continue
            summary={k:v for k,v in row.items() if k!='metric_rows'}
            summary['metric_rows_source']=dict(artifact=execution,sha256=sha,stage_id=row['id'])
            compact.append(summary)
        document['jobs'][uid]={**job,'result':{**result,'stage_outcomes':compact}}
    return document


def retain_result_review(directory,result,plan=None):
    """Called only after artifact verification, once when results are collected."""
    name=result.get('final_view')
    if result.get('status')!='completed' or not name:return
    root=Path(directory).resolve();path=(root/name).resolve()
    if not path.is_relative_to(root) or name not in result.get('files',{}):return
    criteria=final_stage_criterion(result,plan)
    try:view=json.loads(path.read_text())
    except (OSError,ValueError):return
    retain_review_evidence(result,view,criteria)


def verify_result_files(directory,result, *, action=None):
    root=Path(directory).resolve()
    if action is not None:
        statuses={'prepare':{'model_prepared','failed'},'inspect':{'inspected','failed'},'edit':{'model_edited','failed'},
                  'preview':{'preview_passed','failed'},'run':{'completed','needs_review','failed'}}
        if (result.get('schema')!='radar-refinement-worker/v1' or result.get('action')!=action
                or result.get('status') not in statuses.get(action,set())):
            raise ValueError('Scientific worker response does not match the requested operation')
    manifest=result.get('files',{})
    if not isinstance(manifest,dict):raise ValueError('Invalid refinement artifact manifest')
    for name,expected in manifest.items():
        if not isinstance(name,str) or not isinstance(expected,str):raise ValueError('Invalid refinement artifact manifest')
        path=(root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or file_sha(path)!=expected:
            raise ValueError('A saved refinement artifact changed or is missing: '+Path(name).name)
    if result.get('status') in {'completed','model_edited','model_prepared'}:
        final=(root/str(result.get('final_gpx') or '')).resolve()
        if not final.is_relative_to(root) or not final.is_file() or file_sha(final)!=result.get('output_sha256'):
            raise ValueError('The checked final GPX changed or is unavailable')
    if result.get('status')=='completed' and any(not result.get(k) for k in ('final_view','baseline_view','execution','stages')):
        raise ValueError('The completed refinement is missing its scientific review artifacts')
    if result.get('status') not in {'completed','model_edited','model_prepared'} and any(result.get(k) for k in ('final_gpx','final_view','output_sha256','trend_views')):
        raise ValueError('An incomplete refinement cannot publish a final result')
    references=[result[k] for k in ('view','baseline_view','final_view','execution') if result.get(k)]
    stages=result.get('stages') or []
    if not isinstance(stages,list) or any(not isinstance(s,dict) for s in stages):
        raise ValueError('Invalid verified stage list')
    ids=[s.get('id') for s in stages]
    if any(not isinstance(i,str) or not i for i in ids) or len(set(ids))!=len(ids):
        raise ValueError('Invalid verified stage identities')
    for stage in stages:
        if not stage.get('gpx') or not stage.get('view'):raise ValueError('A verified stage is missing its GPX or review view')
        references.extend((stage['gpx'],stage['view']))
    for name in references:
        if not isinstance(name,str) or name not in manifest:
            raise ValueError('A referenced scientific artifact is absent from the verified manifest')


class RuntimeUnavailable(RuntimeError):
    pass


class LocalRefinementRuntime:
    kind='local'

    def __init__(self, root, *, python, gsasii_root=None, timeout=900):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.python=str(python);self.gsasii_root=gsasii_root;self.timeout=timeout

    def resolve(self, source):
        source=dict(source);path=Path(source.get('source_path') or '')
        if not path.is_file():raise ValueError('The published source GPX is not downloaded yet')
        checksum=file_sha(path)
        if source.get('source_sha256') and checksum!=source['source_sha256']:
            raise ValueError('The queued original GPX changed; reload the RADAR-PD result')
        target=self.root/'sources'/(checksum+'.gpx');target.parent.mkdir(exist_ok=True)
        if target.exists():
            if file_sha(target)!=checksum:raise ValueError('An immutable source copy was modified')
        else:
            import uuid
            temporary=target.with_name(target.name+'.'+uuid.uuid4().hex+'.partial');shutil.copy2(path,temporary)
            if file_sha(temporary)!=checksum:raise ValueError('Source GPX changed while it was being captured')
            temporary.replace(target)
        source.update(source_path=str(target),source_sha256=checksum)
        return source

    def call(self, action, source, directory, *, plan=None, approval=None, dispatch_key='',
             on_submitted=None, resume=None, recover=False, cancel_event=None, on_progress=None):
        # A previously registered battery source can outlive a converter fix.
        # Retain inspection/recovery of historical artifacts, but recheck the
        # captured legacy coefficients before any new preview or execution.
        metadata=source.get('metadata') or {}
        preparation = metadata.get('preparation') or metadata.get('parent_preparation')
        if action in {'preview', 'run'} and preparation:
            from .battery_preparation import verify_prepared, check_prepared_profiles
            job = Path(preparation).parent.parent
            check_prepared_profiles(job, verify_prepared(job))
        source=self.resolve(source);directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
        command=[self.python,'-m','gpx_recipe.worker',action,source['source_path'],str(directory)]
        for name,value in [('plan',plan),('approval',approval)]:
            if value is not None:
                path=directory/(name+'.json');path.write_text(json.dumps(value,allow_nan=False)+'\n');command.extend(['--'+name,str(path)])
        environment=scientific_environment(self.gsasii_root)
        with (directory/'native.log').open('w') as log:
            process=run_scientific(command,cwd=directory,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=self.timeout)
        response=directory/'response.json'
        if process.returncode or not response.exists():
            raise RuntimeError('Native refinement worker stopped before its result was published; inspect its captured log')
        result=json.loads(response.read_text());verify_result_files(directory,result,action=action);result['directory']=str(directory)
        if result.get('source_sha256')!=source['source_sha256']:
            raise ValueError('Scientific worker processed a different source revision')
        retain_result_review(directory,result,plan)
        return result

    def save_workspace(self, manifest):
        return None

    def recover_workspace(self):
        return None


class GalaxyRefinementRuntime(LocalRefinementRuntime):
    kind='galaxy'
    tool_id=REFINEMENT_TOOL_ID
    workspace_label=WORKSPACE_LABEL
    workspace_schema='radar-refinement-workspace/v1'

    def __init__(self, root, service, *, timeout=7200,poll_seconds=3):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.service=service;self.timeout=timeout;self.poll_seconds=poll_seconds
        self._collections={}

    def resolve(self,source):
        source=dict(source)
        path=Path(source.get('source_path') or '')
        if not path.is_file():
            dataset=self.dataset_id(source)
            target=self.root/'downloads'/(dataset+'.gpx');target.parent.mkdir(exist_ok=True)
            if not target.exists():
                import uuid
                temporary=target.with_name(target.name+'.'+uuid.uuid4().hex+'.partial')
                try:
                    self.service._download_dataset(dataset,temporary)
                    if source.get('source_sha256') and file_sha(temporary)!=source['source_sha256']:
                        raise ValueError('The downloaded GPX differs from its reviewed publication')
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
            source['source_path']=str(target)
        return super().resolve(source)

    def dataset_id(self,source):
        if source.get('dataset_id'):return source['dataset_id']
        collection=source.get('collection_id')
        if not collection:raise ValueError('This source has no Galaxy GPX collection reference')
        if collection not in self._collections:self._collections[collection]=self.service.collection_elements(collection)
        name=source.get('collection_name')
        matches=[r for r in self._collections[collection] if r['name']==name]
        if not matches and isinstance(name,str) and name.lower().endswith('.gpx'):
            # Older RADAR-PD indexes include the filename suffix, while Galaxy's
            # collection element identifier omits it. Only accept one such match.
            matches=[r for r in self._collections[collection] if r['name']==name[:-4]]
        if len(matches)!=1:raise ValueError('The published GPX does not map to one exact Galaxy collection element')
        return matches[0]['id']

    def call(self,action,source,directory,*,plan=None,approval=None,dispatch_key='',on_submitted=None,resume=None,recover=False,cancel_event=None,on_progress=None):
        from .models import UtilityActionRecord,RunStatus
        from .galaxy_service import _extract_results_archive
        source=self.resolve(source);directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        if resume:
            job=UtilityActionRecord.model_validate(resume)
        else:
            # A unique dispatch key survives lost HTTP acknowledgements/restarts.
            # Never blindly retry an ambiguous submission.
            import re
            if not re.fullmatch(r'[a-f0-9]{24}',dispatch_key):raise ValueError('A stable refinement dispatch key is required')
            ledger=self.root/'dispatches';ledger.mkdir(exist_ok=True)
            acknowledged=ledger/(dispatch_key+'.json');pending=ledger/(dispatch_key+'.pending')
            if acknowledged.exists():
                existing=UtilityActionRecord.model_validate(json.loads(acknowledged.read_text()))
            else:
                existing=self.find_dispatch(dispatch_key) if recover or pending.exists() else None
            if existing is None and pending.exists():
                raise RuntimeUnavailable('This dispatch has an unacknowledged submission. No matching Galaxy job is visible yet; refresh or inspect Galaxy before retrying. No duplicate was submitted.')
            if existing:job=existing
            else:
                request=dict(plan=plan,approval=approval,expected_source_sha256=source['source_sha256'],dispatch_key=dispatch_key)
                capability=getattr(self.service,'tool_has_input',None)
                inline=bool(capability and capability(self.tool_id,'recipe_request_b64'))
                inputs={'action':action,'dispatch_key':dispatch_key}
                if inline:
                    import base64
                    # Match the legacy document bytes, including its final LF.
                    # This changes only transport, not the reviewed request.
                    raw=(json.dumps(request,indent=2,sort_keys=True,allow_nan=False)+'\n').encode('utf-8')
                    if len(raw)>4*1024*1024:raise ValueError('Refinement request exceeds the 4 MiB transport bound')
                    inputs['recipe_request_b64']=base64.b64encode(raw).decode('ascii')
                else:
                    # Older wrappers keep the request as a Galaxy dependency.
                    # Durable workspace claims/checkpoints remain separate.
                    request_id=self.service.upload_json_document(request,name='refinement-request.json',label='RADAR-PD refinement request '+dispatch_key,wait_ready=False)
                    inputs['recipe_request']={'dataset_id':request_id}
                source_dataset=self.dataset_id(source)
                inputs['source_gpx']={'dataset_id':source_dataset}
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeUnavailable('Scientific submission paused before dispatch')
                try:
                    with pending.open('x') as handle:handle.write(action+'\n')
                except FileExistsError:
                    raise RuntimeUnavailable('Another worker already claimed this dispatch; recover its existing Galaxy job')
                try:
                    job=self.service.submit_utility(tool_id=REFINEMENT_TOOL_ID,name='RADAR-PD refinement '+dispatch_key,
                        associated_run_uid=source.get('record_uid'),cancel_event=cancel_event,inputs=inputs)
                except Exception as exc:
                    from .native_job_admission import AdmissionUnavailable
                    if isinstance(exc,AdmissionUnavailable):
                        pending.unlink(missing_ok=True)
                        raise RuntimeUnavailable(str(exc)) from exc
                    raise RuntimeUnavailable('Galaxy did not acknowledge this refinement submission. Its dispatch key is retained; recover the existing job before retrying.') from exc
            # The workspace already retains the exact plan, approval and source.
            # Do not duplicate the encoded request in every checkpoint/receipt.
            record=job.model_dump(mode='json',exclude={'inputs':{'recipe_request_b64'}})
            write_json(acknowledged,record)
            if on_submitted:on_submitted(record)
        started=time.monotonic()
        reported_status=None
        def report_progress():
            nonlocal reported_status
            if on_progress and job.status!=reported_status:
                on_progress(job.model_dump(mode='json',exclude={'inputs':{'recipe_request_b64'}}))
                reported_status=job.status
        report_progress()
        while job.status not in {RunStatus.OK,RunStatus.ERROR,RunStatus.CANCELLED}:
            if time.monotonic()-started>self.timeout:raise RuntimeUnavailable('Galaxy refinement is still pending; recover the existing job before retrying')
            time.sleep(self.poll_seconds);self.service.refresh_utility(job)
            report_progress()
        if job.status==RunStatus.ERROR:raise RuntimeError(job.message or 'The Galaxy refinement worker failed')
        if job.status==RunStatus.CANCELLED:raise RuntimeUnavailable('The Galaxy refinement job was cancelled; it is not still queued')
        bundle=job.outputs.get('result_bundle')
        if not bundle:raise RuntimeError('Galaxy refinement completed without its result bundle')
        archive=directory/'result.zip';self.service._download_dataset(bundle,archive)
        _extract_results_archive(archive,directory)
        result=json.loads((directory/'response.json').read_text());verify_result_files(directory,result,action=action)
        if result.get('source_sha256')!=source['source_sha256']:
            raise ValueError('Galaxy refinement result belongs to another source GPX revision')
        result['directory']=str(directory);result['galaxy_job_id']=job.galaxy_job_id
        collection=job.outputs.get('stage_projects')
        if collection:
            elements=self.service.collection_elements(collection)
            stages={v['id'] for v in result.get('stages') or []}
            result['stage_dataset_ids']={x['name']:x['id'] for x in elements if x['name'] in stages}
            final=[x for x in elements if x['name']=='final']
            if len(final)==1:result['final_dataset_id']=final[0]['id']
            model=[x for x in elements if x['name']=='model']
            if len(model)==1:result['model_dataset_id']=model[0]['id']
        retain_result_review(directory,result,plan)
        return result

    def find_dispatch(self,key):
        """Recover a matching prior utility without starting a duplicate job."""
        if not key:return None
        import requests
        from .galaxy_service import _parameter_value, normalize_status
        from .models import UtilityActionRecord
        response=requests.get(self.service.galaxy_url+'/api/jobs',headers=self.service._headers,
            params={'history_id':self.service.history_id,'tool_id':self.tool_id,'limit':1000},timeout=30)
        response.raise_for_status();jobs=response.json()
        if not isinstance(jobs,list):raise RuntimeUnavailable('Could not verify existing refinement submissions')
        matches=[]
        for row in jobs:
            if row.get('tool_id')!=self.tool_id:continue
            details=self.service._job_details(str(row['id']))
            if str(_parameter_value(details.get('params') or details.get('parameters') or {},'dispatch_key') or '')==key:
                matches.append(details)
        if len(matches)>1:raise RuntimeUnavailable('Several Galaxy jobs share this dispatch key; review them before retrying')
        if not matches:
            if len(jobs)>=1000:raise RuntimeUnavailable('Galaxy returned a truncated job history; verify the previous dispatch before retrying')
            return None
        row=matches[0]
        return UtilityActionRecord(uid='refinement-'+key,tool_id=self.tool_id,name='Recovered refinement '+key,
            galaxy_job_id=str(row['id']),status=normalize_status(row.get('state')),outputs=self.service._job_output_ids(row))

    def save_workspace(self,manifest):
        import gzip,uuid
        manifest=portable_refinement_workspace(manifest)
        path=self.root/('workspace-'+uuid.uuid4().hex+'.json.gz')
        payload=json.dumps(manifest,allow_nan=False,separators=(',',':'),sort_keys=True).encode()
        path.write_bytes(gzip.compress(payload,mtime=0))
        try:
            bounded=getattr(self.service,'upload_checkpoint_document',None)
            if bounded:
                stable=dict(manifest);stable.pop('updated_utc',None)
                fingerprint=hashlib.sha256(json.dumps(stable,allow_nan=False,separators=(',',':'),sort_keys=True).encode()).hexdigest()
                return bounded(path,label=self.workspace_label,semantic_sha256=fingerprint)
            return self.service.upload_document(path,label=self.workspace_label)
        finally:path.unlink(missing_ok=True)

    def recover_workspace(self):
        rows=self.service.search_history_datasets(query=self.workspace_label,limit=5,include_generated=True)
        for row in rows:
            if row.get('state')!='ok':continue
            # A real six-scan workspace already exceeds the generic service's
            # 1 MB text preview limit. Restore the owned dataset directly.
            import gzip,uuid
            path=self.root/('recovery-'+uuid.uuid4().hex)
            self.service._download_dataset(row['id'],path)
            try:
                with path.open('rb') as handle:compressed=handle.read(2)==b'\x1f\x8b'
                opener=gzip.open if compressed else open
                with opener(path,'rb') as handle:payload=handle.read(128*1024*1024+1)
                if len(payload)>128*1024*1024:raise RuntimeUnavailable('The saved workspace exceeds the 128 MB recovery bound; preserve it for an explicit scoped restore')
                document=json.loads(payload)
            finally:path.unlink(missing_ok=True)
            if document.get('schema')==self.workspace_schema and document.get('history_id')==self.service.history_id:
                return document
        return None


def configured_runtime(root,service):
    python=os.getenv('RADAR_PD_REFINEMENT_PYTHON','').strip()
    if python:
        return LocalRefinementRuntime(root,python=python,gsasii_root=os.getenv('GSASII_ROOT'))
    if getattr(service,'galaxy_url',None) and getattr(service,'galaxy_key',None) and getattr(service,'history_id',None):
        return GalaxyRefinementRuntime(root,service)
    return None
