"""One fresh native process per sealed symmetry-search trial."""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from pathlib import Path
import resource
import sys
import time
import traceback

from ..snapshot import gsas,json_safe
from ..execution import write_json
from .contracts import digest,trial_id,validate_protocol
from .compiler import confined,file_sha,package_signature,verify_inputs
from .runtime import native_stack,VALIDATED_STACKS


class TrialDisposition(ValueError):
    def __init__(self,status,message):super().__init__(message);self.status=status


VALIDATED_MAGNETIC_FUNCTIONS={'82076dc0571cad6d6e2882c5f7ced9003f08af94449aa293b0fb4c5c2ae7b9f6'}


def require_runtime(runtime,declared):
    actual=runtime['magnetic_function_sha256']
    if actual not in VALIDATED_MAGNETIC_FUNCTIONS or actual!=declared or runtime.get('native_stack_sha256') not in VALIDATED_STACKS:
        raise TrialDisposition('runtime_incompatible','This native runtime has not passed the declared representation controls')


class Artifacts:
    def __init__(self,directory):self.run_dir=Path(directory).resolve();self.metrics={};self.files=[]
    def register_file(self,*,source_path,relative_path,**metadata):
        path=Path(source_path).resolve()
        if not path.is_relative_to(self.run_dir):raise ValueError('Artifact escaped the trial directory')
        self.files.append(dict(path=path.name,sha256=file_sha(path),**metadata))
    def save_json(self,name,data,**metadata):write_json(confined(self.run_dir,name,exists=False),data)
    def log_metric(self,name,value):self.metrics[name]=value
    def finish(self,message):self.save_json('artifacts.json',dict(files=self.files,metrics=self.metrics,message=message))


def runtime_identity():
    G=gsas();module=importlib.import_module((G.__package__+'.' if G.__package__ else '')+'GSASIIstrMath')
    version=importlib.import_module((G.__package__+'.' if G.__package__ else '')+'GSASIIpath')
    return dict(version=str(version.GetVersionNumber()),magnetic_function_sha256=hashlib.sha256(inspect.getsource(module.MagStructureFactor2).encode()).hexdigest(),**native_stack())


def execute(directory):
    directory=Path(directory).resolve();started=time.monotonic()
    target=directory/'result.json'
    if target.exists():raise ValueError('A completed trial cannot be silently rerun')
    output=directory/'results';output.mkdir(exist_ok=False)
    response=dict(schema='radar-symmetry-trial/v1',status='technical_failure',llm_calls=0)
    protocol=None
    try:
        job=json.loads((directory/'job.json').read_text());protocol=validate_protocol(job['protocol'])
        response.update(id=job['id'],protocol_sha256=digest(protocol),candidate_id=job['candidate_id'],seed_id=job['seed_id'])
        if job['schema']!='radar-symmetry-job/v1' or job['protocol_sha256']!=digest(protocol) or job['id']!=trial_id(protocol,job['candidate_id'],job['seed_id']):
            raise TrialDisposition('integrity_failure','Job identity or approved protocol changed')
        if job['package_sha256']!=protocol['implementation_sha256'] or job['package_sha256']!=package_signature():raise TrialDisposition('integrity_failure','The approved scientific implementation changed')
        verify_inputs(protocol,directory)
        runtime=runtime_identity();response['runtime']=runtime
        require_runtime(runtime,protocol['magnetic_function_sha256'])
        candidate=next(c for c in protocol['candidates'] if c['id']==job['candidate_id'])
        seed=next(s for s in protocol['seeds'] if s['id']==job['seed_id'])
        if 'candidate_ids' in seed and candidate['id'] not in seed['candidate_ids']:raise TrialDisposition('integrity_failure','Start is outside this candidate’s approved finite set')
        source=confined(directory,protocol['source']['path']);ctx=Artifacts(output)
        if protocol['kind']=='magnetic':
            from .native_magnetic import run
        else:
            from .native_structural import run
        result=run(source,ctx,protocol,candidate,seed)
        response.update(result)
        from .native_audit import audit_trial
        response['audit']=audit_trial(source,output/result['gpx'],protocol,candidate)
        response['native_reported_rwp']=response.get('rwp');response['rwp']=response['audit']['rwp_recomputed']
        registered=output/'artifacts.json'
        if registered.exists():
            saved=json.loads(registered.read_text())
            for item in saved.get('files',[]):item['sha256']=file_sha(confined(output,item['path']))
            write_json(registered,saved)
        response['inputs_verified']=True;response['effects_verified']=True
        if response['status']=='valid' and response.get('rwp',101)>protocol['policy']['adequate_rwp_percent']:
            response['status']='inadequate_fit'
            response['reason']='Native parameter checks passed, but the declared experiment-specific fit adequacy criterion did not'
    except Exception as exc:
        response.update(status=exc.status if isinstance(exc,TrialDisposition) else 'technical_failure',reason=str(exc),error_type=type(exc).__name__)
        (output/'failure.log').write_text(traceback.format_exc())
    finally:
        if protocol is not None:
            try:verify_inputs(protocol,directory)
            except Exception as exc:response.update(status='integrity_failure',reason=str(exc),inputs_verified=False)
        usage=resource.getrusage(resource.RUSAGE_SELF)
        response['resources']=dict(wall_seconds=time.monotonic()-started,cpu_seconds=usage.ru_utime+usage.ru_stime,
                                  peak_rss_bytes=int(usage.ru_maxrss*(1024 if sys.platform.startswith('linux') else 1)))
        response['files']={str(p.relative_to(directory)):file_sha(p) for p in output.rglob('*') if p.is_file() and p.suffix in {'.gpx','.json','.log'}}
        write_json(target,response)
    return response


def main(directory=None):
    response=execute(directory or sys.argv[1]);print(json.dumps({k:response[k] for k in ['id','status','reason'] if k in response}))


if __name__=='__main__':main()
