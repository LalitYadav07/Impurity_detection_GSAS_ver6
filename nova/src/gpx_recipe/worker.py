"""Credential-free native worker shared by local NOVA and Galaxy transports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import traceback
import zipfile

from .execution import write_json, run_plan
from .snapshot import snapshot
from .artifacts import write_project_view


def work(action, source, directory, *, plan=None, approval=None, expected_source_sha256=None):
    directory=Path(directory).resolve(); directory.mkdir(parents=True,exist_ok=True)
    source=Path(source).resolve(); before=hashlib.sha256(source.read_bytes()).hexdigest()
    response=dict(schema='radar-refinement-worker/v1',action=action,source_sha256=before,status='running')
    try:
        if expected_source_sha256 and before!=expected_source_sha256:raise ValueError('The supplied dataset is not the approved source GPX revision')
        if action=='inspect':
            response['context']=snapshot(source)
            write_project_view(source,directory/'baseline.view.json')
            response['view']='baseline.view.json';response['status']='inspected'
        elif action=='preview':
            response['preview']=run_plan(source,directory/'preview.gpx',plan,preview=True)
            response['status']='preview_passed'
        elif action=='run':
            from .compiler import compile_script
            script=directory/'refine.py';code=compile_script(plan,approval,script)
            # Execute the compiled wrapper, not a separate direct-engine path.
            import sys
            previous=sys.argv
            try:
                sys.argv=[str(script),'--source-gpx',str(source),'--output-gpx',str(directory/'polished.gpx')]
                exec(compile(code,str(script),'exec'),{'__name__':'__main__'})
            finally:sys.argv=previous
            receipt=json.loads((directory/'polished.execution.json').read_text())
            if receipt['status']!='completed_needs_scientific_review':
                raise ValueError('The worker did not complete every approved stage')
            write_project_view(source,directory/'baseline.view.json')
            stages=[]; last_refined={}
            for stage in plan['stages']:
                project=directory/('polished_'+stage['id']+'.gpx')
                target=directory/('verified_'+stage['id']+'.gpx');shutil.copy2(project,target)
                view=directory/(stage['id']+'.view.json');stage_view=write_project_view(project,view)
                for parameter in stage_view.get('parameters',[]):
                    if parameter['status']=='refined':last_refined[parameter['key']]={**parameter,'refined_stage':stage['id']}
                stages.append(dict(id=stage['id'],title=stage['title'],gpx=target.name,view=view.name))
            shutil.copy2(directory/'polished.gpx',directory/'verified_final.gpx')
            final_view=write_project_view(directory/'polished.gpx',directory/'final.view.json')
            for parameter in final_view.get('parameters',[]):
                prior=last_refined.get(parameter['key'])
                if prior and abs(parameter['value']-prior['value'])<=1e-10*max(1,abs(parameter['value'])):
                    parameter.update(sigma=prior.get('sigma'),refined_stage=prior['refined_stage'],status='refined' if parameter['status']=='refined' else 'refined_in_prior_stage')
            write_json(directory/'final.view.json',final_view)
            response.update(status='completed',execution='polished.execution.json',baseline_view='baseline.view.json',
                final_view='final.view.json',final_gpx='verified_final.gpx',stages=stages,
                output_sha256=hashlib.sha256((directory/'verified_final.gpx').read_bytes()).hexdigest())
        else:raise ValueError('Unknown refinement worker action')
    except Exception as exc:
        response.update(status='needs_review' if isinstance(exc,ValueError) and action=='run' else 'failed',
                        error=str(exc),error_type=type(exc).__name__)
        (directory/'failure.log').write_text(traceback.format_exc())
        # A partially refined project is retained as evidence, never published as a final result.
        partial=directory/'polished.gpx'
        if partial.exists():partial.replace(directory/'stopped.gpx')
        receipt=directory/'polished.execution.json'
        if receipt.exists():
            response['execution']=receipt.name
            saved=json.loads(receipt.read_text());stages=[]
            for stage in saved.get('stages',[]):
                project=directory/('polished_'+stage['id']+'.gpx')
                if stage.get('status')!='completed' or not project.exists():continue
                target=directory/('verified_'+stage['id']+'.gpx');shutil.copy2(project,target)
                view=directory/(stage['id']+'.view.json')
                try:
                    write_project_view(project,view)
                    stages.append(dict(id=stage['id'],title=stage['title'],gpx=target.name,view=view.name))
                except Exception as export_error:
                    response.setdefault('artifact_errors',[]).append(dict(stage=stage['id'],error=str(export_error)))
            response['stages']=stages
        try:
            write_project_view(source,directory/'baseline.view.json');response['baseline_view']='baseline.view.json'
        except Exception:pass
    if hashlib.sha256(source.read_bytes()).hexdigest()!=before:
        response.update(status='failed',error='The original RADAR-PD GPX changed during processing')
    response['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()
        if p.is_file() and (p.suffix in {'.json','.py'} or p.name.startswith('verified_')) and p.name!='response.json'}
    write_json(directory/'response.json',response)
    return response


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['inspect','preview','run']);p.add_argument('source');p.add_argument('directory');p.add_argument('--plan');p.add_argument('--approval');p.add_argument('--archive');args=p.parse_args()
    load=lambda value:json.loads(Path(value).read_text()) if value else None
    response=work(args.action,args.source,args.directory,plan=load(args.plan),approval=load(args.approval))
    if args.archive:
        dest=Path(args.archive).resolve();root=Path(args.directory).resolve()
        with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
            for path in sorted(root.iterdir()):
                if path.is_file() and path.resolve()!=dest:z.write(path,path.name)
    print(json.dumps({k:response[k] for k in ['action','status','error'] if k in response}))


if __name__=='__main__':main()
