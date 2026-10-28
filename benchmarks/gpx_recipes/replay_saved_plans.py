"""Re-execute genuine, semantically accepted saved model plans without new inference.

This is an engine regression, never a new estimate of planner success. Source
clarifications and prior errors remain distinct and are never executed.
"""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import uuid

from gpx_recipe.contracts import validate_plan,plan_hash
from gpx_recipe.execution import write_json
from gpx_recipe.intent import check_literal_free
from gpx_recipe.planner import build_payload
from gpx_recipe.jobs import scientific_environment
from gpx_recipe.process import run_scientific
from run_campaign import SOURCE,source_hash,compare


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('target');p.add_argument('source_round');p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    root=Path(a.target).resolve();old=Path(a.source_round).resolve();statepath=root/'model-status.json'
    if statepath.exists():raise ValueError('Use a fresh replay round; previous outcomes must remain immutable')
    source_state=json.loads((old/'model-status.json').read_text());source_cases={c['id']:c for c in json.loads((old/'catalog.json').read_text())};cases=json.loads((root/'catalog.json').read_text())
    deadline=datetime.fromisoformat(json.loads((root/'run-window.json').read_text())['deadline_utc']).timestamp();revision=source_hash('oracle')
    image=subprocess.check_output(['docker','image','inspect','radar-pd-ndip:validated-runtime-fix','--format','{{.Id}}'],text=True).strip()
    write_json(root/'execution-environment.json',{'image_id':image,'requested_image':'radar-pd-ndip:validated-runtime-fix'})
    state={'mode':'saved_model_plan_replay','denominator':len(cases),'selected_cases':[c['id'] for c in cases],'current_code':revision,'image_id':image,'model_calls_this_round':0,'source_round':str(old),'source_model_code':source_state['current_code'],'results':{},'history':[]}
    lock=threading.Lock()
    def update(case,result):
        with lock:
            result.update(case=case['id'],project=case['project'],category=case['category'],source_sha256=revision,image_id=image,case_sha256=hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest())
            state['results'][case['id']]=result;state['counts']=dict(collections.Counter(r['status'] for r in state['results'].values()));state['updated_unix']=time.time();write_json(statepath,state)
        print(json.dumps({'case':case['id'],'status':result['status'],'error':result.get('error')}),flush=True)
    def native(case,plan,work):
        if time.time()>=deadline:update(case,{'status':'deferred_deadline','artifact_directory':str(work)});return
        env=scientific_environment();env.update(RADAR_GPX_DOCKER_WORKDIR=str(root),RADAR_GPX_DOCKER_IMAGE=image)
        command=[str(SOURCE/'scripts/recipe_docker_python.py'),str(Path(__file__).with_name('native_worker.py')),str(root),case['id'],'model','--workdir',str(work)]
        try:
            with (work/'native.log').open('w') as log:r=run_scientific(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=min(900,max(420,45*len(plan['stages']))))
            result=json.loads((work/'result.json').read_text()) if (work/'result.json').exists() else {'status':'runtime_defect','error':'Worker exited without result '+str(r.returncode)}
        except subprocess.TimeoutExpired:result={'status':'execution_timeout','error':'Owned scientific process group terminated at timeout'}
        except Exception as e:result={'status':'runtime_defect','error':str(e)}
        result.update(artifact_directory=str(work),planning={'calls':0,'source':'Saved actual Luna proposal, no new model call'})
        update(case,result)
    futures=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for case in cases:
            original=source_cases.get(case['id'])
            if original!=case:raise ValueError('Replay case differs from source round: '+case['id'])
            project=case['project'];assert sha(root/'inputs'/(project+'.gpx'))==sha(old/'inputs'/(project+'.gpx')),'Source observations differ'
            context=json.loads((root/'inputs'/(project+'.context.json')).read_text());prior=source_state.get('results',{}).get(case['id'],{})
            if prior.get('status')=='correctly_blocked':update(case,{'status':'source_clarification_not_reexecuted','source_outcome':prior['status']});continue
            if prior.get('status') not in {'verified_execution','scientific_stop','runtime_defect'}:
                update(case,{'status':'prior_error_not_replayed','source_outcome':prior.get('status','pending'),'error':prior.get('error')});continue
            source_work=Path(prior['artifact_directory']);planning=json.loads((source_work/'planning.json').read_text());plan=planning['plan']
            proof=json.loads((source_work/'semantic-plan-accepted.json').read_text());assert proof['plan_sha256']==plan_hash(plan),'Saved semantic proof differs'
            assert proof['source_sha256']==source_state['current_code']==prior['source_sha256'],'Source model version differs'
            assert proof['case_sha256']==hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest(),'Source semantic proof is for another request'
            assert planning['calls'] and all(c.get('model')=='gpt-6-luna' and c.get('reasoning_effort')=='none' for c in planning['calls']),'Expected a genuine Luna proposal'
            validate_plan(plan,context);errors=compare(plan,case['oracle'],context)+check_literal_free(plan,build_payload(case['request'],context)['stage_instruction_cards'])
            if errors:update(case,{'status':'contract_mismatch_not_replayed','error':str(errors)});continue
            work=root/'cases'/case['id']/'model'/('replay-'+uuid.uuid4().hex[:8]);work.mkdir(parents=True)
            write_json(work/'plan.json',plan);shutil.copy2(source_work/'planning.json',work/'source-planning.json')
            write_json(work/'replay-provenance.json',{'source_artifact':str(source_work),'planning_sha256':sha(source_work/'planning.json'),'source_model_code':source_state['current_code'],'target_native_code':revision,'model_calls':0,'plan_sha256':plan_hash(plan)})
            futures.append(pool.submit(native,case,plan,work))
        for f in as_completed(futures):f.result()
    assert len(state['results'])==len(cases)


if __name__=='__main__':main()
