"""Resumable model → independent semantic oracle → compiled GSAS-II benchmark."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import collections
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from gpx_recipe.execution import write_json
from gpx_recipe.jobs import scientific_environment
from gpx_recipe.planner import make_plan, build_payload
from gpx_recipe.providers import configure_provider, ProviderRequestError, ProviderUnavailable, make_client, duration, IncompleteResponse

SOURCE=Path(__file__).resolve().parents[2]


class PacedClient:
    """Use reported token availability; do not guess a model-specific quota multiplier."""
    def __init__(self, client, root):
        self.client,self.root=client,root;self.quota=None;self.case=None

    def update(self, metadata):
        headers=metadata.get('rate_limits') or {}
        try:
            limit=float(headers['x-ratelimit-limit-tokens']);remaining=float(headers['x-ratelimit-remaining-tokens'])
            reset=duration(headers.get('x-ratelimit-reset-tokens'))
            refill=(limit-remaining)/reset if reset and limit>remaining else limit/60
            self.quota=(time.monotonic(),limit,remaining,max(1.,refill*.85))
        except (KeyError,TypeError,ValueError):pass

    def complete(self,messages):
        if self.quota:
            stamp,limit,remaining,refill=self.quota
            available=min(limit,remaining+(time.monotonic()-stamp)*refill)
            need=len(json.dumps(messages))/2.2+self.client.max_output_tokens+min(25000,limit*.1)
            delay=max(0,(need-available)/refill)
            if delay:
                write_json(self.root/'model-wait.json',dict(case=self.case,reason='reported token headroom',wait_seconds=delay,until_unix=time.time()+delay))
                time.sleep(delay)
        try:
            raw,metadata=self.client.complete(messages)
        except IncompleteResponse as exc:
            self.update(exc.metadata);raise
        self.update(metadata)
        return raw,metadata


def source_hash():
    paths=list((SOURCE/'nova/src/gpx_recipe').glob('*.py'))+[SOURCE/'nova/src/gpx_recipe/knowledge.json',*Path(__file__).resolve().parent.glob('*.py')]
    digest=hashlib.sha256()
    for p in sorted(paths):digest.update(p.name.encode());digest.update(p.read_bytes())
    return digest.hexdigest()


def compare(plan, oracle, context=None):
    """Titles/IDs are presentation. Active sets, ordering and scientific edits must agree."""
    def targets(stage):
        result=[]
        for t in stage['free']:
            t=dict(t)
            if 'flags' in t:t['flags']=''.join(sorted(t['flags']))
            result.append(json.dumps(t,sort_keys=True))
        return sorted(result)
    def equations(stage):
        return sorted(json.dumps({'terms':sorted(c['terms'],key=lambda t:(t['phase'],t['atom'])),'total':c['total']},sort_keys=True) for c in stage['constraints'])
    def ordered_edits(stage,key):
        return sorted(stage.get(key,[]),key=lambda v:json.dumps(v,sort_keys=True))
    differences=[]
    bg_got={h['key']:(h['background']['function'],h['background']['coefficients']) for h in (context or {}).get('histograms',[])}
    bg_want=dict(bg_got)
    def background_edit(stage,state):
        edit=stage.get('background')
        if not edit:return None
        key=edit['histogram'];old=state.get(key);new=(edit['function'],edit['coefficients']);state[key]=new
        # Reasserting the current type/count preserves coefficients. Removing
        # support points is never treated as a no-op, even if currently empty.
        return None if old==new and edit['fixed_points']=='preserve' else edit
    if plan['questions'] or plan['unsupported']:return ['Executable request was not resolved']
    if len(plan['stages'])!=len(oracle['stages']):return ['Stage count differs from request']
    norm1=set();norm2=set();eq1=set();eq2=set()
    for i,(got,want) in enumerate(zip(plan['stages'],oracle['stages']),1):
        if targets(got)!=targets(want):differences.append(f'Stage {i}: complete active set differs')
        if got['cycles']!=want['cycles']:differences.append(f'Stage {i}: cycle limit differs')
        norm1.update(got.get('phase_normalization',[]));norm2.update(want.get('phase_normalization',[]))
        if norm1!=norm2:differences.append(f'Stage {i}: normalization history differs')
        eq1.update(equations(got));eq2.update(equations(want))
        if eq1!=eq2:differences.append(f'Stage {i}: occupancy equations differ')
        for key in ['background','fit_ranges','exclusions','bounds','texture']:
            a=got.get(key);b=want.get(key)
            if key=='background':a=background_edit(got,bg_got);b=background_edit(want,bg_want)
            else:a=ordered_edits(got,key);b=ordered_edits(want,key)
            if a!=b:differences.append(f'Stage {i}: {key} edits differ')
    return differences


def main():
    parser=argparse.ArgumentParser();parser.add_argument('campaign');parser.add_argument('--mode',choices=['oracle','model'],required=True);parser.add_argument('--workers',type=int,default=2);parser.add_argument('--only',nargs='*');parser.add_argument('--key-file');args=parser.parse_args()
    root=Path(args.campaign).resolve();catalog=json.loads((root/'catalog.json').read_text());revision=source_hash()
    if args.only:catalog=[c for c in catalog if c['id'] in args.only]
    if args.mode=='oracle':catalog=[c for c in catalog if c['expectation']=='execute']
    root.mkdir(parents=True,exist_ok=True)
    statepath=root/(args.mode+'-status.json');lock=__import__('threading').RLock()
    state=json.loads(statepath.read_text()) if statepath.exists() else {'denominator':100,'results':{},'history':[]}
    def update(case,result):
        with lock:
            previous=state['results'].get(case['id'])
            if previous:state['history'].append(previous)
            result.update(case=case['id'],project=case['project'],category=case['category'],source_sha256=revision,
                          case_sha256=hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest())
            state['results'][case['id']]=result;state['counts']=dict(collections.Counter(v['status'] for v in state['results'].values()));state['updated_unix']=time.time();state['current_code']=revision
            write_json(statepath,state)
        print(json.dumps({k:result.get(k) for k in ['case','status','error','elapsed_seconds']}),flush=True)
    def verified(case):
        value=state['results'].get(case['id'],{})
        return value.get('source_sha256')==revision and value.get('case_sha256')==hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest() and value.get('status') in {'verified_execution','scientific_stop','correctly_blocked'}
    def attempt(case):
        work=root/'cases'/case['id']/args.mode/(time.strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:6]);work.mkdir(parents=True)
        return work
    def native(case,plan,work,planning=None):
        write_json(work/'plan.json',plan)
        env=scientific_environment();env['RADAR_GPX_DOCKER_WORKDIR']=str(root);env['RADAR_GPX_DOCKER_IMAGE']='radar-pd-ndip:validated-runtime-fix'
        cmd=[str(SOURCE/'scripts/recipe_docker_python.py'),str(Path(__file__).with_name('native_worker.py')),str(root),case['id'],args.mode,'--workdir',str(work)]
        try:
            with (work/'native.log').open('w') as log:process=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=420)
            result=json.loads((work/'result.json').read_text()) if (work/'result.json').exists() else {'status':'runtime_defect','error':'Native worker exited without a result: '+str(process.returncode)}
        except Exception as e:result={'status':'runtime_defect','error':str(e)}
        result.update(artifact_directory=str(work),planning=planning);update(case,result)
    if args.mode=='oracle':
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(native,c,c['oracle'],attempt(c)) for c in catalog if not verified(c)]
            for f in as_completed(futures):f.result()
        return
    configure_provider('illumine-luna',args.key_file)
    client=PacedClient(make_client(),root)
    next_call=0;futures=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for case in catalog:
            if verified(case):continue
            context=json.loads((root/'inputs'/(case['project']+'.context.json')).read_text());work=attempt(case)
            client.case=case['id'];client.client.max_output_tokens=2048
            for retry in range(5):
                delay=max(0,next_call-time.monotonic())
                if delay:
                    write_json(root/'model-wait.json',dict(case=case['id'],wait_seconds=delay,reason='token-rate pacing',until_unix=time.time()+delay))
                    time.sleep(delay)
                started=time.monotonic()
                try:
                    planning=make_plan(case['request'],context,max_calls=2,client=client);write_json(work/'planning.json',planning)
                    summary={'calls':len(planning['calls']),'reasoning_tokens':[c.get('usage',{}).get('output_tokens_details',{}).get('reasoning_tokens') for c in planning['calls']],'seconds':sum(c['elapsed_seconds'] for c in planning['calls'])}
                    if case['expectation']=='needs_input':
                        if planning['status']=='needs_input':update(case,dict(status='correctly_blocked',planning=summary,artifact_directory=str(work)))
                        else:update(case,dict(status='semantic_mismatch',error='Unresolvable request was accepted',planning=summary,artifact_directory=str(work)))
                    else:
                        errors=compare(planning['plan'],case['oracle'],context)
                        if errors:update(case,dict(status='semantic_mismatch',error='; '.join(errors),planning=summary,artifact_directory=str(work)))
                        else:futures.append(pool.submit(native,case,planning['plan'],work,summary))
                    break
                except ProviderRequestError as e:
                    if e.http_status==429 and retry<4:
                        delay=max(5,e.retry_after_seconds or 60);next_call=time.monotonic()+delay
                        write_json(work/f'rate-limit-{retry}.json',{'http_status':429,'wait_seconds':delay,'reported_retry_seconds':e.retry_after_seconds})
                        print(json.dumps({'case':case['id'],'status':'rate_wait','seconds':delay}),flush=True);continue
                    update(case,dict(status='provider_blocked',error=str(e),artifact_directory=str(work)));break
                except ProviderUnavailable as e:
                    write_json(work/'planning-error.json',{'error':str(e),'calls':getattr(e,'planning_calls',[])})
                    update(case,dict(status='provider_blocked',error=str(e),artifact_directory=str(work)));break
                except Exception as e:
                    calls=getattr(e,'calls',getattr(e,'planning_calls',[]))
                    write_json(work/'planning-error.json',{'error':str(e),'calls':calls})
                    update(case,dict(status='planner_validation_failure',error=str(e),artifact_directory=str(work)));break
        for f in as_completed(futures):f.result()


if __name__=='__main__':main()
