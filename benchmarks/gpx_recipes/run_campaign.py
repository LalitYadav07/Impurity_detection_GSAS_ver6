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
from gpx_recipe.contracts import plan_hash,validate_plan
from gpx_recipe.jobs import scientific_environment
from gpx_recipe.process import run_scientific
from gpx_recipe.planner import make_plan, build_payload
from gpx_recipe.providers import configure_provider, ProviderRequestError, ProviderUnavailable, make_client, duration, IncompleteResponse

SOURCE=Path(__file__).resolve().parents[2]


class CampaignDeadline(RuntimeError):pass


class ServiceCircuit:
    """An authorization/service failure is not a new failure of every recipe."""
    def __init__(self):self.halt=None;self.transport_failures=0
    def success(self):self.transport_failures=0
    def fail(self,error,case):
        if isinstance(error,ProviderRequestError):
            self.halt={'case':case,'http_status':error.http_status,'error_code':error.error_code,'reason':str(error)}
        elif isinstance(error,ProviderUnavailable):
            self.transport_failures+=1
            if self.transport_failures>=3:self.halt={'case':case,'reason':'Three consecutive transport failures; '+str(error)}
        return self.halt


class PacedClient:
    """Use reported token availability; do not guess a model-specific quota multiplier."""
    def __init__(self, client, root):
        self.client,self.root=client,root;self.quota=None;self.case=None
        window=json.loads((root/'run-window.json').read_text()) if (root/'run-window.json').exists() else {}
        self.deadline=__import__('datetime').datetime.fromisoformat(window['deadline_utc']).timestamp() if window.get('deadline_utc') else float('inf')

    def update(self, metadata):
        headers=metadata.get('rate_limits') or {}
        try:
            limit=float(headers['x-ratelimit-limit-tokens']);remaining=float(headers['x-ratelimit-remaining-tokens'])
            reset=duration(headers.get('x-ratelimit-reset-tokens'))
            refill=(limit-remaining)/reset if reset and limit>remaining else limit/60
            self.quota=(time.monotonic(),limit,remaining,max(1.,refill*.85))
        except (KeyError,TypeError,ValueError):pass

    def prepare_output(self, allowance):
        if hasattr(self.client,'prepare_output'):self.client.prepare_output(allowance)

    def prepare_contract(self, signature):
        if hasattr(self.client,'prepare_contract'):self.client.prepare_contract(signature)

    def complete(self,messages):
        if time.time()>=self.deadline:raise CampaignDeadline('No new model request after the authorized work window')
        if self.quota:
            stamp,limit,remaining,refill=self.quota
            available=min(limit,remaining+(time.monotonic()-stamp)*refill)
            need=len(json.dumps(messages))/2.2+self.client.max_output_tokens+min(25000,limit*.1)
            delay=max(0,(need-available)/refill)
            if delay:
                if time.time()+delay>=self.deadline:raise CampaignDeadline('Quota wait would exceed the authorized work window')
                write_json(self.root/'model-wait.json',dict(case=self.case,reason='reported token headroom',wait_seconds=delay,until_unix=time.time()+delay))
                time.sleep(delay)
        # Retry transport throttling in-place, preserving the same messages and
        # the prior model answer if this is the bounded repair call.
        for retry in range(5):
            if time.time()>=self.deadline:raise CampaignDeadline('No new model request after the authorized work window')
            try:
                raw,metadata=self.client.complete(messages)
            except IncompleteResponse as exc:
                self.update(exc.metadata);raise
            except ProviderRequestError as exc:
                if exc.http_status!=429 or retry==4:raise
                delay=max(5,exc.retry_after_seconds or 60)
                if time.time()+delay>=self.deadline:raise CampaignDeadline('Rate-limit wait would exceed the authorized work window')
                stamp=time.time_ns()
                write_json(self.root/f'rate-limit-{stamp}.json',dict(case=self.case,http_status=429,wait_seconds=delay,reported_retry_seconds=exc.retry_after_seconds))
                print(json.dumps({'case':self.case,'status':'rate_wait','seconds':delay}),flush=True)
                time.sleep(delay);continue
            self.update(metadata)
            return raw,metadata



def source_hash(mode='model'):
    core=SOURCE/'nova/src/gpx_recipe'
    native=['__init__.py','advanced.py','compiler.py','contracts.py','effects.py','identity.py','execution.py','snapshot.py','studio_state.py','jobs.py','process.py','cli.py']
    paths=[core/n for n in native]+[Path(__file__).with_name('native_worker.py'),SOURCE/'scripts/recipe_docker_python.py',SOURCE/'scripts/gpx_recipe/__init__.py']
    if mode=='model':paths+=list(core.glob('*.py'))+[core/'knowledge.json',Path(__file__)]
    digest=hashlib.sha256()
    for p in sorted(set(paths)):digest.update(p.relative_to(SOURCE).as_posix().encode());digest.update(p.read_bytes())
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
        def canonical(c):
            if c['kind']=='uiso_equal':return json.dumps({'kind':'uiso_equal','atoms':sorted(c['atoms'],key=lambda t:(t['phase'],t['atom']))},sort_keys=True)
            terms=sorted(c['terms'],key=lambda t:(t['phase'],t['atom']));factor=float(terms[0]['coefficient'])
            return json.dumps({'terms':[{**t,'coefficient':round(float(t['coefficient'])/factor,12)} for t in terms], 'total':round(float(c['total'])/factor,12)},sort_keys=True)
        return sorted(set(canonical(c) for c in stage['constraints']))
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
    root=Path(args.campaign).resolve();catalog=json.loads((root/'catalog.json').read_text());revision=source_hash(args.mode)
    window=json.loads((root/'run-window.json').read_text()) if (root/'run-window.json').exists() else {}
    deadline=__import__('datetime').datetime.fromisoformat(window['deadline_utc']).timestamp() if window.get('deadline_utc') else float('inf')
    environment_path=root/'execution-environment.json'
    if environment_path.exists():environment=json.loads(environment_path.read_text())
    else:
        selected='radar-pd-ndip:validated-runtime-fix'
        image_id=subprocess.check_output(['docker','image','inspect',selected,'--format','{{.Id}}'],text=True).strip()
        environment={'requested_image':selected,'image_id':image_id};write_json(environment_path,environment)
    image_id=environment['image_id']
    all_cases={c['id']:c for c in catalog}
    if args.mode=='oracle':catalog=[c for c in catalog if c['expectation']=='execute']
    denominator=len(catalog)
    if args.only:catalog=[c for c in catalog if c['id'] in args.only]
    root.mkdir(parents=True,exist_ok=True)
    statepath=root/(args.mode+'-status.json');lock=__import__('threading').RLock()
    state=json.loads(statepath.read_text()) if statepath.exists() else {'denominator':len(catalog),'results':{},'history':[]}
    state['denominator']=denominator;state['selected_cases']=[c['id'] for c in catalog];state['image_id']=image_id
    def update(case,result):
        with lock:
            previous=state['results'].get(case['id'])
            if previous:state['history'].append(previous)
            result.update(case=case['id'],project=case['project'],category=case['category'],source_sha256=revision,
                          image_id=image_id,
                          case_sha256=hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest())
            state['results'][case['id']]=result;state['counts']=dict(collections.Counter(v['status'] for v in state['results'].values() if v.get('source_sha256')==revision and v.get('image_id')==image_id));state['updated_unix']=time.time();state['current_code']=revision
            write_json(statepath,state)
        print(json.dumps({k:result.get(k) for k in ['case','status','error','elapsed_seconds']}),flush=True)
    def verified(case):
        value=state['results'].get(case['id'],{})
        return value.get('source_sha256')==revision and value.get('image_id')==image_id and value.get('case_sha256')==hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest() and value.get('status') in {'verified_execution','scientific_stop','correctly_blocked','semantic_mismatch','planner_validation_failure','provider_blocked','runtime_defect','execution_timeout','deferred_deadline','blocked_dependency','deferred_provider'}
    def attempt(case):
        work=root/'cases'/case['id']/args.mode/(time.strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:6]);work.mkdir(parents=True)
        return work
    def native(case,plan,work,planning=None):
        if time.time()>=deadline:
            update(case,dict(status='deferred_deadline',artifact_directory=str(work)));return
        write_json(work/'plan.json',plan)
        env=scientific_environment();env['RADAR_GPX_DOCKER_WORKDIR']=str(root);env['RADAR_GPX_DOCKER_IMAGE']=image_id
        cmd=[str(SOURCE/'scripts/recipe_docker_python.py'),str(Path(__file__).with_name('native_worker.py')),str(root),case['id'],args.mode,'--workdir',str(work)]
        try:
            with (work/'native.log').open('w') as log:process=run_scientific(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=min(900,max(420,45*len(plan['stages']))))
            result=json.loads((work/'result.json').read_text()) if (work/'result.json').exists() else {'status':'runtime_defect','error':'Native worker exited without a result: '+str(process.returncode)}
        except subprocess.TimeoutExpired as e:result={'status':'execution_timeout','error':'Scientific worker exceeded its case timeout; its process group was terminated'}
        except Exception as e:result={'status':'runtime_defect','error':str(e)}
        result.update(artifact_directory=str(work),planning=planning);update(case,result)
    if args.mode=='oracle':
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(native,c,c['oracle'],attempt(c)) for c in catalog if not verified(c)]
            for f in as_completed(futures):f.result()
        return
    configure_provider('illumine-luna',args.key_file)
    client=PacedClient(make_client(),root)
    next_call=0;futures=[];accepted_plans={};service=ServiceCircuit()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for case in catalog:
            if verified(case):continue
            if time.time()>=deadline:break
            if service.halt:
                update(case,dict(status='deferred_provider',error='No model call attempted after provider circuit opened: '+service.halt['reason']))
                continue
            context=json.loads((root/'inputs'/(case['project']+'.context.json')).read_text());work=attempt(case)
            client.case=case['id'];client.client.max_output_tokens=2048
            previous=case.get('previous_plan')
            if case.get('previous_case'):
                prior_id=case['previous_case'];actual=accepted_plans.get(prior_id)
                if actual is None:
                    prior=state['results'].get(prior_id,{})
                    path=Path(prior.get('artifact_directory',''))/'planning.json'
                    prior_case=all_cases.get(prior_id)
                    if prior.get('source_sha256')==revision and path.is_file() and prior_case and prior_case['project']==case['project']:
                        proposed=json.loads(path.read_text())['plan']
                        prior_context=json.loads((root/'inputs'/(prior_case['project']+'.context.json')).read_text())
                        validate_plan(proposed,prior_context)
                        if not compare(proposed,prior_case['oracle'],prior_context):actual=(proposed,str(path))
                if actual is None:
                    update(case,dict(status='blocked_dependency',error='The actual preceding model plan was not semantically verified; no oracle plan substituted',artifact_directory=str(work)));continue
                previous,previous_path=actual
                write_json(work/'previous-model-plan.json',dict(case=prior_id,planning_artifact=previous_path,plan=previous))
            for retry in range(1):
                delay=max(0,next_call-time.monotonic())
                if delay:
                    if time.time()+delay>=deadline:break
                    write_json(root/'model-wait.json',dict(case=case['id'],wait_seconds=delay,reason='token-rate pacing',until_unix=time.time()+delay))
                    time.sleep(delay)
                started=time.monotonic()
                try:
                    planning=make_plan(case['request'],context,previous=previous,max_calls=2,client=client);write_json(work/'planning.json',planning)
                    service.success()
                    summary={'calls':len(planning['calls']),'reasoning_tokens':[c.get('usage',{}).get('output_tokens_details',{}).get('reasoning_tokens') for c in planning['calls']],'seconds':sum(c['elapsed_seconds'] for c in planning['calls']),'literal_guard_rejections':len(planning.get('literal_guard_rejections',[])),'literal_free_stages':planning.get('literal_free_stages',[])}
                    if case['expectation']=='needs_input':
                        if planning['status']=='needs_input':update(case,dict(status='correctly_blocked',planning=summary,artifact_directory=str(work)))
                        else:update(case,dict(status='semantic_mismatch',error='Unresolvable request was accepted',planning=summary,artifact_directory=str(work)))
                    else:
                        errors=compare(planning['plan'],case['oracle'],context)
                        if errors:update(case,dict(status='semantic_mismatch',error='; '.join(errors),planning=summary,artifact_directory=str(work)))
                        else:
                            # A revision depends on the reviewed proposal, not on
                            # an asynchronous native job having finished first.
                            accepted_plans[case['id']]=(planning['plan'],str(work/'planning.json'))
                            write_json(work/'semantic-plan-accepted.json',dict(plan_sha256=plan_hash(planning['plan']),source_sha256=revision,case_sha256=hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest()))
                            futures.append(pool.submit(native,case,planning['plan'],work,summary))
                    break
                except ProviderRequestError as e:
                    if service.fail(e,case['id']):write_json(root/'model-service-halt.json',service.halt)
                    update(case,dict(status='provider_blocked',error=str(e),artifact_directory=str(work)));break
                except CampaignDeadline as e:
                    write_json(work/'planning-error.json',{'error':str(e),'calls':getattr(e,'planning_calls',[])})
                    update(case,dict(status='deferred_deadline',error=str(e),artifact_directory=str(work)));break
                except ProviderUnavailable as e:
                    if service.fail(e,case['id']):write_json(root/'model-service-halt.json',service.halt)
                    write_json(work/'planning-error.json',{'error':str(e),'calls':getattr(e,'planning_calls',[]),'transport_cause':getattr(e,'cause',None),'transport_seconds':getattr(e,'elapsed_seconds',None)})
                    update(case,dict(status='provider_blocked',error=str(e),artifact_directory=str(work)));break
                except Exception as e:
                    calls=getattr(e,'calls',getattr(e,'planning_calls',[]))
                    write_json(work/'planning-error.json',{'error':str(e),'calls':calls})
                    update(case,dict(status='planner_validation_failure',error=str(e),artifact_directory=str(work)));break
        for f in as_completed(futures):f.result()


if __name__=='__main__':main()
