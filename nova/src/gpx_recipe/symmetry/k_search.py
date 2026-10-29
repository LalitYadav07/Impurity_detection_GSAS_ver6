"""Bounded, deterministic, cached k survey before reviewed full refinement."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from collections import defaultdict,deque
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
import uuid
import numpy as np

from ..commensurate_k import rank_profiles,reciprocal_equivalent
from ..execution import write_json
from ..jobs import scientific_environment
from ..process import run_scientific
from .compiler import compile_bundle,file_sha,package_signature,confined
from .contracts import digest,validate_protocol
from .plans import build_plan


class SearchStopped(RuntimeError):pass


class NativeRunner:
    supports_portfolio=True
    def __init__(self,command,*,environment=None,cancel_event=None):
        self.command=list(command);self.environment=dict(environment or {});self.cancel_event=cancel_event
        if set(self.environment)-{'GSASII_ROOT','RADAR_GPX_DOCKER_IMAGE','PYTHONPATH'}:
            raise ValueError('Only scientific runtime settings belong in a native k worker')

    def call(self,action,request,directory,*,timeout):
        directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
        uid=uuid.uuid4().hex;inp=directory/(uid+'-request.json');out=directory/(uid+'-response.json')
        write_json(inp,request)
        env={**scientific_environment(),**self.environment,'RADAR_GPX_DOCKER_WORKDIR':str(directory)}
        with (directory/(uid+'-native.log')).open('w') as log:
            completed=run_scientific(self.command+['-m','gpx_recipe.symmetry.k_native',action,str(inp),str(out)],
                timeout=timeout,cancel_event=self.cancel_event,env=env,stdout=log,stderr=subprocess.STDOUT)
        if completed.returncode or not out.is_file():raise RuntimeError('Native k preparation stopped before publishing its checked response')
        response=json.loads(out.read_text())
        if response.get('status') not in (None,'simulated','k_periodicity_mismatch'):raise ValueError(response.get('reason','Native k preparation failed'))
        return response


def representative_starts(eligible,sites,*,limit,relative_signs=True):
    """Round robin through BNS families, then embeddings, axes and site signs.

    The bounded sample is explicit; unsampled embeddings remain in coverage.
    This is not an exhaustive magnetic structure search.
    """
    if type(limit) is not int or not 1<=limit<=48:raise ValueError('Choose 1–48 native templates per k')
    grouped=defaultdict(deque)
    rows=sorted(eligible,key=lambda r:(-r['operation_count'],r['cell_multiplier'],r['candidate_id'],r['axis']))
    signs=[dict.fromkeys([s['label'] for s in sites],1)]
    if relative_signs and len(sites)>1:
        if len(sites)>5:raise ValueError('Relative-sign starts are bounded to five selected parent sites')
        signs=[dict(zip([s['label'] for s in sites],(1,*tail))) for tail in itertools.product([1,-1],repeat=len(sites)-1)]
    # Even a one-model family must not exhaust the entire budget before other
    # symmetries get a representative. Each complete pass adds another start.
    for sign in signs:
        for row in rows:grouped[row['family']].append(dict(row,site_signs=sign))
    selected=[]
    axis_count=defaultdict(int);sign_count=defaultdict(int);pair_count=defaultdict(int)
    while len(selected)<limit and any(grouped.values()):
        for values in grouped.values():
            if values and len(selected)<limit:
                def priority(row):
                    sign=tuple(sorted(row['site_signs'].items()));axis=row['axis']
                    return pair_count[(axis,sign)],axis_count[axis],sign_count[sign]
                choice=min(values,key=priority);values.remove(choice);selected.append(choice)
                sign=tuple(sorted(choice['site_signs'].items()));axis=choice['axis']
                pair_count[(axis,sign)]+=1;axis_count[axis]+=1;sign_count[sign]+=1
    return selected


class KSearch:
    def __init__(self,cache_root,native,discover,*,clock=time.monotonic):
        self.cache=Path(cache_root).resolve();self.cache.mkdir(parents=True,exist_ok=True)
        self.native=native;self.discover=discover;self.clock=clock

    def run(self,source,root,*,parent,histogram,sites,directions=('x','y','z'),
            maximum_moment=6.,magnetic_d_min=0.,relative_signs=True,templates_per_k=6,
            budget_seconds=60,workers=2,progress=None,cancel=None):
        if type(budget_seconds) is not int or not 10<=budget_seconds<=600:raise ValueError('Choose a 10–600 second survey budget')
        if type(workers) is not int or not 1<=workers<=4:raise ValueError('Choose 1–4 native workers')
        if type(templates_per_k) is not int or not 1<=templates_per_k<=48:raise ValueError('Choose 1–48 templates per k')
        root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
        cancel=cancel or threading.Event();started=self.clock();deadline=started+budget_seconds
        def remaining():
            if cancel.is_set():raise SearchStopped('K search cancelled')
            return max(0.,deadline-self.clock())
        def announce(stage,message,**extra):
            if progress:progress(dict(stage=stage,message=message,**extra))
        def native(action,payload,directory,cap=60):
            duration=remaining()
            if duration<.25:raise SearchStopped('Survey time budget reached')
            return self.native.call(action,payload,directory,timeout=min(cap,duration))
        captured=root/'source.gpx';shutil.copyfile(source['source_path'],captured)
        expected=source['source_sha256']
        if file_sha(captured)!=expected:raise ValueError('The selected GPX changed while the survey captured it')
        announce('prepare','Reading the exact parent, bank calibration and nuclear residual')
        prepared=native('prepare',dict(source=str(captured),source_sha256=expected,parent=parent,histogram=histogram),root)
        if prepared['source_sha256']!=expected or not prepared['input_unchanged']:raise ValueError('Native preparation used a different source revision')
        baseline=confined(root,prepared['profile'])
        if file_sha(baseline)!=prepared['profile_sha256']:raise ValueError('The prepared nuclear profile changed')
        context=prepared['context'];parent_context=next(p for p in context['phases'] if p['name']==parent)
        # Validate all chemistry/bounds even if an empty service response would
        # otherwise prevent those inputs from reaching the native worker.
        from .chemistry import validate_sites
        validate_sites(parent_context['atoms'],sites)
        if not 0<float(maximum_moment)<=20 or not 0<=float(magnetic_d_min)<=10:raise ValueError('Invalid magnetic bounds')
        points=prepared['points']['candidates'];families=[];catalogs={}
        for point in points:
            families.append(dict(k=point['k'],labels=point['labels'],discovery='untested',discovered=0,eligible=0,
                planned=0,attempted=0,simulated=0,status='untested',reason=''))
        def catalog_for(index):
            family=families[index];key=digest(dict(parent=parent_context['setting_sha256'],k=family['k'],star=False,service='Bilbao-kSUBGROUPSMAG_GSAS-v1'))
            target=self.cache/'catalogs'/(key+'.json');target.parent.mkdir(exist_ok=True)
            if target.exists():
                saved=json.loads(target.read_text());catalog=saved['catalog']
                if saved['sha256']!=digest(catalog) or catalog['parent_setting_sha256']!=parent_context['setting_sha256'] or catalog['k']!=family['k']:
                    raise ValueError('A cached subgroup catalog changed')
                return index,catalog,True
            # Reuse a previously requested representative of the same k-star
            # without relabelling its catalog as a request we never made.
            # Centred-cell reciprocal equivalence must use native translations.
            for alternative in sorted(target.parent.glob('*.json')):
                saved=json.loads(alternative.read_text());catalog=saved['catalog']
                if catalog.get('parent_setting_sha256')!=parent_context['setting_sha256']:continue
                if saved['sha256']!=digest(catalog):raise ValueError('A cached subgroup catalog changed')
                if reciprocal_equivalent(catalog['k'],family['k'],parent_context['operations']):
                    family['generated_k']=family['k'];family['k']=catalog['k']
                    return index,catalog,True
            duration=remaining()
            if duration<1:raise SearchStopped('Survey time budget reached before discovery')
            catalog=self.discover(parent_context,family['k'],timeout_seconds=min(15.,duration))
            if catalog['parent_setting_sha256']!=parent_context['setting_sha256'] or catalog['k']!=family['k']:raise ValueError('Subgroup discovery returned another setting or k')
            temporary=target.with_name(target.name+'.'+uuid.uuid4().hex)
            write_json(temporary,dict(catalog=catalog,sha256=digest(catalog)));temporary.replace(target)
            return index,catalog,False
        announce('discovery',f'Looking up {len(points)} k families; cached catalogs are reused')
        pool=ThreadPoolExecutor(max_workers=min(4,max(1,len(points))))
        pending={pool.submit(catalog_for,i):i for i in range(len(points))}
        try:
            while pending and remaining()>.1:
                ready,_=wait(pending,timeout=min(.2,remaining()),return_when=FIRST_COMPLETED)
                for future in ready:
                    i=pending.pop(future)
                    try:
                        _,catalog,cached=future.result();catalogs[i]=catalog
                        families[i].update(discovery='cached' if cached else 'complete',discovered=len(catalog['candidates']),status='catalogued')
                    except Exception as exc:families[i].update(discovery='unavailable',status='untested',reason=str(exc))
                    announce('discovery',f'{len(families)-len(pending)} / {len(families)} k catalogs checked')
        finally:pool.shutdown(wait=True,cancel_futures=True)
        profiles=[];template_records={};errors=[];inventory=None
        if catalogs and remaining()>.25:
            announce('inventory','Checking magnetic sites, allowed directions and subgroup settings')
            try:
                inventory_request=dict(source=str(captured),source_sha256=expected,parent=parent,
                    parent_setting_sha256=parent_context['setting_sha256'],sites=sites,directions=list(directions),
                    catalogs=[catalogs[i] for i in sorted(catalogs)])
                identity={k:v for k,v in inventory_request.items() if k!='source'}
                key=digest(dict(inputs=identity,implementation=package_signature(),runtime=context['runtime']['native_stack_sha256']))
                inventory_cache=self.cache/'inventories'/(key+'.json');inventory_cache.parent.mkdir(exist_ok=True)
                if inventory_cache.exists():
                    saved=json.loads(inventory_cache.read_text());inventory=saved['inventory']
                    if digest(inventory)!=saved['sha256'] or inventory.get('source_sha256')!=expected:raise ValueError('A cached native inventory changed')
                else:
                    inventory=native('inventory',inventory_request,root,cap=600)
                    temporary=inventory_cache.with_name(inventory_cache.name+'.'+uuid.uuid4().hex)
                    write_json(temporary,dict(inventory=inventory,sha256=digest(inventory)));temporary.replace(inventory_cache)
            except (ValueError,RuntimeError,subprocess.TimeoutExpired) as exc:inventory=None;errors.append(str(exc))
        tasks=[]
        if inventory:
            by_k={tuple(row['k']):row for row in inventory['families']}
            for i,catalog in sorted(catalogs.items()):
                family=families[i];record=by_k[tuple(family['k'])]
                family['eligible']=len(record['eligible']);family['dispositions']=record['dispositions']
                family['compatible_candidate_ids']=sorted({r['candidate_id'] for r in record['eligible']})
                family['available_starts']=len(record['eligible'])*(2**(len(sites)-1) if relative_signs else 1)
                choices=representative_starts(record['eligible'],sites,limit=templates_per_k,relative_signs=relative_signs)
                family.update(planned=len(choices),status='pending' if choices else 'outside_supported_scope')
                if not choices:family['reason']='No supported collinear start for these selected sites and directions; this is not a physical rejection of the k family.'
                if not choices:continue
                inputs=root/('k-'+str(i));inputs.mkdir();shutil.copyfile(captured,inputs/'source.gpx')
                write_json(inputs/'catalog.json',catalog)
                settings=dict(parent=parent,histogram=histogram,kind='magnetic',k=family['k'],sites=sites,directions=list(directions),
                    candidates=sorted({choice['candidate_id'] for choice in choices}),relative_sign_starts=relative_signs,
                    cycles=1,adequate_rwp=100.,maximum_moment=float(maximum_moment),magnetic_d_min=float(magnetic_d_min),
                    maximum_mode=.5,modes=[],start_mode=.1)
                protocol=build_plan(context,catalog,inputs,settings)
                # The simulation path fits no parameters. A single declared
                # nonzero moment supplies a shape; intensity is scored later.
                for choice in choices:
                    seed=next(s for s in protocol['seeds'] if s['axis']==choice['axis'] and s.get('site_signs',dict.fromkeys([v['label'] for v in sites],1))==choice['site_signs'])
                    tasks.append((i,protocol,inputs,choice['candidate_id'],seed['id']))
        # Interleave families so a short deadline does not spend its entire
        # budget on Gamma before trying any nonzero propagation vector.
        by_family=defaultdict(deque)
        for task in tasks:by_family[task[0]].append(task)
        tasks=[]
        while any(by_family.values()):
            for items in by_family.values():
                if items:tasks.append(items.popleft())
        def stage(task):
            i,protocol,inputs,candidate_id,seed_id=task
            key=digest(dict(protocol=digest(protocol),candidate=candidate_id,seed=seed_id,runtime=context['runtime']['native_stack_sha256']))
            cache=self.cache/'templates'/key;cache.mkdir(parents=True,exist_ok=True)
            receipt=cache/'receipt.json'
            if receipt.exists():
                saved=json.loads(receipt.read_text());path=confined(cache,saved['profile'])
                if file_sha(path)!=saved['profile_sha256'] or saved['source_sha256']!=expected or saved.get('protocol_sha256')!=digest(protocol) or not saved['audit']['passed'] or not saved.get('periodicity',{}).get('passed'):
                    raise ValueError('A cached simulation artifact changed')
                return dict(cached=(i,saved,path,True))
            directory=cache/('attempt-'+uuid.uuid4().hex)
            compile_bundle(protocol,inputs,directory,candidate_id=candidate_id,seed_id=seed_id,approved_hash=digest(protocol))
            return dict(task=task,directory=directory,cache=cache,receipt=receipt)
        def collect(staged,response):
            task=staged['task'];i,protocol,_,candidate_id,seed_id=task
            directory=staged['directory'];cache=staged['cache'];receipt=staged['receipt']
            if response.get('status')=='k_periodicity_mismatch':
                return i,dict(response,candidate_id=candidate_id,seed_id=seed_id),None,False
            if response.get('status')!='simulated':raise ValueError(response.get('reason','Native simulation did not complete'))
            job=json.loads((directory/'job.json').read_text())
            if response.get('id')!=job['id'] or response.get('protocol_sha256')!=job['protocol_sha256']:
                raise ValueError('Native simulation returned another trial identity')
            path=confined(directory,response['profile'])
            if response['source_sha256']!=expected or not response.get('no_refinement') or not response.get('audit',{}).get('passed') or not response.get('periodicity',{}).get('passed') or file_sha(path)!=response['profile_sha256']:
                raise ValueError('Native magnetic simulation failed its effect/identity audit')
            response.update(profile=str(path.relative_to(cache)),candidate_id=candidate_id,seed_id=seed_id,k=protocol['k'])
            temporary=cache/('receipt-'+uuid.uuid4().hex+'.json');write_json(temporary,response);temporary.replace(receipt)
            return i,response,path,False
        def simulate(task):
            staged=stage(task)
            if 'cached' in staged:return staged['cached']
            response=native('template',dict(directory=str(staged['directory'])),staged['directory'],cap=20)
            return collect(staged,response)
        def publish(value):
            i,response,path,cached=value
            if response.get('status')=='k_periodicity_mismatch':
                families[i].setdefault('simulation_rejections',[]).append(dict(candidate_id=response['candidate_id'],seed=response['seed_id'],reason=response['reason']))
                return
            with np.load(path,allow_pickle=False) as data:intensity=data['intensity'].copy()
            uid=digest([i,response['id']])[:24]
            profiles.append(dict(id=uid,k=families[i]['k'],intensity=intensity,
                maximum_amplitude=(float(maximum_moment)/response['initial_maximum_moment'])**2))
            template_records[uid]=dict(path=str(path),sha256=file_sha(path),response=response,cached=cached)
            families[i]['simulated']+=1
        announce('simulate',f'Simulating up to {len(tasks)} representative native profiles')
        pending={};todo=deque(tasks)
        if getattr(self.native,'supports_portfolio',False):
            staged_jobs={}
            while todo and remaining()>.25:
                task=todo.popleft();families[task[0]]['attempted']+=1
                try:
                    staged=stage(task)
                    if 'cached' in staged:publish(staged['cached'])
                    else:staged_jobs[str(staged['directory'].relative_to(self.cache))]=staged
                except Exception as exc:families[task[0]].setdefault('simulation_failures',[]).append(dict(candidate_id=task[3],seed=task[4],reason=str(exc)))
            if staged_jobs and remaining()>2:
                announce('simulate',f'{len(profiles)} cached profiles · {len(staged_jobs)} fresh native starts in {workers} workers')
                try:
                    response=native('portfolio',dict(directories=list(staged_jobs),workers=workers,seconds=max(.1,remaining()-2)),self.cache,cap=600)
                    for item in response['results']:
                        if item['directory'] not in staged_jobs:raise ValueError('Portfolio returned an unknown or duplicate trial')
                        staged=staged_jobs.pop(item['directory'])
                        try:publish(collect(staged,item['response']))
                        except Exception as exc:
                            task=staged['task'];families[task[0]].setdefault('simulation_failures',[]).append(dict(candidate_id=task[3],seed=task[4],reason=str(exc)))
                except (ValueError,RuntimeError,subprocess.SubprocessError) as exc:errors.append(str(exc))
            pending=staged_jobs
        else:
            pool=ThreadPoolExecutor(max_workers=workers)
            try:
                while (todo or pending) and remaining()>.1:
                    while todo and len(pending)<workers and remaining()>.25:
                        task=todo.popleft();families[task[0]]['attempted']+=1;pending[pool.submit(simulate,task)]=task
                    if not pending:break
                    ready,_=wait(pending,timeout=min(.2,remaining()),return_when=FIRST_COMPLETED)
                    for future in ready:
                        task=pending.pop(future);i=task[0]
                        try:publish(future.result())
                        except Exception as exc:
                            families[i].setdefault('simulation_failures',[]).append(dict(candidate_id=task[3],seed=task[4],reason=str(exc)))
                        announce('simulate',f'{len(profiles)} profiles ready · {sum(f["attempted"] for f in families)} starts checked',
                                 completed=sum(f['simulated'] for f in families),total=len(tasks))
            finally:pool.shutdown(wait=True,cancel_futures=True)
        complete=bool(inventory) and not pending and not todo and len(catalogs)==len(points) and not errors and not any(f.get('simulation_failures') for f in families)
        with np.load(baseline,allow_pickle=False) as data:
            ranking=rank_profiles(data['observed'],data['nuclear'],data['weights'],profiles,mask=data['mask'],complete=complete)
        by_k={tuple(row['k']):row for row in ranking['families']}
        for family in families:
            scored=by_k.get(tuple(family['k']))
            if scored:family.update(scored,status='sampled' if family['simulated']+len(family.get('simulation_rejections',[]))==family['planned'] else 'partial')
            elif family['status']=='pending':
                family['status']='outside_supported_scope' if len(family.get('simulation_rejections',[]))==family['planned'] else 'untested'
                if family['status']=='outside_supported_scope':family['reason']='The tested moment starts did not carry the requested k; other starts or a more general executor may be needed.'
        # A missing or unsupported family is never assigned a zero evidence
        # score, and never silently disappears from the shortlist table.
        families.sort(key=lambda f:(f.get('score') is None,-f.get('score',0),f['k']))
        result=dict(schema='radar-k-search/v1',status=ranking['status'],complete=complete,
            source_sha256=expected,parent=parent,histogram=histogram,settings=dict(sites=sites,directions=list(directions),
                maximum_moment=maximum_moment,magnetic_d_min=magnetic_d_min,relative_signs=relative_signs,templates_per_k=templates_per_k),
            elapsed_seconds=self.clock()-started,budget_seconds=budget_seconds,points=prepared['points'],
            baseline=str(baseline),baseline_sha256=file_sha(baseline),ranking=ranking,families=families,
            templates=template_records,catalogs={','.join(v['k']):v for v in catalogs.values()},errors=errors,
            coordinate='tof' if 'T' in str(next(h for h in context['histograms'] if h['name']==histogram)['instrument']['Type']) else 'two_theta',
            source_unchanged=file_sha(captured)==expected,llm_calls=0,scope='Finite high-symmetry, collinear template survey; not exhaustive magnetic identification')
        write_json(root/'survey.json',result);announce('finished','K survey ready to inspect')
        return result
