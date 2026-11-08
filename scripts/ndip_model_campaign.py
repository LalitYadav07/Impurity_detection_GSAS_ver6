#!/usr/bin/env python3
"""Galaxy entry point for bounded, credential-free model jobs."""
import argparse
import copy
import json
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

sys.path.insert(0,str(Path(__file__).resolve().parent))
from ndip_runtime import runtime_limits, process_usage, usage_delta

from gpx_recipe.execution import write_json
from gpx_recipe.process import run_scientific
from gpx_recipe.symmetry.bundle import unpack,pack,hydrate_source,DATA_SUFFIXES
from gpx_recipe.symmetry.compiler import confined,file_sha,package_signature,require_adapters
from gpx_recipe.symmetry.contracts import digest
from gpx_recipe.symmetry.chunks import execute_chunk
from gpx_recipe.symmetry.resources import allocation
from gpx_recipe.symmetry.native_protocol import validate_native_payload,validate_native_timeout,STRUCTURAL_SNAPSHOT_OPERATIONS


def native(request,inputs,root,cancel):
    if set(request)!={'schema','operation','payload','implementation_sha256','expected_source_sha256','dispatch_key','timeout','input_hashes'} or request['schema']!='radar-model-native/v1':
        raise ValueError('Invalid native model request')
    operation=request['operation'];payload=copy.deepcopy(request['payload'])
    validate_native_payload(operation,payload)
    if request['implementation_sha256']!=package_signature():raise ValueError('The native worker image differs from NOVA')
    require_adapters()
    actual={str(p.relative_to(inputs)):file_sha(p) for p in inputs.rglob('*') if p.is_file()}
    if actual!=request['input_hashes']:raise ValueError('The native packet differs from its captured file hashes')
    validate_native_timeout(operation,request['timeout'])
    # Work on a private input capture; every path comes from a confined data packet.
    for key in ['source','child','instrument','directory','plan','anchor_snapshot','snapshot']:
        if key in payload:payload[key]=str(confined(inputs,payload[key]))
    if 'source' in payload and file_sha(payload['source'])!=request['expected_source_sha256']:
        raise ValueError('Native source packet differs from the Galaxy GPX')
    for key in ['destination','temporary']:
        if key in payload:
            path=confined(root,payload[key],exists=False);path.parent.mkdir(parents=True,exist_ok=True);payload[key]=str(path)
    if operation=='modes-many':payload['children']=[str(confined(inputs,p)) for p in payload['children']]
    if operation=='k-portfolio':
        for directory in payload['directories']:confined(inputs,directory)
        payload['workers']=allocation(payload['workers'])['workers']
        # portfolio outputs are created beside each captured template.
        root=inputs
    if operation=='k-template':root=inputs
    request_file=root/'native-request.json';output=root/'native-response.json'
    write_json(request_file,payload)
    if operation=='inspect':command=['-m','gpx_recipe.symmetry.inspect',payload['source'],str(output)]
    elif operation=='structural-snapshot':command=['-m','gpx_recipe.structural_live.job',str(request_file),str(output)]
    elif operation=='structural-snapshot-ranking':command=['-m','gpx_recipe.structural_live.ranking',str(request_file),str(output)]
    elif operation.startswith('preparation-'):command=['-m','gpx_recipe.preparation.worker',operation.removeprefix('preparation-'),str(request_file),str(output)]
    elif operation.startswith('k-'):command=['-m','gpx_recipe.symmetry.k_native',operation[2:],str(request_file),str(output)]
    else:command=['-m','gpx_recipe.symmetry.isodistort_native',operation,str(request_file),str(output)]
    with (root/'native.log').open('w') as log:
        # The repository's scripts/gpx_recipe compatibility package forwards
        # submodule imports but has no adapters beside its own __init__.py.
        # Resolve from the actual reviewed compiler, not that package shim.
        from gpx_recipe.symmetry.compiler import native_environment
        environment=native_environment()
        process=run_scientific([sys.executable,*command],timeout=request['timeout'],cancel_event=cancel,
            cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT)
    if process.returncode or not output.is_file():
        reason=None
        if operation.startswith('preparation-'):
            from gpx_recipe.preparation.failures import read as read_failure
            reason=read_failure(output,operation.removeprefix('preparation-'),payload)
        raise RuntimeError(reason or 'The native operation stopped before producing its checked response')
    return json.loads(output.read_text()),root


def _work(action,source,assets,request_file,directory,archive,*,asset_parts=()):
    entered=time.monotonic()
    entry_utc=datetime.now(timezone.utc).isoformat()
    initial_usage=process_usage()
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    request=json.loads(Path(request_file).read_text())
    if asset_parts and action!='native':raise ValueError('Multipart inputs only support structural training')
    from gpx_recipe.preparation.structural_input_parts import stage
    inputs=stage(assets,asset_parts,request,directory/'inputs-packet')
    started=time.monotonic()
    expected=(request.get('protocol') or {}).get('source',{}).get('sha256') if action=='chunk' else request.get('expected_source_sha256')
    if not expected or file_sha(source)!=expected:raise ValueError('Galaxy scientific source differs from its captured input')
    if action=='chunk':
        from gpx_recipe.symmetry.contracts import validate_protocol
        from gpx_recipe.symmetry.compiler import assets as protocol_assets
        protocol=validate_protocol(request.get('protocol'))
        hashes={asset['path']:asset['sha256'] for asset in protocol_assets(protocol)}
    else:hashes=request.get('input_hashes')
    # Multipart structural training contains its required nuclear CIF already;
    # its closed per-part and whole-cohort inventory was verified by stage().
    if not asset_parts:hydrate_source(inputs,source,hashes,expected)
    before={str(p.relative_to(inputs)):file_sha(p) for p in inputs.rglob('*') if p.is_file()}
    staged=time.monotonic()
    cancel=threading.Event()
    def stop(*_):cancel.set()
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        if action=='chunk':
            result=execute_chunk(request,inputs,directory/'chunk',cancel=cancel)
            base=directory/'chunk'
            paths=['response.json','progress.json']+[str(p.relative_to(base)) for p in (base/'trials').rglob('*')
                if p.is_file() and p.suffix.lower() in {'.gpx','.json','.log','.npz'} and p.name!='job.json']
            paths += [row['path'] for row in result.get('profiled_cache',{}).get('entries',[])]
            from gpx_recipe.symmetry.campaign import verify_receipt
            for row in result['trials']:
                if row['state']!='finished' or row.get('status')!='valid':continue
                trial=base/'trials'/row['id']
                receipt=verify_receipt(trial,dict(id=row['id'],protocol_sha256=result['protocol_sha256']))
                relative='results/'+receipt['gpx']
                if relative not in receipt['files']:raise ValueError('A qualified project is absent from its audited artifact list')
                target=directory/'verified_projects'/('verified_'+row['id']+'.gpx')
                target.parent.mkdir(exist_ok=True);shutil.copyfile(confined(trial,relative),target)
        elif action=='native':
            root=directory/'native';root.mkdir()
            result,root=native(request,inputs,root,cancel)
            if request['operation'].startswith('preparation-') or request['operation'] in STRUCTURAL_SNAPSHOT_OPERATIONS:
                # Only declared portable data leaves theoretical preparation.
                # Native construction GPXs and repeated scratch are ephemeral.
                owned=Path(result['output']).resolve()
                if not owned.is_relative_to(root):raise ValueError('Preparation output escaped its native workspace')
                files=[confined(owned,name) for name in result['artifacts']]
                if request['operation']=='structural-snapshot':
                    # These GPXs have passed the native observation/export
                    # checks, not optimization or structure-identification
                    # qualification. Keep their collection explicitly separate.
                    for role in ('frozen','continuation'):
                        item=result['result'][role];path=confined(owned,str(Path(item['path']).relative_to(owned)))
                        if file_sha(path)!=item['sha256']:raise ValueError('Structural publication changed before collection')
                        target=directory/'reviewable_projects'/('reviewable_'+digest([request['dispatch_key'],role])+'.gpx')
                        target.parent.mkdir(exist_ok=True);shutil.copyfile(path,target)
            else:
                files=[p for p in directory.rglob('*') if p.is_file() and p.suffix.lower() in DATA_SUFFIXES
                    and not (p.is_relative_to(inputs) and str(p.relative_to(inputs)) in before)]
            paths=[str(p.relative_to(directory)) for p in files]
            result=dict(schema='radar-model-native-result/v1',dispatch_key=request['dispatch_key'],
                request_sha256=digest(request),source_sha256=expected,result=result,
                worker_root=str(root),input_root=str(inputs),files={str(p.relative_to(directory)):file_sha(p) for p in files},
                native_seconds=time.monotonic()-started,llm_calls=0)
            write_json(directory/'response.json',result);base=directory;paths.append('response.json')
        else:raise ValueError('Unsupported model job action')
        calculated=time.monotonic()
        for relative,sha in before.items():
            if file_sha(confined(inputs,relative))!=sha:raise ValueError('A native job changed its captured inputs')
        checked=time.monotonic()
        timing=dict(schema='radar-worker-timing/v1',entered_utc=entry_utc,
            staging_seconds=staged-entered,calculation_seconds=calculated-staged,
            final_input_verification_seconds=checked-calculated,
            scope='Inside the worker only; Galaxy dispatch, image pull and pre-entry staging are outside this clock')
        result['worker_timing']=timing
        write_json(base/'response.json',result)
        pack(base,archive,paths)
        print('[RADAR MODEL RUNTIME] '+json.dumps(dict(**timing,
            archive_seconds=time.monotonic()-checked,total_worker_seconds=time.monotonic()-entered,
            limits=runtime_limits(),usage=usage_delta(initial_usage))),flush=True)
        return result
    except Exception:
        # Logs survive a controlled failure; never publish an unaudited GPX as a fit.
        with __import__('zipfile').ZipFile(archive,'w') as z:
            for p in directory.rglob('*.log'):z.write(p,str(p.relative_to(directory)))
        raise


def work(action,source,assets,request_file,directory,archive,*,preparation_evidence=None,preparation_certificate=None,asset_parts=()):
    from gpx_recipe.preparation.authority import server_scope
    from gpx_recipe.preparation.certificates import produce
    parent=Path(directory).resolve().parent;parent.mkdir(parents=True,exist_ok=True)
    with server_scope(preparation_evidence,parent):
        result=_work(action,source,assets,request_file,directory,archive,asset_parts=asset_parts)
        if preparation_certificate is not None:
            produce(result,directory,archive,preparation_certificate)
        return result


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['native','chunk']);p.add_argument('source');p.add_argument('assets');p.add_argument('request');p.add_argument('directory');p.add_argument('archive')
    p.add_argument('--preparation-evidence');p.add_argument('--preparation-certificate')
    p.add_argument('--asset-part',action='append',default=[])
    a=p.parse_args();work(a.action,a.source,a.assets,a.request,a.directory,a.archive,
        preparation_evidence=a.preparation_evidence,preparation_certificate=a.preparation_certificate,asset_parts=a.asset_part)


if __name__=='__main__':main()
