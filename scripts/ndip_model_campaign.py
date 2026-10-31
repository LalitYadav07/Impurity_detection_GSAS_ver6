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

from gpx_recipe.execution import write_json
from gpx_recipe.jobs import scientific_environment
from gpx_recipe.process import run_scientific
from gpx_recipe.symmetry.bundle import unpack,pack,hydrate_source
from gpx_recipe.symmetry.compiler import confined,file_sha,package_signature,require_adapters
from gpx_recipe.symmetry.contracts import digest
from gpx_recipe.symmetry.chunks import execute_chunk
from gpx_recipe.symmetry.resources import allocation

FIELDS={
    'inspect':({'source'},set()),
    'export':({'source','phase','destination'},set()),
    'match':({'cif','k','options'},set()),
    'modes':({'child','temporary'},set()),
    'modes-many':({'children','workers'},set()),
    'k-prepare':({'source','source_sha256','parent','histogram'},set()),
    'k-inventory':({'source','source_sha256','parent','parent_setting_sha256','sites','directions','catalogs'},set()),
    'k-template':({'directory'},set()),
    'k-portfolio':({'directories','workers','seconds'},set()),
}


def native(request,inputs,root,cancel):
    if set(request)!={'schema','operation','payload','implementation_sha256','expected_source_sha256','dispatch_key','timeout','input_hashes'} or request['schema']!='radar-model-native/v1':
        raise ValueError('Invalid native model request')
    operation=request['operation'];payload=copy.deepcopy(request['payload'])
    if operation not in FIELDS or not isinstance(payload,dict):raise ValueError('Unsupported native operation')
    required,optional=FIELDS[operation]
    if set(payload)-required-optional or required-set(payload):raise ValueError('Unexpected native operation fields')
    if request['implementation_sha256']!=package_signature():raise ValueError('The native worker image differs from NOVA')
    require_adapters()
    actual={str(p.relative_to(inputs)):file_sha(p) for p in inputs.rglob('*') if p.is_file()}
    if actual!=request['input_hashes']:raise ValueError('The native packet differs from its captured file hashes')
    import math
    if type(request['timeout']) not in {int,float} or not math.isfinite(request['timeout']) or not 1<=request['timeout']<=600:raise ValueError('Invalid native deadline')
    # Work on a private input capture; every path comes from a confined data packet.
    for key in ['source','child','directory']:
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
    elif operation.startswith('k-'):command=['-m','gpx_recipe.symmetry.k_native',operation[2:],str(request_file),str(output)]
    else:command=['-m','gpx_recipe.symmetry.isodistort_native',operation,str(request_file),str(output)]
    with (root/'native.log').open('w') as log:
        environment=scientific_environment()
        import gpx_recipe
        environment['PYTHONPATH']+=':'+str(Path(gpx_recipe.__file__).parent/'symmetry/adapters')
        process=run_scientific([sys.executable,*command],timeout=request['timeout'],cancel_event=cancel,
            cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT)
    if process.returncode or not output.is_file():raise RuntimeError('The native operation stopped before producing its checked response')
    return json.loads(output.read_text()),root


def work(action,source,assets,request_file,directory,archive):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    inputs=unpack(assets,directory/'inputs-packet')
    request=json.loads(Path(request_file).read_text())
    started=time.monotonic()
    expected=(request.get('protocol') or {}).get('source',{}).get('sha256') if action=='chunk' else request.get('expected_source_sha256')
    if not expected or file_sha(source)!=expected:raise ValueError('Galaxy source GPX differs from the captured acquisition')
    if action=='chunk':
        from gpx_recipe.symmetry.contracts import validate_protocol
        from gpx_recipe.symmetry.compiler import assets as protocol_assets
        protocol=validate_protocol(request.get('protocol'))
        hashes={asset['path']:asset['sha256'] for asset in protocol_assets(protocol)}
    else:hashes=request.get('input_hashes')
    hydrate_source(inputs,source,hashes,expected)
    before={str(p.relative_to(inputs)):file_sha(p) for p in inputs.rglob('*') if p.is_file()}
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
            files=[p for p in directory.rglob('*') if p.is_file() and p.suffix.lower() in {'.gpx','.json','.cif','.npz','.log','.txt','.lst'}
                and not (p.is_relative_to(inputs) and str(p.relative_to(inputs)) in before)]
            paths=[str(p.relative_to(directory)) for p in files]
            result=dict(schema='radar-model-native-result/v1',dispatch_key=request['dispatch_key'],
                request_sha256=digest(request),source_sha256=expected,result=result,
                worker_root=str(root),input_root=str(inputs),files={str(p.relative_to(directory)):file_sha(p) for p in files},
                native_seconds=time.monotonic()-started,llm_calls=0)
            write_json(directory/'response.json',result);base=directory;paths.append('response.json')
        else:raise ValueError('Unsupported model job action')
        for relative,sha in before.items():
            if file_sha(confined(inputs,relative))!=sha:raise ValueError('A native job changed its captured inputs')
        pack(base,archive,paths)
        return result
    except Exception:
        # Logs survive a controlled failure; never publish an unaudited GPX as a fit.
        with __import__('zipfile').ZipFile(archive,'w') as z:
            for p in directory.rglob('*.log'):z.write(p,str(p.relative_to(directory)))
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['native','chunk']);p.add_argument('source');p.add_argument('assets');p.add_argument('request');p.add_argument('directory');p.add_argument('archive')
    a=p.parse_args();work(a.action,a.source,a.assets,a.request,a.directory,a.archive)


if __name__=='__main__':main()
