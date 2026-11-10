#!/usr/bin/env python3
"""Start, inspect or stop the integrated NDIP candidate on localhost only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

IMAGE='radar-pd-planner-local:0.7.0'
NAME='ndip-experiment-planner-local'
OWNER='org.ndip.experiment-planner.workspace'


def docker(*args,check=True):
    return subprocess.run(['docker',*args],check=check,text=True,capture_output=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['start','status','stop'])
    parser.add_argument('--work-dir',type=Path,default=Path.cwd()/'planner-workspace')
    parser.add_argument('--image',default=IMAGE)
    parser.add_argument('--port',type=int,default=8099)
    parser.add_argument('--ca-bundle',type=Path,help='Optional complete PEM bundle of trusted CAs; mounted read-only')
    parser.add_argument('--replace',action='store_true',help='Replace only this owned container, retaining all workspace files')
    args=parser.parse_args();root=args.work_dir.expanduser().resolve()
    if not 1024<=args.port<=65535:parser.error('Use a port from 1024 to 65535.')
    docker('info','--format','{{.ServerVersion}}')
    response=docker('container','inspect',NAME,check=False)
    container=json.loads(response.stdout)[0] if response.returncode==0 else None
    if container and container['Config']['Labels'].get(OWNER)!=str(root):
        raise RuntimeError('The container name belongs to another workspace. Supply its --work-dir; nothing was changed.')
    if args.action=='status':
        print(json.dumps({'running':bool(container and container['State']['Running']),
                          'workspace':str(root),'url':f'http://127.0.0.1:{args.port}/index.html'},indent=2))
        return
    if args.action=='stop':
        if container and container['State']['Running']:docker('stop','--time','65',NAME)
        print('Local NDIP stopped. Saved plans and results remain in '+str(root));return
    if container and args.replace:
        if container['State']['Running']:docker('stop','--time','65',NAME)
        docker('container','rm',NAME);container=None
    expected=docker('image','inspect',args.image,'--format','{{.Id}}').stdout.strip()
    if container:
        port=container['HostConfig']['PortBindings'].get('8080/tcp',[{}])[0].get('HostPort')
        if container['Image']!=expected or port!=str(args.port):
            raise RuntimeError('The owned container uses a different image or port. Re-run start with --replace; saved plans are retained.')
        if args.ca_bundle:
            raise RuntimeError('Changing CA configuration requires start --replace.')
        if not container['State']['Running']:docker('start',NAME)
    else:
        root.mkdir(parents=True,exist_ok=True)
        command=['run','-d','--name',NAME,'--platform','linux/amd64','--entrypoint','python',
                 '-e','RADAR_PD_LOCAL_EXPERIMENTAL_BACKEND=1',
                 '--label',OWNER+'='+str(root),'-p',f'127.0.0.1:{args.port}:8080',
                 '--mount',f'type=bind,src={root},dst=/work']
        if os.environ.get('BCS_API_KEY'):command+=['-e','BCS_API_KEY']
        if args.ca_bundle:
            ca=args.ca_bundle.expanduser().resolve(strict=True)
            if not ca.is_file():raise RuntimeError('The CA bundle must be a PEM file.')
            command+=['--mount',f'type=bind,src={ca},dst=/opt/network-ca.pem,readonly',
                      '-e','REQUESTS_CA_BUNDLE=/opt/network-ca.pem']
        docker(*command,args.image,'-m','radar_pd_nova','--host','0.0.0.0','--port','8080')
    url=f'http://127.0.0.1:{args.port}/index.html'
    for attempt in range(45):
        try:
            with urllib.request.urlopen(url,timeout=2) as result:
                if result.status==200:break
        except (OSError,TimeoutError):time.sleep(1)
    else:raise RuntimeError('The local app is still starting. Check docker logs '+NAME+'.')
    print('NDIP with Experiment Planner: '+url)
    print('Saved workspace: '+str(root))


if __name__=='__main__':
    try:main()
    except (OSError,RuntimeError,subprocess.CalledProcessError) as exc:
        print('Local launch failed: '+str(exc),file=sys.stderr);sys.exit(1)
