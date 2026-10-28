"""Launch only a sealed source snapshot, keeping model/native logs out of git."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('round');parser.add_argument('--mode',choices=['oracle','model'],required=True);parser.add_argument('--workers',type=int,default=2);parser.add_argument('--selection');parser.add_argument('--key-file');args=parser.parse_args()
    root=Path(args.round).resolve();runtime=Path((root/'active-runtime.txt').read_text().strip())
    command=[sys.executable,str(runtime/'benchmarks/gpx_recipes/run_campaign.py'),str(root),'--mode',args.mode,'--workers',str(args.workers)]
    if args.selection:command+=['--only',*json.loads(Path(args.selection).read_text())['cases']]
    if args.mode=='model':
        if not args.key_file:parser.error('The model campaign needs its explicit owner-only backend key file')
        command+=['--key-file',args.key_file]
    environment=dict(os.environ);environment['PYTHONPATH']=str(runtime/'scripts');stamp=time.strftime('%Y%m%dT%H%M%S');log=root/(args.mode+'-'+stamp+'.log')
    (root/(args.mode+'-latest-log.txt')).write_text(str(log)+'\n')
    print(json.dumps({'mode':args.mode,'runtime':str(runtime),'log':str(log),'workers':args.workers}),flush=True)
    with log.open('w') as handle:result=subprocess.run(command,env=environment,stdout=handle,stderr=subprocess.STDOUT)
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
