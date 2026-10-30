#!/usr/bin/env python3
"""Native inspect/run entry point; secrets and LLM clients are not required."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from gpx_recipe.execution import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    inspect=sub.add_parser('inspect');inspect.add_argument('source');inspect.add_argument('destination')
    run=sub.add_parser('run');run.add_argument('directory')
    args=parser.parse_args()
    if args.action=='inspect':
        from gpx_recipe.symmetry.native_audit import inspect_source
        context=inspect_source(Path(args.source).resolve());write_json(args.destination,context)
        print(json.dumps(dict(status='inspected',phases=len(context['phases']),histograms=len(context['histograms']))))
    else:
        from gpx_recipe.process import run_scientific
        root=Path(args.directory).resolve();job=json.loads((root/'job.json').read_text())
        timeout=job['protocol']['policy']['timeout_seconds']
        # Fresh process imports the frozen job package, not this launcher's
        # possibly newer installed package. Child process groups are bounded.
        with (root/'native.log').open('w') as log:
            try:result=run_scientific([sys.executable,str(root/'worker.py')],timeout=timeout,cwd=root,stdout=log,stderr=subprocess.STDOUT)
            except subprocess.TimeoutExpired:
                write_json(root/'result.json',dict(schema='radar-symmetry-trial/v1',id=job['id'],status='timed_out',reason='Declared trial walltime exhausted',protocol_sha256=job['protocol_sha256'],llm_calls=0))
                return
        if not (root/'result.json').exists():
            write_json(root/'result.json',dict(schema='radar-symmetry-trial/v1',id=job['id'],status='technical_failure',reason='Native worker exited without a terminal receipt',return_code=result.returncode,protocol_sha256=job['protocol_sha256'],llm_calls=0))


if __name__=='__main__':main()
