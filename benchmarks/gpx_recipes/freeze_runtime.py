"""Freeze the minimal controller/worker source tree for reproducible local campaigns."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

SOURCE=Path(__file__).resolve().parents[2]


def freeze(campaign):
    paths=[]
    for folder in ['nova/src/gpx_recipe','scripts/gpx_recipe','benchmarks/gpx_recipes']:
        paths.extend(p for p in (SOURCE/folder).iterdir() if p.is_file() and p.suffix in ['.py','.json','.html'])
    paths.extend(SOURCE/'scripts'/n for n in ['recipe_docker_python.py','ndip_recipe.py'])
    records={p.relative_to(SOURCE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    fingerprint=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest();root=Path(campaign).resolve()/'runtime'/fingerprint[:12]
    for relative in records:
        dest=root/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert hashlib.sha256(dest.read_bytes()).hexdigest()==records[relative]
        else:shutil.copy2(SOURCE/relative,dest)
    (root/'source-receipt.json').write_text(json.dumps({'source_sha256':fingerprint,'files':records,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()},indent=2)+'\n')
    (Path(campaign)/'active-runtime.txt').write_text(str(root)+'\n');return root


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('campaign');args=p.parse_args();print(freeze(args.campaign))
