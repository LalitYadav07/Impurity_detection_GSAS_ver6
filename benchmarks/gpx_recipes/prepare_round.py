"""Seal a new independent round without overwriting earlier attempts or contexts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from freeze_runtime import freeze


def prepare(corpus,name,partition=None,selection=None):
    corpus=Path(corpus).resolve();root=corpus/'rounds'/name
    if root.exists():raise ValueError('Use a new round name; existing rounds are immutable')
    root.mkdir(parents=True);(root/'inputs').mkdir()
    inventory=json.loads((corpus/'inventory.json').read_text());cases=json.loads((corpus/'catalog.json').read_text())
    if partition:cases=[c for c in cases if c.get('partition')==partition]
    if selection is not None:
        missing=set(selection)-{c['id'] for c in cases}
        if missing:raise ValueError('Selection contains unavailable case IDs: '+str(sorted(missing)))
        cases=[c for c in cases if c['id'] in selection]
    needed={c['project'] for c in cases}
    for key in sorted(needed):
        for suffix in ['.gpx','.context.json']:shutil.copy2(corpus/'inputs'/(key+suffix),root/'inputs'/(key+suffix))
    (root/'catalog.json').write_text(json.dumps(cases,indent=2)+'\n')
    (root/'projects.json').write_text(json.dumps([r for r in inventory if r['id'] in needed],indent=2)+'\n')
    shutil.copy2(corpus/'run-window.json',root/'run-window.json')
    for name in ['axis-domains-all.json','source-review.json']:
        if (corpus/name).exists():shutil.copy2(corpus/name,root/name)
    files=[root/'catalog.json',root/'projects.json',*sorted((root/'inputs').iterdir())]
    seal={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files};(root/'input-seal.json').write_text(json.dumps(seal,indent=2)+'\n')
    runtime=freeze(root);return {'round':str(root),'runtime':str(runtime),'cases':len(cases),'projects':len(needed),'partition':partition,'source_inputs_sealed':True}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('corpus');p.add_argument('name');p.add_argument('--partition');p.add_argument('--selection');a=p.parse_args();print(json.dumps(prepare(a.corpus,a.name,a.partition,json.loads(Path(a.selection).read_text())['cases'] if a.selection else None)))
