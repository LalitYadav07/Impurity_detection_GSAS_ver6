"""Compile an approved finite protocol into a relocatable, credential-free job."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import uuid

from .contracts import validate_candidate, digest, trial_id, validate_protocol, iter_trials


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def confined(root,relative,*,exists=True):
    root=Path(root).resolve();path=(root/relative).resolve(strict=exists)
    if not path.is_relative_to(root):raise ValueError('An asset escaped its job directory')
    return path


def assets(protocol):
    return [protocol['source'],protocol['catalog']]+[c['child'] for c in protocol['candidates'] if c['child']]


SHARED_ENTRYPOINT = 'from pathlib import Path\nimport sys\nimport gpx_recipe\nsys.path.insert(0,str(Path(gpx_recipe.__file__).parent/"symmetry/adapters"))\nfrom gpx_recipe.symmetry.worker import main\nif __name__ == "__main__":\n    main(Path(__file__).resolve().parent)\n'


def require_adapters():
    """Installed workers must import the same adapters bound by package_signature."""
    import importlib.util
    import sys
    for name in ['gsas_magnetic_preflight','gsas_isodistort_import','gsas_cif_import']:
        origin=getattr(sys.modules.get(name),'__file__',None)
        if origin is None:
            spec=importlib.util.find_spec(name);origin=spec.origin if spec else None
        if not origin or file_sha(origin)!=file_sha(Path(__file__).parent/'adapters'/(name+'.py')):
            raise ValueError('The native adapter differs from the pinned package: '+name)


def job_inputs(directory, job):
    """Resolve only the fixed, confined shared-input layout; no client paths."""
    directory = Path(directory).resolve()
    if job.get('schema') == 'radar-symmetry-job/v1':
        return directory
    if job.get('schema') != 'radar-symmetry-job/v2' or directory.name != job['id'] or directory.parent.name not in {'trials','preparing'}:
        raise ValueError('Invalid shared trial layout')
    if job['protocol_sha256'] != digest(validate_protocol(job['protocol'])):
        raise ValueError('Shared inputs differ from the approved protocol')
    return confined(directory.parent.parent, 'inputs/' + job['protocol_sha256'])


def prepare_shared_inputs(protocol, input_root, campaign_root):
    protocol = validate_protocol(protocol)
    sha = digest(protocol)
    target = Path(campaign_root).resolve() / 'inputs' / sha
    if target.exists():
        if digest(json.loads((target/'protocol.json').read_text())) != sha:
            raise ValueError('A shared protocol changed')
        verify_inputs(protocol, target)
        return target
    verify_inputs(protocol, input_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(sha + '.' + uuid.uuid4().hex + '.partial')
    staging.mkdir()
    try:
        for asset in assets(protocol):
            source = confined(input_root, asset['path'])
            destination = confined(staging, asset['path'], exists=False)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        (staging/'protocol.json').write_text(json.dumps(protocol, indent=2, allow_nan=False)+'\n')
        verify_inputs(protocol, staging)
        staging.replace(target)
    finally:
        if staging.exists():shutil.rmtree(staging)
    return target


def compile_shared_job(protocol, destination, *, candidate_id, seed_id, approved_hash):
    """Thin immutable manifest; inputs staged once and code pinned in the image.

    Enrollment validates/captures common inputs once; every native trial still
    independently rehashes them before and after its scientific calculation.
    """
    if digest(protocol) != approved_hash:
        raise ValueError('Approval does not bind this protocol')
    if not any(c['id'] == candidate_id and s['id'] == seed_id for c,s in iter_trials(protocol)):
        raise ValueError('Trial is outside the approved finite portfolio')
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Use a fresh trial directory')
    destination.mkdir(parents=True, exist_ok=True)
    job = dict(schema='radar-symmetry-job/v2', protocol=protocol, protocol_sha256=approved_hash,
        candidate_id=candidate_id, seed_id=seed_id, id=trial_id(protocol, candidate_id, seed_id),
        package_sha256=protocol['implementation_sha256'])
    (destination/'job.json').write_text(json.dumps(job, separators=(',',':'), allow_nan=False)+'\n')
    (destination/'worker.py').write_text(SHARED_ENTRYPOINT)
    return job


def canonical_candidate(value):
    value=validate_candidate(value)
    if value['child']:value['child']={'sha256':value['child']['sha256']}
    return value


def verify_inputs(protocol,root):
    for asset in assets(protocol):
        path=confined(root,asset['path'])
        if file_sha(path)!=asset['sha256']:raise ValueError('A sealed input changed: '+asset['path'])
    catalog=json.loads(confined(root,protocol['catalog']['path']).read_text())
    if catalog.get('schema')!='radar-symmetry-catalog/v1' or catalog.get('parent_setting_sha256')!=protocol['parent_setting_sha256'] or catalog.get('k')!=protocol['k']:
        raise ValueError('Catalog setting/k-vector is not bound to this protocol')
    rows=catalog.get('candidates',[])
    if len({c['id'] for c in rows})!=len(rows):raise ValueError('Ambiguous candidate IDs in catalog')
    registry={c['id']:canonical_candidate(c) for c in rows}
    for candidate in protocol['candidates']:
        registered=registry.get(candidate['id']);declared=canonical_candidate(candidate)
        if not registered or not set(declared['modes']).issubset(registered['modes']) or dict(registered,modes=declared['modes'])!=declared:
            raise ValueError('A candidate differs from the sealed discovery record')


def package_signature():
    """Bind the scientific implementation independently of installation paths."""
    root=Path(__file__).parent.parent
    # Include transitive native dependencies such as freeze(), extraction and
    # constraint auditing, not just the new symmetry entry points.
    values={str(p.relative_to(root)):file_sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in {'.py','.json'}}
    return digest(values)


def freeze_runtime(destination):
    """Ship the approved code as data beside the compiled, trusted entry point."""
    root=Path(__file__).parent.parent;destination=Path(destination)
    for path in root.rglob('*'):
        if path.is_file() and (path.suffix in {'.py','.json'} or path.name.endswith('LICENSE')):
            target=destination/'gpx_recipe'/path.relative_to(root)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    for name in ['gsas_magnetic_preflight.py','gsas_isodistort_import.py','gsas_cif_import.py']:
        target=destination/'scripts'/name;target.parent.mkdir(exist_ok=True);shutil.copyfile(root/'symmetry/adapters'/name,target)


def compile_bundle(protocol,input_root,destination,*,candidate_id,seed_id,approved_hash):
    protocol=validate_protocol(protocol)
    if approved_hash!=digest(protocol):raise ValueError('Approval does not bind the complete protocol')
    if protocol['implementation_sha256']!=package_signature():raise ValueError('The scientific implementation changed after approval')
    verify_inputs(protocol,input_root)
    candidate=next((c for c in protocol['candidates'] if c['id']==candidate_id),None)
    seed=next((s for s in protocol['seeds'] if s['id']==seed_id),None)
    if candidate is None or seed is None:raise ValueError('Trial is outside the approved finite portfolio')
    if 'candidate_ids' in seed and candidate_id not in seed['candidate_ids']:raise ValueError('This start belongs to another candidate')
    destination=Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):raise ValueError('Use a fresh job directory; completed evidence is immutable')
    destination.mkdir(parents=True,exist_ok=True)
    for asset in assets(protocol):
        source=confined(input_root,asset['path']);target=confined(destination,asset['path'],exists=False)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    job=dict(schema='radar-symmetry-job/v1',protocol=protocol,protocol_sha256=approved_hash,
             candidate_id=candidate_id,seed_id=seed_id,id=trial_id(protocol,candidate_id,seed_id),package_sha256=package_signature())
    freeze_runtime(destination/'runtime')
    (destination/'job.json').write_text(json.dumps(job,indent=2,allow_nan=False)+'\n')
    # Only this trusted entry point is generated; no model-written Python runs.
    code='from pathlib import Path\nimport sys\nroot=Path(__file__).resolve().parent\nsys.path[:0]=[str(root/"runtime"),str(root/"runtime/scripts")]\nfrom gpx_recipe.symmetry.worker import main\nif __name__ == "__main__":\n    main(root)\n'
    (destination/'worker.py').write_text(code)
    return job
