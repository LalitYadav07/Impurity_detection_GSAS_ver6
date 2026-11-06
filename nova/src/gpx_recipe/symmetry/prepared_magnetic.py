"""Reuse immutable subgroup construction, with fresh GPXs and native audits.

This capture is local to one sealed dispatch. It is never accepted as a
replacement for the original nuclear source or the final independent audit.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from ..execution import write_json
from ..snapshot import json_safe
from .compiler import confined,file_sha,package_signature
from .contracts import digest

SCHEMA='radar-prepared-magnetic/v1'


def seal_prepared(ctx,protocol,candidate,template,state,gpx):
    """Bind the native construction and its diagnostics to the exact inputs."""
    state=dict(state,magnetic_elements=sorted(state['magnetic_elements']))
    write_json(ctx.run_dir/'prepared-state.json',json_safe(state))
    files={p.name:file_sha(p) for p in ctx.run_dir.iterdir()
        if p.is_file() and (p.suffix=='.json' or p==gpx)}
    manifest=dict(schema=SCHEMA,protocol_sha256=digest(protocol),
        candidate_id=candidate['id'],implementation_sha256=package_signature(),
        source_sha256=protocol['source']['sha256'],gpx=gpx.name,
        seed_template=template,files=files)
    write_json(ctx.run_dir/'prepared-manifest.json',json_safe(manifest))
    return dict(root=str(ctx.run_dir.resolve()),
        manifest_sha256=file_sha(ctx.run_dir/'prepared-manifest.json'))


def verify_prepared(reference,ctx,protocol,candidate):
    """A digest-bound sibling capture only; never an arbitrary cached GPX."""
    root=Path(reference['root']).resolve()
    relative=root.relative_to(ctx.run_dir.parent.parent.resolve())
    if len(relative.parts)!=3 or relative.parts[1:]!=('group-data','prepared'):
        raise ValueError('Prepared magnetic capture escaped this dispatch')
    path=confined(root,'prepared-manifest.json')
    if file_sha(path)!=reference['manifest_sha256']:
        raise ValueError('Prepared magnetic manifest changed')
    manifest=json.loads(path.read_text())
    expected=dict(schema=SCHEMA,protocol_sha256=digest(protocol),
        candidate_id=candidate['id'],implementation_sha256=package_signature(),
        source_sha256=protocol['source']['sha256'])
    if any(manifest.get(k)!=v for k,v in expected.items()):
        raise ValueError('Prepared magnetic capture differs from the approved model')
    files=manifest.get('files',{})
    if not {'prepared-state.json',manifest.get('gpx')}<=files.keys():
        raise ValueError('Prepared magnetic capture is incomplete')
    for name,sha in files.items():
        if Path(name).name!=name or file_sha(confined(root,name))!=sha:
            raise ValueError('Prepared magnetic artifact changed')
    return root,manifest


def run_prepared(source,ctx,protocol,candidate,seed,*,screened):
    """Reseed a fresh native copy; keep every final validation unchanged."""
    from GSASII import GSASIIscriptable as G2sc
    from .native_magnetic import _finish
    from .profiled_groups import seed_moments
    began=time.monotonic()
    reference=screened['prepared']
    root,manifest=verify_prepared(reference,ctx,protocol,candidate)
    # The original source remains the worker's authoritative audit input.
    if file_sha(source)!=manifest['source_sha256']:
        raise ValueError('Prepared magnetic source changed')
    state=json.loads((root/'prepared-state.json').read_text())
    for profile in state['baseline_target_profiles'].values():
        profile['arrays']=tuple(np.asarray(v,dtype=float) for v in profile['arrays'])
        profile['valid']=np.asarray(profile['valid'],dtype=bool)
    project=G2sc.G2Project(gpxfile=str(root/manifest['gpx']),
        newgpx=str(ctx.run_dir/'magnetic-refinement.gpx'))
    phase=project.phase(state['magnetic_name'])
    cx,ct,_,_=phase.data['General']['AtomPtrs']
    labels=[a[ct-1] for a in phase.data['Atoms']]
    moments=seed_moments(manifest['seed_template'],seed,labels)
    audit={a['label']:a for a in state['moment_audit']}
    for atom,moment in zip(phase.data['Atoms'],moments):
        atom[cx+4:cx+7]=moment.tolist()
        audit[atom[ct-1]]['initial_moment']=moment.tolist()
    direction=seed.get('parent_direction')
    state['initial_moment_direction']=direction
    child_direction=None
    if direction is not None:
        transformed=np.asarray(manifest['seed_template']['parent_to_child'])@np.asarray(direction)
        # Match transform_native_moment's physical normalization.
        metric=np.asarray(manifest['seed_template']['sites'][0]['metric'])
        child_direction=(transformed/np.sqrt(transformed@metric@transformed)).tolist()
    state['candidate_seed_moment_direction']=child_direction
    # Copy verified construction diagnostics, updating only seed-dependent data.
    for name in manifest['files']:
        if name.endswith('.json') and name!='prepared-state.json':
            data=json.loads((root/name).read_text())
            if name=='site-policy-audit.json':data['kept']=state['moment_audit']
            if name=='moment-model.json':
                data.update(seed_axis=seed['axis'],independent_mode_seed=seed.get('independent_mode_seed'))
            ctx.save_json(name,data)
    ctx.save_json('prepared-reuse.json',dict(schema=SCHEMA,
        manifest_sha256=reference['manifest_sha256'],source_sha256=manifest['source_sha256'],
        candidate_id=candidate['id'],fresh_native_project=True,
        seed_id=seed['id'],independent_final_audit_required=True))
    project.save(str(ctx.run_dir/'magnetic-refinement.gpx'))
    result=_finish(ctx,protocol,candidate,project,state,
        construction_seconds=time.monotonic()-began,screened=screened)
    verify_prepared(reference,ctx,protocol,candidate)
    return result
