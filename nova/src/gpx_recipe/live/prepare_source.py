"""Magnetic review around the shared nuclear live-snapshot transfer."""
from __future__ import annotations
import json
from pathlib import Path
from .contracts import file_hash
from .io import atomic_json
from .nuclear_transfer import validate_transfer, prepare_nuclear_snapshot


def validate_preparation(request):
    validate_transfer(request)
    if not request.get('catalog_sha256') or file_hash(request['catalog'])!=request['catalog_sha256']:
        raise ValueError('Live preparation input changed: catalog')
    if not request.get('sites'):raise ValueError('Review a magnetic parent and sites in Model exploration')
    return request


def prepare(request,root):
    from gsas_magnetic_preflight import setting_fingerprint,preflight_magnetic_sites
    from gpx_recipe.symmetry.chemistry import validate_sites
    validate_preparation(request)
    root=Path(root);dest=root/'prepared'
    catalog=json.loads(Path(request['catalog']).read_text());preflight=[]
    def review(project):
        parent=project.phase(request['parent_phase'])
        cx,ct,_,_=parent.data['General']['AtomPtrs']
        validate_sites([dict(label=a[ct-1],type=a[ct],occupancy=a[cx+3]) for a in parent.data['Atoms']],request['sites'])
        fingerprint=setting_fingerprint(parent.data['General']['SGData'])
        if catalog.get('parent_setting_sha256')!=fingerprint:
            raise ValueError('Magnetic catalogue belongs to a different nuclear setting')
        if catalog.get('k')!=request.get('k'):
            raise ValueError('Magnetic catalogue has a different propagation vector')
        for candidate in catalog['candidates']:
            try:
                _,audit=preflight_magnetic_sites(parent.data,dict(candidate,candidate_id=candidate['id']),
                    magnetic_sites=[s['label'] for s in request['sites']],site_policy='all',expected_parent_setting=fingerprint)
                preflight.append(dict(id=candidate['id'],compatible=audit['compatible'],dimensions=sum(s['independent_moment_count'] for s in audit['sites']),sites=audit['sites']))
            except Exception as exc:preflight.append(dict(id=candidate['id'],error=str(exc)))
    transferred=prepare_nuclear_snapshot(request,root,review=review)
    atomic_json(dest/'preflight.json',preflight)
    inference={key:request[key] for key in ('k','sites','source_revision','reference_revision')}
    inference.update(source_gpx='prepared/source.gpx',source_sha256=transferred['source_sha256'],catalog='inputs/catalog.json',catalog_sha256=request['catalog_sha256'],
        preflight='prepared/preflight.json',preflight_sha256=file_hash(dest/'preflight.json'),lower=float(request.get('lower',0)),upper=float(request.get('upper',9)),
        particles=int(request.get('particles',256)),workers=int(request.get('workers',1)),maximum_dimensions=int(request.get('maximum_dimensions',32)),
        candidate_seconds=float(request.get('candidate_seconds',45)),preparation_timeout_seconds=float(request.get('preparation_timeout_seconds',600)),
        excluded_parent_space_groups=request.get('excluded_parent_space_groups',[]))
    atomic_json(dest/'inference-request.json',inference)
    return dict(transferred,request='prepared/inference-request.json')
