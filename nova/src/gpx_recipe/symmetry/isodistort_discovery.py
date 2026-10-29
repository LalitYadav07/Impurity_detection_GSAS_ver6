"""Cache authoritative Method-2 children; deferred OPDs remain in coverage."""
from __future__ import annotations
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from .compiler import file_sha
from .contracts import validate_candidate
from .isodistort_forms import IsodistortFormClient


def fixed_post(url,**kwargs):
    import requests
    # No response may redirect an uploaded structure to an unrelated host.
    return requests.post(url,allow_redirects=False,**kwargs)


def choose_tier(candidates,policy,maximum):
    if policy not in {'all','highest-symmetry-per-irrep'} or type(maximum) is not int or not 1<=maximum<=500:raise ValueError('Declare a finite OPD tier and construction budget')
    if policy=='all':chosen=list(candidates)
    else:
        by_irrep=defaultdict(list)
        for row in candidates:by_irrep[row['irrep']].append(row)
        chosen=[]
        for irrep,rows in sorted(by_irrep.items()):
            # Subgroup index measures lost parent symmetry; SG number does not.
            if any(not row.get('catalog_subgroup_index') for row in rows):raise ValueError('Authoritative subgroup indices are needed for a highest-symmetry tier')
            chosen.append(min(rows,key=lambda row:(row['catalog_subgroup_index'],row['catalog_lattice_index'],row['candidate_id'])))
    if len(chosen)>maximum:raise ValueError('Construction budget is smaller than the declared tier; choose a narrower scope explicitly')
    return chosen


def discover_structural(parent_cif,parent,k,directory,*,matcher,inventory,opd_policy='highest-symmetry-per-irrep',maximum=16,client=None):
    """matcher(cif,k,options) and inventory(path) run in the native subprocess.

    This is preparation only. A worker later verifies parent-orbit mapping and
    zero-mode forward equivalence before refining any selected displacement.
    """
    if len(k)!=3 or any(not isinstance(v,str) or Fraction(v).denominator>96 for v in k):raise ValueError('Provide a single rational k vector in the supplied parent basis')
    parent_cif=Path(parent_cif);root=Path(directory).resolve();root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):raise ValueError('Use a fresh discovery directory')
    client=client or IsodistortFormClient(post=fixed_post,max_attempts=2)
    raw=client.method2_k_vector_search(parent_cif.name,parent_cif.read_bytes(),k_vector=tuple(float(Fraction(v)) for v in k),k_form_matcher=lambda cif,vector,options:matcher(cif,k,options))
    chosen=choose_tier(raw['candidates'],opd_policy,maximum);ids={r['candidate_id'] for r in chosen}
    coverage=[dict(id=r['candidate_id'],label=r['label'],state='awaiting_construction' if r['candidate_id'] in ids else 'deferred',reason='' if r['candidate_id'] in ids else 'Outside the explicitly declared OPD tier') for r in raw['candidates']]
    (root/'discovery.json').write_text(json.dumps({k:v for k,v in raw.items() if k!='raw_responses'},indent=2)+'\n')
    raw_hashes={}
    for step,text in raw['raw_responses'].items():
        path=root/'raw'/('response-'+hashlib.sha256(step.encode()).hexdigest()[:16]+'.txt');path.parent.mkdir(exist_ok=True);path.write_text(text);raw_hashes[step]=file_sha(path)
    candidates=[]
    for row in chosen:
        entry=next(r for r in coverage if r['id']==row['candidate_id'])
        try:
            generated=client.method2_generate(raw['session_state'],row['candidate_id'])
            path=root/'children'/(row['candidate_id'].replace(':','-')+'.cif');path.parent.mkdir(exist_ok=True);path.write_text(generated['cif'])
            native=inventory(path);(path.with_suffix('.native.json')).write_text(json.dumps(native,indent=2)+'\n')
            # All available modes are retained; choosing primary/secondary
            # modes and their finite starts belongs to the reviewed protocol.
            candidate=validate_candidate(dict(id=row['candidate_id'],irrep=row['irrep_label'],transformation_matrix=row['catalog_basis'],origin_shift=row['catalog_origin'],child=dict(path=str(path.relative_to(root)),sha256=file_sha(path)),modes=native['modes']))
            candidates.append(candidate);entry.update(state='constructed',reason='')
        except Exception as exc:entry.update(state='construction_failed',reason=type(exc).__name__+': '+str(exc))
    catalog=dict(schema='radar-symmetry-catalog/v1',parent_setting_sha256=parent['setting_sha256'],k=[str(Fraction(v)) for v in k],candidates=candidates,
        coverage=dict(discovery_complete=not raw['irrep_failures'],discovered_count=len(coverage),constructed_count=len(candidates),opd_policy=opd_policy,candidates=coverage,irrep_failures=raw['irrep_failures']),
        provenance=dict(service='ISODISTORT Method 2',parent_cif_sha256=file_sha(parent_cif),raw_response_sha256=raw_hashes,standardized_parent_sha256=hashlib.sha256(raw['standardized_parent_cif'].encode()).hexdigest()))
    (root/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n');return catalog
