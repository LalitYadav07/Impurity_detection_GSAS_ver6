"""Fixed-service discovery and provenance-bound catalogs; no LLM decisions."""
from __future__ import annotations

from fractions import Fraction
import hashlib,json,math
from .bilbao_parser import parse_magnetic_subgroups_gsas_output
from .contracts import validate_candidate

BCS_ENDPOINT='https://cryst.ehu.es/apibcs/v1/run'


def operation_text(operation):
    matrix=operation['rotation'];translation=operation['translation'];parts=[]
    if len(matrix)!=3 or any(len(row)!=3 for row in matrix) or len(translation)!=3:raise ValueError('Invalid native symmetry operation')
    for row,t in zip(matrix,translation):
        terms=[]
        for n,axis in zip(row,'xyz'):
            if not math.isfinite(n) or abs(n-round(n))>1e-8 or abs(n)>3:raise ValueError('Unsupported symmetry rotation')
            n=int(round(n))
            if n:terms.append(('+' if n>0 else '-')+(str(abs(n))+'*' if abs(n)!=1 else '')+axis)
        f=Fraction(float(t)%1).limit_denominator(24)
        if abs(float(f)-float(t)%1)>1e-8:raise ValueError('Unsupported symmetry translation precision')
        if f:terms.append('+'+str(f))
        parts.append(''.join(terms).lstrip('+') or '0')
    return ','.join(parts)


def magnetic_request(parent,k,*,whole_star=False):
    if type(whole_star) is not bool:raise ValueError('The star policy must be an explicit boolean')
    if len(k)!=3 or any(not isinstance(v,str) or Fraction(v).denominator>96 for v in k):raise ValueError('Provide one exact commensurate k-vector')
    generators=[operation_text(op) for op in parent['operations']]
    if not 1<=len(generators)<=192 or len(set(generators))!=len(generators):raise ValueError('A complete, unique native operation inventory is required')
    inputs=dict(generators=generators,starmagnetica='yes' if whole_star else 'no',celtodas='yes',landau='no',limite='spgroup',pointgroup='1',crystalsystem='1',centrosymmetry='0',polarity='0',listado='lista',sub='1.1')
    inputs.update({'km1'+axis:str(Fraction(value)) for axis,value in zip('xyz',k)})
    return dict(tool='kSUBGROUPSMAG_GSAS',inputs=inputs)


def magnetic_catalog(output,parent,k,*,request=None):
    parsed=parse_magnetic_subgroups_gsas_output(output)
    candidates=[validate_candidate(dict(id=c['candidate_id'],magnetic_space_group=c['magnetic_space_group'],bns_number=c.get('bns_number'),bns_lattice=c['bns_lattice'],transformation_matrix=c['transformation_matrix'],origin_shift=c['origin_shift'])) for c in parsed['candidates']]
    return dict(schema='radar-symmetry-catalog/v1',parent_setting_sha256=parent['setting_sha256'],k=[str(Fraction(v)) for v in k],
        candidates=candidates,coverage=dict(discovery_complete=True,constructed_count=len(candidates),scope='single declared propagation vector; selected star policy'),
        provenance=dict(service='Bilbao kSUBGROUPSMAG_GSAS',output_sha256=hashlib.sha256(output.encode()).hexdigest(),request=request))


def discover_magnetic(parent,k,*,api_key,whole_star=False,post=None):
    """Credential supplied by deployment configuration, never serialized.

    Cached output can be handed to magnetic_catalog instead. Native refinement
    workers consume that immutable catalog and do not call external services.
    """
    if not api_key or not api_key.strip():raise ValueError('Configure the Bilbao service credential outside the repository')
    if post is None:
        import requests
        post=requests.post
    payload=magnetic_request(parent,k,whole_star=whole_star)
    try:
        response=post(BCS_ENDPOINT,json=payload,headers={'Authorization':'Bearer '+api_key},timeout=90,allow_redirects=False)
        if response.status_code!=200:raise ValueError('Bilbao service returned HTTP '+str(response.status_code))
        if len(response.content)>20*1024*1024:raise ValueError('Bilbao response size budget exceeded')
        envelope=response.json();output=envelope['result']['output']
    except Exception as exc:
        # Requests/exceptions can retain request headers. Only a sanitized
        # category crosses the application boundary.
        raise ValueError('Bilbao discovery did not produce a complete catalog ('+type(exc).__name__+')') from None
    return magnetic_catalog(output,parent,k,request=payload)
