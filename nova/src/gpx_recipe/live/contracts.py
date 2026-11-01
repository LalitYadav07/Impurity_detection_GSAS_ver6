"""Fail-closed boundaries between browser orchestration and native jobs."""
import hashlib,json,math,re
from fractions import Fraction
from pathlib import Path


def file_hash(path):
    with Path(path).open('rb') as stream:
        h=hashlib.sha256()
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def validate_request(req,binding=None):
    if not isinstance(req,dict):raise ValueError('A scientific request object is required')
    for field,maximum in [('source_gpx',128),('catalog',4),('preflight',4),('observation',32),('observation_source',64)]:
        if req.get(field) and Path(req[field]).stat().st_size>maximum*1024*1024:
            raise ValueError('Scientific input exceeds its byte budget: '+field)
    for field in ('source_gpx','catalog','preflight'):
        if file_hash(req[field])!=req[field.replace('_gpx','')+'_sha256']:raise ValueError('Scientific input changed: '+field)
    for key,lo,hi,integer in [('particles',64,4096,True),('workers',1,4,True),('maximum_dimensions',1,64,True),('candidate_seconds',.01,600,False),('preparation_timeout_seconds',1,3600,False)]:
        value=req.get(key,{'workers':1,'maximum_dimensions':32,'candidate_seconds':60,'preparation_timeout_seconds':600}.get(key))
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not lo<=value<=hi or (integer and not isinstance(value,int)):
            raise ValueError('Invalid worker budget: '+key)
    lower,upper=req.get('lower'),req.get('upper')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in (lower,upper)) or not 0<=lower<upper:raise ValueError('Invalid ordered-moment bounds')
    if len(req.get('k',[]))!=3:raise ValueError('Three propagation-vector components required')
    try:k=[Fraction(str(v)) for v in req['k']]
    except (ValueError,ZeroDivisionError):raise ValueError('Finite rational propagation vector required') from None
    sites=req.get('sites') or []
    if not sites or len({s['label'] for s in sites})!=len(sites):raise ValueError('Distinct magnetic sites required')
    if binding:
        if req.get('observation') and req.get('observation_sha256')!=binding.get('observation_sha256'):raise ValueError('Observation is not bound to the reviewed update')
        if binding.get('request'):
            template=json.loads(Path(binding['request']).read_text())
            for field in ('catalog','catalog_sha256','preflight','preflight_sha256','sites','excluded_parent_space_groups'):
                if req.get(field)!=template.get(field):raise ValueError('Request changes the reviewed candidate set or magnetic sites')
        for field in ('source_gpx','source_sha256'):
            if req[field]!=binding[field]:raise ValueError('Request does not match reviewed nuclear preparation')
        if req['source_revision'].split(':')[-1]!=binding['source_dat_sha256'] or req['reference_revision'].split(':')[-1]!=binding['reference_sha256']:raise ValueError('Request revision does not match reviewed inputs')
        expected_forms=binding.get('site_forms') or {s['label']:binding['ion'] for s in sites}
        if k!=[Fraction(str(v)) for v in binding['k']] or {s['label']:s['form_factor'] for s in sites}!=expected_forms:raise ValueError('Request magnetic assumptions differ from reviewed inputs')
        if upper>binding['maximum_moment']:raise ValueError('Moment bound exceeds reviewed policy')
    excluded=req.get('excluded_parent_space_groups',[])
    if not isinstance(excluded,list) or any(isinstance(v,bool) or not isinstance(v,int) or not 1<=v<=230 for v in excluded):
        raise ValueError('Explicit excluded space-group policy must contain valid group numbers')
    catalog=json.loads(Path(req['catalog']).read_text());preflight=json.loads(Path(req['preflight']).read_text())
    ids=[c['id'] for c in catalog['candidates']]
    if not ids or len(ids)!=len(set(ids)) or any(not re.fullmatch(r'[A-Za-z0-9:_-]{1,160}',i) for i in ids):raise ValueError('Invalid or duplicate candidate identifiers')
    if len({v.replace(':','_') for v in ids})!=len(ids):raise ValueError('Candidate identifiers collide after directory normalization')
    pre_ids=[c['id'] for c in preflight]
    if len(pre_ids)!=len(set(pre_ids)) or set(pre_ids)!=set(ids):raise ValueError('Candidate preflight does not cover exactly this catalogue')
    if [Fraction(str(v)) for v in catalog['k']]!=k:raise ValueError('Catalogue propagation vector differs from request')
    if req.get('observation'):
        if req.get('cumulative_policy')!='replace':raise ValueError('Cumulative snapshots must replace the observation likelihood')
        for key in ('observation','observation_source'):
            if file_hash(req[key])!=req[key+'_sha256']:raise ValueError('Changed stream observation')
        if req['source_revision'].split(':')[-1]!=req['observation_source_sha256']:raise ValueError('Observation source revision mismatch')
        from .array_limits import validate_npz
        validate_npz(req['observation'], observation=True)
        import numpy as np
        with np.load(req['observation'],allow_pickle=False) as a:
            if set(a.files)!={'x','y','sigma','mask'}:raise ValueError('Invalid observation arrays')
            x,y,s,m=(a[k] for k in ('x','y','sigma','mask'))
            if x.ndim!=1 or len(x)<2 or len(x)>1000000 or any(v.shape!=x.shape for v in (y,s,m)):raise ValueError('Invalid observation dimensions')
            if not all(np.isfinite(v).all() for v in (x,y,s)) or np.any(np.diff(x)<=0) or m.dtype!=bool or np.any(s[~m]<=0):raise ValueError('Invalid observation grid/errors/mask')
    return req


def verify_result(directory,*,binding=None,complete=False):
    root=Path(directory).resolve();request=root/'request.json';req=json.loads(request.read_text());result=json.loads((root/'result.json').read_text())
    if result.get('request_sha256')!=file_hash(request):raise ValueError('Result request checksum mismatch')
    if complete and result.get('status')!='complete':raise ValueError('Only completed results can be restored')
    for key in ('source_revision','reference_revision'):
        if result.get(key)!=req.get(key):raise ValueError('Result scientific revision mismatch')
    if result.get('forecast_enabled') is not False or result.get('exclusion_authorized') is not False:raise ValueError('This experimental adapter cannot authorize forecasts or exclusions')
    validate_request(req,binding)
    return req,result


def seal_posterior(directory):
    root=Path(directory)
    names = ['kernel.json','kernel.npz','posterior.json','posterior.npz','inference-request.json']
    if (root/'posterior-gpx.json').is_file() and (root/'posterior-maximum.gpx').is_file():
        names += ['posterior-gpx.json', 'posterior-maximum.gpx']
    return {name:file_hash(root/name) for name in names}


def verify_posterior(path,expected=None):
    path=Path(path);root=path.parent
    if not expected:raise ValueError('Posterior has no artifact checksums; rerun or explicitly verify the legacy archive')
    required={'kernel.json','kernel.npz','posterior.json','posterior.npz','inference-request.json'}
    exported = {'posterior-gpx.json', 'posterior-maximum.gpx'}
    if set(expected) not in (required, required | exported):raise ValueError('Incomplete posterior artifact receipt')
    for name,checksum in expected.items():
        if not (root/name).resolve().is_relative_to(root.resolve()) or file_hash(root/name)!=checksum:raise ValueError('Posterior artifact changed: '+name)
    return True
