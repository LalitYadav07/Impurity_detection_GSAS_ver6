"""Closed, relocatable symmetry-job contracts; standard library only.

The native scientific image intentionally does not inherit NOVA's UI dependency
stack. Every accepted object is normalized before hashing or execution.
"""
from __future__ import annotations
from fractions import Fraction
import hashlib,json,math,re
from pathlib import PurePosixPath


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def fields(value,required,defaults=None):
    defaults=defaults or {}
    if not isinstance(value,dict):raise ValueError('A contract object must be a JSON object')
    if set(value)-set(required)-set(defaults):raise ValueError('Unknown fields in closed contract: '+str(sorted(set(value)-set(required)-set(defaults))))
    if set(required)-set(value):raise ValueError('Missing contract fields: '+str(sorted(set(required)-set(value))))
    return {**defaults,**value}


def text(value,*,pattern=None,maximum=200):
    if not isinstance(value,str) or not 1<=len(value)<=maximum or any(ord(c)<32 for c in value) or (pattern and not re.fullmatch(pattern,value)):
        raise ValueError('Invalid contract string')
    return value


def number(value,low,high,*,positive=False,integer=False):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low<=value<=high or (positive and value==low) or (integer and not isinstance(value,int)):
        raise ValueError('Nonfinite, mistyped or out-of-budget contract number')
    return int(value) if integer else float(value)


def boolean(value):
    if type(value) is not bool:raise ValueError('A contract flag must be a JSON boolean')
    return value


def choices(value,allowed):
    if value not in allowed:raise ValueError('Unsupported declared choice: '+str(value))
    return value


def vector(value):
    if not isinstance(value,list) or len(value)!=3:raise ValueError('Three finite components are required')
    return [number(x,-1e8,1e8) for x in value]


def asset(value):
    value=fields(value,['path','sha256']);path=text(value['path'],maximum=500);p=PurePosixPath(path)
    if p.is_absolute() or any(c in path for c in ['\\',':','\x00']) or '..' in p.parts or p.as_posix()!=path:
        raise ValueError('Use a normalized job-relative asset path')
    if p.parts[0] in {'job.json','worker.py','result.json','results','runtime'}:raise ValueError('Asset path is reserved for the job runner')
    return dict(path=path,sha256=text(value['sha256'],pattern=r'[a-f0-9]{64}'))


def validate_candidate(value):
    c=fields(value,['id','transformation_matrix','origin_shift'],dict(magnetic_space_group=None,bns_number=None,bns_lattice='',child=None,modes=[],irrep=None))
    c['id']=text(c['id'],pattern=r'[a-zA-Z0-9:_-]{1,120}')
    t=c['transformation_matrix']
    if not isinstance(t,list) or len(t)!=3:raise ValueError('A candidate has three basis rows')
    t=c['transformation_matrix']=[vector(row) for row in t];c['origin_shift']=vector(c['origin_shift'])
    det=(t[0][0]*(t[1][1]*t[2][2]-t[1][2]*t[2][1])-t[0][1]*(t[1][0]*t[2][2]-t[1][2]*t[2][0])+t[0][2]*(t[1][0]*t[2][1]-t[1][1]*t[2][0]))
    if abs(det)<1e-10:raise ValueError('A candidate basis cannot be singular')
    if c['magnetic_space_group'] is not None:c['magnetic_space_group']=text(c['magnetic_space_group'],pattern=r"[A-Za-z0-9_/'\- ]+")
    if c['bns_lattice']:text(c['bns_lattice'],pattern=r'[A-Za-z]+')
    if c['bns_number'] is not None:c['bns_number']=text(c['bns_number'],pattern=r'[1-9][0-9]{0,2}\.[1-9][0-9]{0,3}')
    if c['child'] is not None:c['child']=asset(c['child'])
    if not isinstance(c['modes'],list) or any(not isinstance(x,str) for x in c['modes']) or len(c['modes'])!=len(set(c['modes'])):raise ValueError('Native mode IDs must be unique')
    c['modes']=[text(x,maximum=300) for x in c['modes']]
    if c['irrep'] is not None:c['irrep']=text(c['irrep'])
    return c


def seed(value):
    defaults=dict(axis='x',magnitude=1.,parent_direction=None,mode_values={})
    if isinstance(value,dict) and 'candidate_ids' in value:defaults['candidate_ids']=None
    if isinstance(value,dict) and 'site_signs' in value:defaults['site_signs']=None
    s=fields(value,['id'],defaults)
    if 'site_signs' in s:
        if not isinstance(s['site_signs'],dict) or not s['site_signs']:raise ValueError('Parent-site signs need an explicit nonempty mapping')
        s['site_signs']={text(k,maximum=100):number(v,-1,1,integer=True) for k,v in s['site_signs'].items()}
        if any(v not in (-1,1) for v in s['site_signs'].values()):raise ValueError('Parent-site signs must be +1 or -1')
    if 'candidate_ids' in s:
        ids=s['candidate_ids']
        if not isinstance(ids,list) or not ids or len(ids)!=len(set(ids)):raise ValueError('A scoped start needs unique candidate IDs')
        s['candidate_ids']=[text(v,pattern=r'[a-zA-Z0-9:_-]{1,120}') for v in ids]
    s['id']=text(s['id'],pattern=r'[a-zA-Z0-9_-]{1,80}');s['axis']=choices(s['axis'],['x','y','z']);s['magnitude']=number(s['magnitude'],0,20,positive=True)
    if s['parent_direction'] is not None:
        s['parent_direction']=vector(s['parent_direction'])
        if sum(v*v for v in s['parent_direction'])<1e-20:raise ValueError('A parent direction must be nonzero')
    if not isinstance(s['mode_values'],dict):raise ValueError('Native mode starts need a keyed object')
    s['mode_values']={text(k,maximum=300):number(v,-2,2) for k,v in s['mode_values'].items()}
    return s


def policy(value):
    defaults=dict(site_policy='all',refine_cell=False,refine_background=False,refine_histogram_scale=False,maximum_cell_relative_change=.02,maximum_angle_change_degrees=3.)
    if isinstance(value,dict) and 'magnetic_d_min' in value:defaults['magnetic_d_min']=None
    p=fields(value,['cycles','timeout_seconds','maximum_trials','maximum_cell_multiplier','maximum_moment','fit_band_percent','maximum_mode','adequate_rwp_percent'],defaults)
    if 'magnetic_d_min' in p:p['magnetic_d_min']=number(p['magnetic_d_min'],0,10)
    for k,lo,hi in [('cycles',1,100),('timeout_seconds',1,3600),('maximum_trials',1,2000),('maximum_cell_multiplier',1,64)]:p[k]=number(p[k],lo,hi,integer=True)
    for k,hi in [('maximum_moment',20),('maximum_mode',2),('adequate_rwp_percent',100),('maximum_cell_relative_change',.1),('maximum_angle_change_degrees',10)]:p[k]=number(p[k],0,hi,positive=True)
    p['fit_band_percent']=number(p['fit_band_percent'],0,10);p['site_policy']=choices(p['site_policy'],['all','any'])
    for k in ['refine_cell','refine_background','refine_histogram_scale']:p[k]=boolean(p[k])
    return p


def validate_protocol(value):
    defaults=dict(schema='radar-symmetry-protocol/v1',sites=[])
    if isinstance(value,dict) and 'acquisition_id' in value:defaults['acquisition_id']=None
    p=fields(value,['kind','source','parent_phase','histogram','parent_setting_sha256','magnetic_function_sha256','implementation_sha256','catalog','k','candidates','seeds','policy'],defaults)
    if 'acquisition_id' in p:p['acquisition_id']=text(p['acquisition_id'],maximum=300)
    choices(p['schema'],['radar-symmetry-protocol/v1']);choices(p['kind'],['magnetic','structural'])
    for name in ['source','catalog']:p[name]=asset(p[name])
    for name in ['parent_phase','histogram']:p[name]=text(p[name])
    for name in ['parent_setting_sha256','magnetic_function_sha256','implementation_sha256']:p[name]=text(p[name],pattern=r'[a-f0-9]{64}')
    if not isinstance(p['k'],list) or len(p['k'])!=3 or any(not isinstance(x,str) for x in p['k']):raise ValueError('A single k-vector needs three exact rational strings')
    if any(Fraction(v).denominator>96 for v in p['k']):raise ValueError('Commensurate denominator budget exceeded')
    p['k']=[str(Fraction(v)) for v in p['k']]
    if not isinstance(p['candidates'],list) or not 1<=len(p['candidates'])<=500:raise ValueError('Invalid finite candidate set')
    if not isinstance(p['seeds'],list) or not 1<=len(p['seeds'])<=64:raise ValueError('Invalid finite start set')
    p['candidates']=[validate_candidate(c) for c in p['candidates']];p['seeds']=[seed(s) for s in p['seeds']];p['policy']=policy(p['policy'])
    if len({c['id'] for c in p['candidates']})!=len(p['candidates']) or len({s['id'] for s in p['seeds']})!=len(p['seeds']):raise ValueError('Candidate and seed IDs must be unique')
    candidate_ids={c['id'] for c in p['candidates']}
    if any(set(s.get('candidate_ids',candidate_ids))-candidate_ids for s in p['seeds']):raise ValueError('A start references an undiscovered candidate')
    if any(not any(c['id'] in s.get('candidate_ids',candidate_ids) for s in p['seeds']) for c in p['candidates']):raise ValueError('Every declared candidate needs a finite start')
    if sum(1 for _ in iter_trials(p))>p['policy']['maximum_trials']:raise ValueError('Finite trial budget exceeded')
    if p['source']['path']==p['catalog']['path']:raise ValueError('Source and catalog are distinct assets')
    if not isinstance(p['sites'],list):raise ValueError('Magnetic sites must be a list')
    sites=[]
    for s in p['sites']:
        s=fields(s,['label','form_factor','lande_g']);s['label']=text(s['label'],maximum=100);s['form_factor']=text(s['form_factor'],pattern=r'[A-Z][a-z]?(?:[+-][1-9])?');s['lande_g']=number(s['lande_g'],0,10,positive=True);sites.append(s)
    p['sites']=sites
    if p['kind']=='magnetic':
        if not sites or len({s['label'] for s in sites})!=len(sites):raise ValueError('Select unique magnetic sites and form factors')
        if any(not c['magnetic_space_group'] or c['child'] or c['modes'] for c in p['candidates']):raise ValueError('Magnetic candidates require native MSG records')
        if any(p['policy'][k] for k in ['refine_cell','refine_background','refine_histogram_scale']):raise ValueError('Magnetic screening holds the nuclear/nuisance state fixed')
        if any(s['mode_values'] or s['magnitude']>p['policy']['maximum_moment'] for s in p['seeds']):raise ValueError('Invalid magnetic start or moment ceiling')
        if any(set(s.get('site_signs',{}))-{site['label'] for site in sites} for s in p['seeds']):raise ValueError('A start gives a sign to an unselected magnetic site')
        for form in {s['form_factor'] for s in sites}:
            if len({s['lande_g'] for s in sites if s['form_factor']==form})!=1:raise ValueError('Native Lande g is shared by each form-factor species')
    else:
        if any(s.get('site_signs') for s in p['seeds']):raise ValueError('Parent-site moment signs are magnetic-only')
        if sites or any(c['magnetic_space_group'] or not c['child'] or not c['modes'] for c in p['candidates']):raise ValueError('Structural candidates need authoritative displacement-mode CIFs')
        if any(abs(v)>p['policy']['maximum_mode'] for s in p['seeds'] for v in s['mode_values'].values()):raise ValueError('A displacement-mode start exceeds its bound')
    return p


def trial_id(protocol,candidate,seed):return digest(dict(protocol=digest(protocol),candidate=candidate,seed=seed))[:24]


def iter_trials(protocol):
    """Different OPDs can have different native mode sets and finite starts."""
    for candidate in protocol['candidates']:
        for start in protocol['seeds']:
            if 'candidate_ids' not in start or candidate['id'] in start['candidate_ids']:yield candidate,start
