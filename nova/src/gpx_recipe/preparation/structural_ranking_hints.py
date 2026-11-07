"""Portable trained structural rankers; an attachment never authorizes pruning.

The UI verifies the data inventory without loading Torch or numerical arrays.
Inference belongs in a numerical worker, with the caller supplying the checked
physical parent/profile identities. Candidate routes retain exact constructions.
"""
import copy
import json
from pathlib import Path
import re

from ..symmetry.compiler import file_sha
from ..symmetry.contracts import validate_candidate
from .library import confined
from .reference import validate_grid

SCHEMA='radar-prepared-structural-ranking-input/v1'
PREFIX='_prepared_structural/ranking/'
LIMITS={'parent':10*1024**2,'instrument':1024**2,'model':2*1024**2,'weights':32*1024**2}
SUFFIXES={'parent':'.cif','instrument':'.instprm','model':'.json','weights':'.npz'}
MAX_BYTES=64*1024**2
MAX_MODELS=128


def _hash(value):
    return isinstance(value,str) and re.fullmatch('[a-f0-9]{64}',value) is not None


def validate(value,protocol):
    if (not isinstance(value,dict) or set(value)!={'schema','entries','eligible_entries','omitted_entries'}
            or value['schema']!=SCHEMA or protocol['kind']!='structural'
            or not isinstance(value['entries'],list) or len(value['entries'])>MAX_MODELS
            or any(type(value[k]) is not int or not 0<=value[k]<=5000 for k in ('eligible_entries','omitted_entries'))
            or value['eligible_entries']!=len(value['entries'])+value['omitted_entries']):
        raise ValueError('Invalid bounded structural ranking inventory')
    identifiers={c['id'] for c in protocol['candidates']};seen=set()
    for row in value['entries']:
        if (not isinstance(row,dict) or set(row)!={'entry','bank','grid','files','routes','entry_sha256'}
                or not isinstance(row['entry'],str) or not re.fullmatch('[a-f0-9]{24}',row['entry'])
                or row['entry'] in seen or not _hash(row['entry_sha256'])
                or not isinstance(row['bank'],str) or not 1<=len(row['bank'])<=100
                or not isinstance(row['files'],dict) or set(row['files'])!=set(LIMITS)
                or not isinstance(row['routes'],dict) or not row['routes']
                or any(not isinstance(k,str) or not k or len(k)>300 or not isinstance(v,str) or v not in identifiers
                       for k,v in row['routes'].items())
                or len(set(row['routes'].values()))!=len(row['routes'])):
            raise ValueError('Structural ranking routes need exact distinct native constructions')
        if not isinstance(row['grid'],dict) or set(row['grid'])!={'kind','lower','upper','bins','bank'}:
            raise ValueError('Structural ranking needs its declared bank grid')
        validate_grid(**row['grid'])
        for role,asset in row['files'].items():
            if (not isinstance(asset,dict) or set(asset)!={'path','sha256'} or not _hash(asset['sha256'])
                    or asset['path']!=PREFIX+asset['sha256']+SUFFIXES[role]):
                raise ValueError('Invalid structural ranking asset identity')
        seen.add(row['entry'])
    return value


def paths(document):
    return sorted({a['path'] for r in document.get('ranking_models',{}).get('entries',[])
                   for a in r['files'].values()})


def _metadata(path):
    if path.is_symlink() or not 0<path.stat().st_size<=LIMITS['model']:
        raise ValueError('Oversized or changed structural model metadata')
    d=json.loads(path.read_text())
    from .structural_attention_weights import SCHEMA as THREE_CURVE_SCHEMA,ARCHITECTURE as THREE_CURVE_ARCH
    if (not isinstance(d,dict) or d.get('schema') not in {'radar-structural-ranking-model/v4',THREE_CURVE_SCHEMA}
            or (d.get('schema')==THREE_CURVE_SCHEMA and d.get('architecture')!=THREE_CURVE_ARCH)
            or not isinstance(d.get('candidates'),list) or not 1<=len(d['candidates'])<=5000
            or any(not _hash(d.get(k)) for k in ('parent_sha256','instrument_sha256','weights_sha256'))):
        raise ValueError('Structural ranking requires typed candidate and physical-context metadata')
    candidates=[validate_candidate(c) for c in d['candidates']]
    if len({c['id'] for c in candidates})!=len(candidates):
        raise ValueError('Structural model contains duplicate native construction identities')
    return d,candidates


def checked_files(root,row,protocol):
    from .structural_hints import candidate_identity
    result={}
    for role,asset in row['files'].items():
        path=confined(root,asset['path'])
        if path.is_symlink() or not 0<path.stat().st_size<=LIMITS[role] or file_sha(path)!=asset['sha256']:
            raise ValueError('Structural ranking data changed or exceed the declared budget')
        result[role]=path
    d,candidates=_metadata(result['model'])
    if any(d[key]!=row['files'][role]['sha256'] for key,role in (
            ('parent_sha256','parent'),('instrument_sha256','instrument'),('weights_sha256','weights'))):
        raise ValueError('Structural ranker belongs to another parent, profile or weight file')
    current={c['id']:candidate_identity(c) for c in protocol['candidates']}
    prepared={c['id']:candidate_identity(c) for c in candidates}
    if any(prepared.get(key)!=current[target] for key,target in row['routes'].items()):
        raise ValueError('Structural ranking route changed a native construction')
    return result


def capture(library,document,protocol,root):
    from .hints import _copy
    from .structural_hints import candidate_identity
    from .operations import portable_entry_artifacts
    current={candidate_identity(c):c['id'] for c in protocol['candidates']}
    rows=[];eligible=omitted=total=0;copied=set()
    inputs={a['path']:a['sha256'] for a in document['inputs']}
    for row in sorted(document['entries'],key=lambda r:r['id']):
        model=row['model']
        if model['kind']!='structural-training' or row['state']!='prepared':continue
        # File placement is library-owned; use the closed relative inventory
        # without assuming the generation attempt's directory layout.
        named={Path(a['path']).name:a for a in row['files']}
        if not {'entry.json','model.json','weights.npz','training.json'}<=set(named):continue
        assets=library.verify_assets([document['specification']['source'],model['instrument']]+
                                    [named[n]['path'] for n in ('entry.json','model.json','weights.npz')])
        d,candidates=_metadata(assets[named['model.json']['path']])
        routes={c['id']:current[candidate_identity(c)] for c in candidates if candidate_identity(c) in current}
        if not routes:continue
        eligible+=1
        entry=assets[named['entry.json']['path']]
        if entry.stat().st_size>LIMITS['model']:raise ValueError('Structural training entry metadata is oversized')
        manifest=json.loads(entry.read_text())
        if (not isinstance(manifest,dict) or manifest.get('kind')!='structural-training'
                or manifest.get('schema')!='radar-experiment-preparation-entry/v1'
                or manifest.get('settings')!=model['settings'] or manifest.get('grid')!=model['grid']
                or manifest.get('input_sha256')!=[inputs[document['specification']['source']],inputs[model['instrument']]]
                or manifest.get('observations_included') is not False or manifest.get('scientific_result') is not False):
            raise ValueError('Structural training entry differs from its captured physical context')
        portable_entry_artifacts('structural-training',manifest)
        sources={'parent':(document['specification']['source'],inputs[document['specification']['source']]),
                 'instrument':(model['instrument'],inputs[model['instrument']]),
                 'model':(named['model.json']['path'],named['model.json']['sha256']),
                 'weights':(named['weights.npz']['path'],named['weights.npz']['sha256'])}
        if (d['parent_sha256']!=sources['parent'][1] or d['instrument_sha256']!=sources['instrument'][1]
                or d['weights_sha256']!=sources['weights'][1]
                or manifest['artifacts'].get('model/model.json')!=sources['model'][1]
                or manifest['artifacts'].get('model/weights.npz')!=sources['weights'][1]):
            raise ValueError('Structural training artifacts changed after publication')
        if any(not 0<assets[name].stat().st_size<=LIMITS[role] for role,(name,_) in sources.items()):
            raise ValueError('A structural ranking attachment exceeds its portable budget')
        required=sum(assets[name].stat().st_size for role,(name,sha) in sources.items()
                     if PREFIX+sha+SUFFIXES[role] not in copied)
        if total+required>MAX_BYTES or len(rows)>=MAX_MODELS:omitted+=1;continue
        files={}
        for role,(name,sha) in sources.items():
            relative=PREFIX+sha+SUFFIXES[role]
            _copy(assets[name],confined(root,relative,exists=False),sha)
            files[role]=dict(path=relative,sha256=sha)
            if relative not in copied:total+=assets[name].stat().st_size;copied.add(relative)
        rows.append(dict(entry=row['id'],bank=model['bank'],grid=copy.deepcopy(model['grid']),
                         routes=routes,files=files,entry_sha256=named['entry.json']['sha256']))
    return validate(dict(schema=SCHEMA,entries=rows,eligible_entries=eligible,omitted_entries=omitted),protocol)


def restore(document,protocol,source_root,target_root):
    from .hints import _copy
    inventory=document.get('ranking_models')
    if inventory is None:return dict(attached=0,discarded_entries=[],candidate_pruning_authorized=False)
    validate(inventory,protocol);accepted=[];discarded=[];total=0;copied=set()
    for row in inventory['entries']:
        try:
            files=checked_files(source_root,row,protocol)
            required=sum(p.stat().st_size for role,p in files.items() if row['files'][role]['path'] not in copied)
            if total+required>MAX_BYTES:raise ValueError('Structural ranking transfer exceeds its budget')
            for role,path in files.items():
                asset=row['files'][role];target=confined(target_root,asset['path'],exists=False)
                if path!=target:_copy(path,target,asset['sha256'])
                if asset['path'] not in copied:total+=path.stat().st_size;copied.add(asset['path'])
            accepted.append(row['entry'])
        except (OSError,ValueError,EOFError,KeyError,TypeError):discarded.append(row['entry'])
    return dict(attached=len(accepted),discarded_entries=discarded,candidate_pruning_authorized=False)


def rank(root,protocol,q,intensity,*,background,parent_sha256,instrument_sha256,entry_id,valid=None,reference=None):
    """Worker-only advisory order; native callers must verify current context.

    Every declared candidate is returned, even with damaged data, missing Q
    intervals, unsupported patterns, or a model trained for a larger k scope.
    The caller selects an exact checked entry: one instrument file may contain
    several banks, so its checksum alone cannot select the correct model.
    Scores are neither probabilities of a physical structure nor fit adequacy.
    """
    from .structural_hints import read
    order=[c['id'] for c in protocol['candidates']]
    fallback=dict(candidate_order=order,ranking_available=False,candidate_pruning_authorized=False,
                  native_refinement_required=True,complete_native_search_required=True,discarded_entries=[])
    try:document=read(root,protocol)
    except (OSError,ValueError,EOFError):return dict(fallback,reason='Preparation snapshot is unavailable')
    if document is None:return dict(fallback,reason='No prepared ranking model attached')
    for row in document.get('ranking_models',{}).get('entries',[]):
        if (row['entry']!=entry_id or row['files']['parent']['sha256']!=parent_sha256 or
                row['files']['instrument']['sha256']!=instrument_sha256):continue
        try:
            files=checked_files(root,row,protocol)
            from .structural_ranker import Ranker
            # Portable names are content-addressed, not model.json/weights.npz.
            model=Ranker(files['model'].parent,metadata_name=files['model'].name,weights_name=files['weights'].name)
            result=model.rank(q,intensity,background=background,valid=valid,reference=reference,
                              parent_sha256=parent_sha256,instrument_sha256=instrument_sha256)
            if not result['ranking_available']:
                fallback.update(support=result['support'],reason=result['interpretation']);continue
            routes=row['routes'];families=[]
            for family in result['families']:
                found=list(dict.fromkeys(routes[key] for key in family['candidate_ids'] if key in routes))
                if found:families.append(dict(family,candidate_ids=found))
            ranked=list(dict.fromkeys(key for family in families for key in family['candidate_ids']))
            return dict(result,families=families,candidate_order=ranked+[key for key in order if key not in ranked],
                        entry=row['entry'],bank=row['bank'],snapshot_sha256=document['snapshot_sha256'],
                        complete_native_search_required=True,discarded_entries=fallback['discarded_entries'])
        except (ImportError,OSError,ValueError,EOFError,KeyError,TypeError,IndexError) as exc:
            fallback['discarded_entries'].append(dict(entry=row['entry'],error_type=type(exc).__name__))
    return fallback
