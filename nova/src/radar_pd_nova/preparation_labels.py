"""Bounded descriptive labels from the library's captured discovery scope.

These labels describe the declared model; they never authorize scientific
reuse or replace the native parent/site/operator checks.
"""
import json

from gpx_recipe.preparation.library import confined
from gpx_recipe.symmetry.isodistort_forms import _method1_catalog_group


def scope_labels(library):
    name=library.document['specification'].get('scope')
    if not name:return {}
    path=confined(library.root,name)
    if path.stat().st_size>4*1024*1024:raise ValueError('Preparation scope display exceeds its metadata budget')
    path=library.verify_assets([name])[name]
    document=json.loads(path.read_text())
    if not isinstance(document,dict) or document.get('schema')!='radar-preparation-declared-scope/v1':return {}
    labels={}
    searches=document.get('searches',[])
    if not isinstance(searches,list):return {}
    for search in searches:
        if not isinstance(search,dict):continue
        selected=search.get('selected_candidates',[]);vector=search.get('k');coverage=search.get('coverage')
        if (search.get('kind') not in {'structural','magnetic'} or not isinstance(selected,list)
                or any(not isinstance(v,str) for v in selected) or not isinstance(vector,list)
                or len(vector)!=3 or any(not isinstance(v,str) for v in vector)
                or not isinstance(coverage,dict) or not isinstance(coverage.get('candidates'),list)):continue
        selected=set(selected)
        for row in coverage['candidates']:
            if not isinstance(row,dict) or not isinstance(row.get('id'),str) or row['id'] not in selected:continue
            key=(search.get('kind'),row['id'])
            item=dict(k=vector,label=str(row.get('label',''))[:2000])
            values=labels.setdefault(key,[])
            if item not in values:values.append(item)
    return labels


def entry_label(model, labels):
    settings=model.get('settings',{});candidate=settings.get('candidate',{})
    identifier=candidate.get('id','');kind=model['kind']
    label=candidate.get('magnetic_space_group') or candidate.get('irrep') or identifier or 'Parent and profile'
    vector=settings.get('k',[])
    choices=labels.get((kind,identifier),[])
    if vector:choices=[row for row in choices if row['k']==vector]
    if len(choices)==1:
        if not vector:vector=choices[0]['k']
        if kind=='structural':
            group=_method1_catalog_group(choices[0]['label'])
            if group:label=group['catalog_child_space_group_symbol']+' · '+label
    return dict(model=label,k=', '.join(vector),candidate=identifier,
        coordinates=(str(len(candidate.get('modes',[])))+' native displacement coordinates') if kind=='structural' else '',
        basis=str(candidate.get('transformation_matrix',[])),origin=str(candidate.get('origin_shift',[])))
