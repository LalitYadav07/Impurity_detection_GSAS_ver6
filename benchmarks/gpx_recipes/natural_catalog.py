"""Human-style wording and real model-plan revisions, with independent oracles.

These requests do not use the literal Free: grammar. An oracle checks them
before any model-generated plan may reach the native solver.
"""
import copy
import json
from pathlib import Path


def generate(root):
    root=Path(root);rows=json.loads((root/'inventory.json').read_text());result=[]
    def stage(free,cycles=4,**kw):return dict(id='s',title='Requested refinement',free=free,cycles=cycles,constraints=[],**kw)
    def add(key,c,category,text,stages,previous_case=None):
        stages=copy.deepcopy(stages)
        for i,s in enumerate(stages,1):s.update(id=f'pass_{i}',title=f'Requested pass {i}')
        uid=f'N{len(result)+1:03d}'
        oracle=dict(schema='radar-gpx-recipe/v1',model_signature=c['model_signature'],request=text,assumptions=[],questions=[],unsupported=[],stages=stages)
        value=dict(id=uid,project=key,category=category,scenario=text.split('\n')[0][:120],request=text,expectation='execute',oracle=oracle,partition='development')
        if previous_case:value['previous_case']=previous_case
        result.append(value);return uid,oracle
    # All chosen projects belong to development; no use of held-out projects.
    selected={'ipts-36501_pg3_64036','ipts-38280_pg3_64206','ipts-38059_pg3_64142','ipts-39883_pg3_64365','ipts-39838_pg3_64320','nom-21886-138502-b5','aps-ciftutorial-nxpbso4','aps-fresh-joint-nac-caf2'}
    for item in rows:
        if item['status']!='inspected' or item['id'] not in selected:continue
        key=item['id'];c=json.loads((root/'inputs'/(key+'.context.json')).read_text());hist=c['histograms'];h=hist[0]
        eligible=[(p,a) for p in c['phases'] for a in p['atoms'] if a['adp_type']=='I' and a['position_dof']>0]
        bg=[dict(kind='background',histogram=h['key'])]
        if eligible:
            p,a=eligible[0];site=dict(kind='atom',phase=p['key'],atom=a['label'],flags='XU');scale=dict(kind='phase_scale',phase=p['key'],histogram=h['key'])
            text=f"Please polish this without changing the structural model. Use exactly two stages. In the first stage, refine only the current polynomial background of histogram {h['key']} ({h['name']!r}) and the HAP scale of phase {p['key']} ({p['name']!r}) in that histogram, for at most 4 cycles. In stage two fix both of those and refine only {p['key']}:{a['label']} positions and Uiso jointly, for at most 5 cycles. Hold every other parameter fixed at each stage. Keep all background settings, calibration, constraints and observations unchanged; do not infer occupancy refinement."
            base,oracle=add(key,c,'natural_atom_sequence',text,[stage(bg+[scale]),stage([site],5)])
            revised=copy.deepcopy(oracle['stages']);revised[0]['cycles']=3;revised[1]['free'][0]['flags']='X';revised[1]['cycles']=6
            add(key,c,'conversation_revision',f"Change the plan you just proposed: the first stage gets only 3 cycles. In the second stage, refine positions only, fixing that atom's Uiso, and allow 6 cycles. Keep all other choices and stage ordering exactly as in your last plan. Do not add stages.",revised,previous_case=base)
        # Normalization in every histogram is independent; shared fractions are
        # deliberately not claimed. Only the requested scale family is active.
        free=[]
        for hh in hist:
            free.append(dict(kind='histogram_scale',histogram=hh['key']))
            pp=[p for p in c['phases'] if hh['name'] in p['histograms']]
            if len(pp)>1:free.extend(dict(kind='phase_scale',phase=p['key'],histogram=hh['key']) for p in pp)
        text="Use one stage with a limit of 5 cycles. I want normalized HAP phase scales summing to one, independently in every existing histogram, and the global histogram scales refined. A single linked phase has its HAP scale fixed at one. Preserve the starting intensity products when normalizing. Refine only these scales; keep backgrounds, cells, atoms, texture and instruments fixed. Do not constrain the fractions in different histograms to equal each other."
        add(key,c,'natural_normalization',text,[stage(free,5,phase_normalization=[hh['key'] for hh in hist])])
        if len(hist)>1:
            a,b=hist[:2]
            text=f"Please do exactly two stages, each with at most 3 cycles. First refine the existing polynomial background only in {a['key']} ({a['name']!r}). Then fix that background and refine the existing polynomial background only in {b['key']} ({b['name']!r}). Every other histogram and parameter stays fixed throughout. Do not change either background's function, coefficient count or support points."
            add(key,c,'natural_histogram_switch',text,[stage([dict(kind='background',histogram=a['key'])],3),stage([dict(kind='background',histogram=b['key'])],3)])
    from gpx_recipe.contracts import validate_plan
    for case in result:validate_plan(case['oracle'],json.loads((root/'inputs'/(case['project']+'.context.json')).read_text()))
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('campaign');a=p.parse_args();root=Path(a.campaign);cases=generate(root)
    catalog=json.loads((root/'catalog.json').read_text());catalog=[c for c in catalog if not c['id'].startswith('N')]+cases
    (root/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n');print(json.dumps({'natural_cases':len(cases),'all_cases':len(catalog)}))
