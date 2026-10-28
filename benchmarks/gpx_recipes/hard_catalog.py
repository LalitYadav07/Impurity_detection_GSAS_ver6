"""Harder operation courses and revisions on independently inspected real GPXs."""
import copy
import json
from pathlib import Path


def generate(root):
    root=Path(root);inventory=json.loads((root/'inventory.json').read_text());cases=[]
    def stage(free,cycles=6,**edits):return dict(id='s',title='Requested stage',cycles=cycles,free=copy.deepcopy(free),constraints=[],**edits)
    def atom(p,a,flags):return dict(kind='atom',phase=p,atom=a,flags=flags)
    def label(t,c):
        p=next((p for p in c['phases'] if p['key']==t.get('phase')),None);h=next((h for h in c['histograms'] if h['key']==t.get('histogram')),None)
        if t['kind']=='atom':return f"{t['atom']} in phase {p['name']!r}: "+', '.join({'X':'symmetry-independent coordinates','U':'Uiso','F':'site occupancy'}[f] for f in t['flags'])
        value={'background':'existing polynomial background','phase_scale':'HAP phase scale','histogram_scale':'global histogram scale','cell':'unit cell','preferred_orientation':'preferred orientation'}[t['kind']]
        return value+(f" of phase {p['name']!r}" if p else '')+(f" for histogram {h['key']} ({h['name']!r})" if h else '')
    def request(stages,c):
        lines=['Use exactly '+str(len(stages))+' stages. Each numbered stage is a joint refinement of its complete free set; everything unlisted stays fixed. Preserve calibration, original observations, existing constraints and all settings not explicitly edited. Retain each new constraint and setting in later stages.']
        for i,s in enumerate(stages,1):
            line=f"Stage {i}: at most {s['cycles']} cycles. Free: "+'; '.join(label(t,c) for t in s['free'])+'.'
            if s.get('background'):
                b=s['background'];line+=f" Set {b['histogram']} background to {b['function']}, {b['coefficients']} coefficients; {b['fixed_points']} support points."
            if s.get('phase_normalization'):line+=' Normalize HAP phase scales to sum to one separately for '+', '.join(s['phase_normalization'])+'; preserve starting intensity products and refine the global scales. Single-phase HAP scales stay fixed at one.'
            for e in s.get('bounds',[]):line+=f" Set {e['phase']}:{e['atom']} {'occupancy' if e['parameter']=='F' else 'Uiso'} bounds [{e['lower']}, {e['upper']}]"+(f", starting at {e['value']}" if 'value' in e else '')+'.'
            for e in s.get('exclusions',[]):line+=f" For {e['histogram']}, {e['mode']} excluded regions {e['regions']} in {e['unit']} units."
            for e in s.get('fit_ranges',[]):line+=f" Set fit range of {e['histogram']} to [{e['lower']}, {e['upper']}] in {e['unit']} units."
            for e in s.get('texture',[]):line+=f" Set {e['phase']} / {e['histogram']} texture to {e['model']}"+(f" along hkl {e['direction']} with initial ratio {e.get('value',1.)}" if 'direction' in e else f" at order {e['order']}")+'.'
            for e in s['constraints']:
                if e['kind']=='uiso_equal':line+=' Constrain one common isotropic Uiso for '+', '.join(t['phase']+':'+t['atom'] for t in e['atoms'])+'. '+e['reason']+'.'
                else:line+=' Constrain '+' + '.join(str(t['coefficient'])+' * occupancy('+t['phase']+':'+t['atom']+')' for t in e['terms'])+' = '+str(e['total'])+'. '+e['reason']+'.'
            lines.append(line)
        return '\n'.join(lines)
    def add(key,c,category,title,stages,previous=None,text=None,expect='execute',partition='development'):
        stages=copy.deepcopy(stages)
        for i,s in enumerate(stages,1):s.update(id=f'stage_{i}',title=title+f' · stage {i}')
        text=text or request(stages,c)
        oracle=dict(schema='radar-gpx-recipe/v1',model_signature=c['model_signature'],request=text,assumptions=[],questions=[],unsupported=[],stages=stages) if expect=='execute' else None
        case=dict(id=f'H{len(cases)+1:03d}',project=key,category=category,scenario=title,request=text,expectation=expect,oracle=oracle,partition=partition)
        if previous:case['previous_plan']=copy.deepcopy(previous)
        cases.append(case);return oracle
    for item in inventory:
        if item['status']!='inspected':continue
        key=item['id'];c=json.loads((root/'inputs'/(key+'.context.json')).read_text());hist=c['histograms'];phases=c['phases'];bg=[dict(kind='background',histogram=h['key']) for h in hist];scales=[dict(kind='phase_scale',phase=p['key'],histogram=h['key']) for p in phases for h in hist if h['name'] in p['histograms']];cells=[dict(kind='cell',phase=p['key']) for p in phases if p['histograms']]
        eligible=[(p,a) for p in phases for a in p['atoms'] if a['adp_type']=='I' and a['position_dof']>0]
        isotropic=[(p,a) for p in phases for a in p['atoms'] if a['adp_type']=='I']
        partition='held_out' if key in {'ipts-38729_pg3_64315','ipts-38845_pg3_64339','nom-21886-138719-b5'} else 'development'
        # Long complete active sets; explicitly fix prior atoms/positions later.
        if eligible:
            p,a=eligible[0];q,b=eligible[-1]
            count=min(36,max(hist[0]['background']['coefficients']+2,8));edit=dict(histogram=hist[0]['key'],function='chebyschev-1',coefficients=count,fixed_points='preserve')
            stages=[stage(bg+scales,4,background=edit),stage(bg+scales+cells,6),stage(scales+cells+[atom(p['key'],a['label'],'X')],5),stage(scales+cells+[atom(p['key'],a['label'],'XU')],6),stage(scales+[atom(q['key'],b['label'],'U')],5),stage(bg+scales+cells,4)]
            original=add(key,c,'six_stage','Coordinates → joint XU → selected Uiso → cells',stages,partition=partition)
            revised=copy.deepcopy(stages);revised[3]['cycles']=9;revised[4]['free']=[atom(q['key'],b['label'],'U')];revised[5]['background']=dict(histogram=hist[0]['key'],function='cosine',coefficients=6,fixed_points='preserve')
            text=f"Revise only these decisions in the previous six-stage plan: stage 4 maximum cycles becomes 9; stage 5 frees only {q['key']}:{b['label']} Uiso, fixing all phase scales too; stage 6 changes {hist[0]['key']} background to cosine with 6 coefficients, preserving support points. Preserve every other stage, free/fixed choice, constraint, setting and cycle limit exactly."
            add(key,c,'revision','Three targeted edits; preserve the remaining recipe',revised,previous=original,text=text,partition=partition)
        # Independent normalization for every linked histogram, not a shared scale.
        if phases:
            normal_free=bg+[dict(kind='histogram_scale',histogram=h['key']) for h in hist]
            for h in hist:
                linked=[p for p in phases if h['name'] in p['histograms']]
                if len(linked)>1:normal_free.extend(dict(kind='phase_scale',phase=p['key'],histogram=h['key']) for p in linked)
            stages=[stage(normal_free,6,phase_normalization=[h['key'] for h in hist]),stage(normal_free+cells,6)]
            add(key,c,'multihist_normalization','Independent scale normalization and shared cells',stages,partition=partition)
        # Shared isotropic U among explicit existing atoms; no inferred chemistry.
        for p in phases:
            group=[a for a in p['atoms'] if a['adp_type']=='I' and a['element']=='O']
            if len(group)<2:continue
            selected=group[:5];free=[atom(p['key'],a['label'],'U') for a in selected];eq=dict(kind='uiso_equal',atoms=[dict(phase=p['key'],atom=a['label']) for a in selected],reason='Explicit user-selected shared Uiso model for this validation course; not automatically inferred from chemistry')
            stages=[stage(bg+scales+free,6),stage(bg+scales+free,6)]
            stages[0]['constraints']=[eq];stages[1]['constraints']=[copy.deepcopy(eq)]
            add(key,c,'uiso_equality','One shared oxygen Uiso carried across stages',stages,partition=partition)
            break
        # Real mixed sites: simultaneous occupancies followed by joint F/U, then fix them.
        for p in phases:
            groups=p.get('occupancy_groups',[])
            if not groups:continue
            names=groups[0];eq=dict(kind='occupancy_sum',terms=[dict(phase=p['key'],atom=n,coefficient=1.) for n in names],total=1.,reason='User explicitly requests total occupancy one for the existing mixed site')
            first=stage(bg+[atom(p['key'],n,'F') for n in names],6);first['constraints']=[eq]
            second=stage(bg+[atom(p['key'],n,'FU') for n in names if next(a['adp_type'] for a in p['atoms'] if a['label']==n)=='I'],6);second['constraints']=[copy.deepcopy(eq)]
            if len(second['free'])!=len(names)+len(bg):continue
            add(key,c,'mixed_site_sequence','Mixed-site F → FU → held occupancies', [first,second,stage(bg+scales,5)],partition=partition)
            break
        # Histogram-specific changes and phase-global atom parameters must stay distinct.
        if len(hist)>1:
            h0,h1=hist[:2];stages=[stage([dict(kind='background',histogram=h0['key'])],5,background=dict(histogram=h0['key'],function='chebyschev-1',coefficients=8,fixed_points='preserve')),stage([dict(kind='background',histogram=h1['key'])],5,background=dict(histogram=h1['key'],function='cosine',coefficients=10,fixed_points='preserve')),stage(bg+scales,6)]
            add(key,c,'histogram_scopes','Change one histogram, then another, then all',stages,partition=partition)
        # Explicit Uiso starts/bounds with coordinates carried/fixed separately.
        if eligible:
            p,a=eligible[-1];bound=dict(phase=p['key'],atom=a['label'],parameter='U',lower=.0001,upper=.1,value=.02)
            stages=[stage(bg+scales+[atom(p['key'],a['label'],'XU')],6,bounds=[bound]),stage(scales+[atom(p['key'],a['label'],'U')],4),stage(bg+scales,5)]
            add(key,c,'bounded_joint','Bounded joint XU followed by position/ADP freezing',stages,partition=partition)
        # MD texture, then switch to a symmetry-supported harmonic model, then fix.
        if phases and hist:
            p=phases[0];h=next((h for h in hist if h['name'] in p['histograms']),None)
            if h and p.get('texture_orders'):
                po=dict(kind='preferred_orientation',phase=p['key'],histogram=h['key']);target=dict(phase=p['key'],histogram=h['key']);order=p['texture_orders'][0]
                add(key,c,'texture_sequence','March-Dollase → spherical harmonics → fixed', [stage(bg+scales+[po],5,texture=[dict(**target,model='march-dollase',direction=[0,0,1],value=1.)]),stage(bg+scales+[po],5,texture=[dict(**target,model='spherical-harmonics',order=order)]),stage(bg+scales,5)],partition=partition)
        # Requests that must remain unresolved, never rescued by changing the science.
        if isotropic:
            p,a=isotropic[0]
            add(key,c,'negative','No silent anisotropic conversion',[],text=f"Convert {p['key']}:{a['label']} from isotropic to anisotropic ADP and refine all Uij. Do not keep it isotropic or substitute another atom.",expect='needs_input',partition=partition)
        if len(hist)>1 and eligible:
            p,a=eligible[0]
            add(key,c,'negative','No histogram-local fiction for one shared atom',[],text=f"Keep a single existing phase {p['name']!r}, but refine {a['label']} atomic coordinates only for {hist[0]['key']}. Its coordinates for {hist[1]['key']} must remain independent and fixed. Do not split or add phases and do not share that atom coordinate between histograms.",expect='needs_input',partition=partition)
    # Calibrated mixed-axis ranges and exclusion histories, independently derived
    # by native GSAS-II conversions from each real histogram's current limits.
    domains=json.loads((root/'axis-domains-all.json').read_text()) if (root/'axis-domains-all.json').exists() else {}
    for item in inventory:
        if item['status']!='inspected':continue
        key=item['id'];c=json.loads((root/'inputs'/(key+'.context.json')).read_text())
        partition='held_out' if key in {'ipts-38729_pg3_64315','ipts-38845_pg3_64339','nom-21886-138719-b5'} else 'development'
        bg=[dict(kind='background',histogram=h['key']) for h in c['histograms']]
        if key in domains:
            ranges=[];replace=[];additions=[];clear=[]
            for index,h in enumerate(c['histograms']):
                unit='d' if index%2==0 else 'Q';lo,hi=domains[key][h['key']][unit];span=hi-lo
                values=[round(lo+span*f,8) for f in [.04,.33,.34,.51,.52,.94]]
                ranges.append(dict(histogram=h['key'],unit=unit,lower=values[0],upper=values[-1]))
                replace.append(dict(histogram=h['key'],unit=unit,mode='replace',regions=[[values[1],values[2]]]))
                additions.append(dict(histogram=h['key'],unit=unit,mode='add',regions=[[values[3],values[4]]]))
                clear.append(dict(histogram=h['key'],unit=unit,mode='clear',regions=[]))
            add(key,c,'ranges_exclusions','Calibrated d/Q limits; replace → add → clear masks', [stage(bg,3,fit_ranges=ranges,exclusions=replace),stage(bg,3,exclusions=additions),stage(bg,3,exclusions=clear)],partition=partition)
        groups=c.get('uiso_equivalence_groups',[])
        if groups:
            members=groups[0];free=[atom(t['phase'],t['atom'],'U') for t in members]
            add(key,c,'source_equivalence','Release an entire original Uiso equivalence group, then fix it',[stage(bg+free,5),stage(bg,4)],partition=partition)
        for phase in c['phases']:
            sites=[a for a in phase['atoms'] if a['adp_type']=='A' and a['position_dof']>0]
            if sites:
                target=atom(phase['key'],sites[0]['label'],'X')
                add(key,c,'anisotropic_preservation','Refine positions of an existing anisotropic atom; preserve all Uij',[stage(bg+[target],5),stage(bg,4)],partition=partition)
                add(key,c,'negative','No Uiso substitution for an existing anisotropic atom',[],text=f"Refine only Uiso of {phase['key']}:{sites[0]['label']}, preserving its existing anisotropic Uij model. Do not convert ADP types or refine Uij instead.",expect='needs_input',partition=partition)
                break
    from gpx_recipe.contracts import validate_plan
    for case in cases:
        c=json.loads((root/'inputs'/(case['project']+'.context.json')).read_text())
        if case['oracle']:
            # Independent source-relation oracle: holding an equality peer fixed
            # removes the selected site's displacement degree of freedom.
            groups=[{(t['phase'],t['atom']) for t in g} for g in c.get('uiso_equivalence_groups',[])]
            invalid=False
            for stage in case['oracle']['stages']:
                active={(t['phase'],t['atom']) for t in stage['free'] if t['kind']=='atom' and 'U' in t['flags']}
                if any(group & active and group-active for group in groups):invalid=True;break
                groups.extend({(t['phase'],t['atom']) for t in eq['atoms']} for eq in stage['constraints'] if eq['kind']=='uiso_equal')
            if invalid:
                case.update(expectation='needs_input',rejected_recipe=case['oracle'],oracle=None,
                            oracle_reason='Requested complete free set leaves members of a preserved source Uiso equality fixed; native GSAS-II cannot refine the selected Uiso as requested')
        if case['oracle']:validate_plan(case['oracle'],c)
    return cases


if __name__=='__main__':
    import argparse,collections
    p=argparse.ArgumentParser();p.add_argument('campaign');args=p.parse_args();root=Path(args.campaign);cases=generate(root)
    (root/'catalog.json').write_text(json.dumps(cases,indent=2)+'\n');print(json.dumps({'cases':len(cases),'categories':dict(collections.Counter(c['category'] for c in cases)),'partitions':dict(collections.Counter(c['partition'] for c in cases))}))
