"""One hundred reviewable natural-language recipe cases with independent oracles.

Inputs and generated cases stay in a user-selected campaign directory, outside git.
These are execution/control tests, not asserted experimental ground truth.
"""
import copy
import json
from pathlib import Path


def generate(root):
    root = Path(root)
    contexts = {p.stem.replace('.context', ''): json.loads(p.read_text()) for p in (root/'inputs').glob('*.context.json')}
    domains = json.loads((root/'axis-domains.json').read_text())
    cases = []
    bg = {'kind':'background','histogram':'h0'}
    def atom(p, a, f): return {'kind':'atom','phase':p,'atom':a,'flags':f}
    def scale(p): return {'kind':'phase_scale','phase':p,'histogram':'h0'}
    def cells(key): return [{'kind':'cell','phase':p['key']} for p in contexts[key]['phases']]
    def scales(key): return [scale(p['key']) for p in contexts[key]['phases']]
    def st(free, **edits):
        result=dict(id='stage',title='Requested refinement',cycles=4,free=copy.deepcopy(free),constraints=[]);result.update(edits);return result
    def describe_target(t,c):
        kind=t['kind'];p=next((x for x in c['phases'] if x['key']==t.get('phase')),None)
        phase=' in phase named '+repr(p['name']) if p else ''
        if kind=='atom': return t['atom']+phase+' ('+', '.join({'F':'site occupancy','X':'symmetry-independent positions','U':'Uiso'}[f] for f in t['flags'])+')'
        return {'background':'existing background','phase_scale':'phase scale'+phase,'cell':'unit cell'+phase,'histogram_scale':'histogram scale','preferred_orientation':'preferred orientation'+phase}[kind]
    def request_for(stages,c):
        lines=['Apply these stages in order, with 4 maximum refinement cycles in each. Every stage lists the complete free set; hold all unlisted parameters fixed. Preserve existing constraints, instrument calibration and all settings not explicitly changed.']
        for i,s in enumerate(stages,1):
            line=f'{i}. Refine '+ '; '.join(describe_target(t,c) for t in s['free'])+'.'
            if s.get('phase_normalization'): line+=' Normalize the linked HAP phase scales to sum to one without changing starting calculated intensities; refine histogram scale. A single phase stays fixed at one.'
            if s.get('background'):
                b=s['background'];line+=f' Set {b["function"]} background with {b["coefficients"]} coefficients and {b["fixed_points"]} existing support points.'
            for x in s.get('bounds',[]):
                phase=next(p['name'] for p in c['phases'] if p['key']==x['phase']);line+=f' Bound {x["atom"]} in phase {phase!r} '+{'F':'occupancy','U':'Uiso'}[x['parameter']]+f' between {x["lower"]} and {x["upper"]}'+(f', starting at {x["value"]}' if 'value' in x else '')+'.'
            for x in s.get('fit_ranges',[]):line+=f' Set fit range for h0 from {x["lower"]} to {x["upper"]} in {x["unit"]} units.'
            for x in s.get('exclusions',[]):line+=f' Excluded regions for h0: {x["mode"]} {x["regions"]} in {x["unit"]} units.'
            for x in s.get('texture',[]):
                phase=next(p['name'] for p in c['phases'] if p['key']==x['phase']);line+=f' Configure {x["model"]} texture for phase {phase!r} in h0'+(f' with hkl {x["direction"]}, ratio {x.get("value",1.0)}' if 'direction' in x else f' at order {x["order"]}')+'.'
            for x in s['constraints']:
                terms=' + '.join(f'{t["coefficient"]} times occupancy of {t["atom"]} in phase {next(p["name"] for p in c["phases"] if p["key"]==t["phase"])!r}' for t in x['terms']);line+=' Constrain '+terms+f' = {x["total"]}. '+x['reason']+'.'
            lines.append(line)
        return '\n'.join(lines)
    def add(category,key,stages,scenario,request=None,expect='execute'):
        c=contexts[key];stages=copy.deepcopy(stages)
        for i,s in enumerate(stages,1):s.update(id=f'stage_{i}',title=f'{scenario}: step {i}')
        request=request or request_for(stages,c)
        plan=dict(schema='radar-gpx-recipe/v1',model_signature=c['model_signature'],request=request,assumptions=[],questions=[],unsupported=[],stages=stages)
        cases.append(dict(id=f'R{len(cases)+1:03d}',category=category,project=key,scenario=scenario,request=request,expectation=expect,oracle=plan if expect=='execute' else None))
    # 1–12: individual occupancies, vacancies and explicitly bounded starts.
    targets=[('alumina-warm','p0','O1'),('alumina-cold','p0','Al1'),('nomad-battery','p4','Li0'),('nomad-battery','p3','Li0'),('lapr-chromite','p0','La0'),('lapr-chromite','p0','Pr1'),('layered-nmc','p0','Li0'),('layered-nmc','p0','Ni3'),('namnfe-oxide','p0','Na1'),('fe-mg-oxide','p0','Mg0'),('fe-mg-oxide','p0','Fe3'),('babisb-oxide','p0','Bi1')]
    for i,(key,p,a) in enumerate(targets):
        edits={}
        if i in {2,4,6,9}:
            current=next(x['occupancy'] for q in contexts[key]['phases'] if q['key']==p for x in q['atoms'] if x['label']==a)
            edits['bounds']=[dict(phase=p,atom=a,parameter='F',lower=max(0.,current-.15),upper=min(1.,current+.15),value=current)]
        add('occupancy',key,[st([bg,atom(p,a,'F')],**edits)],'Selected site occupancy '+a)
    # 13–24: individual and grouped Uiso, with explicit physical bounds.
    u_targets=[('alumina-warm','p0','Al1'),('alumina-cold','p0','O1'),('alumina-warm','p1','Mg1'),('nomad-battery','p0','Al0'),('nomad-battery','p3','Li0'),('bacusi','p0','Cu2'),('lapr-chromite','p0','La0'),('layered-nmc','p0','Ni3'),('namnfe-oxide','p0','O1'),('fe-mg-oxide','p0','O4'),('babisb-oxide','p0','Bi1'),('prfeaso-two-phase','p1','As2')]
    for i,(key,p,a) in enumerate(u_targets):
        edits={}
        if i%3==0: edits['bounds']=[dict(phase=p,atom=a,parameter='U',lower=.0001,upper=.05)]
        add('uiso',key,[st([bg,*scales(key),atom(p,a,'U')],**edits)],'Selected Uiso '+a)
    # 25–36: positions, joint XU and explicit atom selection.
    x_targets=[('alumina-warm','p0','O1','X'),('alumina-warm','p0','Al1','XU'),('alumina-cold','p0','O1','XU'),('bacusi','p0','Si1','X'),('bacusi','p0','O3','XU'),('bacusi','p0','O4','X'),('lapr-chromite','p0','O3','XU'),('layered-nmc','p0','O6','XU'),('namnfe-oxide','p0','O1','X'),('fe-mg-oxide','p0','O4','XU'),('babisb-oxide','p0','O3','XU'),('nomad-battery','p3','C1','XU')]
    for key,p,a,f in x_targets:add('positions_joint',key,[st([bg,*scales(key),atom(p,a,f)])],'Joint atom flags '+a+' '+f)
    # 37–48: releasing one phase and then another, and deliberately fixing prior flags.
    for i in range(12):
        key=['alumina-warm','alumina-cold','nomad-battery','prfeaso-two-phase','silica-two-phase','nomad-battery'][i%6]
        if key.startswith('alumina'):pa,aa,pb,ab='p0','O1','p1','O1'
        elif key=='nomad-battery':pa,aa,pb,ab='p3','C1','p4','C2'
        elif key=='prfeaso-two-phase':pa,aa,pb,ab='p0','As2','p1','As2'
        else:pa,aa,pb,ab='p0','Si1','p1','Si1'
        first='U' if i%2==0 else 'XU';second='U'
        if key=='silica-two-phase':first=second='X'
        if key.startswith('alumina') and pb=='p1': second='U'
        stages=[st([bg,*scales(key),atom(pa,aa,first)]),st([*scales(key),atom(pb,ab,second)])]
        if i>=6:stages.append(st([bg,*scales(key),*cells(key)]))
        add('phase_sequence',key,stages,'Phase-specific release/fix sequence '+str(i+1))
    # 49–60: both legitimate scale conventions, normalized single and multiphase cases.
    f_keys=['alumina-warm','alumina-cold','nomad-battery','prfeaso-two-phase','silica-two-phase','alumina-warm','nomad-battery','lapr-chromite','layered-nmc','alumina-warm','nomad-battery','silica-two-phase']
    for i,key in enumerate(f_keys):
        if i in {0,3}:stages=[st([bg,*scales(key)])]
        else:
            free=[bg,dict(kind='histogram_scale',histogram='h0')]+(scales(key) if len(contexts[key]['phases'])>1 else [])
            stages=[st(free,phase_normalization=['h0'])]
            if i in {5,6,9,10}:stages.append(st([*free,*cells(key)]))
        add('phase_fractions',key,stages,'Scale convention '+str(i+1))
    # 61–70: fit ranges and masked intervals in three coordinate systems.
    for i in range(10):
        key=['alumina-warm','nomad-battery','bacusi','layered-nmc','babisb-oxide'][i%5];unit=['native','d','Q'][i%3];lo,hi=domains[key][unit];width=hi-lo
        lower=round(lo+.08*width,6);upper=round(hi-.08*width,6);one=[round(lo+.4*width,6),round(lo+.405*width,6)]
        edits={'fit_ranges':[dict(histogram='h0',unit=unit,lower=lower,upper=upper)]}
        if i>1:edits['exclusions']=[dict(histogram='h0',unit=unit,mode='add' if i%2 else 'replace',regions=[one])]
        stages=[st([bg,*scales(key)],**edits)]
        if i>=7:stages.append(st([bg,*scales(key)],exclusions=[dict(histogram='h0',unit='native',mode='clear',regions=[])]))
        add('ranges_exclusions',key,stages,'Calibrated ranges and exclusions '+str(i+1))
    # 71–80: two-, three- and four-species real mixed sites, with carried constraints.
    mix=[('lapr-chromite',['La0','Pr1']),('namnfe-oxide',['Mn1','Fe1']),('fe-mg-oxide',['Mg0','Fe1']),('fe-mg-oxide',['Mg2','Fe3']),('layered-nmc',['Li0','Ni1']),('layered-nmc',['Li2','Ni3','Co4','Mn5'])]
    for i in range(10):
        if i==9:
            key='namnfe-oxide';free=[bg,atom('p0','Na1','F'),atom('p0','Mn1','F'),atom('p0','Fe1','F')]
            weighted=[dict(kind='occupancy_sum',terms=[dict(phase='p0',atom='Mn1',coefficient=1),dict(phase='p0',atom='Fe1',coefficient=1)],total=1,reason='Preserve the existing transition-metal site total'),dict(kind='occupancy_sum',terms=[dict(phase='p0',atom='Na1',coefficient=1),dict(phase='p0',atom='Mn1',coefficient=-2)],total=0,reason='Explicit weighted coupling for this validation recipe, not inferred experiment chemistry')]
            a=st(free,constraints=weighted);b=st([bg,atom('p0','Na1','F'),atom('p0','Mn1','FU'),atom('p0','Fe1','F')],constraints=copy.deepcopy(weighted))
            add('site_mixing',key,[a,b],'Coupled weighted occupancy equations and selected Uiso')
            continue
        key,names=mix[i%len(mix)];s=st([bg,*[atom('p0',a,'F') for a in names]])
        s['constraints']=[dict(kind='occupancy_sum',terms=[dict(phase='p0',atom=a,coefficient=1) for a in names],total=1,reason='User-requested total occupancy of this existing mixed site')]
        stages=[s]
        if i>=6:stages.append(st([bg,*[atom('p0',a,'FU') for a in names]],constraints=copy.deepcopy(s['constraints'])))
        add('site_mixing',key,stages,'Existing mixed site '+','.join(names))
    # 81–90: MD and SH models, chosen axes, orders and phase-specific changes.
    tex=[('alumina-warm','p0','march-dollase'),('bacusi','p0','march-dollase'),('nomad-battery','p2','march-dollase'),('layered-nmc','p0','march-dollase'),('babisb-oxide','p0','march-dollase'),('alumina-cold','p0','spherical-harmonics'),('bacusi','p0','spherical-harmonics'),('nomad-battery','p2','spherical-harmonics'),('layered-nmc','p0','spherical-harmonics'),('lapr-chromite','p0','spherical-harmonics')]
    for i,(key,p,model) in enumerate(tex):
        edit=dict(phase=p,histogram='h0',model=model)
        if model=='march-dollase':edit.update(direction=[0,0,1] if i%2==0 else [1,0,0],value=1.)
        else:edit['order']=4 if i%2==0 else 2
        stages=[st([bg,*scales(key),dict(kind='preferred_orientation',phase=p,histogram='h0')],texture=[edit])]
        if i in {2,6}:stages.append(st([bg,*scales(key),*cells(key)]))
        add('texture',key,stages,'Texture model '+model+' '+str(i+1))
    # 91–100: useful requests that cannot be executed as written, not silently substituted.
    negatives=[('Unknown atom','Refine only O99 occupancy in corundum. Do not substitute another atom.'),('Symmetry-fixed position','Refine only O1 positional coordinates in the MgO phase without changing symmetry. Do not refine Uiso or anything else.'),('Anisotropic displacement','Convert corundum Al1 to anisotropic displacement and refine all its anisotropic components.'),('Magnetic model','Add a magnetic phase and refine its ordered magnetic moment.'),('Missing phase','Add a new NaCl phase and refine its fraction without supplying a CIF or structure.'),('Invalid texture axis','Refine March-Dollase texture along hkl [0,0,0]. Keep that exact direction; do not choose another.'),('Ambiguous range unit','Fit between 1 and 3. I have not decided whether these numbers mean d-spacing or Q; ask me before changing the range.'),('Unphysical occupancy bounds','Refine corundum Al1 occupancy with lower bound -0.2 and upper bound 1.5. Do not change these bounds.'),('Scale degeneracy','Refine histogram scale and every phase scale simultaneously. Do not normalize them, do not add a scale constraint, and do not fix any scale.'),('Fixed weight-percent relation','Constrain MgO to exactly 20 weight percent while refining both phases atomic occupancies. Do not substitute a raw phase-scale fraction of 0.2.')]
    for title,request in negatives:add('clarify_or_reject','alumina-warm',[],title,request=request,expect='needs_input')
    # Independent physical oracle: a held co-occupant reduces available site capacity.
    # Preserve these original requests; reclassify their incompatible bounds rather
    # than silently tightening them or counting their execution as a success.
    for case in cases:
        if case['expectation']!='execute':continue
        c=contexts[case['project']]
        for stage in case['oracle']['stages']:
            active={(t['phase'],t['atom']) for t in stage['free'] if t['kind']=='atom' and 'F' in t['flags']}
            for bound in stage.get('bounds',[]):
                if bound['parameter']!='F':continue
                phase=next(p for p in c['phases'] if p['key']==bound['phase'])
                atom=next(a for a in phase['atoms'] if a['label']==bound['atom'])
                point=tuple(round(float(x)%1,7)%1 for x in atom['coordinates'])
                held=sum(a['occupancy'] for a in phase['atoms'] if a['label']!=atom['label'] and a['multiplicity']==atom['multiplicity'] and tuple(round(float(x)%1,7)%1 for x in a['coordinates'])==point and (phase['key'],a['label']) not in active)
                if bound['upper']>1-held+1e-7:
                    case.update(expectation='needs_input',rejected_recipe=case['oracle'],oracle_reason='Requested bounds permit overfilling a shared site while its other occupants are fixed')
                    case['oracle']=None
                    break
            if case['expectation']=='needs_input':break
    assert len(cases)==100
    for case in cases:
        if case['expectation']=='execute':
            from gpx_recipe.contracts import validate_plan
            validate_plan(case['oracle'],contexts[case['project']])
    return cases


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('campaign');args=parser.parse_args();root=Path(args.campaign)
    cases=generate(root);(root/'catalog.json').write_text(json.dumps(cases,indent=2)+'\n')
    print(json.dumps({'cases':len(cases),'execute':sum(c['expectation']=='execute' for c in cases),'clarify_or_reject':sum(c['expectation']=='needs_input' for c in cases)}))
