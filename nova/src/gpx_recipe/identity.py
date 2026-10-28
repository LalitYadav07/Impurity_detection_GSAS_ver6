"""Explicit source relations and stable constraint identities across native IDs."""
import math


def explicit_constraints(project, json_safe):
    rows=[]
    for section,records in project.data.get('Constraints',{}).get('data',{}).items():
        if not isinstance(records,list):continue
        for record in records:
            if not isinstance(record,list) or len(record)<4 or record[-1] not in {'h','e','c','f'}:continue
            terms=[]
            for coefficient,variable in record[:-3]:
                terms.append({'variable':variable.varname() if hasattr(variable,'varname') else str(variable),'coefficient':json_safe(coefficient)})
            rows.append({'section':section,'kind':record[-1],'terms':terms,'value':json_safe(record[-3]),'definition':json_safe(record[-2])})
    return rows


def variable_identity(name, phases, histograms):
    fields=name.split(':')
    if len(fields)<3:return {'literal':name}
    p=next((p for p in phases if str(p['native_id'])==fields[0]),None)
    h=next((h for h in histograms if str(h['native_id'])==fields[1]),None)
    atom=next((a for a in p['atoms'] if str(a['native_id'])==fields[3]),None) if p and len(fields)>3 else None
    return {'phase':p['key'] if p else fields[0],'histogram':h['key'] if h else fields[1],
            'family':fields[2],'atom':atom['label'] if atom else fields[3] if len(fields)>3 else None,
            'suffix':fields[4:]}


def constraint_binding(records,phases,histograms):
    import json
    result=[]
    for record in records:
        terms=[{**t,'variable':variable_identity(t['variable'],phases,histograms)} for t in record['terms']]
        result.append({**record,'terms':sorted(terms,key=lambda t:json.dumps(t,sort_keys=True))})
    return sorted(result,key=lambda t:json.dumps(t,sort_keys=True))


def uiso_groups(records,phases,histograms):
    neighbours={}
    for record in records:
        if record['kind']!='e':continue
        terms=record['terms']
        try:
            first=float(terms[0]['coefficient'])
            if not first or any(not math.isclose(float(t['coefficient']),first,rel_tol=1e-12,abs_tol=1e-12) for t in terms):continue
        except (TypeError,ValueError,IndexError):continue
        atoms=[]
        for term in terms:
            v=variable_identity(term['variable'],phases,histograms)
            if v.get('family')!='AUiso' or not v.get('atom'):break
            atoms.append((v['phase'],v['atom']))
        else:
            for atom in atoms[1:]:neighbours.setdefault(atoms[0],set()).add(atom);neighbours.setdefault(atom,set()).add(atoms[0])
    groups=[];seen=set()
    for atom in sorted(neighbours):
        if atom in seen:continue
        pending=[atom];group=set()
        while pending:
            item=pending.pop()
            if item in group:continue
            group.add(item);pending.extend(neighbours.get(item,set())-group)
        seen.update(group);groups.append([{'phase':p,'atom':a} for p,a in sorted(group)])
    return groups


def validate_joint_uiso_release(stage,groups):
    active={(t['phase'],t['atom']) for t in stage['free'] if t['kind']=='atom' and 'U' in t['flags']}
    for group in groups:
        members={(t['phase'],t['atom']) for t in group}
        if members & active and not members<=active:
            raise ValueError('Requested Uiso refinement is tied by a preserved equality to held atoms. Release the full existing equality group with user authorization, or clarify the requested fixed/free choices; never silently remove constraints or free extra atoms.')


def bound_binding(controls,phases,histograms,json_safe):
    """Source limits are part of a recipe's scientific compatibility contract."""
    import json
    result={}
    for key in ['parmMinDict','parmMaxDict']:
        rows=[]
        for variable,value in controls.get(key,{}).items():
            name=variable.varname() if hasattr(variable,'varname') else str(variable)
            rows.append({'variable':variable_identity(name,phases,histograms),'value':json_safe(value)})
        result[key]=sorted(rows,key=lambda row:json.dumps(row,sort_keys=True))
    return result


def model_topology(phase):
    """Bind model definitions while allowing per-scan refined values to vary."""
    texture=phase.get('global_texture',{})
    result={'global_texture':{'model':texture.get('Model'),'order':texture.get('Order',0),
                             'coefficients':sorted((texture.get('SH Coeff') or [False,{}])[1])},'hap':{}}
    for key,record in phase.get('histogram_phase_models',{}).items():
        pref=record.get('Pref.Ori.',[])
        value={'preferred_orientation':{'model':pref[0] if pref else None,
            'direction':pref[3] if len(pref)>3 and pref[0]=='MD' else None,
            'order':pref[4] if len(pref)>4 and pref[0]=='SH' else None,
            'coefficients':sorted(pref[5]) if len(pref)>5 and pref[0]=='SH' else []}}
        for name in ['Size','Mustrain']:
            data=record.get(name,[]);value[name]={'model':data[0] if data else None,'direction':data[3] if len(data)>3 else None}
        result['hap'][key]=value
    return result
