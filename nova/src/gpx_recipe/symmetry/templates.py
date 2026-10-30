"""Reviewed search scope and conservative rebinding to a new acquisition."""
from copy import deepcopy
import math

from .contracts import digest,validate_protocol,iter_trials


def validate_template(template):
    required={'schema','protocol','context','scope','origin','existing_ids','seen_ids','future','maximum_scans'}
    if not isinstance(template,dict) or set(template)!=required or template['schema']!='radar-model-template/v1':
        raise ValueError('Invalid reviewed model-search template')
    validate_protocol(template['protocol'])
    if template['context'].get('sha256')!=template['protocol']['source']['sha256']:
        raise ValueError('Template context differs from its reviewed GPX')
    if set(template['scope'])!={'experiment','sample_key','bank'}:raise ValueError('Invalid model sample/bank scope')
    if type(template['future']) is not bool or type(template['maximum_scans']) is not int or not 1<=template['maximum_scans']<=2000:
        raise ValueError('Declare future enrollment and a one-to-2000 acquisition budget')
    for field in ['existing_ids','seen_ids']:
        values=template[field]
        if not isinstance(values,list) or len(values)!=len(set(values)) or any(not isinstance(v,str) or not v for v in values):
            raise ValueError('Template acquisitions must have unique exact identities')
    if not set(template['existing_ids']).issubset(template['seen_ids']) or len(template['existing_ids'])>template['maximum_scans']:
        raise ValueError('Existing scans exceed the reviewed campaign scope/budget')
    if template['future'] and (not template['scope']['sample_key'] or template['scope']['sample_key'].startswith('scan:')):
        raise ValueError('Identify a specimen before following future scans')
    return template


def matches(template,source,origin):
    scope=template['scope']
    return (source.get('experiment')==scope['experiment'] and source.get('sample_key')==scope['sample_key']
        and str(source.get('bank') or '')==scope['bank'] and origin==template['origin'])


def make_template(protocol,context,reference,candidates,*,origin,existing='selected',future=False,maximum_scans=200):
    if existing not in {'selected','sample'}:raise ValueError('Choose selected scan or matching sample/bank')
    template=dict(schema='radar-model-template/v1',protocol=deepcopy(validate_protocol(protocol)),context=deepcopy(context),
        scope=dict(experiment=reference['experiment'],sample_key=reference['sample_key'],bank=str(reference.get('bank') or '')),
        origin=origin,existing_ids=[],seen_ids=[],future=future,maximum_scans=maximum_scans)
    seen=[s['id'] for s,o in candidates if matches(template,s,o)]
    if reference['id'] not in seen:raise ValueError('The reference scan is outside the current model scope')
    template['seen_ids']=list(dict.fromkeys(seen))
    template['existing_ids']=template['seen_ids'] if existing=='sample' else [reference['id']]
    return validate_template(template)


def _phase(context,name):
    found=[p for p in context['phases'] if p['name']==name]
    if len(found)!=1:raise ValueError('The reviewed parent phase is missing or ambiguous; inspect a new plan')
    return found[0]


def _value(histogram,key):
    value=histogram['instrument'].get(key)
    return value[0] if isinstance(value,(tuple,list)) and value else value


def rebind(template,context,source,origin):
    validate_template(template)
    if not matches(template,source,origin):raise ValueError('This acquisition is outside the reviewed sample/bank/recipe scope')
    protocol=deepcopy(template['protocol']);old=_phase(template['context'],protocol['parent_phase']);new=_phase(context,protocol['parent_phase'])
    if context.get('sha256')!=source.get('source_sha256'):raise ValueError('The inspected source GPX revision changed')
    if old['setting_sha256']!=new['setting_sha256'] or old['type']!=new['type']:
        raise ValueError('Parent setting changed; discover and review new candidates')
    if context['runtime'].get('native_stack_sha256')!=template['context']['runtime'].get('native_stack_sha256'):
        raise ValueError('The acquisition was inspected with a different native runtime')
    prior={a['label']:a for a in old['atoms']};atoms={a['label']:a for a in new['atoms']}
    if set(prior)!=set(atoms):raise ValueError('Parent site identities changed; inspect a new plan')
    for key,a in atoms.items():
        previous=prior[key]
        if a['type']!=previous['type'] or not math.isclose(a['occupancy'],previous['occupancy'],rel_tol=0,abs_tol=1e-8):
            raise ValueError('Parent chemistry/occupancy changed; inspect a new plan')
        if protocol['kind']=='structural' and any(abs((v-w+.5)%1-.5)>1e-6 for v,w in zip(a['xyz'],previous['xyz'])):
            raise ValueError('Parent positions changed; regenerate the structural mode basis for this scan')
    old_hist=next(h for h in template['context']['histograms'] if h['name']==protocol['histogram'])
    choices=[h for h in context['histograms'] if h['name'] in new['histograms'] and _value(h,'Type')==_value(old_hist,'Type')]
    if len(choices)>1:choices=[h for h in choices if _value(h,'Bank')==_value(old_hist,'Bank')]
    if len(choices)!=1:raise ValueError('The corresponding histogram/bank is ambiguous; select it in a new plan')
    protocol['histogram']=choices[0]['name'];protocol['source']['sha256']=source['source_sha256']
    protocol['acquisition_id']=source['id'];protocol['template_sha256']=digest(template)
    return validate_protocol(protocol)


def trial_count(template):return sum(1 for _ in iter_trials(template['protocol']))
