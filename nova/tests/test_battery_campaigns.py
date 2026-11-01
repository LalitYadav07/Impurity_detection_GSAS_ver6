"""Closed-set comparison and recovery controls; native acceptance is separate."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from gpx_recipe.contracts import plan_hash
from gpx_recipe.fit_domain import NATIVE_FIT_CONVENTION
from radar_pd_nova import battery_campaigns as m
from radar_pd_nova.refinement_workspace import RefinementWorkspace


def hashed(path):
    return sha256(path.read_bytes()).hexdigest()


class Runtime:
    kind='local'
    def __init__(self):
        self.views={}; self.contexts={}; self.calls=[]; self.fail=set()

    def resolve(self,source):
        if hashed(Path(source['source_path']))!=source['source_sha256']:
            raise ValueError('Immutable source changed')
        return deepcopy(source)

    def call(self,action,source,directory,**kwargs):
        self.calls.append((action,source['id']))
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        view=deepcopy(self.views[source['id']]);context=deepcopy(self.contexts[source['id']])
        context['source']['sha256']=source['source_sha256']
        result=dict(schema='radar-refinement-worker/v1',action=action,status='inspected',source_sha256=source['source_sha256'],context=context,
                    directory=str(directory),view='baseline.view.json',files={})
        if action=='preview':
            final=kwargs['plan']['stages'][-1]
            free_atoms={(t['phase'],t['atom']):t['flags'] for t in final['free'] if t['kind']=='atom'}
            configured=dict(phases=[dict(key=p['key'],atoms=[dict(label=a['label'],occupancy=a['occupancy'],
                flags=free_atoms.get((p['key'],a['label']),'')) for a in p['atoms']]) for p in context['phases']],
                effective_bounds=[dict(b) for b in final.get('bounds',[])])
            varied=['scale','cell']+[v['name'] for p in context['phases'] for a in p['atoms']
                if 'F' in free_atoms.get((p['key'],a['label']),'') for v in a.get('variables',[]) if v['group']=='occupancy']
            result.update(status='preview_passed',preview=dict(status='preview_passed',
                model_signature=context['model_signature'],source_sha256=source['source_sha256'],
                plan_sha256=plan_hash(kwargs['plan']),stages=[dict(effective_vary=varied,configured_state=configured)]))
        elif action=='run':
            if source['id'] in self.fail:
                return dict(status='needs_review',error='Physical bound failure; prior checkpoint retained')
            view['histograms'][0]['fit_statistics']=dict(point_count=3,weighted_residual_sum=6.)
            view['histograms'][0]['rwp']=2.
            view.setdefault('convergence',dict(schema='radar-native-convergence/v1',
                native_converged=True,last_cycle_complete=True,last_cycle_max_shift_over_sigma=.02,
                relative_objective_change=1e-6,aborted=False))
            result.update(status='completed',final_gpx='verified.gpx',final_view='final.view.json')
            (directory/'verified.gpx').write_bytes(b'verified')
            result['output_sha256']=hashed(directory/'verified.gpx')
            (directory/'final.view.json').write_text(json.dumps(view))
            stages=[]
            for stage in kwargs['plan']['stages']:
                gpx='verified_'+stage['id']+'.gpx';report=stage['id']+'.view.json'
                (directory/gpx).write_bytes(b'verified stage')
                (directory/report).write_text(json.dumps(view))
                stages.append(dict(id=stage['id'],title=stage['title'],gpx=gpx,view=report))
            (directory/'polished.execution.json').write_text(json.dumps(dict(status='completed_needs_scientific_review',stages=stages)))
            result.update(baseline_view='baseline.view.json',execution='polished.execution.json',stages=stages)
        (directory/'baseline.view.json').write_text(json.dumps(view))
        result['files']={p.name:hashed(p) for p in directory.iterdir() if p.name!='response.json'}
        (directory/'response.json').write_text(json.dumps(result))
        return result


@pytest.fixture
def setup(tmp_path,monkeypatch):
    runtime=Runtime(); workspace=RefinementWorkspace(tmp_path/'workspace',history_id='local',runtime=runtime)
    inputs={}
    for index in range(1,4):
        key=str(index);path=tmp_path/(key+'.gpx');path.write_text('initial model '+key)
        source=dict(id=key,experiment='cell-series',scan_key=key,scan_label=key,record_uid=key,
                    source_path=str(path),source_sha256=hashed(path),metadata={'physical_banks':{'h':2}})
        atom=dict(label='Li1',element='Li',coordinates=[0,0,0],occupancy=.9,
                  adp_type='I',adp=.01,multiplicity=3,refinement_flags='')
        runtime.contexts[key]=dict(model_signature='same-topology',source={'sha256':hashed(path)},
            histograms=[dict(key='h0',name='h',background={'function':'chebyschev-1','coefficients':3})],
            phases=[dict(key='p0',name='oxide',histograms=['h'],atoms=[atom],cell=[3,3,10],
                         phase_scales=[dict(histogram='h0',value=float(index),refine=False)])])
        runtime.views[key]=dict(histograms=[dict(name='h',coordinate='tof',x=[1.,2.,3.],observed=[4.,7.,3.],
            weights=[1.,1.,1.],weight_factor=1.,valid=[True]*3,fitted=[True]*3,limits=[1.,3.],
            exclusions=[],instrument_parameters={'difC':[1.,1.,False]},rwp=None,fit_range_convention=NATIVE_FIT_CONVENTION)])
        acquisition=dict(instrument='NOM',ipts='IPTS-1',run=4,sample_id='sample',cell_id='cell',
            start_utc='2020-01-01T00:00:00Z',end_utc='2020-01-01T00:01:00Z',metadata_sha256='a'*64)
        inputs[key]=dict(source=source,specification=dict(acquisition=acquisition,reduction_family='reduced',
            banks=[dict(data={'sha256':'b'*64})]))
    monkeypatch.setattr(m,'verify_prepared',lambda key:deepcopy(inputs[key]))
    def register(w,key,result):
        w.observe([result['source']]);return result['source']
    monkeypatch.setattr(m,'prepared_source',register)
    entries=[dict(model=str(i),family='layered',seed='start-'+str(i)) for i in (1,2)]
    return workspace,runtime,inputs,entries


def test_review_is_finite_read_only_and_recovery_never_repeats_a_trial(setup):
    w,rt,_,entries=setup
    token,doc=m.prepare_portfolio(w,entries,note='Two explicit scales')
    assert w.state['jobs']=={} and w.state['recipes']=={}
    assert [action for action,_ in rt.calls].count('preview')==2
    assert m.portfolio_summary(w,token)['status']=='awaiting_trials'
    record=m.start_portfolio(w,token,approver='reviewer')
    assert len(record['jobs'])==2
    assert all(r['scope']['mode']=='scans' and not r['future'] for r in w.state['recipes'].values())
    assert w.run_next()['status']=='completed'
    restored=RefinementWorkspace(w.root,history_id='local',runtime=rt)
    assert m.start_portfolio(restored,token,approver='reviewer')['jobs']==record['jobs']
    assert restored.run_next()['status']=='completed' and restored.run_next() is None
    summary=m.portfolio_summary(restored,token)
    assert summary['completed_count']==2 and summary['declared_count']==2
    assert summary['status']=='reviewable_unqualified'
    assert summary['scientific_assignment']=='not_established'
    assert summary['composition_status']=='not_evaluated'
    assert summary['trials'][0]['wss_per_point']==2.
    assert summary['families'][0]['start_check']=='within_screening_band'


def test_small_identical_residuals_do_not_establish_start_agreement_without_convergence(setup):
    w,rt,_,entries=setup
    rt.views['1']['convergence']=dict(schema='radar-native-convergence/v1',native_converged=True,
        last_cycle_complete=True,last_cycle_max_shift_over_sigma=3.9,relative_objective_change=-.002)
    rt.views['2']['convergence']=None
    token,doc=m.prepare_portfolio(w,entries,note='Convergence control',maximum_last_cycle_shift_over_sigma=.1)
    assert doc['policy']['maximum_last_cycle_shift_over_sigma']==.1
    m.start_portfolio(w,token,approver='reviewer');w.run_next();w.run_next()
    summary=m.portfolio_summary(w,token)
    assert summary['completed_count']==2 and summary['status']=='optimization_unresolved'
    assert [r['convergence']['status'] for r in summary['trials']]==['needs_review','unavailable']
    assert summary['families'][0]['start_spread']==0.
    assert summary['families'][0]['start_check']=='convergence_unresolved'


@pytest.mark.parametrize('maximum',[None,True,-1,float('nan'),float('inf'),'0.1'])
def test_invalid_convergence_ceiling_stops_before_preflight(setup,maximum):
    w,rt,_,entries=setup
    with pytest.raises(ValueError):
        m.prepare_portfolio(w,entries,note='invalid policy',maximum_last_cycle_shift_over_sigma=maximum)
    assert rt.calls==[] and not w.state['jobs']


@pytest.mark.parametrize('mixed',[False,True])
def test_preexisting_portfolio_keeps_its_metrics_and_never_mixes_native_conventions(setup,mixed):
    w,rt,_,entries=setup
    token,doc=m.prepare_portfolio(w,entries,note='Two explicit scales')
    # Reproduce a previously sealed review with the original v2 digest and
    # without the schema field introduced with zero-weight canonicalization.
    doc['observation'].pop('fit_domain_schema')
    doc['observation']['observations']=m.observation_contract(rt.views['1'],{'h':2},
        domain_schema='weighted-observations/v2')
    legacy_token=m.digest(doc)
    path=w.root/'battery-portfolios'/(legacy_token+'.json')
    path.write_text(json.dumps(doc))
    w.state['battery_portfolios'][legacy_token]=dict(token=legacy_token,path=str(path),status='ready',jobs={})
    w.save()
    m.start_portfolio(w,legacy_token,approver='reviewer')
    # Reproduce retained legacy output. A job finishing after the upgrade can
    # use the corrected native report; it must not join the old ranking.
    rt.views['1']['histograms'][0].pop('fit_range_convention')
    if not mixed: rt.views['2']['histograms'][0].pop('fit_range_convention')
    assert w.run_next()['status']=='completed'
    assert w.run_next()['status']=='completed'
    summary=m.portfolio_summary(w,legacy_token)
    assert summary['completed_count']==(1 if mixed else 2)
    assert summary['trials'][0]['wss_per_point']==2.
    assert 'historical' in summary['metric_convention']
    assert m.digest(m.load_portfolio(w,legacy_token))==legacy_token
    if mixed:
        row=summary['trials'][1]
        assert row['status']=='comparison_unavailable' and row['wss_per_point'] is None
        assert row['banks']==[] and row['point_count'] is None
        assert 'new reviewed comparison' in row['message']
        assert summary['families'][0]['start_check']=='incomplete'


def test_new_review_refreshes_old_cached_fit_domain_without_rewriting_its_artifacts(setup):
    w,rt,inputs,entries=setup
    w.observe([inputs['1']['source']]);source=w.inspect('1')
    folder=Path(source['inspection']);path=folder/'baseline.view.json'
    old=json.loads(path.read_text());old['histograms'][0].pop('fit_range_convention')
    path.write_text(json.dumps(old));old_hash=hashed(path)
    response=json.loads((folder/'response.json').read_text());response['files'][path.name]=old_hash
    (folder/'response.json').write_text(json.dumps(response))
    token,doc=m.prepare_portfolio(w,entries,note='Native-bin boundary correction')
    assert 'battery-inspections' in doc['trials'][0]['source']['inspection']
    assert hashed(path)==old_hash
    assert doc['observation']['fit_domain_schema']=='weighted-observations/v4'


def test_new_review_reexports_reused_legacy_fit_without_rerunning_or_rewriting_it(setup):
    w,rt,_,entries=setup
    old_token,_=m.prepare_portfolio(w,entries[:1],note='Previous review')
    m.start_portfolio(w,old_token,approver='reviewer')
    rt.views['1']['histograms'][0].pop('fit_range_convention')
    job=w.run_next()
    old_report=Path(job['directory'])/job['result']['final_view'];old_hash=hashed(old_report)
    rt.views['1']['histograms'][0].update(fit_range_convention=NATIVE_FIT_CONVENTION,
        fit_statistics=dict(point_count=3,weighted_residual_sum=6.))
    token,doc=m.prepare_portfolio(w,entries[:1],note='Corrected native interval')
    assert doc['trials'][0]['retained_result_report']['output_sha256']==job['result']['output_sha256']
    m.start_portfolio(w,token,approver='reviewer')
    assert w.run_next() is None
    assert w.state['jobs'][job['id']]['attempts']==1
    assert len([c for c in rt.calls if c[0]=='run'])==1
    summary=m.portfolio_summary(w,token)
    assert summary['completed_count']==1 and summary['trials'][0]['wss_per_point']==2.
    assert 'not repeated' in summary['trials'][0]['message']
    assert hashed(old_report)==old_hash
    assert m.digest(m.load_portfolio(w,old_token))==old_token
    reference=doc['trials'][0]['retained_result_report']
    (Path(reference['directory'])/'response.json').write_text('{}')
    changed=m.portfolio_summary(w,token)['trials'][0]
    assert changed['status']=='integrity_failure' and changed['wss_per_point'] is None


@pytest.mark.parametrize('change',[ 'cell', 'run', 'weights', 'limits', 'profile', 'reduction', 'source', 'evidence' ])
def test_incomparable_inputs_stop_before_any_native_preflight(setup,change):
    w,rt,inputs,entries=setup
    h=rt.views['2']['histograms'][0];spec=inputs['2']['specification']
    if change=='cell':spec['acquisition']['cell_id']='other'
    elif change=='run':spec['acquisition']['run']=9
    elif change=='weights':h['weights'][0]=2.
    elif change=='limits':h['limits'][0]=1.1
    elif change=='profile':h['instrument_parameters']['difC'][0]=2.
    elif change=='reduction':spec['reduction_family']='alternative'
    elif change=='source':spec['banks'][0]['data']['sha256']='c'*64
    elif change=='evidence':spec['reduction_evidence']=[{'sha256':'d'*64}]
    with pytest.raises(ValueError,match='same cell'):
        m.prepare_portfolio(w,entries,note='review')
    assert not any(action=='preview' for action,_ in rt.calls)
    assert not w.state['jobs']


def test_fixed_lithium_is_a_hypothesis_change_not_an_optimizer_start(setup):
    w,rt,_,entries=setup
    rt.contexts['2']['phases'][0]['atoms'][0]['occupancy']=.6
    with pytest.raises(ValueError,match='fixed atomic'):
        m.prepare_portfolio(w,entries,note='review')
    entries[1]['family']='lower fixed Li'
    token,_=m.prepare_portfolio(w,entries,note='Two conditional hypotheses')
    assert len(m.load_portfolio(w,token)['trials'])==2


def test_only_released_widths_can_differ_as_starts_in_one_family(setup):
    from gpx_recipe.battery.refinement_policy import validate_policy
    w,rt,_,entries=setup
    for key in ('1','2'):
        phase=rt.contexts[key]['phases'][0]
        phase['phase_scales'][0]['value']=1.
        phase['histogram_phase_models']={'h0':{'Size':['isotropic',[float(key),1.,1.],[False]*3],
                                             'Mustrain':['isotropic',[1000.,1000.,1000.],[False]*3]}}
    before=deepcopy(rt.contexts)
    with pytest.raises(ValueError,match='fixed atomic'):
        m.prepare_portfolio(w,entries,note='Fixed widths differ')
    policy=validate_policy();policy['sample_widths']=[dict(phase='oxide',physical_bank=2,kind='sample_size')]
    token,doc=m.prepare_portfolio(w,entries,note='Declared independent size starts',refinement_policy=policy)
    assert doc['refinement_policy']==policy
    assert doc['trials'][0]['starting_values_sha256']!=doc['trials'][1]['starting_values_sha256']
    assert rt.contexts==before
    # A changed fixed input is a new prepared revision, not a mutation of an
    # already inspected immutable source through a shared fixture dictionary.
    rt.contexts['3']['phases']=deepcopy(rt.contexts['2']['phases'])
    rt.contexts['3']['phases'][0]['histogram_phase_models']['h0']['Mustrain'][1][0]=2000.
    with pytest.raises(ValueError,match='fixed atomic'):
        m.prepare_portfolio(w,[entries[0],dict(entries[1],model='3')],note='Unreleased strain differs',refinement_policy=policy)


def test_declared_background_policy_is_sealed_into_the_exact_review_and_plan(setup):
    from gpx_recipe.battery.refinement_policy import validate_policy
    w,_,_,entries=setup;policy=validate_policy()
    policy['background']=dict(function='chebyschev-1',coefficients=6)
    token,doc=m.prepare_portfolio(w,entries,note='Review background sensitivity',refinement_policy=policy)
    assert all(t['plan']['stages'][0]['background']['coefficients']==6 for t in doc['trials'])
    assert all('starting_backgrounds' in t for t in doc['trials'])
    policy['background']['coefficients']=12
    assert m.load_portfolio(w,token)['refinement_policy']['background']['coefficients']==6
    revised,_=m.prepare_portfolio(w,entries,note='Review background sensitivity',refinement_policy=policy)
    assert revised!=token and not w.state['jobs']


def test_identical_physical_starts_do_not_count_as_independent_trials(setup):
    w,rt,_,entries=setup
    rt.contexts['2']['phases']=deepcopy(rt.contexts['1']['phases'])
    with pytest.raises(ValueError,match='Duplicate scientific'):
        m.prepare_portfolio(w,entries,note='review')


def test_changed_source_or_review_cannot_partially_enqueue_a_set(setup):
    w,rt,inputs,entries=setup
    token,doc=m.prepare_portfolio(w,entries,note='review')
    path=Path(inputs['2']['source']['source_path']);path.write_text('changed')
    with pytest.raises(ValueError,match='source changed'):
        m.start_portfolio(w,token,approver='reviewer')
    assert not w.state['jobs'] and not w.state['recipes']
    path.write_text('initial model 2')
    saved=w.root/'battery-portfolios'/(token+'.json')
    doc['trials'].pop();saved.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='portfolio changed'):
        m.start_portfolio(w,token,approver='reviewer')
    assert not w.state['jobs']


def test_failed_trial_remains_in_denominator_and_never_seeds_another(setup):
    w,rt,_,entries=setup
    token,_=m.prepare_portfolio(w,entries,note='review',maximum_wss_per_point=1.)
    m.start_portfolio(w,token,approver='reviewer');rt.fail.add('1')
    w.run_next();w.run_next()
    result=m.portfolio_summary(w,token)
    assert result['status']=='incomplete' and result['completed_count']==1 and result['declared_count']==2
    assert result['trials'][0]['wss_per_point'] is None
    assert result['trials'][1]['adequacy']=='threshold_exceeded'
    assert [uid for action,uid in rt.calls if action=='run']==['1','2']


def test_tampered_completed_artifact_loses_comparison_eligibility(setup):
    w,rt,_,entries=setup
    token,_=m.prepare_portfolio(w,entries,note='review');m.start_portfolio(w,token,approver='reviewer')
    job=w.run_next();w.run_next()
    (Path(job['directory'])/'verified.gpx').write_bytes(b'modified')
    result=m.portfolio_summary(w,token)
    assert result['status']=='incomplete' and result['trials'][0]['status']=='integrity_failure'
    assert result['trials'][0]['wss_per_point'] is None


def test_only_the_native_fixed_zero_width_default_is_equivalent(setup):
    _,rt,_,_=setup
    original=deepcopy(rt.views['1']);original['histograms'][0]['instrument_parameters']['Type']=['PNT','PNT',False]
    final=deepcopy(original);params=final['histograms'][0]['instrument_parameters'];params['Z']=[0.,0.,False]
    assert m.observation_contract(original,{'h':2})==m.observation_contract(final,{'h':2})
    params['Z'][1]=.01
    assert m.observation_contract(original,{'h':2})!=m.observation_contract(final,{'h':2})
    params['Z']=[0.,0.,True]
    assert m.observation_contract(original,{'h':2})!=m.observation_contract(final,{'h':2})


def test_a_queue_registration_failure_rolls_back_the_whole_trial_set(setup,monkeypatch):
    w,_,_,entries=setup
    token,_=m.prepare_portfolio(w,entries,note='review')
    enqueue=w._enqueue
    def fail_second(recipe,source):
        if source['id']=='2':raise ValueError('Injected registration failure')
        return enqueue(recipe,source)
    monkeypatch.setattr(w,'_enqueue',fail_second)
    with pytest.raises(ValueError,match='registration failure'):
        m.start_portfolio(w,token,approver='reviewer')
    assert not w.state['jobs'] and not w.state['recipes']
    assert not w.state['battery_portfolios'][token]['jobs']
    assert json.loads(w.manifest.read_text())['jobs']=={}


def test_pause_and_resume_are_scoped_and_do_not_repeat_completed_work(setup):
    w,rt,_,entries=setup
    token,_=m.prepare_portfolio(w,entries,note='review');m.start_portfolio(w,token,approver='reviewer')
    m.pause_portfolio(w,token)
    assert w.run_next() is None and m.portfolio_summary(w,token)['status']=='paused'
    m.pause_portfolio(w,token,False)
    assert m.portfolio_summary(w,token)['status']=='running'
    w.run_next();w.run_next()
    m.pause_portfolio(w,token);m.pause_portfolio(w,token,False)
    assert w.run_next() is None and len([a for a,_ in rt.calls if a=='run'])==2


def test_review_labels_map_banks_and_distinguish_fixed_atoms(setup):
    from radar_pd_nova.battery_campaign_controller import review_start_values
    w,_,_,entries=setup
    _,doc=m.prepare_portfolio(w,entries,note='review')
    trial=doc['trials'][0];before=deepcopy(trial)
    values=json.loads(review_start_values(trial))
    assert values[0]['starting_HAP_scales']==[dict(physical_bank=2,value=1.)]
    assert values[0]['fixed_atoms'][0]['occupancy']==.9
    assert trial==before


def test_start_review_retains_all_phases_and_labels_widths_and_backgrounds(setup):
    from radar_pd_nova.battery_campaign_controller import review_start_values,review_backgrounds
    w,rt,_,entries=setup
    for context in rt.contexts.values():
        context['phases'][0]['histogram_phase_models']={'h0':{'Size':['isotropic',[.1,1.,1.],[False]*3],
                                                           'Mustrain':['isotropic',[500.,1000.,1000.],[False]*3]}}
    _,doc=m.prepare_portfolio(w,entries,note='review')
    trial=doc['trials'][0];before=deepcopy(trial)
    values=json.loads(review_start_values(trial))
    assert isinstance(values,list) and values[0]['name']=='oxide'
    row=values[0]['starting_sample_widths'][0]
    assert row['physical_bank']==2 and row['sample_size']['value']==.1
    assert row['sample_microstrain']['unit']=='µstrain'
    assert json.loads(review_backgrounds(trial))[0]['physical_bank']==2
    assert trial==before


def test_changing_transport_after_preflight_cannot_dispatch_a_local_portfolio(setup):
    w,rt,_,entries=setup
    token,_=m.prepare_portfolio(w,entries,note='review')
    rt.kind='galaxy'
    with pytest.raises(ValueError,match='local scientific worker'):
        m.start_portfolio(w,token,approver='reviewer')
    assert not w.state['jobs']


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1,True])
def test_invalid_comparison_policy_cannot_be_sealed(setup,value):
    w,_,_,entries=setup
    with pytest.raises(ValueError,match='finite'):
        m.prepare_portfolio(w,entries,note='review',close_fit_band=value)


def li_grid(setup, tmp_path):
    """Six immutable sources, with three held occupancies and two scale starts."""
    w,rt,inputs,_=setup
    original_context=deepcopy(rt.contexts['1']);original_view=deepcopy(rt.views['1']);original_input=deepcopy(inputs['1'])
    entries=[]
    for i,value in enumerate((.45,.45,.65,.65,.85,.85),10):
        key=str(i);path=tmp_path/(key+'.gpx');path.write_text('Li model '+key)
        source=deepcopy(original_input['source']);source.update(id=key,scan_key=key,record_uid=key,source_path=str(path),source_sha256=hashed(path))
        context=deepcopy(original_context);context['source']['sha256']=source['source_sha256']
        context['histograms'][0].update(background_auxiliary={'FixedPoints':[]},sample_parameters={'Absorption':[0,False]})
        atom=context['phases'][0]['atoms'][0];atom.update(occupancy=value,variables=[dict(name='0::Afrac:0',group='occupancy')])
        context['phases'][0]['phase_scales'][0]['value']=float(i%2+1)
        context['phases'][0].update(neutron_isotopes={'Li':'Nat. Abund.'},neutron_scattering={'Li':{'SL':[-.193,0.]}})
        rt.contexts[key]=context
        view=deepcopy(original_view)
        view['phases']=[dict(name='oxide',atoms=[dict(label='Li1',element='Li',occupancy=value,flags='')])]
        view['parameters']=[dict(kind='atom',phase='oxide',atom='Li1',parameter='occupancy',value=value,sigma=None,status='fixed')]
        rt.views[key]=view
        inputs[key]=dict(source=source,specification=deepcopy(original_input['specification']))
        entries.append(dict(model=key,family='oxide',seed=f'Li {value} / start {i%2}'))
    profile=dict(schema='radar-held-li-profile/v1',phase='oxide',atom='Li1',mixing_atom=None,refine_cells=False)
    return entries,profile


def test_held_li_review_binds_exact_grid_and_recovers_all_trials(setup,tmp_path):
    w,rt,_,_=setup;entries,profile=li_grid(setup,tmp_path)
    token,doc=m.prepare_portfolio(w,entries,note='Declared Li grid',li_profile=profile)
    assert not w.state['jobs'] and doc['li_profile']==profile
    assert [t['li_occupancy'] for t in doc['trials']]==[.45,.45,.65,.65,.85,.85]
    assert all(not any(t['kind']=='cell' for s in trial['plan']['stages'] for t in s['free']) for trial in doc['trials'])
    m.start_portfolio(w,token,approver='reviewer')
    for _ in entries:assert w.run_next()['status']=='completed'
    restored=RefinementWorkspace(w.root,history_id='local',runtime=rt)
    m.start_portfolio(restored,token,approver='recovery')
    assert restored.run_next() is None
    summary=m.portfolio_summary(restored,token)
    assert summary['li_profile']['status']=='flat_over_tested_grid'
    assert summary['composition_status']=='not_identified'
    assert len(summary['families'])==3 and all(f['declared_starts']==2 for f in summary['families'])
    assert len([action for action,_ in rt.calls if action=='run'])==6


@pytest.mark.parametrize('change',['adp','absorption','background_support','cell','solver','other_fixed_occupancy'])
def test_extra_fixed_assumption_changes_stop_li_grid_before_native_preflight(setup,tmp_path,change):
    w,rt,_,_=setup;entries,profile=li_grid(setup,tmp_path)
    c=rt.contexts[entries[-1]['model']]
    if change=='adp':c['phases'][0]['atoms'][0]['adp']=.02
    elif change=='absorption':c['histograms'][0]['sample_parameters']['Absorption'][0]=.3
    elif change=='background_support':c['histograms'][0]['background_auxiliary']['FixedPoints']=[[1.,2.]]
    elif change=='cell':c['phases'][0]['cell']=[3,3,11]
    elif change=='solver':c['controls']={'min dM/M':.1}
    else:
        for context in rt.contexts.values():
            context['phases'][0]['atoms'].append(dict(context['phases'][0]['atoms'][0],label='Li2',coordinates=[0,0,.5],occupancy=.1))
        c['phases'][0]['atoms'][-1]['occupancy']=.2
    with pytest.raises(ValueError,match='same topology'):m.prepare_portfolio(w,entries,note='Invalid grid',li_profile=profile)
    assert not any(action=='preview' for action,_ in rt.calls) and not w.state['jobs']


def test_held_li_grid_keeps_failed_points_and_rejects_changed_final_occupancy(setup,tmp_path):
    w,rt,_,_=setup;entries,profile=li_grid(setup,tmp_path)
    token,_=m.prepare_portfolio(w,entries,note='Adverse control',li_profile=profile)
    rt.fail.add(entries[0]['model'])
    rt.views[entries[2]['model']]['phases'][0]['atoms'][0]['occupancy']=.66
    m.start_portfolio(w,token,approver='reviewer')
    for _ in entries:w.run_next()
    summary=m.portfolio_summary(w,token)
    assert summary['declared_count']==6 and summary['completed_count']==4
    assert summary['trials'][2]['status']=='integrity_failure'
    assert summary['li_profile']['status']=='optimization_unresolved'
    assert summary['li_profile']['grid_preference']==[]


def test_li_grid_requires_three_distinct_native_values_and_cannot_invent_from_labels(setup,tmp_path):
    w,rt,_,_=setup;entries,profile=li_grid(setup,tmp_path)
    with pytest.raises(ValueError,match='three to twelve'):m.prepare_portfolio(w,entries[:4],note='Too short',li_profile=profile)
    assert not any(action=='preview' for action,_ in rt.calls)


def test_li_sync_uses_sealed_policy_and_clears_diagnostic_when_opening_ordinary_review(setup,tmp_path):
    from types import SimpleNamespace
    from radar_pd_nova.battery_campaign_controller import BatteryCampaignMixin
    w,rt,_,ordinary=setup;entries,profile=li_grid(setup,tmp_path)
    token,_=m.prepare_portfolio(w,entries,note='Display Li policy',li_profile=profile)
    app=BatteryCampaignMixin();app.server=SimpleNamespace(state=SimpleNamespace())
    app._refinement_store=lambda:w;app._battery_portfolio_token=token
    app._battery_portfolio_sync()
    s=app.server.state
    assert s.battery_li_result_ready and len(s.battery_li_points)==3
    assert 'All phase cells held' in s.battery_portfolio_policy
    assert 'unidentified' in s.battery_portfolio_policy
    assert 'held grid value' in s.battery_portfolio_review[0]['values']
    ordinary_token,_=m.prepare_portfolio(w,ordinary,note='Ordinary comparison')
    app._battery_portfolio_token=ordinary_token;app._battery_portfolio_sync()
    assert not s.battery_li_result_ready and s.battery_li_points==[] and s.battery_li_notice==''


def test_saved_review_choices_show_cell_and_mixing_policy_and_retain_unavailable_entries(setup,tmp_path):
    from radar_pd_nova.battery_campaign_controller import saved_portfolio_choices
    w,_,_,_=setup;entries,profile=li_grid(setup,tmp_path)
    token,_=m.prepare_portfolio(w,entries,note='Named Li review',li_profile=profile)
    choice=saved_portfolio_choices(w)[0]
    assert choice['value']==token and 'cell · Li oxide/Li1 · mixing fixed' in choice['title']
    assert '0/6 completed' in choice['title']
    (w.root/'battery-portfolios'/(token+'.json')).write_text('{}')
    assert 'unavailable' in saved_portfolio_choices(w)[0]['title']
