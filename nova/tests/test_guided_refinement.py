import copy
import json

import pytest

from gpx_recipe import guided
from gpx_recipe.model_edits import validate_edit, SCHEMA as EDIT_SCHEMA


@pytest.fixture
def context():
    def atom(label, element, occupancy, multiplicity, **kwargs):
        return dict(label=label, element=element, occupancy=occupancy, multiplicity=multiplicity,
                    adp_type='I', adp=.01, coordinates=[0, 0, .2], position_dof=0, **kwargs)
    return dict(model_signature='model-a', source={'sha256': 'source-a'},
        histograms=[dict(key='h0', name='PWDR one')], phases=[dict(key='p0', name='Na host', histograms=['PWDR one'], cell={'length_a': 4.},
        atoms=[atom('Na1', 'Na', .6, 2), atom('Na2', 'Na', .2, 4), atom('Na3', 'Na', 1., 1), atom('O1', 'O', 1., 4)],
        occupancy_groups=[['Na1'], ['Na2'], ['Na3'], ['O1']])])


def complete(context, recipe='redistribution', **overrides):
    values=dict(recipe=recipe, model_review='reviewed', cell_phases=['p0'], phase='p0',
                sites=['Na1', 'Na2'] if recipe=='redistribution' else 'Na1', total_ions=3., positions=[],
                reference='Independent reference model at the same temperature; vacancies allowed, Na content known.', conditional='accepted', reference_scope='single')
    values.update(overrides); state=guided.start(context)
    while (question:=guided.next_question(context,state)) is not None:
        state=guided.answer(context,state,question['id'],values[question['id']])
    return state


def test_weighted_conservation_subtracts_held_sites_and_retains_full_free_set(context):
    plan=guided.build_plan(context,complete(context))
    last=plan['stages'][-1]
    assert last['constraints'][0]['total']==2.
    assert [x['coefficient'] for x in last['constraints'][0]['terms']]==[2,4]
    assert last['free'][:len(plan['stages'][-2]['free'])]==plan['stages'][-2]['free']
    assert not any(x['kind']=='histogram_scale' or 'U' in x.get('flags','') for x in last['free'])
    assert len(last['bounds'])==2


def test_mixture_has_no_occupancy_or_atom_releases(context):
    plan=guided.build_plan(context,complete(context,'mixture'))
    assert len(plan['stages'])==2
    assert not any(x['kind']=='atom' for s in plan['stages'] for x in s['free'])
    assert all(s['convergence']=={'max_shift_over_sigma':.5} for s in plan['stages'])
    assert plan['stages'][1]['title']=='Selected unit cells'


def test_reference_length_matches_the_compiled_plan_contract(context):
    from gpx_recipe.contracts import validate_plan
    state=complete(context,reference='  '+'x'*3000+'  ')
    assert validate_plan(guided.build_plan(context,state),context)
    with pytest.raises(ValueError,match='3000'):
        complete(context,reference='x'*3001)


@pytest.mark.parametrize('name',['baseline','final','BASELINE','Final'])
def test_stage_names_cannot_overwrite_baseline_or_final_artifacts(context,name):
    from gpx_recipe.contracts import validate_plan
    plan=guided.build_plan(context,complete(context,'mixture'))
    plan['stages'][0]['id']=name
    with pytest.raises(ValueError,match='reserved'):
        validate_plan(plan,context)


def test_stage_names_cannot_collide_on_a_case_insensitive_filesystem(context):
    from gpx_recipe.contracts import validate_plan
    plan=guided.build_plan(context,complete(context,'mixture'))
    plan['stages'][1]['id']=plan['stages'][0]['id'].upper()
    with pytest.raises(ValueError,match='unique'):
        validate_plan(plan,context)


def test_ordinary_material_is_not_offered_battery_only_recipes(context):
    for atom in context['phases'][0]['atoms']:
        if atom['element']=='Na':atom['element']='Al'
    question=guided.next_question(context,guided.start(context))
    assert [o['value'] for o in question['options']]==['mixture']
    assert guided.build_plan(context,complete(context,'mixture'))['stages']


def test_redistribution_requires_multiple_distinct_sites_in_one_phase(context):
    p=context['phases'][0];p['atoms']=p['atoms'][:1]
    assert guided.available_recipes(context)=={'mixture','vacancy'}
    p['atoms'].append(dict(p['atoms'][0],label='Na_alternate'))
    p['occupancy_groups']=[['Na1','Na_alternate']]
    assert guided.available_recipes(context)=={'mixture','vacancy'}
    p['occupancy_groups']=[['Na1'],['Na_alternate']]
    assert 'redistribution' in guided.available_recipes(context)


def test_recorded_decisions_use_phase_names_and_clear_held_choices(context):
    rows=guided.decision_summary(context,complete(context,'vacancy',reference_scope='series'))
    assert any(r['answer']=='Na host' for r in rows)
    assert any(r['answer']=='Na1' for r in rows)
    assert any(r['answer']=='None (held)' for r in rows)
    assert all(r['answer']!='p0' and r['question']!='cell_phases' for r in rows)


def test_guided_progress_restores_only_for_unchanged_source_and_clears_stale_correction(context):
    from types import SimpleNamespace
    from radar_pd_nova.guided_refinement_controller import GuidedRefinementMixin
    state=SimpleNamespace(rf_source='s1',flush=lambda:None)
    saved=complete(context,'mixture')
    app=GuidedRefinementMixin();app.server=SimpleNamespace(state=state);app._init_guided_refinement()
    app._refinement_workspace=SimpleNamespace(state=dict(sources={'s1':dict(context=context,source_sha256='source-a')},guided_setups={'s1':saved}))
    app._guided_source_changed()
    assert state.rf_guide_complete and app._guided_state==saved and state.rf_guide_summary
    state.rf_edit_phase='p0';state.rf_edit_reason='Earlier source';state.rf_edit_cif_text='earlier structure'
    context['source']['sha256']='new-revision'
    app._refinement_workspace.state['sources']['s1']['source_sha256']='new-revision'
    app._guided_source_changed()
    assert app._guided_state is None and not state.rf_guide_complete and not state.rf_guide_summary
    assert state.rf_edit_phase is None and not state.rf_edit_reason and not state.rf_edit_cif_text


def test_reused_guide_decisions_are_original_provenance_not_an_editable_new_setup(context):
    from types import SimpleNamespace
    from radar_pd_nova.guided_refinement_controller import GuidedRefinementMixin
    saved=complete(context,'mixture')
    current=copy.deepcopy(context);current['source']['sha256']='another-project'
    state=SimpleNamespace(rf_source='s2',rf_experiment='exp',flush=lambda:None)
    app=GuidedRefinementMixin();app.server=SimpleNamespace(state=state);app._init_guided_refinement()
    app._refinement_workspace=SimpleNamespace(state=dict(sources={'s2':dict(context=current,source_sha256='another-project')},
        drafts={'exp':dict(source_id='s2',context=current,guided=saved,plan={'stages':['present']})}))
    app._guided_source_changed()
    assert app._guided_state is None and not state.rf_guide_complete
    assert state.rf_guide_summary and 'original starting project' in state.rf_guide_summary_notice


@pytest.mark.parametrize('change', [dict(aborted=True),dict(native_converged=False),dict(last_cycle_complete=False),dict(last_cycle_max_shift_over_sigma=.6)])
def test_required_convergence_cannot_accept_abort_missing_shifts_or_large_motion(change):
    from gpx_recipe.convergence import SCHEMA,require_convergence
    evidence=dict(schema=SCHEMA,aborted=False,native_converged=True,last_cycle_complete=True,last_cycle_max_shift_over_sigma=.01,relative_objective_change=0.)
    assert require_convergence(evidence,{'max_shift_over_sigma':.5})['status']=='passed'
    with pytest.raises(ValueError,match='required convergence screen'):
        require_convergence(dict(evidence,**change),{'max_shift_over_sigma':.5})


@pytest.mark.parametrize('value',[True,0,-1,2,float('nan')])
def test_invalid_convergence_contract_cannot_compile(context,value):
    from gpx_recipe.contracts import validate_plan
    plan=guided.build_plan(context,complete(context,'mixture'))
    plan['stages'][0]['convergence']['max_shift_over_sigma']=value
    with pytest.raises(ValueError,match='Convergence'):
        validate_plan(plan,context)


def test_single_site_respects_held_cooccupant_capacity(context):
    p=context['phases'][0]
    p['atoms'].append(dict(p['atoms'][0],label='Ni1',element='Ni',occupancy=.15))
    p['occupancy_groups'][0]=['Na1','Ni1']
    last=guided.build_plan(context,complete(context,'vacancy'))['stages'][-1]
    assert last['bounds'][0]['upper']==.85
    assert last['constraints']==[]


@pytest.mark.parametrize('total', [2.9, 8., 1.])
def test_inconsistent_or_degenerate_total_stops_before_refinement(context,total):
    with pytest.raises(ValueError,match='starting model|declared total'):
        complete(context,total_ions=total)


def test_stale_reference_values_invalidate_even_if_topology_matches(context):
    state=complete(context)
    plan=guided.build_plan(context,state)
    context['source']['sha256']='changed-values-same-topology'
    with pytest.raises(ValueError,match='starting model changed'):
        guided.build_plan(context,state)
    from gpx_recipe.contracts import validate_plan
    with pytest.raises(ValueError,match='different source GPX revision'):
        validate_plan(plan,context)


def test_series_recipe_accepts_new_data_and_released_starts_without_reinterpreting(context):
    from gpx_recipe.contracts import validate_plan
    state=complete(context,'vacancy',reference_scope='series')
    plan=guided.build_plan(context,state)
    assert 'source_sha256' not in plan and plan['reference_policy']['recipe']=='vacancy'
    other=copy.deepcopy(context);other['source']['sha256']='second-scan'
    other['phases'][0]['cell']['length_a']=4.1
    other['phases'][0]['atoms'][0]['occupancy']=.4
    other['phases'][0]['phase_scales']=[dict(histogram='h0',value=.7)]
    assert validate_plan(plan,other)==plan
    with pytest.raises(ValueError,match='starting model changed'):
        guided.build_plan(other,state)  # Setup decisions themselves remain immutable provenance.


@pytest.mark.parametrize('field,value',[('adp',.02),('coordinates',[0,0,.3]),('occupancy',.8)])
def test_series_recipe_rejects_changed_held_reference(context,field,value):
    from gpx_recipe.contracts import validate_plan
    plan=guided.build_plan(context,complete(context,'vacancy',reference_scope='series'))
    context['phases'][0]['atoms'][1][field]=value
    with pytest.raises(ValueError,match='Held reference model does not match'):
        validate_plan(plan,context)


def test_series_reference_rejects_changed_sample_correction_but_not_unverified_condition_labels(context):
    from gpx_recipe.contracts import validate_plan
    context['histograms'][0]['sample_parameters']={'Scale':[1.,False], 'Time':0., 'Temperature':300.}
    plan=guided.build_plan(context,complete(context,'vacancy',reference_scope='series'))
    context['histograms'][0]['sample_parameters'].update(Time=10.,Temperature=301.)
    validate_plan(plan,context)
    context['histograms'][0]['sample_parameters']['Scale'][0]=2.
    with pytest.raises(ValueError,match='sample_corrections'):
        validate_plan(plan,context)


def test_redistribution_rechecks_total_on_every_scan(context):
    from gpx_recipe.contracts import validate_plan
    plan=guided.build_plan(context,complete(context,reference_scope='series'))
    atoms=context['phases'][0]['atoms'];atoms[0]['occupancy']=.5;atoms[1]['occupancy']=.25
    validate_plan(plan,context)
    atoms[1]['occupancy']=.3
    with pytest.raises(ValueError,match='Occupancy reference model does not match.*starting model'):
        validate_plan(plan,context)


@pytest.mark.parametrize('sites',[['missing'],['O1']])
def test_reference_policy_rejects_unknown_or_ineligible_sites(context,sites):
    from gpx_recipe.contracts import validate_plan
    plan=guided.build_plan(context,complete(context,'vacancy',reference_scope='series'))
    plan['reference_policy']['sites']=sites
    with pytest.raises(ValueError,match='existing isotropic Li/Na'):
        validate_plan(plan,context)


def test_old_guided_mixture_stays_exact_source_bound(context):
    from gpx_recipe.contracts import validate_plan
    state=complete(context,'mixture');state['schema']=guided.LEGACY_SCHEMA
    plan=guided.build_plan(context,state)
    assert plan['source_sha256']=='source-a'
    context['source']['sha256']='source-b'
    with pytest.raises(ValueError,match='different source GPX revision'):
        validate_plan(plan,context)


def test_guided_compile_preserves_selected_series_and_future_enrollment(context):
    import asyncio
    from types import SimpleNamespace
    from radar_pd_nova.guided_refinement_controller import GuidedRefinementMixin
    setup=complete(context,'mixture');calls=[]
    state=SimpleNamespace(rf_busy=False,rf_guided_busy=False,rf_source='s1',rf_experiment='exp',
        rf_scope='samples',rf_samples=['sample-a'],rf_scans=[],rf_banks=['2'],rf_existing=True,rf_future=True,flush=lambda:None)
    class App(GuidedRefinementMixin):
        def _guided_context(self):return context
        def _refinement_sync(self):pass
        def _refinement_checkpoint(self):pass
        def _refinement_schedule(self,work,label):asyncio.run(work)
    app=App();app.server=SimpleNamespace(state=state);app._guided_state=setup;app._guided_epoch='test'
    app._refinement_workspace=SimpleNamespace(prepare=lambda *a,**k:calls.append((a,k)))
    app.compile_guided_refinement()
    assert calls and state.rf_scope=='samples' and state.rf_samples==['sample-a']
    assert state.rf_banks==['2'] and state.rf_existing and state.rf_future and not state.rf_error


def test_unasked_fields_and_mixed_species_are_rejected(context):
    state=complete(context,'mixture');state['answers']['python']='evil'
    with pytest.raises(ValueError,match='unsupported'):
        guided.build_plan(context,state)
    context['phases'][0]['atoms'][1]['element']='Li'
    with pytest.raises(ValueError,match='one element'):
        complete(context)


class Client:
    def __init__(self,**overrides):
        self.value=dict(status='answer',choices=['mixture'],number=None,text=None,evidence='mixture',unhandled=[])
        self.value.update(overrides)
    def complete_structured(self,messages,schema,name):
        assert schema['additionalProperties'] is False
        assert name=='guided_refinement_answer'
        return json.dumps(self.value),dict(model='test')


def test_interpretation_never_changes_state_and_needs_current_question(context):
    state=guided.start(context);before=copy.deepcopy(state)
    proposal=guided.interpret(context,state,'Use a mixture',Client())
    assert state==before
    updated=guided.accept_proposal(context,state,proposal)
    assert updated['answers']=={'recipe':'mixture'}
    with pytest.raises(ValueError,match='earlier question'):
        guided.accept_proposal(context,updated,proposal)


def test_unhandled_intent_and_invalid_output_cannot_be_applied(context):
    state=guided.start(context)
    proposal=guided.interpret(context,state,'Use a mixture and add oxygen',Client(unhandled=['add oxygen']))
    with pytest.raises(ValueError,match='unresolved'):
        guided.accept_proposal(context,state,proposal)
    with pytest.raises(ValueError,match='ground'):
        guided.interpret(context,state,'Do something',Client())
    with pytest.raises(ValueError,match='options'):
        guided.interpret(context,state,'mixture',Client(choices=['invented']))


def test_phase_edit_requires_explicit_unique_component_and_revision(context):
    from hashlib import sha256
    cif='data_example\n'
    request=dict(schema=EDIT_SCHEMA,source_sha256='source-a',model_signature='model-a',operation='add',
                 name='Second Na host',cif=cif,cif_sha256=sha256(cif.encode()).hexdigest(),
                 links=[dict(histogram='h0',scale=.1)],reason='Independent coexisting state')
    assert validate_edit(context,request)['name']=='Second Na host'
    for field,value in [('name','Na host'),('source_sha256','old'),('links',[]),('cif_sha256','wrong')]:
        with pytest.raises(ValueError):validate_edit(context,dict(request,**{field:value}))
