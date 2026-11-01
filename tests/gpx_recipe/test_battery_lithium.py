"""Li curves must expose fixed assumptions and survive adverse scientific cases."""
from copy import deepcopy

import pytest

from gpx_recipe.battery import lithium as m
from gpx_recipe.battery.refinement_policy import validate_policy


@pytest.fixture
def context():
    atoms = [dict(label=label, element=element, occupancy=value, coordinates=xyz,
                  multiplicity=3, adp_type='I', adp=.012, position_dof=0,
                  variables=[dict(name=f'0::Afrac:{i}', group='occupancy')])
             for i, (label, element, value, xyz) in enumerate([
                 ('Li1','Li',.65,[0,0,0]), ('Ni_Li','Ni',.08,[0,0,0]), ('Ni2','Ni',1.,[0,0,.5])])]
    return dict(model_signature='model', controls={'min dM/M':.001},
        histograms=[dict(key='h0',name='bank',background_auxiliary={'FixedPoints':[]}, sample_parameters={'Absorption':[0.,False]})],
        phases=[dict(key='p0',name='oxide',histograms=['bank'],atoms=atoms,cell=[3,3,14,90,90,120],
                     neutron_isotopes={'Li':'Nat. Abund.'},neutron_scattering={'Li':{'SL':[-.193,0.]}})])


def spec(mixing=None, cells=True):
    return dict(schema=m.SCHEMA,phase='oxide',atom='Li1',mixing_atom=mixing,refine_cells=cells)


def trial(value, score, seed, **kwargs):
    return dict(id=f'{value}/{seed}',li_occupancy=value,status='completed',wss_per_point=score,
                convergence={'status':'passed'},adequacy='not_evaluated',li_nuisance={'status':'optimized'},**kwargs)


def curve(values):
    return [trial(value,score,seed) for value,score in values for seed in ('a','b')]


@pytest.mark.parametrize('mixing',[None,'Ni_Li'])
@pytest.mark.parametrize('cells',[False,True])
def test_closed_plan_never_releases_li_and_only_releases_selected_nuisance(context,mixing,cells):
    before=deepcopy(context)
    plan=m.profile_recipe(context,{'bank':2},validate_policy(),spec(mixing,cells))
    assert any(t['kind']=='cell' for s in plan['stages'] for t in s['free']) == cells
    atoms=[t for s in plan['stages'] for t in s['free'] if t['kind']=='atom']
    assert atoms == ([dict(kind='atom',phase='p0',atom='Ni_Li',flags='F')]*2 if mixing else [])
    if mixing: assert plan['stages'][-1]['bounds']==[dict(phase='p0',atom='Ni_Li',parameter='F',lower=0.,upper=.35)]
    assert context==before


@pytest.mark.parametrize('change',['other_site','li_mixing','absent','full_site','bad_target','bad_value'])
def test_mixing_targets_and_physical_capacity_are_not_guessed(context,change):
    profile=spec('Ni_Li')
    if change=='other_site':profile['mixing_atom']='Ni2'
    elif change=='li_mixing':context['phases'][0]['atoms'][1]['element']='Li_7'
    elif change=='absent':profile['mixing_atom']='missing'
    elif change=='full_site':context['phases'][0]['atoms'][0]['occupancy']=1.
    elif change=='bad_target':profile['atom']='Ni2'
    else:context['phases'][0]['atoms'][0]['occupancy']=float('nan')
    with pytest.raises(ValueError):m.profile_target(context,profile)


def test_other_fixed_occupant_reduces_mixing_capacity_without_forcing_full_site(context):
    context['phases'][0]['atoms'].append(dict(context['phases'][0]['atoms'][1],label='Co_Li',element='Co',occupancy=.1))
    assert m.profile_target(context,spec('Ni_Li'))[-1]==pytest.approx(.25)


def test_profile_binding_excludes_only_declared_freedoms(context):
    fixed=[dict(name='oxide',atoms=deepcopy(context['phases'][0]['atoms']))]
    base=m.profile_binding(context,fixed,spec('Ni_Li'))
    random_id=deepcopy(context);random_id['histograms'][0]['sample_parameters']['ranId']=123456789
    assert m.profile_binding(random_id,fixed,spec('Ni_Li'))==base
    changed=deepcopy(fixed);changed[0]['atoms'][0]['occupancy']=.45;changed[0]['atoms'][1]['occupancy']=.04
    assert m.profile_binding(context,changed,spec('Ni_Li'))==base
    changed[0]['atoms'][0]['adp']=.02
    assert m.profile_binding(context,changed,spec('Ni_Li'))!=base
    other=deepcopy(context);other['histograms'][0]['sample_parameters']['Absorption'][0]=.5
    assert m.profile_binding(other,fixed,spec('Ni_Li'))!=base
    other=deepcopy(context);other['controls']['min dM/M']=.01
    assert m.profile_binding(other,fixed,spec('Ni_Li'))!=base
    other=deepcopy(context);other['phases'][0]['neutron_isotopes']['Li']='7'
    assert m.profile_binding(other,fixed,spec('Ni_Li'))!=base
    other=deepcopy(context);other['phases'][0]['neutron_scattering']['Li']['SL'][0]=-.222
    assert m.profile_binding(other,fixed,spec('Ni_Li'))!=base
    held=m.profile_binding(context,fixed,spec(cells=False))
    other=deepcopy(context);other['phases'][0]['cell'][2]=14.1
    assert m.profile_binding(other,fixed,spec(cells=False))!=held
    other=deepcopy(context);del other['histograms'][0]['background_auxiliary']
    with pytest.raises(ValueError,match='fresh native'):m.profile_binding(other,fixed,spec())


@pytest.mark.parametrize('scores,status',[
    ([(.45,2),(.65,1),(.85,2)],'interior_grid_preference'),
    ([(.45,1),(.65,1),(.85,1)],'flat_over_tested_grid'),
    ([(.45,1),(.65,2),(.85,3)],'grid_boundary_unresolved'),
    ([(.2,2),(.4,1),(.6,2),(.8,1),(.9,2)],'separated_low_residual_regions')])
def test_curve_classification_never_claims_composition_or_confidence(scores,status):
    result=m.summarize_profile(spec(),curve(scores),.1)
    assert result['status']==status
    assert result['composition_status']=='not_identified'
    assert 'confidence interval' in result['interpretation']


@pytest.mark.parametrize('problem',['failed','unconverged','single_start','start_sensitive','residual','mixing_boundary'])
def test_small_residual_cannot_hide_incomplete_or_unqualified_grid(problem):
    rows=curve([(.45,2),(.65,1),(.85,2)])
    if problem=='failed':rows[0].update(status='needs_review',wss_per_point=None)
    elif problem=='unconverged':rows[0]['convergence']['status']='unavailable'
    elif problem=='single_start':rows.pop(0)
    elif problem=='start_sensitive':rows[0]['wss_per_point']=3.
    elif problem=='residual':rows[0]['adequacy']='threshold_exceeded'
    else:rows[0]['li_nuisance']['status']='bound_or_unresolved'
    result=m.summarize_profile(spec('Ni_Li'),rows,.1)
    assert result['status']=='optimization_unresolved' and result['grid_preference']==[]
    assert sum(p['declared_starts'] for p in result['points'])==len(rows)
    assert result['points'][0]['reasons']


def test_preview_requires_fixed_li_and_independent_mixing_with_exact_bounds(context):
    state=dict(phases=[dict(key='p0',atoms=[dict(label='Li1',flags='',occupancy=.65)])],
               effective_bounds=[dict(phase='p0',atom='Ni_Li',parameter='F',lower=0.,upper=.35)])
    preview=dict(stages=[dict(effective_vary=['0::Afrac:1'],configured_state=state)])
    m.verify_profile_preview(context,preview,spec('Ni_Li'))
    for change in ('release_li','change_li','constrain_mixing','wrong_bounds'):
        p=deepcopy(preview)
        if change=='release_li':p['stages'][0]['effective_vary'].append('0::Afrac:0')
        elif change=='change_li':p['stages'][0]['configured_state']['phases'][0]['atoms'][0]['occupancy']=.8
        elif change=='constrain_mixing':p['stages'][0]['effective_vary']=[]
        else:p['stages'][0]['configured_state']['effective_bounds'][0]['upper']=1.
        with pytest.raises(ValueError):m.verify_profile_preview(context,p,spec('Ni_Li'))


def test_final_model_must_prove_held_value_and_report_boundary_nuisance():
    atoms=[dict(label='Li1',element='Li',occupancy=.65,flags=''),dict(label='Ni_Li',element='Ni',occupancy=.08,flags='F')]
    metrics=[dict(kind='atom',phase='oxide',atom=a['label'],parameter='occupancy',value=a['occupancy'],sigma=None,
                  status='fixed' if i==0 else 'refined',native_status='fixed' if i==0 else 'refined_independent') for i,a in enumerate(atoms)]
    view=dict(phases=[dict(name='oxide',atoms=atoms)],parameters=metrics)
    assert m.verify_profile_result(view,spec('Ni_Li'),.65,[0,.35])['status']=='optimized'
    with pytest.raises(ValueError,match='fixed Li'):m.verify_profile_result(view,spec('Ni_Li'),.75,[0,.25])
    view['phases'][0]['atoms'][1]['occupancy']=0.;view['parameters'][1]['value']=0.
    assert m.verify_profile_result(view,spec('Ni_Li'),.65,[0,.35])['status']=='bound_or_unresolved'
