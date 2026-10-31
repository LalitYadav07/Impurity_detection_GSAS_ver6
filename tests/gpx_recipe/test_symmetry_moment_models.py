"""Geometry and unreviewed-k gates for finite magnetic searches."""
import copy

import pytest
import numpy as np

from gpx_recipe.symmetry.contracts import digest,validate_protocol,iter_trials
from gpx_recipe.symmetry.plans import build_plan
from test_symmetry_contracts import make_protocol
from test_symmetry_plans import inputs


def test_legacy_sealed_protocols_keep_their_collinear_policy_and_hash(tmp_path):
    original=validate_protocol(make_protocol(tmp_path))
    assert 'moment_model' not in original['policy']
    assert digest(original)==digest(validate_protocol(original))


def test_canted_starts_cover_component_signs_without_selecting_a_structure(tmp_path):
    catalog,context,settings=inputs(tmp_path)
    settings.update(moment_model='symmetry_allowed',directions=['x','y','z'])
    result=build_plan(context,catalog,tmp_path,settings)
    assert len(result['seeds'])==24
    assert {s['independent_mode_seed'] for s in result['seeds']}==set(range(1,25))
    assert result['policy']['moment_model']=='symmetry_allowed'
    assert len(list(iter_trials(result)))==24*len(result['candidates'])
    for axis in 'xyz':
        starts=[s for s in result['seeds'] if s['axis']==axis]
        assert len(starts)==8
        assert all(s['parent_direction']['xyz'.index(axis)]==1 for s in starts)
        assert len({tuple(s['parent_direction']) for s in starts})==4
        assert {s['magnitude'] for s in starts}=={.25*settings['maximum_moment'],.75*settings['maximum_moment']}
    legacy=build_plan(context,catalog,tmp_path,dict(settings,moment_model='collinear'))
    assert len(legacy['seeds'])==3 and digest(legacy)!=digest(result)
    # Nuclear and nuisance parameters stay fixed for a fair candidate screen.
    assert not any(result['policy'][k] for k in ['refine_cell','refine_background','refine_histogram_scale'])


@pytest.mark.parametrize('value',['arbitrary','multi_k',True,None])
def test_unknown_moment_models_cannot_execute(tmp_path,value):
    p=make_protocol(tmp_path);p['policy']['moment_model']=value
    with pytest.raises(ValueError):validate_protocol(p)


def test_unknown_k_remains_nonexecutable_at_the_worker_boundary(tmp_path):
    p=make_protocol(tmp_path);p['k']=None
    with pytest.raises(ValueError):validate_protocol(p)


def test_common_scale_is_an_explicit_shared_option_in_the_protocol(tmp_path):
    catalog,context,settings=inputs(tmp_path)
    fixed=build_plan(context,catalog,tmp_path,settings)
    released=build_plan(context,catalog,tmp_path,dict(settings,refine_histogram_scale=True))
    assert not fixed['policy']['refine_histogram_scale']
    assert released['policy']['refine_histogram_scale']
    assert digest(fixed)!=digest(released)
    assert not released['policy']['refine_cell'] and not released['policy']['refine_background']
    for field in ['refine_cell']:
        changed=copy.deepcopy(released);changed['policy'][field]=True
        with pytest.raises(ValueError):validate_protocol(changed)


def test_existing_background_is_reviewed_separately_and_changes_approval_identity(tmp_path):
    catalog,context,settings=inputs(tmp_path)
    fixed=build_plan(context,catalog,tmp_path,settings)
    released=build_plan(context,catalog,tmp_path,dict(settings,refine_background=True))
    assert released['policy']['refine_background'] and not released['policy']['refine_histogram_scale']
    assert not released['policy']['refine_cell'] and digest(fixed)!=digest(released)
    with pytest.raises(ValueError):build_plan(context,catalog,tmp_path,dict(settings,refine_background='yes'))


@pytest.mark.parametrize('value',['true',1,None])
def test_scale_switch_cannot_be_coerced_from_non_boolean(tmp_path,value):
    catalog,context,settings=inputs(tmp_path)
    with pytest.raises(ValueError):build_plan(context,catalog,tmp_path,dict(settings,refine_histogram_scale=value))


def test_independent_child_starts_preserve_coupled_components_and_break_equal_orbits():
    from gpx_recipe.symmetry.moment_starts import independent_orbit_start
    # A skew cell and a coupled x/y basis exercise physical rather than
    # Euclidean magnitude. z is forbidden and must remain exactly zero.
    metric=np.array([[1.,.5,0],[.5,1.,0],[0,0,1.]])
    starts=[]
    for seed in range(1,25):
        pair=[independent_orbit_start([1,1,0],[1.,-2.,0],[1.,-2.,0],metric,
              magnitude=4.5,seed=seed,orbit=orbit) for orbit in ['0:0','0:1']]
        for value in pair:
            assert value[2]==0 and value[1]==pytest.approx(-2*value[0])
            assert 0<np.sqrt(value @ metric @ value)<=4.5
        starts.append(pair)
    assert any(a[0]*b[0]<0 for a,b in starts)  # antisymmetric split-orbit modes
    assert any(not np.isclose(abs(a[0]),abs(b[0])) for a,b in starts)
    again=independent_orbit_start([1,1,0],[1.,-2.,0],[1.,-2.,0],metric,
        magnitude=4.5,seed=1,orbit='0:0')
    assert np.array_equal(starts[0][0],again)


def test_independent_components_are_nonzero_and_not_one_shared_direction():
    from gpx_recipe.symmetry.moment_starts import independent_orbit_start
    values=[independent_orbit_start([1,2,3],[1,1,1],[1,0,0],np.eye(3),
            magnitude=2.,seed=seed,orbit='0:0') for seed in range(1,25)]
    assert all(np.all(np.abs(v)>0) and np.linalg.norm(v)<=2. for v in values)
    assert len({tuple(np.sign(v)) for v in values})>1


def test_independent_mode_seed_is_closed_and_full_model_only(tmp_path):
    p=make_protocol(tmp_path);p['seeds'][0]['independent_mode_seed']=1
    with pytest.raises(ValueError,match='symmetry-allowed'):validate_protocol(p)
    p['policy']['moment_model']='symmetry_allowed'
    assert validate_protocol(p)['seeds'][0]['independent_mode_seed']==1
    for bad in [True,0,2001,'1']:
        p['seeds'][0]['independent_mode_seed']=bad
        with pytest.raises(ValueError):validate_protocol(p)


def test_parent_site_sign_pairs_keep_the_same_independent_child_start(tmp_path):
    catalog,context,settings=inputs(tmp_path)
    parent=context['phases'][0]
    parent['atoms'].append(dict(parent['atoms'][0],label='Mn2'))
    settings.update(moment_model='symmetry_allowed',directions=['x','y','z'],relative_sign_starts=True,
        sites=[*settings['sites'],dict(settings['sites'][0],label='Mn2')])
    p=build_plan(context,catalog,tmp_path,settings)
    assert len(p['seeds'])==48
    for axis in 'xyz':
        starts=[s for s in p['seeds'] if s['axis']==axis]
        for cant in range(1,5):
            for amplitude in [1,2]:
                pair=[s for s in starts if s['id'].endswith(f'-cant-{cant}-amp-{amplitude}')]
                assert len(pair)==2 and pair[0]['independent_mode_seed']==pair[1]['independent_mode_seed']
                assert {s['site_signs']['Mn2'] for s in pair}=={-1,1}
