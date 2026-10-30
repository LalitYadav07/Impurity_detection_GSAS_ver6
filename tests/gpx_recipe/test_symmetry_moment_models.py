"""Geometry and unreviewed-k gates for finite magnetic searches."""
import copy

import pytest

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
