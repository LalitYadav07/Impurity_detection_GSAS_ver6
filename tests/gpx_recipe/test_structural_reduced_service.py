import json
import hashlib
from pathlib import Path
import numpy as np
import pytest
from gpx_recipe.structural_live.reduced import from_arrays, from_snapshot
from gpx_recipe.structural_live.service import SearchSettings
from gpx_recipe.symmetry.native_audit import profile_residual_metrics

def saved_result(path,parameters):
    result=dict(parameters=parameters,previous_likelihood_accumulated=False)
    for role in ('frozen','continuation'):
        gpx=path/(role+'.gpx');gpx.write_bytes(b'synthetic publication integrity test')
        result[role]=dict(path=str(gpx),sha256=hashlib.sha256(gpx.read_bytes()).hexdigest())
    (path/'fit.json').write_text(json.dumps(result))
    return result


def observation(revision=0, exposure=None):
    return from_arrays([np.arange(5.),np.array([1.,1.,100.,1.,1.]),np.ones(5)],
        [[0,4],[.2,3.5],[2,2]],acquisition='run-bank-reduction-epoch',protocol_sha256=hashlib.sha256(b'{}').hexdigest(),
        revision=revision,exposure=exposure)


def test_reduced_exclusions_and_nonaligned_limits_match_native_convention():
    o=observation()
    assert o.exposure is None
    np.testing.assert_array_equal(o.mask,[False,True,False,True,True])
    np.testing.assert_array_equal(o.normalization_mask,[False,True,True,True,True])
    metric=profile_residual_metrics([o.x,o.y,np.ones(5)],np.zeros(5),[[0,4],[.2,3.5],[2,2]])
    assert metric['normalization_points']==4 and metric['fitted_points']==3
    assert metric['rwp_native']<metric['rwp_fit_points_only']
    assert not o.x.flags.writeable and not o.mask.flags.writeable


def test_live_invalid_mask_is_inverted_without_count_or_exposure_invention():
    o=from_snapshot(np.arange(5.),np.ones(5),np.ones(5),np.array([False,True,False,False,False]),
        [[0,4],[0,4]],acquisition='a',protocol_sha256='b',revision=1)
    assert o.exposure is None and not o.mask[1] and o.mask[0]
    with pytest.raises(ValueError,match='Boolean'):
        from_snapshot(np.arange(5.),np.ones(5),np.ones(5),np.zeros(5),[[0,4],[0,4]],acquisition='a',protocol_sha256='b')


def test_reduced_revision_replay_and_unknown_exposure_are_explicit():
    old=observation(1)
    assert observation(1).require_continuation(old)=='duplicate'
    assert observation(2).require_continuation(old)=='replacement'
    with pytest.raises(ValueError):observation(0).require_continuation(old)
    with pytest.raises(ValueError):observation(2,10).require_continuation(old)
    with pytest.raises(ValueError):observation(2,5).require_continuation(observation(1,10))


@pytest.mark.parametrize('settings',[dict(starts=1),dict(starts=True),dict(seconds=float('nan')),
    dict(mode_radius=0),dict(cell_radius=.1),dict(recovery_seconds=-1),dict(background_scale=0)])
def test_invalid_structural_execution_scope_rejected(settings):
    with pytest.raises(ValueError):SearchSettings(**settings)


def test_failed_updates_restore_and_duplicate_updates_do_not_accumulate_likelihood(tmp_path,monkeypatch):
    import gpx_recipe.structural_live.session as module
    prepared=tmp_path/'prepared';prepared.mkdir();(prepared/'prepared.json').write_text('{}')
    calls=[]
    def run(p,path,**kw):
        calls.append(kw)
        if kw['observation'].revision==3:raise RuntimeError('export failed')
        path.mkdir();return saved_result(path,[.1,.2])
    session=module.StructuralSession(prepared,tmp_path/'session',runner=run)
    session.update(observation(1));before=(session.root/'latest.json').read_bytes()
    assert session.update(observation(1))['new_fit'] is False and len(calls)==1
    with pytest.raises(RuntimeError):session.update(observation(3))
    assert (session.root/'latest.json').read_bytes()==before
    restored=module.StructuralSession(prepared,tmp_path/'session',runner=run)
    assert restored.update(observation(1))['state']=='duplicate'
    restored.update(observation(2));assert calls[-1]['warm_starts']==[[.1,.2]]
    with pytest.raises(ValueError):restored.update(observation(0))
    with pytest.raises(ValueError):restored.update(observation(4,20))


def test_full_mode_plan_has_one_portfolio_work_unit_per_child(tmp_path):
    from test_symmetry_plans import inputs
    from gpx_recipe.symmetry.plans import build_plan
    from gpx_recipe.symmetry.compiler import file_sha,verify_inputs
    from gpx_recipe.symmetry.contracts import iter_trials
    catalog,context,settings=inputs(tmp_path)
    (tmp_path/'child.cif').write_text('data_checked')
    c=catalog['candidates'][0]
    c.update(magnetic_space_group=None,child=dict(path='child.cif',sha256=file_sha(tmp_path/'child.cif')),
             modes=['mode'+str(i) for i in range(27)],irrep='complete child')
    (tmp_path/'catalog.json').write_text(json.dumps(catalog))
    settings.update(kind='structural',sites=[],modes=[],start_mode=.1,structural_optimizer='prepared_nonlinear',
                    structural_starts=4,structural_seconds=8.)
    p=build_plan(context,catalog,tmp_path,settings);verify_inputs(p,tmp_path)
    assert len(list(iter_trials(p)))==1
    assert len(p['candidates'][0]['modes'])==27
    assert p['policy']['structural_starts']==4
    assert not p['seeds'][0]['mode_values']
    (tmp_path/'reference.gpx').write_bytes(b'reviewed parent reference')
    settings['structural_reference']=dict(path='reference.gpx',sha256=file_sha(tmp_path/'reference.gpx'))
    reference_plan=build_plan(context,catalog,tmp_path,settings)
    assert 'structural_reference' not in reference_plan['policy']
    assert reference_plan['structural_reference']==settings['structural_reference']
    verify_inputs(reference_plan,tmp_path)
    (tmp_path/'reference.gpx').write_bytes(b'changed reference')
    with pytest.raises(ValueError,match='sealed input changed'):verify_inputs(reference_plan,tmp_path)


def test_packaged_contact_policy_is_relocatable_and_checksum_bound():
    from gpx_recipe.structural_live.optimization.empirical_catalog import load_declared_catalog
    from gpx_recipe.structural_live.optimization import native_fast_candidate
    path=Path(native_fast_candidate.__file__).with_name('empirical_chemistry_policy.json')
    policy=json.loads(path.read_text())
    for key in ('statistics','mp_statistics','cod_statistics','radii'):
        assert not Path(policy[key]).is_absolute()
    assert load_declared_catalog(path).fingerprint


def test_restored_session_rejects_changed_settings_and_unbound_observations(tmp_path):
    from gpx_recipe.structural_live.session import StructuralSession
    from gpx_recipe.structural_live.reduced import ReducedObservation
    prepared=tmp_path/'prepared';prepared.mkdir();(prepared/'prepared.json').write_text('{}')
    def run(p,path,**kw):
        path.mkdir();return saved_result(path,[.1])
    session=StructuralSession(prepared,tmp_path/'session',runner=run)
    good=observation(1);session.update(good)
    with pytest.raises(ValueError,match='changed'):
        StructuralSession(prepared,session.root,settings=SearchSettings(starts=3),runner=run)
    wrong=ReducedObservation(good.acquisition,'another preparation',2,good.x,good.y,
                             good.variance,good.mask,good.normalization_mask)
    before=(session.root/'latest.json').read_bytes()
    with pytest.raises(ValueError,match='bind'):
        session.update(wrong)
    assert (session.root/'latest.json').read_bytes()==before
    restored=StructuralSession(prepared,session.root,runner=run)
    (session.root/restored.record['result']).write_text('{"parameters":[999]}')
    with pytest.raises(ValueError,match='result changed'):
        StructuralSession(prepared,session.root,runner=run)
