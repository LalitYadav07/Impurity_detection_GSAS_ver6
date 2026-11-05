"""Regressions for the independent 088cb51 review; no private data."""
import json
import shutil
from dataclasses import replace
from types import SimpleNamespace as NS
import numpy as np
import pytest

from gpx_recipe.structural_live.reduced import from_arrays,native_limits,verify_native_observation


def test_magnetic_setup_assistant_cannot_silently_replace_a_structural_search(monkeypatch):
    import asyncio
    from radar_pd_nova import symmetry_controller as module
    app=module.SymmetryMixin()
    state=NS(sx_kind='structural',sx_request='Find a distorted nuclear structure',flush=lambda:None)
    app.server=NS(state=state)
    app._clear_symmetry_setup_details=lambda:None
    app._refinement_schedule=lambda coroutine,name:asyncio.run(coroutine)
    monkeypatch.setattr(module,'make_intent',lambda *a,**kw:pytest.fail('Magnetic planner called for structural search'))
    app.plan_symmetry_request()
    assert state.sx_kind=='structural' and not state.sx_busy
    assert 'structural controls' in state.sx_error


def test_extra_exclusion_preserves_measurement_and_normalization_domains():
    x=np.arange(9.);y=np.ones(9);y[4]=1000.
    limits=[[0,8],[1.2,6.4]]
    original=from_arrays([x,y,np.full(9,4.)],limits,acquisition='a',protocol_sha256='b')
    mask=original.mask.copy();mask[4]=False
    observation=replace(original,mask=mask)
    converted=native_limits(observation,limits)
    assert converted[-1]==[4.,4.]
    actual=from_arrays([x,y,np.full(9,4.)],converted,acquisition='a',protocol_sha256='b')
    np.testing.assert_array_equal(actual.mask,mask)
    assert actual.measurement_mask.all() and actual.normalization_mask[4]
    receipt=verify_native_observation(observation,[x,y,np.full(9,4.)],converted,np.ones(9),0.,weight_factor=2.)
    assert receipt['weighted_objective_verified']
    with pytest.raises(ValueError,match='domains'):
        verify_native_observation(observation,[x,y,np.full(9,4.)],limits,np.ones(9),0.)
    with pytest.raises(ValueError,match='objective'):
        verify_native_observation(original,[x,y,np.full(9,4.)],limits,np.ones(9),0.)
    bad=observation.normalization_mask.copy();bad[4]=False
    with pytest.raises(ValueError,match='normalization'):
        native_limits(replace(observation,normalization_mask=bad),limits)


def test_masked_abscissa_is_excluded_but_keeps_measured_normalization():
    x=np.ma.array(np.arange(5.),mask=[0,0,1,0,0])
    o=from_arrays([x,np.ones(5),np.ones(5)],[[0,4],[0,4]],acquisition='a',protocol_sha256='b')
    assert not o.mask[2] and o.normalization_mask[2] and o.measurement_mask[2]
    assert native_limits(o,[[0,4],[0,4]])[-1]==[2.,2.]


@pytest.mark.parametrize('det,maximum,accepted',[(1,1,True),(2,2,True),(2,1,False),(0,1,False)])
def test_cell_expansion_admission_precedes_both_native_executors(det,maximum,accepted,monkeypatch):
    from gpx_recipe.symmetry.contracts import structural_cell_budget
    from gpx_recipe.symmetry import native_structural,native_prepared_structural
    from gpx_recipe.symmetry.worker import TrialDisposition
    c=dict(transformation_matrix=[[det,0,0],[0,1,0],[0,0,1]])
    p=dict(maximum_cell_multiplier=maximum)
    if accepted:
        assert structural_cell_budget(c,p)==det
    else:
        with pytest.raises(ValueError):structural_cell_budget(c,p)
        def forbidden(*a,**kw):raise AssertionError('Native work entered before admission')
        monkeypatch.setattr(native_structural,'gsas',forbidden)
        monkeypatch.setattr(native_prepared_structural,'prepare_model',forbidden)
        for executor in (native_structural,native_prepared_structural):
            with pytest.raises(TrialDisposition):executor.run(None,None,dict(policy=p),c,{})


def test_protocol_admission_rejects_oversized_child(tmp_path):
    from test_symmetry_plans import inputs
    from gpx_recipe.symmetry.plans import build_plan
    from gpx_recipe.symmetry.compiler import file_sha
    from gpx_recipe.symmetry.contracts import validate_protocol
    catalog,context,settings=inputs(tmp_path)
    child=tmp_path/'child.cif';child.write_text('data_synthetic')
    catalog['candidates'][0].update(magnetic_space_group=None,child=dict(path=child.name,sha256=file_sha(child)),modes=['m0'])
    (tmp_path/'catalog.json').write_text(json.dumps(catalog))
    settings.update(kind='structural',sites=[],modes=[],structural_optimizer='prepared_nonlinear')
    p=build_plan(context,catalog,tmp_path,settings)
    p['policy']['maximum_cell_multiplier']=1
    p['candidates'][0]['transformation_matrix']=[[2.,0,0],[0,2.,0],[0,0,2.]]
    with pytest.raises(ValueError,match='cell-expansion'):validate_protocol(p)


@pytest.mark.parametrize('dimension',[255,256,257])
def test_dimension_limit_switches_to_native_without_dropping_coordinates(dimension,monkeypatch):
    import gpx_recipe.structural_live.optimization.native_cell_intensity as module
    calls=[]
    source=NS(radii=np.ones(dimension),mode_dimensions=dimension-1,cell_dimensions=1,
        fast=NS(native=NS(_check=lambda:None,math=NS(StructureFactor2=None))),
        check=lambda p:p,native=lambda p:calls.append(p.copy()) or np.arange(3.))
    def stop(*a,**kw):raise RuntimeError('cache entry reached')
    source.check_domain=lambda p:p
    monkeypatch.setattr(module,'NativeProfileBank',stop)
    if dimension<=256:
        with pytest.raises(RuntimeError,match='cache entry'):module.ExactCellIntensity(source).evaluate(np.zeros(dimension))
    else:
        np.testing.assert_array_equal(module.ExactCellIntensity(source).evaluate(np.zeros(dimension)),np.arange(3.))
        assert len(calls)==1 and len(calls[0])==257


@pytest.mark.parametrize('tamper',['receipt','missing_frozen','changed_continuation'])
@pytest.mark.parametrize('operation',['restore','duplicate','warm'])
def test_session_checks_all_artifacts_on_every_reuse(tmp_path,tamper,operation):
    from test_structural_reduced_service import saved_result,observation
    from gpx_recipe.structural_live.session import StructuralSession,PublicationUnavailable
    p=tmp_path/'prepared';p.mkdir();(p/'prepared.json').write_text('{}')
    calls=[]
    def runner(_,out,**kw):
        calls.append(kw);out.mkdir();return saved_result(out,[.1])
    session=StructuralSession(p,tmp_path/'session',runner=runner)
    result=session.update(observation(1));before=(session.root/'latest.json').read_bytes()
    if tamper=='receipt':(session.root/session.record['result']).write_text('{}')
    elif tamper=='missing_frozen':__import__('pathlib').Path(result['frozen']['path']).unlink()
    else:__import__('pathlib').Path(result['continuation']['path']).write_bytes(b'changed')
    with pytest.raises(PublicationUnavailable):
        if operation=='restore':StructuralSession(p,session.root,runner=runner)
        else:session.update(observation(1 if operation=='duplicate' else 2))
    assert len(calls)==1 and (session.root/'latest.json').read_bytes()==before


def test_session_artifact_paths_relocate_without_rewriting_sealed_receipt(tmp_path):
    from test_structural_reduced_service import saved_result,observation
    from gpx_recipe.structural_live.session import StructuralSession
    p=tmp_path/'prepared';p.mkdir();(p/'prepared.json').write_text('{}')
    def runner(_,out,**kw):out.mkdir();return saved_result(out,[.1])
    session=StructuralSession(p,tmp_path/'session',runner=runner);session.update(observation(1))
    target=tmp_path/'relocated';shutil.move(session.root,target)
    restored=StructuralSession(p,target,runner=runner)
    duplicate=restored.update(observation(1))
    assert all(__import__('pathlib').Path(path).is_relative_to(target) for path in duplicate['artifacts'].values())


@pytest.mark.parametrize('failure',['missing','escape','false_digest'])
def test_failed_publication_retains_previous_fit(tmp_path,failure):
    from test_structural_reduced_service import saved_result,observation
    from gpx_recipe.structural_live.session import StructuralSession,PublicationUnavailable
    p=tmp_path/'prepared';p.mkdir();(p/'prepared.json').write_text('{}')
    def runner(_,out,**kw):
        out.mkdir();result=saved_result(out,[.1])
        if kw['observation'].revision==2:
            artifact=__import__('pathlib').Path(result['continuation']['path'])
            if failure=='missing':artifact.unlink()
            elif failure=='escape':
                artifact.unlink();artifact.symlink_to(tmp_path/'session'/'latest.json')
            else:result['continuation']['sha256']='0'*64
        return result
    session=StructuralSession(p,tmp_path/'session',runner=runner);session.update(observation(1))
    before=(session.root/'latest.json').read_bytes()
    with pytest.raises(PublicationUnavailable):session.update(observation(2))
    assert (session.root/'latest.json').read_bytes()==before
    assert session.update(observation(1))['state']=='duplicate'


@pytest.mark.parametrize('stage',[0,1])
def test_recovery_timeout_retains_checked_incumbent_but_integrity_error_propagates(monkeypatch,stage):
    from test_structural_live_offline_start_bank import source
    from gpx_recipe.structural_live.optimization import automatic_fit_recovery as module
    from gpx_recipe.structural_live.optimization.feasible_gauss_newton import EvaluationBudgetExpired
    from gpx_recipe.structural_live.optimization import quadratic_start_proposals
    s,t,o,_=source(monkeypatch);z=np.zeros(len(s.radii)+2);calls=[]
    def fit(*a,**kw):
        i=len(calls);calls.append(i)
        if i==stage:raise EvaluationBudgetExpired('deadline')
        rate=s.native(z[:-2]*s.radii)
        residual,_=module.residual_and_factor(o,rate)
        return dict(best=dict(parameters=z.tolist(),deviance=float(residual@residual))),rate
    monkeypatch.setattr(module,'fit_feasible_gn',fit)
    monkeypatch.setattr(quadratic_start_proposals,'explore',lambda *a,**kw:([],{}))
    result,prediction=module.improve(s,t,o,z,policy=module.RecoveryPolicy(local_seconds=.01,proposal_seconds=.01,joint_seconds=.01))
    assert result['stages'][stage*2]['incumbent_retained']
    residual,_=module.residual_and_factor(o,prediction)
    assert result['best']['deviance']==pytest.approx(residual@residual)
    def broken(*a,**kw):raise ValueError('native identity changed')
    monkeypatch.setattr(module,'fit_feasible_gn',broken)
    with pytest.raises(ValueError,match='identity'):
        module.improve(s,t,o,z)


@pytest.mark.parametrize('field',['sx_max_mode','sx_start_mode','sx_cycles','sx_structural_optimizer'])
def test_prepared_form_ignores_hidden_legacy_fields_but_validates_visible_settings(monkeypatch,tmp_path,field):
    from radar_pd_nova import symmetry_controller as module
    state=NS(sx_error='',sx_adequate=5.,sx_kind='structural',sx_parent='parent',sx_histogram='bank',sx_k='0,0,0',
        sx_candidates=['child'],sx_modes=[],sx_directions=['x'],sx_max_mode='',sx_start_mode='',sx_cycles='',
        sx_sign_starts=False,sx_structural_optimizer='prepared_nonlinear',sx_structural_starts=2,sx_structural_seconds=8.,
        sx_structural_radius=.05,sx_structural_cells=.0008,sx_structural_recovery=0.,sx_parent_reference='',
        sx_magnetic_optimizer='gsas_hessian',sx_optimizer_iterations=60)
    setattr(state,field,'invalid hidden legacy value')
    app=module.SymmetryMixin();app.server=NS(state=state)
    app._model_form=lambda payload:'form';app._discovery_identity=lambda:'identity'
    app._symmetry_draft=dict(discovery_identity='identity',source_id='scan',context={},catalog={},root=tmp_path)
    reached=[]
    def stop(context,catalog,root,settings):
        reached.append(settings);raise RuntimeError('Reached plan validation')
    monkeypatch.setattr(module,'build_plan',stop)
    app.build_symmetry_plan()
    assert len(reached)==1 and reached[0]['cycles']==12
    assert reached[0]['structural_optimizer']=='prepared_nonlinear'
    state.sx_structural_radius='invalid visible value';app.build_symmetry_plan()
    assert len(reached)==1 and state.sx_error
