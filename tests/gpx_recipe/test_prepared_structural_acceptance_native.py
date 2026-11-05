"""Generated public synthetic native controls, mandatory in the Linux gate.

The child CIF's header records the ISODISTORT generator and parent sites.
No measured dataset, network service or external fixture variable is needed.
"""
import copy
import json
import os
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest

if os.environ.get('RADAR_REQUIRE_STRUCTURAL_NATIVE')=='1':
    import GSASII.GSASIIscriptable as G
else:
    G=pytest.importorskip('GSASII.GSASIIscriptable')

from gpx_recipe.structural_live.native import prepare,NativeCandidate,sha,modules,apply_modes
from gpx_recipe.structural_live.service import prepare_model,fit_snapshot,SearchSettings
from gpx_recipe.structural_live.reduced import from_arrays


@pytest.fixture
def synthetic(tmp_path):
    from gpx_recipe.execution import freeze
    from GSASII import GSASIIstrMain as MAIN
    inst=tmp_path/'synthetic.instprm'
    inst.write_text('#GSAS-II instrument parameter file\nType:PNC\nBank:1.0\nLam:1.54\nZero:0.0\nU:0.12\nV:0.0\nW:0.08\nX:0.0\nY:0.0\nZ:0.0\nSH/L:0.002\nPolariz.:0.0\nAzimuth:0.0\n')
    path=tmp_path/'source.gpx';p=G.G2Project(newgpx=str(path))
    phase=p.add_phase(phasename='Synthetic ErSi',cell=[4.195,10.353,3.779,90,90,90],spacegroup='C m c m')
    phase.add_atom(0.,.413,.25,'Si','Si1');phase.add_atom(0.,.142,.25,'Er','Er1')
    h=p.add_simulated_powder_histogram('synthetic',str(inst),12.,125.,Npoints=701,scale=120.,phases=[phase])
    h.data['Background'][0][2]=3;h.data['Background'][0][3:]=[3.,.2,.1]
    freeze(p);h.data['data'][1][1][:]=1.;p.save()
    y=np.asarray(MAIN.DoNoFit(str(path),h.name),float)
    h.data['data'][1][1]=y.copy();h.data['data'][1][2]=np.full(len(y),4.);p.save()
    child=Path(__file__).with_name('data')/'synthetic-ersi-cmcm-isodistort.cif'
    return path,child,[[-1.,0,0],[0,-1.,0],[0,0,1.]],h.name,phase.name


def test_prepare_fit_export_reopen_and_native_continuation(tmp_path,synthetic):
    source,cif,transform,histogram,phase=synthetic;settings=SearchSettings(seconds=2.,starts=2)
    prepared=tmp_path/'prepared'
    meta=prepare_model(source,cif,transform,[0,0,0],prepared,settings=settings)
    result=fit_snapshot(prepared,tmp_path/'fit',settings=settings)
    assert result['state']=='verified_point_fit'
    assert result['verification']['saved_observations']['continuation']['weighted_objective_verified']
    for role in ('frozen','continuation'):assert sha(result[role]['path'])==result[role]['sha256']
    from gpx_recipe.structural_live.optimization.refinable_gpx import mode_amplitudes,continue_native_refinement
    p=G.G2Project(gpxfile=result['continuation']['path'],newgpx=str(tmp_path/'continued.gpx'))
    current=p.phase(phase)
    target=mode_amplitudes(current.data);_,math,*_=modules()
    apply_modes(current.data,target+np.array([.001,-.001]),math)
    check=continue_native_refinement(p,result['continuation']['contract']['native_active_variables'],phase_name=phase,histogram_name=histogram)
    assert check['objective_verified'] and check['chemistry_feasible']
    np.testing.assert_allclose(mode_amplitudes(current.data),target,rtol=0,atol=2e-6)
    expected=meta['inventory']['physical_rank']+meta['inventory']['cell_rank']+2
    assert len(check['covariance']['varyList'])==expected
    assert np.shape(check['covariance']['covMatrix'])==(expected,expected)


def test_reused_start_bank_matches_every_direct_native_curve(tmp_path,synthetic):
    from gpx_recipe.structural_live.service import make_source
    from gpx_recipe.structural_live.optimization.offline_native_start_bank import build, OfflineNativeStartBank
    source,cif,transform,*_=synthetic;prepared=tmp_path/'prepared'
    prepare(source,cif,transform,[0,0,0],prepared)
    settings=SearchSettings(seconds=2.,starts=2)
    direct_source=make_source(prepared,tmp_path/'direct-context',settings)
    direct=tmp_path/'direct.npz';build(direct_source,direct)
    reused_source=make_source(prepared,tmp_path/'reuse-context',settings)
    reused=tmp_path/'reused.npz';receipt=build(reused_source,reused,reuse_fixed_cell_profiles=True)
    left=OfflineNativeStartBank(direct_source,direct)
    right=OfflineNativeStartBank(reused_source,reused)
    np.testing.assert_array_equal(left.points,right.points)
    np.testing.assert_allclose(left.curves,right.curves,rtol=1e-11,atol=1e-9)
    np.testing.assert_allclose(left.proposal_cell_steps,right.proposal_cell_steps,rtol=0,atol=0)
    assert receipt['construction_execution']['cached_calls']>0
    assert receipt['construction_execution']['diagnostics']['maximum_check_error']<1e-9


def test_sequential_distortion_cell_shift_and_parent_return_recover_native_truth(tmp_path,synthetic):
    from gpx_recipe.structural_live.session import StructuralSession
    source,cif,transform,*_=synthetic;prepared=tmp_path/'prepared'
    settings=SearchSettings(seconds=8.,starts=4)
    metadata=prepare_model(source,cif,transform,[0,0,0],prepared,settings=settings)
    oracle=NativeCandidate(prepared,tmp_path/'oracle')
    m=metadata['inventory']['physical_rank'];c=metadata['inventory']['cell_rank']
    assert m==2 and c==3
    truth=[(np.array([.015,-.012]),np.zeros(c)),
           (np.zeros(m),np.array([.00015,0.,0.])),
           (np.zeros(m),np.zeros(c))]
    observations=[]
    for revision,(modes,cells) in enumerate(truth):
        y=oracle.evaluate(modes,cells);arrays=oracle.hist.data['data'][1]
        observations.append(from_arrays([arrays[0],y,np.full(len(y),4.)],oracle.hist.data['Limits'],
            acquisition='synthetic-native-sequence',revision=revision,protocol_sha256=sha(prepared/'prepared.json')))
    session=StructuralSession(prepared,tmp_path/'session',settings=settings)
    for observation,(modes,cells) in zip(observations,truth):
        result=session.update(observation)
        np.testing.assert_allclose(result['physical_parameters'][:m],modes,rtol=0,atol=1e-4)
        np.testing.assert_allclose(result['physical_parameters'][m:],cells,rtol=0,atol=2e-7)
        assert result['metrics']['rwp_native']<.02
        assert result['starting_curves_reused'] and not result['previous_likelihood_accumulated']
        assert all(check['weighted_objective_verified'] for check in result['verification']['saved_observations'].values())


def test_extra_mask_and_tiny_optional_recovery_preserve_saved_objective(tmp_path,synthetic):
    source,cif,transform,histogram,phase=synthetic;prepared=tmp_path/'prepared'
    settings=SearchSettings(seconds=2.,starts=2,recovery_seconds=1e-6)
    prepare_model(source,cif,transform,[0,0,0],prepared,settings=settings)
    n=NativeCandidate(prepared,tmp_path/'oracle')
    truth=n.evaluate(np.zeros(n.basis['physical_basis'].shape[1]),np.zeros(n.basis['cell_basis'].shape[1]))
    arrays=n.hist.data['data'][1];observed=truth.copy();weights=np.ones(len(truth))
    observed[30]=np.nan;weights[30]=0.
    o=from_arrays([arrays[0],observed,weights],n.hist.data['Limits'],
        acquisition='synthetic',protocol_sha256=sha(prepared/'prepared.json'))
    y=o.y.copy();index=int(np.argmax(np.where(o.mask,y,-np.inf)));y[index]+=1000.;mask=o.mask.copy();mask[index]=False
    o=replace(o,y=y,mask=mask)
    result=fit_snapshot(prepared,tmp_path/'fit',settings=settings,observation=o)
    checks=result['verification']['saved_observations']
    assert all(c['weighted_objective_verified'] for c in checks.values())
    assert checks['continuation']['search_weighted_cost']<1e-8
    assert any(stage.get('incumbent_retained') for stage in result['result']['recovery']['stages'])
    saved=G.G2Project(gpxfile=result['continuation']['path']).histogram(histogram).data['data'][1]
    assert saved[2][index]>0 and saved[2][30]==0


def test_binding_preserves_measurements_outside_fit_limits_and_replaces_exclusions(tmp_path,synthetic):
    source,cif,transform,*_=synthetic;prepared=tmp_path/'prepared'
    prepare(source,cif,transform,[0,0,0],prepared)
    n=NativeCandidate(prepared,tmp_path/'evaluation');data=n.hist.data['data'][1]
    weights=np.asarray(data[2]).copy();n.hist.data['Limits'][1]=[40.,90.]
    o=from_arrays(data,n.hist.data['Limits'],acquisition='a',protocol_sha256=sha(prepared/'prepared.json'))
    mask=o.mask.copy();index=int(np.flatnonzero(mask)[5]);mask[index]=False
    n.bind_observation(replace(o,mask=mask))
    np.testing.assert_array_equal(n.hist.data['data'][1][2],weights)
    assert len(n.hist.data['Limits'])==3
    n.bind_observation(o)
    assert len(n.hist.data['Limits'])==2


def test_worker_audit_reconstructs_declared_freedom_and_bounds(tmp_path,synthetic):
    from gpx_recipe.symmetry.native_audit import audit_trial,inspect_source
    from gpx_recipe.symmetry.compiler import compile_bundle
    from gpx_recipe.symmetry.contracts import digest
    from gpx_recipe.symmetry.worker import execute,TrialDisposition
    from test_symmetry_plans import inputs
    from gpx_recipe.symmetry.plans import build_plan
    source,cif,transform,histogram,phase=synthetic
    geometry=tmp_path/'geometry';meta=prepare(source,cif,transform,[0,0,0],geometry)
    root=tmp_path/'inputs';root.mkdir();catalog,_,settings=inputs(root)
    (root/'source.gpx').write_bytes(source.read_bytes());source=root/'source.gpx'
    local=root/'child.cif';local.write_bytes(cif.read_bytes())
    c=catalog['candidates'][0];c.update(magnetic_space_group=None,bns_number=None,
        transformation_matrix=transform,child=dict(path=local.name,sha256=sha(local)),
        modes=meta['inventory']['native_mode_names'],irrep='synthetic displacement modes')
    context=inspect_source(source);catalog['parent_setting_sha256']=context['phases'][0]['setting_sha256']
    (root/'catalog.json').write_text(json.dumps(catalog))
    settings.update(kind='structural',parent=phase,histogram=histogram,sites=[],modes=[],
        structural_optimizer='prepared_nonlinear',structural_starts=2,structural_seconds=2.)
    protocol=build_plan(context,catalog,root,settings)
    job=tmp_path/'worker'
    compile_bundle(protocol,root,job,candidate_id=protocol['candidates'][0]['id'],
        seed_id=protocol['seeds'][0]['id'],approved_hash=digest(protocol))
    result=execute(job)
    assert result['status']=='optimization_unresolved',result
    assert result['inputs_verified'] and result['effects_verified'] and result['audit']['passed']
    root=job;source=root/'source.gpx';out=job/'results'
    output=out/result['gpx'];original=output.read_bytes()
    good=audit_trial(source,output,protocol,protocol['candidates'][0],input_root=root)
    assert good['passed'] and good['checks']['positive_bounded_common_scale']
    for mutation in ('negative_scale','cell_flag_false','hidden_cell','mode_bound','cell_bound','background_bound','phase_scale','extra_phase','species'):
        output.write_bytes(original);p=G.G2Project(gpxfile=str(output));h=p.histogram(histogram);child=p.phase(phase)
        scope=copy.deepcopy(protocol)
        if mutation=='negative_scale':h.data['Sample Parameters']['Scale'][0]=-1.
        elif mutation=='cell_flag_false':scope['policy']['refine_cell']=False
        elif mutation=='hidden_cell':child.set_refinements({'Cell':False})
        elif mutation=='mode_bound':
            _,math,*_=modules();apply_modes(child.data,np.array([1.,0.]),math)
        elif mutation=='cell_bound':child.data['General']['Cell'][1]*=1.01
        elif mutation=='background_bound':h.data['Background'][0][3]=1e6
        elif mutation=='phase_scale':child.data['Histograms'][histogram]['Scale'][0]*=.8
        elif mutation=='species':
            ct=child.data['General']['AtomPtrs'][1];child.data['Atoms'][0][ct]='Al'
            G.SetupGeneral(child.data,None)
        else:p.add_phase(phasename='Undeclared additional phase',cell=[5,5,5,90,90,90])
        p.save()
        with pytest.raises(TrialDisposition):audit_trial(source,output,scope,scope['candidates'][0],input_root=root)


def test_native_session_restores_relocates_and_reuses_without_new_fit(tmp_path,synthetic):
    from gpx_recipe.structural_live.session import StructuralSession
    source,cif,transform,*_=synthetic;prepared=tmp_path/'prepared'
    settings=SearchSettings(seconds=2.,starts=2)
    prepare_model(source,cif,transform,[0,0,0],prepared,settings=settings)
    n=NativeCandidate(prepared,tmp_path/'observation')
    o=from_arrays(n.hist.data['data'][1],n.hist.data['Limits'],acquisition='session-control',protocol_sha256=sha(prepared/'prepared.json'))
    session=StructuralSession(prepared,tmp_path/'session',settings=settings)
    result=session.update(o)
    restored=StructuralSession(prepared,session.root,settings=settings)
    assert restored.update(o)['state']=='duplicate'
    receipt=json.loads((session.root/session.record['result']).read_text())
    assert receipt['verification']['saved_observations']['continuation']['weighted_objective_verified']
