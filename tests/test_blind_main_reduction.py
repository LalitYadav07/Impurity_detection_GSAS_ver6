"""Independent reduced-model recovery; no private experimental inputs."""
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import blind_main_reduction as reduction


def evidence_case():
    phases = [SimpleNamespace(name=name, data={'Histograms': {'h': {'Use': True, 'Scale': [scale, True]}}})
              for name, scale in [('blind', 0.), ('survivor', 1.)]]
    project = SimpleNamespace(phases=lambda: phases, data={'Covariance': {'data': {'freshCOV': True}}})
    rows = {name: {'phase_fraction_pct': value, 'weight_fraction_pct': value}
            for name, value in [('blind', 0.), ('survivor', 100.)]}
    return project, phases, rows


def test_complete_native_zero_proposes_only_one_independent_hypothesis():
    project, _, rows = evidence_case()
    result = reduction.collapse_evidence(project, 'h', 'blind', rows, label='trial')
    assert result['proposal_only'] is True and result['survivor'] == 'survivor'
    assert result['rows']['blind']['hap_scale'] == 0


@pytest.mark.parametrize('bad', ['missing_row','missing_value','nan','missing_scale','positive_scale','stale_covariance','inactive'])
def test_incomplete_or_nonzero_evidence_never_triggers(bad):
    project, phases, rows = evidence_case()
    if bad == 'missing_row':rows.pop('blind')
    if bad == 'missing_value':rows['blind'].pop('weight_fraction_pct')
    if bad == 'nan':rows['blind']['weight_fraction_pct'] = float('nan')
    if bad == 'missing_scale':phases[0].data['Histograms']['h'].pop('Scale')
    if bad == 'positive_scale':phases[0].data['Histograms']['h']['Scale'][0] = 1e-8
    if bad == 'stale_covariance':project.data['Covariance']['data']['freshCOV'] = False
    if bad == 'inactive':phases[0].data['Histograms']['h']['Use'] = False
    assert reduction.collapse_evidence(project, 'h', 'blind', rows, label='trial') is None


def test_known_main_is_protected_without_opening_project(tmp_path):
    result = reduction.try_reduced_main(tmp_path/'absent.gpx',tmp_path/'hypothesis','blind',
        [{'schema':'radar-blind-main-collapse/v1','main':'blind'}],user_supplied_main=True)
    assert result['reason'] == 'user_supplied_main_protected' and not result['attempted']


def test_forged_or_incomplete_proposal_cannot_start_native_attempt(tmp_path):
    value={'schema':'radar-blind-main-collapse/v1','main':'blind','survivor':'survivor'}
    result=reduction.try_reduced_main(tmp_path/'absent.gpx',tmp_path/'trial','blind',[value],user_supplied_main=False)
    assert not result['attempted'] and result['reason']=='no_authoritative_zero_scale_proposal'


def test_promotion_updates_main_fractions_and_publication_identity_together(tmp_path):
    from scripts.ndip_outputs import _publish_gpx
    from scripts.gsas_main_phase_refiner import reject_gpx_checkpoint
    receipt={'adopted':True,'removed_phase':'blind','main_after':'survivor',
             'output_gpx':'single.gpx','main_cif':'single.cif','rwp':3.,
             'fractions':{'survivor':{'weight_fraction_pct':100.}}}
    accepted=['survivor'];labels={'blind':'Old hypothesis','survivor':'Surviving hypothesis'}
    values=reduction.promotion_values(receipt,'blind',accepted,labels)
    assert values[2]==receipt['fractions'] and values[3]=='survivor' and values[5]==[]
    assert values[6]=={'survivor':'Surviving hypothesis'}
    assert accepted==['survivor'] and 'blind' in labels
    folder=tmp_path/'Technical/GSAS_Projects';folder.mkdir(parents=True)
    old=folder/'seq_final_joint_polished.gpx';old.write_bytes(b'old model')
    old.with_suffix('.lst').write_bytes(b'old fractions')
    reject_gpx_checkpoint(str(old))
    saved=old.with_suffix('.rejected.gpx')
    assert saved.read_bytes()==b'old model' and not _publish_gpx(saved,tmp_path)
    final=folder/'seq_final_accepted_model.gpx';final.write_bytes(b'new sole model')
    assert _publish_gpx(final,tmp_path)


@pytest.mark.parametrize('adopted',[False,True])
def test_attempted_main_hypothesis_remains_visible_in_published_review(tmp_path,adopted):
    from scripts.ndip_outputs import collect_outputs
    run=tmp_path/'run';run.mkdir()
    (run/'run_manifest.json').write_text(json.dumps({'status':'complete','metrics':{
        'blind_main_reduction':{'attempted':True,'adopted':adopted,
            'main_before':'blind','main_after':'survivor','reason':'native decision'}}}))
    result=collect_outputs(run,tmp_path/'portal',mode='full',include_archive=False)
    warning=next(row for row in result['warnings'] if row['code']=='blind_main_hypothesis_review')
    assert warning['adopted']==adopted and result['scientific_status']=='needs_review'


def synthetic(tmp_path, kind):
    G = pytest.importorskip('GSASII.GSASIIscriptable')
    from GSASII import GSASIIstrMain as native
    from gpx_recipe.execution import freeze
    parameters = ('Lam:1.54\nZero:0\nU:.12\nV:0\nW:.08\nX:0\nY:0\nZ:0\nSH/L:.002\nPolariz.:0\nAzimuth:0\n'
                  if kind=='PNC' else 'difC:6200\ndifA:2\ndifB:.4\nZero:1\n2-theta:105\nfltPath:15.5\nalpha:.99\nbeta-0:.02\nbeta-1:.03\nbeta-q:0\nsig-0:1\nsig-1:66\nsig-2:0\nsig-q:0\nX:0\nY:0\nZ:0\nAzimuth:0\n')
    profile = tmp_path/'profile.instprm'
    profile.write_text('#GSAS-II instrument parameter file\nType:'+kind+'\nBank:1\n'+parameters)
    source = tmp_path/'source.gpx'
    project = G.G2Project(newgpx=str(source))
    good = project.add_phase(phasename='survivor',spacegroup='F m -3 m',cell=[4.6,4.6,4.6,90,90,90])
    good.add_atom(0,0,0,'Mg','Mg1',uiso=.01)
    good.add_atom(.5,.5,.5,'O','O1',uiso=.01)
    limits = (12,125) if kind=='PNC' else (3,40)
    hist = project.add_simulated_powder_histogram('public synthetic '+kind,str(profile),*limits,
                                                 Npoints=1201,scale=1000,phases=[good])
    hist.data['Background'][0][2]=3
    hist.data['Background'][0][3:]=[25.,0.,0.]
    hist.data['data'][1][1][:]=1.
    freeze(project);project.save()
    truth=np.asarray(native.DoNoFit(str(source),hist.name),float)
    sigma=np.sqrt(np.maximum(truth,1.))
    hist.data['data'][1][1][:]=truth+np.random.default_rng(8272).normal(size=len(truth))*sigma
    hist.data['data'][1][2][:]=1./sigma**2
    hist.data['data'][0]['Dummy']=False
    # Include a gap and explicit boundary exclusions to check exact ownership.
    hist.data['data'][1][2][300:303]=0
    x=hist.data['data'][1][0]
    hist.data['Limits'].append([float(x[501]),float(x[510])])
    bad=project.add_phase(phasename='blind',spacegroup='F m -3 m',cell=[5.7,5.7,5.7,90,90,90],histograms=[hist])
    bad.add_atom(0,0,0,'Mg','Mg1',uiso=.01)
    bad.add_atom(.5,.5,.5,'O','O1',uiso=.01)
    bad.data['Histograms'][hist.name]['Scale'][0]=.5
    good.data['Histograms'][hist.name]['Scale'][0]=.5
    freeze(project);project.data['Controls']['data']['max cyc']=0
    project.data['Controls']['data']['deriv type']='analytic Hessian'
    project.save();project.do_refinements([{}]);project.save()
    return source


@pytest.mark.parametrize('kind',['PNC','PNT'])
def test_native_single_model_recovery_preserves_original_and_gaps(tmp_path,kind):
    from gpx_recipe.snapshot import gsas
    source=synthetic(tmp_path,kind)
    before=sha256(source.read_bytes()).hexdigest()
    project=gsas().G2Project(gpxfile=str(source))
    original=reduction._observation_state(project)
    fake,_,rows=evidence_case()
    trigger=reduction.collapse_evidence(fake,'h','blind',rows,label='synthetic zero-scale control')
    result=reduction.try_reduced_main(source,tmp_path/'reduction','blind',[trigger],user_supplied_main=False)
    assert result['adopted'], result.get('error', result)
    assert result['rwp'] < result['baseline_rwp']
    completed=[row for row in result['native_stages'] if row['status']=='completed']
    assert completed and all(row['convergence']['native_converged'] is True for row in completed)
    assert result['retained_stage']==completed[-1]['stage']
    assert result['cell_refined']==completed[-1]['cell_refined']
    final=gsas().G2Project(gpxfile=result['output_gpx'])
    assert [p.name for p in final.phases()] == ['survivor']
    assert reduction._observation_state(final) == original
    assert sha256(source.read_bytes()).hexdigest() == before
    repeated=reduction.try_reduced_main(source,tmp_path/'reduction','blind',[trigger],user_supplied_main=False)
    assert not repeated['attempted'] and repeated['reason']=='prior_attempt_not_repeated'


def test_rejected_native_trial_never_replaces_baseline(tmp_path,monkeypatch):
    source=synthetic(tmp_path,'PNC');before=source.read_bytes()
    monkeypatch.setattr(reduction,'_fit_single_model',lambda *a,**k: (_ for _ in ()).throw(ValueError('nonconverged')))
    fake,_,rows=evidence_case()
    trigger=reduction.collapse_evidence(fake,'h','blind',rows,label='synthetic zero-scale control')
    result=reduction.try_reduced_main(source,tmp_path/'reduction','blind',[trigger],user_supplied_main=False)
    assert result['attempted'] and not result['adopted']
    assert result['error'] == 'nonconverged'
    assert source.read_bytes() == before


def test_failed_cell_stage_restores_only_verified_scale_background(tmp_path,monkeypatch):
    from gpx_recipe.snapshot import gsas
    import gpx_recipe.convergence
    source=synthetic(tmp_path,'PNC');before=source.read_bytes()
    original=gpx_recipe.convergence.require_convergence
    calls=[]
    def strict_check(evidence,policy):
        calls.append(evidence)
        if len(calls)==2:
            raise ValueError('synthetic strict cell convergence rejection')
        return original(evidence,policy)
    monkeypatch.setattr(gpx_recipe.convergence,'require_convergence',strict_check)
    fake,_,rows=evidence_case();trigger=reduction.collapse_evidence(fake,'h','blind',rows,label='test')
    folder=tmp_path/'reduction'
    result=reduction.try_reduced_main(source,folder,'blind',[trigger],user_supplied_main=False)
    assert result['adopted'] and result['retained_stage']=='scale_background'
    assert result['cell_refined'] is False
    assert result['native_stages'][1]['status']=='rejected'
    assert result['rwp']==result['native_stages'][0]['rwp']
    assert Path(result['output_gpx']).read_bytes()==(folder/'scale_background_verified.gpx').read_bytes()
    final=gsas().G2Project(gpxfile=result['output_gpx'])
    assert final.phases()[0].data['General']['Cell'][0] is False
    assert source.read_bytes()==before


def test_actual_native_scale_and_listing_capture_zero_without_inference(tmp_path):
    from gpx_recipe.snapshot import gsas
    from gpx_recipe.execution import freeze
    from scripts.gsas_main_phase_refiner import parse_gsas_lst
    source=synthetic(tmp_path,'PNC');project=gsas().G2Project(gpxfile=str(source))
    hist=project.histograms()[0]
    project.phase('blind').data['Histograms'][hist.name]['Scale'][0]=0.
    freeze(project)
    hist.set_refinements({'Sample Parameters':['Scale'],'Background':{'refine':True}})
    project.data['Controls']['data']['max cyc']=10
    project.save();project.do_refinements([{}]);project.save()
    rows=parse_gsas_lst(source.with_suffix('.lst'),hist.name)
    result=reduction.collapse_evidence(project,hist.name,'blind',rows,label='native zero control')
    assert result is not None, {'rows': rows, 'fresh': project.data['Covariance']['data'].get('freshCOV'),
                                'hap': {p.name:p.data['Histograms'][hist.name]['Scale'] for p in project.phases()}}
    assert result['rows']['blind']['hap_scale']==reduction.NATIVE_PHASE_SCALE_FLOOR
    rows.pop('blind')
    assert reduction.collapse_evidence(project,hist.name,'blind',rows,label='missing row') is None


def test_profile_requiring_trim_is_rejected_without_changing_baseline(tmp_path,monkeypatch):
    import gpx_recipe.tof_domain
    source=synthetic(tmp_path,'PNT');before=source.read_bytes()
    monkeypatch.setattr(gpx_recipe.tof_domain,'assess_tof_domain',lambda *a: {'trimmed_lower':True,'native_limits':[0,1]})
    fake,_,rows=evidence_case();trigger=reduction.collapse_evidence(fake,'h','blind',rows,label='test')
    result=reduction.try_reduced_main(source,tmp_path/'reduction','blind',[trigger],user_supplied_main=False)
    assert not result['adopted'] and 'changed fitted domain' in result['error']
    assert source.read_bytes()==before


def test_unbounded_numeric_method_is_rejected_before_new_fit(tmp_path):
    from gpx_recipe.snapshot import gsas
    source=synthetic(tmp_path,'PNC');project=gsas().G2Project(gpxfile=str(source))
    project.data['Controls']['data']['deriv type']='numeric';project.save()
    before=source.read_bytes()
    fake,_,rows=evidence_case();trigger=reduction.collapse_evidence(fake,'h','blind',rows,label='test')
    result=reduction.try_reduced_main(source,tmp_path/'reduction','blind',[trigger],user_supplied_main=False)
    assert not result['adopted'] and 'cycle budget' in result['error']
    assert source.read_bytes()==before
