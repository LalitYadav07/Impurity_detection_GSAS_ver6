"""Prepared construction is a sealed hint, never a substitute audit source."""
import copy
from types import SimpleNamespace

import pytest

from gpx_recipe.symmetry.prepared_magnetic import seal_prepared,verify_prepared


@pytest.fixture
def capture(tmp_path):
    directory=tmp_path/'trials'/'owner'/'group-data'/'prepared'
    directory.mkdir(parents=True)
    gpx=directory/'magnetic-refinement.gpx';gpx.write_bytes(b'prepared geometry')
    protocol=dict(source={'sha256':'a'*64},scope='one source')
    candidate={'id':'one candidate'}
    ref=seal_prepared(SimpleNamespace(run_dir=directory),protocol,candidate,
        {'schema':'seed-template'},{'magnetic_elements':{'Ni'}},gpx)
    output=tmp_path/'trials'/'different-start'/'results';output.mkdir(parents=True)
    ctx=SimpleNamespace(run_dir=output)
    return directory,ref,ctx,protocol,candidate


def test_prepared_capture_verifies_only_inside_its_dispatch(capture):
    directory,ref,ctx,protocol,candidate=capture
    root,manifest=verify_prepared(ref,ctx,protocol,candidate)
    assert root==directory
    assert manifest['source_sha256']=='a'*64
    assert 'magnetic-refinement.gpx' in manifest['files']


@pytest.mark.parametrize('artifact',['prepared-state.json','magnetic-refinement.gpx','prepared-manifest.json'])
def test_changed_prepared_artifact_is_rejected(capture,artifact):
    directory,ref,ctx,protocol,candidate=capture
    (directory/artifact).write_text('changed')
    with pytest.raises(ValueError,match='changed'):
        verify_prepared(ref,ctx,protocol,candidate)


@pytest.mark.parametrize('change',['source','protocol','candidate'])
def test_prepared_capture_cannot_cross_input_or_model_boundaries(capture,change):
    _,ref,ctx,protocol,candidate=capture
    protocol=copy.deepcopy(protocol);candidate=dict(candidate)
    if change=='source':protocol['source']['sha256']='b'*64
    if change=='protocol':protocol['scope']='another policy'
    if change=='candidate':candidate['id']='another model'
    with pytest.raises(ValueError,match='approved model'):
        verify_prepared(ref,ctx,protocol,candidate)


def test_prepared_capture_cannot_be_taken_from_another_dispatch(capture,tmp_path):
    _,ref,_,protocol,candidate=capture
    output=tmp_path/'another-dispatch'/'trial'/'results';output.mkdir(parents=True)
    with pytest.raises(ValueError):
        verify_prepared(ref,SimpleNamespace(run_dir=output),protocol,candidate)


@pytest.mark.parametrize('sites,masked',[(1,False),(2,True)])
def test_native_prepared_copy_matches_independent_reconstruction(tmp_path,sites,masked):
    """Public P1 CW controls complement the private TOF multi-start pilot."""
    pytest.importorskip('GSASII.GSASIIscriptable')
    import numpy as np
    from test_native_profiled import artificial_project
    from gsas_magnetic_preflight import setting_fingerprint
    from gpx_recipe.symmetry.compiler import file_sha
    from gpx_recipe.symmetry.native_magnetic import run,prepare
    from gpx_recipe.symmetry.prepared_magnetic import run_prepared
    from gpx_recipe.symmetry.native_profiled import NativeMagneticStartingPoint
    from gpx_recipe.symmetry.profiled_groups import native_seed
    from gpx_recipe.symmetry.native_audit import audit_trial
    from gpx_recipe.symmetry.worker import Artifacts

    project,source=artificial_project(tmp_path,sites=sites,masked=masked)
    nuclear,magnetic=project.phases()
    del project.data['Phases'][magnetic.name]
    for names in project.names:
        if names[0]=='Phases':names.remove(magnetic.name)
    # Real GSAS-II saves can retain the removed phase's output cache. Both
    # independently reconstructed and prepared fits must ignore that orphan
    # while keeping all active nuclear/magnetic scattering dependencies.
    assert magnetic.name in project.histograms()[0].data['Reflection Lists']
    project.save()
    candidate=dict(id='public-p1',magnetic_space_group='P1',bns_number='1.1',
        bns_lattice='',origin_shift=[0.,0.,0.],transformation_matrix=np.eye(3).tolist())
    policy=dict(moment_model='symmetry_allowed',magnetic_optimizer='profiled_quadratic',
        optimizer_iterations=100,cycles=20,timeout_seconds=120,maximum_moment=6.,
        refine_histogram_scale=True,refine_background=True,refine_cell=False,
        maximum_cell_multiplier=8,site_policy='all',magnetic_d_min=0)
    protocol=dict(kind='magnetic',source={'sha256':file_sha(source)},
        parent_phase=nuclear.name,histogram=project.histograms()[0].name,
        parent_setting_sha256=setting_fingerprint(nuclear.data['General']['SGData']),
        sites=[dict(label=f'Fe{i+1}',form_factor='Fe+3',lande_g=2.) for i in range(sites)],
        policy=policy,k=['0','0','0'])
    initial=dict(id='preparation',axis='x',magnitude=1.,parent_direction=None,
        independent_mode_seed=1,site_signs={})
    selected=dict(initial,id='different-start',axis='y',magnitude=1.8,
        independent_mode_seed=7)
    full=tmp_path/'reconstructed';full.mkdir()
    direct=run(source,Artifacts(full,input_root=tmp_path),protocol,candidate,selected)
    prepared=tmp_path/'trials'/'owner'/'group-data'/'prepared';prepared.mkdir(parents=True)
    built=prepare(source,Artifacts(prepared,input_root=tmp_path),protocol,candidate,initial)
    context=NativeMagneticStartingPoint(prepared/built['gpx'],refine_scale=True,
        refine_background=True,maximum_moment=6.)
    model,_=context.build()
    labels=[context.phases[name]['Atoms'][i][context.phases[name]['General']['AtomPtrs'][1]-1]
        for name,i,_ in context.atoms]
    start=native_seed(built['seed_template'],selected,context.component_maps,labels)
    fit=model.fit(start,max_iterations=100,objective_ftol=1e-14)
    reused=tmp_path/'trials'/'different-start'/'results';reused.mkdir(parents=True)
    result=run_prepared(source,Artifacts(reused,input_root=tmp_path),protocol,candidate,
        selected,screened=dict(fit=fit,magnetic_variables=context.moment_names,prepared=built['prepared']))
    assert direct['status']==result['status']
    assert direct['rwp']==pytest.approx(result['rwp'],rel=1e-9,abs=1e-9)
    np.testing.assert_allclose([m['moment'] for m in direct['moments']],
        [m['moment'] for m in result['moments']],rtol=1e-8,atol=1e-8)
    audits=[audit_trial(source,directory/r['gpx'],protocol,candidate)
        for directory,r in [(full,direct),(reused,result)]]
    assert audits[0]['rwp_recomputed']==pytest.approx(audits[1]['rwp_recomputed'],rel=1e-9)
    import json
    receipts=[json.loads((directory/'magnetic-refinement-receipt.json').read_text())
        for directory in [full,reused]]
    assert receipts[0]['moment_initialization']==receipts[1]['moment_initialization']
    assert verify_prepared(built['prepared'],SimpleNamespace(run_dir=reused),protocol,candidate)
