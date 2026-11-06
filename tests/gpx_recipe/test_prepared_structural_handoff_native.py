"""CIF preparation -> portable structural bank -> actual scientific worker.

Public artificial PNC/PNT controls. Observation data are revealed only after
preparation. This tests handoff fidelity, not material identification or speed.
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

G=pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_profile_capture import native_source
from gpx_recipe.execution import write_json
from gpx_recipe.preparation import hints,structural_hints
from gpx_recipe.preparation.library import import_library
from gpx_recipe.preparation.operations import create_library,stage_entry,publish_entry
from gpx_recipe.preparation.worker import execute
from gpx_recipe.structural_live.native import prepare,NativeCandidate,sha
from gpx_recipe.structural_live.reduced import from_arrays
from gpx_recipe.structural_live.service import SearchSettings
from gpx_recipe.symmetry.bundle import pack,unpack
from gpx_recipe.symmetry.compiler import file_sha,package_signature
from gpx_recipe.symmetry.contracts import digest,iter_trials,trial_id,validate_candidate,validate_protocol
from gpx_recipe.symmetry.native_audit import inspect_source


@pytest.mark.parametrize('kind,change',[('PNC','same'),('PNT','same'),('PNC','cell'),('PNC','damaged')])
def test_structural_library_reaches_real_model_and_owned_snapshot_consumers(tmp_path,kind,change):
    # Only a public parent CIF, authoritative child and instrument enter the
    # preparation. No fitted parameters or observations enter its library.
    public_gpx,public=native_source(tmp_path/'public-inputs',kind)
    parent=G.G2Project(gpxfile=str(public_gpx))
    cif=tmp_path/'parent.cif';parent.phase(0).export_CIF(str(cif))
    child=Path(__file__).with_name('data')/'synthetic-ersi-cmcm-isodistort.cif'
    candidate=validate_candidate(dict(id='struct:public-ersi',
        transformation_matrix=[[-1.,0,0],[0,-1.,0],[0,0,1.]],origin_shift=[0,0,0],
        child=dict(path='child.cif',sha256=file_sha(child)),
        modes=public.fast.meta['inventory']['native_mode_names']))
    settings=dict(candidate=candidate,mode_radius=.05,cell_radius=.005)
    grid=dict(kind=kind,lower=12. if kind=='PNC' else 3000.,
        upper=125. if kind=='PNC' else 40000.,bins=701,bank=1)
    library=create_library(tmp_path/'library',name='Public before-measurement structural library',source=cif,
        banks={'selected bank':dict(instrument=tmp_path/'public-inputs/profile.instprm',grid=grid)},
        models=[dict(kind='structural',bank='selected bank',settings=settings,child=child)])
    entry=library.document['entries'][0]['id'];attempt=library.start(entry)
    request=stage_entry(library,entry,attempt,tmp_path/'offline-calculation')
    built=execute('structural',request)
    assert publish_entry(library,entry,attempt,request,built)['all_prepared']
    offline=Path(built['output'])
    library.export(tmp_path/'portable.zip')
    restored=import_library(tmp_path/'portable.zip',tmp_path/'restored-library')
    assert all(not row['path'].endswith('.gpx') for row in restored.row(entry)['files'])
    manifest=json.loads((offline/'entry.json').read_text())
    assert not manifest['observations_included'] and not manifest['scientific_result']

    # A later observation may have a different actual parent lattice. The
    # intensity comes from an independent native saved-project calculation.
    actual=G.G2Project(gpxfile=str(offline/'native-reference/reference.gpx'))
    if change=='cell':
        actual.phase(0).data['General']['Cell'][1]*=1.0003
    blank=tmp_path/'later-parent.gpx';actual.save(str(blank))
    truth_prepared=tmp_path/'independent-truth'
    prepare(blank,child,candidate['transformation_matrix'],candidate['origin_shift'],truth_prepared)
    truth=NativeCandidate(truth_prepared,tmp_path/'independent-native-calculation')
    intensity=truth.evaluate(np.array([.005,-.004]))
    histogram=actual.histograms()[0];arrays=histogram.data['data'][1]
    sigma=max(float(np.max(abs(intensity)))*1e-4,1e-4)
    arrays[1]=intensity+np.random.default_rng(39403).normal(0.,sigma,len(intensity))
    arrays[2]=np.full(len(intensity),1/sigma**2)
    histogram.data['data'][0]['Dummy']=False
    inputs=tmp_path/'model-inputs';inputs.mkdir()
    source=inputs/'source.gpx';actual.save(str(source))
    shutil.copyfile(child,inputs/'child.cif')
    context=inspect_source(source);nuclear=context['phases'][0]
    catalog=dict(schema='radar-symmetry-catalog/v1',parent_setting_sha256=nuclear['setting_sha256'],
        k=['0','0','0'],candidates=[candidate])
    write_json(inputs/'catalog.json',catalog)
    protocol=validate_protocol(dict(kind='structural',source=dict(path='source.gpx',sha256=file_sha(source)),
        catalog=dict(path='catalog.json',sha256=file_sha(inputs/'catalog.json')),
        parent_phase=nuclear['name'],histogram=histogram.name,parent_setting_sha256=nuclear['setting_sha256'],
        magnetic_function_sha256=context['runtime']['magnetic_function_sha256'],implementation_sha256=package_signature(),
        k=['0','0','0'],sites=[],candidates=[candidate],seeds=[dict(id='prepared')],
        policy=dict(cycles=20,timeout_seconds=180,maximum_trials=1,maximum_cell_multiplier=8,maximum_moment=6.,
            fit_band_percent=.25,maximum_mode=.5,adequate_rwp_percent=12.,refine_cell=True,
            refine_histogram_scale=True,refine_background=True,
            structural_optimizer='prepared_nonlinear',structural_starts=2,structural_seconds=8.,
            structural_mode_radius=.05,structural_cell_radius=.005,structural_recovery_seconds=0.)))
    write_json(inputs/'protocol.json',protocol)
    document=hints.capture(restored,protocol,inputs)
    assert document['schema']=='radar-prepared-structural-input/v1'
    assert len(document['entries'])==1 and not document['observations_included']
    if change=='damaged':
        row=document['entries'][0]
        bad=inputs/'invalid.npz';bad.write_bytes(b'invalid optional archive with matching declared checksum')
        row['sha256']=file_sha(bad);row['path']='_prepared_structural/'+row['sha256']+'.npz'
        bad.rename(inputs/row['path'])
        receipt=json.loads((inputs/row['receipt_path']).read_text());receipt['sha256']=row['sha256']
        replacement=inputs/'replacement.json';write_json(replacement,receipt)
        row['receipt_sha256']=file_sha(replacement)
        row['receipt_path']='_prepared_structural/'+row['receipt_sha256']+'.json'
        replacement.rename(inputs/row['receipt_path'])
        document['snapshot_sha256']=digest({k:v for k,v in document.items() if k!='snapshot_sha256'})
        write_json(inputs/structural_hints.MANIFEST,document)
    names=['catalog.json','protocol.json','child.cif']+hints.packet_paths(inputs,protocol)
    assert not any(name.endswith('.gpx') for name in names)
    pack(inputs,tmp_path/'model-assets.zip',names)
    ids=[trial_id(protocol,c['id'],s['id']) for c,s in iter_trials(protocol)]
    message=dict(schema='radar-model-chunk/v1',protocol=protocol,protocol_sha256=digest(protocol),
        trials=ids,workers=1,dispatch_key='b'*24,seconds=240,prepared_scattering=document)
    write_json(tmp_path/'model-request.json',message)
    script=Path(__file__).resolve().parents[2]/'scripts/ndip_model_campaign.py'
    with (tmp_path/'model-wrapper.log').open('w') as log:
        subprocess.run([sys.executable,str(script),'chunk',str(source),str(tmp_path/'model-assets.zip'),
            str(tmp_path/'model-request.json'),str(tmp_path/'model-job'),str(tmp_path/'model-result.zip')],
            check=True,timeout=300,stdout=log,stderr=subprocess.STDOUT)
    received=unpack(tmp_path/'model-result.zip',tmp_path/'received')
    response=json.loads((received/'response.json').read_text())
    assert response['prepared_scattering']['attached']==(0 if change=='damaged' else 1)
    assert {row['id'] for row in response['trials']}==set(ids) and not response['pending']
    assert response['trials'][0]['status']=='optimization_unresolved',response['trials']
    results=received/'trials'/ids[0]/'results'
    fitted=json.loads((results/'structural-fit.json').read_text())
    assert fitted['state']=='verified_point_fit' and fitted['all_physical_modes_retained']
    assert fitted['physical_mode_count']==2 and fitted['cell_parameter_count']==3
    assert fitted['preparation_attachment']['attached']==(0 if change=='damaged' else 1)
    qualification=fitted['portable_preparation']
    if change=='same':
        assert qualification['status']=='native_requalified',qualification
        assert qualification['supplied_curves_match_current']
    elif change=='cell':
        assert qualification['status']=='rebuilt_current_context',qualification
        assert qualification['discarded_entries']
    else:
        assert qualification['status']=='not_selected',qualification
        assert response['prepared_scattering']['discarded_entries']==[document['entries'][0]['entry']]
    assert not qualification['curve_bytes_reused']
    assert not qualification['scientific_candidates_removed']
    assert fitted['verification']['relative_error']<1e-8
    assert fitted['verification']['max_sigma_error']<.01
    assert fitted['result']['native_response_cache']['captured_native_calls']>0
    assert fitted['result']['native_response_cache']['derivative_calls']>0
    saved=G.G2Project(gpxfile=str(results/'child-refinement.gpx'))
    np.testing.assert_array_equal(saved.histograms()[0].data['data'][1][1],arrays[1])
    assert not fitted['optimization_qualified'] and not fitted['model_exclusion_authorized']
    # This backend session uses the same fitting service. It is not a claim
    # that a structural WebMon GUI/polling path has already been connected.
    if kind=='PNC' and change=='same':
        from dataclasses import replace
        from gpx_recipe.structural_live.session import StructuralSession
        prepared=results/'structural/prepared'
        session=StructuralSession(prepared,tmp_path/'snapshot-session',
            settings=SearchSettings(mode_radius=.05,cell_radius=.005,seconds=4.,starts=2))
        observation=from_arrays(arrays,histogram.data['Limits'],acquisition='public streaming control',
            protocol_sha256=sha(prepared/'prepared.json'),revision=0,exposure=1.)
        first=session.update(observation)
        assert first['state']=='verified_point_fit' and first['starting_curves_reused']
        duplicate=session.update(observation)
        assert duplicate['state']=='duplicate' and not duplicate['new_fit']
        later=replace(observation,revision=1,exposure=2.,variance=observation.variance/2)
        second=session.update(later)
        assert second['observation_sha256']==later.observation_sha256
        assert second['starting_curves_reused'] and second['result']['starts']>=1
        restored_session=StructuralSession(prepared,tmp_path/'snapshot-session',settings=session.settings)
        assert restored_session.update(later)['state']=='duplicate'
    write_json(tmp_path/'handoff.json',dict(kind=kind,change=change,
        portable_preparation=qualification,timings=fitted['timings'],status=response['trials'][0]['status'],
        scope='Public artificial native handoff control; no real material, convergence or performance claim'))
