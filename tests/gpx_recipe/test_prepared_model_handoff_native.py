"""CIF preparation -> portable factors -> actual Galaxy model worker wrapper.

All patterns are generated from a public artificial parent. This is an
integration/physics control, not a held-out material identification benchmark.
"""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

G=pytest.importorskip('GSASII.GSASIIscriptable')
from GSASII import GSASIIstrMain as MAIN
from test_preparation_worker_native import request as preparation_request
from gpx_recipe.execution import write_json
from gpx_recipe.preparation import hints
from gpx_recipe.preparation.operations import create_library,stage_entry,publish_entry
from gpx_recipe.preparation.worker import execute
from gpx_recipe.symmetry.bundle import pack,unpack
from gpx_recipe.symmetry.compiler import file_sha,package_signature
from gpx_recipe.symmetry.contracts import digest,iter_trials,trial_id,validate_protocol
from gpx_recipe.symmetry.native_audit import inspect_source
from gpx_recipe.symmetry.native_magnetic import construct_theory
from gpx_recipe.symmetry.worker import Artifacts


@pytest.mark.parametrize('kind,change',[('PNC','same'),('PNT','same'),('PNC','cell')])
def test_theory_library_reaches_real_model_jobs_and_current_native_checks(tmp_path,kind,change):
    payload,construction,candidate=preparation_request(tmp_path/'before-measurement',kind)
    settings=dict(candidate=candidate,k=construction['k'],sites=construction['sites'],maximum_moment=6.,
        refine_histogram_scale=True,refine_background=True)
    library=create_library(tmp_path/'library',name='Before measurement',source=payload['source'],
        banks={'selected bank':dict(instrument=payload['instrument'],grid=payload['grid'])},
        models=[dict(kind='magnetic',bank='selected bank',settings=settings)])
    entry=library.document['entries'][0]['id'];attempt=library.start(entry)
    preparation=stage_entry(library,entry,attempt,tmp_path/'offline-calculation')
    result=execute('magnetic',preparation)
    assert publish_entry(library,entry,attempt,preparation,result)['all_prepared']
    offline=Path(result['output'])
    # Reveal the actual observation only after the preparation is complete.
    parent=G.G2Project(gpxfile=str(offline/'native-reference/reference.gpx'))
    if change=='cell':
        cell=parent.phase(0).data['General']['Cell']
        cell[1]*=1.003;cell[2]*=.998
    blank=tmp_path/'actual-parent-empty.gpx';parent.save(str(blank))
    truth_root=tmp_path/'independent-truth';truth_root.mkdir()
    truth_protocol=copy.deepcopy(construction)
    truth_protocol['source']={'sha256':file_sha(blank)}
    truth=construct_theory(blank,Artifacts(truth_root),truth_protocol,candidate,
        dict(axis='x',magnitude=1.7,parent_direction=None,independent_mode_seed=31,site_signs={}))
    truth_project=G.G2Project(gpxfile=str(truth_root/truth['gpx']))
    for phase in truth_project.phases():
        if phase.data['General']['Type']=='magnetic':
            phase.data['Atoms'][0][7:10]=[1.7,2.1,1.2]
    truth_project.save()
    intensity=np.asarray(MAIN.DoNoFit(str(truth_root/truth['gpx']),truth_project.histograms()[0].name),float)
    histogram=parent.histograms()[0];arrays=histogram.data['data'][1]
    random=np.random.default_rng(99173)
    arrays[1]=intensity+random.normal(0.,.001,len(intensity))
    arrays[2]=np.full(len(intensity),1e6)
    histogram.data['data'][0]['Dummy']=False
    inputs=tmp_path/'model-inputs';inputs.mkdir()
    source=inputs/'source.gpx';parent.save(str(source))
    context=inspect_source(source);nuclear=context['phases'][0]
    catalog=dict(schema='radar-symmetry-catalog/v1',parent_setting_sha256=nuclear['setting_sha256'],
        k=construction['k'],candidates=[candidate])
    write_json(inputs/'catalog.json',catalog)
    protocol=validate_protocol(dict(kind='magnetic',source=dict(path='source.gpx',sha256=file_sha(source)),
        catalog=dict(path='catalog.json',sha256=file_sha(inputs/'catalog.json')),
        parent_phase=nuclear['name'],histogram=histogram.name,parent_setting_sha256=nuclear['setting_sha256'],
        magnetic_function_sha256=context['runtime']['magnetic_function_sha256'],implementation_sha256=package_signature(),
        k=construction['k'],sites=construction['sites'],candidates=[candidate],
        seeds=[dict(id='first',axis='x',magnitude=.7,independent_mode_seed=17),
               dict(id='second',axis='y',magnitude=1.1,independent_mode_seed=31)],
            policy=dict(cycles=20,timeout_seconds=180,maximum_trials=2,maximum_cell_multiplier=8,maximum_moment=6.,
                magnetic_d_min=0.,
            fit_band_percent=.25,maximum_mode=.5,adequate_rwp_percent=12.,refine_cell=False,
            refine_histogram_scale=True,refine_background=True,moment_model='symmetry_allowed',
            magnetic_optimizer='profiled_quadratic',optimizer_iterations=120,grouped_surrogate=True)))
    write_json(inputs/'protocol.json',protocol)
    document=hints.capture(library,protocol,inputs)
    assert len(document['entries'])==1
    names=['catalog.json','protocol.json']+hints.packet_paths(inputs,protocol)
    assert not any(name.endswith('response.npz') for name in names)
    pack(inputs,tmp_path/'model-assets.zip',names)
    ids=[trial_id(protocol,c['id'],s['id']) for c,s in iter_trials(protocol)]
    message=dict(schema='radar-model-chunk/v1',protocol=protocol,protocol_sha256=digest(protocol),
        trials=ids,workers=1,dispatch_key='a'*24,seconds=240,prepared_scattering=document)
    write_json(tmp_path/'model-request.json',message)
    script=Path(__file__).resolve().parents[2]/'scripts/ndip_model_campaign.py'
    with (tmp_path/'model-wrapper.log').open('w') as log:
        subprocess.run([sys.executable,str(script),'chunk',str(source),str(tmp_path/'model-assets.zip'),
            str(tmp_path/'model-request.json'),str(tmp_path/'model-job'),str(tmp_path/'model-result.zip')],
            check=True,timeout=300,stdout=log,stderr=subprocess.STDOUT)
    received=unpack(tmp_path/'model-result.zip',tmp_path/'received')
    response=json.loads((received/'response.json').read_text())
    assert response['prepared_scattering']['attached']==1
    assert {row['id'] for row in response['trials']}==set(ids) and not response['pending']
    assert all(row['status'] in {'valid','profile_equivalent'} for row in response['trials']),response['trials']
    owner=received/'trials'/ids[0]/'group-data'
    metadata=json.loads((owner/'kernel.json').read_text())
    proof=metadata['compilation']
    assert proof['prepared_scattering']['attached']==1
    compiler=proof['compiler_diagnostics']
    assert compiler['scattering_factors_reused'] == (change=='same')
    assert not compiler['scattering_qualification_reused']
    if change=='same':
        assert compiler['native_amplitude_calls']==0
        assert compiler['coefficient_structure_factor_calls']==19
    else:assert compiler['native_amplitude_calls']==6
    # Final GPXs are real native outputs, separate from the theory library.
    published=list(received.glob('trials/*/results/*.gpx'))
    assert published
    for path in published:
        final=G.G2Project(gpxfile=str(path))
        phases=[p for p in final.phases() if p.data['General']['Type']=='magnetic']
        assert phases
        moment=np.asarray(phases[0].data['Atoms'][0][7:10],float)
        assert np.isfinite(moment).all() and np.linalg.norm(moment)>1.
    # The independent Live packet uses the same library and native compiler.
    # This intentionally tiny sampling budget checks handoffs, not posterior
    # convergence, magnetic identification or an exposure recommendation.
    from gpx_recipe.live.packet import create_packet
    from gpx_recipe.live.galaxy_worker import run_packet
    preflight=tmp_path/'live-preflight.json'
    write_json(preflight,[dict(id=candidate['id'],compatible=True,dimensions=3)])
    live_request=dict(source_gpx=str(source),source_sha256=file_sha(source),
        catalog=str(inputs/'catalog.json'),catalog_sha256=file_sha(inputs/'catalog.json'),
        preflight=str(preflight),preflight_sha256=file_sha(preflight),k=protocol['k'],sites=protocol['sites'],
        source_revision='artificial:current',reference_revision='artificial:nominal',
        lower=0.,upper=6.,particles=64,workers=1,candidate_seconds=5.,preparation_timeout_seconds=180.)
    live_inputs=tmp_path/'live-inputs'
    live_document=hints.capture(library,hints.live_context(live_request,catalog),live_inputs)
    live_request['prepared_scattering']=live_document
    packet=tmp_path/'live-packet.zip';create_packet(live_request,packet,prepared_root=live_inputs)
    live=run_packet(packet,tmp_path/'live-native',tmp_path/'live-result.zip',tmp_path/'live-cache.zip')
    assert len(live['rows'])==1 and live['rows'][0]['status'] in {'conditional_posterior','sampling_unresolved'},live['rows']
    assert not live['rows'][0].get('sampling_incomplete'),live['rows']
    assert live['rows'][0].get('posterior')
    assert live['rows'][0]['gpx_status']=='conditional_checkpoint'
    checkpoint=tmp_path/'live-native'/live['rows'][0]['fitted_gpx']
    assert file_sha(checkpoint)==live['rows'][0]['fitted_gpx_sha256']
    saved=G.G2Project(gpxfile=str(checkpoint))
    assert any(p.data['General']['Type']=='magnetic' for p in saved.phases())
    assert live['rows'][0]['prepared_scattering']['attached']==1
    model_name=candidate['id'].replace(':','_')
    live_meta=json.loads((tmp_path/'live-native/models'/model_name/'kernel.json').read_text())
    live_compiler=live_meta['compiler']['native_reflection_diagnostics']
    assert live_compiler['scattering_factors_reused']==(change=='same')
    assert not live_compiler['scattering_qualification_reused']
    assert live_compiler['native_amplitude_calls']==(0 if change=='same' else 6)
    assert live_compiler['coefficient_structure_factor_calls']==19
    write_json(tmp_path/'handoff.json',dict(kind=kind,change=change,library_id=document['library_id'],
        snapshot_sha256=document['snapshot_sha256'],native_compiler=compiler,
        statuses=[row['status'] for row in response['trials']],published_gpx=len(published),
        live_compiler=live_compiler,live_status=live['rows'][0]['status']))
