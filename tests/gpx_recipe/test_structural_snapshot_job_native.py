"""Full Model wrapper transport of a public structural acquisition snapshot."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

G=pytest.importorskip('GSASII.GSASIIscriptable')
from GSASII import GSASIIstrMain as MAIN
from test_native_profile_capture import native_source
from test_structural_acquisition_native import ascii_pattern
from gpx_recipe.execution import write_json
from gpx_recipe.live.snapshot import parse_live_ascii,uniform_density_to_tof
from gpx_recipe.structural_live.native import sha
from gpx_recipe.symmetry.bundle import pack,unpack
from gpx_recipe.symmetry.compiler import compile_native_bundle,compile_shared_job,prepare_shared_inputs,file_sha,package_signature
from gpx_recipe.symmetry.contracts import digest,validate_candidate,validate_protocol,trial_id
from gpx_recipe.symmetry.native_audit import inspect_source


@pytest.mark.parametrize('shared',[False,True])
def test_model_wrapper_preserves_reviewed_reference_and_exports_native_structural_snapshot(tmp_path,shared):
    source,public=native_source(tmp_path/'public','PNT');source_sha=sha(source)
    parent=G.G2Project(gpxfile=str(source));warm=tmp_path/'warm.gpx'
    # A separately reviewed fractional structure/ADP reference must survive
    # transfer. It is not interchangeable with silently using the source GPX.
    atom=parent.phase(0).data['Atoms'][0];cia=parent.phase(0).data['General']['AtomPtrs'][3]
    atom[cia+1]=float(atom[cia+1])*1.1+.0001
    parent.save(str(warm));warm_sha=sha(warm)
    exporter=G.G2Project(gpxfile=str(source),newgpx=str(tmp_path/'exporter.gpx'))
    h=exporter.histogram(0);x=np.linspace(float(h.data['data'][1][0][0]),float(h.data['data'][1][0][-1]),701)
    h.data['data'][1]=[x,np.ones_like(x),np.ones_like(x)*4,np.zeros_like(x),np.zeros_like(x),np.zeros_like(x)]
    exporter.save();native=np.asarray(MAIN.DoNoFit(str(exporter.filename),h.name),float)
    errors=np.full(len(x),max(float(np.max(abs(native)))*1e-3,.001));factor=6200.
    anchor=ascii_pattern(x,native,errors,factor)
    observed=native+np.random.default_rng(902).normal(0.,errors,len(x))
    raw=ascii_pattern(x,observed,errors,factor)
    inputs=tmp_path/'input';inputs.mkdir()
    import shutil
    shutil.copyfile(source,inputs/'source.gpx');shutil.copyfile(warm,inputs/'warm.gpx')
    child=Path(__file__).with_name('data')/'synthetic-ersi-cmcm-isodistort.cif'
    shutil.copyfile(child,inputs/'child.cif')
    candidate=validate_candidate(dict(id='struct:public',child=dict(path='child.cif',sha256=file_sha(child)),
        modes=public.fast.meta['inventory']['native_mode_names'],
        transformation_matrix=[[-1.,0,0],[0,-1.,0],[0,0,1.]],origin_shift=[0,0,0]))
    inspected=inspect_source(source);phase=inspected['phases'][0]
    write_json(inputs/'catalog.json',dict(schema='radar-symmetry-catalog/v1',parent_setting_sha256=phase['setting_sha256'],
        k=['0','0','0'],candidates=[candidate]))
    protocol=validate_protocol(dict(kind='structural',source=dict(path='source.gpx',sha256=source_sha),
        structural_reference=dict(path='warm.gpx',sha256=warm_sha),catalog=dict(path='catalog.json',sha256=file_sha(inputs/'catalog.json')),
        parent_phase=phase['name'],histogram=h.name,parent_setting_sha256=phase['setting_sha256'],
        magnetic_function_sha256=inspected['runtime']['magnetic_function_sha256'],implementation_sha256=package_signature(),
        k=['0','0','0'],sites=[],candidates=[candidate],seeds=[dict(id='prepared')],
        policy=dict(cycles=20,timeout_seconds=180,maximum_trials=1,maximum_cell_multiplier=8,maximum_moment=6.,
            fit_band_percent=.25,maximum_mode=.5,adequate_rwp_percent=12.,refine_cell=True,refine_histogram_scale=True,
            refine_background=True,structural_optimizer='prepared_nonlinear',structural_starts=2,structural_seconds=4.,
            structural_mode_radius=.05,structural_cell_radius=.005)))
    packet=tmp_path/'packet';packet.mkdir()
    if shared:
        captured=prepare_shared_inputs(protocol,inputs,packet/'models')
        model=packet/'models/trials'/trial_id(protocol,candidate['id'],'prepared')
        compile_shared_job(protocol,model,candidate_id=candidate['id'],seed_id='prepared',approved_hash=digest(protocol))
    else:
        model=packet/'model';captured=model
        compile_native_bundle(protocol,inputs,model,candidate_id=candidate['id'],seed_id='prepared',approved_hash=digest(protocol))
    from gpx_recipe.preparation import structural_hints
    preparation=dict(schema=structural_hints.SCHEMA,protocol_sha256=digest(protocol),library_id='c'*24,
        library_name='Empty public handoff control',entries=[],eligible_entries=0,omitted_entries=0,observations_included=False)
    preparation['snapshot_sha256']=digest(preparation)
    hint=captured/structural_hints.MANIFEST;hint.parent.mkdir();write_json(hint,preparation)
    (packet/'anchor.dat').write_bytes(anchor);(packet/'snapshot.dat').write_bytes(raw)
    identity=dict(facility='SNS',instrument='PG3',ipts=1,run=2,epoch='public-native-roundtrip')
    mapping=dict(mapping_kind='user_reviewed',source_gpx_sha256=source_sha,snapshot_sha256=hashlib.sha256(anchor).hexdigest(),
        physical_bank=1,identity=identity,export_geometry=dict(kind='linear_tof_from_d',microseconds_per_angstrom=factor),
        normalization='Public native reduced density control',producer_config_sha256='a'*64)
    payload=dict(source=str((captured/'source.gpx').relative_to(packet)),directory=str(model.relative_to(packet)),anchor_snapshot='anchor.dat',snapshot='snapshot.dat',
        receipt=dict(identity=identity,content_sha256=hashlib.sha256(raw).hexdigest(),metadata=dict(physical_bank=1)),
        mapping=mapping,source_scope=dict(run='2',sample_id='public',sample='Public ErSi',wavelength='1.5'),revision=2,
        source_revision='PG3_2:'+mapping['snapshot_sha256'],reference_revision='PG3_ref:'+source_sha,
        refine_transfer=False,destination='snapshot-output',prepared_scattering=preparation)
    hashes={str(p.relative_to(packet)):sha(p) for p in packet.rglob('*') if p.is_file() and p.suffix!='.py'}
    request=dict(schema='radar-model-native/v1',operation='structural-snapshot',payload=payload,
        implementation_sha256=package_signature(),expected_source_sha256=source_sha,input_hashes=hashes,
        timeout=180,dispatch_key='b'*24)
    # Omit the source bytes in the archive; the normal Galaxy HDA hydration
    # path must reconstruct exactly the same captured model input.
    pack(packet,tmp_path/'assets.zip',[name for name in hashes if name!=payload['source']])
    write_json(tmp_path/'request.json',request)
    script=Path(__file__).resolve().parents[2]/'scripts/ndip_model_campaign.py'
    with (tmp_path/'wrapper.log').open('w') as log:
        completed=subprocess.run([sys.executable,str(script),'native',str(source),str(tmp_path/'assets.zip'),str(tmp_path/'request.json'),
            str(tmp_path/'worker'),str(tmp_path/'result.zip'),'--preparation-certificate',str(tmp_path/'certificate.json')],
            check=False,timeout=210,stdout=log,stderr=subprocess.STDOUT)
    assert completed.returncode==0,(tmp_path/'wrapper.log').read_text()[-16000:]
    received=unpack(tmp_path/'result.zip',tmp_path/'received')
    response=json.loads((received/'response.json').read_text());result=response['result'];fitted=result['result']
    assert result['schema']=='radar-structural-snapshot-result/v1' and result['revision']==2
    stages=result['workflow_timings']
    assert stages['scope']=='acquisition' and stages['preparation_seconds']>0
    assert stages['fit_service_seconds']>=fitted['wall_seconds']
    assert stages['combined_service_seconds']==stages['preparation_seconds']+stages['fit_service_seconds']
    assert stages['queue_startup_transfer_included'] is False
    assert result['protocol_sha256']==digest(protocol) and result['source_sha256']==source_sha
    assert result['reference_sha256']==warm_sha and result['snapshot_sha256']==hashlib.sha256(raw).hexdigest()
    assert fitted['all_physical_modes_retained'] and fitted['physical_mode_count']==2 and fitted['cell_parameter_count']==3
    assert fitted['state']=='verified_point_fit' and not fitted['optimization_qualified']
    assert fitted['verification']['relative_error']<1e-8 and fitted['verification']['max_sigma_error']<.01
    assert not result['forecast_enabled'] and not result['statistical_policy_qualified']
    metadata=json.loads((received/'native/snapshot-output/acquisition/model/prepared.json').read_text())
    assert metadata['reference_gpx_sha256']==warm_sha
    assert not list((tmp_path/'worker/verified_projects').glob('*.gpx'))
    published=list((tmp_path/'worker/reviewable_projects').glob('*.gpx'))
    assert len(published)==2
    assert {sha(p) for p in published}=={fitted[k]['sha256'] for k in ('frozen','continuation')}
    measurement=uniform_density_to_tof(parse_live_ascii(raw),export_difc=factor)
    for path in published:
        saved=G.G2Project(gpxfile=str(path));arrays=saved.histogram(0).data['data'][1]
        np.testing.assert_array_equal(arrays[0],measurement['x'])
        np.testing.assert_array_equal(arrays[1],measurement['y'])
        np.testing.assert_allclose(arrays[2],measurement['weights'],rtol=1e-12,atol=0)
    display=json.loads((received/'native/snapshot-output/view.json').read_text())
    assert display['histogram']==h.name and display['ticks'] and len(display['x'])==701
    np.testing.assert_array_equal(display['observed'],measurement['y'])
    assert sha(source)==source_sha and sha(warm)==warm_sha
    assert json.loads((tmp_path/'certificate.json').read_text())['entries']==[]
    assert not any(p.suffix=='.dat' for p in received.rglob('*'))
    write_json(tmp_path/'acceptance.json',dict(result=result['state'],rwp=fitted['metrics']['rwp_native'],
        timings=fitted['timings'],worker_timing=response['worker_timing'],reference_preserved=True,
        complete_modes=2,cell_coordinates=3,
        scope='Artificial public POWGEN-format Model wrapper round trip; no NDIP or material-identification claim'))
