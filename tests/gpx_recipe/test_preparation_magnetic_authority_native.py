"""Cold magnetic factor reuse after real native preparation and producer removal.

The creating-job metadata is simulated. The installed native wrapper, its
certificate, portable library, fresh consumer processes and fitted native
residual checks are real. No facility measurement or global-search claim.
"""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest

pytest.importorskip('GSASII.GSASIIscriptable')

from gpx_recipe.execution import write_json
from gpx_recipe.preparation import authority,certificates
from gpx_recipe.preparation.library import import_library
from gpx_recipe.preparation.operations import create_library,stage_entry,publish_entry
from gpx_recipe.symmetry.bundle import pack,unpack
from gpx_recipe.symmetry.compiler import file_sha,package_signature
from test_preparation_worker_native import request


CONSUMER='''import json,sys
from pathlib import Path
import numpy as np
from GSASII import GSASIIscriptable as G
from gpx_recipe.preparation.authority import server_scope
from gpx_recipe.preparation.reference import create_reference
from gpx_recipe.preparation.consume import restore_magnetic_entry
from gpx_recipe.preparation.library import PreparationLibrary
from gpx_recipe.live.reflection_factors import digest_array
from gpx_recipe.native_forward import forward_profile
from gpx_recipe.symmetry.native_magnetic import construct_theory
from gpx_recipe.symmetry.worker import Artifacts
from gpx_recipe.symmetry.compiler import file_sha
from gsas_magnetic_preflight import setting_fingerprint
sys.path.insert(0,sys.argv[5])
from test_prepared_k_series_native import fit_portfolio
library=PreparationLibrary(sys.argv[1]);work=Path(sys.argv[2]);work.mkdir()
origin=sys.argv[3];variant=sys.argv[4]
row=library.document['entries'][0];model=row['model'];settings=model['settings']
assets={Path(a['path']).name:library.root/a['path'] for a in row['files']}
response,cache,receipt=restore_magnetic_entry(assets['entry.json'].parent)
assert response._native_seal is None and receipt['native_qualification_required']
manifest=json.loads(assets['entry.json'].read_text())
if variant=='corrupt':
    with np.load(assets['scattering.npz'],allow_pickle=False) as data:
        factors=data['factors'].copy()*1.01;metadata=json.loads(str(data['manifest']))
    metadata['sha256']=digest_array(factors)
    injected=work/'changed-factors.npz'
    np.savez_compressed(injected,factors=factors,manifest=np.array(json.dumps(metadata)))
    cache.attach_source(injected,identity=manifest['detail']['scattering']['identity'])
elif variant=='invalid':
    injected=work/'invalid-factors.npz';injected.write_bytes(b'not a NumPy archive')
    cache.attach_source(injected,identity=manifest['detail']['scattering']['identity'])
reference=create_reference(library.root/library.document['specification']['source'],
    library.root/model['instrument'],work/'reference',**model['grid'])
project=G.G2Project(gpxfile=str(reference['path']));phase=project.phase(0)
if variant=='width':
    instrument=project.histograms()[0].data['Instrument Parameters'][0]
    for key,value in [('X',.1),('Y',.08),('Z',.03)]:instrument[key][0]=instrument[key][1]=value
elif variant=='cell':
    phase.data['General']['Cell'][1]*=1.003;phase.data['General']['Cell'][2]*=.998
elif variant=='adp':
    cia=phase.data['General']['AtomPtrs'][3]
    for atom in phase.data['Atoms']:atom[cia+1]+=.01
project.save()
protocol=dict(source={'sha256':file_sha(reference['path'])},parent_phase=phase.name,
    histogram=reference['histogram'],parent_setting_sha256=setting_fingerprint(phase.data['General']['SGData']),
    k=settings['k'],sites=settings['sites'],policy=dict(maximum_cell_multiplier=8,magnetic_d_min=0.,cycles=1,
        maximum_moment=6.,site_policy='all',refine_histogram_scale=False,refine_background=False,moment_model='symmetry_allowed'))
output=work/'model';output.mkdir()
built=construct_theory(reference['path'],Artifacts(output),protocol,settings['candidate'],
    dict(axis='x',magnitude=1.,parent_direction=None,independent_mode_seed=17,site_signs={}))
path=output/built['gpx'];project=G.G2Project(gpxfile=str(path))
for phase in project.phases():
    if phase.data['General']['Type']=='magnetic':
        cx=phase.data['General']['AtomPtrs'][0]
        phase.data['Atoms'][0][cx+4:cx+7]=(1.4*np.array([.8,.5,.3])).tolist()
project.save();observed,_=forward_profile(path,project.histograms()[0].name)
project=G.G2Project(gpxfile=str(path));hist=project.histograms()[0]
arrays=hist.data['data'][1];arrays[1]=observed.copy();arrays[2]=np.full(len(observed),250000.)
hist.data['data'][0]['Dummy']=False;project.save()
with server_scope(None if origin=='none' else origin,work):
    fitted=fit_portfolio(path,None if variant=='fresh' else cache)
proof=fitted['compilation']['compiler_diagnostics']
dimension=proof['dimensions']
expected_reuse=variant in {'same','width'}
assert proof['scattering_factors_reused']==expected_reuse,proof
assert proof['scattering_authenticated_origin_reused']==(expected_reuse and origin!='none'),proof
assert not proof['scattering_qualification_reused']
if expected_reuse:
    assert proof['native_amplitude_calls']==0
    assert proof['coefficient_structure_factor_calls']==(3 if origin!='none' else 2*dimension**2+1),proof
elif variant!='fresh':
    assert proof['native_amplitude_calls']==2*dimension,proof
    assert proof['coefficient_structure_factor_calls']>=2*dimension**2+1,proof
if variant=='corrupt':assert 'failed native checks' in proof['scattering_cache_note']
if variant=='invalid':assert 'archive is invalid' in proof['scattering_cache_note']
assert len(fitted['rows'])==2*dimension and fitted['best']['cost']<1e-5
assert proof['native_profile_calls']>=1
np.savez_compressed(work/'checked.npz',residuals=np.array([r['residual'] for r in fitted['rows']]),observed=observed)
results=dict(variant=variant,seconds=fitted['seconds'],stages_seconds=fitted['stages_seconds'],
    compiler=proof,costs=[r['cost'] for r in fitted['rows']],starts=len(fitted['rows']),
    maximum_native_error=max(r['forward_error'] for r in fitted['rows']),
    native_calls=fitted['native_calls'],native_derivative_calls=fitted['native_derivative_calls'])
(work/'result.json').write_text(json.dumps(results))
'''


@pytest.mark.parametrize('kind,site_count',[('PNC',1),('PNT',1),('PNT',3)])
def test_cold_authenticated_magnetic_reuse_keeps_current_physics_and_all_fit_checks(tmp_path,kind,site_count):
    payload,protocol,candidate=request(tmp_path/'public-inputs',kind)
    if site_count==3:
        cif=Path(payload['source'])
        cif.write_text(cif.read_text()+'Fe2 Fe .31 .43 .17 .006 1\nFe3 Fe .61 .72 .48 .004 1\n')
        protocol['sites']+=[dict(label=label,form_factor='Fe+3',lande_g=2.) for label in ('Fe2','Fe3')]
    settings=dict(candidate=candidate,k=['0','0','0'],sites=protocol['sites'],maximum_moment=6.,
        refine_histogram_scale=False,refine_background=False)
    library=create_library(tmp_path/'library',name='Public cold magnetic preparation',source=payload['source'],
        banks={kind:dict(instrument=payload['instrument'],grid=payload['grid'])},
        models=[dict(kind='magnetic',bank=kind,settings=settings)])
    row=library.document['entries'][0];attempt=library.start(row['id'])
    captured=tmp_path/'captured';staged=stage_entry(library,row['id'],attempt,captured)
    native_payload=copy.deepcopy(staged);names=[]
    for key in ('source','instrument'):
        native_payload[key]=str(Path(native_payload[key]).relative_to(captured));names.append(native_payload[key])
    native_payload['destination']='compiled';source=captured/native_payload['source']
    pack(captured,tmp_path/'inputs.zip',names)
    write_json(tmp_path/'request.json',dict(schema='radar-model-native/v1',operation='preparation-magnetic',
        payload=native_payload,implementation_sha256=package_signature(),expected_source_sha256=file_sha(source),
        dispatch_key='e'*24,timeout=180,input_hashes={name:file_sha(captured/name) for name in names}))
    wrapper=Path(__file__).resolve().parents[2]/'scripts/ndip_model_campaign.py'
    certificate=tmp_path/'certificate.json';began=time.monotonic()
    with (tmp_path/'producer.log').open('w') as stream:
        subprocess.run([sys.executable,str(wrapper),'native',str(source),str(tmp_path/'inputs.zip'),
            str(tmp_path/'request.json'),str(tmp_path/'job'),str(tmp_path/'result.zip'),
            '--preparation-certificate',str(certificate)],check=True,timeout=210,stdout=stream,stderr=subprocess.STDOUT)
    preparation_seconds=time.monotonic()-began
    document=certificates.read(certificate)
    assert len(document['entries'])==1 and document['entries'][0]['kind']=='magnetic'
    assert set(document['entries'][0]['artifacts'])=={'response.npz','scattering.npz'}
    assert document['result_archive_sha256']==file_sha(tmp_path/'result.zip')
    received=unpack(tmp_path/'result.zip',tmp_path/'received')
    result=json.loads((received/'response.json').read_text())['result']
    shutil.copytree(received/'native/compiled',staged['destination']);result['output']=staged['destination']
    publish_entry(library,row['id'],attempt,staged,result)
    archive=tmp_path/'portable.zip';library.export(archive)
    restored=import_library(archive,tmp_path/'cold-library');frozen=file_sha(archive)
    for path in (tmp_path/'job',library.root,captured,received,tmp_path/'public-inputs'):shutil.rmtree(path)
    assert not list(restored.root.rglob('*.gpx'))
    envelope=tmp_path/'server.json'
    write_json(envelope,dict(schema=authority.SCHEMA,records=[dict(certificate_path=str(certificate),
        tool_id='neutrons_radar_pd_model_campaign_prototype',tool_version='0.2.17',job_state='ok',dataset_state='ok',
        dataset_deleted=False,dataset_purged=False,job_stdout=certificates.MARKER+file_sha(certificate)+'\n')]))
    script=tmp_path/'consumer.py';script.write_text(CONSUMER);results={}
    cases=[('certified','same',str(envelope)),('uncertified','same','none'),('fresh','fresh','none'),
        ('width','width',str(envelope)),('cell','cell',str(envelope)),
        ('adp','adp',str(envelope)),('corrupted','corrupt',str(envelope)),('invalid','invalid',str(envelope))]
    # Keep the existing full invalidation controls, and add an independent
    # higher-dimensional cold timing/control case without dropping any start.
    if site_count==3:cases=cases[:3]
    for label,variant,origin in cases:
        output=tmp_path/label
        with (tmp_path/(label+'.log')).open('w') as stream:
            subprocess.run([sys.executable,str(script),str(restored.root),str(output),origin,variant,
                str(Path(__file__).parent)],check=True,timeout=180,stdout=stream,stderr=subprocess.STDOUT)
        results[label]=json.loads((output/'result.json').read_text())
        assert file_sha(archive)==frozen
        write_json(tmp_path/'progress.json',dict(kind=kind,completed=list(results),expected=len(cases)))
    for label in ('uncertified','fresh')+(('corrupted','invalid') if site_count==1 else ()):
        np.testing.assert_allclose(results[label]['costs'],results['certified']['costs'],rtol=1e-6,atol=1e-6)
        with np.load(tmp_path/label/'checked.npz') as a,np.load(tmp_path/'certified/checked.npz') as b:
            np.testing.assert_allclose(a['residuals'],b['residuals'],rtol=1e-6,atol=2e-5)
    write_json(tmp_path/'cold-reuse-measurement.json',dict(schema='radar-public-cold-magnetic-reuse-control/v1',
        preparation_seconds=preparation_seconds,results=results,complete_starts_per_case=6*site_count,
        native_checked_fitted_profiles=len(cases)*6*site_count,archive_sha256=frozen,
        scope='Public artificial '+kind+'; actual installed native wrapper and fresh consumer processes; '+
            'simulated Galaxy creating-job metadata; native profile and every fitted-state check retained; '+
            'no real material identification, global optimization or NDIP latency claim'))
