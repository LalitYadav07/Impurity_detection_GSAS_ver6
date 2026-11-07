"""Real producer wrapper, portable library and independent native consumer.

Only the Galaxy creating-job metadata is simulated. Public artificial CIF,
native compilation, files, processes and current-context probes are real.
"""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

G=pytest.importorskip('GSASII.GSASIIscriptable')
from gpx_recipe.execution import write_json
from gpx_recipe.preparation import authority,certificates
from gpx_recipe.preparation.library import import_library
from gpx_recipe.preparation.operations import create_library,stage_entry,publish_entry
from gpx_recipe.symmetry.bundle import pack,unpack
from gpx_recipe.symmetry.compiler import file_sha,package_signature
from test_native_profile_capture import native_source


CONSUMER='''import json,sys,time
from pathlib import Path
import numpy as np
from gpx_recipe.preparation.authority import server_scope
from gpx_recipe.preparation.library import PreparationLibrary
from gpx_recipe.preparation import hints,tof_profile_hints
from gpx_recipe.preparation.tof_profile_arrays import read,admit
from gpx_recipe.structural_live.optimization.tof_profile_table import CompiledTOFTable
from gpx_recipe.structural_live.optimization.tof_profile_proposals import TOFProfileProposals,for_source
from gpx_recipe.structural_live.optimization import tof_profile_table
from gpx_recipe.structural_live.optimization.native_cell_proposals import ProposalUnavailable
from gpx_recipe.symmetry.bundle import pack,unpack
from test_native_profile_capture import native_source
library=PreparationLibrary(sys.argv[1]);work=Path(sys.argv[2]);work.mkdir()
protocol={'kind':'structural','candidates':[]}
captured=work/'captured';document=hints.capture(library,protocol,captured)
assert document['entries']==[] and len(document['profile_tables']['entries'])==1
assert len(document['evidence'])==1
pack(captured,work/'packet.zip',hints.packet_paths(captured,protocol))
received=unpack(work/'packet.zip',work/'received')
hints.restore(document,protocol,received,work/'inputs')
choices,attachment=tof_profile_hints.candidates(work/'inputs',protocol)
assert len(choices)==1
source_gpx,source=native_source(work/'actual','PNT')
from GSASII import GSASIIscriptable as G
original_x=G.G2Project(gpxfile=str(source_gpx)).histograms()[0].data['data'][1][0].copy()
assert np.array_equal(source.fast.slow.hist.data['data'][1][0],original_x)
row=choices[0];data,sha,receipt_sha=read(row['files']['path'],row['files']['receipt_path'])
mode=sys.argv[4]
if mode=='shape':source.fast.native.parms[f':{source.fast.slow.hist.id}:X']+=.1
if mode=='runtime':
    import gpx_recipe.preparation.kernel_runtime as runtime
    original=runtime.powder_fingerprint
    runtime.powder_fingerprint=lambda powder:(original(powder)[0],'f'*64)
envelope=None if mode=='unverified' else sys.argv[3]
tof_profile_table._shared=None
began=time.monotonic()
with server_scope(envelope,work):
    report=tof_profile_hints.bind(source,choices)
assert report['native_likelihood_authority'] is False
if mode!='certified':
    assert report['status']=='rebuild_current_context',report
    assert tof_profile_table._shared is None
    proposal=TOFProfileProposals(source)
    assert proposal.table.portable_origin is None
    actual=proposal.evaluate(np.zeros(len(source.radii)));expected=source.native(np.zeros(len(source.radii)))
    assert np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-30)<=1e-8
    (work/'result.json').write_text(json.dumps(report));sys.exit(0)
assert report['status']=='bound_certified_proposal_table',report
table=tof_profile_table._shared
assert table.portable_origin and table.portable_probes>0
def forbidden(*a,**k):raise AssertionError('A cold imported table was recompiled')
CompiledTOFTable.__init__=forbidden
proposal=for_source(source)
assert proposal is not None,'The default small-workload policy ignored the imported table'
assert proposal.table is table and proposal.diagnostics()['portable_table_bound']
rng=np.random.default_rng(91023);errors=[]
for point in [np.zeros(len(source.radii)),*[rng.uniform(-.5,.5,len(source.radii))*source.radii for _ in range(5)]]:
    actual=proposal.evaluate(point);expected=source.native(point)
    hist=source.fast.slow.hist.data['data'][1]
    background=np.zeros(len(expected));background[source.fast.slice]=hist[4][source.fast.slice]
    delta=actual-expected
    relative=float(np.linalg.norm(delta)/max(np.linalg.norm(expected-background),1e-30))
    sigma=float(np.max(abs(delta)*np.sqrt(hist[2])))
    assert relative<=1e-8 and sigma<=.01,(relative,sigma)
    errors.append(dict(relative_error=relative,maximum_sigma_error=sigma))
assert proposal.diagnostics()['portable_table_reused']
from functools import partial
from gpx_recipe.structural_live import service
from gpx_recipe.structural_live.optimization import profiled_native_fit
from gpx_recipe.symmetry.compiler import file_sha
before=file_sha(source_gpx)
child=Path(native_source.__code__.co_filename).with_name('data')/'synthetic-ersi-cmcm-isodistort.cif'
settings=service.SearchSettings(seconds=4,starts=2)
prepared=work/'service-prepared'
with server_scope(envelope,work):
    service.prepare_model(source_gpx,child,[[-1.,0,0],[0,-1.,0],[0,0,1.]],[0,0,0],prepared,
        settings=settings,prepared_profiles=choices)
contract=json.loads((prepared/'starting-curve-contract.json').read_text())
assert contract['portable_preparation']['profile_table']['status']=='bound_certified_proposal_table'
execution=contract['construction_execution']['continuous_profile']
assert execution['portable_table_reused'] and execution['statistics']['interpolated_bins']>0
assert contract['starting_curves']+contract['chemically_infeasible_points']==185
profiled_native_fit.fit=partial(profiled_native_fit.fit,qualification_iterations=1)
fit=service.fit_snapshot(prepared,work/'service-fit',settings=settings)
assert fit['verification']['relative_error']<=1e-8 and fit['verification']['max_sigma_error']<=.01
assert all(v['weighted_objective_verified'] for v in fit['verification']['saved_observations'].values())
assert fit['all_physical_modes_retained'] and fit['optimization_qualified'] is False
assert file_sha(source_gpx)==before
report.update(errors=errors,renderer=proposal.diagnostics(),total_current_seconds=time.monotonic()-began,
    attachment=attachment,actual_bins=len(source.fast.slow.hist.data['data'][1][0]),
    original_source_bins=len(original_x),source_grid_preserved=True,
    likelihood_bins=len(source.fast.x),prepared_bins=library.document['entries'][0]['model']['grid']['bins'],
    service_preparation=contract['portable_preparation'],service_execution=execution,
    service_verification=fit['verification'])
(work/'result.json').write_text(json.dumps(report))
'''


@pytest.fixture(scope='module')
def portable_reference(tmp_path_factory):
    root=tmp_path_factory.mktemp('portable-tof-profile')
    source,_=native_source(root/'public-inputs','PNT')
    cif=root/'parent.cif';G.G2Project(gpxfile=str(source)).phase(0).export_CIF(str(cif))
    library=create_library(root/'library',name='Public theory-only TOF preparation',source=cif,
        banks={'PNT reference':dict(instrument=root/'public-inputs/profile.instprm',
            grid=dict(kind='PNT',lower=3000.,upper=40000.,bins=701,bank=1))},
        models=[dict(kind='reference',bank='PNT reference',settings=dict(maximum_denominator=8))])
    row=library.document['entries'][0];attempt=library.start(row['id']);captured=root/'captured'
    request=stage_entry(library,row['id'],attempt,captured);payload=copy.deepcopy(request)
    names=[]
    for key in ('source','instrument'):
        payload[key]=str(Path(payload[key]).relative_to(captured));names.append(payload[key])
    payload['destination']='compiled';source=captured/payload['source'];pack(captured,root/'inputs.zip',names)
    message=dict(schema='radar-model-native/v1',operation='preparation-reference',payload=payload,
        implementation_sha256=package_signature(),expected_source_sha256=file_sha(source),dispatch_key='d'*24,
        timeout=120,input_hashes={name:file_sha(captured/name) for name in names})
    write_json(root/'request.json',message)
    wrapper=Path(__file__).resolve().parents[2]/'scripts/ndip_model_campaign.py'
    certificate=root/'certificate.json'
    with (root/'producer.log').open('w') as stream:
        result=subprocess.run([sys.executable,str(wrapper),'native',str(source),str(root/'inputs.zip'),
            str(root/'request.json'),str(root/'job'),str(root/'result.zip'),
            '--preparation-certificate',str(certificate)],timeout=150,stdout=stream,stderr=subprocess.STDOUT)
    assert result.returncode==0,(root/'producer.log').read_text()[-16000:]
    announced=subprocess.check_output([sys.executable,'-m','gpx_recipe.preparation.certificates',str(certificate)],text=True)
    cert=certificates.read(certificate)
    assert len(cert['entries'])==1 and cert['entries'][0]['kind']=='reference'
    assert set(cert['entries'][0]['artifacts'])=={'profile.npz','profile.json'}
    received=unpack(root/'result.zip',root/'received');outer=json.loads((received/'response.json').read_text())
    output=outer['result'];shutil.copytree(received/'native/compiled',request['destination']);output['output']=request['destination']
    publish_entry(library,row['id'],attempt,request,output)
    library.add_evidence(dict(site='https://galaxy.invalid',dataset_id='profile-origin',sha256=file_sha(certificate),
        entries=[output['entry_sha256']]))
    library.export(root/'portable.zip');cold=import_library(root/'portable.zip',root/'cold-library')
    for directory in (root/'job',library.root,captured,received,root/'public-inputs'):shutil.rmtree(directory)
    assert not list(cold.root.rglob('*.gpx'))
    envelope=root/'server.json';write_json(envelope,dict(schema=authority.SCHEMA,records=[dict(
        certificate_path=str(certificate),tool_id='neutrons_radar_pd_model_campaign_prototype',tool_version='0.2.17',
        job_state='ok',dataset_state='ok',dataset_deleted=False,dataset_purged=False,job_stdout=announced)]))
    consumer=root/'consumer.py';consumer.write_text(CONSUMER)
    return root,cold.root,envelope,consumer


@pytest.mark.parametrize('mode',['certified','unverified','shape','runtime'])
def test_new_worker_uses_only_certified_matching_profile_bytes(portable_reference,mode):
    root,library,envelope,consumer=portable_reference;work=root/mode
    with (root/(mode+'.log')).open('w') as stream:
        result=subprocess.run([sys.executable,str(consumer),str(library),str(work),str(envelope),mode],
            timeout=150,stdout=stream,stderr=subprocess.STDOUT)
    assert result.returncode==0,(root/(mode+'.log')).read_text()[-16000:]
    report=json.loads((work/'result.json').read_text())
    if mode=='certified':
        assert report['prepared_bins']==701 and report['actual_bins']!=report['prepared_bins']
        assert report['actual_bins']==report['original_source_bins'] and report['source_grid_preserved']
        assert 2<=report['likelihood_bins']<=report['actual_bins']
        assert len(report['errors'])==6
        assert report['renderer']['portable_origin']['native_likelihood_authority'] is False
    else:assert report['discarded_entries']
