"""Actual CIF preparation, producer removal and cold native bank consumption.

The Galaxy creating-job envelope is simulated here. Native preparation,
wrapper output, portable arrays and both fresh consumer processes are real.
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
from gpx_recipe.preparation.reference import create_reference
from gpx_recipe.preparation.structural_consume import prepare_starting_bank
from gpx_recipe.preparation.library import PreparationLibrary
from gpx_recipe.structural_live.native import prepare
from gpx_recipe.structural_live.service import make_source,SearchSettings
from gpx_recipe.structural_live.optimization.offline_native_start_bank import OfflineNativeStartBank
from gpx_recipe.symmetry.compiler import file_sha
library=PreparationLibrary(sys.argv[1]);work=Path(sys.argv[2]);work.mkdir()
row=library.document['entries'][0];model=row['model'];settings=model['settings']
began=time.monotonic()
reference=create_reference(library.root/library.document['specification']['source'],
    library.root/model['instrument'],work/'reference',**model['grid'])
changed_nuisance=len(sys.argv)>4 and sys.argv[4]=='changed'
if changed_nuisance:
    from GSASII import GSASIIscriptable as G
    project=G.G2Project(gpxfile=str(reference['path']));hist=project.histograms()[0]
    hist.data['Sample Parameters']['Scale'][0]*=1.7
    project.phase(0).data['Histograms'][hist.name]['Scale'][0]*=.73
    hist.data['Background'][0][3:]=[-2.]+[.1]*(len(hist.data['Background'][0])-4)
    project.save()
prepare(reference['path'],library.root/model['child'],settings['candidate']['transformation_matrix'],
    settings['candidate']['origin_shift'],work/'prepared')
source=make_source(work/'prepared',work/'evaluation',SearchSettings(mode_radius=settings['mode_radius'],cell_radius=settings['cell_radius']))
context_seconds=time.monotonic()-began
assets={Path(a['path']).name:library.root/a['path'] for a in row['files']}
choice=dict(entry=row['id'],bank=model['bank'],files=dict(path=str(assets['starts.npz']),receipt_path=str(assets['starts.json'])),
    sha256=file_sha(assets['starts.npz']),receipt_sha256=file_sha(assets['starts.json']))
calls=[];native=source.native
def counted(point):calls.append(1);return native(point)
source.native=counted
with server_scope(None if sys.argv[3]=='none' else sys.argv[3],work):
    receipt,report=prepare_starting_bank(source,work/'starts.npz',[choice])
bank=OfflineNativeStartBank(source,work/'starts.npz')
if sys.argv[3]!='none':
    assert report['status']=='reused_certified_native_bank',report
    assert report['native_curves_recomputed']==3
    assert len(calls)==(5 if changed_nuisance else 3),len(calls)
    assert bool(report['affine_nuisance_binding'])==changed_nuisance
else:
    assert report['status']=='native_requalified',report
    schedule=receipt['construction_execution']
    forwarded=schedule.get('native_forward_requests',0)
    assert schedule['direct_calls']+schedule['cached_calls']+forwarded>=receipt['native_curves']
    if forwarded:
        native_forward=schedule['native_forward']
        assert native_forward['requests']==forwarded
        assert native_forward['rendered']+native_forward['native_fallback_calls']==forwarded
        assert native_forward['qualified_anchor'] and not native_forward['disabled']
        assert native_forward['native_anchor_checks']==1
assert receipt['proposed_points']==185
assert len(bank.points)==receipt['native_curves']
assert report['scientific_candidates_removed'] is False
np.savez_compressed(work/'checked.npz',points=bank.points,curves=bank.curves,steps=bank.proposal_cell_steps)
(work/'result.json').write_text(json.dumps(dict(context_seconds=context_seconds,qualification=report,
    complete_native_curves=receipt['native_curves'],source_native_calls=len(calls),
    construction_execution=receipt.get('construction_execution'),mode_dimensions=source.mode_dimensions,
    cell_dimensions=source.cell_dimensions)))
'''


@pytest.mark.parametrize('kind,changed_nuisance',[('PNC',False),('PNT',False),('PNT',True)])
def test_cold_portable_structural_bank_reuses_native_receipt_without_producer_files(tmp_path,kind,changed_nuisance):
    source,_=native_source(tmp_path/'public-inputs',kind)
    cif=tmp_path/'parent.cif';G.G2Project(gpxfile=str(source)).phase(0).export_CIF(str(cif))
    child=Path(__file__).with_name('data')/'synthetic-ersi-cmcm-isodistort.cif'
    settings=dict(candidate=dict(id='structural:test',transformation_matrix=[[-1.,0,0],[0,-1.,0],[0,0,1.]],
        origin_shift=[0,0,0]),mode_radius=.05,cell_radius=.005)
    library=create_library(tmp_path/'library',name='Public cold preparation',source=cif,
        banks={kind:dict(instrument=tmp_path/'public-inputs/profile.instprm',
            grid=dict(kind=kind,lower=12. if kind=='PNC' else 3000.,upper=125. if kind=='PNC' else 40000.,bins=701,bank=1))},
        models=[dict(kind='structural',bank=kind,settings=settings,child=child)])
    row=library.document['entries'][0];attempt=library.start(row['id'])
    captured=tmp_path/'captured';staged=stage_entry(library,row['id'],attempt,captured)
    payload=copy.deepcopy(staged)
    names=[]
    for key in ('source','instrument','child'):
        payload[key]=str(Path(payload[key]).relative_to(captured));names.append(payload[key])
    payload['destination']='compiled'
    source=captured/payload['source'];pack(captured,tmp_path/'inputs.zip',names)
    message=dict(schema='radar-model-native/v1',operation='preparation-structural',payload=payload,
        implementation_sha256=package_signature(),expected_source_sha256=file_sha(source),
        dispatch_key='d'*24,timeout=240,input_hashes={name:file_sha(captured/name) for name in names})
    write_json(tmp_path/'request.json',message)
    wrapper=Path(__file__).resolve().parents[2]/'scripts/ndip_model_campaign.py'
    certificate=tmp_path/'certificate.json';began=time.monotonic()
    with (tmp_path/'producer.log').open('w') as stream:
        subprocess.run([sys.executable,str(wrapper),'native',str(source),str(tmp_path/'inputs.zip'),
            str(tmp_path/'request.json'),str(tmp_path/'job'),str(tmp_path/'result.zip'),
            '--preparation-certificate',str(certificate)],check=True,timeout=270,stdout=stream,stderr=subprocess.STDOUT)
    preparation_seconds=time.monotonic()-began
    announced=subprocess.check_output([sys.executable,'-m','gpx_recipe.preparation.certificates',str(certificate)],text=True)
    assert announced==certificates.MARKER+file_sha(certificate)+'\n'
    document=certificates.read(certificate)
    assert len(document['entries'])==1 and document['entries'][0]['kind']=='structural'
    assert document['result_archive_sha256']==file_sha(tmp_path/'result.zip')
    received=unpack(tmp_path/'result.zip',tmp_path/'received')
    outer=json.loads((received/'response.json').read_text());result=outer['result']
    shutil.copytree(received/'native/compiled',staged['destination'])
    result['output']=staged['destination']
    publish_entry(library,row['id'],attempt,staged,result)
    library.add_evidence(dict(site='https://galaxy.invalid',dataset_id='native-receipt',sha256=file_sha(certificate),
        entries=[result['entry_sha256']]))
    archive=tmp_path/'portable.zip';library.export(archive)
    restored=import_library(archive,tmp_path/'cold-library')
    shutil.rmtree(tmp_path/'job');shutil.rmtree(library.root);shutil.rmtree(captured);shutil.rmtree(received)
    assert not list(restored.root.rglob('*.gpx'))
    envelope=tmp_path/'server.json'
    write_json(envelope,dict(schema=authority.SCHEMA,records=[dict(certificate_path=str(certificate),
        tool_id='neutrons_radar_pd_model_campaign_prototype',tool_version='0.3.4',job_state='ok',dataset_state='ok',
        dataset_deleted=False,dataset_purged=False,job_stdout=announced)]))
    consumer=tmp_path/'consumer.py';consumer.write_text(CONSUMER)
    results={}
    for label,origin in (('certified',str(envelope)),('unverified','none')):
        output=tmp_path/label
        with (tmp_path/(label+'.log')).open('w') as stream:
            subprocess.run([sys.executable,str(consumer),str(restored.root),str(output),origin,
                'changed' if changed_nuisance else 'same'],
                check=True,timeout=240,stdout=stream,stderr=subprocess.STDOUT)
        results[label]=json.loads((output/'result.json').read_text())
    assert results['certified']['mode_dimensions']==2 and results['certified']['cell_dimensions']==3
    assert results['certified']['complete_native_curves']==results['unverified']['complete_native_curves']
    with np.load(tmp_path/'certified/checked.npz') as a,np.load(tmp_path/'unverified/checked.npz') as b:
        np.testing.assert_array_equal(a['points'],b['points'])
        np.testing.assert_allclose(a['curves'],b['curves'],rtol=1e-11,atol=1e-11)
        if not changed_nuisance:np.testing.assert_array_equal(a['steps'],b['steps'])
        else:assert np.isfinite(a['steps']).all() and np.all(a['steps']>0)
    write_json(tmp_path/'cold-reuse-measurement.json',dict(preparation_seconds=preparation_seconds,results=results,
        scope='Public artificial '+kind+' child; real native wrapper and independent cold processes; Galaxy origin metadata simulated; '+
            ('changed nuclear scale/background at fixed geometry' if changed_nuisance else 'exact context')))
