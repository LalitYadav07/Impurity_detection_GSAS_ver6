"""Shared k physics versus independent ordinary native simulations."""
import copy
import json
from pathlib import Path
import time

import numpy as np
import pytest

G=pytest.importorskip('GSASII.GSASIIscriptable')
from test_preparation_theory_native import inputs
from gpx_recipe.execution import write_json
from gpx_recipe.symmetry.compiler import compile_bundle,file_sha,package_signature
from gpx_recipe.symmetry.contracts import digest,validate_protocol
from gpx_recipe.symmetry.k_group import simulate_group
from gpx_recipe.symmetry.k_native import template,portfolio
from gpx_recipe.symmetry.native_audit import inspect_source


def setup(root,kind,masked):
    root.mkdir();reference,construction,candidate,_=inputs(root,kind)
    project=G.G2Project(gpxfile=reference['path']);h=project.histogram(0)
    arrays=h.data['data'][1];arrays[1]=np.ones(len(arrays[0]));arrays[2]=np.ones(len(arrays[0]))
    h.data['data'][0]['Dummy']=False
    if masked:
        lo,hi=h.data['Limits'][1];h.data['Limits'].append([lo+.35*(hi-lo),lo+.4*(hi-lo)])
        a=np.ma.array(np.vstack(arrays),mask=False);a.mask[2,40:47]=True
        h.data['data'][1]=a
    source=root/'source.gpx';project.save(str(source));context=inspect_source(source)
    candidate=copy.deepcopy(candidate)
    catalog=dict(schema='radar-symmetry-catalog/v1',parent_setting_sha256=construction['parent_setting_sha256'],
        k=construction['k'],candidates=[candidate]);write_json(root/'catalog.json',catalog)
    protocol=validate_protocol(dict(kind='magnetic',source=dict(path='source.gpx',sha256=file_sha(source)),
        catalog=dict(path='catalog.json',sha256=file_sha(root/'catalog.json')),
        parent_phase=construction['parent_phase'],histogram=construction['histogram'],
        parent_setting_sha256=construction['parent_setting_sha256'],
        magnetic_function_sha256=context['runtime']['magnetic_function_sha256'],implementation_sha256=package_signature(),
        k=construction['k'],sites=construction['sites'],candidates=[candidate],
        seeds=[dict(id=f'{axis}-{i}',axis=axis,magnitude=amplitude,independent_mode_seed=seed)
               for i,(amplitude,seed) in enumerate([(.7,17),(1.7,31)]) for axis in 'xyz'],
        policy=dict(cycles=1,timeout_seconds=180,maximum_trials=6,maximum_cell_multiplier=8,maximum_moment=6.,
            magnetic_d_min=0.,fit_band_percent=.25,maximum_mode=.5,adequate_rwp_percent=100.,
            refine_cell=False,refine_histogram_scale=False,refine_background=False,moment_model='symmetry_allowed')))
    return source,protocol


@pytest.mark.parametrize('kind,masked',[('PNC',False),('PNT',True)])
def test_shared_profiles_preserve_every_seed_native_curve_and_original_masks(tmp_path,kind,masked):
    source,protocol=setup(tmp_path/'input',kind,masked);before=source.read_bytes()
    direct=[];grouped=[];timing={}
    for label,target in [('ordinary',direct),('grouped',grouped)]:
        for seed in protocol['seeds']:
            directory=tmp_path/label/seed['id']
            compile_bundle(protocol,source.parent,directory,candidate_id=protocol['candidates'][0]['id'],
                seed_id=seed['id'],approved_hash=digest(protocol));target.append(directory)
    began=time.monotonic();ordinary=[template(dict(directory=str(p)),p) for p in direct]
    timing['ordinary_seconds']=time.monotonic()-began
    began=time.monotonic();accelerated=simulate_group(grouped)
    timing['grouped_seconds']=time.monotonic()-began
    assert len(accelerated)==len(protocol['seeds'])
    for old,new,old_dir,new_dir in zip(ordinary,accelerated,direct,grouped):
        result=new['response'];assert result['status']=='simulated',result
        assert result['id']==old['id'] and result['protocol_sha256']==old['protocol_sha256']
        assert result['audit']['passed'] and result['periodicity']['passed'] and result['no_refinement']
        with np.load(old_dir/old['profile']) as a,np.load(new_dir/result['profile']) as b:
            assert a.files==b.files
            np.testing.assert_allclose(a['intensity'],b['intensity'],rtol=1e-8,atol=1e-10)
            for field in a.files:
                if field=='intensity':continue
                if a[field].dtype.kind in 'US':np.testing.assert_array_equal(a[field],b[field])
                else:np.testing.assert_allclose(a[field],b[field],rtol=1e-10,atol=1e-10)
        assert result['initial_maximum_moment']==pytest.approx(old['initial_maximum_moment'],rel=1e-12)
        assert result['shared_physics']['native_constructions']==1
        assert result['shared_physics']['declared_starts']==6
        assert result['shared_physics']['independent_seed_error']<=1e-8
    assert source.read_bytes()==before
    write_json(tmp_path/'timing.json',dict(timing,starts=6,kind=kind,masked=masked,
        scope='Artificial native equivalence and timing; not a material identification or NDIP latency benchmark'))


def test_large_group_consumes_prepared_factors_and_checks_all_native_seed_profiles(tmp_path):
    from gpx_recipe.preparation import hints
    from gpx_recipe.preparation.operations import create_library,stage_entry,publish_entry
    from gpx_recipe.preparation.worker import execute
    source,protocol=setup(tmp_path/'input','PNC',False)
    protocol['seeds']=[dict(id=f'{axis}-{i}',axis=axis,magnitude=.3+.08*i,independent_mode_seed=17+i)
        for i in range(16) for axis in 'xyz']
    protocol['policy']['maximum_trials']=48;protocol=validate_protocol(protocol)
    library=create_library(tmp_path/'library',name='Public before-measurement k control',
        source=source.parent/'parent.cif',
        banks={'bank 1':dict(instrument=source.parent/'bank.instprm',
            grid=dict(kind='PNC',lower=12.,upper=130.,bins=701,bank=1))},
        models=[dict(kind='magnetic',bank='bank 1',settings=dict(candidate=protocol['candidates'][0],
            k=protocol['k'],sites=protocol['sites'],maximum_moment=6.,
            refine_histogram_scale=False,refine_background=False))])
    entry=library.document['entries'][0]['id'];attempt=library.start(entry)
    request=stage_entry(library,entry,attempt,tmp_path/'offline')
    published=execute('magnetic',request);assert publish_entry(library,entry,attempt,request,published)['all_prepared']
    directories=[]
    for seed in protocol['seeds']:
        directory=tmp_path/'grouped'/seed['id']
        compile_bundle(protocol,source.parent,directory,candidate_id=protocol['candidates'][0]['id'],
            seed_id=seed['id'],approved_hash=digest(protocol))
        hints.capture(library,protocol,directory);directories.append(directory)
    began=time.monotonic();results=simulate_group(directories)
    grouped_seconds=time.monotonic()-began
    assert len(results)==48
    proof=results[0]['response']['shared_physics']
    assert proof['prepared_scattering']['attached']==1 and proof['prepared_scattering']['used']
    compiler=proof['native_compiler'];assert compiler['method']=='qualified_quadratic_response'
    assert compiler['proof']['compiler']['native_reflection_diagnostics']['scattering_factors_reused']
    began=time.monotonic()
    for seed,item,directory in zip(protocol['seeds'],results,directories):
        result=item['response'];assert result['status']=='simulated',result
        independent=tmp_path/'independent'/seed['id']
        compile_bundle(protocol,source.parent,independent,candidate_id=protocol['candidates'][0]['id'],
            seed_id=seed['id'],approved_hash=digest(protocol))
        check=template(dict(directory=str(independent)),independent)
        with np.load(directory/result['profile']) as actual,np.load(independent/check['profile']) as expected:
            np.testing.assert_allclose(actual['intensity'],expected['intensity'],rtol=1e-8,atol=1e-10)
        assert result['id']==check['id'] and result['audit']['passed'] and result['periodicity']['passed']
    write_json(tmp_path/'timing.json',dict(starts=48,grouped_seconds=grouped_seconds,
        ordinary_seconds=time.monotonic()-began,scope='Public artificial prepared-factor/native-equivalence control'))


def test_real_galaxy_wrapper_runs_a_shared_context_and_returns_all_original_identities(tmp_path):
    import subprocess
    import sys
    from gpx_recipe.symmetry.bundle import pack,unpack
    from gpx_recipe.symmetry.compiler import assets
    source,protocol=setup(tmp_path/'input','PNC',False)
    packet_root=tmp_path/'packet';names=[];directories=[]
    for seed in protocol['seeds']:
        directory=packet_root/seed['id'];directories.append(seed['id'])
        compile_bundle(protocol,source.parent,directory,candidate_id=protocol['candidates'][0]['id'],
            seed_id=seed['id'],approved_hash=digest(protocol))
        names.extend(str((directory/a['path']).relative_to(packet_root)) for a in assets(protocol))
        names.append(str((directory/'job.json').relative_to(packet_root)))
    pack(packet_root,tmp_path/'assets.zip',names)
    message=dict(schema='radar-model-native/v1',operation='k-portfolio',
        payload=dict(directories=directories,workers=2,seconds=180),
        implementation_sha256=package_signature(),expected_source_sha256=file_sha(source),
        dispatch_key='a'*24,timeout=200,
        input_hashes={name:file_sha(packet_root/name) for name in names})
    write_json(tmp_path/'request.json',message)
    script=Path(__file__).resolve().parents[2]/'scripts/ndip_model_campaign.py'
    with (tmp_path/'wrapper.log').open('w') as log:
        subprocess.run([sys.executable,str(script),'native',str(source),str(tmp_path/'assets.zip'),
            str(tmp_path/'request.json'),str(tmp_path/'worker'),str(tmp_path/'results.zip')],
            check=True,timeout=240,stdout=log,stderr=subprocess.STDOUT)
    restored=unpack(tmp_path/'results.zip',tmp_path/'received')
    response=json.loads((restored/'response.json').read_text())
    result=response['result']
    assert not result['pending'] and len(result['results'])==6 and len(result['group_runs'])==1
    expected_ids={json.loads((packet_root/d/'job.json').read_text())['id'] for d in directories}
    assert {r['response']['id'] for r in result['results']}==expected_ids
    assert all(r['response']['status']=='simulated' and r['response']['audit']['passed']
        and r['response']['periodicity']['passed'] for r in result['results'])
    assert all(r['response']['shared_physics']['native_constructions']==1 for r in result['results'])
