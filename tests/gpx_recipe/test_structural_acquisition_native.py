"""Actual POWGEN-format transfer, nonlinear updates and independently saved GPXs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

G=pytest.importorskip('GSASII.GSASIIscriptable')
from GSASII import GSASIIstrMain as MAIN
from test_native_profile_capture import native_source
from gpx_recipe.structural_live.acquisition import prepare_acquisition,StructuralAcquisition
from gpx_recipe.structural_live.service import make_source,SearchSettings
from gpx_recipe.structural_live.native import sha
from gpx_recipe.live.snapshot import parse_live_ascii,uniform_density_to_tof


def ascii_pattern(x,y,sigma,factor):
    return ('# X , Y , E Distribution=true\n0\n'+'\n'.join(
        f'{d:.17g},{intensity:.17g},{error:.17g}'
        for d,intensity,error in zip(x/factor,y*factor,np.asarray(sigma)*factor))+'\n').encode()


def test_real_native_acquisition_replaces_snapshots_and_publishes_the_matching_structural_gpx(tmp_path):
    source,_=native_source(tmp_path/'source','PNT')
    checksum=sha(source);factor=6200.
    # Generate native observations on a separate uniform exporter grid.
    project=G.G2Project(gpxfile=str(source),newgpx=str(tmp_path/'exporter-control.gpx'))
    histogram=project.histogram(0);arrays=histogram.data['data'][1]
    x=np.linspace(float(arrays[0][0]),float(arrays[0][-1]),len(arrays[0]))
    histogram.data['data'][1]=[x,np.ones_like(x),np.ones_like(x)*4,
        np.zeros_like(x),np.zeros_like(x),np.zeros_like(x)]
    project.save();initial=np.asarray(MAIN.DoNoFit(str(project.filename),histogram.name),float)
    sigma=np.full(len(x),max(float(np.max(abs(initial)))*1e-4,.001))
    raw=ascii_pattern(x,initial,sigma,factor);snapshot=tmp_path/'snapshot.dat';snapshot.write_bytes(raw)
    identity=dict(facility='SNS',instrument='PG3',ipts=1,run=2,epoch='public-native-control')
    scope=dict(run='2',sample_id='public-ersi',sample='Public ErSi',wavelength='1.5')
    mapping=dict(mapping_kind='user_reviewed',source_gpx_sha256=checksum,
        snapshot_sha256=sha(snapshot),physical_bank=1,identity=identity,
        export_geometry=dict(kind='linear_tof_from_d',microseconds_per_angstrom=factor),
        normalization='Public controlled reduced density',producer_config_sha256='a'*64)
    request=dict(source_gpx=str(source),source_sha256=checksum,
        observation_source=str(snapshot),observation_source_sha256=sha(snapshot),
        source_revision='PG3_2:'+sha(snapshot),reference_revision='PG3_1:'+checksum,
        parent_phase='Public ErSi',mapping=mapping,refine_transfer=False)
    child=Path(__file__).with_name('data')/'synthetic-ersi-cmcm-isodistort.cif'
    settings=SearchSettings(seconds=6.,starts=2)
    prepared=tmp_path/'acquisition'
    contract=prepare_acquisition(request,child,[[-1.,0,0],[0,-1.,0],[0,0,1.]],
        [0,0,0],prepared,source_scope=scope,settings=settings)
    assert contract['kind']=='structural_point_fit' and not contract['forecast_enabled']
    transfer=json.loads((prepared/'transfer.json').read_text())
    assert (prepared/transfer['source_dat']).read_bytes()==raw
    assert sha(prepared/transfer['source_dat'])==transfer['source_dat_sha256']
    assert sha(source)==checksum
    context=make_source(prepared/'model',tmp_path/'new-snapshot-truth',settings)
    point=np.linspace(-.04,.06,len(context.radii))*context.radii
    true_curve=context.native(point)
    actual_x=np.asarray(context.fast.x,float)
    sigma=np.full(len(actual_x),max(float(np.max(abs(true_curve)))*1e-4,.001))
    owner=StructuralAcquisition(prepared,tmp_path/'session');rows=[]
    for revision in (1,2):
        noise=np.random.default_rng(310+revision).normal(0.,sigma/revision)
        observed=true_curve+noise;errors=sigma/revision
        if revision==1:
            observed[37]=0.;errors=errors.copy();errors[37]=0.
        raw=ascii_pattern(actual_x,observed,errors,factor)
        receipt=dict(identity=identity,content_sha256=hashlib.sha256(raw).hexdigest(),
            metadata=dict(cumulative_exposure=10.*revision,sample_id=scope['sample_id'],
                          physical_bank=1,producer_config_sha256='a'*64))
        result=owner.update(raw,receipt,revision=revision,current_scope=scope,current_mapping=mapping)
        assert result['state']=='verified_point_fit' and result['kind']=='structural_point_fit'
        assert result['previous_likelihood_accumulated'] is False
        assert result['all_physical_modes_retained']
        assert result['physical_mode_count']==context.mode_dimensions
        assert result['cell_parameter_count']==context.cell_dimensions
        assert result['starting_curves_reused'] and not result['model_exclusion_authorized']
        assert result['verification']['relative_error']<1e-8 and result['verification']['max_sigma_error']<.01
        measured=uniform_density_to_tof(parse_live_ascii(raw),export_difc=factor)
        for role in ('frozen','continuation'):
            item=result[role];saved=G.G2Project(gpxfile=item['path'])
            arrays=saved.histogram(0).data['data'][1]
            np.testing.assert_array_equal(arrays[0],measured['x'])
            np.testing.assert_array_equal(arrays[1],measured['y'])
            np.testing.assert_allclose(arrays[2],measured['weights'],rtol=1e-12,atol=0)
            assert result['verification']['saved_observations'][role]['weighted_objective_verified']
            assert sha(item['path'])==item['sha256']
        rows.append(dict(revision=revision,rwp=result['metrics']['rwp_native'],
            timings=result['timings'],physical_modes=result['physical_mode_count'],
            cell_parameters=result['cell_parameter_count'],snapshot_sha256=result['snapshot_sha256']))
        # A fresh owner restores the same native publication and its archived
        # source bytes. Duplicate delivery performs no new nonlinear work.
        owner=StructuralAcquisition(prepared,owner.root)
        repeated=owner.update(raw,receipt,revision=revision,current_scope=scope,current_mapping=mapping)
        assert repeated['state']=='duplicate' and repeated['new_fit'] is False
    assert sha(source)==checksum
    (tmp_path/'acceptance.json').write_text(json.dumps(dict(rows=rows,
        scope='Public artificial POWGEN-format acquisition control; no material or phase-transition identification claim'),indent=2)+'\n')
