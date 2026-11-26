"""Offline preparation must preserve every native curve under four CPUs."""
import json
from pathlib import Path
import time

import numpy as np
import pytest

G=pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_profile_capture import native_source
from gpx_recipe.preparation.worker import execute
from gpx_recipe.symmetry.compiler import file_sha


@pytest.mark.parametrize('kind',['PNC','PNT'])
def test_offline_bank_matches_serial_native_arrays_and_keeps_full_portfolio(tmp_path,monkeypatch,kind):
    # Force both schedules on the small, completely public native fixture;
    # production keeps this inexpensive two-mode case serial.
    from gpx_recipe.preparation import worker
    monkeypatch.setattr(worker,'MINIMUM_PARALLEL_STRUCTURAL_MODES',0)
    source,_=native_source(tmp_path/'reference',kind)
    parent=tmp_path/'parent.cif';G.G2Project(gpxfile=str(source)).phase(0).export_CIF(str(parent))
    child=Path(__file__).with_name('data')/'synthetic-ersi-cmcm-isodistort.cif'
    profile=tmp_path/'reference/profile.instprm'
    hashes=[file_sha(p) for p in (parent,child,profile)]
    payload=dict(source=str(parent),instrument=str(profile),child=str(child),
        grid=dict(kind=kind,lower=12. if kind=='PNC' else 3000.,
                  upper=125. if kind=='PNC' else 40000.,bins=701,bank=1),
        settings=dict(candidate=dict(id='structural:allocation-control',
            transformation_matrix=[[-1.,0,0],[0,-1.,0],[0,0,1.]],origin_shift=[0,0,0]),
            mode_radius=.05,cell_radius=.005))
    arrays=[];receipts=[];timings=[]
    for workers in (1,4):
        monkeypatch.setenv('GALAXY_SLOTS',str(workers));monkeypatch.setenv('GALAXY_MEMORY_MB','8192')
        destination=tmp_path/f'workers-{workers}'
        start=time.monotonic();result=execute('structural',dict(payload,destination=str(destination)))
        timings.append(time.monotonic()-start)
        assert result['status']=='prepared' and not result['observations_included']
        assert not result['scientific_result']
        with np.load(destination/'starts.npz',allow_pickle=False) as data:
            arrays.append({name:data[name].copy() for name in data.files})
        receipts.append(json.loads((destination/'starts.json').read_text()))
    assert arrays[0].keys()==arrays[1].keys()
    for name in arrays[0]:
        if name=='curves':np.testing.assert_allclose(arrays[0][name],arrays[1][name],rtol=1e-10,atol=1e-10)
        else:np.testing.assert_array_equal(arrays[0][name],arrays[1][name])
    one,four=receipts
    assert one['context']==four['context']
    assert one['proposed_points']==four['proposed_points']==185
    assert one['native_curves']==four['native_curves']==len(arrays[0]['curves'])
    assert not one['interpolation'] and not four['interpolation']
    execution=four['parallel_execution']
    assert execution['workers']==execution['peak_active_tasks']==4
    assert len(execution['worker_pids'])==4
    assert execution['completed_curves']==four['native_curves']
    assert execution['every_context_verified'] and execution['all_points_retained']
    assert hashes==[file_sha(p) for p in (parent,child,profile)]
    (tmp_path/'timings.json').write_text(json.dumps(dict(kind=kind,serial_seconds=timings[0],
        parallel_seconds=timings[1],execution=execution),indent=2)+'\n')
