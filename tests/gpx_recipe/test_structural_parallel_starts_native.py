"""Matched-work controls for isolated, allocation-bounded native starts."""
import numpy as np
import pytest

pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_profile_capture import native_source
from gpx_recipe.structural_live.service import SearchSettings,prepare_model,make_fit_context
from gpx_recipe.structural_live.optimization.offline_native_start_bank import OfflineNativeStartBank
from gpx_recipe.structural_live.optimization.parallel_starts import fit_allocated
from gpx_recipe.structural_live.optimization.captured_cell_derivative import CapturedCellDerivative


@pytest.mark.parametrize('kind',['PNC','PNT'])
def test_parallel_starts_keep_the_same_native_objective_and_complete_portfolio(tmp_path,monkeypatch,kind):
    from pathlib import Path
    source_gpx,_=native_source(tmp_path/'inputs',kind)
    child=Path(__file__).with_name('data')/'synthetic-ersi-cmcm-isodistort.cif'
    settings=SearchSettings(seconds=45.,starts=4)
    prepared=tmp_path/'prepared'
    prepare_model(source_gpx,child,[[-1.,0,0],[0,-1.,0],[0,0,1.]],
                  [0,0,0],prepared,settings=settings)
    results=[]
    for workers in (1,4):
        monkeypatch.setenv('GALAXY_SLOTS',str(workers));monkeypatch.setenv('GALAXY_MEMORY_MB','8192')
        folder=tmp_path/f'workers-{workers}'
        source,observation,table,_=make_fit_context(prepared,folder/'context',settings)
        bank=OfflineNativeStartBank(source,prepared/'starting-curves.npz')
        result,rate=fit_allocated(prepared,folder/'parallel',settings,source,table,observation,bank,
            evaluator=CapturedCellDerivative(source,max_cells=2),qualification_iterations=2)
        assert len(result['attempts'])==result['starts']==4
        assert result['all_allowed_modes_retained'] and not result['global_optimum_certified']
        np.testing.assert_allclose(result['best']['deviance'],
            np.sum((rate[observation.mask]-observation.y[observation.mask])**2/observation.variance[observation.mask]),rtol=1e-10)
        results.append((result,rate))
    serial,parallel=results
    np.testing.assert_allclose(serial[0]['best']['parameters'],parallel[0]['best']['parameters'],rtol=1e-9,atol=1e-10)
    np.testing.assert_allclose(serial[1],parallel[1],rtol=1e-9,atol=1e-9)
    np.testing.assert_allclose([a['initial_parameters'] for a in serial[0]['attempts']],
                               [a['initial_parameters'] for a in parallel[0]['attempts']],rtol=0,atol=0)
    execution=parallel[0]['parallel_execution']
    assert execution['workers']==execution['peak_active_workers']==4
    assert len({row['pid'] for row in execution['processes']})==4
    assert execution['total_start_allocation_seconds']<=settings.seconds
    assert all(row['cpu_seconds']>0 for row in execution['processes'])


def test_parallel_start_bank_retains_every_native_curve(tmp_path,monkeypatch):
    from pathlib import Path
    import json
    source_gpx,_=native_source(tmp_path/'inputs','PNT')
    child=Path(__file__).with_name('data')/'synthetic-ersi-cmcm-isodistort.cif'
    banks=[]
    for workers in (1,4):
        monkeypatch.setenv('GALAXY_SLOTS',str(workers));monkeypatch.setenv('GALAXY_MEMORY_MB','8192')
        prepared=tmp_path/f'prepared-{workers}'
        prepare_model(source_gpx,child,[[-1.,0,0],[0,-1.,0],[0,0,1.]],
                      [0,0,0],prepared,settings=SearchSettings())
        with np.load(prepared/'starting-curves.npz',allow_pickle=False) as arrays:
            banks.append({name:arrays[name].copy() for name in arrays.files})
    assert banks[0].keys()==banks[1].keys()
    for name in banks[0]:
        if name=='curves':np.testing.assert_allclose(banks[0][name],banks[1][name],rtol=1e-10,atol=1e-10)
        else:np.testing.assert_array_equal(banks[0][name],banks[1][name])
    receipt=json.loads((prepared/'starting-curves.json').read_text())
    assert receipt['parallel_execution']['peak_active_tasks']==4
    assert receipt['parallel_execution']['completed_curves']==len(banks[0]['curves'])
    assert receipt['native_curves']==len(banks[0]['curves']) and not receipt['interpolation']
