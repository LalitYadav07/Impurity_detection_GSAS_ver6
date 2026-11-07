from functools import partial
from pathlib import Path
import json
import numpy as np
import pytest

pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_profile_capture import native_source
from gpx_recipe.structural_live.optimization.tof_profile_proposals import TOFProfileProposals
from gpx_recipe.structural_live.optimization.native_cell_proposals import ProposalUnavailable


def tail_source(root):
    filename,source=native_source(root,'PNT')
    n=source.fast.native;prefix=f':{source.fast.slow.hist.id}:'
    for name,value in [('alpha',.01),('beta-0',.0002),('beta-1',.0003),('sig-0',1.),('sig-1',2.)]:
        n.parms[prefix+name]=value
    source.native(np.zeros(len(source.radii)))
    return filename,source


@pytest.mark.parametrize('profile',['regular','rough','packaged_powgen'])
def test_native_compiled_proposals_retain_mixed_geometry_and_observations(tmp_path,profile):
    if profile=='rough':_,source=tail_source(tmp_path/'input')
    else:_,source=native_source(tmp_path/'input','PNT')
    n=source.fast.native;hist=n.histo_phases[0][source.fast.slow.hist.name]
    if profile=='packaged_powgen':
        preset=Path(__file__).parents[2]/'examples/four_phase/2018B_HighRes_60HzB1_CWL0p8.instprm'
        prefix=f':{source.fast.slow.hist.id}:'
        names={'alpha','beta-0','beta-1','beta-q','sig-0','sig-1','sig-2','sig-q','X','Y','Z'}
        for line in preset.read_text().splitlines():
            name,_,value=line.partition(':')
            if name in names:n.parms[prefix+name]=float(value)
        source.native(np.zeros(len(source.radii)))
    original=[np.ma.array(v,copy=True) for v in hist['Data'][:3]]
    proposal=TOFProfileProposals(source)
    rng=np.random.default_rng(172905)
    for point in [np.zeros(len(source.radii)),
            *[rng.uniform(-.7,.7,len(source.radii))*source.radii for _ in range(4)]]:
        got=proposal.evaluate(point);expected=source.native(point)
        background=np.zeros(len(expected));background[source.fast.slice]=hist['Data'][4][source.fast.slice]
        delta=got-expected
        error=np.linalg.norm(delta)/max(np.linalg.norm(expected-background),1e-30)
        sigma=np.max(abs(delta)*np.sqrt(hist['Data'][2]))
        assert error<=1e-8 and sigma<=.01
    report=proposal.diagnostics()
    assert report['statistics']['native_bins']>0
    if profile=='packaged_powgen':assert report['statistics']['interpolated_bins']>0
    if profile=='rough':assert report['rejected_interpolation_cells']>0
    assert report['native_midpoint_checks']>0
    assert report['native_origin_authority'] is False
    assert report['observations_used_for_compilation'] is False
    assert report['maximum_accounted_workspace_bytes']<=64*1024*1024
    for actual,expected in zip(hist['Data'][:3],original):np.testing.assert_array_equal(actual,expected)
    with pytest.raises(ValueError):proposal.coefficients.flags.writeable=True


def test_later_profile_edit_requires_regeneration(tmp_path):
    _,source=tail_source(tmp_path/'input');proposal=TOFProfileProposals(source)
    source.fast.native.parms[f':{source.fast.slow.hist.id}:X']+=.04
    with pytest.raises(ProposalUnavailable,match='profile changed'):
        proposal.evaluate(np.zeros(len(source.radii)))
    assert np.isfinite(source.native(np.zeros(len(source.radii)))).all()


def test_tabulated_profile_keeps_original_native_renderer(tmp_path):
    _,source=tail_source(tmp_path/'input')
    source.fast.native.histdict[f':{source.fast.slow.hist.id}:pdabc']={'d':[.1,3.,20.],'sig':[.01,.07,.03]}
    with pytest.raises(ProposalUnavailable,match='Tabulated'):
        TOFProfileProposals(source)


@pytest.mark.parametrize('clear_owned_origin',[False,True])
def test_shared_service_uses_proposals_without_private_initialization_injection(tmp_path,monkeypatch,clear_owned_origin):
    from gpx_recipe.structural_live import service
    from gpx_recipe.structural_live.optimization import starting_proposals,profiled_native_fit,tof_profile_proposals
    filename,_=native_source(tmp_path/'input','PNT')
    child=Path(__file__).with_name('data')/'synthetic-ersi-cmcm-isodistort.cif'
    monkeypatch.setattr(tof_profile_proposals,'MINIMUM_REFERENCE_BINS',0)
    monkeypatch.setattr(profiled_native_fit,'fit',partial(profiled_native_fit.fit,qualification_iterations=1))
    settings=service.SearchSettings(seconds=30,starts=2)
    prepared=tmp_path/'current'
    service.prepare_model(filename,child,[[-1.,0,0],[0,-1.,0],[0,0,1.]],[0,0,0],prepared,settings=settings)
    receipt=json.loads((prepared/'starting-curves.json').read_text())
    assert receipt['curve_role']=='starting_proposals' and receipt['native_curves']==0
    assert receipt['starting_curves']+receipt['chemically_infeasible_points']==185
    before=(prepared/'starting-curves.npz').read_bytes()
    if clear_owned_origin:starting_proposals._origins.clear()
    result=service.fit_snapshot(prepared,tmp_path/'fit',settings=settings)
    assert result['initialization_execution']['curve_role']=='starting_proposals'
    assert result['initialization_execution']['regenerated'] is clear_owned_origin
    assert result['starting_curves_reused'] is (not clear_owned_origin)
    assert result['verification']['relative_error']<=1e-8 and result['verification']['max_sigma_error']<=.01
    assert all(v['weighted_objective_verified'] for v in result['verification']['saved_observations'].values())
    assert result['all_physical_modes_retained'] and result['optimization_qualified'] is False
    assert (prepared/'starting-curves.npz').read_bytes()==before
