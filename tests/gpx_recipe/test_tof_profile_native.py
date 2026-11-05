"""Controlled public-model recovery, independent of phase-discovery data."""
from pathlib import Path
import numpy as np
import pytest

G = pytest.importorskip('GSASII.GSASIIscriptable')
from GSASII import GSASIIstrMain as main
from gpx_recipe.execution import freeze, run_plan
from gpx_recipe.snapshot import snapshot
from gpx_recipe.contracts import plan_hash


@pytest.mark.parametrize('term', ['X', 'Y', 'Z'])
@pytest.mark.parametrize('other_bank', [False, True])
def test_recovers_one_profile_term_without_moving_cell_or_position_calibration(tmp_path, term, other_bank):
    inst = Path(__file__).resolve().parents[2]/'nova/src/radar_pd_nova/data/powgen_calibrations/2025_Jan_HighRes_60HzB3_CWL2p665.instprm'
    path = tmp_path/'source.gpx'
    p = G.G2Project(newgpx=str(path))
    phase = p.add_phase(phasename='Public artificial oxide', cell=[4.7,5.6,6.8,79,93,104])
    phase.add_atom(.37,.29,.19,'O','O1')
    phase.add_atom(.13,.21,.34,'Fe','Fe1')
    h = p.add_simulated_powder_histogram('TOF control',str(inst),10,100,Npoints=4000,phases=[phase],scale=100000)
    held = None
    if other_bank:
        held = p.add_simulated_powder_histogram('Second bank held fixed',str(inst),10,100,Npoints=4000,phases=[phase],scale=100000)
    freeze(p)
    # Use a broad-domain artificial positive profile; the facility profile's
    # empirical asymmetry is only qualified in its original reduction range.
    truth_values = {'X':2., 'Y':.8, 'Z':2.5}
    rng = np.random.default_rng(46100)
    for bank in p.histograms():
        for name,value in {'alpha':.3,'beta-0':.02,'beta-1':0.,'beta-q':0.,'sig-0':100.,'sig-1':0.,'sig-2':0.,'sig-q':0.}.items():
            bank.InstrumentParameters[name][1]=value
        for name,value in truth_values.items():bank.InstrumentParameters[name][1]=value
        bank.data['data'][1][1][:]=1.
    p.save()
    for bank in p.histograms():
        p.save()
        truth = np.asarray(main.DoNoFit(str(path),bank.name),float)
        bank.data['data'][1][1]=truth+rng.normal(0,.03,len(truth))
        bank.data['data'][1][2]=np.full(len(truth),1/.03**2)
        bank.data['data'][0]['Dummy']=False
    h.InstrumentParameters[term][1]=truth_values[term]*.4
    p.save()
    context=snapshot(path)
    owner=next(hist for hist in context['histograms'] if hist['name']==h.name)
    plan=dict(schema='radar-gpx-recipe/v1',model_signature=context['model_signature'],
        request='Refine one explicit TOF width coefficient only',assumptions=[],questions=[],unsupported=[],
        stages=[dict(id='profile',title='Width',cycles=20,constraints=[],free=[dict(
            kind='instrument_profile',histogram=owner['key'],parameter=term)])])
    result=run_plan(path,tmp_path/'refined.gpx',plan,approved=dict(
        approved_plan_sha256=plan_hash(plan),model_signature=context['model_signature']))
    stage=result['stages'][0]
    assert stage['status']=='completed'
    assert stage['tof_profile_audit'][0]['coefficients'][term]==pytest.approx(truth_values[term],rel=.01,abs=.02)
    assert stage['parameter_changes']['unexpected_fixed_changes']==[]
    assert stage['actual_vary']==[f':{owner["native_id"]}:'+term]
    if other_bank:
        result_project=G.G2Project(gpxfile=str(tmp_path/'refined.gpx'))
        assert result_project.histogram(held.name).InstrumentParameters[term][1]==truth_values[term]
