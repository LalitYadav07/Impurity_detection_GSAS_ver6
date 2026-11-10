"""The reusable width portfolio keeps real native inputs and calibration fixed."""
import numpy as np
import pytest

G = pytest.importorskip('GSASII.GSASIIscriptable')
from GSASII import GSASIIstrMain as MAIN
from gpx_recipe.execution import freeze
from gpx_recipe.live.contracts import file_hash
from gpx_recipe.live.sample_profile_trials import TRIALS, sample_width_trials


def source(root):
    instrument = root/'profile.instprm'
    instrument.write_text('#GSAS-II instrument parameter file\nType:PNT\nBank:1\n'
        'difC:6200\ndifA:2\ndifB:.4\nZero:1\n2-theta:105\nfltPath:15.5\n'
        'alpha:.99\nbeta-0:.02\nbeta-1:.03\nbeta-q:0\n'
        'sig-0:1\nsig-1:66\nsig-2:0\nsig-q:0\nX:0\nY:0\nZ:0\nAzimuth:0\n')
    path = root/'source.gpx'
    g = G.G2Project(newgpx=str(path))
    p = g.add_phase(phasename='Public synthetic ErSi', spacegroup='C m c m',
        cell=[4.195,10.353,3.779,90.,90.,90.])
    p.add_atom(0.,.413,.25,'Si','Si1');p.add_atom(0.,.142,.25,'Er','Er1')
    h = g.add_simulated_powder_histogram('public control',str(instrument),
        3.,40.,Npoints=701,scale=120.,phases=[p])
    h.data['Background'][0][2]=3;h.data['Background'][0][3:]=[3.,.2,.1]
    freeze(g);h.data['data'][1][1][:]=1.;g.save()
    calculated=np.asarray(MAIN.DoNoFit(str(path),h.name),float)
    h.data['data'][1][1]=calculated+.03*np.sin(np.arange(len(calculated)))
    h.data['data'][1][2][:]=4.
    p.data['Histograms'][h.name]['Scale'][0]*=.9
    g.save()
    return path


def test_real_width_trials_preserve_source_and_exact_native_observations(tmp_path):
    path=source(tmp_path);sha=file_hash(path);original=G.G2Project(gpxfile=str(path))
    report=sample_width_trials(path,tmp_path/'trials')
    assert report['source_sha256']==sha and file_hash(path)==sha
    assert report['scientifically_qualified'] is False
    assert [r['name'] for r in report['trials']]==[r[0] for r in TRIALS]
    assert report['selected_for_review'] is not None
    successful=[r for r in report['trials'] if 'error' not in r]
    assert successful and all(not r['scientifically_accepted'] for r in successful)
    for row in successful:
        result=G.G2Project(gpxfile=row['path'])
        h=result.histogram(0);before=original.histogram(0)
        assert h.data['Instrument Parameters']==before.data['Instrument Parameters']
        for index in (0,1,2):
            np.testing.assert_array_equal(h.data['data'][1][index],before.data['data'][1][index])
        assert row['observations_and_masks_preserved'] and np.isfinite(row['rwp'])
    with pytest.raises(ValueError,match='fresh output GPX'):
        sample_width_trials(path,tmp_path/'trials')
