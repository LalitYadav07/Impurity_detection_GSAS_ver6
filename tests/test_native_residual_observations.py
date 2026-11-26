"""Already-native residuals must not re-enter GSAS's TOF edge conversion."""
import re

import numpy as np
import pytest

from scripts.gsas_native_observations import restore_native_observations
from gpx_recipe.fit_domain import native_powder_fit_mask


def instrument(tmp_path):
    path = tmp_path/'profile.instprm'
    path.write_text('#GSAS-II instrument parameter file\nType:PNT\nBank:1\n'
        'difC:6200\ndifA:2\ndifB:.4\nZero:1\n2-theta:105\nfltPath:15.5\nalpha:.99\n'
        'beta-0:.02\nbeta-1:.03\nbeta-q:0\nsig-0:1\nsig-1:66\nsig-2:0\nsig-q:0\nX:0\nY:0\nZ:0\nAzimuth:0\n')
    return path


def native_histogram(tmp_path, x, y):
    gsas = pytest.importorskip('GSASII.GSASIIscriptable')
    path = tmp_path/'diagnostic.xye'
    np.savetxt(path, np.column_stack((x, y, np.ones(len(x)))), fmt='%.17g')
    project = gsas.G2Project(newgpx=str(tmp_path/'native.gpx'))
    phase = project.add_phase(phasename='Synthetic Si', spacegroup='P 1', cell=[4,4,4,90,90,90])
    phase.add_atom(0,0,0,'Si','Si1')
    hist = project.add_powder_histogram(str(path), str(instrument(tmp_path)), fmthint='Topas', phases=[phase])
    return project, hist


@pytest.mark.parametrize('spacing', ['irregular', 'logarithmic'])
def test_native_roundtrip_preserves_every_center_intensity_weight_and_endpoint(tmp_path, spacing):
    from scripts.plot_payload import histogram_plot_arrays
    gsas = pytest.importorskip('GSASII.GSASIIscriptable')
    x = (2000.123456789+np.cumsum(np.linspace(17.123456789,49.987654321,240))
         if spacing == 'irregular' else np.geomspace(2000.123456789,16000.987654321,240))
    y = 20+np.sin(x/10)
    weights = np.linspace(.31,7.1,len(x))
    project, hist = native_histogram(tmp_path,x,y)
    # Direct proof of why the native-center adapter is required.
    imported = hist.data['data'][1][0]
    assert len(imported) == len(x)-1
    np.testing.assert_allclose(imported, (x[:-1]+x[1:])/2, rtol=0, atol=1e-10)
    originals = [v.copy() for v in (x,y,weights)]
    restore_native_observations(hist,x,y,weights=weights)
    actual = histogram_plot_arrays(hist)
    for key, expected in [('x',x),('yobs',y),('weights',weights)]:
        np.testing.assert_array_equal(actual[key],expected)
    assert actual['fit_valid'].all()
    assert hist.data['Limits'] == [[x[0],x[-1]],[x[0],x[-1]]]
    project.save()
    restored = gsas.G2Project(gpxfile=str(tmp_path/'native.gpx'))
    actual = histogram_plot_arrays(restored.histograms()[0])
    for key, expected in [('x',x),('yobs',y),('weights',weights)]:
        np.testing.assert_array_equal(actual[key],expected)
    assert actual['fit_valid'].all()
    for original, value in zip(originals,(x,y,weights)):
        np.testing.assert_array_equal(original,value)


@pytest.mark.parametrize('gap', ['zero_weight', 'masked_y', 'masked_x', 'masked_weight'])
@pytest.mark.parametrize('gap_width',[1,2])
def test_native_refinement_counts_single_bin_gaps_and_exact_limit_boundaries(tmp_path,gap,gap_width):
    from scripts.plot_payload import histogram_plot_arrays
    x = 2000.123456789+np.cumsum(np.linspace(17.3,49.7,240))
    y = 20+np.sin(x/10)
    weights = np.linspace(.5,4.,len(x))
    project, hist = native_histogram(tmp_path,x,y)
    supplied = [x.copy(), y.copy(), weights.copy()]
    channel = {'zero_weight':2,'masked_y':1,'masked_x':0,'masked_weight':2}[gap]
    if gap == 'zero_weight':supplied[channel][72:72+gap_width]=0
    else:
        supplied[channel]=np.ma.array(supplied[channel],mask=False)
        supplied[channel].mask[72:72+gap_width]=True
    restore_native_observations(hist,*supplied[:2],weights=supplied[2])
    # Include a between-bin upper edge and a separate one-bin excluded region.
    limits=[float((x[10]+x[11])/2),float((x[210]+x[211])/2)]
    excluded=[[float(x[130]),float(x[130])]]
    hist.data['Limits'][1]=limits
    hist.data['Limits'].extend(excluded)
    before=histogram_plot_arrays(hist)
    expected=native_powder_fit_mask(x,limits,excluded)
    expected[72:72+gap_width]=False
    np.testing.assert_array_equal(before['fit_valid'],expected)
    assert not before['fit_valid'][130] and before['fit_valid'][211]
    project.data['Controls']['data']['max cyc']=0
    project.do_refinements([{'set':{}}])
    after=histogram_plot_arrays(hist)
    np.testing.assert_array_equal(after['x'],x)
    np.testing.assert_array_equal(after['fit_valid'],expected)
    listing=(tmp_path/'native.lst').read_text()
    counts=re.findall(r'No\. of observations: (\d+)',listing)
    assert counts and int(counts[-1]) == int(expected.sum())


def test_zero_negative_and_masked_values_are_not_reinterpreted_as_absent_counts():
    from types import SimpleNamespace
    x=np.arange(12.)+1
    y=np.ma.array([0.,-2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.],mask=False)
    y.mask[3]=True
    weights=np.linspace(.25,4.,12);weights[5]=0
    hist=SimpleNamespace(data={'data':[{},[]]})
    restore_native_observations(hist,x,y,weights=weights)
    actual=hist.data['data'][1]
    np.testing.assert_array_equal(actual[0],x)
    np.testing.assert_array_equal(np.ma.getdata(actual[1]),np.ma.getdata(y))
    np.testing.assert_array_equal(np.ma.getmaskarray(actual[1]),np.ma.getmaskarray(y))
    np.testing.assert_array_equal(np.ma.getdata(actual[2]),weights)
    assert np.ma.getmaskarray(actual[2])[3] and np.ma.getmaskarray(actual[2])[5]


@pytest.mark.parametrize('cached_template',[False,True])
def test_actual_pearson_path_restores_fresh_and_cached_tof_observations(tmp_path,monkeypatch,cached_template):
    from scripts import gsas_main_phase_refiner as refiner
    from scripts.plot_payload import histogram_plot_arrays
    gsas=pytest.importorskip('GSASII.GSASIIscriptable')
    x=np.geomspace(2200.123456789,15000.987654321,247)
    y=3+np.sin(np.arange(len(x))/13)
    profile=instrument(tmp_path)
    cif=tmp_path/'silicon.cif'
    cif.write_text('''data_synthetic
_cell_length_a 4
_cell_length_b 4
_cell_length_c 4
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'P 1'
_space_group_IT_number 1
loop_
_space_group_symop_operation_xyz
'x,y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_U_iso_or_equiv
Si1 Si 0 0 0 1 0.005
''')
    template=None
    if cached_template:
        # Deliberately give the real call a pre-fix 246-point template.
        source=tmp_path/'cached.xye'
        np.savetxt(source,np.column_stack((x,y,np.ones(len(x)))),fmt='%.17g')
        p=gsas.G2Project(newgpx=str(tmp_path/'cached.gpx'))
        h=p.add_powder_histogram(str(source),str(profile),fmthint='Topas')
        assert len(h.data['data'][1][0])==246
        p.save();template=str(tmp_path/'cached.gpx')
    original=refiner.GSASDataExtractor.get_fitted_arrays
    seen=[]
    def inspect(hist):
        arrays=histogram_plot_arrays(hist)
        np.testing.assert_array_equal(arrays['x'],x)
        np.testing.assert_array_equal(arrays['yobs'],y)
        np.testing.assert_array_equal(arrays['weights'],np.ones(len(x)))
        assert arrays['fit_valid'].all()
        seen.append(True)
        return original(hist)
    monkeypatch.setattr(refiner.GSASDataExtractor,'get_fitted_arrays',inspect)
    result=refiner.compute_gsas_pearson_for_cif('',str(profile),'xye',str(cif),str(tmp_path/'screen'),
        None,None,'native',x_override=x,y_override=y,template_gpx=template,
        cell_refine_min_r=2,export_refined_cif=False,return_diagnostics=True)
    assert seen and result['fit_point_counts']=={'pass1-scale':len(x)}


@pytest.mark.parametrize('bad', ['length','duplicate_x','nan_x','negative_weight','nan_active_y'])
def test_invalid_native_contract_fails_before_histogram_mutation(bad):
    from types import SimpleNamespace
    x=np.arange(12.)+1;y=np.ones(12);weights=np.ones(12)
    if bad=='length':y=y[:-1]
    if bad=='duplicate_x':x[5]=x[4]
    if bad=='nan_x':x[5]=np.nan
    if bad=='negative_weight':weights[5]=-1
    if bad=='nan_active_y':y[5]=np.nan
    hist=SimpleNamespace(data={'data':[{},['unchanged']]})
    with pytest.raises(ValueError):restore_native_observations(hist,x,y,weights=weights)
    assert hist.data['data'][1] == ['unchanged']
