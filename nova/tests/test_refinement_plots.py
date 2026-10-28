import copy
import math
import pytest
from radar_pd_nova.refinement_plots import fit_figure,trend_figure


def view():
    return dict(histograms=[dict(name='bank2',coordinate='tof',x=[100.,200.,300.],d=[1.,2.,3.],observed=[1.,20.,1.],calculated=[1.,18.,1.],fitted=[True,True,False],limits=[100.,200.],exclusions=[],rwp=5.,instrument_parameters={},ticks=[dict(phase='oxide',d=[2.],native=[200.])])],parameters=[dict(key='cell|oxide|c',label='oxide c',kind='cell',parameter='c',value=4.,sigma=.001,unit='Å')])


def test_fit_preserves_excluded_gaps_and_uses_native_bragg_positions_in_each_coordinate():
    for axis,position in [('native',200.),('d',2.),('Q',math.pi)]:
        fig,meta=fit_figure(view(),view(),axis=axis)
        assert list(fig.data[-1].x)==pytest.approx([position])
        assert any(math.isnan(float(v)) for v in fig.data[1].y)
        assert meta['rwp_comparable']
    p=view();p['histograms'][0]['exclusions']=[[150.,170.]]
    assert not fit_figure(view(),p)[1]['rwp_comparable']


def test_comparison_refuses_different_observed_data():
    modified=view();modified['histograms'][0]['observed'][0]=50.
    with pytest.raises(ValueError,match='same measured'):fit_figure(view(),modified)


def rows():
    values=[]
    for i in range(2):
        final=view();final['parameters'][0]['value']=4.+i*.04
        values.append(dict(id=str(i),recipe_id='recipe',source=dict(sample_key='a',sample_label='a',bank='2',run_number=i+1,metadata={}),baseline=view(),final=final))
    return values


def test_relative_cell_trend_is_relative_to_same_sample_bank_and_recipe():
    records=rows();fig=trend_figure(records,metric='cell|oxide|c',display='relative')
    assert list(fig.data[0].y)==pytest.approx([0.,1.])
    other=copy.deepcopy(records[1]);other['id']='other';other['source']['sample_key']='b';other['source']['sample_label']='b';other['final']['parameters'][0]['value']=20.
    fig=trend_figure(records+[other],metric='cell|oxide|c',display='relative')
    assert list(fig.data[1].y)==[0.]


def test_missing_acquisition_temperature_is_not_replaced_by_gpx_default_or_run_number():
    fig=trend_figure(rows(),metric='cell|oxide|c',x_key='temperature')
    assert list(fig.data[0].x)==[None,None]


def test_relative_occupancy_is_not_mislabeled_as_cell_expansion():
    records=rows()
    for row in records:row['final']['parameters'][0].update(key='atom|oxide|O1|occupancy',kind='atom',parameter='occupancy')
    with pytest.raises(ValueError,match='Relative percent'):trend_figure(records,metric='atom|oxide|O1|occupancy',display='relative')


def test_compact_trends_retain_parameters_and_domain_identity_without_pattern_arrays():
    from gpx_recipe.worker import trend_view
    full=view();compact=trend_view(full)
    assert compact['parameters']==full['parameters']
    assert not {'x','observed','calculated','fitted'}&set(compact['histograms'][0])
    altered=view();altered['histograms'][0]['exclusions']=[[150,160]]
    assert trend_view(altered)['histograms'][0]['fit_domain_sha256']!=compact['histograms'][0]['fit_domain_sha256']
    records=rows()
    for row in records:row.update(baseline=compact,final=trend_view(altered))
    fig=trend_figure(records)
    assert any('not directly comparable' in a.text for a in fig.layout.annotations)


def test_scan_specific_histogram_filenames_do_not_break_rwp_trends():
    records=rows()
    records[1]['baseline']['histograms'][0]['name']='PWDR another_scan.gsa'
    records[1]['final']['histograms'][0]['name']='PWDR another_scan.gsa'
    fig=trend_figure(records)
    assert all(list(trace.y)==[5.,5.] for trace in fig.data)


def test_phase_fraction_overview_has_separate_curves_for_all_modeled_phases():
    records=rows()
    for row in records:
        row['final']['parameters']=[dict(key='weight|A|h0',phase='A',kind='weight',value=30.),dict(key='weight|B|h0',phase='B',kind='weight',value=70.)]
    fig=trend_figure(records,metric='phase_fractions')
    assert len(fig.data)==2 and list(fig.data[0].y)==[30.,30.] and list(fig.data[1].y)==[70.,70.]
