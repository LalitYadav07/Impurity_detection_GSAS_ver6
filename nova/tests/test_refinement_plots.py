import copy
import math
import pytest
from radar_pd_nova.refinement_plots import fit_figure,trend_figure


def view():
    return dict(histograms=[dict(name='bank2',coordinate='tof',x=[100.,200.,300.],d=[1.,2.,3.],observed=[1.,20.,1.],weights=[1.,1.,1.],weight_factor=1.,valid=[True,True,True],calculated=[1.,18.,1.],fitted=[True,True,False],limits=[100.,200.],exclusions=[],rwp=5.,instrument_parameters={},ticks=[dict(phase='oxide',d=[2.],native=[200.])])],parameters=[dict(key='cell|oxide|c',label='oxide c',kind='cell',parameter='c',value=4.,sigma=.001,unit='Å')])


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


def test_native_mask_reset_cannot_turn_an_invalid_zero_weight_bin_into_a_measurement():
    from gpx_recipe.fit_domain import fit_domain_digest
    from gpx_recipe.worker import trend_view
    from radar_pd_nova.refinement_plots import same_fit_domain
    before=view();after=view()
    for record in (before,after):
        record['histograms'][0]['valid'][2]=False
        record['histograms'][0]['weights'][2]=0.
    before['histograms'][0]['observed'][2]=None
    after['histograms'][0]['observed'][2]=0.
    original=copy.deepcopy((before,after))
    figure,notice=fit_figure(before,after)
    assert math.isnan(float(figure.data[0].y[2]))
    assert notice['rwp_comparable']
    assert fit_domain_digest(before['histograms'][0])==fit_domain_digest(after['histograms'][0])
    assert same_fit_domain(trend_view(before)['histograms'][0],trend_view(after)['histograms'][0])
    assert (before,after)==original


@pytest.mark.parametrize('field,value',[('weights',1.),('valid',True),('fitted',True)])
def test_placeholder_equivalence_never_covers_usable_or_fitted_points(field,value):
    before=view();after=view()
    for record in (before,after):
        h=record['histograms'][0]
        h['valid'][2]=False;h['weights'][2]=0.
        h[field][2]=value
    before['histograms'][0]['observed'][2]=None
    after['histograms'][0]['observed'][2]=0.
    with pytest.raises(ValueError,match='same measured'):fit_figure(before,after)


def test_compact_domains_preserve_old_contracts_without_mixing_versions():
    from gpx_recipe.fit_domain import DOMAIN_SCHEMA
    from radar_pd_nova.refinement_plots import same_fit_domain
    old={'fit_domain_schema':'weighted-observations/v2','fit_domain_sha256':'old'}
    assert same_fit_domain(old,old)
    assert not same_fit_domain(old,dict(old,fit_domain_schema=DOMAIN_SCHEMA))


@pytest.mark.parametrize('field,value',[('weights',[1.,2.,1.]),('weight_factor',2.),('valid',[True,False,True]),('observed',[2.,20.,1.])])
def test_compact_comparisons_detect_measurement_and_uncertainty_changes(field,value):
    from gpx_recipe.worker import trend_view
    from radar_pd_nova.refinement_plots import same_fit_domain
    before=view();after=view();after['histograms'][0][field]=value
    assert not same_fit_domain(trend_view(before)['histograms'][0],trend_view(after)['histograms'][0])


def test_legacy_exports_cannot_certify_weighted_comparability():
    from radar_pd_nova.refinement_plots import same_fit_domain
    old=view()['histograms'][0];del old['weights']
    assert not same_fit_domain(old,old)
    assert not same_fit_domain({'fit_domain_sha256':'old'},{'fit_domain_sha256':'old'})
    assert not fit_figure({'histograms':[old]},view())[1]['rwp_comparable']


def test_changed_calculation_is_comparable_but_changed_weights_are_not():
    after=view();after['histograms'][0]['calculated']=[1.,19.,1.]
    assert fit_figure(view(),after)[1]['rwp_comparable']
    after['histograms'][0]['weights'][1]=2.
    assert not fit_figure(view(),after)[1]['rwp_comparable']


def test_uncalculated_starting_model_is_not_drawn_as_a_zero_fit():
    starting=view();starting['histograms'][0].update(calculated=[0.,0.,0.],rwp=None,
        fit_statistics={'calculation_status':'not_yet_evaluated'})
    figure,notice=fit_figure(starting,view())
    assert not any('Original RADAR-PD' in trace.name for trace in figure.data)
    assert 'has not been calculated' in notice['notice']
    assert 'two Rwp values' not in notice['notice']


def test_unevaluated_legacy_baseline_does_not_prevent_weighted_selected_residual():
    starting=view();starting['histograms'][0].pop('weights')
    starting['histograms'][0]['fit_statistics']={'calculation_status':'not_yet_evaluated'}
    _,notice=fit_figure(starting,view())
    assert 'Residuals use sqrt' in notice['notice']


def test_residual_plot_uses_recorded_weights_and_histogram_weight_factor():
    result=view();h=result['histograms'][0];h['weights']=[1.,4.,1.];h['weight_factor']=4.
    h['fit_statistics']={'weighted_residual_sum':64.,'point_count':2}
    figure,notice=fit_figure(result,result,display='polished')
    residual=next(trace for trace in figure.data if trace.name=='Selected refinement residual')
    assert list(residual.y[:2])==pytest.approx([0.,8.])
    assert 'per fitted point: 32' in notice['notice']


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


@pytest.mark.parametrize('metric', ['rwp', 'phase_fractions', 'cell|oxide|c'])
def test_temperature_ranges_follow_each_scan_and_are_distinct_from_parameter_errors(metric):
    records = rows()
    for row in records:
        row['final']['parameters'].append(dict(key='weight|oxide', kind='weight', phase='oxide', value=100.))
    records[0]['source']['metadata']['temperature'] = dict(value=100., minimum=99., maximum=103., unit='K')
    records[1]['source']['metadata']['temperature'] = dict(value=20., minimum=18., maximum=23., unit='C')
    original = copy.deepcopy(records)
    figure = trend_figure(records, metric=metric, x_key='temperature')
    for trace in figure.data:
        assert list(trace.x) == pytest.approx([100., 293.15])
        assert list(trace.error_x.array) == pytest.approx([3., 3.])
        assert list(trace.error_x.arrayminus) == pytest.approx([1., 2.])
        assert trace.error_x.symmetric is False and trace.error_x.visible
    assert figure.layout.meta['temperature_spans']
    assert records == original
    hidden = trend_figure(records, metric=metric, x_key='temperature', show_temperature_span=False)
    assert all(not trace.error_x.visible for trace in hidden.data)
    assert not hidden.layout.meta['temperature_spans']
    run_axis = trend_figure(records, metric=metric)
    assert all(not trace.error_x.visible for trace in run_axis.data)


def test_display_mapping_does_not_borrow_an_unmapped_temperature_range():
    records = rows()
    records[0]['source']['metadata']['temperature'] = dict(value=100., minimum=95., maximum=105., unit='K')
    records[0]['source']['axis_overrides'] = {'temperature': dict(value=20., unit='C', reviewed_mapping=True)}
    records[1]['source']['metadata']['temperature'] = dict(value=100., minimum=101., maximum=103., unit='K')
    figure = trend_figure(records, metric='cell|oxide|c', x_key='temperature')
    assert list(figure.data[0].x) == pytest.approx([293.15, 100.])
    assert not figure.data[0].error_x.visible
    assert list(figure.data[0].error_x.array) == [None, None]


def test_voltage_trend_uses_only_same_cell_evidence_and_keeps_coverage_visible():
    from radar_pd_nova.refinement_plots import refinement_axis_options
    from radar_pd_nova.refinement_controller import refinement_exposure_notice
    records=rows()
    for record in records:
        record['source']['metadata']=dict(sample_id='sample',cell_id='cell',
            electrochemistry=dict(schema='radar-battery-model-electrochemistry/v1',
                exposure=dict(sample_id='sample',cell_id='cell',voltage_v={'mean':3.2},
                    coverage_fraction=.5,current_a=None,clock_uncertainty_seconds=None)))
    records[1]['source']['metadata']['cell_id']='another cell'
    fig=trend_figure(records,metric='cell|oxide|c',x_key='battery_voltage')
    assert list(fig.data[0].x)==[3.2,None]
    assert fig.data[0].mode=='markers'
    assert 'battery_voltage' in {r['value'] for r in refinement_axis_options([records[0]['source']])}
    assert 'battery_voltage' not in {r['value'] for r in refinement_axis_options([records[1]['source']])}
    assert '50.0% voltage coverage' in refinement_exposure_notice(records[0]['source'])
    assert 'current unavailable' in refinement_exposure_notice(records[0]['source'])


def test_reviewed_temperature_mapping_overrides_conflicting_raw_unit_without_changing_raw_metadata():
    records=rows()
    records[0]['source']['metadata']['temperature']={'value':114,'unit':'K','status':'recorded'}
    records[0]['source']['axis_overrides']={'temperature':{'value':114,'unit':'C','reviewed_mapping':True}}
    records[1]['source']['axis_overrides']={'temperature':{'value':153,'unit':'C','reviewed_mapping':True}}
    fig=trend_figure(records,metric='cell|oxide|c',x_key='temperature')
    assert list(fig.data[0].x)==pytest.approx([387.15,426.15])
    assert records[0]['source']['metadata']['temperature']['unit']=='K'


def test_relative_occupancy_is_not_mislabeled_as_cell_expansion():
    records=rows()
    for row in records:row['final']['parameters'][0].update(key='atom|oxide|O1|occupancy',kind='atom',parameter='occupancy')
    with pytest.raises(ValueError,match='Relative percent'):trend_figure(records,metric='atom|oxide|O1|occupancy',display='relative')


def test_fixed_occupancy_trend_is_explicitly_not_a_measurement():
    records=rows()
    for row in records:
        row['final']['parameters'][0].update(key='atom|oxide|Li1|occupancy',kind='atom',parameter='occupancy',
            value=.95,sigma=None,status='fixed_by_model')
    figure=trend_figure(records,metric='atom|oxide|Li1|occupancy')
    assert any('not independent measurements' in item.text for item in figure.layout.annotations)


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


def test_unevaluated_baselines_are_not_empty_trend_legend_entries_or_comparisons():
    records=rows()
    for row in records:
        row['baseline']['histograms'][0]['rwp']=None
        row['baseline']['histograms'][0].pop('weights')
    figure=trend_figure(records)
    assert len(figure.data)==1 and 'polished' in figure.data[0].name
    assert not figure.layout.annotations


def test_bank_specific_sample_widths_have_distinct_parameter_labels():
    from radar_pd_nova.refinement_plots import refinement_parameter_label
    source={'metadata':{'physical_banks':{'native1':2,'native2':5}}}
    p=dict(label='Graphite · sample size',histogram='native1')
    assert refinement_parameter_label(p,source)=='Graphite · sample size · physical bank 2'
    p['histogram']='native2'
    assert refinement_parameter_label(p,source)=='Graphite · sample size · physical bank 5'


def test_phase_fraction_overview_has_separate_curves_for_all_modeled_phases():
    records=rows()
    for row in records:
        row['final']['parameters']=[dict(key='weight|A|h0',phase='A',kind='weight',value=30.),dict(key='weight|B|h0',phase='B',kind='weight',value=70.)]
    fig=trend_figure(records,metric='phase_fractions')
    assert len(fig.data)==2 and list(fig.data[0].y)==[30.,30.] and list(fig.data[1].y)==[70.,70.]


def test_related_phase_names_are_not_collapsed_in_fit_legends():
    result=view()
    result['histograms'][0]['ticks']=[dict(phase=name,d=[2.],native=[200.]) for name in ['Host - state A','Host - state B']]
    fig,_=fit_figure(result,result)
    assert [t.name for t in fig.data[-2:]]==['Host - state A','Host - state B']


def test_multibank_phase_fractions_and_rwp_have_unambiguous_labels():
    records=rows()
    for row in records:
        row['source']['bank']=''
        row['source']['metadata']={'physical_banks':{'bank2':2,'bank5':5}}
        for variant in ['baseline','final']:
            first=row[variant]['histograms'][0];first['key']='h0'
            row[variant]['histograms'].append(dict(first,key='h1',name='bank5'))
        row['final']['parameters']=[dict(key=f'weight|{phase}|{key}',histogram_key=key,phase=phase,kind='weight',value=50.)
                                   for phase in ['Host - state A','Host - state B'] for key in ['h0','h1']]
    labels=[t.name for t in trend_figure(records,metric='phase_fractions').data]
    assert len(set(labels))==4
    assert any('Host - state A · physical bank 2' in s for s in labels)
    assert any('Host - state B · physical bank 5' in s for s in labels)
    assert all('physical bank ' in t.name for t in trend_figure(records).data)
