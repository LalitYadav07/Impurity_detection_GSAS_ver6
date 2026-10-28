"""Failures found while auditing the deployed NDIP user workflows."""
import copy

import pytest

from radar_pd_nova.post_analysis import PostAnalysisSettings, ViewSettings
from radar_pd_nova.post_analysis import AxisMapping
from radar_pd_nova.post_analysis_plot import view_figure
from radar_pd_nova.results import (
    _experiment_axis,
    experiment_axis_options,
    experiment_fit_diagnostics,
)
from test_post_analysis import scan
from test_post_analysis_ui import app_with_data


def test_reviewed_run_mapping_reaches_main_charts_without_rewriting_acquisition_metadata():
    from types import SimpleNamespace
    app=app_with_data()
    source=scan(1,temperature=48.1463)
    original=copy.deepcopy(source['metadata'])
    run=SimpleNamespace(run_id='PG3_1',run_number=1,bank=1,instrument='PG3',source_path='/SNS/PG3/IPTS-1/shared/autoreduce/PG3_1.gsa',
        galaxy_result_ids=[],error='',galaxy_job_id='',scan_metadata=source['metadata'],
        scientific_summary={'rwp':3.,'final_rwp':3.,'analysis_mode':'full',
            'observables':source['observables'],'lattice_parameters':source['observables']['phases'],
            'phases':[{'phase':'phase-a','weight_percent':100.,'space_group':'P1'}]})
    post=app._post_settings()
    post.axis_mappings={'temperature':{'PG3_1':AxisMapping(value=47.3,unit='C',source='Reviewed campaign units',kind='user_supplied',verified=True)}}
    app._powgen_controller=SimpleNamespace(
        state=SimpleNamespace(completed={'PG3_1':run},submitted={},discovered={},failed={},post_analysis=post.model_dump()),
        settings=SimpleNamespace(max_active_jobs=5,instrument='PG3',ipts='IPTS-1'),records={},
        source_directory='/SNS/PG3/IPTS-1/shared/autoreduce')
    app._post_analysis_scope='watch:/SNS/PG3/IPTS-1/shared/autoreduce'
    app._set_post_settings(post)
    app._sync_powgen_rows()
    row=app.server.state.powgen_scientific_rows[0]
    assert row['temperature_display']=='320.45 K'
    assert 'Reviewed run mapping' in row['temperature_provenance']
    assert row['metadata']==original
    assert _experiment_axis([row],'temperature')[0]==pytest.approx([320.45])
    exports=[]
    app.download_file=lambda *args:exports.append(args)
    app.download_powgen_lattice()
    assert '320.45 K' in exports[0][2].decode()


def test_mapping_import_and_clear_refresh_the_main_watch_dashboard(tmp_path):
    from types import SimpleNamespace
    from radar_pd_nova.powgen_watch import WatchState
    app=app_with_data()
    app._post_analysis_scope='watch:test'
    app._powgen_controller=SimpleNamespace(state=WatchState(history_id='test'))
    refreshes=[]
    app._sync_powgen_rows=lambda:refreshes.append(True)
    path=tmp_path/'mapping.csv'
    path.write_text('axis,run_id,value,unit,source,kind,verified\ntemperature,NOM_1_BANK1,47.3,C,Reviewed campaign,user_supplied,true\n')
    app.server.state.post_analysis_mapping_path=str(path)
    app.import_post_analysis_mapping()
    assert not app.server.state.post_analysis_error
    assert len(refreshes)==1
    app.clear_post_analysis_mapping()
    assert len(refreshes)==2


def test_reviewed_mapping_updates_transition_coordinates_without_changing_pattern_states():
    from test_transition_tracking_ui import scans as transition_scans
    app=app_with_data()
    rows=transition_scans()
    app._refresh_transition_tracking(rows)
    before=[(r['run_id'],r.get('symbol'),r.get('boundary')) for r in app._transition_report['records']]
    for i,row in enumerate(rows):
        row['axis_overrides']={'temperature':dict(value=320.+i,unit='K',source='Reviewed run mapping',
            kind='user_supplied',reviewed_mapping=True)}
    app._refresh_transition_tracking(rows)
    after=[(r['run_id'],r.get('symbol'),r.get('boundary')) for r in app._transition_report['records']]
    assert after==before
    app.update_transition_diagram_axis('temperature')
    plotted=[t['x'][0] for t in app.server.state.transition_boundary_map['data']]
    assert plotted==pytest.approx([320.+i for i in range(len(rows))])
    assert 'Reviewed run mapping' in app.server.state.transition_condition_summary


@pytest.mark.parametrize('axis', ['proton_charge', 'wavelength'])
def test_main_dashboard_axes_cannot_break_post_analysis_configuration(axis):
    app=app_with_data()
    app.server.state.powgen_x_axis_options=[{'title':axis,'value':axis}]
    app.update_powgen_dashboard_axis(axis)
    assert app.server.state.powgen_x_axis==axis
    # A main-chart axis without a registered derived-quantity recipe must
    # leave the separately saved post-analysis axis valid and unchanged.
    assert app._post_settings().axis=='auto'
    assert not app.server.state.post_analysis_error


def test_equivalent_celsius_and_kelvin_conditions_have_the_same_plot_coordinate():
    scans=[{'run_number':1,'metadata':{'temperature':{'value':300.,'unit':'K'}}},
        {'run_number':2,'metadata':{'temperature':{'value':26.85,'unit':'C'}}}]
    values,title,_=_experiment_axis(scans,'temperature')
    assert values==pytest.approx([300.,300.])
    assert title=='Sample temperature (K)'
    assert next(o for o in experiment_axis_options(scans) if o['value']=='temperature')['title']==title


def test_unitless_temperature_does_not_enter_a_kelvin_series():
    scans=[{'run_number':1,'metadata':{'temperature':{'value':300.,'unit':'K'}}},
        {'run_number':2,'metadata':{'temperature':{'value':25.}}}]
    values,title,_=_experiment_axis(scans,'temperature')
    assert values==[300.,None]
    assert title=='Sample temperature (K)'


@pytest.mark.parametrize('field,value',[
    ('profile_sha256','different-profile'),('recipe_key','different-recipe'),
    ('reduction_key','different-reduction'),('configuration_sha256','different-config'),
])
def test_rwp_baseline_excludes_a_different_final_model_context(field,value):
    rows=[dict(scan(i),rwp=2.) for i in range(1,5)]
    changed=dict(scan(5),rwp=20.)
    changed['observables']['context'][field]=value
    if field=='configuration_sha256':
        for row in rows:row['observables']['context'][field]='original-config'
    diagnostics=experiment_fit_diagnostics([*rows,changed])
    assert diagnostics[changed['run_id']]['label']=='Collecting baseline'
    assert diagnostics[changed['run_id']]['baseline_median']==20.


def test_rwp_baseline_excludes_changed_hb2a_setup():
    rows=[dict(scan(i),rwp=2.) for i in range(1,5)]
    for row in rows:row['metadata']['hb2a_setup']={'cycle':'506','normalization':'vanadium'}
    changed=dict(scan(5),rwp=20.)
    changed['metadata']['hb2a_setup']={'cycle':'508','normalization':'monitor'}
    diagnostics=experiment_fit_diagnostics([*rows,changed])
    assert diagnostics[changed['run_id']]['label']=='Collecting baseline'


def mixed_display_app():
    app=app_with_data()
    app._set_post_settings(PostAnalysisSettings(axis='temperature',views=[
        ViewSettings(id='kelvin',kind='thermal_expansion',temperature_display_unit='K'),
        ViewSettings(id='celsius',kind='thermal_expansion',temperature_display_unit='C'),
    ]))
    app._refresh_post_analysis([scan(1,temperature=300),scan(2,temperature=350,a=4.01)])
    return app


def test_linked_temperature_zoom_translates_between_display_units():
    app=mixed_display_app()
    app.relayout_post_analysis('celsius',{'xaxis.range':[20.,80.]})
    by_id={v.id:v for v in app._post_settings().views}
    assert by_id['celsius'].x_range==(20.,80.)
    assert by_id['kelvin'].x_range==pytest.approx((293.15,353.15))
    app.relayout_post_analysis('kelvin',{'xaxis.range':[300.,330.]})
    by_id={v.id:v for v in app._post_settings().views}
    assert by_id['celsius'].x_range==pytest.approx((26.85,56.85))
    app.relayout_post_analysis('kelvin',{'xaxis.autorange':True})
    assert all(v.x_range is None for v in app._post_settings().views)


def test_observation_table_and_inspector_use_the_plotted_temperature_unit():
    app=mixed_display_app()
    panel=next(v for v in app._post_analysis_result['views'] if v['id']=='celsius')
    row=app._post_observations(panel)[0]
    assert row['x']=='26.85 °C'
    setting=next(v for v in app._post_settings().views if v.id=='celsius')
    custom=view_figure(panel,setting).data[0].customdata[0]
    app.select_post_analysis_point({'points':[{'customdata':custom}]},'celsius')
    details={d['label']:d['value'] for d in app.server.state.post_analysis_point_details}
    assert float(details['Horizontal coordinate'])==pytest.approx(26.85)
    assert details['Axis unit']=='°C'
