"""Scientific review flags stay separate from completed transport jobs."""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import zipfile

import pytest

from gpx_recipe.convergence import SCHEMA
from radar_pd_nova.refinement_controller import RefinementMixin
from radar_pd_nova.refinement_messages import refinement_review_status,retain_review_evidence
from radar_pd_nova.refinement_plots import trend_figure
from radar_pd_nova.refinement_runtime import retain_result_review,verify_result_files,LocalRefinementRuntime,GalaxyRefinementRuntime
from test_refinement_integration import create,source,approved
from radar_pd_nova.refinement_sources import RefinementScope
from radar_pd_nova.refinement_workspace import RefinementWorkspace
from test_refinement_plots import view as plotted_view
from test_refinement_trend_arrivals import row,job


def diagnostics(**changes):
    # Serialized field shape from the actual 64314 result; no source GPX/data.
    value=dict(schema=SCHEMA,method='analytic Hessian',native_converged=False,
        aborted=False,native_message='',maximum_cycles=3,
        relative_objective_change=.002,relative_objective_tolerance=.001,
        independent_parameters=18,last_cycle_complete=True,missing_last_cycle_parameters=[],
        last_cycle_ratios={'0::dAx:0':10.155876963661079},
        last_cycle_max_shift_over_sigma=10.155876963661079,
        largest_shift_parameter='0::dAx:0',total_run_max_shift_over_sigma=21.,
        covariance_rank=18,high_correlation_count=4,top_correlations=[])
    value.update(changes)
    return json.loads(json.dumps(value))


def result(evidence=None,criteria=None):
    output={'status':'completed','source_sha256':'source','output_sha256':'final'}
    view={'source_sha256':'final','convergence':evidence or diagnostics()}
    assert retain_review_evidence(output,view,criteria)
    return output


def test_explicit_nonconvergence_marks_review_without_changing_completed_status():
    output=result();before=deepcopy(output)
    review=refinement_review_status(output)
    assert review['needs_review'] and review['status']=='needs_review'
    assert 'did not report convergence' in review['message'] and '10.156' in review['message']
    assert output==before and output['status']=='completed'


def test_native_true_large_shift_is_reported_without_inventing_a_cutoff():
    output=result(diagnostics(native_converged=True,last_cycle_max_shift_over_sigma=10.8929443943163,
        relative_objective_change=-.00034064))
    review=refinement_review_status(output)
    assert not review['needs_review'] and review['status']=='recorded'
    assert '10.893' in review['message'] and 'do not establish model adequacy' in review['message']
    strict=result(output['review_evidence']['convergence'],{'max_shift_over_sigma':.1})
    assert refinement_review_status(strict)['needs_review']


@pytest.mark.parametrize('change',[{'aborted':True},{'last_cycle_complete':False},
    {'missing_last_cycle_parameters':['0::A0']},{'last_cycle_max_shift_over_sigma':float('inf')},
    {'last_cycle_max_shift_over_sigma':float('nan')},{'relative_objective_change':float('inf')}])
def test_explicit_bad_diagnostics_require_review(change):
    assert refinement_review_status(result(diagnostics(native_converged=True,**change)))['needs_review']


def test_legacy_or_cross_source_evidence_is_unavailable_not_nonconvergence():
    for output in ({'status':'completed'},result()):
        if 'review_evidence' in output:output['output_sha256']='another final'
        review=refinement_review_status(output)
        assert review['status']=='unavailable' and not review['needs_review']
        assert 'unavailable' in review['message'] and 'did not' not in review['message']
    assert refinement_review_status(result(diagnostics(native_converged=None,last_cycle_complete=None)))['status']=='unavailable'


def test_capture_is_small_independent_and_survives_json_reload(tmp_path):
    gpx=b'unchanged verified GPX';(tmp_path/'final.gpx').write_bytes(gpx)
    view={'source_sha256':hashlib.sha256(gpx).hexdigest(),'convergence':diagnostics(),'histograms':[{'x':[1,2,3]}]}
    raw=json.dumps(view).encode();(tmp_path/'view.json').write_bytes(raw)
    output={'status':'model_edited','source_sha256':'original','output_sha256':view['source_sha256'],
        'final_gpx':'final.gpx','final_view':'view.json',
        'files':{'final.gpx':view['source_sha256'],'view.json':hashlib.sha256(raw).hexdigest()}}
    verify_result_files(tmp_path,output)
    output.update(status='completed')
    retain_result_review(tmp_path,output,{'stages':[{'convergence':None}]})
    restored=json.loads(json.dumps(output))
    assert refinement_review_status(restored)['needs_review']
    assert 'histograms' not in restored['review_evidence']
    assert (tmp_path/'view.json').read_bytes()==raw and (tmp_path/'final.gpx').read_bytes()==gpx
    assert not retain_review_evidence(output,view)
    view['convergence']['native_converged']=True
    assert output['review_evidence']['convergence']['native_converged'] is False


@pytest.mark.parametrize('contents',['legacy text','[]','{}'])
def test_legacy_view_without_diagnostics_does_not_turn_completed_transport_into_failure(tmp_path,contents):
    path=tmp_path/'legacy.json';path.write_text(contents)
    output={'status':'completed','source_sha256':'source','output_sha256':'final',
        'final_view':'legacy.json','files':{'legacy.json':hashlib.sha256(path.read_bytes()).hexdigest()}}
    before=deepcopy(output);retain_result_review(tmp_path,output)
    assert output==before and refinement_review_status(output)['status']=='unavailable'


@pytest.mark.parametrize('transport',['local','galaxy'])
def test_real_runtime_collection_retains_verified_diagnostics_without_changing_native_receipt(tmp_path,monkeypatch,transport):
    import radar_pd_nova.refinement_runtime as module
    original=tmp_path/'source.gpx';original.write_bytes(b'original')
    source_info={'source_path':str(original),'source_sha256':hashlib.sha256(b'original').hexdigest()}
    plan={'stages':[{'convergence':{'max_shift_over_sigma':.1}}]}
    def artifacts(directory):
        directory=Path(directory);directory.mkdir(exist_ok=True)
        final=b'final';sha=hashlib.sha256(final).hexdigest()
        files={'final.gpx':final,'final.json':json.dumps({'source_sha256':sha,'convergence':diagnostics()}).encode(),
            'baseline.json':b'{}','execution.json':b'{}'}
        for name,blob in files.items():(directory/name).write_bytes(blob)
        response=dict(schema='radar-refinement-worker/v1',action='run',status='completed',
            source_sha256=source_info['source_sha256'],output_sha256=sha,final_gpx='final.gpx',
            final_view='final.json',baseline_view='baseline.json',execution='execution.json',
            stages=[{'id':'last','title':'Last','gpx':'final.gpx','view':'final.json'}],
            files={name:hashlib.sha256(blob).hexdigest() for name,blob in files.items()})
        (directory/'response.json').write_text(json.dumps(response))
    if transport=='local':
        def run(*args,cwd,**kwargs):artifacts(Path(cwd));return SimpleNamespace(returncode=0)
        monkeypatch.setattr(module,'run_scientific',run)
        monkeypatch.setattr(module,'scientific_environment',lambda *_:{})
        runtime=LocalRefinementRuntime(tmp_path/'runtime',python='unused');resume=None
    else:
        bundle=tmp_path/'bundle';artifacts(bundle);archive=tmp_path/'result.zip'
        with zipfile.ZipFile(archive,'w') as target:
            for path in bundle.iterdir():target.write(path,path.name)
        service=SimpleNamespace(_download_dataset=lambda _,path:shutil.copyfile(archive,path))
        runtime=GalaxyRefinementRuntime(tmp_path/'runtime',service)
        resume={'uid':'existing','tool_id':'refine','name':'Existing','status':'ok',
            'galaxy_job_id':'accepted','outputs':{'result_bundle':'bundle'}}
    directory=tmp_path/'run'
    output=runtime.call('run',source_info,directory,plan=plan,approval={},resume=resume)
    assert refinement_review_status(output)['needs_review']
    assert output['review_evidence']['criteria']=={'max_shift_over_sigma':.1}
    assert 'review_evidence' not in json.loads((directory/'response.json').read_text())
    assert (directory/'final.gpx').read_bytes()==b'final'


def test_queue_counts_and_filters_do_not_double_count_completed_review_jobs(tmp_path):
    w,_=create(tmp_path);w.observe([source('1'),source('2')]);recipe=approved(w,RefinementScope('exp'))
    w.start(recipe['id'],recipe['selection_token'])
    values=list(w.state['jobs'].values())
    for index,entry in enumerate(values):entry.update(status='completed',result=result(diagnostics(native_converged=bool(index))))
    class App(RefinementMixin):
        def _refinement_sync_source(self):pass
        def _refinement_sync_approval(self):pass
        def _sync_recipe_preparation_progress(self):pass
        def _refinement_setup_matches(self,*args):return False
    app=App();app._refinement_workspace=w;app._refinement_approving=False;app._refinement_inspecting=False
    app.server=SimpleNamespace(state=SimpleNamespace(**app._refinement_state_defaults(),flush=lambda:None))
    app.server.state.rf_experiment='exp';app._refinement_sync();state=app.server.state
    counts={r['label']:r['value'] for r in state.rf_metrics}
    assert counts['Completed jobs']==2 and counts['Needs attention']==1
    assert state.rf_progress_percent==100 and state.rf_progress_text.startswith('2 of 2 processed')
    needing=[r for r in state.rf_jobs if 'needs_review' in r['filter_statuses']]
    assert len(needing)==1 and needing[0]['status']=='completed'
    assert needing[0]['status_label']=='Completed · needs review'
    assert all('completed' in r['filter_statuses'] for r in state.rf_jobs)


def test_old_completed_row_hydrates_once_and_retains_review_on_restore(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1')]);recipe=approved(w,RefinementScope('exp'))
    entry=w.start(recipe['id'],recipe['selection_token'])[0];key=entry['id']
    directory=tmp_path/'result';directory.mkdir()
    baseline=plotted_view();final=plotted_view()
    final.update(source_sha256='final-hash',convergence=diagnostics())
    for view in (baseline,final):view['parameters'][0]['status']='refined'
    files={}
    for name,value in [('baseline.json',baseline),('final.json',final)]:
        raw=json.dumps(value).encode();(directory/name).write_bytes(raw);files[name]=hashlib.sha256(raw).hexdigest()
    w.state['jobs'][key].update(status='completed',result=dict(status='completed',source_sha256='source',output_sha256='final-hash',
        directory=str(directory),baseline_view='baseline.json',final_view='final.json',files=files))
    w.save();w=RefinementWorkspace(w.root,history_id='history',runtime=rt)
    class App(RefinementMixin):
        def _refinement_sync_source(self):pass
        def _refinement_sync_approval(self):pass
        def _sync_recipe_preparation_progress(self):pass
        def _refinement_setup_matches(self,*args):return False
    app=App();app._refinement_workspace=w;app._refinement_approving=False;app._refinement_inspecting=False;app._refinement_view_cache={}
    state=SimpleNamespace(**app._refinement_state_defaults(),flush=lambda:None)
    state.rf_experiment='exp';state.rf_selected_job=key;state.rf_histogram='bank2';state.rf_has_output=True
    app.server=SimpleNamespace(state=state);app._refinement_sync()
    assert state.rf_jobs[0]['status_label']=='Completed · diagnostics unavailable'
    assert not state.rf_jobs[0]['needs_review']
    app._refinement_plot_changed()
    assert state.rf_jobs[0]['status_label']=='Completed · needs review'
    assert w.state['jobs'][key]['status']=='completed' and state.rf_has_output
    with pytest.raises(ValueError,match='Only interrupted or failed'):w.retry(key)
    restored=RefinementWorkspace(w.root,history_id='history',runtime=rt)
    assert refinement_review_status(restored.state['jobs'][key]['result'])['needs_review']
    before=(w.manifest.read_bytes(),len(app._refinement_view_cache))
    app._refinement_plot_changed()
    assert (w.manifest.read_bytes(),len(app._refinement_view_cache))==before


@pytest.mark.parametrize('metric',['rwp','cell|oxide|c'])
def test_trend_keeps_values_and_marks_only_explicit_review_points(metric):
    rows=[row(1,75,4.),row(2,100,4.04)]
    rows[0]['review']=refinement_review_status(result())
    rows[1]['review']=refinement_review_status(result(diagnostics(native_converged=True)))
    before=deepcopy(rows);fig=trend_figure(rows,metric=metric,x_key='temperature')
    trace=fig.data[-1]
    assert list(trace.customdata)==['1','2'] and list(trace.marker.symbol)==['x','circle']
    assert 'needs review' in trace.name and 'did not report convergence' in trace.hovertext[0]
    assert 'solver reported convergence' in trace.hovertext[1]
    assert list(trace.y)==([5.,5.] if metric=='rwp' else [4.,4.04])
    assert rows==before


def test_compact_arrival_updates_trend_without_replacing_pinned_fit_or_loading_curves():
    first,second=row(1,75,4.),row(2,100,4.04)
    jobs={r['id']:job(r,'completed') for r in [first,second]}
    for entry in jobs.values():entry['result'].update(result())
    class App(RefinementMixin):
        def _read_refinement_view(self,*args):raise AssertionError('Compact review must not read curves')
    app=App();app._refinement_workspace=SimpleNamespace(state={'jobs':jobs,'recipes':{'recipe':{'version':1}}})
    state=SimpleNamespace(**app._refinement_state_defaults(),flush=lambda:None)
    state.rf_experiment='exp';state.rf_selected_job='1';state.rf_fit={'pinned':'unchanged'};state.rf_zoom=17
    state.rf_trend_metric='cell|oxide|c';state.rf_trend_axis='temperature';app.server=SimpleNamespace(state=state)
    app._refinement_trend_changed()
    assert state.rf_selected_job=='1' and state.rf_fit=={'pinned':'unchanged'} and state.rf_zoom==17
    assert state.rf_trend['data'][0]['marker']['symbol']==['x','x']
    assert not state.rf_error
