"""Selected refinement output belongs to its experiment and job."""
import asyncio
import copy
from types import SimpleNamespace

from radar_pd_nova.refinement_controller import RefinementMixin


class State(SimpleNamespace):
    def flush(self):pass


class App(RefinementMixin):
    def _refinement_sync(self):pass
    def _refinement_plot_changed(self,**_):
        if self.server.state.rf_has_output:
            self.server.state.rf_fit={'restored':'own fit'}
    def _refinement_trend_changed(self,**_):pass
    def _read_refinement_view(self,job,name):return self._refinement_view_cache['verified']


def setup(tmp_path):
    state=State(**RefinementMixin._refinement_state_defaults())
    state.rf_experiment='empty';state.rf_selected_job='old';state.rf_selected_title='Old experiment result'
    state.rf_selected_message='Old accepted stage';state.rf_has_output=True
    state.rf_fit={'old':'curve'};state.rf_parameters=[{'old':'parameter'}]
    state.rf_downloads=[{'value':'old.gpx'}];state.rf_download='old.gpx'
    state.rf_can_open_gsas=True;state.rf_output_provenance='old provenance';state.rf_expanded=True
    state.rf_stages=[{'value':'old-stage'}];state.rf_histograms=[{'value':'old-histogram'}]
    state.rf_histogram='old-histogram';state.rf_parameter_warnings=['old warning']
    state.rf_convergence_notice='old convergence';state.rf_rwp_summary='old Rwp';state.rf_fit_notice='old notice'
    source=dict(id='source',experiment='populated',scan_label='Original scan',sample_label='sample',metadata={})
    job=dict(id='old',source=source,message='Completed',recipe_id='r',recipes=['r'],
        result=dict(directory=str(tmp_path),baseline_view='baseline.json',final_view='final.json',
            final_gpx='final.gpx',files={'final.gpx':'hash'},status='completed'))
    app=App();app.server=SimpleNamespace(state=state);app._refinement_internal=False
    app._refinement_workspace=SimpleNamespace(state={'jobs':{'old':job},'recipes':{'r':{'version':1}},'drafts':{},'sources':{}})
    app._refinement_view_cache={'verified':{'histograms':[{'name':'histogram','fit_statistics':{'calculation_status':'evaluated'}}]}}
    return app,state


def assert_output_cleared(state):
    assert not state.rf_selected_job and not state.rf_selected_title and not state.rf_selected_message
    assert not state.rf_has_output and not state.rf_can_open_gsas and not state.rf_expanded
    assert state.rf_parameters==[] and state.rf_downloads==[] and state.rf_download==''
    assert state.rf_stages==[] and state.rf_histograms==[] and state.rf_histogram==''
    assert state.rf_parameter_warnings==[] and state.rf_convergence_notice==''
    assert state.rf_output_provenance=='' and state.rf_rwp_summary=='' and state.rf_fit_notice==''
    assert 'old' not in state.rf_fit


def test_empty_experiment_clears_detail_and_returning_can_reopen_saved_result(tmp_path):
    app,state=setup(tmp_path);saved=copy.deepcopy(app._refinement_workspace.state)
    cache=app._refinement_view_cache;cached_view=cache['verified']
    app._refinement_experiment_changed()
    assert_output_cleared(state)
    assert app._refinement_workspace.state==saved
    assert app._refinement_view_cache is cache and cache['verified'] is cached_view
    state.rf_experiment='populated';app._refinement_experiment_changed()
    asyncio.run(app._select_refinement_job('old'))
    assert state.rf_selected_job=='old' and state.rf_has_output
    assert state.rf_selected_title=='Original scan · sample · Recipe 1'
    assert state.rf_fit=={'restored':'own fit'} and state.rf_download=='final.gpx'
    assert app._refinement_workspace.state==saved and cache['verified'] is cached_view


def test_explicit_empty_selection_clears_visible_output_only(tmp_path):
    app,state=setup(tmp_path);state.rf_experiment='populated'
    saved=copy.deepcopy(app._refinement_workspace.state)
    asyncio.run(app._select_refinement_job(''))
    assert_output_cleared(state)
    assert app._refinement_workspace.state==saved and 'verified' in app._refinement_view_cache


def test_late_old_view_cannot_restore_detail_after_experiment_changes(tmp_path,monkeypatch):
    app,state=setup(tmp_path);state.rf_experiment='populated'
    async def run():
        entered=asyncio.Event();release=asyncio.Event()
        async def delayed(function,*args):
            entered.set();await release.wait();return function(*args)
        monkeypatch.setattr(asyncio,'to_thread',delayed)
        pending=asyncio.create_task(app._select_refinement_job('old'))
        await entered.wait()
        state.rf_experiment='empty';app._refinement_experiment_changed()
        assert_output_cleared(state)
        release.set();await pending
    asyncio.run(run())
    assert_output_cleared(state)


def test_clearing_selection_from_state_change_releases_visible_output(tmp_path):
    app,state=setup(tmp_path)
    state.rf_selected_job=''
    app._refinement_selection_changed()
    assert_output_cleared(state)
    assert 'old' in app._refinement_workspace.state['jobs']
    assert 'verified' in app._refinement_view_cache


def test_plot_callback_rejects_selection_owned_by_a_different_experiment(tmp_path):
    app,state=setup(tmp_path)
    # Exercise the real callback rather than the fixture's lightweight plot hook.
    RefinementMixin._refinement_plot_changed(app)
    assert_output_cleared(state)
    assert 'old' in app._refinement_workspace.state['jobs']


def test_new_selection_hides_and_clears_old_fit_through_pending_baseline(tmp_path,monkeypatch):
    app,state=setup(tmp_path);state.rf_experiment='populated'
    job=copy.deepcopy(app._refinement_workspace.state['jobs']['old'])
    job['id']='new';job['source']['scan_label']='New scan'
    app._refinement_workspace.state['jobs']['new']=job
    snapshots=[]
    state.flush=lambda:snapshots.append(dict(job=state.rf_selected_job,heading=state.rf_selected_title,
        visible=state.rf_has_output,fit=copy.deepcopy(state.rf_fit)))
    async def run():
        entered=asyncio.Event();release=asyncio.Event()
        async def delayed(function,*args):
            entered.set();await release.wait();return function(*args)
        monkeypatch.setattr(asyncio,'to_thread',delayed)
        pending=asyncio.create_task(app._select_refinement_job('new'))
        await entered.wait()
        # An unrelated progress flush while the executor is occupied cannot
        # expose the previous scan's fit under this scan's new title.
        state.flush()
        assert state.rf_selected_job=='new' and state.rf_selected_title.startswith('New scan ·')
        assert not state.rf_has_output and 'old' not in state.rf_fit
        assert state.rf_parameters==[] and state.rf_downloads==[]
        release.set();await pending
    asyncio.run(run())
    assert state.rf_has_output and state.rf_fit=={'restored':'own fit'}
    assert all(not row['visible'] or row['fit']=={'restored':'own fit'} for row in snapshots)
