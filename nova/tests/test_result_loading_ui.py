"""Timing progress follows the same source/selection owner as the result."""
import asyncio
import pytest

from radar_pd_nova.result_loading import measure
from test_fit_first_opening import setup_preview, until


def test_loading_timing_records_partial_and_complete_without_authorizing_preview(tmp_path, monkeypatch):
    app,row,preview,_,entered,release,_=setup_preview(tmp_path)
    collect=app.service.collect_results
    def measured_collect(record,**kwargs):
        with measure('response_body','summary'):
            return collect(record,**kwargs)
    app.service.collect_results=measured_collect
    def measured_preview(*_):
        with measure('preview_plot'):
            return preview
    monkeypatch.setattr('radar_pd_nova.result_preview.collect_result_preview',measured_preview)
    async def run():
        app._select_record(row);app._open_record_results(row)
        await until(lambda:app.server.state.result_preview)
        assert not app._result_actions_ready(row.uid)
        assert 'observed' in app.server.state.selected_result_load_elapsed
        release.set();await asyncio.gather(*list(app._history_result_requests.values()))
        profile=app.records[row.uid].result_load_profile
        assert profile['job_id']==row.uid and 'events' not in profile
        assert {'open_requested','preview_display_queued','canonical_ready','canonical_display_queued'} <= profile['milestones'].keys()
        assert {e['branch'] for e in profile['aggregates']}=={'complete','preview','display'}
        assert not app.server.state.result_preview and app._result_actions_ready(row.uid)
    asyncio.run(run())


def test_old_progress_after_selection_change_cannot_replace_new_selection(tmp_path,monkeypatch):
    app,row,preview,_,entered,release,_=setup_preview(tmp_path)
    monkeypatch.setattr('radar_pd_nova.result_preview.collect_result_preview',lambda *_:preview)
    other=row.model_copy(deep=True,update={'uid':'B','galaxy_job_id':'B','output_dataset_ids':{'summary':'summary-B'}})
    app.records[other.uid]=other
    async def run():
        app._select_record(row);app._open_record_results(row)
        await until(lambda:app.server.state.result_preview)
        app._select_record(other)
        assert not app.server.state.selected_result_load_rows
        app.server.state.selected_result_load_elapsed='New selection owns this value'
        release.set();await asyncio.gather(*list(app._history_result_requests.values()))
        assert app.server.state.selected_run_uid=='B'
        assert app.server.state.selected_result_load_elapsed=='New selection owns this value'
        assert not app.server.state.selected_result_load_rows
    asyncio.run(run())


@pytest.mark.parametrize('replacement',[False,True])
def test_loading_profile_follows_artifact_identity_on_metadata_refresh(tmp_path,replacement):
    app,row,*_=setup_preview(tmp_path)
    row.output_dir=str(tmp_path/'canonical')
    row.result_load_profile={'trace_id':'old-attempt'}
    refreshed=row.model_copy(deep=True)
    if replacement:refreshed.output_dataset_ids['summary']='replacement-summary'
    else:refreshed.output_dataset_ids['plots']='additional-plots'
    merged=app._merge_refreshed_result(refreshed,row)
    assert merged.result_load_profile==({} if replacement else {'trace_id':'old-attempt'})
