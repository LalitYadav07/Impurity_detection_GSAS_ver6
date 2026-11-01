import asyncio
import hashlib
import json
from types import SimpleNamespace

from radar_pd_nova.webmon.ui import WebmonMixin


def test_terminal_snapshot_without_models_does_not_display_previous_fit(tmp_path):
    current=tmp_path/'current';current.mkdir()
    request={'source_revision':'PG3_2:bbbbbbbbbbbb','reference_revision':'PG3_1:aaaa'}
    path=current/'request.json';path.write_text(json.dumps(request))
    result={**request,'request_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'status':'complete','rows':[{'id':'new','name':'Model','bns':'1.1','status':'numerical_failure'}]}
    (current/'result.json').write_text(json.dumps(result))
    state=SimpleNamespace(webmon_has_result=True,webmon_run='2',webmon_revision='bbbbbbbbbbbb',
        webmon_compared=['old'],webmon_table_selected=['old'],webmon_model='old',webmon_filter='All',flush=lambda:None)
    view=WebmonMixin();view.server=SimpleNamespace(state=state);view._webmon_out=tmp_path/'old'
    view._webmon_model_paths={'old'};view._webmon_payloads={};view._webmon_request={}
    rendered=[];view.webmon_render=lambda:rendered.append(state.webmon_has_result)
    view._webmon_load_result(current)
    assert state.webmon_has_result is False and not view._webmon_model_paths
    assert state.webmon_rows[0]['id']=='new' and not state.webmon_model
    assert state.webmon_analysis_revision=='bbbbbbbbbbbb'
    assert rendered==[False] and not state.webmon_pending_rows


def test_changing_live_run_during_reference_resolution_never_dispatches(tmp_path,monkeypatch):
    state=SimpleNamespace(webmon_busy=False,webmon_prepare_accepted=True,
        webmon_published_source='source',powgen_ipts='1',webmon_run='2',sx_kind='magnetic',flush=lambda:None)
    view=WebmonMixin();view.server=SimpleNamespace(state=state)
    view._webmon_published_catalog={'source':{'id':'source'}}
    view._symmetry_draft={'source_id':'source','discovery_identity':'reviewed'}
    view._webmon_reviewed_mapping={'source_id':'source','snapshot_sha256':'a'*64}
    view._discovery_identity=lambda:'reviewed';view._model_form=lambda:{}
    view._webmon=SimpleNamespace(config={'jobs':str(tmp_path)},archive=SimpleNamespace(read=lambda r:b'captured'))
    view._webmon.capture=lambda *a:({'path':'old.dat'},{'content_sha256':'a'*64},None)
    def resolve(source):
        state.webmon_run='3'
        return {'source_path':'reference.gpx','source_sha256':'b'*64}
    view._refinement_store=lambda:SimpleNamespace(runtime=SimpleNamespace(resolve=resolve))
    def forbidden(*args,**kwargs):raise AssertionError('Stale request must not construct a runtime')
    monkeypatch.setattr('radar_pd_nova.webmon.galaxy_runtime.GalaxyLiveRuntime',forbidden)
    asyncio.run(view.webmon_prepare_from_models())
    assert 'selection changed' in state.webmon_source_notice
    assert not state.webmon_busy
