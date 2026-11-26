"""A display-only preview cannot mutate independently verified native work."""
import asyncio
from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from radar_pd_nova.refinement_runtime import GalaxyRefinementRuntime
from radar_pd_nova.refinement_sources import RefinementScope
from test_fit_first_opening import setup_preview
from test_refinement_catalog_io import App as CatalogApp, Service, record
from test_refinement_integration import approved, create, source


def approved_preview(tmp_path):
    view=tmp_path/'view';view.mkdir()
    app,row,preview,_,_,_,_=setup_preview(view)
    workspace,runtime=create(tmp_path/'refinement')
    workspace.observe([source(row.uid)])
    recipe=approved(workspace,RefinementScope('exp'),future=True)
    workspace.start(recipe['id'],recipe['selection_token'])
    app._refinement_workspace=workspace
    app._refinement_maybe_refresh=lambda:pytest.fail('A display preview must not refresh or enroll native sources')
    return app,row,preview,workspace,runtime


def test_opening_preview_leaves_approved_sources_jobs_and_queue_unchanged(tmp_path):
    app,row,preview,workspace,runtime=approved_preview(tmp_path)
    before=deepcopy(workspace.state);calls=deepcopy(runtime.calls)
    record_before=row.model_dump(mode='json')
    app._select_record(row)
    assert app._display_result_preview(row,preview,app.service,
        app._result_source_identity(row),app._result_context)
    assert app.server.state.result_preview
    assert workspace.state==before and runtime.calls==calls
    assert row.model_dump(mode='json')==record_before and app.records[row.uid].output_dir is None
    assert not workspace._inflight and not workspace._submission_cancels
    assert not list(preview.root.rglob('*.gpx'))


def test_verified_approved_job_remains_runnable_during_same_uid_preview(tmp_path):
    app,row,preview,workspace,runtime=approved_preview(tmp_path)
    app._select_record(row)
    assert app._display_result_preview(row,preview,app.service,
        app._result_source_identity(row),app._result_context)
    queued=next(iter(workspace.state['jobs'].values()))
    identity=deepcopy(queued['source']);approval=deepcopy(workspace.state['recipes'][queued['recipe_id']]['approval'])
    runtime.calls.clear()
    result=workspace.run_next()
    assert result['status']=='completed' and result['attempts']==1
    assert [(action,uid) for action,uid,_ in runtime.calls]==[('run',row.uid)]
    assert result['source']==identity
    assert workspace.state['recipes'][queued['recipe_id']]['approval']==approval
    assert 'result_preview_hold' not in result


def test_native_source_materialization_keeps_exact_published_hash_verification(tmp_path):
    content=b'opaque synthetic published GPX'
    requests=[]
    class NativeService:
        def collection_elements(self,identifier):
            assert identifier=='published-collection'
            return [{'id':'published-dataset','name':'accepted'}]
        def _download_dataset(self,identifier,path):
            requests.append(identifier);Path(path).write_bytes(content)
    runtime=GalaxyRefinementRuntime(tmp_path/'native',NativeService())
    descriptor=dict(source_path=None,source_sha256=hashlib.sha256(content).hexdigest(),
        collection_id='published-collection',collection_name='accepted.gpx')
    original=deepcopy(descriptor)
    resolved=runtime.resolve(descriptor)
    assert descriptor==original and requests==['published-dataset']
    assert Path(resolved['source_path']).read_bytes()==content
    changed=dict(descriptor,source_sha256='0'*64)
    with pytest.raises(ValueError,match='GPX changed'):
        runtime.resolve(changed)
    assert requests==['published-dataset']


def test_catalog_uses_canonical_publication_not_the_display_preview(tmp_path):
    rows=[record(1)];service=Service(rows);app=CatalogApp(tmp_path/'catalog',rows,service)
    preview=tmp_path/'preview';preview.mkdir()
    (preview/'forged.gpx').write_bytes(b'preview data has no native authority')
    app.server.state.result_preview=True
    app._result_preview_pending=lambda _:True
    # Independent source catalogs use canonical records and their published
    # identities. A UI preview flag cannot remove an otherwise usable source.
    asyncio.run(app._refresh_refinement_sources())
    native=next(iter(app._refinement_workspace.state['sources'].values()))
    assert native['record_uid']=='record-1'
    assert native['collection_id']=='collection-1'
    assert native['source_sha256']==f'{101:064x}'
    assert native['source_path'] is None and rows[0].output_dir is None
    assert sorted(service.calls)==['index-1','summary-1']
    assert not app._refinement_workspace.state['jobs']
