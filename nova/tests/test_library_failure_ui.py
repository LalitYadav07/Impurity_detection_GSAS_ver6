"""Library errors stay concise, inspectable and bound to the owning draft."""
import asyncio
from copy import deepcopy
from html.parser import HTMLParser
from types import SimpleNamespace
from uuid import uuid4

import pytest
from trame.app import get_server

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.library_review import library_review_signature
from radar_pd_nova.library_setup import LibraryDraft
from radar_pd_nova.models import RunStatus, UtilityActionRecord


TRACE = ("[NDIP ERROR] All CIF phases failed to build: CIF symmetry review required: "
         "Declared CIF symmetry symbol needs setting review\nTraceback (most recent call last):\n"
         + "  File '/native/worker.py', line 42, in build\n" * 300
         + "ValueError: CIF symmetry review required\n<script>not executable</script>")


@pytest.fixture
def failure_case(tmp_path, monkeypatch):
    server=get_server('library-failure-'+uuid4().hex,client_type='vue3')
    monkeypatch.setattr('radar_pd_nova.app.get_server',lambda *_a,**_k:server)
    app=RadarPdNovaApp();server.state.ready()
    app.service=SimpleNamespace(galaxy_url='https://galaxy.example')
    draft=LibraryDraft(server.state)
    path=tmp_path/'parent.cif';path.write_text('data_parent\n')
    draft.database_source='custom_mini';draft.library_builder_local_paths=[str(path)]
    draft.library_builder_name='Parent review';draft.library_builder_generation='generation-one';draft.flush()
    owner=dict(workflow='single',generation=draft.library_builder_generation,
               signature=draft.signature(),source_signature=library_review_signature(draft))
    action=UtilityActionRecord(uid='failed-library',tool_id='neutrons_radar_pd_library_builder_prototype',
        name='Build parent library',galaxy_job_id='123abc',status=RunStatus.ERROR,message=TRACE)
    app.utility_actions[action.uid]=action
    return app,draft,owner,action,path


def test_long_native_trace_has_short_summary_full_collapsed_diagnostics_and_exact_job(failure_case):
    app,draft,owner,action,_=failure_case
    assert app._record_library_build_failure(draft,TRACE,owner,action)
    draft.flush()
    assert draft.library_builder_status=='error' and action.status==RunStatus.ERROR
    assert len(draft.library_builder_error['summary'])<150
    assert 'symmetry setting needs review' in draft.library_builder_error['summary']
    assert 'Traceback' not in draft.library_builder_error['summary']
    assert draft.library_builder_message==TRACE
    assert draft.library_builder_error['job_url']=='https://galaxy.example/jobs/123abc/view'
    assert app.utility_actions[action.uid].message==TRACE
    # These are real Trame widgets: raw diagnostics are in a closed native
    # disclosure and a text-bound <pre>, never interpreted as HTML.
    class Elements(HTMLParser):
        def __init__(self):super().__init__();self.details=[];self.pre=[];self.stack=[]
        def handle_starttag(self,tag,attrs):
            values=dict(attrs)
            if tag=='details' and values.get('class')=='radar-library-error-details':self.details.append(values)
            if tag=='pre':self.pre.append((values,'details' in self.stack))
            if tag not in {'input','br','img','hr','meta','link'}:self.stack.append(tag)
        def handle_endtag(self,tag):
            if tag in self.stack:
                while self.stack.pop()!=tag:pass
    parser=Elements();markup=app.layout.html;parser.feed(markup)
    assert parser.details and all('open' not in row for row in parser.details)
    assert any(inside and 'max-height:20rem' in row.get('style','') for row,inside in parser.pre)
    assert "library_builder_status !== 'error'" in markup
    assert 'Open Galaxy job' in markup and 'library_builder_error.job_url' in markup


@pytest.mark.parametrize('change',['name','sources','radiation','rounding','file-bytes'])
def test_editing_the_owning_inputs_clears_only_its_error_and_job_link(failure_case,change):
    app,draft,owner,action,path=failure_case
    app._record_library_build_failure(draft,TRACE,owner,action)
    other=LibraryDraft(app.server.state,'live');other.database_source='custom_mini';other.flush()
    other_owner=dict(workflow='live',signature=other.signature(),source_signature=library_review_signature(other))
    app._record_library_build_failure(other,'Other draft error',other_owner)
    other_error=deepcopy(other.library_builder_error)
    if change=='name':draft.library_builder_name='Changed parent'
    elif change=='sources':draft.library_builder_local_paths=[]
    elif change=='radiation':draft.radiation='xray'
    elif change=='rounding':draft.library_builder_allow_rounding=True
    else:path.write_text('data_changed_parent\n')
    app._library_review_changed();draft.flush()
    assert draft.library_builder_error=={} and draft.library_builder_status=='idle'
    assert not draft.library_builder_message
    assert other.library_builder_error==other_error
    assert action.status==RunStatus.ERROR and action.message==TRACE


def test_late_failure_cannot_replace_changed_draft(failure_case):
    app,draft,owner,action,_=failure_case
    draft.library_builder_name='New selection';draft.library_builder_message='New draft';draft.flush()
    before=deepcopy(draft.library_builder_error)
    assert not app._record_library_build_failure(draft,TRACE,owner,action)
    assert draft.library_builder_message=='New draft' and draft.library_builder_error==before


@pytest.mark.parametrize('change',['no-action','unverified-id','different-tool','credential-url'])
def test_missing_or_unsafe_job_identity_has_no_job_link(failure_case,change):
    app,draft,owner,action,_=failure_case
    if change=='no-action':action=None
    elif change=='unverified-id':action.galaxy_job_id='../wrong'
    elif change=='different-tool':action.tool_id='another-tool'
    else:app.service.galaxy_url='https://secret@galaxy.example'
    assert app._record_library_build_failure(draft,TRACE,owner,action)
    assert draft.library_builder_status=='error' and not draft.library_builder_error['job_url']
    assert draft.library_builder_message==TRACE


@pytest.mark.parametrize('change_during_build',[False,True])
def test_actual_builder_failure_path_publishes_only_current_draft(failure_case,tmp_path,monkeypatch,change_during_build):
    app,draft,_,action,_=failure_case
    bundle=tmp_path/'sources.zip';bundle.write_bytes(b'fixture source bundle')
    monkeypatch.setattr('radar_pd_nova.app.build_cif_source_archive',lambda *_:(bundle,dict(source_count=1,cif_count=1)))
    app.service._upload_one=lambda *_:('sources.zip','owned-source-hda')
    async def submit(**_):
        if change_during_build:draft.library_builder_name='New draft'
        return action
    app._submit_utility_action=submit;pending=[]
    app._schedule_utility=lambda task,_:pending.append(task)
    app.build_candidate_library();asyncio.run(pending.pop());draft.flush()
    assert not draft.library_builder_active and action.status==RunStatus.ERROR
    if change_during_build:
        assert draft.library_builder_name=='New draft' and draft.library_builder_error=={}
        assert draft.library_builder_status=='idle' and 'Inputs changed' in draft.library_builder_message
    else:
        assert draft.library_builder_status=='error' and draft.library_builder_message==TRACE
        assert draft.library_builder_error['job_url'].endswith('/jobs/123abc/view')
