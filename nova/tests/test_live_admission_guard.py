"""Optional admission hooks fail closed before a fresh Analyze POST."""
import threading
from types import SimpleNamespace

import pytest

from radar_pd_nova.powgen_controller import PowgenWatchController


@pytest.mark.parametrize('guard', ['missing', None, True, lambda: False],
                         ids=['missing', 'none', 'noncallable', 'revoked'])
def test_active_admission_requires_explicit_callable_owner_guard(guard):
    from radar_pd_nova.galaxy_service import SubmissionDeferred
    controller = object.__new__(PowgenWatchController)
    controller._submission_pause_event = threading.Event()
    if guard != 'missing':
        controller._retry_submission_allowed = guard
    posts = []
    def submit(snapshot, *, before_submit, admission_pause):
        assert admission_pause is controller._submission_pause_event
        before_submit()
        posts.append(snapshot)
    controller.service = SimpleNamespace(submit_snapshot=submit)
    with pytest.raises(SubmissionDeferred, match='no longer admits'):
        controller._submit_live_snapshot(SimpleNamespace(scan_metadata={}), object())
    assert not posts


@pytest.mark.parametrize('retry', [False, True])
def test_active_admission_composes_owner_launch_and_retry_guards(retry):
    controller = object.__new__(PowgenWatchController)
    controller._submission_pause_event = threading.Event()
    calls = []
    def allowed():
        calls.append('owner')
        return True
    controller._retry_submission_allowed = allowed
    controller._before_live_submit = lambda: calls.append('launch')
    controller._validate_retry_submission = lambda run: calls.append('retry')
    def submit(snapshot, *, before_submit, admission_pause):
        assert admission_pause is controller._submission_pause_event
        before_submit()
        calls.append('post')
        return 'accepted'
    controller.service = SimpleNamespace(submit_snapshot=submit)
    run = SimpleNamespace(scan_metadata={'retry_source_proof': {'job': 'old'}} if retry else {})
    assert controller._submit_live_snapshot(run, object()) == 'accepted'
    assert calls == (['owner', 'launch', 'retry', 'post'] if retry
                     else ['owner', 'launch', 'post'])


@pytest.mark.parametrize('revocation', ['pause', 'owner'])
def test_active_admission_is_rechecked_after_input_preparation(revocation):
    from radar_pd_nova.galaxy_service import SubmissionDeferred
    controller = object.__new__(PowgenWatchController)
    pause = controller._submission_pause_event = threading.Event()
    owner = {'allowed': True}
    controller._retry_submission_allowed = lambda: owner['allowed']
    posts = []
    def submit(snapshot, *, before_submit, admission_pause):
        assert admission_pause is pause
        if revocation == 'pause':
            pause.set()
        else:
            owner['allowed'] = False
        before_submit()
        posts.append(snapshot)
    controller.service = SimpleNamespace(submit_snapshot=submit)
    with pytest.raises(SubmissionDeferred, match='no longer admits'):
        controller._submit_live_snapshot(SimpleNamespace(scan_metadata={}), object())
    assert not posts


