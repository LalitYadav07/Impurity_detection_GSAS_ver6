import pytest

from radar_pd_nova.webmon.galaxy_runtime import GalaxyLiveRuntime, LIVE_TOOL_ID
from radar_pd_nova.refinement_runtime import RuntimeUnavailable
from radar_pd_nova.models import UtilityActionRecord, RunStatus
from test_webmon_hardening import scientific_request
from test_webmon_galaxy_runtime import Service


@pytest.mark.parametrize('initial', [RunStatus.CANCELLED, RunStatus.RUNNING])
def test_cancelled_live_job_is_terminal_and_dispatch_remains_recoverable(tmp_path, initial):
    src = tmp_path/'src'; src.mkdir(); request, _ = scientific_request(src)
    class CancelledService(Service):
        def __init__(self):
            super().__init__(None)
            self.refreshes = 0
        def submit_utility(self, **kw):
            self.submissions.append(kw)
            return UtilityActionRecord(uid='job', name='Live', tool_id=LIVE_TOOL_ID,
                galaxy_job_id='galaxy-job', status=initial)
        def refresh_utility(self, job):
            self.refreshes += 1
            job.status = RunStatus.CANCELLED
        def _download_dataset(self, *args):
            raise AssertionError('Cancelled jobs have no accepted results')
    service = CancelledService()
    runtime = GalaxyLiveRuntime(tmp_path/'runtime', service, timeout=2, poll_seconds=0)
    with pytest.raises(RuntimeUnavailable, match='was cancelled'):
        runtime.call(request, tmp_path/'first', dispatch_key='d'*24)
    assert service.refreshes == (0 if initial == RunStatus.CANCELLED else 1)
    ledger = tmp_path/'runtime/dispatches'/('d'*24+'.json')
    assert ledger.is_file()
    with pytest.raises(RuntimeUnavailable, match='was cancelled'):
        runtime.call(request, tmp_path/'second', dispatch_key='d'*24)
    assert len(service.submissions) == 1
