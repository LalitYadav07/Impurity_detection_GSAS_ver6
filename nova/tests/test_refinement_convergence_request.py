"""The hosted planner preserves numerical criteria before native preflight."""
import copy
import json

from radar_pd_nova.refinement_planner import make_plan
from radar_pd_nova.refinement_workspace import RefinementWorkspace
from test_refinement_planner_payload import ContractClient, Runtime, four_stage_plan, native_context


def test_host_repair_keeps_requested_ceiling_in_final_messages():
    context = native_context()
    context['controls'] = {'min dM/M': .001}
    request = 'Use exactly four stages. Require final-cycle shift/sigma <= 0.5.'
    correct = four_stage_plan(context, request)
    for stage in correct['stages']:
        stage['convergence'] = {'max_shift_over_sigma': .5}
    wrong = copy.deepcopy(correct)
    wrong['stages'][0]['convergence']['max_shift_over_sigma'] = .001
    client = ContractClient([json.dumps(wrong), json.dumps(correct)])
    result = make_plan(request, context, client=client)
    assert len(client.calls) == 2
    assert result['plan'] == correct
    for call in client.calls:
        payload = json.loads(call[1]['content'])
        assert payload['convergence_request']['global_maximum'] == .5
        assert 'relative-objective tolerance' in payload['convergence_instruction']


def test_host_missing_retained_ceiling_does_not_dispatch_preflight(tmp_path, monkeypatch):
    context = native_context()
    context['controls'] = {'min dM/M': .001}
    runtime = Runtime(context)
    client = ContractClient([])
    monkeypatch.setattr('radar_pd_nova.refinement_workspace.configure_provider', lambda *a: None)
    monkeypatch.setattr('radar_pd_nova.refinement_workspace.make_client', lambda: client)
    workspace = RefinementWorkspace(tmp_path, history_id='test-history', runtime=runtime)
    workspace.observe([dict(id='source', experiment='exp', scan_key='scan', source_sha256='test-source',
                            sample_key='sample', bank='2', scan_label='scan', run_number=1)])
    draft = workspace.prepare('exp', 'source', 'Retain the original convergence threshold.')
    assert draft['status'] == 'needs_input'
    assert draft['preview'] is None and draft['plan']['stages'] == []
    assert client.calls == [] and runtime.calls == ['inspect']
    assert 'relative-objective tolerance' in draft['plan']['questions'][0]
