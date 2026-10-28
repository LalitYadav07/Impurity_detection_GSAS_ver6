"""The benchmark transport must preserve bounded model conversations under 429s."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
path=Path(__file__).resolve().parents[2]/'benchmarks/gpx_recipes/run_campaign.py'
spec=importlib.util.spec_from_file_location('recipe_campaign',path);campaign=importlib.util.module_from_spec(spec);spec.loader.exec_module(campaign)


def test_throttled_repair_retries_identical_conversation_without_replanning(monkeypatch,tmp_path):
    from gpx_recipe.providers import ProviderRequestError
    class Client:
        max_output_tokens=2048
        def __init__(self):self.messages=[]
        def complete(self,messages):
            self.messages.append(json.loads(json.dumps(messages)))
            if len(self.messages)==2:raise ProviderRequestError('test',429,1)
            return ('first' if len(self.messages)==1 else 'repair'),{}
    sleeps=[];monkeypatch.setattr(campaign.time,'sleep',lambda seconds:sleeps.append(seconds))
    real=Client();client=campaign.PacedClient(real,tmp_path);client.case='test'
    original=[{'role':'user','content':'recipe'}];assert client.complete(original)[0]=='first'
    repair=original+[{'role':'assistant','content':'first'},{'role':'user','content':'fix'}]
    assert client.complete(repair)[0]=='repair'
    assert real.messages==[original,repair,repair] and sleeps==[5]
    record=json.loads(next(tmp_path.glob('rate-limit-*.json')).read_text());assert record['http_status']==429


def test_throttle_cannot_sleep_or_send_request_after_deadline(monkeypatch,tmp_path):
    from gpx_recipe.providers import ProviderRequestError
    class Client:
        max_output_tokens=2048
        def complete(self,messages):raise ProviderRequestError('test',429,60)
    monkeypatch.setattr(campaign.time,'sleep',lambda seconds:pytest.fail('No out-of-window wait'))
    client=campaign.PacedClient(Client(),tmp_path);client.deadline=campaign.time.time()+1
    with pytest.raises(campaign.CampaignDeadline,match='Rate-limit wait'):client.complete([])
    client.deadline=campaign.time.time()-1
    with pytest.raises(campaign.CampaignDeadline,match='No new model'):client.complete([])


def test_actual_plan_revision_does_not_wait_for_unrelated_native_job_completion(monkeypatch,tmp_path):
    import copy,threading
    from types import SimpleNamespace
    context={'model_signature':'x','histograms':[{'key':'h0','name':'h','background':{'function':'chebyschev-1','coefficients':6}}],'phases':[]}
    def plan(cycles):return {'schema':'radar-gpx-recipe/v1','model_signature':'x','request':'explicit background','assumptions':[],'questions':[],'unsupported':[],'stages':[{'id':'s1','title':'BG','cycles':cycles,'free':[{'kind':'background','histogram':'h0'}],'constraints':[]}]}
    cases=[{'id':'A','project':'p','category':'test','expectation':'execute','request':'first','oracle':plan(4)}, {'id':'B','project':'p','category':'test','expectation':'execute','request':'revision','previous_case':'A','oracle':plan(5)}]
    (tmp_path/'inputs').mkdir();(tmp_path/'inputs/p.context.json').write_text(json.dumps(context));(tmp_path/'catalog.json').write_text(json.dumps(cases));(tmp_path/'execution-environment.json').write_text(json.dumps({'image_id':'test-only'}))
    revised=threading.Event();seen=[]
    def planning(request,ctx,previous=None,**kw):
        seen.append(previous)
        if request=='revision':
            assert previous==plan(4);revised.set()
        return {'plan':plan(4 if request=='first' else 5),'calls':[{'elapsed_seconds':.01,'usage':{}}],'status':'ready_for_preflight'}
    def native(command,**kw):
        if command[3]=='A':assert revised.wait(2),'Revision was incorrectly blocked by an unfinished scientific job'
        work=Path(command[-1]);(work/'result.json').write_text(json.dumps({'status':'verified_execution'}));return SimpleNamespace(returncode=0)
    monkeypatch.setattr(campaign,'make_plan',planning);monkeypatch.setattr(campaign,'run_scientific',native)
    monkeypatch.setattr(campaign,'configure_provider',lambda *a:None);monkeypatch.setattr(campaign,'make_client',lambda:SimpleNamespace(max_output_tokens=2048))
    monkeypatch.setattr(sys,'argv',['campaign',str(tmp_path),'--mode','model'])
    campaign.main();status=json.loads((tmp_path/'model-status.json').read_text())
    assert status['counts']=={'verified_execution':2} and seen[0] is None
    previous=next((tmp_path/'cases/B/model').glob('*/previous-model-plan.json'))
    assert json.loads(previous.read_text())['plan']==plan(4)


def test_authorization_failure_opens_circuit_before_any_further_recipe_calls(monkeypatch,tmp_path):
    from gpx_recipe.providers import ProviderRequestError
    from types import SimpleNamespace
    context={'model_signature':'x','histograms':[{'key':'h0','name':'h','background':{'function':'chebyschev-1','coefficients':6}}],'phases':[]}
    cases=[{'id':k,'project':'p','category':'test','expectation':'needs_input','request':'Explain an unsupported operation','oracle':None} for k in ['A','B','C']]
    (tmp_path/'inputs').mkdir();(tmp_path/'inputs/p.context.json').write_text(json.dumps(context));(tmp_path/'catalog.json').write_text(json.dumps(cases));(tmp_path/'execution-environment.json').write_text(json.dumps({'image_id':'test-only'}))
    calls=[]
    def planning(*a,**kw):calls.append(True);raise ProviderRequestError('test',403,error_code='permissiondenied')
    monkeypatch.setattr(campaign,'make_plan',planning);monkeypatch.setattr(campaign,'configure_provider',lambda *a:None);monkeypatch.setattr(campaign,'make_client',lambda:SimpleNamespace(max_output_tokens=2048))
    monkeypatch.setattr(sys,'argv',['campaign',str(tmp_path),'--mode','model'])
    campaign.main();state=json.loads((tmp_path/'model-status.json').read_text())
    assert len(calls)==1 and state['counts']=={'provider_blocked':1,'deferred_provider':2}
    assert json.loads((tmp_path/'model-service-halt.json').read_text())['http_status']==403


def test_transport_circuit_requires_consecutive_failures_and_success_resets_it():
    from gpx_recipe.providers import ProviderUnavailable
    gate=campaign.ServiceCircuit();error=ProviderUnavailable('timeout')
    assert gate.fail(error,'A') is None and gate.fail(error,'B') is None
    gate.success();assert gate.fail(error,'C') is None and gate.fail(error,'D') is None
    assert gate.fail(error,'E')['case']=='E'
