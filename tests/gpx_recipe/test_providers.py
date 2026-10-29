"""Provider isolation, lowest Luna effort and credential-free scientific workers."""
import io
import json
from pathlib import Path
import sys

import pytest
from test_contracts import context, plan

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from gpx_recipe import providers
from gpx_recipe.jobs import scientific_environment


def response(text='{"stages": []}'):
    return {"status": "completed", "model": "gpt-6-luna", "usage": {"output_tokens": 20},
            "output": [{"type": "reasoning", "encrypted_content": "NEVER_LOG_THIS"},
                       {"type": "message", "content": [{"type": "output_text", "text": text}]}]}


def test_explicit_sambanova_selection_does_not_change_when_luna_key_exists(monkeypatch):
    monkeypatch.setenv("RADAR_GPX_PLANNER_PROVIDER", "sambanova")
    monkeypatch.setenv("AZURE_ILLUMINE_API_KEY", "only-luna-key")
    assert providers.provider_info()["model"] == "gpt-oss-120b"
    with pytest.raises(ValueError, match="SAMBANOVA_API_KEY"):
        monkeypatch.delenv("SAMBANOVA_API_KEY", raising=False)
        providers.make_client()


def test_luna_uses_none_and_only_its_resource_credential(monkeypatch):
    monkeypatch.setenv("AZURE_ILLUMINE_API_KEY", "luna-secret")
    monkeypatch.setenv("SAMBANOVA_API_KEY", "oss-secret")
    calls = []
    class Opener:
        def open(self, request, timeout):
            calls.append((request, timeout))
            handle = io.BytesIO(json.dumps(response()).encode()); handle.headers = {}
            return handle
    monkeypatch.setattr(providers.urllib.request, "build_opener", lambda *a: Opener())
    raw, meta = providers.IllumineLuna().complete([{"role": "user", "content": "JSON recipe"}])
    req, timeout = calls[0]; body = json.loads(req.data)
    assert req.full_url == "https://openai-eus-illumine.openai.azure.com/openai/v1/responses"
    assert req.get_header("Api-key") == "luna-secret" and req.get_header("Authorization") is None
    assert body["reasoning"] == {"effort": "none"} and body["store"] is False
    assert body["model"] == "gpt-6-luna" and timeout == 180
    assert meta["reasoning_effort"] == "none"
    assert "NEVER_LOG_THIS" not in raw + json.dumps(meta)


@pytest.mark.parametrize("change", ["incomplete", "model", "refusal", "empty"])
def test_incomplete_wrong_model_or_refusal_never_becomes_a_plan(monkeypatch, change):
    monkeypatch.setenv("AZURE_ILLUMINE_API_KEY", "test-key")
    data = response()
    if change == "incomplete": data["status"] = "incomplete"
    if change == "model": data["model"] = "another-model"
    if change == "refusal": data["output"][1]["content"] = [{"type": "refusal"}]
    if change == "empty": data["output"] = []
    monkeypatch.setattr(providers, "_post", lambda *a: data)
    with pytest.raises(providers.ProviderResultError) as raised:
        providers.IllumineLuna().complete([])
    assert raised.value.metadata['usage']['output_tokens'] == 20
    assert 'NEVER_LOG_THIS' not in json.dumps(raised.value.metadata)


def test_rejected_response_usage_is_counted_without_accepting_a_plan(monkeypatch, context):
    from gpx_recipe.planner import make_plan
    monkeypatch.setenv('AZURE_ILLUMINE_API_KEY','test-key')
    data=response();data['output'][1]['content']=[{'type':'refusal'}]
    data['usage']={'input_tokens':900,'output_tokens':30,'total_tokens':930}
    monkeypatch.setattr(providers,'_post',lambda *a:data)
    with pytest.raises(providers.ProviderResultError) as raised:
        make_plan('Refine the selected atom',context,client=providers.IllumineLuna())
    assert len(raised.value.planning_calls)==1
    assert raised.value.planning_calls[0]['usage']['total_tokens']==930
    assert raised.value.planning_calls[0]['response']==''


def test_provider_credentials_cannot_be_redirected_to_another_resource(monkeypatch):
    monkeypatch.setenv("AZURE_ILLUMINE_BASE_URL", "https://aoai-eastus-bead.openai.azure.com/openai/v1")
    monkeypatch.setenv("AZURE_ILLUMINE_API_KEY", "secret")
    with pytest.raises(ValueError, match="restricted"):
        providers.IllumineLuna()
    assert providers.provider_info("sambanova")["model"] == "gpt-oss-120b"


def test_local_credential_file_is_private_and_never_copied_to_workers(tmp_path, monkeypatch):
    path = tmp_path / "key"; path.write_text("private-development-key-value"); path.chmod(0o644)
    with pytest.raises(ValueError, match="owner-only"):
        providers.configure_provider("illumine-luna", path)
    path.chmod(0o600)
    monkeypatch.setenv("RADAR_GPX_PLANNER_PROVIDER", "sambanova")
    monkeypatch.setenv("AZURE_ILLUMINE_API_KEY", "old-key")
    providers.configure_provider("illumine-luna", path)
    assert providers.make_client().info["reasoning_effort"] == "none"
    for name in ("SAMBANOVA_API_KEY", "AZURE_OPENAI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(name, "another-secret")
    env = scientific_environment()
    assert not any("KEY" in k or "SECRET" in k for k in env)
    link = tmp_path / "linked-key"; link.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        providers.configure_provider("illumine-luna", link)


def test_transport_failure_never_substitutes_oss(monkeypatch):
    monkeypatch.setenv("RADAR_GPX_PLANNER_PROVIDER", "illumine-luna")
    monkeypatch.setenv("AZURE_ILLUMINE_API_KEY", "private-key")
    calls = []
    def fail(*args):
        calls.append(args[0]); raise RuntimeError("Provider unavailable")
    monkeypatch.setattr(providers, "_post", fail)
    with pytest.raises(RuntimeError): providers.make_client().complete([])
    assert len(calls) == 1 and "illumine" in calls[0]


def test_rate_limit_has_safe_retry_guidance_and_never_retries(monkeypatch):
    calls = []
    class Opener:
        def open(self, request, timeout):
            calls.append(request)
            raise providers.urllib.error.HTTPError(request.full_url, 429, "rate limited", {"retry-after-ms": "2500"}, io.BytesIO(b"private provider content"))
    monkeypatch.setattr(providers.urllib.request, "build_opener", lambda *a: Opener())
    with pytest.raises(providers.ProviderRequestError) as raised:
        providers._post("https://example.invalid", {}, {}, "test")
    assert raised.value.http_status == 429 and raised.value.retry_after_seconds == 2.5
    assert "private provider content" not in str(raised.value)
    assert len(calls) == 1


def test_incomplete_plan_can_use_one_larger_bounded_repair(monkeypatch,context,plan):
    from gpx_recipe.planner import make_plan
    monkeypatch.setenv('AZURE_ILLUMINE_API_KEY','test-key')
    monkeypatch.delenv('RADAR_GPX_LUNA_MAX_OUTPUT_TOKENS',raising=False)
    caps=[]
    def post(endpoint,headers,body,label):
        caps.append(body['max_output_tokens'])
        if len(caps)==1:
            data=response('{"schema":');data.update(status='incomplete',incomplete_details={'reason':'max_output_tokens'});return data
        return response(json.dumps(plan))
    monkeypatch.setattr(providers,'_post',post)
    result=make_plan('Refine the selected atom',context,client=providers.IllumineLuna())
    assert caps==[2048,4096] and len(result['calls'])==2
    assert result['calls'][0]['finish_reason']=='length'
    assert result['status']=='ready_for_preflight'


def test_luna_is_the_selected_default_with_no_automatic_fallback(monkeypatch):
    monkeypatch.delenv("RADAR_GPX_PLANNER_PROVIDER",raising=False)
    assert providers.provider_info()["provider"]=="illumine-luna"
    assert providers.provider_info()["reasoning_effort"]=="none"
    assert providers.provider_info("sambanova")["purpose"]=="Optional alternative"


def test_transport_timeout_has_safe_diagnostics_without_private_exception_text(monkeypatch):
    class Opener:
        def open(self,request,timeout):raise providers.urllib.error.URLError(TimeoutError('private transport details'))
    monkeypatch.setattr(providers.urllib.request,'build_opener',lambda *a:Opener())
    with pytest.raises(providers.ProviderUnavailable) as raised:providers._post('https://example.invalid',{}, {},'test')
    assert raised.value.cause=='timeout' and raised.value.elapsed_seconds>=0
    assert 'private transport details' not in str(raised.value)


def test_luna_grammar_binds_exact_current_project_without_rewriting_model_targets(monkeypatch,context,plan):
    from gpx_recipe.planner import make_plan, build_payload
    monkeypatch.setenv('AZURE_ILLUMINE_API_KEY','test-key')
    bodies=[]
    def post(endpoint,headers,body,label):
        bodies.append(body)
        return response(json.dumps(plan))
    monkeypatch.setattr(providers,'_post',post)
    client=providers.IllumineLuna()
    result=make_plan('Refine the selected atoms',context,client=client)
    signature=bodies[0]['text']['format']['schema']['properties']['model_signature']
    assert signature=={'type':'string','enum':[context['model_signature']]}
    assert result['plan']==plan
    assert build_payload('Refine the selected atoms',context)['response_schema']['properties']['model_signature']==signature
    changed=dict(context,model_signature='different-project')
    client.prepare_contract(changed['model_signature'])
    client.complete([])
    assert bodies[-1]['text']['format']['schema']['properties']['model_signature']['enum']==['different-project']


def test_provider_error_diagnostic_never_exposes_prose_or_unrecognized_codes():
    assert providers.safe_service_code(io.BytesIO(b'{"error":{"code":"PermissionDenied","message":"private prompt"}}'))=='permissiondenied'
    assert providers.safe_service_code(io.BytesIO(b'{"error":{"code":"private-secret-as-code","message":"private prompt"}}')) is None
    assert providers.safe_service_code(io.BytesIO(b'private non-json error')) is None
