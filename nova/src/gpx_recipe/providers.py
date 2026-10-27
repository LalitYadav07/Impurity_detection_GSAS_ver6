"""Explicit provider selection; development Luna never silently replaces NDIP OSS."""
import json
import os
from pathlib import Path
import stat
import re
import urllib.error
import urllib.request
from urllib.parse import urlsplit

PROVIDERS = {
    "sambanova": {"provider": "sambanova", "label": "OSS 120B", "model": "gpt-oss-120b",
                  "reasoning_effort": "high", "purpose": "NDIP deployment target", "output_format": "validated_json_text"},
    "illumine-luna": {"provider": "illumine-luna", "label": "Illumine Luna", "model": "gpt-6-luna",
                      "reasoning_effort": "none", "purpose": "Development", "output_format": "json_schema"},
}


def provider_info(name=None):
    name = name or os.environ.get("RADAR_GPX_PLANNER_PROVIDER", "sambanova")
    if name not in PROVIDERS:
        raise ValueError("Choose sambanova or illumine-luna; no automatic provider fallback")
    return dict(PROVIDERS[name])


def configure_provider(name=None, key_file=None):
    info = provider_info(name)
    if key_file:
        if info["provider"] != "illumine-luna":
            raise ValueError("The local key-file option is for the Illumine development resource only")
        path = Path(key_file).expanduser().absolute()
        if path.resolve() != path:
            raise ValueError("The development credential path must not contain symlinks")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd) as handle:
            meta = os.fstat(handle.fileno())
            if not stat.S_ISREG(meta.st_mode) or meta.st_mode & 0o077 or meta.st_uid != os.getuid() or meta.st_size > 1024:
                raise ValueError("The development credential must be a small owner-only regular file")
            key = handle.read().strip()
        if not 20 <= len(key) <= 512 or any(c.isspace() for c in key):
            raise ValueError("Invalid development credential; value withheld")
        os.environ["AZURE_ILLUMINE_API_KEY"] = key
    os.environ["RADAR_GPX_PLANNER_PROVIDER"] = info["provider"]
    return info


def _endpoint(value, host, path):
    url = urlsplit(value)
    if url.scheme != "https" or url.hostname != host or url.port not in (None, 443) or url.path.rstrip("/") != path or url.username or url.password or url.query or url.fragment:
        raise ValueError("Provider credentials are restricted to their configured ORNL HTTPS resource")
    return value.rstrip("/")


class ProviderRequestError(RuntimeError):
    def __init__(self, label, status, retry_after=None):
        self.http_status, self.retry_after_seconds = status, retry_after
        wait = f" Retry after {retry_after:g} seconds." if retry_after is not None else ""
        message = f"{label} is rate limited (HTTP 429)." + wait if status == 429 else f"{label} returned HTTP {status}."
        super().__init__(message + " Response content withheld; no automatic retry or provider substitution.")


class ProviderUnavailable(RuntimeError):
    """Transport failure is distinct from an invalid scientific model reply."""


RATE_HEADERS = ("retry-after", "retry-after-ms", "x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens",
                "x-ratelimit-remaining-requests", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests")


def duration(value):
    if value is None: return None
    text = str(value).strip()
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", text): return float(text)
    parts = re.findall(r"([0-9]+(?:\.[0-9]+)?)(ms|s|m|h)", text)
    if not parts or ''.join(a+b for a,b in parts) != text: return None
    return sum(float(a) * {"ms": .001, "s": 1, "m": 60, "h": 3600}[b] for a,b in parts)


def _post(endpoint, headers, body, label):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, response_headers, newurl):
            return None
    request = urllib.request.Request(endpoint, data=json.dumps(body, allow_nan=False).encode(), headers=headers)
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=180) as response:
            data = json.load(response)
            data["_radar_rate_limits"] = {k: response.headers[k] for k in RATE_HEADERS if response.headers.get(k)}
            return data
    except urllib.error.HTTPError as exc:
        status, retry = exc.code, None
        response_headers = {k.lower(): v for k,v in (exc.headers or {}).items()}
        try:
            if response_headers.get("retry-after"):
                retry = duration(response_headers["retry-after"])
            elif response_headers.get("retry-after-ms"):
                retry = float(response_headers["retry-after-ms"]) / 1000
            if status == 429:
                resets = [duration(response_headers.get(k)) for k in ("x-ratelimit-reset-tokens", "x-ratelimit-reset-requests")]
                waits = [x for x in [retry, *resets] if x is not None]
                retry = max(waits) if waits else None
            if retry is not None and not 0 <= retry <= 86400:
                retry = None
        except (ValueError, TypeError):
            retry = None
        exc.close()
        raise ProviderRequestError(label, status, retry) from None
    except (TimeoutError, urllib.error.URLError):
        raise ProviderUnavailable(f"{label} inference could not complete the backend HTTPS request; no plan was accepted or executed") from None


class SambaNova:
    def __init__(self):
        self.info = provider_info("sambanova")
        self.model = self.info["model"]
        self.endpoint = _endpoint(os.environ.get("SAMBANOVA_LLM_BASE_URL", "https://sambanova.cades.ornl.gov/v1"), "sambanova.cades.ornl.gov", "/v1")
        if os.environ.get("SAMBANOVA_LLM_MODEL", self.model) != self.model:
            raise ValueError("The NDIP deployment planner model is gpt-oss-120b")
        self.key = os.environ.get("SAMBANOVA_API_KEY", "")
        if not self.key:
            raise ValueError("SAMBANOVA_API_KEY must be supplied to the backend environment")

    def complete(self, messages):
        body = {"model": self.model, "messages": messages, "max_completion_tokens": 8192,
                "reasoning_effort": self.info["reasoning_effort"], "stream": False}
        data = _post(self.endpoint + "/chat/completions", {"Content-Type": "application/json", "Authorization": "Bearer " + self.key}, body, "SambaNova")
        content = data["choices"][0]["message"].get("content")
        if not isinstance(content, str):
            raise ValueError("SambaNova returned no text plan")
        return content, {**self.info, "model": data.get("model", self.model), "usage": data.get("usage"),
                         "rate_limits": data.get("_radar_rate_limits"), "finish_reason": data["choices"][0].get("finish_reason")}


class IllumineLuna:
    def __init__(self):
        self.info = provider_info("illumine-luna")
        self.model = self.info["model"]
        self.endpoint = _endpoint(os.environ.get("AZURE_ILLUMINE_BASE_URL", "https://openai-eus-illumine.openai.azure.com/openai/v1"), "openai-eus-illumine.openai.azure.com", "/openai/v1")
        self.max_output_tokens = int(os.environ.get("RADAR_GPX_LUNA_MAX_OUTPUT_TOKENS", "2048"))
        if not 1024 <= self.max_output_tokens <= 8192:
            raise ValueError("The Luna output allowance must be 1024–8192 tokens")
        self.key = os.environ.get("AZURE_ILLUMINE_API_KEY", "")
        if not self.key:
            raise ValueError("AZURE_ILLUMINE_API_KEY must be supplied to the backend environment")

    def complete(self, messages):
        from .schema import response_schema
        body = {"model": self.model, "input": messages, "reasoning": {"effort": "none"},
                "store": False, "max_output_tokens": self.max_output_tokens,
                "text": {"format": {"type": "json_schema", "name": "radar_gpx_recipe", "strict": True, "schema": response_schema()}}}
        data = _post(self.endpoint + "/responses", {"Content-Type": "application/json", "api-key": self.key}, body, "Illumine Luna")
        if data.get("model") != self.model:
            raise ValueError("Illumine returned a different model identity; no plan was accepted")
        content = []
        for item in data.get("output", []):
            if item.get("type") != "message":
                continue  # Never persist hidden reasoning or encrypted provider state.
            for part in item.get("content", []):
                if part.get("type") == "refusal":
                    raise ValueError("Illumine Luna refused the request; no plan was accepted")
                if part.get("type") == "output_text":
                    content.append(part.get("text", ""))
        if data.get("status") == "incomplete" and data.get("incomplete_details", {}).get("reason") == "max_output_tokens":
            used_budget = self.max_output_tokens
            self.max_output_tokens = min(8192, self.max_output_tokens * 2)
            raise IncompleteResponse("".join(content), {**self.info, "usage": data.get("usage"), "rate_limits": data.get("_radar_rate_limits"), "finish_reason": "length", "output_allowance": used_budget})
        if data.get("status") != "completed":
            raise ValueError("Illumine Luna did not complete a full response; no plan was accepted")
        if not content or not all(isinstance(text, str) for text in content):
            raise ValueError("Illumine Luna returned no text plan")
        return "".join(content), {**self.info, "usage": data.get("usage"), "output_allowance": self.max_output_tokens, "rate_limits": data.get("_radar_rate_limits"), "finish_reason": "stop"}


def make_client():
    return IllumineLuna() if provider_info()["provider"] == "illumine-luna" else SambaNova()


class IncompleteResponse(ValueError):
    def __init__(self, text, metadata):
        super().__init__("Provider output allowance was exhausted before completing the plan")
        self.text, self.metadata = text, metadata
