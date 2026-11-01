"""NOVA planning context over the unchanged, worker-compatible native planner."""
from copy import deepcopy
import json
import math

from gpx_recipe import planner as native
from gpx_recipe.snapshot import digest

ADAPTER_VERSION = "nova-refinement-planner/v1"
MAX_PAYLOAD_BYTES = 350000
HOST_INSTRUCTION = (
    "Capabilities explicitly listed in project.host_execution are managed by the host, outside "
    "the per-GPX recipe. Do not mark those listed capabilities unsupported or invent refinement "
    "targets for them; host approval and compatibility checks still apply. Do not assume unlisted "
    "host capabilities. Follow fit_range_instruction and the explicit imported versus current "
    "histogram bounds. Keep genuinely unsupported operations and missing decisions blocked."
)
FIT_RANGE_INSTRUCTION = (
    "native_data_range (GSAS Limits[0]) gives the imported native coordinate endpoints; "
    "current_fit_range (Limits[1]) is the currently selected fit interval, which may be narrower. "
    "Never describe a restricted current fit as the full imported/valid range. When the user explicitly "
    "requests the full imported/native/valid range and it differs from the current fit, include the exact "
    "full_imported_range_edit in fit_ranges at the first requested stage; later stages inherit it unless "
    "the user requests another change. Preserve native weights, validity masks and existing exclusions "
    "unless their change is explicitly requested. Imported endpoints do not prove every bin is valid. "
    "If the imported bounds or a supported edit are unavailable, ask a concrete question instead of "
    "inventing bounds or substituting the current fit."
)


def _host_execution(value, histogram_keys):
    if not isinstance(value, dict):
        return {}
    result = {key: value[key] for key in ("recipe_unit", "scan_enrollment")
              if isinstance(value.get(key), str)}
    if isinstance(value.get("histogram_keys"), list):
        result["histogram_keys"] = [key for key in histogram_keys if key in value["histogram_keys"]]
    for name, fields in {
        "future_enrollment": ("supported", "requires_approval"),
        "output_artifacts": ("stage_gpx", "final_gpx", "source_plan_lineage"),
        "condition_mapping": ("preserve_reviewed_temperature_axis",),
    }.items():
        record = value.get(name)
        if not isinstance(record, dict):
            continue
        projected = {key: record[key] for key in fields if type(record.get(key)) is bool}
        if isinstance(record.get("scope"), str):
            projected["scope"] = record["scope"]
        if projected:
            result[name] = projected
    return result


def _range_pair(value):
    if (isinstance(value, (list, tuple)) and len(value) == 2
            and all(type(v) in (int, float) and math.isfinite(v) for v in value)
            and value[0] < value[1]):
        return list(value)
    return None


def _histogram_ranges(histogram):
    limits = histogram.get("limits")
    limits = limits if isinstance(limits, (list, tuple)) else []
    imported = _range_pair(limits[0]) if limits else None
    current = _range_pair(limits[1]) if len(limits) > 1 else None
    exclusions = [_range_pair(pair) for pair in limits[2:]] if len(limits) >= 2 else None
    edit = (dict(histogram=histogram["key"], unit="native", lower=imported[0], upper=imported[1])
            if imported is not None and imported[0] > 0 else None)
    points = histogram.get("point_count")
    return dict(native_data_range=imported, current_fit_range=current,
                current_exclusions=exclusions if exclusions is not None and all(p is not None for p in exclusions) else None,
                point_count=points if type(points) is int and points >= 0 else None,
                fit_range_unit="native", full_imported_range_edit=edit,
                current_fit_covers_imported_range=(current[0] <= imported[0] and current[1] >= imported[1])
                    if imported is not None and current is not None else None)


def _enrich(payload, context):
    payload = deepcopy(payload)
    project = native.expand_context(payload["project"])
    host = _host_execution(context.get("host_execution"), [h["key"] for h in context["histograms"]])
    if host:
        project["host_execution"] = host
    # The native hash binds its original projection. Rebind the exact enriched
    # projection and retain the original separately in the planning receipt.
    project.pop("context_sha256", None)
    project["context_sha256"] = digest(project)
    payload["project"] = native.compact_context(project)
    histograms = {h["key"]: h for h in context["histograms"]}
    for capability in payload["histogram_capabilities"]:
        capability.update(_histogram_ranges(histograms[capability["histogram"]]))
    payload["fit_range_instruction"] = FIT_RANGE_INSTRUCTION
    if len(json.dumps(payload, allow_nan=False).encode()) > MAX_PAYLOAD_BYTES:
        raise ValueError("Project context exceeds the validated planner budget; do not silently truncate scientific state")
    return payload


def build_payload(request, context, previous=None):
    return _enrich(native.build_payload(request, context, previous), context)


class _PlanningClient:
    def __init__(self, client, base, payload):
        self.client, self.base, self.payload = client, base, payload
        self.requests = []

    def _provider(self):
        if self.client is None:
            self.client = native.make_client()
        return self.client

    def __getattr__(self, name):
        # Preserve optional provider contracts, including clients predating
        # prepare_contract/prepare_output. No native module globals change.
        return getattr(self._provider(), name)

    def complete(self, messages):
        outbound = deepcopy(messages)
        base = json.loads(outbound[1]["content"])
        if digest(base) != digest(self.base):
            raise ValueError("Native planning payload changed before host enrichment")
        outbound[0]["content"] += " " + HOST_INSTRUCTION
        outbound[1]["content"] = json.dumps(self.payload, allow_nan=False)
        repair_context = {key: self.payload[key] for key in (
            "target_capabilities", "histogram_capabilities", "stage_instruction_cards", "fit_range_instruction")}
        repair_context["host_execution"] = self.payload["project"].get("host_execution", {})
        for message in outbound[2:]:
            if message["role"] != "user":
                continue
            # Both native repair routes retain their original instruction and
            # receive the same authoritative host/range context.
            head, separator, tail = message["content"].rpartition("\n")
            try:
                old = json.loads(tail) if separator else None
            except ValueError:
                old = None
            if isinstance(old, dict) and "target_capabilities" in old:
                message["content"] = head
            message["content"] += "\n" + HOST_INSTRUCTION + "\n" + json.dumps(repair_context, allow_nan=False)
        request = dict(messages_sha256=digest(outbound),
                       messages_bytes=len(json.dumps(outbound, allow_nan=False).encode()),
                       payload_sha256=digest(self.payload))
        self.requests.append(request)
        try:
            raw, metadata = self._provider().complete(outbound)
        except Exception as exc:
            if isinstance(getattr(exc, "metadata", None), dict):
                exc.metadata = {**exc.metadata, **request}
            raise
        return raw, {**metadata, **request}


def make_plan(request, context, *, previous=None, client=None, max_calls=2):
    context = deepcopy(context)
    base = native.build_payload(request, context, previous)
    payload = _enrich(base, context)
    wrapped = _PlanningClient(client, base, payload)
    provenance = dict(host_planner_adapter=ADAPTER_VERSION, base_payload_sha256=digest(base),
                      payload_sha256=digest(payload), payload_bytes=len(json.dumps(payload, allow_nan=False).encode()))
    try:
        result = native.make_plan(request, context, previous=previous, client=wrapped, max_calls=max_calls)
    except Exception as exc:
        if isinstance(exc, native.PlanningValidationError):
            exc.base_payload_sha256, exc.payload_sha256 = digest(base), digest(payload)
            exc.planning_calls = exc.calls
        exc.planning_provenance = {**provenance, "outbound_requests": deepcopy(wrapped.requests)}
        raise
    return {**result, **provenance, "outbound_requests": wrapped.requests}
