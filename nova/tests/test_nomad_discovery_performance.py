"""Discovery remains a scoped, read-only preview before native verification."""
import asyncio
import json
import time

import pytest

from test_live_assistant import assistant
from radar_pd_nova import nomad, powgen_samples, sample_catalog
from radar_pd_nova.facility_metadata import ExperimentMetadataIndex
from radar_pd_nova.nomad_controller import discover_nomad_samples
from radar_pd_nova.nomad_metadata import NomadSampleIndex
from radar_pd_nova.powgen_samples import PowgenSampleIndex, select_runs
from radar_pd_nova.powgen_watch import WatchDefinition, WatchedRun


def indexed_row(run=42, sample_id="00123"):
    return {"location": f"/SNS/NOM/IPTS-1/nexus/NOM_{run}.nxs.h5",
            "indexed": {"run_number": run, "sample_id": sample_id}}


def parse(rows):
    return sample_catalog.parse_sample_catalog(rows, "NOM", "IPTS-1")


def cache_catalog(monkeypatch, rows):
    monkeypatch.setitem(sample_catalog._CACHE, ("NOM", "IPTS-1"), (time.monotonic(), rows))


def index_at(tmp_path, *, nomad_adapter=False):
    index = NomadSampleIndex("IPTS-1") if nomad_adapter else PowgenSampleIndex("IPTS-1", instrument="NOM")
    index.metadata_index = ExperimentMetadataIndex("NOM", "IPTS-1", root=tmp_path)
    return index


def test_public_index_preserves_exact_identifiers_and_does_not_invent_names():
    result = parse([indexed_row(), indexed_row(43, 9007199254740993)])
    assert result[42]["sample_id"] == "00123"
    assert result[43]["sample_id"] == "9007199254740993"
    assert result[42]["sample_name"] == result[42]["run_title"] == ""
    assert result[42]["identity_verification"] == "catalog_preview"
    assert result[42]["identity_sources"] == [{
        "value": "00123", "path": "/SNS/NOM/IPTS-1/nexus/NOM_42.nxs.h5",
        "kind": "oncat_index", "field": "indexed.sample_id"}]


def test_anonymous_request_explicitly_projects_indexed_ids_and_retains_ttl(monkeypatch):
    sample_catalog._CACHE.clear()
    calls = []
    class Response:
        headers = {}
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def raise_for_status(self): pass
        def iter_content(self, _): yield json.dumps([indexed_row()]).encode()
    def get(url, **kwargs):
        calls.append(kwargs)
        assert "headers" not in kwargs and "auth" not in kwargs
        assert kwargs["timeout"] == (5, 25)
        assert ("projection", "indexed.run_number") in kwargs["params"]
        assert ("projection", "indexed.sample_id") in kwargs["params"]
        return Response()
    monkeypatch.setattr(sample_catalog.requests, "get", get)
    try:
        assert sample_catalog.load_sample_catalog("NOM", "IPTS-1")[42]["sample_id"] == "00123"
        assert sample_catalog.load_sample_catalog("NOM", "IPTS-1")[42]["sample_id"] == "00123"
        assert len(calls) == 1
    finally:
        sample_catalog._CACHE.clear()


@pytest.mark.parametrize("mutation", [
    {"location": "/SNS/NOM/IPTS-2/nexus/NOM_42.nxs.h5"},
    {"location": "/SNS/PG3/IPTS-1/nexus/PG3_42.nxs.h5"},
    {"location": "/SNS/NOM/IPTS-1/0/43/NeXus/NOM_42_event.nxs"},
    {"indexed": {"run_number": 43, "sample_id": "00123"}},
    {"indexed": {"run_number": True, "sample_id": "00123"}},
    {"indexed": {"run_number": 42, "sample_id": 123.0}},
    {"indexed": {"run_number": 42, "sample_id": True}},
    {"indexed": {"run_number": 42, "sample_id": {"value": "00123"}}},
])
def test_index_is_bound_to_exact_raw_path_run_and_identifier_schema(mutation):
    row = indexed_row()
    row.update(mutation)
    assert not parse([row])


def test_raw_aliases_and_duplicate_catalog_rows_cannot_hide_index_conflicts():
    row = indexed_row(sample_id="A")
    row["metadata"] = {"entry": {"sample": {"identifier": "B"}}}
    assert parse([row])[42]["identity_status"] == "conflict"
    assert parse([indexed_row(sample_id="A"), indexed_row(sample_id="B")])[42]["identity_status"] == "conflict"


def test_individual_nomad_discovery_only_reads_representative_raw_files(tmp_path, monkeypatch):
    runs = [WatchedRun(n, str(tmp_path/f"NOM_{n}.gsa"), str(n), instrument="NOM") for n in range(1, 1975)]
    rows = parse([indexed_row(n, "A" if n % 2 else "B") for n in range(1, 1975)])
    index = index_at(tmp_path, nomad_adapter=True)
    monkeypatch.setattr(WatchDefinition, "source_directory", property(lambda self: str(tmp_path)))
    monkeypatch.setattr(nomad, "directory_listing", lambda *_: [])
    monkeypatch.setattr(nomad, "source_options", lambda *_: [])
    monkeypatch.setattr("radar_pd_nova.nomad_controller.discover_from_listing", lambda *_: runs)
    cache_catalog(monkeypatch, rows)
    raw_calls = []
    def raw(run):
        raw_calls.append(run.run_number)
        sample = "A" if run.run_number % 2 else "B"
        return {"sample_id": sample, "sample_name": "Recorded "+sample, "identity_status": "resolved"}
    monkeypatch.setattr(index, "identity", raw)
    monkeypatch.setattr(powgen_samples, "run_role", lambda *_: {"run_title": "", "run_role": "unclassified"})
    progress = []
    result = discover_nomad_samples("IPTS-1", index, progress=lambda *args: progress.append(args))
    assert result["scan_count"] == 1974 and result["catalog_preview_count"] == 1970
    assert {row["value"]: row["scan_count"] for row in result["options"]} == {"A": 987, "B": 987}
    assert sorted(raw_calls) == [1, 2, 1973, 1974]
    assert all("Recorded" in row["title"] and "representative name" in row["title"] for row in result["options"])
    assert not index._cache
    assert progress[-1] == (1974, 1974)


def test_missing_conflicting_and_unset_catalog_ids_use_raw_evidence(tmp_path, monkeypatch):
    rows = parse([indexed_row(1, "A"), indexed_row(2, None),
                  indexed_row(3, "A"), indexed_row(3, "B"),
                  indexed_row(4, 4294967295)])
    cache_catalog(monkeypatch, rows)
    index = index_at(tmp_path)
    calls = []
    def raw(run):
        calls.append(run.run_number)
        return {"sample_id": "4294967295" if run.run_number == 4 else "RAW",
                "identity_status": "resolved", "sample_name": "NoSample" if run.run_number == 4 else "Recorded"}
    monkeypatch.setattr(index, "identity", raw)
    monkeypatch.setattr(powgen_samples, "run_role", lambda *_: {"run_title": "", "run_role": "unclassified"})
    result = index.inventory([WatchedRun(n, f"NOM_{n}.gsa", str(n), instrument="NOM") for n in range(1, 6)], catalog=True)
    assert sorted(calls) == [2, 3, 4, 5]
    assert result["catalog_preview_count"] == 1
    assert next(x for x in result["options"] if x["value"] == "4294967295")["title"].startswith("4294967295 — NoSample")


def test_catalog_hint_never_replaces_selected_source_check_or_filters_unknown_runs(tmp_path, monkeypatch):
    rows = parse([indexed_row(1, "A"), indexed_row(2, "B"), indexed_row(3, "A")])
    cache_catalog(monkeypatch, rows)
    index = index_at(tmp_path)
    runs = [WatchedRun(n, f"NOM_{n}.gsa", str(n), instrument="NOM") for n in range(1, 5)]
    monkeypatch.setattr(index, "identity", lambda *_: {"sample_id": "A", "identity_status": "resolved"})
    monkeypatch.setattr(powgen_samples, "run_role", lambda *_: {"run_title": "", "run_role": "unclassified"})
    index.inventory(runs, catalog=True)
    calls = []
    def raw(run):
        calls.append(run.run_number)
        return {"sample_id": "B" if run.run_number == 3 else "A", "identity_status": "resolved"}
    monkeypatch.setattr(index, "identity", raw)
    monkeypatch.setattr(powgen_samples, "include_run", lambda *a, **k: True)
    # Unknown run4 remains eligible; row3 is independently disproved before
    # selection. A known B preview can prune row2 without claiming calibration.
    selected = select_runs(runs, index, ("A",), limit=2)
    assert calls == [4, 3, 1]
    assert [x.run_number for x in selected] == [1, 4]
    changed = WatchedRun(2, "NOM_2.gsa", "replacement", instrument="NOM")
    assert index.candidate_runs([changed], ("A",)) == [changed]


def test_expired_or_metadata_changed_index_never_prunes_from_stale_identity(tmp_path, monkeypatch):
    index = index_at(tmp_path)
    run = WatchedRun(42, "NOM_42.gsa", "original", instrument="NOM")
    current = parse([indexed_row(42, "B")])
    cache_catalog(monkeypatch, current)
    index.inventory([run], catalog=True)
    assert index.candidate_runs([run], ("A",)) == []
    # An independent table generation change invalidates the pruning hint.
    index.metadata_index.generation += 1
    assert index.candidate_runs([run], ("A",)) == [run]
    index.inventory([run], catalog=True)
    signature, metadata = index._preview_cache[42]
    index._preview_cache[42] = ((*signature[:3], signature[3] - sample_catalog.TTL_SECONDS - 1), metadata)
    current.clear()
    current.update(parse([indexed_row(42, "A")]))
    assert index.candidate_runs([run], ("A",)) == [run]


def test_cached_response_age_is_not_extended_by_a_later_inventory(tmp_path, monkeypatch):
    rows = parse([indexed_row(42, "A")])
    created = time.monotonic() - sample_catalog.TTL_SECONDS + 5
    monkeypatch.setitem(sample_catalog._CACHE, ("NOM", "IPTS-1"), (created, rows))
    monkeypatch.setattr(sample_catalog.requests, "get", lambda *a, **k: pytest.fail("Valid cached response was fetched again"))
    index = index_at(tmp_path)
    run = WatchedRun(42, "NOM_42.gsa", "f", instrument="NOM")
    index.inventory([run], catalog=True)
    assert index._preview_cache[42][0][3] == created


def test_replaced_cache_cannot_make_older_returned_rows_fresh(tmp_path, monkeypatch):
    old = parse([indexed_row(42, "B")])
    fresh = parse([indexed_row(42, "A")])
    calls = []
    real_load = sample_catalog.load_sample_catalog
    def interleaved_load(*args):
        calls.append(args)
        if len(calls) == 1:
            # Another request publishes a replacement after this caller got
            # the old response but before it asks for that response's age.
            monkeypatch.setitem(sample_catalog._CACHE, ("NOM", "IPTS-1"), (time.monotonic(), fresh))
            return old
        return real_load(*args)
    monkeypatch.setattr(powgen_samples, "load_sample_catalog", interleaved_load)
    index = index_at(tmp_path)
    run = WatchedRun(42, "NOM_42.gsa", "f", instrument="NOM")
    index.inventory([run], catalog=True)
    assert index._preview_cache[42][0][3] <= time.monotonic() - sample_catalog.TTL_SECONDS
    assert index.candidate_runs([run], ("A",)) == [run]
    assert len(calls) == 2


def test_fresh_source_fallback_uses_its_own_observation_time(tmp_path, monkeypatch):
    rows = parse([indexed_row(42, None)])
    created = time.monotonic() - sample_catalog.TTL_SECONDS + 5
    monkeypatch.setitem(sample_catalog._CACHE, ("NOM", "IPTS-1"), (created, rows))
    index = index_at(tmp_path)
    monkeypatch.setattr(index, "identity", lambda *_: {"sample_id": "A", "identity_status": "resolved"})
    monkeypatch.setattr(powgen_samples, "run_role", lambda *_: {"run_title": "", "run_role": "unclassified"})
    run = WatchedRun(42, "NOM_42.gsa", "f", instrument="NOM")
    index.inventory([run], catalog=True)
    assert index._preview_cache[42][0][3] > created + 100


def test_representative_name_variation_is_explicit_without_assigning_names_to_unread_runs(tmp_path, monkeypatch):
    cache_catalog(monkeypatch, parse([indexed_row(n, "A") for n in (1, 2, 3)]))
    index = index_at(tmp_path, nomad_adapter=True)
    calls = []
    def raw(run):
        calls.append(run.run_number)
        return {"sample_id": "A", "sample_name": "First label" if run.run_number == 1 else "Last label", "identity_status": "resolved"}
    monkeypatch.setattr(index, "identity", raw)
    monkeypatch.setattr(powgen_samples, "run_role", lambda *_: {"run_title": "", "run_role": "unclassified"})
    result = index.inventory([WatchedRun(n, f"NOM_{n}.gsa", str(n), instrument="NOM") for n in (1, 2, 3)], catalog=True)
    assert sorted(calls) == [1, 3]
    option = result["options"][0]
    assert option["name_variation_observed"]
    assert option["name_run_numbers"] == [1, 3]
    assert option["name_scope"] == "source-checked representatives"
    assert "sampled names differ" in option["title"]
    assert result["metadata_resolution"][1]["verification"] == "catalog_preview"


def test_representative_mismatch_rechecks_the_entire_indexed_group(tmp_path, monkeypatch):
    previews = [indexed_row(n, "A" if n in (1, 3, 4) else "B") for n in range(1, 6)]
    cache_catalog(monkeypatch, parse(previews))
    index = index_at(tmp_path, nomad_adapter=True)
    calls = []
    def raw(run):
        calls.append(run.run_number)
        return {"sample_id": "A" if run.run_number in (1, 3) else "B", "identity_status": "resolved", "sample_name": "Recorded"}
    monkeypatch.setattr(index, "identity", raw)
    monkeypatch.setattr(powgen_samples, "run_role", lambda *_: {"run_title": "", "run_role": "unclassified"})
    result = index.inventory([WatchedRun(n, f"NOM_{n}.gsa", str(n), instrument="NOM") for n in range(1, 6)], catalog=True)
    assert sorted(calls) == [1, 2, 3, 4, 5]
    assert result["catalog_preview_count"] == 0
    assert {x["value"]:x["scan_count"] for x in result["options"]} == {"A": 2, "B": 3}
    assert any("disagreed" in n for n in result["metadata_notices"])


def test_representative_cap_does_not_limit_sample_ids_or_repeat_one_source_path(tmp_path, monkeypatch):
    rows = parse([indexed_row(n, sample) for n, sample in [(1, "A"), (2, "A"), (3, "B"), (4, "B"), (5, "C")]])
    cache_catalog(monkeypatch, rows)
    monkeypatch.setattr(powgen_samples, "MAX_CATALOG_NAME_REPRESENTATIVES", 2)
    index = index_at(tmp_path, nomad_adapter=True)
    calls = []
    def raw(run):
        calls.append(run.source_path)
        return {"sample_id": rows[run.run_number]["sample_id"], "sample_name": "Recorded", "identity_status": "resolved"}
    monkeypatch.setattr(index, "identity", raw)
    monkeypatch.setattr(powgen_samples, "run_role", lambda *_: {"run_title": "", "run_role": "unclassified"})
    runs = [WatchedRun(n, f"NOM_{n if n != 2 else 1}.gsa", str(n), instrument="NOM") for n in range(1, 6)]
    result = index.inventory(runs, catalog=True)
    assert len(calls) == len(set(calls)) == 2
    assert result["scan_count"] == 5
    assert {x["value"] for x in result["options"]} == {"A", "B", "C"}
    assert "Name not reported" in next(x["title"] for x in result["options"] if x["value"] == "C")


def test_unreported_representative_does_not_erase_a_verified_cached_name(tmp_path, monkeypatch):
    cache_catalog(monkeypatch, parse([indexed_row(n, "A") for n in (1, 2, 3)]))
    index = index_at(tmp_path, nomad_adapter=True)
    index._cache[2] = ((), {})  # Trigger the existing verified-cache path.
    monkeypatch.setattr(index, "identity", lambda run: {
        "sample_id": "A", "identity_status": "resolved", "sample_name": "Known name" if run.run_number == 2 else ""})
    monkeypatch.setattr(powgen_samples, "run_role", lambda *_: {"run_title": "", "run_role": "unclassified"})
    result = index.inventory([WatchedRun(n, f"NOM_{n}.gsa", str(n), instrument="NOM") for n in (1, 2, 3)], catalog=True)
    assert "Known name" in result["options"][0]["title"]


@pytest.mark.parametrize("kind", ["slice", "sum"])
def test_derived_products_never_use_a_parent_catalog_hint(tmp_path, monkeypatch, kind):
    index = index_at(tmp_path, nomad_adapter=True)
    if kind == "sum":
        path = tmp_path/"multi_banks_summed/GSAS/NOM_sum.gsa"
        sidecar = path.parent.parent/"Input/NOM_sum.json"
        sidecar.parent.mkdir(parents=True)
        sidecar.write_text(json.dumps({"Sample": {"Runs": "41,42"}}))
        product_id = nomad.product_from_path(path).product_id
    else:
        path = tmp_path/"sliced/reduced/GSAS/nom_42_300.h5.gsa"
        product_id = "SLICE_300"
    raw_calls = []
    def raw(_self, run):
        raw_calls.append(run.run_number)
        return {"sample_id": "A" if run.run_number == 42 else "B", "identity_status": "resolved"}
    monkeypatch.setattr(PowgenSampleIndex, "identity", raw)
    monkeypatch.setattr(powgen_samples, "load_sample_catalog", lambda *_: pytest.fail("Derived product trusted parent catalog"))
    monkeypatch.setattr(powgen_samples, "run_role", lambda *_: {"run_title": "", "run_role": "unclassified"})
    result = index.inventory([WatchedRun(42, str(path), "f", instrument="NOM", product_id=product_id)], catalog=True)
    assert result["catalog_preview_count"] == 0
    assert raw_calls == ([41, 42] if kind == "sum" else [42])
    assert result["unidentified_count"] == (1 if kind == "sum" else 0)


def test_partial_catalog_response_falls_back_to_every_source_without_hiding_foreign_samples(tmp_path, monkeypatch):
    class Response:
        headers = {"Link": '<https://oncat.ornl.gov/api/datafiles?cursor=x>; rel="next"'}
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def raise_for_status(self): pass
        def iter_content(self, _): yield json.dumps([indexed_row(1, "A")]).encode()
    sample_catalog._CACHE.clear()
    monkeypatch.setattr(sample_catalog.requests, "get", lambda *a, **k: Response())
    index = index_at(tmp_path)
    calls = []
    def raw(run):
        calls.append(run.run_number)
        return {"sample_id": "A" if run.run_number == 1 else "B", "identity_status": "resolved"}
    monkeypatch.setattr(index, "identity", raw)
    monkeypatch.setattr(powgen_samples, "run_role", lambda *_: {"run_title": "", "run_role": "unclassified"})
    result = index.inventory([WatchedRun(n, f"NOM_{n}.gsa", str(n), instrument="NOM") for n in (1, 2)], catalog=True)
    assert sorted(calls) == [1, 2]
    assert {x["value"] for x in result["options"]} == {"A", "B"}
    assert result["catalog_preview_count"] == 0
    assert not sample_catalog._CACHE


@pytest.mark.parametrize("folders", [
    [{"title": "Legacy individual reductions", "value": "shared/autoreduce"}],
    [{"title": "Legacy individual reductions", "value": "shared/autoreduce"},
     {"title": "IDL reductions · autoNOM", "value": "shared/autoNOM/GSAS"}],
])
def test_unavailable_default_folder_is_a_choice_not_a_failed_experiment(assistant, monkeypatch, folders):
    state = assistant.server.state
    state.live_beamline = "NOM"
    state.powgen_source_subfolder = "shared/autoreduce/multi_banks/GSAS"
    state.flush()

    async def discover(ipts=None):
        state.powgen_sample_discovery_ipts = state.powgen_ipts
        state.powgen_source_options = folders
        state.powgen_discovered_sample_options = []
        state.powgen_sample_message = "The selected reduction folder is unavailable. Choose one of the available reduction families, then refresh sample IDs."
        return {"source_unavailable": True, "source_options": folders}

    assistant.refresh_powgen_samples = discover
    monkeypatch.setattr("radar_pd_nova.live_assistant_controller.inspect_inventory",
                        lambda *a, **k: pytest.fail("Missing family started raw/bank inspection"))
    asyncio.run(assistant.live_assistant_continue())
    assert state.live_assistant_step == "experiment"
    assert state.live_assistant_error == ""
    assert state.powgen_source_options == folders
    assert state.powgen_source_subfolder == "shared/autoreduce/multi_banks/GSAS"
    assert "Choose one" in state.live_assistant_messages[-1]["text"]
    assert "experiment" not in assistant._live_conversation.confirmed
    assert not state.live_assistant_busy
    assert not assistant.service.jobs
