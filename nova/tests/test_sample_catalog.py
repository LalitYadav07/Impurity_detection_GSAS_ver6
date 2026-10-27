from copy import deepcopy
from types import SimpleNamespace

import pytest

from radar_pd_nova import sample_catalog, powgen_samples
from radar_pd_nova.facility_metadata import ExperimentMetadataIndex
from radar_pd_nova.powgen_samples import PowgenSampleIndex, select_runs
from radar_pd_nova.powgen_watch import WatchedRun


def record(run=42, identifier="00123", **extra):
    return {"location": f"/SNS/PG3/IPTS-1/nexus/PG3_{run}.nxs.h5", "metadata": {"entry": {
        "run_number": str(run), "experiment_identifier": "IPTS-1", "instrument": {"name": "POWGEN3"},
        "title": "sample at 300 K", "sample": {"identifier": identifier, "name": "Copper"}, **extra}}}


def parse(rows):
    return sample_catalog.parse_sample_catalog(rows, "PG3", "IPTS-1")


def test_catalog_is_scoped_to_matching_raw_run_proposal_and_instrument():
    good = record()
    variants = []
    for key, value in (("run_number", "43"), ("experiment_identifier", "IPTS-2"), ("instrument", {"name": "NOMAD"})):
        row = deepcopy(good); row["metadata"]["entry"][key] = value; variants.append(row)
    for path in ("/SNS/PG3/IPTS-2/nexus/PG3_42.nxs.h5", "/SNS/PG3/IPTS-1/0/43/NeXus/PG3_42_event.nxs", "/SNS/PG3/IPTS-1/shared/PG3_42.gsa"):
        row = deepcopy(good); row["location"] = path; variants.append(row)
    assert not parse(variants)
    result = parse([good])[42]
    assert result["sample_id"] == "00123" and result["sample_name"] == "Copper"
    assert result["identity_verification"] == "catalog_preview"
    assert all(e["kind"] == "oncat_catalog" for e in result["identity_sources"])


def test_catalog_keeps_alias_within_run_and_duplicate_file_conflicts():
    first = record(identifier="123")
    second = record(identifier="124")
    second["location"] = "/SNS/PG3/IPTS-1/data/PG3_42_event.nxs"
    assert parse([first, second])[42]["identity_status"] == "conflict"
    first["metadata"]["entry"]["DASlogs"] = {"SampleId": {"value": [123, 124, 123]}}
    assert parse([first])[42]["identity_status"] == "conflict"


def test_known_unidentified_catalog_run_is_not_inferred_from_its_title():
    row = record(identifier="NONE", title="Sample 123")
    assert parse([row])[42]["identity_status"] == "unidentified"
    assert not parse([row])[42]["sample_id"]
    row["metadata"]["entry"]["sample"]["identifier"] = [123] * 1001
    assert not parse([row])


def test_large_inventory_does_not_open_thousands_of_raw_or_reduced_files(tmp_path, monkeypatch):
    rows = parse([record(run=n, identifier="A" if n % 2 else "B") for n in range(1, 3230)])
    monkeypatch.setattr(powgen_samples, "load_sample_catalog", lambda *args: rows)
    monkeypatch.setattr(powgen_samples, "run_role", lambda *args: pytest.fail("Browsing opened a GSA"))
    index = PowgenSampleIndex("IPTS-1", metadata_index=ExperimentMetadataIndex("PG3", "IPTS-1", root=tmp_path))
    monkeypatch.setattr(index, "identity", lambda *args: pytest.fail("Browsing opened a raw file"))
    runs = [WatchedRun(n, f"PG3_{n}.gsa", str(n)) for n in range(1, 3230)]
    result = index.inventory(runs, catalog=True)
    assert result["scan_count"] == result["catalog_preview_count"] == 3229
    assert [o["scan_count"] for o in result["options"]] == [1615, 1614]
    assert not index._cache  # Preview results never satisfy a scientific identity check.


def test_only_selected_latest_runs_get_raw_verified_and_mismatches_are_excluded(tmp_path, monkeypatch):
    rows = parse([record(run=n, identifier="A" if n % 2 else "B") for n in range(1, 1001)])
    monkeypatch.setattr(powgen_samples, "load_sample_catalog", lambda *args: rows)
    index = PowgenSampleIndex("IPTS-1", metadata_index=ExperimentMetadataIndex("PG3", "IPTS-1", root=tmp_path))
    runs = [WatchedRun(n, f"PG3_{n}.gsa", str(n)) for n in range(1, 1001)]
    index.inventory(runs, catalog=True)
    calls = []
    def verify(run):
        calls.append(run.run_number)
        return {"sample_id": "B" if run.run_number == 999 else "A"}
    monkeypatch.setattr(index, "identity", verify)
    monkeypatch.setattr(powgen_samples, "include_run", lambda *args, **kwargs: True)
    stats = {}
    selected = select_runs(runs, index, ("A",), limit=2, selection_stats=stats)
    assert [r.run_number for r in selected] == [995, 997]
    assert calls == [999, 997, 995]
    assert 999 in stats["identity_review"]
    # A replaced reduction cannot reuse its former catalog hint.
    replacement = WatchedRun(1000, "PG3_1000.gsa", "replacement")
    assert index.candidate_runs([replacement], ("A",)) == [replacement]


def test_latest_backfill_counts_only_eligible_runs_and_avoids_other_headers(monkeypatch):
    runs = [WatchedRun(n, f"PG3_{n}.gsa", str(n)) for n in range(1, 1001)]
    inspected = []
    def include(run, **kwargs):
        inspected.append(run.run_number)
        return run.run_number != 999
    monkeypatch.setattr(powgen_samples, "include_run", include)
    result = select_runs(runs, None, None, limit=2)
    assert [r.run_number for r in result] == [998, 1000]
    assert inspected == [1000, 999, 998]


def test_verified_source_identity_is_not_overwritten_by_a_catalog_preview(tmp_path, monkeypatch):
    raw = tmp_path / "raw.h5"
    raw.write_bytes(b"fixture with independent reader")
    index = PowgenSampleIndex("IPTS-1", nexus_path=lambda *args: raw,
        identity_reader=lambda *args: {"sample_id": "B"},
        metadata_index=ExperimentMetadataIndex("PG3", "IPTS-1", root=tmp_path))
    run = WatchedRun(42, "PG3_42.gsa", "unchanged")
    assert index.identity(run)["sample_id"] == "B"
    monkeypatch.setattr(powgen_samples, "load_sample_catalog", lambda *args: parse([record(identifier="A")]))
    result = index.inventory([run], catalog=True)
    assert [o["value"] for o in result["options"]] == ["B"]
    assert not result["catalog_preview_count"]


def test_catalog_outage_preserves_the_independent_source_reader(tmp_path, monkeypatch):
    def unavailable(*args): raise OSError("Catalog unavailable")
    monkeypatch.setattr(powgen_samples, "load_sample_catalog", unavailable)
    index = PowgenSampleIndex("IPTS-1", metadata_index=ExperimentMetadataIndex("PG3", "IPTS-1", root=tmp_path))
    calls = []
    def identity(run):
        calls.append(run.run_number)
        return {"sample_id": "A"}
    monkeypatch.setattr(index, "identity", identity)
    result = index.inventory([WatchedRun(42, "PG3_42.gsa", "42")], catalog=True)
    assert calls == [42] and result["options"][0]["value"] == "A"
    assert "catalog is unavailable" in result["metadata_notices"][0]


def test_public_request_has_no_credentials_and_reuses_a_complete_response(monkeypatch):
    import json
    sample_catalog._CACHE.clear()
    calls = []
    class Response:
        headers = {}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_content(self, size): yield json.dumps([record()]).encode()
    def get(url, **kwargs):
        calls.append(kwargs)
        assert "headers" not in kwargs and "auth" not in kwargs
        assert ("experiment", "IPTS-1") in kwargs["params"]
        return Response()
    monkeypatch.setattr(sample_catalog.requests, "get", get)
    assert sample_catalog.load_sample_catalog("PG3", "IPTS-1")[42]["sample_id"] == "00123"
    assert sample_catalog.load_sample_catalog("PG3", "IPTS-1")[42]["sample_id"] == "00123"
    assert len(calls) == 1
    sample_catalog._CACHE.clear()
