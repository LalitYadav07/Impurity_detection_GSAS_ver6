"""Only resolved nonmatching preview identities can prune raw candidates."""

import time
from types import SimpleNamespace

import pytest

from radar_pd_nova import powgen_samples
from radar_pd_nova.nomad_metadata import NomadSampleIndex
from radar_pd_nova.powgen_samples import PowgenSampleIndex, select_runs
from radar_pd_nova.powgen_watch import WatchedRun
from radar_pd_nova.snap_metadata import SnapSampleIndex


@pytest.fixture(params=["PG3", "NOM", "SNAP"])
def cached_candidate(request):
    instrument = request.param
    index = {"PG3": PowgenSampleIndex, "NOM": NomadSampleIndex, "SNAP": SnapSampleIndex}[instrument]("IPTS-1")
    index.metadata_index = SimpleNamespace(revision=(), generation=0, refresh=lambda **_: None)
    run = WatchedRun(42, f"/fixture/{instrument}_42.gsa", "unchanged", instrument=instrument)

    def cache(metadata):
        index._preview_cache[42] = ((run.fingerprint, (), 0, time.monotonic()), metadata)
        return index, run

    return cache


@pytest.mark.parametrize("metadata", [
    {"sample_id": "", "identity_status": "unidentified"},
    {"sample_id": None, "identity_status": "resolved"},
    {"identity_status": "resolved"},
    {"sample_id": "B", "identity_status": "conflict"},
    {"sample_id": "B", "identity_status": "unidentified"},
    {"sample_id": "B"},
])
def test_unresolved_preview_retains_source_identity_fallback(cached_candidate, metadata):
    index, run = cached_candidate(metadata)
    assert index.candidate_runs([run], ("A",)) == [run]


def test_resolved_nonmatching_preview_still_prunes_unrelated_run(cached_candidate):
    index, run = cached_candidate({"sample_id": "B", "identity_status": "resolved"})
    assert index.candidate_runs([run], ("A",)) == []


def test_matching_preview_is_not_scientific_identity_evidence(cached_candidate, monkeypatch):
    index, run = cached_candidate({"sample_id": "A", "identity_status": "resolved"})
    calls = []
    def source_identity(current):
        calls.append(current.run_number)
        return {"sample_id": "B", "identity_status": "resolved"}
    monkeypatch.setattr(index, "identity", source_identity)
    monkeypatch.setattr(powgen_samples, "include_run", lambda *a, **k: True)
    assert select_runs([run], index, ("A",), limit=1) == []
    assert calls == [42]


def test_unknown_preview_is_reverified_when_raw_metadata_arrives(cached_candidate, monkeypatch):
    index, run = cached_candidate({"sample_id": "", "identity_status": "unidentified"})
    calls = []
    def source_identity(current):
        calls.append(current.run_number)
        return {"sample_id": "A", "identity_status": "resolved"}
    monkeypatch.setattr(index, "identity", source_identity)
    monkeypatch.setattr(powgen_samples, "include_run", lambda *a, **k: True)
    assert select_runs([run], index, ("A",), limit=1) == [run]
    assert calls == [42]


@pytest.mark.parametrize("stale_field", ["fingerprint", "revision", "generation", "age"])
def test_stale_preview_retains_source_identity_fallback(cached_candidate, monkeypatch, stale_field):
    index, run = cached_candidate({"sample_id": "B", "identity_status": "resolved"})
    stamp, metadata = index._preview_cache[42]
    stamp = list(stamp)
    if stale_field == "fingerprint":
        stamp[0] = "old-reduction"
    elif stale_field == "revision":
        stamp[1] = ("old-table",)
    elif stale_field == "generation":
        stamp[2] = -1
    else:
        stamp[3] -= powgen_samples.TTL_SECONDS + 1
        monkeypatch.setattr(index, "inventory", lambda *a, **k: index._preview_cache.clear())
    index._preview_cache[42] = (tuple(stamp), metadata)
    assert index.candidate_runs([run], ("A",)) == [run]
