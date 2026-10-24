"""Isolation and reproducible file-arrival behavior for the simulator."""

from hashlib import sha256
import json
from pathlib import Path

import pytest

from scripts.beamline_simulator import BeamlineSimulator, SCHEMA


def fixture(tmp_path, events=None):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    raw = b"BANK 1 example\n1 2 3\n"
    name = "NOM/IPTS-1/shared/autoreduce/NOM_1.gsa"
    source = corpus / name
    source.parent.mkdir(parents=True)
    source.write_bytes(raw)
    payload = {
        "$schema": SCHEMA,
        "files": [{"path": name, "bytes": len(raw), "sha256": sha256(raw).hexdigest()}],
        "events": events
        or [
            {"path": name, "bytes": 5},
            {"path": name},
            {"path": name, "operation": "remove"},
        ],
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload))
    return manifest, corpus, tmp_path / "simulated", name, raw


def test_partial_completion_and_removal_leave_corpus_unchanged(tmp_path):
    manifest, corpus, tree, name, raw = fixture(tmp_path)
    sim = BeamlineSimulator(manifest, corpus, tree)
    sim.apply(0)
    assert (tree / name).read_bytes() == raw[:5]
    sim.apply(1)
    assert (tree / name).read_bytes() == raw
    sim.apply(2)
    assert not (tree / name).exists()
    assert (corpus / name).read_bytes() == raw


def test_verified_raw_aliases_preserve_inode_identity(tmp_path):
    manifest, corpus, tree, name, raw = fixture(tmp_path)
    alias = "NOM/IPTS-1/data/NOM_1_event.nxs"
    path = corpus / alias
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    payload = json.loads(manifest.read_text())
    payload["files"].append({"path": alias, "bytes": len(raw), "sha256": sha256(raw).hexdigest()})
    payload["events"] = [{"path": name}, {"operation": "link", "path": alias, "source": name}]
    manifest.write_text(json.dumps(payload))
    sim = BeamlineSimulator(manifest, corpus, tree)
    list(sim.play())
    assert (tree / alias).stat().st_ino == (tree / name).stat().st_ino
    assert (tree / alias).read_bytes() == raw


@pytest.mark.parametrize("name", ["../outside", "/SNS/NOM/IPTS-1/data", "a/../../escape", "a\\b"])
def test_rejects_escaping_event_paths(tmp_path, name):
    manifest, corpus, tree, _, _ = fixture(tmp_path, [{"path": name, "operation": "remove"}])
    with pytest.raises(ValueError):
        BeamlineSimulator(manifest, corpus, tree)
    assert not tree.exists()


def test_rejects_changed_corpus_and_injected_symlink(tmp_path):
    manifest, corpus, tree, name, _ = fixture(tmp_path)
    sim = BeamlineSimulator(manifest, corpus, tree)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tree / "NOM").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        sim.apply(0)
    (tree / "NOM").unlink()
    (corpus / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        sim.apply(0)


def test_rejects_real_archives_and_existing_unmarked_destination(tmp_path):
    manifest, corpus, tree, _, _ = fixture(tmp_path)
    for destination in [
        Path("/SNS"),
        Path("/SNS/NOM/IPTS-1"),
        Path("/gpfs/neutronsfs/test"),
        corpus / "child",
    ]:
        with pytest.raises(ValueError):
            BeamlineSimulator(manifest, corpus, destination)
    tree.mkdir()
    (tree / "user-file").write_text("keep")
    with pytest.raises(ValueError, match="empty"):
        BeamlineSimulator(manifest, corpus, tree)
    assert (tree / "user-file").read_text() == "keep"
