#!/usr/bin/env python3
"""Replay a hash-sealed, small facility corpus into an isolated local tree.

This is a filesystem simulator, not a neutron reduction or a Galaxy emulator.
Manifest paths are relative to a simulated SNS root. Real facility roots are
always refused. Private experiment files belong outside the source repository.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import time

SCHEMA = "radar-pd-beamline-simulation/v1"


def confined(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if not name or relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError("Simulation paths must be relative without traversal")
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Simulation path escapes its root through a symlink")
    return path


class BeamlineSimulator:
    def __init__(self, manifest: Path, corpus: Path, destination: Path):
        self.manifest = json.loads(Path(manifest).read_text())
        if self.manifest.get("$schema") != SCHEMA:
            raise ValueError("Unsupported simulation manifest")
        self.corpus = Path(corpus).resolve(strict=True)
        self.destination = Path(destination).resolve()
        protected = (Path("/SNS"), Path("/HFIR"), Path("/gpfs"))
        if any(self.destination == p or self.destination.is_relative_to(p) for p in protected):
            raise ValueError("A simulator cannot write into a facility archive")
        if (
            self.destination == self.corpus
            or self.destination.is_relative_to(self.corpus)
            or self.corpus.is_relative_to(self.destination)
        ):
            raise ValueError("Corpus and simulated tree must be disjoint")
        self.files = {}
        for item in self.manifest.get("files", []):
            name = item["path"]
            if name in self.files:
                raise ValueError("Duplicate corpus path")
            source = confined(self.corpus, name)
            confined(self.destination, name)
            if not source.is_file():
                raise ValueError(f"Missing corpus file: {name}")
            if (
                source.stat().st_size != item["bytes"]
                or sha256(source.read_bytes()).hexdigest() != item["sha256"]
            ):
                raise ValueError(f"Corpus hash/size mismatch: {name}")
            self.files[name] = item
        self.events = list(self.manifest.get("events", []))
        previous = -1.0
        for event in self.events:
            delay = event.get("at_seconds", 0)
            if (
                not isinstance(delay, (int, float))
                or not 0 <= delay < float("inf")
                or delay < previous
            ):
                raise ValueError("Events need finite, ordered nonnegative times")
            previous = delay
            if event.get("operation", "publish") not in {"publish", "remove", "link"}:
                raise ValueError("Unsupported simulation event")
            confined(self.destination, event["path"])
            if event.get("operation") == "link":
                source_name = event.get("source")
                if source_name not in self.files or event["path"] not in self.files:
                    raise ValueError("A link must join two sealed corpus paths")
                if self.files[source_name]["sha256"] != self.files[event["path"]]["sha256"]:
                    raise ValueError("An alias must have the same sealed content as its source")
                confined(self.destination, source_name)
            if event.get("operation", "publish") == "publish":
                source_name = event.get("source", event["path"])
                if source_name not in self.files:
                    raise ValueError("An event source is not in the sealed corpus")
                count = event.get("bytes", self.files[source_name]["bytes"])
                if (
                    isinstance(count, bool)
                    or not isinstance(count, int)
                    or not 0 <= count <= self.files[source_name]["bytes"]
                ):
                    raise ValueError("Invalid partial-file size")
        self.destination.mkdir(parents=True, exist_ok=True)
        marker = self.destination / ".radar-pd-simulator"
        if not marker.exists() and any(self.destination.iterdir()):
            raise ValueError("Use an empty directory or an existing marked simulator")
        marker.write_text(SCHEMA + "\n")

    def apply(self, index: int) -> dict:
        event = self.events[index]
        target = confined(self.destination, event["path"])
        if event.get("operation", "publish") == "remove":
            target.unlink(missing_ok=True)
            return {"event": index, **event}
        if event.get("operation") == "link":
            source = confined(self.destination, event["source"])
            if source.is_symlink() or not source.is_file():
                raise ValueError("Publish the canonical regular file before linking an alias")
            if sha256(source.read_bytes()).hexdigest() != self.files[event["source"]]["sha256"]:
                raise ValueError("The canonical alias source differs from its sealed content")
            target.parent.mkdir(parents=True, exist_ok=True)
            target = confined(self.destination, event["path"])
            with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=".alias-", delete=False
            ) as stream:
                temporary = Path(stream.name)
            try:
                temporary.unlink()
                os.link(source, temporary)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            return {"event": index, **event, "hardlink": True}
        source_name = event.get("source", event["path"])
        source = confined(self.corpus, source_name)
        raw = source.read_bytes()
        if sha256(raw).hexdigest() != self.files[source_name]["sha256"]:
            raise ValueError("Corpus changed after validation")
        raw = raw[: event.get("bytes", len(raw))]
        target.parent.mkdir(parents=True, exist_ok=True)
        # Recheck after creating parents: no symlink may redirect a replay.
        target = confined(self.destination, event["path"])
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=".arriving-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        return {
            "event": index,
            **event,
            "published_bytes": len(raw),
            "published_sha256": sha256(raw).hexdigest(),
        }

    def play(self, *, realtime: bool = False):
        start = time.monotonic()
        for index, event in enumerate(self.events):
            if realtime:
                delay = event.get("at_seconds", 0) - (time.monotonic() - start)
                if delay > 0:
                    time.sleep(delay)
            yield self.apply(index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()
    simulator = BeamlineSimulator(args.manifest, args.corpus, args.destination)
    for receipt in simulator.play(realtime=args.realtime):
        print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
