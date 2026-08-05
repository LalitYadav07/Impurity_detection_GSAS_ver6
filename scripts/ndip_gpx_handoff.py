#!/usr/bin/env python3
"""Prepare one RADAR-PD GPX checkpoint as a typed Galaxy handoff."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_index(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    candidate = Path(path)
    if not candidate.is_file():
        return {}
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def find_record(index: dict[str, Any], names: set[str]) -> dict[str, Any]:
    for record in index.get("projects") or []:
        if not isinstance(record, dict):
            continue
        candidates = {
            str(record.get("name") or ""),
            str(record.get("label") or ""),
            Path(str(record.get("path") or "unknown")).name,
            Path(str(record.get("source_path") or "unknown")).name,
            Path(str(record.get("collection_path") or "unknown")).name,
            str(record.get("collection_name") or ""),
        }
        if names & candidates:
            return record
    return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--project-name", default="")
    parser.add_argument("--index")
    parser.add_argument("--output-project", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    source = Path(args.project).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output = Path(args.output_project)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)

    supplied_name = str(args.project_name or "").strip()
    record = find_record(load_index(args.index), {source.name, supplied_name})
    metadata = {
        "$schema": "radar-pd-gpx-handoff/v1",
        "project_name": supplied_name or source.name,
        "size_bytes": output.stat().st_size,
        "sha256": sha256(output),
        "stage": record.get("stage", "refinement_checkpoint"),
        "checkpoint_status": record.get("status", "unknown"),
        "source_record": record,
        "handoff": {
            "dataset_extension": "gpx",
            "ready_for_interactive_gsas2": True,
            "requires_registered_ndip_tool": "Interactive GSAS-II GPX opener",
        },
    }
    Path(args.metadata).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    title = html.escape(metadata["project_name"])
    stage = html.escape(str(metadata["stage"]))
    status = html.escape(str(metadata["checkpoint_status"]))
    digest = html.escape(metadata["sha256"])
    report = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>RADAR-PD GPX handoff</title>
<style>body{{font:16px system-ui;max-width:900px;margin:2rem auto;color:#17251f}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd8d1;padding:.65rem;text-align:left}}
code{{overflow-wrap:anywhere}}</style></head><body>
<h1>GSAS-II project handoff</h1>
<p>The selected RADAR-PD refinement checkpoint has been preserved as a typed GPX dataset.</p>
<table><tr><th>Project</th><td>{title}</td></tr><tr><th>Pipeline stage</th><td>{stage}</td></tr>
<tr><th>Checkpoint status</th><td>{status}</td></tr><tr><th>SHA-256</th><td><code>{digest}</code></td></tr></table>
<h2>Next connection</h2>
<p>Connect the GPX output to NDIP's interactive GSAS-II GPX opener once that image and datatype are registered. The current batch GSAS2 Refinement tool starts a new project and cannot open this existing GPX file.</p>
</body></html>"""
    Path(args.report).write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
