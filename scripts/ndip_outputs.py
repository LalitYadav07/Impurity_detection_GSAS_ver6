"""Normalize RADAR-PD run folders into stable NDIP/Galaxy outputs."""
from __future__ import annotations

import csv
import html
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterable

try:
    from .ndip_contracts import (
        GPX_INDEX_SCHEMA,
        RESULT_SCHEMA,
        atomic_write_json,
        file_record,
        read_json,
        safe_name,
        utc_now,
    )
except ImportError:
    from ndip_contracts import (  # type: ignore
        GPX_INDEX_SCHEMA,
        RESULT_SCHEMA,
        atomic_write_json,
        file_record,
        read_json,
        safe_name,
        utc_now,
    )


COLLECTION_RULES = {
    "plots": {".png", ".jpg", ".jpeg", ".svg", ".html", ".npz"},
    "tables": {".csv", ".tsv"},
    "phases": {".cif"},
    "gpx": {".gpx"},
    "diagnostics": {".json", ".yaml", ".yml", ".log", ".lst", ".txt"},
}

SKIP_DIRS = {"inputs", "portal", "ndip_outputs", "__pycache__"}


def _git_commit(project_root: Path) -> str:
    explicit = os.environ.get("RADAR_PD_GIT_COMMIT", "").strip()
    if explicit:
        return explicit
    head = project_root / ".git" / "HEAD"
    try:
        value = head.read_text(encoding="utf-8").strip()
        if value.startswith("ref:"):
            ref = project_root / ".git" / value.split(":", 1)[1].strip()
            return ref.read_text(encoding="utf-8").strip()
        return value
    except OSError:
        return "unknown"


def _read_csv(path: Path, limit: int = 500) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for _, row in zip(range(limit), reader)]
    except OSError:
        return []


def _flatten_name(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    stem = relative.replace("/", "__").replace(" ", "_")
    return safe_name(stem, path.name)


def _iter_artifacts(run_dir: Path) -> Iterable[Path]:
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(run_dir).parts
        if any(part in SKIP_DIRS for part in relative_parts[:-1]):
            continue
        yield path


def _gpx_stage(relative: str) -> str:
    lowered = relative.lower()
    if "main" in lowered and ("anchor" in lowered or "phase" in lowered):
        return "main_phase_anchor"
    if "target" in lowered:
        return "targeted_refinement"
    if "hypothesis" in lowered or "gsas" in lowered or "validation" in lowered:
        return "hypothesis_refinement"
    if "pass" in lowered:
        return "full_pipeline_pass"
    if "final" in lowered or "polish" in lowered:
        return "final_refinement"
    return "refinement_checkpoint"


def build_gpx_index(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    projects: list[dict[str, Any]] = []
    for index, path in enumerate(sorted(root.rglob("*.gpx")), start=1):
        relative = path.resolve().relative_to(root).as_posix()
        lowered = relative.lower()
        if "rollback" in lowered:
            status = "rollback"
        elif any(token in lowered for token in ("failed", "error", "bad")):
            status = "failed"
        elif any(token in lowered for token in ("accepted", "final", "polish")):
            status = "accepted"
        else:
            status = "checkpoint"
        record = file_record(path, role="gsas2_project", root=root)
        record.update(
            {
                "index": index,
                "label": path.stem,
                "stage": _gpx_stage(relative),
                "status": status,
                "parent": None,
            }
        )
        projects.append(record)
    return {
        "$schema": GPX_INDEX_SCHEMA,
        "created_utc": utc_now(),
        "run_root": root.name,
        "project_count": len(projects),
        "projects": projects,
    }


def _copy_collections(run_dir: Path, portal: Path) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {name: [] for name in COLLECTION_RULES}
    for source in _iter_artifacts(run_dir):
        suffix = source.suffix.lower()
        collection = next((name for name, suffixes in COLLECTION_RULES.items() if suffix in suffixes), None)
        if collection is None:
            continue
        destination_dir = portal / collection
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / _flatten_name(source, run_dir)
        if destination.exists() and destination.resolve() != source.resolve():
            destination = destination.with_name(f"{destination.stem}_{len(records[collection]) + 1}{destination.suffix}")
        if destination.resolve() != source.resolve():
            shutil.copy2(source, destination)
        record = file_record(destination, role=collection.rstrip("s"), root=portal)
        record["source_path"] = source.relative_to(run_dir).as_posix()
        records[collection].append(record)
    return records


def _zip_run(run_dir: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        for path in _iter_artifacts(run_dir):
            handle.write(path, arcname=path.relative_to(run_dir).as_posix())


def _find_suffix(root: Path, suffix: str) -> Path | None:
    direct = root / suffix
    if direct.is_file():
        return direct
    expected = Path(suffix).as_posix()
    for path in sorted(root.rglob(Path(suffix).name)):
        if path.relative_to(root).as_posix().endswith(expected):
            return path
    return None


def _summary_sources(run_dir: Path, mode: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if mode == "rapid":
        summary_path = _find_suffix(run_dir, "rapid_results/summary.json")
        summary = read_json(summary_path, {}) if summary_path else {}
        hypothesis_path = _find_suffix(run_dir, "rapid_results/all_gsas_validation_summary.csv")
        hypotheses = _read_csv(hypothesis_path) if hypothesis_path else []
        phases: list[dict[str, Any]] = []
    else:
        summary_path = _find_suffix(run_dir, "pipeline_summary.json")
        summary = read_json(summary_path, {}) if summary_path else {}
        hypotheses = []
        phase_path = next(iter(run_dir.rglob("Summary_Fractions.csv")), None)
        phases = _read_csv(phase_path) if phase_path else []
    return summary, hypotheses, phases


def _artifact_href(item: dict[str, Any]) -> str:
    return html.escape(str(item.get("path") or "#"), quote=True)


def _artifact_label(item: dict[str, Any]) -> str:
    return html.escape(str(item.get("source_path") or item.get("name") or "artifact"))


def _artifact_title(item: dict[str, Any]) -> str:
    raw = str(item.get("source_path") or item.get("name") or "artifact")
    name = Path(raw).stem.replace("__", " / ").replace("_", " ")
    return html.escape(" ".join(name.split()))


def _artifact_links(items: list[dict[str, Any]], *, limit: int = 80) -> str:
    if not items:
        return "<p class='empty'>No files were produced for this collection.</p>"
    links = "".join(
        f"<li><a href='{_artifact_href(item)}' target='_blank' rel='noopener'>{_artifact_label(item)}</a></li>"
        for item in items[:limit]
    )
    remaining = len(items) - limit
    suffix = f"<li class='muted'>and {remaining} more file(s)</li>" if remaining > 0 else ""
    return f"<ul class='file-list'>{links}{suffix}</ul>"


def _plot_gallery(items: list[dict[str, Any]], *, limit: int = 24) -> str:
    if not items:
        return "<p class='empty'>No plot artifacts were produced.</p>"
    figures: list[str] = []
    links: list[dict[str, Any]] = []
    def priority(item: dict[str, Any]) -> tuple[int, str]:
        label = str(item.get("source_path") or item.get("name") or "").lower()
        if any(token in label for token in ("final", "accepted_model", "best_fit", "curve")):
            return (0, label)
        if any(token in label for token in ("fit", "refine", "hypothesis")):
            return (1, label)
        return (2, label)

    for item in sorted(items, key=priority):
        suffix = Path(str(item.get("name") or item.get("path") or "")).suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".svg"} and len(figures) < limit:
            href = _artifact_href(item)
            figures.append(
                f"<figure><a href='{href}' target='_blank' rel='noopener'>"
                f"<img src='{href}' loading='lazy' alt='{_artifact_label(item)}'></a>"
                f"<figcaption><strong>{_artifact_title(item)}</strong>"
                f"<span>{_artifact_label(item)}</span></figcaption></figure>"
            )
        else:
            links.append(item)
    gallery = f"<div class='gallery'>{''.join(figures)}</div>" if figures else ""
    extra = _artifact_links(links) if links else ""
    return gallery + extra


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
        return number if number == number else None
    except (TypeError, ValueError):
        return None


def _fmt_number(value: Any, digits: int = 2) -> str:
    number = _as_float(value)
    return "-" if number is None else f"{number:.{digits}f}"


def _timing_values(summary: dict[str, Any]) -> list[tuple[str, float]]:
    timings = ((summary.get("live_run") or {}).get("timings") or {}) if isinstance(summary, dict) else {}
    labels = {
        "signal_seconds": "Input preparation",
        "search64_seconds": "Coarse search",
        "nudge_seconds": "Lattice nudge",
        "rerank512_seconds": "Pattern scoring",
        "gsas_wall_seconds": "Final refinement wall time",
        "gsas_total_seconds": "Summed refinement time",
        "total_seconds": "Total",
    }
    values: list[tuple[str, float]] = []
    for key, label in labels.items():
        value = _as_float(timings.get(key))
        if value is not None:
            values.append((label, value))
    if not any(label == "Total" for label, _ in values):
        stage_values = [value for label, value in values if "Summed" not in label]
        if stage_values:
            values.append(("Reported stages", sum(stage_values)))
    return values


def _hypothesis_table(items: list[dict[str, Any]], *, limit: int = 20) -> str:
    if not items:
        return "<p class='empty'>No final hypothesis table was produced for this run.</p>"
    rows: list[str] = []
    for index, item in enumerate(items[:limit], start=1):
        rank = item.get("gsas_rwp_rank") or item.get("rank") or index
        hypothesis = item.get("hypothesis") or item.get("formulas") or item.get("formula_keys") or "-"
        quality = item.get("rwp") or item.get("Rwp") or item.get("refinement_quality")
        fractions = item.get("weights_json") or item.get("phase_fractions") or item.get("weights") or "-"
        status = item.get("status") or "complete"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(rank))}</td>"
            f"<td class='scientific'>{html.escape(str(hypothesis))}</td>"
            f"<td>{_fmt_number(quality, 3)}</td>"
            f"<td class='scientific'>{html.escape(str(fractions))}</td>"
            f"<td>{html.escape(str(status).replace('_', ' ').title())}</td>"
            "</tr>"
        )
    return (
        "<div class='table-scroll'><table><thead><tr><th>Rank</th><th>Phase hypothesis</th>"
        "<th>Rwp / fit metric</th><th>Refined phase fractions</th><th>Status</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _phase_table(items: list[dict[str, Any]], *, limit: int = 30) -> str:
    if not items:
        return ""
    columns = list(items[0].keys())[:8]
    head = "".join(f"<th>{html.escape(str(column).replace('_', ' ').title())}</th>" for column in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(item.get(column, '')))}</td>" for column in columns) + "</tr>"
        for item in items[:limit]
    )
    return f"<div class='table-scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _message_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("message") or item.get("error") or json.dumps(item, default=str))
    return str(item)

def _render_report(result: dict[str, Any]) -> str:
    summary = result.get("summary") or {}
    projects = result.get("gpx_projects") or []
    collections = result.get("artifacts") or {}
    warnings = result.get("warnings") or []
    errors = result.get("errors") or []
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item.get('index', '')))}</td>"
        f"<td><a href='{html.escape(str(item.get('collection_path') or '#'), quote=True)}' target='_blank' rel='noopener'>{html.escape(str(item.get('label', '')))}</a></td>"
        f"<td>{html.escape(str(item.get('stage', '')))}</td>"
        f"<td>{html.escape(str(item.get('status', '')))}</td>"
        "</tr>"
        for item in projects
    ) or "<tr><td colspan='4'>No GPX project was produced.</td></tr>"
    collection_rows = "".join(
        f"<tr><td>{html.escape(name.title())}</td><td>{len(items)}</td></tr>"
        for name, items in collections.items()
        if isinstance(items, list)
    )
    messages = "".join(f"<li>{html.escape(_message_text(item))}</li>" for item in [*warnings, *errors])
    plots = list(collections.get("plots") or [])
    tables = list(collections.get("tables") or [])
    phases = list(collections.get("phases") or [])
    diagnostics = list(collections.get("diagnostics") or [])
    hypotheses = list(result.get("hypotheses") or [])
    phases_summary = list(result.get("phases") or [])
    timings = _timing_values(summary)
    best_rwp = min(
        (value for value in (_as_float(item.get("rwp") or item.get("Rwp")) for item in hypotheses) if value is not None),
        default=None,
    )
    cards = [
        ("Analysis", "Rapid hypothesis" if str(result.get("analysis_mode")) == "rapid" else "Full RADAR-PD"),
        ("Status", str(result.get("status") or "unknown").title()),
        ("Final hypotheses", str(len(hypotheses)) if hypotheses else "-"),
        ("Reported phases", str(len(phases_summary)) if phases_summary else "-"),
        ("Best Rwp", _fmt_number(best_rwp, 3)),
    ]
    timing_cards = "".join(
        f"<div class='metric'><span>{html.escape(label)}</span><strong>{value:.1f} s</strong></div>"
        for label, value in timings
    )
    overview_cards = "".join(
        f"<div class='metric'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>"
        for label, value in cards
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RADAR-PD result: {html.escape(str(result.get('run_name', 'run')))}</title>
<style>
:root{{--ink:#14231d;--muted:#66756e;--green:#125b42;--green2:#e8f3ed;--line:#d4e0d9;--panel:#f7faf8;--warn:#fff4d6}}
*{{box-sizing:border-box}} body{{font-family:Inter,Segoe UI,Arial,sans-serif;max-width:1240px;margin:0 auto;padding:2.25rem 1.4rem 5rem;color:var(--ink);background:#fff;line-height:1.5}}
h1{{font-size:2rem;letter-spacing:0;margin:.25rem 0}} h2{{color:#103f31;margin:2.35rem 0 .45rem;font-size:1.3rem}} h3{{margin:1.25rem 0 .35rem}}
.eyebrow{{color:var(--green);font-size:.78rem;font-weight:800;text-transform:uppercase}} .lede{{color:var(--muted);max-width:780px;margin:.4rem 0 1.3rem}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:.7rem;margin:1rem 0}}
.metric{{border:1px solid var(--line);border-left:4px solid var(--green);padding:.75rem .85rem;background:#fff;min-height:74px}}
.metric span{{display:block;color:var(--muted);font-size:.75rem;font-weight:700}} .metric strong{{display:block;font-size:1.05rem;margin-top:.2rem;overflow-wrap:anywhere}}
.notice{{border:1px solid #ecd58e;background:var(--warn);padding:.8rem 1rem;margin:1rem 0}}
table{{border-collapse:collapse;width:100%;margin:.8rem 0}} th,td{{border:1px solid var(--line);padding:.58rem;text-align:left;vertical-align:top}} th{{background:var(--green2);font-size:.8rem}}
.table-scroll{{overflow-x:auto}} .scientific{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.86rem}}
a{{color:#126b4e}} pre{{background:#f5f7f6;border:1px solid var(--line);padding:1rem;overflow:auto;max-height:34rem;font-size:.8rem}}
.empty,.muted{{color:var(--muted)}} .file-list{{columns:2;column-gap:2rem;padding-left:1.2rem}} .file-list li{{break-inside:avoid;margin:.22rem 0}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:1rem}}
figure{{margin:0;border:1px solid var(--line);padding:.7rem;background:var(--panel)}} figure img{{display:block;width:100%;height:280px;object-fit:contain;background:#fff}}
figcaption{{font-size:.82rem;color:var(--muted);margin-top:.55rem;overflow-wrap:anywhere}} figcaption strong,figcaption span{{display:block}} figcaption strong{{color:var(--ink);font-size:.9rem}}
details{{border:1px solid var(--line);padding:.7rem .9rem;margin:.8rem 0}} summary{{cursor:pointer;font-weight:700}}
@media(max-width:700px){{body{{padding:1rem}}.file-list{{columns:1}}figure img{{height:auto}}.gallery{{grid-template-columns:1fr}}}}
</style></head><body>
<div class="eyebrow">RADAR-PD scientific analysis</div><h1>{html.escape(str(result.get('run_name')))}</h1>
<p class="lede">Candidate phases, lattice-aware pattern matching, and GSAS-II refinement outputs collected into one inspectable NDIP result.</p>
<div class="metrics">{overview_cards}</div>
{f'<h2>Stage timing</h2><div class="metrics">{timing_cards}</div>' if timing_cards else ''}
{f'<div class="notice"><strong>Run messages</strong><ul>{messages}</ul></div>' if messages else ''}
<h2>Scientific result</h2>
<p class="muted">For Rapid mode, pattern ranking narrows hypotheses and final GSAS-II refinement supplies the fit metric and phase fractions. Treat a low-ranked or zero-weight phase as unsupported by that refinement, not as proof of absence.</p>
{_hypothesis_table(hypotheses)}
{f'<h3>Final phase summary</h3>{_phase_table(phases_summary)}' if phases_summary else ''}
<h2>Fit and diagnostic plots</h2><p class="muted">Final and accepted fit plots are shown first. Open any plot at full size to inspect residuals and Bragg positions.</p>{_plot_gallery(plots)}
<h2>Continue in GSAS-II</h2><p>Each refinement checkpoint is preserved with provenance. Select a GPX project with the RADAR-PD GPX handoff tool, then open it in the hosted GSAS-II interface for manual inspection or continued refinement.</p>
<table><tr><th>#</th><th>Project</th><th>Stage</th><th>Status</th></tr>{rows}</table>
<h2>Downloads and reproducibility</h2>
<details open><summary>Result tables ({len(tables)})</summary>{_artifact_links(tables)}</details>
<details><summary>Phase CIF files ({len(phases)})</summary>{_artifact_links(phases)}</details>
<details><summary>Diagnostics and logs ({len(diagnostics)})</summary>{_artifact_links(diagnostics)}</details>
<details><summary>Output inventory</summary><table><tr><th>Collection</th><th>Files</th></tr>{collection_rows}</table></details>
<details><summary>Machine-readable pipeline summary</summary><pre>{html.escape(json.dumps(summary, indent=2, default=str))}</pre></details>
</body></html>"""


def collect_outputs(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    mode: str,
    run_name: str | None = None,
    project_root: str | Path | None = None,
    include_archive: bool = True,
    status: str = "complete",
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    run_root = Path(run_dir).resolve()
    portal = Path(output_dir).resolve()
    portal.mkdir(parents=True, exist_ok=True)
    if not run_root.exists():
        raise FileNotFoundError(f"RADAR-PD run directory does not exist: {run_root}")

    collection_records = _copy_collections(run_root, portal)
    gpx_index = build_gpx_index(run_root)
    gpx_outputs = {item.get("source_path"): item for item in collection_records.get("gpx", [])}
    for project in gpx_index["projects"]:
        source_path = project.get("path")
        project["source_path"] = source_path
        output_record = gpx_outputs.get(source_path)
        if output_record:
            project["collection_path"] = output_record.get("path")
            project["collection_name"] = output_record.get("name")
    atomic_write_json(portal / "gpx_index.json", gpx_index)
    summary, hypotheses, phases = _summary_sources(run_root, mode)
    manifest_path = _find_suffix(run_root, "run_manifest.json")
    manifest = read_json(manifest_path) if manifest_path else {}
    effective_status = "failed" if status == "failed" else ("complete" if manifest.get("status", status) == "complete" else status)
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1]
    result: dict[str, Any] = {
        "$schema": RESULT_SCHEMA,
        "created_utc": utc_now(),
        "run_name": safe_name(run_name or run_root.name),
        "analysis_mode": mode,
        "status": effective_status,
        "summary": summary,
        "phases": phases,
        "hypotheses": hypotheses,
        "gpx_projects": gpx_index["projects"],
        "provenance": {
            "radar_pd_git_commit": _git_commit(root),
            "container_digest": os.environ.get("RADAR_PD_CONTAINER_DIGEST", "unknown"),
            "database_version": os.environ.get("RADAR_PD_DATABASE_VERSION", "unknown"),
            "source_manifest": manifest,
        },
        "artifacts": collection_records,
        "warnings": [],
        "errors": list(errors or []),
    }
    atomic_write_json(portal / "summary.json", result)
    (portal / "report.html").write_text(_render_report(result), encoding="utf-8")
    if include_archive:
        _zip_run(run_root, portal / "results.zip")
    return result


__all__ = ["build_gpx_index", "collect_outputs"]
