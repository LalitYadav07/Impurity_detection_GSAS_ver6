#!/usr/bin/env python3
"""NDIP/Galaxy adapter for the RADAR-PD scientific pipelines.

The adapter owns path materialization and output normalization.  It does not
reimplement candidate search, lattice nudging or GSAS-II refinement.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import traceback
import zipfile
from pathlib import Path
from typing import Any, Iterable

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for item in (str(PROJECT_ROOT), str(SCRIPT_DIR)):
    if item not in sys.path:
        sys.path.insert(0, item)

from ndip_contracts import (  # noqa: E402
    CONFIG_SCHEMA,
    LIBRARY_SCHEMA,
    atomic_write_json,
    initial_state,
    input_manifest,
    safe_name,
    update_state,
    utc_now,
)
from ndip_outputs import collect_outputs  # noqa: E402


PLACEHOLDER_DATA = "RADAR_PD_NDIP_WILL_REWRITE_DATA"
PLACEHOLDER_INSTRUMENT = "RADAR_PD_NDIP_WILL_REWRITE_INSTRUMENT"


def _load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("RADAR-PD configuration must be a YAML mapping")
    return payload


def _write_yaml(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target


def _split_elements(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, str):
        parts = [str(item) for item in value]
    else:
        parts = re.split(r"[\s,;]+", value.strip())
    result: list[str] = []
    for part in parts:
        token = part.strip()
        if not token:
            continue
        normalized = token[:1].upper() + token[1:].lower()
        if not re.fullmatch(r"[A-Z][a-z]?", normalized):
            raise ValueError(f"Invalid element symbol: {token!r}")
        if normalized not in result:
            result.append(normalized)
    return result


def _parse_regions(values: list[str] | None) -> list[list[float]]:
    regions: list[list[float]] = []
    for value in values or []:
        parts = [part for part in re.split(r"[:,\s]+", str(value).strip()) if part]
        if len(parts) != 2:
            raise ValueError(f"Region must have start and end values: {value!r}")
        start, end = float(parts[0]), float(parts[1])
        if end <= start:
            raise ValueError(f"Region end must be greater than start: {value!r}")
        regions.append([start, end])
    return regions


def _contract_config(args: argparse.Namespace) -> dict[str, Any]:
    mode = "rapid" if args.mode == "rapid" else "full"
    allowed = _split_elements(args.allowed_elements)
    if not allowed:
        raise ValueError("At least one sample element is required")
    limits = None
    if args.limit_start is not None or args.limit_end is not None:
        if args.limit_start is None or args.limit_end is None or args.limit_end <= args.limit_start:
            raise ValueError("Both pattern limits are required and the upper limit must be larger")
        limits = [float(args.limit_start), float(args.limit_end)]
    payload: dict[str, Any] = {
        "$schema": CONFIG_SCHEMA,
        "created_utc": utc_now(),
        "analysis": {
            "mode": mode,
            "radiation": args.radiation,
            "instrument_mode": args.instrument_mode,
        },
        "chemistry": {
            "sample_elements": allowed,
            "environment_elements": _split_elements(args.environment_elements),
        },
        "pattern": {
            "limits": limits,
            "exclude_regions": _parse_regions(args.exclude_region),
        },
        "background": {
            "mode": args.background_mode,
            "type": args.background_type,
            "terms": int(args.background_terms),
        },
        "main_phase": {
            "prenudge": bool(args.main_prenudge),
            "shadow_filter": bool(args.main_shadow_filter),
            "cleanup": {
                "enabled": bool(args.main_cleanup),
                "refine_u_iso": bool(args.refine_uiso),
                "refine_positions": bool(args.refine_positions),
            },
        },
        "magnetic_precheck": {"enabled": bool(args.magnetic_precheck)},
        "full": {
            "max_passes": int(args.max_passes),
            "min_phase_percent": float(args.min_phase_percent),
        },
        "rapid": {
            "phases_per_hypothesis": int(args.phases_per_hypothesis),
            "gsas_validation_limit": int(args.gsas_validation_limit),
        },
    }
    return payload


def command_configure(args: argparse.Namespace) -> int:
    config = _contract_config(args)
    _write_yaml(args.output, config)
    if args.report:
        atomic_write_json(
            args.report,
            {
                "$schema": CONFIG_SCHEMA,
                "status": "valid",
                "output": str(Path(args.output).resolve()),
                "analysis": config["analysis"],
                "chemistry": config["chemistry"],
            },
        )
    print(f"Wrote RADAR-PD NDIP configuration: {args.output}")
    return 0


def _db_root_for(radiation: str, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    environment = os.environ.get("RADAR_PD_DB_ROOT", "").strip()
    if environment:
        candidate = Path(environment).resolve()
        nested = candidate / f"database_{radiation}"
        return nested if nested.exists() else candidate
    return (PROJECT_ROOT / "data" / f"database_{radiation}").resolve()


def _materialize_contract_config(
    contract: dict[str, Any],
    *,
    run_name: str,
    data: Path,
    instrument: Path,
    main_cif: Path | None,
    work_root: Path,
    db_root: Path,
) -> dict[str, Any]:
    from config_builder import build_pipeline_config

    analysis = dict(contract.get("analysis") or {})
    chemistry = dict(contract.get("chemistry") or {})
    pattern = dict(contract.get("pattern") or {})
    background = dict(contract.get("background") or {})
    main_phase = dict(contract.get("main_phase") or {})
    cleanup = dict(main_phase.get("cleanup") or {})
    rapid = dict(contract.get("rapid") or {})
    full = dict(contract.get("full") or {})
    mode = "rapid" if analysis.get("mode") == "rapid" else "full"
    radiation = str(analysis.get("radiation") or "neutron").lower()
    advanced = {
        "analysis_mode": "rapid_hypothesis" if mode == "rapid" else "full_radar_pd",
        "rapid_hypothesis": {
            "enabled": mode == "rapid",
            "phases_per_hypothesis": int(rapid.get("phases_per_hypothesis", 3)),
            "gsas_validation_limit": int(rapid.get("gsas_validation_limit", 10)),
        },
        "background": {
            "mode": background.get("mode", "auto_fixed_points"),
            "type": background.get("type", "chebyschev-1"),
            "terms": int(background.get("terms", 6)),
        },
        "main_phase_prenudge": {"enabled": bool(main_phase.get("prenudge", True))},
        "main_phase_shadow": {"enabled": bool(main_phase.get("shadow_filter", True))},
        "main_phase_cleanup": {
            "enabled": bool(cleanup.get("enabled", False)),
            "refine_u_iso": bool(cleanup.get("refine_u_iso", False)),
            "refine_positions": bool(cleanup.get("refine_positions", False)),
        },
        "magnetic_precheck": dict(contract.get("magnetic_precheck") or {"enabled": False}),
        "stage4": {"radiation": radiation},
        "db_source": radiation,
    }
    text = build_pipeline_config(
        run_name=run_name,
        data_file=str(data),
        instprm_file=str(instrument),
        allowed_elements=_split_elements(chemistry.get("sample_elements")),
        main_cif=str(main_cif) if main_cif else None,
        work_root=str(work_root),
        project_root=str(PROJECT_ROOT),
        db_root=str(db_root),
        min_impurity_percent=float(full.get("min_phase_percent", 0.5)),
        max_passes=int(full.get("max_passes", 3)),
        sample_env_elements=_split_elements(chemistry.get("environment_elements")),
        instrument_mode=str(analysis.get("instrument_mode") or "auto"),
        advanced_params=advanced,
        limits=pattern.get("limits"),
        exclude_regions=pattern.get("exclude_regions") or [],
    )
    return yaml.safe_load(text)


def _materialize_legacy_config(
    config: dict[str, Any],
    *,
    run_name: str,
    data: Path,
    instrument: Path,
    main_cif: Path | None,
    work_root: Path,
    mode: str,
) -> dict[str, Any]:
    from radar_api_server import _rewrite_config_for_job

    job_dir = work_root.parent
    resolved, _ = _rewrite_config_for_job(
        config,
        job_dir=job_dir,
        run_dir=work_root,
        dataset_name=None,
        run_name=run_name,
        data_path=data,
        instprm_path=instrument,
        main_cif_path=main_cif,
        mode=mode,
    )
    resolved.pop("api_job", None)
    return resolved


def _extract_db_pack(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    if archive.is_dir():
        return archive.resolve()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            root = destination.resolve()
            for member in handle.infolist():
                target = (destination / member.filename).resolve()
                if not target.is_relative_to(root):
                    raise ValueError(f"Unsafe path in custom database archive: {member.filename}")
            handle.extractall(destination)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            handle.extractall(destination, filter="data")
    else:
        raise ValueError(f"Unsupported custom database archive: {archive}")
    children = [path for path in destination.iterdir() if path.is_dir()]
    candidates = [destination, *children]
    for candidate in candidates:
        if any((candidate / name).exists() for name in ("db_manifest.json", "highsymm_metadata.json", "catalog.csv")):
            return candidate.resolve()
    return children[0].resolve() if len(children) == 1 else destination.resolve()


def _pipeline_command(config: Path, dataset: str, mode: str) -> list[str]:
    script = "rapid_hypothesis_pipeline.py" if mode == "rapid" else "gsas_complete_pipeline_nomain.py"
    return [sys.executable, str(SCRIPT_DIR / script), "--config", str(config), "--dataset", dataset]


def _stage_from_log(line: str, mode: str) -> tuple[str, str] | None:
    """Map stable pipeline log phrases to portable NDIP stage names."""
    lowered = line.lower()
    if mode == "rapid":
        patterns = (
            (("setup", "load", "input", "background"), "input_preparation", "Preparing signal and inputs"),
            (("coarse", "beam64", "64-bin"), "coarse_search", "Searching coarse hypotheses"),
            (("nudge", "lattice"), "lattice_nudge", "Optimizing candidate cells"),
            (("512", "pattern scoring", "refined shortlist"), "pattern_scoring", "Scoring lattice-aware patterns"),
            (("gsas", "validation", "final refinement"), "final_refinement", "Refining shortlisted hypotheses"),
            (("report", "summary.json"), "report", "Writing normalized results"),
        )
    else:
        patterns = (
            (("stage-0", "stage 0", "main phase"), "main_phase", "Preparing and anchoring the main phase"),
            (("stage-1", "stage 1", "ml"), "candidate_screening", "Screening phase candidates"),
            (("stage-4", "stage 4", "pearson", "nudge"), "lattice_nudge", "Scoring and nudging candidates"),
            (("joint refinement", "polish", "commit"), "refinement", "Refining accepted phases"),
            (("pipeline_summary", "pipeline complete", "final report"), "report", "Writing final results"),
        )
    for needles, stage, message in patterns:
        if any(needle in lowered for needle in needles):
            return stage, message
    return None


def command_analyze(args: argparse.Namespace) -> int:
    data = Path(args.data).resolve()
    main_cif = Path(args.main_cif).resolve() if args.main_cif else None
    if not data.is_file():
        raise FileNotFoundError(f"Missing diffraction data: {data}")
    if main_cif and not main_cif.is_file():
        raise FileNotFoundError(f"Missing main-phase CIF: {main_cif}")

    run_name = safe_name(args.run_name or Path(args.data).stem)
    work = Path(args.work_dir).resolve()
    run_dir = work / "run"
    input_dir = work / "inputs"
    portal = Path(args.output_dir).resolve()
    for directory in (run_dir, input_dir, portal):
        directory.mkdir(parents=True, exist_ok=True)

    copied_data = input_dir / data.name
    shutil.copy2(data, copied_data)
    if args.instrument:
        instrument = Path(args.instrument).resolve()
        if not instrument.is_file():
            raise FileNotFoundError(f"Missing instrument profile: {instrument}")
        copied_instrument = input_dir / instrument.name
        shutil.copy2(instrument, copied_instrument)
    elif args.instrument_preset:
        from instprm_presets import write_builtin_instprm_file
        copied_instrument = write_builtin_instprm_file(args.instrument_preset, input_dir / "generated_CuKa_lab.instprm")
    else:
        raise ValueError("Supply --instrument or select --instrument-preset")
    copied_main = None
    if main_cif:
        copied_main = input_dir / main_cif.name
        shutil.copy2(main_cif, copied_main)

    raw_config = _load_yaml(args.config)
    contract_mode = str((raw_config.get("analysis") or {}).get("mode") or "full")
    mode = args.mode if args.mode != "auto" else ("rapid" if contract_mode == "rapid" else "full")
    input_doc = input_manifest(
        run_name=run_name,
        mode=mode,
        data=copied_data,
        instrument=copied_instrument,
        main_cif=copied_main,
        source=args.input_source,
    )
    atomic_write_json(portal / "input_manifest.json", input_doc)
    state = initial_state(run_name=run_name, mode=mode, input_doc=input_doc)
    state_path = portal / "state.json"
    atomic_write_json(state_path, state)

    radiation = str((raw_config.get("analysis") or {}).get("radiation") or args.radiation or "neutron").lower()
    db_root = _db_root_for(radiation, args.db_root)
    if args.db_pack:
        db_root = _extract_db_pack(Path(args.db_pack).resolve(), work / "custom_database")

    if raw_config.get("$schema") == CONFIG_SCHEMA:
        resolved = _materialize_contract_config(
            raw_config,
            run_name=run_name,
            data=copied_data,
            instrument=copied_instrument,
            main_cif=copied_main,
            work_root=run_dir,
            db_root=db_root,
        )
    else:
        resolved = _materialize_legacy_config(
            raw_config,
            run_name=run_name,
            data=copied_data,
            instrument=copied_instrument,
            main_cif=copied_main,
            work_root=run_dir,
            mode=mode,
        )
    resolved["ndip_job"] = {
        "schema": "radar-pd-state/v1",
        "created_utc": utc_now(),
        "input_source": args.input_source,
        "ephemeral_work_dir": str(work),
    }
    resolved_path = work / "resolved_config.yaml"
    _write_yaml(resolved_path, resolved)
    shutil.copy2(resolved_path, portal / "resolved_config.yaml")

    state = update_state(state, status="running", stage="analysis", stage_status="running", message="RADAR-PD pipeline started")
    atomic_write_json(state_path, state)
    command = _pipeline_command(resolved_path, run_name, mode)
    log_path = portal / "console.log"
    return_code = 0
    if args.dry_run:
        (portal / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
        state = update_state(state, status="complete", stage="analysis", stage_status="skipped", message="Dry run completed")
        atomic_write_json(state_path, state)
        print("Dry run command:", " ".join(command))
        return 0

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SCRIPT_DIR), env.get("PYTHONPATH", "")]))
    current_stage = "analysis"
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_handle.write(line)
                log_handle.flush()
                stage_update = _stage_from_log(line, mode)
                if stage_update and stage_update[0] != current_stage:
                    current_stage = stage_update[0]
                    state = update_state(
                        state,
                        status="running",
                        stage=current_stage,
                        stage_status="running",
                        message=stage_update[1],
                    )
                    atomic_write_json(state_path, state)
            return_code = process.wait()
    except Exception as exc:
        return_code = 1
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(traceback.format_exc())
        state = update_state(state, status="failed", stage="analysis", stage_status="failed", error=str(exc))
        atomic_write_json(state_path, state)

    pipeline_run_dir = run_dir
    nested = run_dir / run_name
    if nested.exists() and any(nested.iterdir()):
        pipeline_run_dir = nested
    errors = list(state.get("errors") or [])
    if return_code:
        errors.append({"stage": "analysis", "message": f"Pipeline exited with code {return_code}"})
    collect_outputs(
        pipeline_run_dir,
        portal,
        mode=mode,
        run_name=run_name,
        project_root=PROJECT_ROOT,
        include_archive=not args.no_archive,
        status="failed" if return_code else "complete",
        errors=errors,
    )
    final_status = "failed" if return_code else "complete"
    state = update_state(
        state,
        status=final_status,
        stage="analysis",
        stage_status=final_status,
        message="RADAR-PD pipeline finished" if not return_code else "RADAR-PD pipeline failed",
        artifacts={"portal": str(portal)},
    )
    atomic_write_json(state_path, state)
    return return_code


def _decode_h5_value(value: Any) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _event_metadata(event_file: Path) -> dict[str, str]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to read NeXus metadata") from exc
    mapping = {
        "run": ("/entry/run_number", "/entry/run_number/value"),
        "instrument": ("/entry/instrument/name", "/entry/instrument/name/value"),
        "ipts": ("/entry/experiment_identifier", "/entry/experiment_identifier/value"),
    }
    result: dict[str, str] = {}
    with h5py.File(event_file, "r") as handle:
        for key, candidates in mapping.items():
            for candidate in candidates:
                if candidate in handle:
                    result[key] = _decode_h5_value(handle[candidate][()]).strip()
                    break
    return result


def _safe_facility_component(value: str, label: str) -> str:
    normalized = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", normalized):
        raise ValueError(f"Invalid {label}: {value!r}")
    return normalized


def _select_match(root: Path, patterns: list[str], replacements: dict[str, str], label: str) -> Path:
    matches: list[Path] = []
    for pattern in patterns:
        expanded = pattern.format(**replacements)
        matches.extend(path for path in root.glob(expanded) if path.is_file())
    unique = sorted({path.resolve() for path in matches}, key=lambda item: item.stat().st_mtime, reverse=True)
    if not unique:
        raise FileNotFoundError(f"No {label} matched under {root}")
    if len(unique) > 1:
        preview = ", ".join(str(path) for path in unique[:8])
        raise RuntimeError(f"Multiple {label} files matched; narrow the run/bank pattern: {preview}")
    return unique[0]


def command_resolve_ipts(args: argparse.Namespace) -> int:
    metadata = _event_metadata(Path(args.event_file).resolve()) if args.event_file else {}
    instrument = _safe_facility_component(args.instrument or metadata.get("instrument", ""), "instrument").upper()
    ipts_raw = args.ipts or metadata.get("ipts", "")
    ipts_match = re.search(r"IPTS[-_ ]?(\d+)", str(ipts_raw), re.IGNORECASE)
    ipts = f"IPTS-{ipts_match.group(1)}" if ipts_match else _safe_facility_component(str(ipts_raw), "IPTS")
    run = _safe_facility_component(args.run or metadata.get("run", ""), "run")
    bank = _safe_facility_component(args.bank, "bank") if args.bank else ""
    root = Path(args.facility_root).resolve() / instrument / ipts
    if not root.is_dir():
        raise FileNotFoundError(f"IPTS directory is not mounted or does not exist: {root}")
    replacements = {"instrument": instrument, "ipts": ipts, "run": run, "bank": bank}
    pattern = _select_match(root, args.pattern_glob, replacements, "reduced powder pattern")
    instrument_file = _select_match(root, args.instrument_glob, replacements, "instrument profile")
    shutil.copy2(pattern, args.pattern_output)
    shutil.copy2(instrument_file, args.instrument_output)
    result = {
        "$schema": "radar-pd-input/v1",
        "source": "ipts",
        "instrument": instrument,
        "ipts": ipts,
        "run": run,
        "bank": bank or None,
        "facility_root": str(Path(args.facility_root).resolve()),
        "resolved_pattern": str(pattern),
        "resolved_instrument": str(instrument_file),
    }
    atomic_write_json(args.metadata_output, result)
    print(json.dumps(result, indent=2))
    return 0


def _archive_directory(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                handle.write(path, arcname=(Path(source.name) / path.relative_to(source)).as_posix())


def command_build_library(args: argparse.Namespace) -> int:
    from db_pack_builder import build_augmented_db_pack, build_mini_db_pack

    cif_paths = [Path(path).resolve() for path in args.cif]
    if args.cif_dir:
        cif_paths.extend(sorted(Path(args.cif_dir).resolve().glob("*.cif")))
    if not cif_paths:
        raise ValueError("At least one CIF file is required")
    output_root = Path(args.output_root).resolve()
    if args.library_mode == "augmented":
        if not args.base_db_root:
            raise ValueError("Augmented libraries require --base-db-root")
        result = build_augmented_db_pack(
            cif_paths,
            output_root,
            source_type=args.radiation,
            base_db_root=Path(args.base_db_root).resolve(),
            overwrite=args.overwrite,
        )
    else:
        result = build_mini_db_pack(
            cif_paths,
            output_root,
            source_type=args.radiation,
            reference_db_root=Path(args.base_db_root).resolve() if args.base_db_root else None,
            overwrite=args.overwrite,
        )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest["$schema"] = LIBRARY_SCHEMA
    portable_db: dict[str, str] = {}
    for key, value in result.db_config.items():
        try:
            relative = Path(value).resolve().relative_to(result.pack_root.resolve()).as_posix()
        except ValueError:
            relative = Path(value).name
        portable_db[key] = relative
    manifest.update(
        {
            "library_mode": args.library_mode,
            "radiation": args.radiation,
            "phase_count": len(result.phase_ids),
            "archive_root": result.pack_root.name,
            "phase_ids": list(result.phase_ids),
            "failures": list(result.failures),
            "database_config": portable_db,
        }
    )
    atomic_write_json(args.manifest_output, manifest)
    if args.archive:
        _archive_directory(result.pack_root, Path(args.archive).resolve())
    print(f"Built {args.library_mode} library with {len(result.phase_ids)} phase(s)")
    return 0


def _summary_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary") or {}
    live = summary.get("live_run") or {}
    timings = live.get("timings") or {}
    return {
        "run_name": payload.get("run_name", path.stem),
        "analysis_mode": payload.get("analysis_mode", "unknown"),
        "status": payload.get("status", "unknown"),
        "best_rwp": (payload.get("provenance") or {}).get("source_manifest", {}).get("metrics", {}).get("best_rwp"),
        "total_seconds": timings.get("total_seconds"),
        "phase_count": len(payload.get("phases") or []),
        "hypothesis_count": len(payload.get("hypotheses") or []),
        "gpx_count": len(payload.get("gpx_projects") or []),
        "source": str(path),
    }


def command_compare(args: argparse.Namespace) -> int:
    paths = [Path(path).resolve() for path in args.summary]
    if args.summary_dir:
        paths.extend(sorted(Path(args.summary_dir).resolve().glob("*.json")))
    rows = [_summary_metrics(path) for path in paths]
    if not rows:
        raise ValueError("No RADAR-PD result summaries were supplied")
    fields = list(rows[0])
    with Path(args.csv_output).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    table_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields) + "</tr>"
        for row in rows
    )
    headers = "".join(f"<th>{html.escape(field.replace('_', ' ').title())}</th>" for field in fields)
    Path(args.html_output).write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><title>RADAR-PD series</title>"
        "<style>body{font-family:Arial;margin:2rem}table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #ccd8d1;padding:.5rem}th{background:#e9f3ee}</style></head>"
        f"<body><h1>RADAR-PD series comparison</h1><table><tr>{headers}</tr>{table_rows}</table></body></html>",
        encoding="utf-8",
    )
    return 0


def command_collect(args: argparse.Namespace) -> int:
    collect_outputs(
        args.run_dir,
        args.output_dir,
        mode=args.mode,
        run_name=args.run_name,
        project_root=PROJECT_ROOT,
        include_archive=not args.no_archive,
        status=args.status,
    )
    return 0


def _add_config_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=("rapid", "full"), default="rapid")
    parser.add_argument("--radiation", choices=("neutron", "xray"), default="neutron")
    parser.add_argument("--instrument-mode", choices=("auto", "cw", "tof"), default="auto")
    parser.add_argument("--allowed-elements", required=True)
    parser.add_argument("--environment-elements", default="")
    parser.add_argument("--limit-start", type=float)
    parser.add_argument("--limit-end", type=float)
    parser.add_argument("--exclude-region", action="append", default=[])
    parser.add_argument("--background-mode", default="auto_fixed_points")
    parser.add_argument("--background-type", default="chebyschev-1")
    parser.add_argument("--background-terms", type=int, default=6)
    parser.add_argument("--max-passes", type=int, default=3)
    parser.add_argument("--min-phase-percent", type=float, default=0.5)
    parser.add_argument("--phases-per-hypothesis", type=int, default=3)
    parser.add_argument("--gsas-validation-limit", type=int, default=10)
    parser.add_argument("--magnetic-precheck", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--main-prenudge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--main-shadow-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--main-cleanup", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--refine-uiso", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--refine-positions", action=argparse.BooleanOptionalAction, default=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    configure = sub.add_parser("configure", help="Create a reusable path-independent NDIP config")
    _add_config_options(configure)
    configure.add_argument("--output", required=True)
    configure.add_argument("--report")
    configure.set_defaults(handler=command_configure)

    analyze = sub.add_parser("analyze", help="Run RADAR-PD and normalize outputs for Galaxy")
    analyze.add_argument("--config", required=True)
    analyze.add_argument("--data", required=True)
    analyze.add_argument("--instrument")
    analyze.add_argument("--instrument-preset", choices=("xray_lab_cuka",))
    analyze.add_argument("--main-cif")
    analyze.add_argument("--db-root")
    analyze.add_argument("--db-pack")
    analyze.add_argument("--mode", choices=("auto", "rapid", "full"), default="auto")
    analyze.add_argument("--radiation", choices=("neutron", "xray"), default="neutron")
    analyze.add_argument("--run-name")
    analyze.add_argument("--input-source", choices=("history", "ipts", "collection", "previous_run", "api"), default="history")
    analyze.add_argument("--work-dir", required=True)
    analyze.add_argument("--output-dir", required=True)
    analyze.add_argument("--dry-run", action="store_true")
    analyze.add_argument("--no-archive", action="store_true")
    analyze.set_defaults(handler=command_analyze)

    resolve = sub.add_parser("resolve-ipts", help="Resolve a reduced powder input from IPTS metadata")
    resolve.add_argument("--event-file")
    resolve.add_argument("--instrument")
    resolve.add_argument("--ipts")
    resolve.add_argument("--run")
    resolve.add_argument("--bank")
    resolve.add_argument("--facility-root", default="/SNS")
    resolve.add_argument("--pattern-glob", action="append", default=["shared/**/*{run}*{bank}*.gsa", "shared/**/*{run}*{bank}*.xye", "shared/**/*{run}*{bank}*.dat"])
    resolve.add_argument("--instrument-glob", action="append", default=["shared/**/*{run}*{bank}*.instprm", "shared/**/*{instrument}*.instprm"])
    resolve.add_argument("--pattern-output", required=True)
    resolve.add_argument("--instrument-output", required=True)
    resolve.add_argument("--metadata-output", required=True)
    resolve.set_defaults(handler=command_resolve_ipts)

    library = sub.add_parser("build-library", help="Build a custom RADAR-PD CIF library")
    library.add_argument("--cif", action="append", default=[])
    library.add_argument("--cif-dir")
    library.add_argument("--library-mode", choices=("mini", "augmented"), default="mini")
    library.add_argument("--radiation", choices=("neutron", "xray"), default="neutron")
    library.add_argument("--base-db-root")
    library.add_argument("--output-root", required=True)
    library.add_argument("--manifest-output", required=True)
    library.add_argument("--archive")
    library.add_argument("--overwrite", action="store_true")
    library.set_defaults(handler=command_build_library)

    compare = sub.add_parser("compare-series", help="Compare a collection of normalized RADAR-PD results")
    compare.add_argument("--summary", action="append", default=[])
    compare.add_argument("--summary-dir")
    compare.add_argument("--csv-output", required=True)
    compare.add_argument("--html-output", required=True)
    compare.set_defaults(handler=command_compare)

    collect = sub.add_parser("collect", help="Normalize an existing RADAR-PD run folder")
    collect.add_argument("--run-dir", required=True)
    collect.add_argument("--output-dir", required=True)
    collect.add_argument("--mode", choices=("rapid", "full"), required=True)
    collect.add_argument("--run-name")
    collect.add_argument("--status", choices=("complete", "partial", "failed"), default="complete")
    collect.add_argument("--no-archive", action="store_true")
    collect.set_defaults(handler=command_collect)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as exc:
        print(f"[NDIP ERROR] {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
