"""Galaxy-backed execution and recovery for the RADAR-PD NOVA client."""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import requests
import yaml

from .configuration import config_from_contract, dump_configuration
from .models import AnalysisConfig, AnalysisMode, InputSelection, InputSource, RunRecord, RunStatus

ANALYZE_TOOL_ID = os.getenv("RADAR_PD_ANALYZE_TOOL_ID", "neutrons_radar_pd_analyze_prototype")
RUN_NAME_PREFIX = "RADAR-PD NOVA"


def _decode_parameter(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{\"":
        return value
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return value


def _parameter_items(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten Galaxy's several job-parameter encodings without losing parent values."""

    decoded = _decode_parameter(value)
    items: list[tuple[str, Any]] = []
    if prefix:
        items.append((prefix, decoded))
    if isinstance(decoded, dict):
        if "name" in decoded and "value" in decoded:
            child = str(decoded["name"])
            path = f"{prefix}.{child}" if prefix else child
            items.extend(_parameter_items(decoded["value"], path))
        else:
            for key, child in decoded.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                items.extend(_parameter_items(child, path))
    elif isinstance(decoded, list):
        for index, child in enumerate(decoded):
            path = f"{prefix}.{index}" if prefix else str(index)
            items.extend(_parameter_items(child, path))
    return items


def _normalized_parameter_path(value: str) -> str:
    return re.sub(r"[|/]+", ".", value).strip(".").lower()


def _parameter_value(parameters: Any, *suffixes: str) -> Any:
    normalized_suffixes = tuple(_normalized_parameter_path(item) for item in suffixes)
    for path, value in _parameter_items(parameters):
        normalized = _normalized_parameter_path(path)
        if any(normalized == suffix or normalized.endswith(f".{suffix}") for suffix in normalized_suffixes):
            return value
    return None


def _scalar(value: Any) -> Any:
    decoded = _decode_parameter(value)
    if isinstance(decoded, dict):
        for key in ("value", "values"):
            if key in decoded:
                return _scalar(decoded[key])
        return None
    if isinstance(decoded, list):
        return _scalar(decoded[0]) if decoded else None
    return decoded


def _text(value: Any) -> str | None:
    scalar = _scalar(value)
    if scalar is None:
        return None
    text = str(scalar).strip()
    return text if text and text not in {"None", "Not available."} else None


def _boolean(value: Any) -> bool | None:
    scalar = _scalar(value)
    if isinstance(scalar, bool):
        return scalar
    text = str(scalar).strip().lower() if scalar is not None else ""
    if text.startswith("--no-") or text in {"false", "no", "0", "off", "none", ""}:
        return False if text else None
    if text.startswith("--") or text in {"true", "yes", "1", "on"}:
        return True
    return None


def _number(value: Any, kind: type[int] | type[float]) -> int | float | None:
    scalar = _scalar(value)
    if scalar is None:
        return None
    try:
        return kind(scalar)
    except (TypeError, ValueError):
        return None


def _dataset_id(value: Any) -> str | None:
    decoded = _decode_parameter(value)
    if isinstance(decoded, dict):
        if decoded.get("id"):
            return str(decoded["id"])
        for key in ("value", "values", "dataset"):
            if key in decoded:
                found = _dataset_id(decoded[key])
                if found:
                    return found
    elif isinstance(decoded, list):
        for child in decoded:
            found = _dataset_id(child)
            if found:
                return found
    return None


def stage_from_console(text: str, status: RunStatus) -> tuple[str, int]:
    """Infer a stable user-facing stage from the existing batch console log."""

    lowered = text.lower()
    if status == RunStatus.OK:
        return "Results ready", 100
    if status in {RunStatus.ERROR, RunStatus.CANCELLED}:
        return "Stopped", 100
    stages = [
        (("final refinement", "gsas validation", "polish"), "Final refinement ranking", 88),
        (("512", "pattern scoring", "rerank"), "Pattern scoring", 72),
        (("pearson", "lattice nudge", "nudg"), "Lattice nudging", 55),
        (("coarse", "histogram", "ml rank"), "Candidate search", 35),
        (("main phase", "background", "signal"), "Signal and main-phase preparation", 18),
    ]
    for needles, label, progress in stages:
        if any(needle in lowered for needle in needles):
            return label, progress
    if status == RunStatus.RUNNING:
        return "Starting RADAR-PD", 8
    return "Waiting for compute", 3


def normalize_status(value: Any) -> RunStatus:
    text = str(getattr(value, "value", value)).lower()
    if any(token in text for token in ("finish", "ok", "complete")):
        return RunStatus.OK
    if any(token in text for token in ("error", "fail")):
        return RunStatus.ERROR
    if any(token in text for token in ("cancel", "delete", "stop")):
        return RunStatus.CANCELLED
    if "run" in text:
        return RunStatus.RUNNING
    return RunStatus.QUEUED


class GalaxyService:
    """Thin service around nova-galaxy plus read-only Galaxy REST endpoints."""

    def __init__(
        self,
        galaxy_url: str | None = None,
        galaxy_key: str | None = None,
        history_id: str | None = None,
        *,
        output_root: str | Path = "/tmp/radar-pd-nova",
    ) -> None:
        self.galaxy_url = (galaxy_url or os.getenv("GALAXY_URL", "")).rstrip("/")
        self.galaxy_key = galaxy_key or os.getenv("GALAXY_API_KEY", "")
        self.history_id = history_id or os.getenv("HISTORY_ID", "")
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._tools: dict[str, Any] = {}
        self._lock = threading.RLock()

    def validate_connection(self) -> None:
        if not self.galaxy_url or not self.galaxy_key or not self.history_id:
            raise RuntimeError("GALAXY_URL, GALAXY_API_KEY, and HISTORY_ID are required")
        response = requests.get(
            f"{self.galaxy_url}/api/version",
            headers=self._headers,
            timeout=20,
        )
        response.raise_for_status()

    @property
    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.galaxy_key, "accept": "application/json"}

    @contextmanager
    def _store(self) -> Iterator[Any]:
        from nova.galaxy import Connection
        from nova.galaxy.data_store import Datastore

        connection = Connection(galaxy_url=self.galaxy_url, galaxy_key=self.galaxy_key)
        helper = connection.connect()
        store = Datastore(RUN_NAME_PREFIX, helper, self.history_id)
        helper.datastores.append(store)
        store.persist()
        try:
            yield store
        finally:
            helper.close()

    @staticmethod
    def _existing_dataset(dataset_id: str, store: Any, name: str) -> Any:
        from nova.galaxy import Dataset

        dataset = Dataset(name=name, force_upload=False)
        dataset.id = dataset_id
        dataset.store = store
        return dataset

    @staticmethod
    def _upload_dataset(path: str, store: Any, label: str) -> Any:
        from nova.galaxy import Dataset

        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"{label} does not exist: {source}")
        dataset = Dataset(path=str(source), name=f"RADAR-PD {label} | {source.name}")
        dataset.upload(store)
        return dataset

    def list_history_datasets(self, *, limit: int = 500) -> list[dict[str, str]]:
        """Return selectable, non-deleted datasets from the active history."""

        response = requests.get(
            f"{self.galaxy_url}/api/histories/{self.history_id}/contents",
            headers=self._headers,
            params={"v": "dev", "limit": limit, "order": "update_time-dsc"},
            timeout=30,
        )
        response.raise_for_status()
        rows = response.json()
        return [
            {
                "id": str(row.get("id")),
                "name": str(row.get("name") or row.get("hid") or "dataset"),
                "extension": str(row.get("extension") or row.get("file_ext") or "data"),
                "state": str(row.get("state") or ""),
            }
            for row in rows
            if row.get("history_content_type", "dataset") == "dataset"
            and not row.get("deleted")
            and str(row.get("state") or "ok") in {"ok", "deferred"}
        ]

    def _job_details(self, uid: str, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fetch the durable Galaxy job record used to recover a NOVA session."""

        response = requests.get(
            f"{self.galaxy_url}/api/jobs/{uid}",
            headers=self._headers,
            params={"full": "true"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        details = dict(fallback or {})
        if isinstance(payload, dict):
            details.update(payload)
        return details

    @staticmethod
    def _job_parameters(job: dict[str, Any]) -> Any:
        candidates = [job.get(key) for key in ("params", "parameters", "inputs", "job_inputs") if job.get(key)]
        if not candidates:
            return {}
        return candidates[0] if len(candidates) == 1 else candidates

    @staticmethod
    def _named_output_ids(value: Any) -> dict[str, str]:
        output_ids: dict[str, str] = {}
        decoded = _decode_parameter(value)
        if isinstance(decoded, dict):
            for name, item in decoded.items():
                identifier = _dataset_id(item)
                if identifier:
                    output_ids[str(name)] = identifier
        elif isinstance(decoded, list):
            for item in decoded:
                if not isinstance(item, dict):
                    continue
                name = item.get("name") or item.get("output_name") or item.get("label")
                identifier = _dataset_id(item)
                if name and identifier:
                    output_ids[str(name)] = identifier
        return output_ids

    def _job_output_ids(self, job: dict[str, Any]) -> dict[str, str]:
        output_ids: dict[str, str] = {}
        for key in ("outputs", "output_datasets", "output_collections"):
            output_ids.update(self._named_output_ids(job.get(key)))
        return output_ids

    def _dataset_document(self, dataset_id: str) -> dict[str, Any] | None:
        response = requests.get(
            f"{self.galaxy_url}/api/datasets/{dataset_id}/display",
            headers=self._headers,
            params={"raw": "true"},
            timeout=30,
        )
        response.raise_for_status()
        payload = yaml.safe_load(response.text)
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _job_metric(job: dict[str, Any], name: str) -> str | None:
        metrics = job.get("job_metrics") or {}
        if isinstance(metrics, dict):
            return _text(metrics.get(name))
        if isinstance(metrics, list):
            for metric in metrics:
                if not isinstance(metric, dict):
                    continue
                metric_name = str(metric.get("name") or metric.get("metric_name") or "")
                if metric_name == name:
                    return _text(metric.get("value"))
        return None

    @staticmethod
    def _elements(value: Any) -> list[str]:
        decoded = _decode_parameter(value)
        if isinstance(decoded, list):
            return [str(item).strip() for item in decoded if str(item).strip()]
        text = _text(decoded)
        return re.split(r"[\s,]+", text) if text else []

    @staticmethod
    def _regions(value: Any) -> list[tuple[float, float]]:
        decoded = _decode_parameter(value)
        if isinstance(decoded, dict):
            decoded = decoded.get("regions") or decoded.get("values") or []
        if not isinstance(decoded, list):
            return []
        regions: list[tuple[float, float]] = []
        for item in decoded:
            if isinstance(item, dict):
                start = _number(item.get("start"), float)
                end = _number(item.get("end"), float)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                start = _number(item[0], float)
                end = _number(item[1], float)
            else:
                continue
            if start is not None and end is not None and start < end:
                regions.append((float(start), float(end)))
        return regions

    def _config_from_parameters(self, parameters: Any) -> AnalysisConfig | None:
        sample_elements = self._elements(
            _parameter_value(parameters, "chemistry.sample_elements", "sample_elements")
        )
        if not sample_elements:
            return None

        kwargs: dict[str, Any] = {"sample_elements": sample_elements}
        text_fields = {
            "run_name": ("reproducibility.run_name", "run_name"),
            "mode": ("analysis.strategy.analysis_mode", "analysis.analysis_mode", "analysis_mode", "mode"),
            "radiation": ("measurement.radiation", "radiation"),
            "instrument_mode": ("measurement.instrument_mode", "instrument_mode"),
            "background_mode": ("background.background_mode", "background.mode", "background_mode"),
            "background_type": ("background.background_type", "background.type", "background_type"),
            "full_profile": ("analysis.strategy.full_profile", "full_profile"),
        }
        for field, aliases in text_fields.items():
            value = _text(_parameter_value(parameters, *aliases))
            if value:
                kwargs[field] = value.lower() if field in {"mode", "radiation", "instrument_mode"} else value

        environment = self._elements(
            _parameter_value(parameters, "chemistry.environment_elements", "environment_elements")
        )
        kwargs["environment_elements"] = environment

        start = _number(_parameter_value(parameters, "pattern.fit_start", "fit_start"), float)
        end = _number(_parameter_value(parameters, "pattern.fit_end", "fit_end"), float)
        if start is not None and end is not None and start < end:
            kwargs["limits"] = (float(start), float(end))
        regions = self._regions(_parameter_value(parameters, "pattern.exclude_regions", "exclude_regions"))
        if regions:
            kwargs["exclude_regions"] = regions

        boolean_fields = {
            "reference_masks_enabled": ("pattern.reference_masks_enabled", "reference_masks_enabled"),
            "include_cu_kbeta": ("pattern.include_cu_kbeta", "include_cu_kbeta"),
            "main_prenudge": ("safeguards.main_prenudge", "main_prenudge"),
            "main_shadow_filter": ("safeguards.main_shadow_filter", "main_shadow_filter"),
            "cleanup_enabled": ("cleanup.cleanup_enabled", "cleanup_enabled"),
            "refine_u_iso": ("cleanup.refine_u_iso", "refine_u_iso"),
            "refine_positions": ("cleanup.refine_positions", "refine_positions"),
            "magnetic_precheck": ("magnetic.magnetic_precheck", "magnetic_precheck"),
            "rapid_show_family_variants": ("rapid.show_family_variants", "show_family_variants"),
            "rapid_final_polish_enabled": ("rapid.final_polish_enabled", "final_polish_enabled"),
        }
        for field, aliases in boolean_fields.items():
            value = _boolean(_parameter_value(parameters, *aliases))
            if value is not None:
                kwargs[field] = value

        number_fields: dict[str, tuple[type[int] | type[float], tuple[str, ...]]] = {
            "background_terms": (int, ("background.background_terms", "background.terms", "background_terms")),
            "magnetic_q_max": (float, ("magnetic.magnetic_q_max", "magnetic.q_max", "magnetic_q_max")),
            "full_max_passes": (int, ("full.max_passes", "full_max_passes")),
            "full_min_phase_percent": (float, ("full.min_phase_percent", "full_min_phase_percent")),
            "full_top_n_ml": (int, ("full.top_n_ml", "full_top_n_ml")),
            "full_nudge_candidates": (int, ("full.nudge_candidates", "full_nudge_candidates")),
            "full_nudge_samples": (int, ("full.nudge_samples", "full_nudge_samples")),
            "full_nudge_representatives": (int, ("full.nudge_representatives", "full_nudge_representatives")),
            "full_compare_candidates": (int, ("full.compare_candidates", "full_compare_candidates")),
            "full_compare_cycles": (int, ("full.compare_cycles", "full_compare_cycles")),
            "rapid_phases_per_hypothesis": (int, ("rapid.phases_per_hypothesis", "phases_per_hypothesis")),
            "rapid_stage_output_limit": (int, ("rapid.stage_output_limit", "stage_output_limit")),
            "rapid_gsas_validation_limit": (int, ("rapid.gsas_validation_limit", "gsas_validation_limit")),
            "rapid_parallel_workers": (int, ("rapid.parallel_workers", "parallel_workers")),
        }
        for field, (kind, aliases) in number_fields.items():
            value = _number(_parameter_value(parameters, *aliases), kind)
            if value is not None:
                kwargs[field] = value
        try:
            config = AnalysisConfig(**kwargs)
            if "run_name" not in kwargs:
                config.run_name = ""
            return config
        except (TypeError, ValueError):
            return None

    def _config_from_job(
        self,
        job: dict[str, Any],
        parameters: Any,
        output_ids: dict[str, str],
    ) -> AnalysisConfig | None:
        dataset_ids = [output_ids.get("resolved_config")]
        dataset_ids.append(_dataset_id(_parameter_value(parameters, "configuration.config_file", "config_file")))
        for dataset_id in dataset_ids:
            if not dataset_id:
                continue
            try:
                payload = self._dataset_document(dataset_id)
                if payload and payload.get("$schema") == "radar-pd-config/v1":
                    config = config_from_contract(payload)
                    run_name = _text(_parameter_value(parameters, "reproducibility.run_name", "run_name"))
                    config.run_name = run_name or self._job_metric(job, "run_name") or ""
                    return config
            except Exception:
                continue
        return self._config_from_parameters(parameters)

    @staticmethod
    def _input_dataset_ids(parameters: Any) -> dict[str, str]:
        aliases = {
            "data": ("data_inputs.input_source.diffraction_pattern", "input_source.data", "diffraction_pattern", "data"),
            "instrument": (
                "data_inputs.input_source.instrument_source.instrument_file",
                "input_source.instrument_source.instrument",
                "instrument_file",
            ),
            "main_cif": ("data_inputs.main_cif", "main_cif"),
            "database_archive": ("library.database.database_archive", "database.database_archive", "database_archive"),
            "event_file": ("data_inputs.input_source.event_file", "input_source.ipts_lookup.event_file", "event_file"),
            "configuration": ("reproducibility.configuration_override.config_file", "configuration.config_file", "config_file"),
        }
        result: dict[str, str] = {}
        for name, paths in aliases.items():
            identifier = _dataset_id(_parameter_value(parameters, *paths))
            if identifier:
                result[name] = identifier
        return result

    def _inputs_from_parameters(self, parameters: Any) -> InputSelection | None:
        dataset_ids = self._input_dataset_ids(parameters)
        source_kind = (
            _text(
                _parameter_value(
                    parameters,
                    "data_inputs.input_source.source_kind",
                    "input_source.source_kind",
                    "source_kind",
                )
            )
            or "history"
        ).lower()
        instrument_kind = (
            _text(
                _parameter_value(
                    parameters,
                    "data_inputs.input_source.instrument_source.kind",
                    "input_source.instrument_source.instrument_kind",
                    "instrument_kind",
                )
            )
            or ""
        ).lower()
        kwargs: dict[str, Any] = {
            "main_cif_dataset_id": dataset_ids.get("main_cif"),
            "database_dataset_id": dataset_ids.get("database_archive"),
        }
        if source_kind in {"history", "upload", "galaxy"}:
            kwargs.update(
                source=InputSource.GALAXY,
                data_dataset_id=dataset_ids.get("data"),
                instrument_dataset_id=dataset_ids.get("instrument"),
                use_builtin_cuka=instrument_kind in {"builtin", "builtin_cuka", "cuka"},
            )
        elif source_kind in {"event", "ipts_event"} or dataset_ids.get("event_file"):
            kwargs.update(
                source=InputSource.IPTS_EVENT,
                event_dataset_id=dataset_ids.get("event_file"),
                bank=_text(_parameter_value(parameters, "data_inputs.input_source.bank", "input_source.ipts_lookup.bank", "bank")),
            )
        else:
            kwargs.update(
                source=InputSource.IPTS_MANUAL,
                facility_root=_text(_parameter_value(parameters, "facility_root")) or "/SNS",
                instrument=_text(
                    _parameter_value(
                        parameters,
                        "data_inputs.input_source.instrument",
                        "input_source.ipts_lookup.instrument",
                        "ipts_lookup.instrument",
                    )
                ),
                ipts=_text(_parameter_value(parameters, "ipts_lookup.ipts", "ipts")),
                run_number=_number(_parameter_value(parameters, "run_number"), int),
                bank=_text(_parameter_value(parameters, "bank")),
            )
        try:
            return InputSelection(**kwargs)
        except (TypeError, ValueError):
            return None

    def _dataset_for_input(
        self,
        *,
        path: str | None,
        dataset_id: str | None,
        store: Any,
        label: str,
    ) -> Any | None:
        if dataset_id:
            return self._existing_dataset(dataset_id, store, label)
        if path:
            return self._upload_dataset(path, store, label)
        return None

    def submit(self, config: AnalysisConfig, inputs: InputSelection) -> RunRecord:
        """Upload inputs and asynchronously start the existing Analyze tool."""

        from nova.galaxy import Parameters, Tool

        submitted_config = config.model_copy(deep=True)
        submitted_inputs = inputs.model_copy(deep=True)
        if inputs.use_builtin_cuka and (config.radiation.value != "xray" or config.instrument_mode == "tof"):
            raise ValueError("The built-in Cu K-alpha profile is only valid for X-ray CW data")
        work_dir = self.output_root / "submissions" / config.run_name
        work_dir.mkdir(parents=True, exist_ok=True)
        config_path = dump_configuration(config, work_dir / "radar_pd_config.yaml")

        with self._store() as store:
            params = Parameters()
            uploaded: dict[str, str] = {}
            config_ds = self._upload_dataset(str(config_path), store, "configuration")
            uploaded["configuration"] = config_ds.id
            params.add_input(name="configuration|config_kind", value="existing")
            params.add_input(name="configuration|config_file", value=config_ds)

            if inputs.source in {InputSource.UPLOAD, InputSource.GALAXY}:
                params.add_input(name="input_source|source_kind", value="history")
                data = self._dataset_for_input(
                    path=inputs.data_path,
                    dataset_id=inputs.data_dataset_id,
                    store=store,
                    label="diffraction data",
                )
                if data is None:
                    raise ValueError("A diffraction pattern is required")
                uploaded["data"] = data.id
                submitted_inputs.data_dataset_id = data.id
                params.add_input(name="input_source|data", value=data)
                if inputs.use_builtin_cuka:
                    params.add_input(name="input_source|instrument_source|instrument_kind", value="builtin_cuka")
                else:
                    instrument = self._dataset_for_input(
                        path=inputs.instrument_path,
                        dataset_id=inputs.instrument_dataset_id,
                        store=store,
                        label="instrument profile",
                    )
                    if instrument is None:
                        raise ValueError("An instrument profile is required")
                    uploaded["instrument"] = instrument.id
                    submitted_inputs.instrument_dataset_id = instrument.id
                    params.add_input(name="input_source|instrument_source|instrument_kind", value="upload")
                    params.add_input(name="input_source|instrument_source|instrument", value=instrument)
            else:
                params.add_input(name="input_source|source_kind", value="ipts")
                if inputs.source == InputSource.IPTS_EVENT:
                    params.add_input(name="input_source|ipts_lookup|lookup_kind", value="event")
                    event = self._dataset_for_input(
                        path=inputs.event_file_path,
                        dataset_id=inputs.event_dataset_id,
                        store=store,
                        label="NeXus event file",
                    )
                    if event is None:
                        raise ValueError("A NeXus event file is required")
                    uploaded["event_file"] = event.id
                    submitted_inputs.event_dataset_id = event.id
                    params.add_input(name="input_source|ipts_lookup|event_file", value=event)
                    params.add_input(name="input_source|ipts_lookup|bank", value=inputs.bank)
                else:
                    params.add_input(name="input_source|ipts_lookup|lookup_kind", value="manual")
                    params.add_input(name="input_source|ipts_lookup|instrument", value=inputs.instrument)
                    params.add_input(name="input_source|ipts_lookup|ipts", value=inputs.ipts)
                    params.add_input(name="input_source|ipts_lookup|run_number", value=inputs.run_number)
                    params.add_input(name="input_source|ipts_lookup|bank", value=inputs.bank)
                params.add_input(name="input_source|facility_root", value=inputs.facility_root)

            main_cif = self._dataset_for_input(
                path=inputs.main_cif_path,
                dataset_id=inputs.main_cif_dataset_id,
                store=store,
                label="main phase CIF",
            )
            if main_cif is not None:
                uploaded["main_cif"] = main_cif.id
                submitted_inputs.main_cif_dataset_id = main_cif.id
                params.add_input(name="main_cif", value=main_cif)

            database = self._dataset_for_input(
                path=inputs.database_archive_path,
                dataset_id=inputs.database_dataset_id,
                store=store,
                label="candidate library",
            )
            if database is None:
                params.add_input(name="database|database_kind", value="builtin")
            else:
                uploaded["database_archive"] = database.id
                submitted_inputs.database_dataset_id = database.id
                params.add_input(name="database|database_kind", value="custom")
                params.add_input(name="database|database_archive", value=database)
            params.add_input(name="run_name", value=config.run_name)

            tool = Tool(id=ANALYZE_TOOL_ID)
            tool.run(store, params, wait=False)
            uid = tool.get_uid()
            if not uid:
                raise RuntimeError("Galaxy accepted the request but did not return a job identifier")
            with self._lock:
                self._tools[uid] = tool
            return RunRecord(
                uid=uid,
                name=config.run_name,
                mode=config.mode,
                history_id=self.history_id,
                status=RunStatus.QUEUED,
                stage="Waiting for compute",
                progress=3,
                input_dataset_ids=uploaded,
                config=submitted_config,
                inputs=submitted_inputs,
            )

    def _recover_tool(self, uid: str) -> Any:
        from nova.galaxy import Tool

        with self._lock:
            if uid in self._tools:
                return self._tools[uid]
        with self._store() as store:
            tool = Tool(id=ANALYZE_TOOL_ID)
            tool.assign_id(uid, store)
            with self._lock:
                self._tools[uid] = tool
            return tool

    def refresh(self, record: RunRecord) -> RunRecord:
        with self._lock:
            tool = self._tools.get(record.uid)
        if tool is not None:
            status = normalize_status(tool.get_status())
            stdout = tool.get_stdout() or ""
            stderr = tool.get_stderr() or ""
        else:
            job = self._job_details(record.uid)
            status = normalize_status(job.get("state"))
            stdout = str(job.get("stdout") or job.get("tool_stdout") or "")
            stderr = str(job.get("stderr") or job.get("tool_stderr") or "")
            parameters = self._job_parameters(job)
            output_ids = self._job_output_ids(job)
            record.output_dataset_ids.update(output_ids)
            record.input_dataset_ids.update(self._input_dataset_ids(parameters))
            recovered_config = self._config_from_job(job, parameters, output_ids)
            if recovered_config is not None:
                record.config = recovered_config
                record.mode = recovered_config.mode
                if recovered_config.run_name:
                    record.name = recovered_config.run_name
            recovered_inputs = self._inputs_from_parameters(parameters)
            if recovered_inputs is not None:
                record.inputs = recovered_inputs
        stage, progress = stage_from_console(stdout, status)
        record.status = status
        record.stage = stage
        record.progress = progress
        record.console_tail = stdout[-12000:]
        record.updated_utc = datetime.now(timezone.utc).isoformat()
        if status == RunStatus.ERROR:
            record.message = (stderr or stdout or "Galaxy reported an error")[-4000:]
        return record

    def cancel(self, uid: str) -> None:
        self._recover_tool(uid).stop()

    def recent_runs(self, *, limit: int = 50) -> list[RunRecord]:
        """Recover RADAR-PD Analyze jobs directly from the active Galaxy history."""

        response = requests.get(
            f"{self.galaxy_url}/api/jobs",
            headers=self._headers,
            params={"history_id": self.history_id, "tool_id": ANALYZE_TOOL_ID, "order": "update_time-dsc", "limit": limit},
            timeout=30,
        )
        response.raise_for_status()
        records: list[RunRecord] = []
        for summary in response.json():
            uid = str(summary.get("id") or "")
            if not uid:
                continue
            try:
                job = self._job_details(uid, summary)
            except Exception:
                job = dict(summary)
            state = normalize_status(job.get("state"))
            parameters = self._job_parameters(job)
            output_ids = self._job_output_ids(job)
            config = self._config_from_job(job, parameters, output_ids)
            inputs = self._inputs_from_parameters(parameters)
            parameter_name = _text(_parameter_value(parameters, "reproducibility.run_name", "run_name"))
            run_name = config.run_name if config is not None and config.run_name else parameter_name
            run_name = run_name or self._job_metric(job, "run_name")
            command = str(job.get("command_line") or job.get("command") or "")
            if not run_name:
                match = re.search(r"--run-name\s+(?:'([^']+)'|\"([^\"]+)\"|(\S+))", command)
                run_name = next((group for group in match.groups() if group), uid) if match else uid
            mode_text = _text(
                _parameter_value(parameters, "analysis.strategy.analysis_mode", "analysis.analysis_mode", "analysis_mode")
            )
            if config is not None:
                mode = config.mode
            elif mode_text and mode_text.lower() in {item.value for item in AnalysisMode}:
                mode = AnalysisMode(mode_text.lower())
            else:
                mode = AnalysisMode.RAPID if "rapid" in command.lower() else AnalysisMode.FULL
            created = str(job.get("create_time") or job.get("created_utc") or datetime.now(timezone.utc).isoformat())
            updated = str(job.get("update_time") or job.get("updated_utc") or created)
            records.append(
                RunRecord(
                    uid=uid,
                    name=run_name,
                    mode=mode,
                    history_id=self.history_id,
                    status=state,
                    stage=stage_from_console("", state)[0],
                    progress=stage_from_console("", state)[1],
                    created_utc=created,
                    updated_utc=updated,
                    input_dataset_ids=self._input_dataset_ids(parameters),
                    output_dataset_ids=output_ids,
                    config=config,
                    inputs=inputs,
                )
            )
        return records

    def collect_results(self, record: RunRecord) -> RunRecord:
        """Download stable scalar outputs and collections when the job completes."""

        destination = self.output_root / "runs" / record.uid
        staging = destination.parent / f".{record.uid}-{threading.get_ident()}.partial"
        try:
            tool = self._recover_tool(record.uid)
            outputs = tool.get_results()
            if outputs is None:
                raise RuntimeError("Galaxy did not return the declared result outputs")
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)
            output_ids: dict[str, str] = {}
            downloaded = 0
            for name in (
                "report",
                "summary",
                "state",
                "resolved_config",
                "input_manifest",
                "gpx_index",
                "results_archive",
                "input_resolution_metadata",
                "console_output",
            ):
                try:
                    dataset = outputs.get_dataset(name)
                except Exception:
                    continue
                suffix = {
                    "report": ".html",
                    "summary": ".json",
                    "state": ".json",
                    "resolved_config": ".yaml",
                    "input_manifest": ".json",
                    "gpx_index": ".json",
                    "results_archive": ".zip",
                    "input_resolution_metadata": ".json",
                    "console_output": ".txt",
                }[name]
                target = staging / f"{name}{suffix}"
                dataset.download(str(target))
                downloaded += 1
                output_ids[name] = str(getattr(dataset, "id", ""))
            for name in ("plots", "tables", "phases", "gpx_projects", "diagnostics"):
                try:
                    collection = outputs.get_collection(name)
                except Exception:
                    continue
                target = staging / name
                target.mkdir(exist_ok=True)
                collection.download(str(target))
                downloaded += 1
                output_ids[name] = str(getattr(collection, "id", ""))
            if not downloaded:
                raise RuntimeError("Galaxy returned no downloadable RADAR-PD result artifacts")
            if destination.exists():
                shutil.rmtree(destination)
            staging.replace(destination)
            record.output_dataset_ids.update(output_ids)
            record.output_dir = str(destination)
            record.status = RunStatus.OK
            record.stage = "Results ready"
            record.progress = 100
            record.message = ""
        except Exception as exc:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            record.status = RunStatus.ERROR
            record.stage = "Result download failed"
            record.progress = 100
            record.message = f"Analysis finished, but result download failed: {exc}"
        record.updated_utc = datetime.now(timezone.utc).isoformat()
        return record

    def result_payload(self, record: RunRecord) -> dict[str, Any]:
        root = record.local_output_dir
        if root is None:
            return {}
        payload: dict[str, Any] = {"artifacts": []}
        for key in ("summary", "state", "gpx_index", "input_manifest"):
            path = root / f"{key}.json"
            if path.is_file():
                try:
                    payload[key] = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    payload[key] = {}
        for path in sorted(root.rglob("*")):
            if path.is_file():
                payload["artifacts"].append(
                    {
                        "name": path.name,
                        "kind": path.parent.name if path.parent != root else "result",
                        "path": str(path),
                        "size": path.stat().st_size,
                    }
                )
        return payload

    def monitor(
        self,
        record: RunRecord,
        callback: Callable[[RunRecord], None],
        *,
        poll_seconds: float = 5.0,
    ) -> None:
        """Blocking monitor intended for a background thread."""

        import time

        while record.status not in {RunStatus.OK, RunStatus.ERROR, RunStatus.CANCELLED}:
            try:
                record = self.refresh(record)
            except Exception as exc:
                record.message = str(exc)
            callback(record)
            if record.status in {RunStatus.OK, RunStatus.ERROR, RunStatus.CANCELLED}:
                break
            time.sleep(poll_seconds)
        if record.status == RunStatus.OK:
            try:
                record = self.collect_results(record)
            except Exception as exc:
                record.status = RunStatus.ERROR
                record.stage = "Result download failed"
                record.progress = 100
                record.message = f"Analysis finished, but result download failed: {exc}"
            callback(record)
