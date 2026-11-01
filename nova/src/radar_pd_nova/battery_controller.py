"""Battery workspace state and asynchronous read-only input qualification."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import zipfile

from .battery.discovery import discover_proposal, family_runs, inspect_run
from .battery.electrochemistry import CyclerBinding, MAX_BYTES, read_cycler, summarize_exposure
from .battery.timeline import timeline_figure
from .battery.catalog import catalog_client, compare_acquisition, unavailable_context
from .facility_metadata import ExperimentMetadataIndex
from .battery_preparation import BatteryPreparationMixin
from .battery_campaign_controller import BatteryCampaignMixin


def acquisition_row(record):
    fields = record.get("fields", {})
    get = lambda name: fields.get(name, {}).get("value", "")
    identity = record.get("identity", {})
    exposure = record.get("exposure", {})
    issues = [item["message"] for item in record.get("issues", [])]
    issues.extend(identity.get("metadata_notices", []))
    if identity.get("identity_warning"): issues.append(identity["identity_warning"])
    if identity.get("identity_status") != "resolved":
        issues.append("Sample identity is unresolved; do not attach a cycler yet")
    comparison = record.get('catalog_comparison', {})
    if comparison.get('status') == 'conflict': issues.append(comparison['message'])
    return {"run": record["run"], "sample_id": identity.get("sample_id", ""),
            "sample": str(get("sample_name")), "title": str(get("run_title")),
            "start": exposure.get("start_utc", "Unavailable"),
            "duration": round(exposure["duration_seconds"], 3) if "duration_seconds" in exposure else None,
            "state": record["status"], "issues": " · ".join(issues) or "Metadata available · sample ID resolved",
            "catalog": comparison.get('status','not_requested').replace('_',' '),
            "voltage": "—", "coverage": "Not joined", "branch": "—"}


def read_candidate(inventory, candidate):
    """Read a selected inventoried source; archive members stay in memory."""
    if candidate not in inventory.get("cycler_candidates", []):
        raise ValueError("Cycler selection is no longer in the current inventory")
    root = Path(inventory["root"]).resolve()
    path = (root / candidate["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Cycler source resolves outside the selected IPTS")
    before = path.stat()
    if (before.st_size, before.st_mtime_ns) != (candidate["size_bytes"], candidate["mtime_ns"]):
        raise ValueError("Cycler source changed after discovery; refresh the proposal")
    if "member" in candidate:
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo(candidate["member"])
            if info.file_size > MAX_BYTES or info.file_size != candidate["member_size_bytes"] or f"{info.CRC:08x}" != candidate["member_crc32"]:
                raise ValueError("Cycler archive member changed or exceeds the import budget")
            with archive.open(info) as stream: raw = stream.read(MAX_BYTES + 1)
    else:
        if before.st_size > MAX_BYTES:
            raise ValueError("Cycler source exceeds the 64 MiB import budget")
        with path.open("rb") as stream: raw = stream.read(MAX_BYTES + 1)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Cycler source changed during import; wait for a completed file")
    return raw


class BatteryMixin(BatteryPreparationMixin, BatteryCampaignMixin):
    def _init_battery(self):
        state = self.server.state
        state.battery_instrument = "NOM"
        state.battery_ipts = ""
        state.battery_busy = False
        state.battery_message = "Choose an IPTS to find diffraction, sample metadata and possible cycler records."
        state.battery_error = ""
        state.battery_issues = []
        state.battery_families = []
        state.battery_family = ""
        state.battery_cycler_options = []
        state.battery_cycler = ""
        state.battery_metadata_sources = []
        state.battery_rows = []
        state.battery_sample_options = []
        state.battery_sample = ""
        state.battery_cell = ""
        state.battery_channel = ""
        state.battery_positive = ""
        state.battery_origin = ""
        state.battery_date_format = ""
        state.battery_timezone = ""
        state.battery_offset = 0.
        state.battery_clock_uncertainty = None
        state.battery_gap = 120.
        state.battery_cycler_path = ""
        state.battery_echem_source = "archive"
        state.battery_echem_summary = ""
        state.battery_timeline = {"data": [], "layout": {}}
        state.battery_inspection_limit = 12
        state.battery_total_runs = 0
        state.battery_export_ready = False
        state.battery_catalog_message = 'Catalog context is read with the selected proposal and acquisitions.'
        state.battery_catalog_proposal_message = ''
        state.battery_catalog_title = ''; state.battery_catalog_abstract = ''
        state.battery_catalog_source = ''; state.battery_catalog_issues = []
        state.battery_catalog_rows = []
        self._battery_catalog_client = catalog_client()
        self._battery_catalog_proposal = None
        self._battery_inventory = None
        self._battery_acquisitions = []
        self._battery_series = None
        self._battery_cycler_raw = None
        self._battery_joins = []
        self._battery_generation = 0
        self._battery_binding_revision = 0
        self._battery_task = None
        self._init_battery_preparation()
        self._init_battery_portfolio()
        self.server.state.change("battery_instrument", "battery_ipts")(self._battery_context_changed)
        self.server.state.change("battery_family")(self._battery_family_changed)
        self.server.state.change("battery_sample", "battery_cell", "battery_channel", "battery_positive", "battery_origin",
            "battery_offset", "battery_clock_uncertainty", "battery_gap", "battery_cycler", "battery_cycler_path", "battery_echem_source", "battery_date_format", "battery_timezone")(self._battery_binding_changed)

    def open_battery_workspace(self, **_):
        self.server.state.workspace_view = "battery"
        self.server.state.setup_collapsed = True
        self.server.state.flush()

    def close_battery_workspace(self, **_):
        self.server.state.workspace_view = "monitor"
        self.server.state.setup_collapsed = False
        self.server.state.flush()

    def _battery_context_changed(self, **_):
        self._battery_generation += 1
        self._battery_inventory = None
        state = self.server.state
        state.battery_families = []; state.battery_family = ""
        state.battery_cycler_options = []; state.battery_cycler = ""
        state.battery_metadata_sources = []; state.battery_issues = []
        self._battery_catalog_proposal = None
        state.battery_catalog_title = ''; state.battery_catalog_abstract = ''
        state.battery_catalog_source = ''; state.battery_catalog_issues = []
        state.battery_catalog_proposal_message = ''
        state.battery_catalog_message = 'Discover the proposal to retrieve its catalog context.'
        state.battery_error = ""
        state.battery_cell = ""; state.battery_channel = ""; state.battery_origin = ""
        state.battery_date_format = ""; state.battery_timezone = ""
        state.battery_positive = ""; state.battery_cycler_path = ""
        state.battery_offset = 0.; state.battery_clock_uncertainty = None
        state.battery_message = "Discover this proposal to refresh its sources."
        self._battery_family_changed()

    def _battery_family_changed(self, **_):
        self._battery_generation += 1
        self._battery_acquisitions = []
        state = self.server.state
        state.battery_rows = []; state.battery_sample_options = []; state.battery_sample = ""
        state.battery_catalog_rows = []
        state.battery_catalog_message = ('Select a reduction family and inspect its acquisition context.'
            if self._battery_inventory else 'Discover the proposal to retrieve its catalog context.')
        state.battery_total_runs = 0
        self._battery_reset_preparation()
        self._battery_binding_changed()

    def _battery_binding_changed(self, **_):
        self._battery_binding_revision += 1
        self._battery_series = None; self._battery_joins = []
        self._battery_cycler_raw = None
        if self.server.state.battery_model_include_echem:
            self._battery_preparation_edited()
        self.server.state.battery_echem_summary = ""
        self.server.state.battery_timeline = {"data": [], "layout": {}}
        self.server.state.battery_rows = [acquisition_row(row) for row in self._battery_acquisitions]
        self.server.state.battery_export_ready = bool(self._battery_acquisitions or self._battery_inventory)

    @staticmethod
    def _battery_root():
        return Path(os.getenv("RADAR_PD_SNS_ROOT") or os.getenv("RADAR_PD_FACILITY_ROOT") or "/SNS")

    def _battery_start(self, coroutine):
        if self._battery_task is not None and not self._battery_task.done():
            coroutine.close()
            return
        self._battery_task = asyncio.create_task(coroutine)

    def _battery_running(self):
        return self._battery_task is not None and not self._battery_task.done()

    async def _battery_read_catalog(self, scope, instrument, ipts, *, run=None, location=None):
        # Optional catalog failures cannot discard independently read archive metadata.
        # Do not expose provider exceptions: they may contain a response or credential.
        try:
            if scope == 'proposal':
                return await asyncio.to_thread(self._battery_catalog_client.proposal, instrument, ipts)
            return await asyncio.to_thread(self._battery_catalog_client.acquisition,
                instrument, ipts, run, location=location)
        except Exception:
            return unavailable_context(scope, instrument, ipts, run=run)

    def discover_battery_proposal(self, payload=None, **_):
        if self._battery_running(): return
        state = self.server.state
        values = payload or {"instrument": state.battery_instrument, "ipts": state.battery_ipts}
        instrument, ipts = str(values["instrument"]).upper(), str(values["ipts"]).strip().upper()
        if ipts.isdigit(): ipts = "IPTS-" + ipts
        state.battery_instrument = instrument; state.battery_ipts = ipts
        state.flush()
        self._battery_context_changed()
        state.flush()
        self._battery_start(self._discover_battery_async(instrument, ipts, self._battery_generation))

    async def _discover_battery_async(self, instrument, ipts, generation):
        state = self.server.state
        state.battery_busy = True; state.battery_error = ""
        state.battery_message = "Inspecting known reduction layouts and shallow auxiliary folders…"; state.flush()
        try:
            result = await asyncio.to_thread(discover_proposal, instrument, ipts, facility_root=self._battery_root())
            if generation != self._battery_generation: return
            self._battery_inventory = result
            state.battery_families = [{"title": f"{r['folder']} · {r['run_count']} runs · {', '.join(r['formats'])}", "value": r["folder"]} for r in result["product_families"]]
            state.battery_cycler_options = [{"title": r["path"] + (" :: " + r["member"] if "member" in r else ""), "value": str(i)} for i, r in enumerate(result["cycler_candidates"])]
            state.battery_metadata_sources = [dict(row, kind=kind) for key, kind in [
                ("metadata_tables", "Scan table"), ("profile_candidates", "Peak profile · review required"),
                ("reduction_setups", "IDL reduction setup · evidence only"),
                ("calibration_candidates", "Detector calibration · not a GSAS peak profile")]
                for row in result.get(key, [])]
            state.battery_issues = result["issues"]
            state.battery_message = f"Found {len(result['product_families'])} reduction families and {len(result['cycler_candidates'])} possible cycler sources. Select a family; sources are not yet bound to a cell."
            if result["status"] == "unavailable": state.battery_error = result["issues"][0]["message"]
            state.battery_catalog_message = 'Reading proposal context from ONCat…'; state.flush()
            self._battery_catalog_client.begin_batch()
            catalog = await self._battery_read_catalog('proposal', instrument, ipts)
            if generation != self._battery_generation: return
            self._battery_catalog_proposal = catalog
            state.battery_catalog_title = catalog.get('fields',{}).get('title','')
            state.battery_catalog_abstract = catalog.get('fields',{}).get('abstract','')
            state.battery_catalog_source = ' · '.join(filter(None,[catalog.get('source_url'),catalog.get('retrieved_utc')]))
            state.battery_catalog_proposal_message = catalog['message']
            state.battery_catalog_message = 'Select a reduction family and inspect its acquisition context.'
            state.battery_catalog_issues = catalog.get('issues',[])
            state.battery_export_ready = True
        except Exception as exc:
            if generation == self._battery_generation: state.battery_error = str(exc)
        finally:
            state.battery_busy = False; state.flush()

    def inspect_battery_acquisitions(self, **_):
        if self._battery_running(): return
        inventory = self._battery_inventory
        if not inventory:
            self.server.state.battery_error = "Discover a proposal first"; return
        family = self.server.state.battery_family
        try:
            value = self.server.state.battery_inspection_limit
            if isinstance(value, bool) or not str(value).isdigit():
                raise ValueError("Whole number required")
            limit = int(value)
        except (ValueError, TypeError):
            self.server.state.battery_error = "Choose a whole number of runs between 1 and 200"
            self.server.state.flush(); return
        if not 1 <= limit <= 200:
            self.server.state.battery_error = "Choose between 1 and 200 representative runs"
            self.server.state.flush(); return
        self._battery_family_changed()
        self.server.state.flush()
        self._battery_start(self._inspect_battery_async(inventory, family, max(1, min(limit, 200)), self._battery_generation))

    async def _inspect_battery_async(self, inventory, family, limit, generation):
        state = self.server.state
        state.battery_busy = True; state.battery_error = ""; state.flush()
        try:
            runs = await asyncio.to_thread(family_runs, inventory, family)
            chosen = runs if len(runs) <= limit else [runs[round(i*(len(runs)-1)/(limit-1))] for i in range(limit)] if limit>1 else runs[-1:]
            records = []
            self._battery_catalog_client.begin_batch()
            metadata = ExperimentMetadataIndex(inventory["instrument"], inventory["ipts"], root=Path(inventory["root"]))
            for index, run in enumerate(chosen):
                if generation != self._battery_generation: return
                state.battery_message = f"Reading metadata {index+1}/{len(chosen)} · run {run}. Detector events are not loaded."; state.flush()
                try:
                    record = await asyncio.to_thread(inspect_run, inventory["instrument"], inventory["ipts"], run, facility_root=self._battery_root())
                except (OSError,ValueError) as exc:
                    record=dict(run=run,status='unavailable',fields={},issues=[dict(code='acquisition_unavailable',message=str(exc))])
                try:
                    record['identity'] = await asyncio.to_thread(metadata.enrich, run, record.get('identity',{}))
                except (OSError,ValueError) as exc:
                    record.setdefault('issues',[]).append(dict(code='sample_metadata_unavailable',message=str(exc)))
                    if record['status']=='available':record['status']='partial'
                if generation != self._battery_generation: return
                location = None
                if record.get('path') and record.get('source_signature'):
                    try:
                        relative = Path(record['path']).relative_to(Path(inventory['root']))
                        location = f"/SNS/{inventory['instrument']}/{inventory['ipts']}/"+relative.as_posix()
                    except ValueError: pass
                catalog = await self._battery_read_catalog('run',
                    inventory['instrument'],inventory['ipts'],run=run,location=location)
                record['catalog'] = catalog
                record['catalog_comparison'] = compare_acquisition(record,catalog,self._battery_catalog_proposal)
                records.append(record)
            if generation != self._battery_generation: return
            self._battery_acquisitions = records
            self._battery_model_acquisitions()
            state.battery_rows = [acquisition_row(r) for r in records]
            def display(value):
                text = str(value) if value is not None else 'Unavailable in file metadata'
                return text if len(text)<=300 else text[:297]+'…'
            state.battery_catalog_rows = [dict(run=r['run'],field=f['field'].replace('_',' '),
                archive=display(f['archive_value']),catalog=display(f['catalog_value']),status=f['status'].replace('_',' '))
                for r in records for f in r['catalog_comparison']['fields']]
            missing = sum(not r['catalog'].get('fields') for r in records)
            if missing:
                state.battery_catalog_message = f'Catalog acquisition fields unavailable for {missing}/{len(records)} inspected runs. '+next(
                    r['catalog']['message'] for r in records if not r['catalog'].get('fields'))
            else:
                conflicts = sum(r['catalog_comparison']['status']=='conflict' for r in records)
                state.battery_catalog_message = f'Catalog context retrieved for {len(records)} inspected runs; {conflicts} with recorded disagreements. File-derived binding is retained.'
            samples = sorted({r.get("identity", {}).get("sample_id", "") for r in records} - {""})
            state.battery_sample_options = [{"title": value + " · " + " / ".join(sorted({
                str(r.get("fields", {}).get("run_title", {}).get("value", ""))
                for r in records if r.get("identity", {}).get("sample_id") == value})[:2]), "value": value} for value in samples]
            state.battery_total_runs = len(runs)
            state.battery_message = f"Inspected {len(records)} of {len(runs)} runs across this family. Review sample IDs and titles: references and other cells may share the same folder."
            state.battery_export_ready = True
        except Exception as exc:
            if generation == self._battery_generation: state.battery_error = str(exc)
        finally:
            state.battery_busy = False; state.flush()

    def join_battery_cycler(self, payload=None, **_):
        if self._battery_running(): return
        state = self.server.state
        if not self._battery_acquisitions:
            state.battery_error = "Inspect acquisitions before binding electrochemistry"; state.flush(); return
        values = payload or {key: getattr(state, "battery_"+key) for key in (
            "sample", "cell", "channel", "positive", "origin", "offset", "clock_uncertainty", "gap", "date_format", "timezone")}
        for key, value in values.items():
            if key in {"sample", "cell", "channel", "positive", "origin", "offset", "clock_uncertainty", "gap", "date_format", "timezone"}:
                setattr(state, "battery_"+key, value)
        state.flush()
        try:
            binding = CyclerBinding(str(values["sample"]), str(values["cell"]), str(values["channel"]),
                str(values["positive"]), str(values["origin"]), float(values["offset"]),
                float(values["clock_uncertainty"]) if values["clock_uncertainty"] not in (None, "") else None, float(values["gap"]),
                local_time_format=str(values.get("date_format", "")), timezone_name=str(values.get("timezone", "")))
            binding.validate()
            if binding.sample_id not in {r.get("identity", {}).get("sample_id") for r in self._battery_acquisitions}:
                raise ValueError("Select a sample verified in the inspected acquisitions")
        except (ValueError, TypeError) as exc:
            state.battery_error = str(exc); state.flush(); return
        selection = (state.battery_echem_source, state.battery_cycler, state.battery_cycler_path)
        self._battery_start(self._join_battery_async(binding, selection, self._battery_generation, self._battery_binding_revision))

    async def _join_battery_async(self, binding, selection, generation, revision):
        state = self.server.state
        state.battery_busy = True; state.battery_error = ""; state.flush()
        try:
            source, candidate_id, upload = selection
            if source == "archive":
                if not str(candidate_id).isdigit(): raise ValueError("Choose a discovered cycler source")
                candidate = self._battery_inventory["cycler_candidates"][int(candidate_id)]
                raw = await asyncio.to_thread(read_candidate, self._battery_inventory, candidate)
                name = candidate.get("member") or candidate["path"]
            elif source == "upload":
                path = getattr(getattr(self, "_battery_upload", None), "current_path", None)
                if path is None or str(path) != upload or not path.is_file() or path.stat().st_size > MAX_BYTES:
                    raise ValueError("Upload a completed cycler file smaller than 64 MiB")
                raw = await asyncio.to_thread(path.read_bytes); name = path.name
            else:
                raise ValueError("Choose an archive or uploaded cycler source")
            series = await asyncio.to_thread(read_cycler, raw, binding, source_name=name)
            if generation != self._battery_generation or revision != self._battery_binding_revision: return
            joins, rows = [], []
            for record in self._battery_acquisitions:
                row = acquisition_row(record)
                if row["sample_id"] == binding.sample_id and record.get("exposure") and record["status"] != "conflict":
                    exposure = record["exposure"]
                    joined = summarize_exposure(series, exposure["start_utc"], exposure["end_utc"], sample_id=row["sample_id"])
                    joins.append({"run": record["run"], **joined})
                    row["voltage"] = f"{joined['voltage_v']['mean']:.4f} V" if joined["voltage_v"] else "Unavailable"
                    current = f"{joined['current_coverage_fraction']:.1%}" if joined['current_status'] != 'unavailable' else "unavailable"
                    row["coverage"] = f"Voltage {joined['coverage_fraction']:.1%}; current {current}"
                    row["branch"] = ", ".join(joined["branches"]) or "Unavailable"
                rows.append(row)
            self._battery_series = series; self._battery_joins = joins; self._battery_cycler_raw = raw
            self._battery_preparation_edited()
            state.battery_model_include_echem=True
            if self._battery_model_sample==binding.sample_id and not state.battery_model_cell:
                state.battery_model_cell=binding.cell_id
            state.battery_rows = rows
            state.battery_timeline = timeline_figure(series, joins)
            summary = series.summary()
            state.battery_echem_summary = f"{summary['point_count']:,} points · {summary['first_utc']} to {summary['last_utc']} · {len(joins)} matched exposures. Recorded current: {summary['current_status']}. Typical log interval: {summary['median_recording_interval_seconds']:.3g} s; longest: {summary['largest_recording_interval_seconds']:.3g} s. Means and charge describe covered intervals only; no SOC or Li occupancy is inferred."
            state.battery_echem_summary += (" Clock uncertainty is unknown." if binding.clock_uncertainty_seconds is None
                else f" Reviewed clock uncertainty: {binding.clock_uncertainty_seconds:g} s.")
            if summary['median_recording_interval_seconds'] > binding.maximum_gap_seconds:
                state.battery_echem_summary += " The allowed gap is shorter than the typical recording interval; review the coverage before changing that limit."
            state.battery_export_ready = True
        except Exception as exc:
            if generation == self._battery_generation and revision == self._battery_binding_revision: state.battery_error = str(exc)
        finally:
            state.battery_busy = False; state.flush()

    def export_battery_evidence(self, **_):
        data = {"schema": "radar-battery-workspace/v1", "inventory": self._battery_inventory,
                "catalog_proposal": self._battery_catalog_proposal,
                "acquisitions": self._battery_acquisitions,
                "electrochemistry": self._battery_series.summary() if self._battery_series else None,
                "exposure_joins": self._battery_joins, "scientific_qualification": "not_evaluated"}
        self.download_file("battery-evidence.json", "application/json", json.dumps(data, indent=2, allow_nan=False).encode())
        return data
