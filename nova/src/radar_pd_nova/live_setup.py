"""Self-contained Live setup, immutable launch recipes and a one-scan trial."""

from __future__ import annotations

import asyncio
import json
import zipfile
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from trame.widgets import html
from trame.widgets import vuetify3 as vuetify

from .configuration import load_configuration
from .library_setup import LIBRARY_FIELDS
from .models import AnalysisConfig, RunStatus
from .post_analysis import RECIPES, PostAnalysisSettings, ViewSettings
from .setup_forms import (
    CONFIG_FORM_FIELDS,
    configuration_form_values,
    configuration_from_form,
)
from .uploads import NamedFileUpload


def configuration_digest(config: AnalysisConfig) -> str:
    contract = config.portable_contract()
    contract.pop("created_utc", None)
    return sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def local_asset_signature(path: str) -> tuple[str, int, int]:
    source = Path(path)
    stat = source.stat()
    if not source.is_file():
        raise ValueError(f"{source.name}: not a regular file")
    return str(source.resolve()), stat.st_size, stat.st_mtime_ns


class LiveSetupMixin:
    def _init_live_setup(self):
        state = self.server.state
        for field in CONFIG_FORM_FIELDS:
            setattr(state, "live_" + field, deepcopy(getattr(state, field)))
        state.live_radiation = "neutron"
        state.live_instrument_mode = "tof"
        state.live_analysis_mode = "full"
        state.live_configuration_source = "inline"
        state.live_configuration_source_options = [
            {"title": "Configure here", "value": "inline"},
            {"title": "Reuse a saved configuration", "value": "history"},
        ]
        state.live_config_import_path = ""
        state.live_setup_panels = [0]
        state.live_launch_busy = False
        state.live_saved_configuration_id = ""
        state.live_test_bank = None
        state.live_test_bank_options = []
        state.live_test_message = ""
        state.live_post_analysis_selected = [
            v["kind"] for v in state.live_post_analysis["views"]
        ]
        for field in LIBRARY_FIELDS:
            default = (
                []
                if field in {"library_builder_elements", "library_builder_file_rows"}
                else 0
                if field == "library_builder_unknown_elements"
                else getattr(state, field, None)
            )
            setattr(state, "live_" + field, deepcopy(default))
        state.library_builder_elements = []
        state.library_builder_unknown_elements = 0
        self._live_saved_configurations = {}
        self._live_uploaded_assets = {}
        self._library_action_contexts = {}
        fields = ["live_" + f for f in CONFIG_FORM_FIELDS]
        fields += [
            "live_" + f
            for f in LIBRARY_FIELDS
            if f
            in {"database_source", "library_archive_source", "database_archive_path"}
        ]
        fields += ["live_configuration_source", "powgen_database_dataset_id"]
        self.server.state.change(*fields)(self._live_setup_changed)

    def _live_setup_changed(self, **_):
        state = self.server.state
        state.powgen_preflight_ready = False
        state.powgen_preflight_status = "idle"
        state.powgen_preflight_details = []
        state.powgen_preflight_message = (
            "Live settings changed. Check experiment and inputs again."
        )
        state.live_test_message = ""

    def _live_configuration(self):
        state = self.server.state
        # Old checkpoints/test harnesses without a draft can still select their
        # explicit configuration; there is no inheritance from Single pattern.
        source = getattr(state, "live_configuration_source", None) or "history"
        if source == "history":
            dataset_id = str(state.powgen_configuration_dataset_id or "").strip()
            if not dataset_id:
                raise ValueError(
                    "Choose a reusable RADAR-PD configuration, or create settings here"
                )
            config = self.service.load_configuration_dataset(dataset_id)
        elif source == "inline":
            payload = {
                name: deepcopy(getattr(state, "live_" + name))
                for name in CONFIG_FORM_FIELDS
            }
            source_main = str(state.powgen_main_cif_source or "none")
            payload.update(
                radiation="neutron",
                instrument_mode="tof",
                run_name=str(payload.get("run_name") or "").strip()
                or f"{self._live_adapter().instrument}_{self._normalize_powgen_ipts(state.powgen_ipts)}",
                main_cif_source={"computer": "upload", "galaxy": "galaxy"}.get(
                    source_main, "none"
                ),
                main_cif_path=str(state.powgen_main_cif_path or ""),
                history_main_cif_id=str(state.powgen_main_cif_dataset_id or ""),
                remote_main_cif_uri="",
                facility_main_cif_path="",
            )
            config = configuration_from_form(SimpleNamespace(**payload))
        else:
            raise ValueError("Choose how to configure the Live analysis")
        if config.radiation.value != "neutron" or config.instrument_mode == "cw":
            raise ValueError(
                "Live mode requires a neutron configuration with automatic or time-of-flight instrument mode"
            )
        return config

    def _live_analysis_setup(self):
        locked = "powgen_monitoring || live_launch_busy"
        vuetify.VSelect(
            label="Analysis setup",
            v_model=("live_configuration_source",),
            items=("live_configuration_source_options",),
            item_title="title",
            item_value="value",
            density="compact",
            variant="outlined",
            disabled=(locked,),
        )
        with html.Div(v_show="live_configuration_source === 'history'"):
            vuetify.VAutocomplete(
                label="Reusable RADAR-PD configuration",
                v_model=("powgen_configuration_dataset_id",),
                items=("history_configuration_datasets",),
                item_title="display_name",
                item_value="id",
                density="compact",
                variant="outlined",
                disabled=(locked,),
                clearable=True,
                no_data_text="No reusable configurations are in this Galaxy History",
                hint="Reuses analysis settings. Scans, instrument profile, candidate library and main phase come from this Live setup.",
                persistent_hint=True,
                update_modelValue=(self._powgen_configuration_changed, "[$event]"),
            )
            vuetify.VAlert(
                v_show="!!powgen_configuration_error",
                text=("powgen_configuration_error",),
                type="warning",
                variant="tonal",
                density="compact",
            )
            with html.Details(
                v_show="powgen_configuration_summary.length > 0",
                classes="radar-soft-panel mb-2",
            ):
                html.Summary("Saved analysis settings")
                with html.Div(
                    v_for="item in powgen_configuration_summary",
                    key="item.label",
                    classes="radar-preflight-row",
                ):
                    html.Span("{{ item.label }}", classes="radar-preflight-label")
                    html.Span("{{ item.value }}", classes="radar-preflight-value")
            vuetify.VBtn(
                "Edit a copy here",
                click=self.edit_live_configuration,
                disabled=(locked + " || !powgen_configuration_dataset_id",),
                variant="outlined",
                size="small",
                block=True,
            )
        with html.Fieldset(
            v_show="live_configuration_source === 'inline'",
            disabled=(locked,),
            classes="radar-plain-fieldset",
        ):
            html.P(
                "Neutron time-of-flight analysis. These settings belong to Live and are saved automatically when an analysis starts.",
                classes="radar-help-copy",
            )
            vuetify.VTextField(
                label="Sample elements",
                v_model=("live_sample_elements",),
                placeholder="Li, C, O, Al, Cu, Mn, Co, Ni",
                disabled=(locked,),
                density="compact",
                variant="outlined",
                hint="Include every sample element that the candidate search should allow.",
                persistent_hint=True,
            )
            html.P(
                "{{ live_library_builder_elements.length ? 'Suggested from input CIFs: ' + live_library_builder_elements.join(', ') + (live_library_builder_unknown_elements ? '. Some structures could not be read; check completeness.' : '. Review sample and environment assignments.') : '' }}",
                v_show="live_library_builder_elements.length > 0",
                classes="radar-help-copy",
            )
            vuetify.VBtn(
                "Use suggested elements",
                click=self.use_live_element_suggestion,
                v_show="live_library_builder_elements.length > 0",
                disabled=(locked,),
                variant="text",
                size="small",
            )
            vuetify.VSelect(
                label="Analysis mode",
                v_model=("live_analysis_mode",),
                items=("mode_options",),
                item_title="title",
                item_value="value",
                disabled=(locked,),
                density="compact",
                variant="outlined",
            )
            self._budget_controls("live_", locked=locked, compact=True)
            with html.Details(classes="radar-soft-panel mb-2"):
                html.Summary("Advanced analysis settings")
                vuetify.VTextField(
                    label="Can / environment elements",
                    v_model=("live_environment_elements",),
                    disabled=(locked,),
                    density="compact",
                    variant="outlined",
                    hint="Additional elements permitted for sample environment contributions.",
                    persistent_hint=True,
                )
                html.H4("Fit region")
                self._regions_controls("live_", locked=locked)
                html.H4("Background")
                self._background_controls("live_", locked=locked)
                html.H4("Refinement and candidate controls")
                main_ready = "((powgen_main_cif_source === 'computer' && !!powgen_main_cif_path) || (powgen_main_cif_source === 'galaxy' && !!powgen_main_cif_dataset_id))"
                self._expert_controls(
                    "live_", main_phase_ready=main_ready, locked=locked
                )
                vuetify.VSwitch(
                    v_model=("live_magnetic_precheck",),
                    disabled=(locked,),
                    v_show=main_ready,
                    label="Check residual peaks for commensurate magnetic indexing",
                    density="compact",
                )
                self._number_field(
                    label="Magnetic Q maximum",
                    v_model=("live_magnetic_q_max",),
                    disabled=(locked,),
                    v_show=f"{main_ready} && live_magnetic_precheck",
                    density="compact",
                    variant="outlined",
                )
                vuetify.VTextField(
                    label="Magnetic denominators",
                    v_model=("live_magnetic_denominators",),
                    disabled=(locked,),
                    v_show=f"{main_ready} && live_magnetic_precheck",
                    density="compact",
                    variant="outlined",
                )
                vuetify.VTextField(
                    label="Live analysis name (optional)",
                    v_model=("live_run_name",),
                    disabled=(locked,),
                    density="compact",
                    variant="outlined",
                )
            with html.Details(classes="radar-soft-panel mb-2"):
                html.Summary("Import a configuration file")
                NamedFileUpload(
                    "live_config_import_path",
                    label="RADAR-PD configuration",
                    help_text="Load a YAML or JSON configuration into the Live draft.",
                    extensions=[".yaml", ".yml", ".json"],
                    optional=True,
                    key="live-config-import-upload",
                )
                vuetify.VBtn(
                    "Apply to Live",
                    click=self.apply_live_uploaded_configuration,
                    disabled=(locked + " || !live_config_import_path",),
                    variant="outlined",
                    size="small",
                    block=True,
                )
        vuetify.VAutocomplete(
            v_show="live_configuration_source === 'inline'",
            label="Additional result plots",
            v_model=("live_post_analysis_selected",),
            items=("post_analysis_recipes",),
            multiple=True,
            chips=True,
            closable_chips=True,
            disabled=(locked,),
            density="compact",
            variant="outlined",
            hint="Choose views to update as scans finish.",
            persistent_hint=True,
            update_modelValue=(self.update_live_post_analysis_selection, "[$event]"),
        )

    def _apply_live_configuration(self, config):
        if config.radiation.value != "neutron" or config.instrument_mode == "cw":
            raise ValueError(
                "Live mode requires a neutron time-of-flight configuration"
            )
        state = self.server.state
        for name, value in configuration_form_values(config).items():
            setattr(state, "live_" + name, deepcopy(value))
        state.live_instrument_mode = "tof"
        state.live_configuration_source = "inline"
        state.live_post_analysis_selected = list(
            dict.fromkeys(v.kind for v in config.post_analysis.views)
        )
        self._live_setup_changed()

    def edit_live_configuration(self, **_):
        state = self.server.state
        if state.powgen_monitoring or state.live_launch_busy:
            return
        try:
            self._apply_live_configuration(
                self.service.load_configuration_dataset(
                    str(state.powgen_configuration_dataset_id or "")
                )
            )
            state.notice = "Copied saved analysis settings into Live. Single pattern was not changed."
            state.error_message = ""
        except Exception as exc:  # noqa: BLE001 - report Galaxy and parser failures in the UI
            state.error_message = f"Could not load Live configuration: {exc}"
        state.flush()

    def apply_live_uploaded_configuration(self, **_):
        state = self.server.state
        if state.powgen_monitoring or state.live_launch_busy:
            return
        try:
            self._apply_live_configuration(
                load_configuration(state.live_config_import_path)
            )
            state.error_message = ""
            state.notice = "Configuration loaded into the Live draft."
        except Exception as exc:  # noqa: BLE001 - preserve the draft on import failure
            state.error_message = f"Could not load Live configuration: {exc}"
        state.flush()

    def update_live_post_analysis_selection(self, selected=None, **_):
        state = self.server.state
        if (
            state.powgen_monitoring
            or state.live_launch_busy
            or state.live_configuration_source != "inline"
        ):
            return
        try:
            chosen = list(
                dict.fromkeys(
                    selected
                    if isinstance(selected, list)
                    else state.live_post_analysis_selected or []
                )
            )
            if any(kind not in RECIPES for kind in chosen):
                raise ValueError("Unknown post-analysis recipe")
            settings = PostAnalysisSettings.model_validate(state.live_post_analysis)
            views = [v for v in settings.views if v.kind in chosen]
            for kind in chosen:
                if not any(v.kind == kind for v in views):
                    views.append(ViewSettings(id=kind, kind=kind))
            settings.views = views
            state.live_post_analysis = settings.model_dump(mode="json")
            state.live_post_analysis_selected = chosen
        except ValueError as exc:
            state.error_message = str(exc)
        state.flush()

    def _use_live_element_suggestion(self, only_if_empty=False):
        state = self.server.state
        elements = list(state.live_library_builder_elements or [])
        incomplete = bool(state.live_library_builder_unknown_elements)
        if elements and not (
            only_if_empty
            and (str(state.live_sample_elements or "").strip() or incomplete)
        ):
            state.live_sample_elements = ", ".join(elements)
            state.notice = "Element suggestion copied from the input CIFs. Review sample and environment assignments."

    def use_live_element_suggestion(self, **_):
        if (
            not self.server.state.powgen_monitoring
            and not self.server.state.live_launch_busy
        ):
            self._use_live_element_suggestion()
            self.server.state.flush()

    def _live_library_selection(self):
        state = self.server.state
        source = getattr(state, "live_database_source", None)
        if not source:
            source = (
                "archive"
                if getattr(state, "powgen_database_dataset_id", None)
                else "builtin"
            )
        if source == "builtin":
            return {
                "dataset_id": None,
                "path": None,
                "label": "Built-in MP/COD catalog",
            }
        if source in {"custom_mini", "custom_augmented"}:
            raise ValueError(
                "Build and use the custom CIF library before checking the experiment"
            )
        if source != "archive":
            raise ValueError("Choose a candidate library for Live")
        archive_source = getattr(state, "live_library_archive_source", None) or "galaxy"
        if archive_source == "galaxy":
            dataset_id = str(state.powgen_database_dataset_id or "").strip()
            if not dataset_id:
                raise ValueError("Choose a candidate library from Galaxy History")
            label = self._powgen_dataset_label(
                state.history_archive_datasets,
                dataset_id,
                "Selected custom Galaxy library",
            )
            return {"dataset_id": dataset_id, "path": None, "label": label}
        if archive_source != "computer":
            raise ValueError("Choose where the reusable library is stored")
        path = Path(str(state.live_database_archive_path or ""))
        if not path.is_file() or path.suffix.lower() != ".zip":
            raise ValueError("Choose a RADAR-PD library ZIP archive from this computer")
        if not zipfile.is_zipfile(path):
            raise ValueError("The candidate library is not a valid ZIP archive")
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            roots = [
                name.removesuffix("catalog_deduplicated.csv")
                for name in names
                if Path(name).name == "catalog_deduplicated.csv"
            ]
            complete = any(
                root + "mp_experimental_stable.csv" in names
                and any(
                    name.startswith(root + "profiles64/") and not name.endswith("/")
                    for name in names
                )
                for root in roots
            )
            if not complete:
                raise ValueError(
                    "Select a built RADAR-PD portable library archive. To use a ZIP of source CIFs, choose 'Create a mini-library from my CIFs' and build it here."
                )
        return {"dataset_id": None, "path": str(path), "label": path.name}

    async def _save_live_configuration(self, config):
        state = self.server.state
        if (
            getattr(state, "live_configuration_source", None) or "history"
        ) == "history":
            return str(state.powgen_configuration_dataset_id)
        key = configuration_digest(config)
        cache = self._live_saved_configurations
        if key not in cache:
            action = await asyncio.to_thread(self.service.save_configuration, config)
            if action.status != RunStatus.OK or not action.outputs.get("config_output"):
                raise RuntimeError(
                    action.message or "Galaxy did not save the Live configuration"
                )
            cache[key] = str(action.outputs["config_output"])
            self._register_utility(action)
            self._remember_history_dataset(
                cache[key], config.run_name + "_radar_pd_config.yaml", "configuration"
            )
        return cache[key]

    async def _upload_live_asset(self, path, label):
        before = local_asset_signature(str(path))
        key = (label, *before)
        cache = self._live_uploaded_assets
        if key not in cache:
            dataset_id = await asyncio.to_thread(
                self.service.upload_document, Path(path), label=label
            )
            if before != local_asset_signature(str(path)):
                raise ValueError(
                    f"{Path(path).name} changed during upload; check the Live setup again"
                )
            if not dataset_id:
                raise RuntimeError(f"Galaxy did not acknowledge {label}")
            cache[key] = str(dataset_id)
        return cache[key]

    async def _prepare_live_settings(self, *, test=False):
        state = self.server.state
        if not await self.check_powgen_experiment():
            return None
        checked = self._powgen_preflight_signature()
        config = self._live_checked_configuration.model_copy(deep=True)
        library = deepcopy(self._live_library_selection())
        adapter = self._live_adapter()
        values = {
            key: deepcopy(getattr(state, key, None))
            for key in (
                "powgen_ipts",
                "powgen_instrument_source",
                "powgen_instrument_path",
                "powgen_instrument_dataset_id",
                "powgen_instrument_label",
                "powgen_main_cif_source",
                "powgen_main_cif_path",
                "powgen_main_cif_dataset_id",
                "powgen_backfill_mode",
                "powgen_wavelength",
                "powgen_excluded_runs",
                "powgen_include_background",
                "powgen_include_ramps",
                "powgen_source_subfolder",
                "nomad_profile_bank",
                "nomad_profile_acknowledged",
                "nomad_allow_exploratory",
                "live_test_bank",
            )
        }
        selected_samples = self._powgen_processing_ids()
        selected_banks = self._powgen_processing_bank_ids()
        if test:
            bank = values["live_test_bank"]
            allowed = [int(v["value"]) for v in (state.live_test_bank_options or [])]
            if bank is None or isinstance(bank, bool) or int(bank) not in allowed:
                raise ValueError("Choose a diffraction bank for the one-scan test")
            selected_banks = (int(bank),)
        if not self.service.history_id:
            raise RuntimeError(
                "The active Galaxy History is not available to this NOVA session"
            )
        state.powgen_message = "Inputs verified. Saving the reproducible Live configuration..."
        state.flush()
        config_id = await self._save_live_configuration(config)
        main_id = (
            str(values["powgen_main_cif_dataset_id"] or "") or None
            if values["powgen_main_cif_source"] == "galaxy"
            else None
        )
        if values["powgen_main_cif_source"] == "computer":
            state.powgen_message = "Saving the selected main-phase CIF to Galaxy..."
            state.flush()
            main_id = await self._upload_live_asset(
                values["powgen_main_cif_path"], f"{adapter.name} main phase CIF"
            )
        profile_id = (
            str(values["powgen_instrument_dataset_id"] or "") or None
            if values["powgen_instrument_source"] == "galaxy"
            else None
        )
        if values["powgen_instrument_source"] == "computer":
            state.powgen_message = "Saving the selected instrument profile to Galaxy..."
            state.flush()
            profile_id = await self._upload_live_asset(
                values["powgen_instrument_path"], f"{adapter.name} instrument profile"
            )
        library_id = library["dataset_id"]
        if library["path"]:
            state.powgen_message = "Saving the selected candidate library to Galaxy..."
            state.flush()
            library_id = await self._upload_live_asset(
                library["path"], "Live candidate library"
            )
        if checked != self._powgen_preflight_signature():
            raise ValueError(
                "Live setup changed while inputs were saved. Check the current settings before starting."
            )
        # All adapter inputs below were captured before the first upload. Later
        # Single-pattern edits cannot affect a queued or running Live analysis.
        backfill = {
            "latest_1": 1,
            "latest_5": 5,
            "latest_20": 20,
            "new_only": 0,
            "all": None,
        }[values["powgen_backfill_mode"]]
        extra = (
            {}
            if adapter.instrument == "PG3"
            else {
                "profile_bank": int(values["nomad_profile_bank"] or 0),
                "profile_acknowledged": bool(values["nomad_profile_acknowledged"]),
                "allow_exploratory": bool(values["nomad_allow_exploratory"]),
            }
        )
        settings = adapter.settings(
            ipts=self._normalize_powgen_ipts(values["powgen_ipts"]),
            history_id=self.service.history_id,
            configuration_dataset_id=config_id,
            wavelength_angstrom=str(values["powgen_wavelength"] or ""),
            instrument_profile_dataset_id=profile_id,
            instrument_profile_name=str(values["powgen_instrument_label"] or "")
            or None,
            main_cif_dataset_id=main_id,
            database_dataset_id=library_id,
            initial_backfill_limit=1 if test else backfill,
            selected_sample_ids=selected_samples,
            selected_banks=selected_banks,
            excluded_run_numbers=tuple(
                int(n) for n in (values["powgen_excluded_runs"] or [])
            ),
            include_background_runs=bool(values["powgen_include_background"]),
            include_ramp_runs=bool(values["powgen_include_ramps"]),
            subfolder=str(values["powgen_source_subfolder"] or adapter.default_folder),
            **extra,
        )
        state.live_saved_configuration_id = config_id
        return settings, config

    async def test_live_scan(self, **_):
        state = self.server.state
        if state.powgen_monitoring or state.live_launch_busy:
            return
        state.live_launch_busy = True
        state.live_test_message = "Checking and preparing one scan..."
        state.flush()
        try:
            prepared = await self._prepare_live_settings(test=True)
            if prepared is None:
                return
            settings, _config = prepared
            controller = self._live_adapter().controller(
                self.service,
                settings,
                sample_index=self._powgen_sample_index(settings.ipts),
            )
            checked = self._powgen_preflight_signature()
            # The disposable discovery never writes a watch checkpoint or
            # schedules the remaining banks. Exactly one analysis is launched.
            await asyncio.to_thread(controller.discover, checkpoint=False)
            runs = controller.due_submissions()
            if not runs:
                raise ValueError(
                    "No completed matching scan is ready for the one-scan test"
                )
            if checked != self._powgen_preflight_signature():
                raise ValueError(
                    "Live setup changed during scan discovery; check it again"
                )
            run = runs[-1]
            record = await asyncio.to_thread(controller.launch_submission, run)
            record.provenance["live_trial"] = {
                "run": run.run_number,
                "bank": run.bank,
                "configuration_dataset_id": settings.configuration_dataset_id,
            }
            self.records[record.uid] = record
            self._sync_runs()
            self._select_record(record)
            state.workspace_view = "monitor"
            state.active_page = "runs"
            state.setup_collapsed = True
            state.live_test_message = f"One-scan test submitted: {run.run_id}. Continuous monitoring is stopped."
            state.notice = state.live_test_message
            state.error_message = ""
            self._start_monitor(record)
        except Exception as exc:  # noqa: BLE001 - leave monitoring stopped after any launch failure
            state.live_test_message = f"One-scan test could not start: {exc}"
            state.error_message = state.live_test_message
        finally:
            state.live_launch_busy = False
            state.flush()
