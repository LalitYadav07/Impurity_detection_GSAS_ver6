"""Live setup at a local Galaxy boundary: no network or facility writes."""

import asyncio
import threading
import zipfile
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import ClassVar

import pytest
from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.configuration import config_from_contract, dump_configuration
from radar_pd_nova.live_beamlines import BEAMLINES
from radar_pd_nova.live_setup import configuration_digest
from radar_pd_nova.models import (
    AnalysisConfig,
    InputSelection,
    RunRecord,
    RunStatus,
    UtilityActionRecord,
)
from radar_pd_nova.post_analysis import PostAnalysisSettings
from radar_pd_nova.setup_forms import configuration_form_values, configuration_from_form
from radar_pd_nova.uploads import build_cif_source_archive, cif_element_suggestion


class LocalGalaxy:
    history_id = "local-history"
    galaxy_url = "https://local.invalid"

    def __init__(self):
        self.configs = {}
        self.saved = []
        self.uploads = []
        self.jobs = []
        self.build_inputs = []

    def save_configuration(self, config):
        key = f"config-{len(self.saved) + 1}"
        # Exercise the actual wire-contract round trip.
        self.configs[key] = config_from_contract(config.portable_contract())
        self.saved.append(deepcopy(config))
        return UtilityActionRecord(
            uid=key,
            tool_id="configuration",
            name="Save",
            status=RunStatus.OK,
            outputs={"config_output": key},
        )

    def load_configuration_dataset(self, key):
        return self.configs[key].model_copy(deep=True)

    def load_json_document(self, key):
        return self.configs[key].portable_contract()

    def upload_document(self, path, *, label):
        self.uploads.append((str(path), label))
        return f"upload-{len(self.uploads)}"

    def _upload_one(self, key, path, label):
        self.uploads.append((str(path), label))
        return key, "cif-bundle"

    def submit_utility(self, **kwargs):
        self.build_inputs.append(kwargs["inputs"])
        return UtilityActionRecord(
            uid=f"build-{len(self.build_inputs)}",
            tool_id=kwargs["tool_id"],
            name="Build",
            status=RunStatus.OK,
            outputs={
                "library_archive": "built-library",
                "library_manifest": "manifest",
            },
        )

    def _dataset_document(self, key):
        return {"phase_count": 2, "failures": []}


class TrialController:
    instances: ClassVar[list] = []

    def __init__(self, service, settings, **kwargs):
        self.service, self.settings = service, settings
        self.source_directory = "/SNS/local-read-only"
        self.state = SimpleNamespace(post_analysis={}, scan_interpretation={})
        self.records = {}
        self.instances.append(self)

    def restore_latest_state(self):
        return False

    def discover(self, *, checkpoint=True):
        self.checkpoint = checkpoint
        self.runs = [SimpleNamespace(run_id="scan-100-bank2", run_number=100, bank=2)]

    def due_submissions(self):
        return self.runs

    def launch_submission(self, run):
        record = RunRecord(
            uid="trial-1",
            name=run.run_id,
            mode="full",
            history_id=self.service.history_id,
            status=RunStatus.QUEUED,
            galaxy_job_id="job-1",
            config=self.service.load_configuration_dataset(
                self.settings.configuration_dataset_id
            ),
            inputs=InputSelection(
                source="galaxy",
                data_dataset_id="pattern",
                instrument_dataset_id="profile",
                bank="2",
            ),
        )
        self.service.jobs.append(record)
        return record


@pytest.fixture
def live(monkeypatch):
    app = RadarPdNovaApp()
    state = app.server.state
    app.service = LocalGalaxy()
    app.refresh_history = lambda **_: None
    app._sync_activities = lambda: None
    app._sync_powgen_rows = lambda: None
    app._sync_workspace_options = lambda *_: None
    app._sync_runs = lambda: None
    app._select_record = lambda record: setattr(state, "selected_run_uid", record.uid)
    app._start_monitor = lambda record: None
    app._powgen_sample_index = lambda _: None

    async def no_background_work():
        return

    app._powgen_monitor_loop = no_background_work
    state.powgen_ipts = "2767"
    state.live_sample_elements = "Mg, Al, O"
    state.powgen_processing_scope = "selected"
    state.powgen_processing_sample_ids = ["sample-A"]
    state.powgen_bank_scope = "selected"
    state.powgen_processing_banks = [2]
    state.flush()

    def preflight(*args, **kwargs):
        return {
            "ready": True,
            "source_directory": "/SNS/PG3/IPTS-2767/shared/autoreduce",
            "scan_count": 2,
            "available_banks": [2],
            "analysis_count": 2,
            "message": "Ready",
            "selected_wavelength": "1.5",
            "profile_filename": "official.instprm",
        }

    monkeypatch.setattr("radar_pd_nova.app.preflight_powgen_experiment", preflight)
    monkeypatch.setattr("radar_pd_nova.app.PowgenWatchController", TrialController)
    monkeypatch.setitem(
        BEAMLINES, "PG3", replace(BEAMLINES["PG3"], controller=TrialController)
    )
    TrialController.instances = []
    return app


def test_live_defaults_are_complete_separate_drafts(live):
    state = live.server.state
    state.sample_elements = "Fe"
    state.radiation = "xray"
    state.instrument_mode = "cw"
    state.database_source = "archive"
    state.library_archive_source = "galaxy"
    state.history_database_id = "single-only"
    state.flush()
    config = live._live_configuration()
    assert (
        config.sample_elements == ["Mg", "Al", "O"]
        and config.radiation.value == "neutron"
        and config.instrument_mode == "tof"
    )
    assert live._resolve_powgen_database_id() is None
    state.powgen_database_dataset_id = "leftover-custom"
    assert (
        live._resolve_powgen_database_id() is None
    )  # Explicit built-in means built-in.
    state.live_sample_elements = "Cu"
    assert state.sample_elements == "Fe"


def test_preflight_is_read_only_and_requires_no_history_configuration(live):
    assert asyncio.run(live.check_powgen_experiment())
    state = live.server.state
    assert not state.powgen_configuration_dataset_id
    assert not live.service.saved and not live.service.uploads and not live.service.jobs
    assert state.powgen_preflight_ready
    assert state.live_test_bank == 2
    assert any(
        row["value"].startswith("Live settings")
        for row in state.powgen_preflight_details
        if row["label"] == "RADAR-PD configuration"
    )


@pytest.mark.parametrize('scope,expected',[('selected','[2]'),('all','[1, 2, 3, 4, 5, 6]')])
def test_bank_summary_distinguishes_selected_from_available_histograms(live,monkeypatch,scope,expected):
    live.server.state.powgen_bank_scope=scope
    def preflight(*args,**kwargs):
        return dict(ready=True,source_directory='/SNS/PG3/IPTS-2767/shared/autoreduce',
            scan_count=2,analysis_count=2 if kwargs['selected_banks'] else 12,
            available_banks=[1,2,3,4,5,6],message='Ready',selected_wavelength='1.5',profile_filename='official.instprm')
    monkeypatch.setattr('radar_pd_nova.app.preflight_powgen_experiment',preflight)
    assert asyncio.run(live.check_powgen_experiment())
    row=next(r for r in live.server.state.powgen_preflight_details if r['label']=='Bank analyses')
    assert 'selected source banks '+expected+';' in row['value']


def test_start_saves_exact_draft_and_freezes_the_watch(live):
    state = live.server.state
    state.live_full_profile = "custom"
    state.live_full_max_passes = 8
    state.live_full_dedup_threshold = 0.999
    state.live_fit_start = 1000
    state.live_fit_end = 50000
    state.flush()

    async def run():
        await live.start_powgen_monitoring()
        await asyncio.sleep(0)
        settings = live._powgen_controller.settings
        assert settings.configuration_dataset_id == "config-1"
        assert settings.selected_banks == (2,) and settings.selected_sample_ids == (
            "sample-A",
        )
        saved = live.service.load_configuration_dataset("config-1")
        assert (
            saved.full_max_passes == 8
            and saved.full_dedup_threshold == 0.999
            and saved.limits == (1000, 50000)
        )
        state.full_max_passes = 1
        state.live_full_max_passes = 3
        assert saved.full_max_passes == 8
        assert (
            live.service.load_configuration_dataset(
                settings.configuration_dataset_id
            ).full_max_passes
            == 8
        )

    asyncio.run(run())
    assert state.powgen_monitoring and not state.live_launch_busy
    assert len(live.service.saved) == 1 and not live.service.jobs


def test_test_latest_submits_exactly_one_and_never_starts_monitor(live):
    asyncio.run(live.test_live_scan())
    state = live.server.state
    assert not state.powgen_monitoring and not state.live_launch_busy
    assert len(live.service.jobs) == 1 and state.selected_run_uid == "trial-1"
    controller = TrialController.instances[0]
    assert controller.checkpoint is False
    assert controller.settings.initial_backfill_limit == 1
    assert controller.settings.selected_banks == (2,)
    assert (
        live.service.jobs[0].provenance["live_trial"]["configuration_dataset_id"]
        == "config-1"
    )


def test_trial_then_monitor_reuses_saved_configuration(live):
    async def run():
        await live.test_live_scan()
        await live.start_powgen_monitoring()
        await asyncio.sleep(0)

    asyncio.run(run())
    assert len(live.service.saved) == 1 and len(live.service.jobs) == 1
    assert live._powgen_controller.settings.configuration_dataset_id == "config-1"


@pytest.mark.parametrize(
    "field,value",
    [
        ("live_sample_elements", ""),
        ("live_fit_start", 10),
        ("live_database_source", "custom_mini"),
    ],
)
def test_invalid_draft_never_saves_or_submits(live, field, value):
    setattr(live.server.state, field, value)
    asyncio.run(live.start_powgen_monitoring())
    assert not live.service.saved and not live.service.jobs
    assert (
        not live.server.state.powgen_monitoring
        and not live.server.state.live_launch_busy
    )
    assert live.server.state.error_message


def test_failed_configuration_save_never_starts_watch(live):
    def fail(_):
        raise RuntimeError("History unavailable")

    live.service.save_configuration = fail
    asyncio.run(live.start_powgen_monitoring())
    assert not live.service.jobs and live._powgen_controller is None
    assert "History unavailable" in live.server.state.error_message


def test_changed_live_draft_during_save_aborts_before_job_submission(live):
    started, release = threading.Event(), threading.Event()
    save = live.service.save_configuration

    def delayed(config):
        started.set()
        assert release.wait(5)
        return save(config)

    live.service.save_configuration = delayed

    async def run():
        task = asyncio.create_task(live.start_powgen_monitoring())
        assert await asyncio.to_thread(started.wait, 5)
        live.server.state.live_sample_elements = "Fe"
        live.server.state.flush()
        release.set()
        await task

    asyncio.run(run())
    assert (
        len(live.service.saved) == 1
        and not live.service.jobs
        and not TrialController.instances
    )
    assert "changed while inputs were saved" in live.server.state.error_message
    assert not live.server.state.powgen_monitoring


def test_two_start_clicks_cannot_create_two_monitors(live):
    async def run():
        await asyncio.gather(
            live.start_powgen_monitoring(), live.start_powgen_monitoring()
        )
        await asyncio.sleep(0)

    asyncio.run(run())
    assert len(live.service.saved) == 1 and len(TrialController.instances) == 1


def test_checkpoint_restoration_does_not_block_the_ui_event_loop(live, monkeypatch):
    started, release = threading.Event(), threading.Event()
    threads = []

    def restore(_):
        threads.append(threading.get_ident())
        started.set()
        assert release.wait(3), "UI could not release the restoration worker"
        return False

    monkeypatch.setattr(TrialController, "restore_latest_state", restore)

    async def run():
        event_loop_thread = threading.get_ident()
        task = asyncio.create_task(live.start_powgen_monitoring())
        try:
            assert await asyncio.to_thread(started.wait, 3)
            assert "previous watch" in live.server.state.powgen_message
            assert threads != [event_loop_thread]
            assert live.server.state.live_launch_busy
        finally:
            release.set()
        await task
        await asyncio.sleep(0)
        assert live.server.state.powgen_monitoring

    asyncio.run(run())


def test_completed_scan_is_visible_while_another_upload_is_still_waiting(live):
    second_started, release_second = threading.Event(), threading.Event()
    first_visible, checkpoint_started = threading.Event(), threading.Event()
    final_checkpoint = threading.Event()
    state = live.server.state
    runs = [SimpleNamespace(run_id=f"scan-{n}", n=n) for n in (1, 2)]

    class Controller:
        source_directory = "/read-only/experiment"
        refresh_errors = {}
        settings = SimpleNamespace(max_active_jobs=5)

        def __init__(self):
            self.state = SimpleNamespace(submitted={}, completed={}, discovered={r.run_id: r for r in runs}, failed={})

        def discover(self, *, checkpoint): assert not checkpoint
        def reconcile_recent_runs(self, *, checkpoint):
            assert not checkpoint
            return 0
        def due_submissions(self, *, reserved_run_ids=None):
            return [r for r in self.state.discovered.values() if r.run_id not in (reserved_run_ids or ())]

        def launch_submission(self, run):
            if run.n == 1:
                assert second_started.wait(4)
            else:
                second_started.set()
                assert release_second.wait(4)
            return RunRecord(uid=run.run_id, name=run.run_id, mode="full", history_id="local-history", galaxy_job_id=run.run_id)

        def acknowledge_submission(self, run, record, *, persist):
            assert not persist
            self.state.submitted[run.run_id] = run
            del self.state.discovered[run.run_id]
            return record

        def refresh(self, *, checkpoint):
            assert not checkpoint
            records = {}
            for identifier, run in list(self.state.submitted.items()):
                self.state.completed[identifier] = self.state.submitted.pop(identifier)
                records[identifier] = RunRecord(uid=identifier, name=identifier, mode="full", history_id="local-history", status=RunStatus.OK)
            return records

        def capture_checkpoint(self): return len(self.state.completed)

        def persist_state(self, snapshot):
            checkpoint_started.set()
            if snapshot == 2: final_checkpoint.set()

    controller = Controller()
    live._powgen_controller = controller

    def sync():
        if "scan-1" in controller.state.completed and live.records.get("scan-1", None):
            first_visible.set()

    live._sync_powgen_rows = sync

    async def run():
        state.powgen_monitoring = True
        task = asyncio.create_task(RadarPdNovaApp._powgen_monitor_loop(live))
        try:
            assert await asyncio.to_thread(first_visible.wait, 4)
            assert second_started.is_set() and not release_second.is_set()
            assert checkpoint_started.is_set()
            assert live.records["scan-1"].status == RunStatus.OK
        finally:
            release_second.set()
        try:
            assert await asyncio.to_thread(final_checkpoint.wait, 4)
            assert set(controller.state.completed) == {"scan-1", "scan-2"}
        finally:
            state.powgen_monitoring = False
            live._powgen_wake_event.set()
            await asyncio.wait_for(task, 3)

    asyncio.run(run())


def test_existing_configuration_can_be_reused_or_copied_without_changing_single(live):
    state = live.server.state
    original = AnalysisConfig(
        run_name="shared",
        mode="full",
        sample_elements=["Ni", "O"],
        background_terms=8,
        full_profile="custom",
        full_max_passes=8,
    )
    live.service.configs["existing"] = original
    state.live_configuration_source = "history"
    state.powgen_configuration_dataset_id = "existing"
    state.sample_elements = "Si"
    state.background_terms = 4
    state.flush()
    assert asyncio.run(live.check_powgen_experiment())
    live.edit_live_configuration()
    assert (
        state.live_configuration_source == "inline"
        and state.live_sample_elements == "Ni, O"
    )
    assert state.live_background_terms == 8 and state.live_full_max_passes == 8
    assert state.sample_elements == "Si" and state.background_terms == 4
    state.live_sample_elements = "Ni, O, Cu"
    assert live.service.configs["existing"].sample_elements == ["Ni", "O"]


def test_configuration_import_is_atomic_and_rejects_xray(live, tmp_path):
    state = live.server.state
    path = dump_configuration(
        AnalysisConfig(sample_elements=["Fe"], instrument_mode="tof"),
        tmp_path / "saved.yaml",
    )
    state.live_config_import_path = str(path)
    live.apply_live_uploaded_configuration()
    assert state.live_sample_elements == "Fe"
    path = dump_configuration(
        AnalysisConfig(sample_elements=["Cu"], radiation="xray", instrument_mode="cw"),
        tmp_path / "xray.yaml",
    )
    state.live_config_import_path = str(path)
    before = state.live_sample_elements
    live.apply_live_uploaded_configuration()
    assert state.live_sample_elements == before and "neutron" in state.error_message


def test_live_derived_recipe_choices_are_independent(live):
    state = live.server.state
    old = deepcopy(state.post_analysis)
    live.update_live_post_analysis_selection(["thermal_expansion"])
    assert state.post_analysis == old
    assert [v["kind"] for v in state.live_post_analysis["views"]] == [
        "thermal_expansion"
    ]
    state.post_analysis = PostAnalysisSettings(views=[]).model_dump(mode="json")
    assert [v.kind for v in live._live_configuration().post_analysis.views] == [
        "thermal_expansion"
    ]


def test_library_zip_upload_is_durable_and_cannot_inherit_single_archive(
    live, tmp_path
):
    path = tmp_path / "library.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("pack/catalog_deduplicated.csv", "id\n")
        archive.writestr("pack/mp_experimental_stable.csv", "id\n")
        archive.writestr("pack/profiles64/phase.npy", "profile")
    state = live.server.state
    state.live_database_source = "archive"
    state.live_library_archive_source = "computer"
    state.live_database_archive_path = str(path)
    state.database_archive_path = "unrelated-single.zip"
    state.flush()

    async def run():
        await live.start_powgen_monitoring()
        await asyncio.sleep(0)

    asyncio.run(run())
    assert live._powgen_controller.settings.database_dataset_id == "upload-1"
    assert live.service.uploads == [(str(path), "Live candidate library")]
    assert state.database_archive_path == "unrelated-single.zip"


CIF = """data_test
_chemical_formula_sum 'Al2 O3'
_cell_length_a 4.7
_cell_length_b 4.7
_cell_length_c 13
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 120
"""


@pytest.mark.parametrize("context", ["single", "live"])
def test_library_build_selects_only_its_own_workflow(live, tmp_path, context):
    state = live.server.state
    cif = tmp_path / "Al2O3.cif"
    cif.write_text(CIF)
    prefix = "live_" if context == "live" else ""
    setattr(state, prefix + "library_builder_local_paths", [str(cif)])
    setattr(state, prefix + "database_source", "custom_mini")
    state.live_sample_elements = ""
    pending = []
    live._schedule_utility = lambda coroutine, name: pending.append(coroutine)
    live.build_candidate_library(context)
    asyncio.run(pending[0])
    if context == "live":
        assert (
            state.powgen_database_dataset_id == "built-library"
            and state.live_database_source == "archive"
        )
        assert state.database_source == "builtin" and not state.history_database_id
        assert state.live_sample_elements == "Al, O"
    else:
        assert (
            state.history_database_id == "built-library"
            and state.database_source == "archive"
        )
        assert (
            not state.powgen_database_dataset_id
            and state.live_database_source == "builtin"
        )
        assert state.live_sample_elements == ""
    assert live.service.build_inputs[0]["radiation"] == "neutron"
    assert getattr(state, prefix + "library_builder_status") == "ready"


def test_library_build_does_not_replace_a_changed_selection(live, tmp_path):
    state = live.server.state
    cif = tmp_path / "Al.cif"
    cif.write_text(CIF)
    state.live_database_source = "custom_mini"
    state.live_library_builder_local_paths = [str(cif)]
    pending = []
    live._schedule_utility = lambda coroutine, name: pending.append(coroutine)
    live.build_candidate_library("live")
    state.live_database_source = "builtin"
    asyncio.run(pending[0])
    assert (
        state.live_database_source == "builtin" and not state.powgen_database_dataset_id
    )
    assert "changed selection was kept" in state.live_library_builder_message


def test_suggestion_never_overwrites_entered_elements(live):
    state = live.server.state
    state.live_library_builder_elements = ["Al", "O"]
    live._use_live_element_suggestion(only_if_empty=True)
    assert state.live_sample_elements == "Mg, Al, O"
    live.use_live_element_suggestion()
    assert state.live_sample_elements == "Al, O"
    state.live_sample_elements = ""
    state.live_library_builder_unknown_elements = 1
    live._use_live_element_suggestion(only_if_empty=True)
    assert state.live_sample_elements == ""


@pytest.mark.parametrize(
    "config",
    [
        AnalysisConfig(sample_elements=["Al", "O"]),
        AnalysisConfig(
            sample_elements=["Li", "C"],
            mode="full",
            full_profile="custom",
            full_max_passes=8,
            full_dedup_threshold=0.999,
            limits=(1000, 50000),
            exclude_regions=[(3000, 4000)],
            reference_masks_enabled=True,
            reference_window_mode="fixed",
            reference_fixed_half_width=0.2,
        ),
    ],
)
def test_shared_form_conversion_preserves_scientific_contract(config):
    values = configuration_form_values(config)
    values.update(
        main_cif_source="none",
        main_cif_path="",
        history_main_cif_id="",
        remote_main_cif_uri="",
        facility_main_cif_path="",
    )
    restored = configuration_from_form(SimpleNamespace(**values))
    # Main-phase controls are intentionally disabled without an actual CIF.
    assert restored.portable_contract()["full"] == config.portable_contract()["full"]
    assert restored.sample_elements == config.sample_elements
    assert (
        restored.limits == config.limits
        and restored.exclude_regions == config.exclude_regions
    )
    assert restored.reference_fixed_half_width == config.reference_fixed_half_width


def test_digest_is_stable_and_changes_for_scientific_settings():
    config = AnalysisConfig(sample_elements=["Si"])
    assert configuration_digest(config) == configuration_digest(
        config.model_copy(deep=True)
    )
    assert configuration_digest(config) != configuration_digest(
        config.model_copy(update={"background_terms": 8})
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "_chemical_formula_sum 'Li0.5 Ni0.6 Mn0.2 Co0.2 O2'",
            ["Co", "Li", "Mn", "Ni", "O"],
        ),
        (
            "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_occupancy\nFe1 Fe3+ 1\nO1 O2- 1\n",
            ["Fe", "O"],
        ),
        ("loop_\n_atom_type_symbol\nFe3+\nO2-\n", ["Fe", "O"]),
        ("_chemical_formula_sum 'NMC622'", []),
        ("_chemical_formula_sum '?'", []),
        ("loop_\n_atom_site_label\n_atom_site_type_symbol\nX1 ?\n", []),
    ],
)
def test_cif_element_suggestion_uses_explicit_chemistry_only(text, expected):
    assert cif_element_suggestion(text) == expected


def test_zip_bundle_suggests_union_with_unknown_count(tmp_path):
    path = tmp_path / "sources.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("nested/known.cif", CIF)
        archive.writestr(
            "unknown.cif", CIF.replace("_chemical_formula_sum 'Al2 O3'\n", "")
        )
    _, stats = build_cif_source_archive([str(path)], "test")
    assert (
        stats["elements"] == ["Al", "O"]
        and stats["unknown_elements"] == 1
        and stats["cif_count"] == 2
    )


def test_live_ui_has_own_stable_uploads_and_no_saved_configuration_gate(live):
    template = live.layout.html
    assert (
        "under Single pattern first" not in template
        and "!powgen_configuration_dataset_id || powgen_sample_loading" not in template
    )
    for title in (
        "Candidate structures",
        "Analysis settings",
        "Review and start",
        "Test latest scan",
        "Edit a copy here",
    ):
        assert title in template
    for key in (
        "live-cif-library-upload-native",
        "live-database-upload-native",
        "radar-cif-library-upload-native",
        "radar-database-upload-native",
    ):
        assert template.count(key) == 1
    assert "html-error" not in template


def test_reusing_saved_configuration_does_not_create_another_configuration(live):
    state = live.server.state
    live.service.configs["shared"] = AnalysisConfig(
        sample_elements=["Fe"], instrument_mode="tof"
    )
    state.live_configuration_source = "history"
    state.powgen_configuration_dataset_id = "shared"
    state.live_sample_elements = ""

    async def run():
        await live.start_powgen_monitoring()
        await asyncio.sleep(0)

    asyncio.run(run())
    assert live._powgen_controller.settings.configuration_dataset_id == "shared"
    assert not live.service.saved


def test_source_cif_zip_is_not_mistaken_for_a_built_library(live, tmp_path):
    path = tmp_path / "cifs.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("Al2O3.cif", CIF)
    state = live.server.state
    state.live_database_source = "archive"
    state.live_library_archive_source = "computer"
    state.live_database_archive_path = str(path)
    asyncio.run(live.start_powgen_monitoring())
    assert not live.service.saved and not live.service.uploads and not live.service.jobs
    assert "source CIFs" in state.error_message


def test_unknown_phase_count_never_selects_a_library(live, tmp_path):
    state = live.server.state
    path = tmp_path / "Al.cif"
    path.write_text(CIF)
    state.live_database_source = "custom_mini"
    state.live_library_builder_local_paths = [str(path)]
    live.service._dataset_document = lambda _: {
        "phase_count": 0,
        "failures": [{"id": "Al", "error": "No atoms"}],
    }
    pending = []
    live._schedule_utility = lambda coroutine, name: pending.append(coroutine)
    live.build_candidate_library("live")
    asyncio.run(pending[0])
    assert (
        not state.powgen_database_dataset_id
        and state.live_database_source == "custom_mini"
    )
    assert state.live_library_builder_status == "error"


def test_nomad_inline_setup_keeps_physical_and_profile_banks_distinct(
    live, tmp_path, monkeypatch
):
    state = live.server.state
    state.live_beamline = "NOM"
    live._live_beamline_changed()
    path = tmp_path / "nomad.instprm"
    path.write_text("# supplied profile\n")
    state.powgen_ipts = "21886"
    state.live_sample_elements = "Li, C, O, Ni, Mn, Co, Cu, Al"
    state.powgen_processing_scope = "all"
    state.nomad_physical_bank = 5
    state.nomad_profile_bank = 1
    state.nomad_profile_acknowledged = True
    state.powgen_instrument_path = str(path)
    captured = []

    def preflight(*args, **kwargs):
        captured.append(kwargs)
        return {
            "ready": True,
            "source_directory": "/SNS/NOM/IPTS-21886/shared/autoreduce/multi_banks/GSAS",
            "available_banks": [5],
            "scan_count": 1,
            "message": "Supplied profile reviewed",
        }

    monkeypatch.setitem(
        BEAMLINES,
        "NOM",
        replace(BEAMLINES["NOM"], preflight=preflight, controller=TrialController),
    )
    state.flush()

    async def run():
        await live.start_powgen_monitoring()
        await asyncio.sleep(0)

    asyncio.run(run())
    assert live._powgen_controller is not None, state.error_message
    settings = live._powgen_controller.settings
    assert (
        settings.instrument == "NOM"
        and settings.selected_banks == (5,)
        and settings.profile_bank == 1
    )
    assert (
        settings.profile_acknowledged
        and settings.instrument_profile_dataset_id == "upload-1"
    )
    assert captured[0]["selected_banks"] == (5,) and captured[0]["profile_bank"] == 1
    assert (
        live.service.saved[0].instrument_mode == "tof"
        and live.service.saved[0].radiation.value == "neutron"
    )
