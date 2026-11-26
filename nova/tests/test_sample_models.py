"""Mixed samples must not borrow sister-compound models or main phases."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import jsonschema
import pytest

from radar_pd_nova.configuration import config_from_contract, dump_configuration, load_configuration
from radar_pd_nova.live_setup import LiveSetupMixin, configuration_digest
from radar_pd_nova.models import AnalysisConfig
from radar_pd_nova.sample_models import bind_sample_models, check_model_selection, single_sample_cohort
from radar_pd_nova.setup_forms import configuration_form_values, configuration_from_form


def routing(instrument="NOM"):
    return dict(galaxy_url="https://galaxy.invalid", history_id="history",
        assignments=[dict(instrument=instrument, ipts="IPTS-123", sample_id="A",
                         candidate_library_dataset_id="library-A", known_main_cif_dataset_id="main-A"),
                     dict(instrument=instrument, ipts="IPTS-123", sample_id="B",
                         candidate_library_dataset_id="library-B", known_main_cif_dataset_id=None)])


def config(instrument="NOM"):
    return AnalysisConfig(sample_elements=["Ni", "Mn", "Co", "O"], sample_models=routing(instrument))


def bind(cfg, sample="A", **changes):
    values=dict(instrument="NOM", ipts="IPTS-123", metadata=dict(sample_id=sample, identity_status="resolved"),
        selected_sample_ids=("A", "B"), galaxy_url="https://galaxy.invalid", history_id="history",
        shared_library="wrong-global-library", shared_main="wrong-global-main")
    values.update(changes)
    return bind_sample_models(cfg, **values)


def test_sister_sample_has_its_own_library_and_keeps_blind_discovery():
    a, b = bind(config()), bind(config(), "B")
    assert (a.candidate_library_dataset_id, a.known_main_cif_dataset_id) == ("library-A", "main-A")
    assert (b.candidate_library_dataset_id, b.known_main_cif_dataset_id) == ("library-B", None)
    assert a.receipt()["binding_sha256"] != b.receipt()["binding_sha256"]
    assert b.receipt()["sample_id"] == "B"


@pytest.mark.parametrize("changes,match", [
    ({"metadata":dict(sample_id="A", identity_status="resolved", identity_verification="catalog_preview")}, "source-file"),
    ({"metadata":dict(sample_id="", identity_status="conflict", sample_name="A", formula="NMC333", rwp=0.1)}, "source-file"),
    ({"metadata":dict(sample_id="C", identity_status="resolved"), "selected_sample_ids":None}, "No model assignment"),
    ({"metadata":dict(sample_id="B", identity_status="resolved"), "selected_sample_ids":("A",)}, "selected sample scope"),
    ({"galaxy_url":"https://another.invalid"}, "another Galaxy"),
    ({"history_id":"another"}, "another Galaxy"),
    ({"ipts":"IPTS-124"}, "No model assignment"),
    ({"instrument":"SNAP"}, "No model assignment"),
])
def test_unproven_or_foreign_scope_cannot_select_models(changes, match):
    with pytest.raises(ValueError, match=match):
        bind(config(), **changes)


def test_legacy_config_requires_one_explicit_verified_sample():
    cfg=AnalysisConfig(sample_elements=["Ni", "O"])
    selected=bind(cfg, selected_sample_ids=("A",))
    assert selected.known_main_cif_dataset_id == "wrong-global-main"
    for ids in (None, ("A", "B")):
        with pytest.raises(ValueError, match="Select one sample|verified sample ID"):
            bind(cfg, selected_sample_ids=ids)


def test_default_all_samples_accepts_one_verified_cohort_but_not_names_as_identity():
    cfg=AnalysisConfig(sample_elements=["O"])
    kwargs=dict(instrument="NOM",ipts="IPTS-21886",selected_sample_ids=None,
        galaxy_url="https://galaxy.invalid",history_id="history")
    rows=[dict(status="resolved", sample_id="66546", sample_name="ZDND002") for _ in range(2)]
    assert "66546" in check_model_selection(cfg, **kwargs, cohort_metadata=rows)
    rows.append(dict(status="resolved",sample_id="66547",sample_name="ZDND002"))
    with pytest.raises(ValueError,match="multiple sample IDs"):
        check_model_selection(cfg, **kwargs, cohort_metadata=rows)


def test_related_real_sample_ids_remain_distinct_even_with_shared_metadata():
    # Only public identifier/name metadata is used. These are identity tests,
    # not assertions about actual sample composition or fit correctness.
    for ids in [("66546", "66548"), ("66548", "66549"),
                ("120472","120473","120474","120475","120476","120477","120845")]:
        rows=[dict(status="resolved",sample_id=i,sample_name="same label",registered_formula="same formula") for i in ids]
        with pytest.raises(ValueError,match="multiple sample IDs"):
            single_sample_cohort(rows)


def test_unknown_live_source_explains_exact_single_pattern_fallback():
    unknown=dict(sample_id="",identity_status="unidentified",sample_name="")
    for action in (lambda:single_sample_cohort([unknown]),
                   lambda:bind(config(),metadata=unknown)):
        with pytest.raises(ValueError) as error:
            action()
        message=str(error.value)
        assert "Single pattern" in message and "exact reduced file" in message
        assert "matching instrument profile" in message
        assert "Future Live enrollment remains blocked" in message


def test_full_contract_yaml_schema_and_form_roundtrip(tmp_path):
    cfg=config()
    wire=cfg.portable_contract()
    schema=json.loads((Path(__file__).resolve().parents[2]/"schemas/ndip/radar-pd-config-v1.schema.json").read_text())
    jsonschema.validate(wire, schema)
    path=dump_configuration(cfg, tmp_path/"models.yaml")
    assert load_configuration(path).sample_models == cfg.sample_models
    assert config_from_contract(wire).sample_models == cfg.sample_models
    state=SimpleNamespace(**configuration_form_values(cfg), main_cif_source="none", main_cif_path="",
        history_main_cif_id="", remote_main_cif_uri="", facility_main_cif_path="")
    assert configuration_from_form(state).sample_models == cfg.sample_models
    other=config().model_copy(update={"sample_models":None})
    assert configuration_digest(cfg) != configuration_digest(other)


def test_duplicate_or_implicit_assignments_rejected():
    value=routing()
    value["assignments"].append(deepcopy(value["assignments"][0]))
    with pytest.raises(ValueError, match="only one model"):
        AnalysisConfig(sample_elements=["O"], sample_models=value)


def test_per_scan_profile_cannot_leak_from_known_main_into_blind_assignment():
    cfg=config().model_copy(update={"tof_profile_mode":"per_scan"})
    with pytest.raises(ValueError,match="supplied main CIF for every assigned sample"):
        bind(cfg, "B")
    assert bind(cfg,"A",selected_sample_ids=("A",)).known_main_cif_dataset_id=="main-A"
    value=routing()
    del value["assignments"][0]["known_main_cif_dataset_id"]
    with pytest.raises(ValueError):
        AnalysisConfig(sample_elements=["O"], sample_models=value)


@pytest.mark.parametrize("instrument,module_name,class_name", [
    ("PG3", "powgen_controller", "PowgenWatchController"),
    ("NOM", "nomad_controller", "NomadWatchController"),
    ("HB2A", "hb2a_controller", "Hb2aWatchController"),
    ("SNAP", "snap_controller", "SnapWatchController"),
])
def test_every_controller_rejects_unassigned_sample_before_staging(monkeypatch, instrument, module_name, class_name):
    import importlib
    module=importlib.import_module("radar_pd_nova."+module_name)
    controller=getattr(module,class_name).__new__(getattr(module,class_name))
    controller._configuration=config(instrument)
    controller._configuration_lock=threading.Lock()
    controller.settings=SimpleNamespace(instrument=instrument, ipts="IPTS-123", selected_sample_ids=None,
        database_dataset_id="wrong-global-library", main_cif_dataset_id="wrong-global-main")
    controller.service=SimpleNamespace(galaxy_url="https://galaxy.invalid", history_id="history")
    controller.sample_index=SimpleNamespace(identity=lambda _:dict(sample_id="C",identity_status="resolved"))
    controller.definition=object()
    controller.state=object()
    monkeypatch.setattr(module,"build_submission_plan",lambda *_:SimpleNamespace(idempotency_key="plan"))
    controller._prepare_run=lambda _:pytest.fail("Unassigned sample reached staging")
    controller._setup=lambda _:pytest.fail("Unassigned sample reached staging")
    with pytest.raises(ValueError, match="No model assignment"):
        controller.launch_submission(SimpleNamespace(scan_metadata={"sample_id":"C"},run_id="run"))


def test_changed_source_identity_rejected_even_if_both_samples_are_assigned():
    from radar_pd_nova.powgen_controller import PowgenWatchController
    controller=PowgenWatchController.__new__(PowgenWatchController)
    controller._configuration=config()
    controller.sample_index=SimpleNamespace(identity=lambda _:dict(sample_id="B",identity_status="resolved"))
    with pytest.raises(ValueError, match="changed after discovery"):
        controller._sample_model_binding(SimpleNamespace(scan_metadata={"sample_id":"A"},run_id="run"))


def test_legacy_single_sample_watch_pins_and_restores_scope_then_rejects_new_sample():
    from radar_pd_nova.powgen_controller import PowgenWatchController
    from radar_pd_nova.powgen_watch import WatchState, WatchedRun
    controller=PowgenWatchController.__new__(PowgenWatchController)
    controller._configuration=AnalysisConfig(sample_elements=["O"])
    controller._configuration_lock=threading.Lock()
    controller.settings=SimpleNamespace(instrument="NOM",ipts="IPTS-21886",selected_sample_ids=None,
        database_dataset_id="library",main_cif_dataset_id="main")
    controller.service=SimpleNamespace(galaxy_url="https://galaxy.invalid",history_id="history")
    first=WatchedRun(138719,"/input/one.gsa","one",instrument="NOM")
    controller.state=WatchState("history",instrument="NOM",discovered={first.run_id:first})
    ids={138719:"66546",138720:"66548"}
    controller.sample_index=SimpleNamespace(identity=lambda r:dict(sample_id=ids[r.run_number],identity_status="resolved"))
    assert controller._sample_model_binding(first).sample_id == "66546"
    controller.state=WatchState.from_json(controller.state.to_json())
    assert controller._sample_model_binding(first).sample_id == "66546"
    later=WatchedRun(138720,"/input/two.gsa","two",instrument="NOM")
    with pytest.raises(ValueError,match="selected sample scope"):
        controller._sample_model_binding(later)
    controller.state.sample_model_scope["candidate_library_dataset_id"]="different-library"
    with pytest.raises(ValueError,match="model binding changed"):
        controller._sample_model_binding(first)


def test_hb2a_equal_scan_number_remains_experiment_scoped():
    from radar_pd_nova.powgen_watch import WatchedRun
    a=WatchedRun(21,"/input/exp853_scan21.dat","a",instrument="HB2A",experiment_number=853)
    b=WatchedRun(21,"/input/exp854_scan21.dat","b",instrument="HB2A",experiment_number=854)
    assert a.run_id != b.run_id
    value=routing("HB2A")
    value["assignments"][0]["sample_id"]="exp853:Er"
    cfg=AnalysisConfig(sample_elements=["Er","O"],sample_models=value)
    with pytest.raises(ValueError,match="No model assignment"):
        bind(cfg,instrument="HB2A",sample="exp854:Er",selected_sample_ids=None)


@pytest.mark.parametrize("instrument,module_name,class_name", [
    ("PG3", "powgen_controller", "PowgenWatchController"),
    ("NOM", "nomad_controller", "NomadWatchController"),
    ("HB2A", "hb2a_controller", "Hb2aWatchController"),
    ("SNAP", "snap_controller", "SnapWatchController"),
])
def test_submitted_input_uses_assignment_and_no_main_leaks_to_blind_sample(
        monkeypatch,tmp_path,instrument,module_name,class_name):
    import importlib
    module=importlib.import_module("radar_pd_nova."+module_name)
    controller=getattr(module,class_name).__new__(getattr(module,class_name))
    controller._configuration=config(instrument)
    controller._configuration_lock=threading.Lock()
    controller.settings=SimpleNamespace(instrument=instrument,ipts="IPTS-123",selected_sample_ids=("A","B"),
        database_dataset_id="wrong-global-library",main_cif_dataset_id="wrong-global-main",
        instrument_profile_dataset_id="profile",configuration_dataset_id="config",subfolder="reduced",condition_bindings_json="{}")
    controller.sample_index=SimpleNamespace(identity=lambda r:dict(sample_id=r.scan_metadata["sample_id"],identity_status="resolved"))
    snapshots=[]
    controller.service=SimpleNamespace(galaxy_url="https://galaxy.invalid",history_id="history",
        create_submission_snapshot=lambda cfg,inputs,**kw:snapshots.append(SimpleNamespace(config=cfg,inputs=inputs,**kw)) or snapshots[-1])
    controller.definition=object();controller.state=object()
    controller._run_name=lambda r:r.run_id
    controller._submission_identity=lambda r:r.run_id
    controller._submit_live_snapshot=lambda r,s:SimpleNamespace(galaxy_job_id="job",provenance={})
    controller._profile_dataset_id=lambda _:"profile"
    source=tmp_path/"pattern.gsa";source.write_text("1 2 3\n")
    setup=SimpleNamespace(banks=[SimpleNamespace(bank=1)],cycle="2026",mode="HR",wavelength=1.5,frequency=60,reduction_script_sha256="x",source_signature="source")
    controller._setup=lambda _:setup
    controller._check_discovered_setup=lambda *args:None
    controller._prepare_run=lambda _: (source,source,{"source_bank":1,"staged_bank":1})
    controller._manual_setup_lock=threading.Lock();controller._manual_setup_signature=None
    controller._staging=SimpleNamespace(name=str(tmp_path))
    monkeypatch.setattr(module,"build_submission_plan",lambda *_:SimpleNamespace(idempotency_key="plan"))
    if instrument=="NOM":
        monkeypatch.setattr(module,"_profile_fit_domain",lambda *args:{})
        monkeypatch.setattr(module.nomad,"read_profile_snapshot",lambda _:object())
    for sample in ("A","B"):
        run=SimpleNamespace(run_id=sample,run_number=1,source_path=str(source),fingerprint=sample,
            scan_metadata={"sample_id":sample},bank=1)
        record=controller.launch_submission(run)
        assert record.provenance["sample_model_assignment"]["sample_id"]==sample
    assert snapshots[0].inputs.main_cif_dataset_id=="main-A"
    assert snapshots[1].inputs.main_cif_dataset_id is None
    assert snapshots[1].inputs.database_dataset_id=="library-B"
    assert snapshots[1].inputs.sample_id=="B"


def test_recovery_rejects_wrong_sister_library_even_with_same_name():
    from radar_pd_nova.powgen_controller import PowgenWatchController
    controller=PowgenWatchController.__new__(PowgenWatchController)
    controller._configuration=config()
    controller._run_name=lambda _:"same-run-name"
    controller._profile_matches_watch=lambda *args:True
    controller.settings=SimpleNamespace(instrument="NOM",ipts="IPTS-123",selected_sample_ids=("A","B"),
        database_dataset_id="wrong-global-library",main_cif_dataset_id="wrong-global-main")
    controller.service=SimpleNamespace(galaxy_url="https://galaxy.invalid",history_id="history")
    controller.sample_index=SimpleNamespace(identity=lambda _:dict(sample_id="B",identity_status="resolved"))
    record=SimpleNamespace(name="same-run-name",created_utc="",inputs=SimpleNamespace(
        database_dataset_id="library-A",main_cif_dataset_id="main-A",instrument_dataset_id="profile",sample_id="B"),provenance={})
    run=SimpleNamespace(discovered_utc="",scan_metadata={"sample_id":"B"},run_id="run")
    assert not controller._record_matches_watch(record,run)
    record.inputs.database_dataset_id="library-B";record.inputs.main_cif_dataset_id=None
    assert controller._record_matches_watch(record,run)
    record.provenance["sample_model_assignment"]={"sample_id":"A"}
    assert not controller._record_matches_watch(record,run)


def test_name_only_hb2a_requires_disjoint_observed_acquisition_scopes():
    value=routing("HB2A")
    a=value["assignments"][0]
    a.update(sample_id="EXP853:NAME:abc",acquisition_ids=["HB2A_EXP853_SCAN0021"])
    b=value["assignments"][1]
    b.update(sample_id="EXP853:NAME:abc",acquisition_ids=["HB2A_EXP853_SCAN0022"])
    cfg=AnalysisConfig(sample_elements=["Er","Tb","O"],sample_models=value)
    metadata=dict(sample_id="EXP853:NAME:abc",identity_status="resolved")
    a_binding=bind(cfg,instrument="HB2A",metadata=metadata,selected_sample_ids=None,acquisition_id="HB2A_EXP853_SCAN0021")
    b_binding=bind(cfg,instrument="HB2A",metadata=metadata,selected_sample_ids=None,acquisition_id="HB2A_EXP853_SCAN0022")
    assert a_binding.known_main_cif_dataset_id=="main-A"
    assert b_binding.known_main_cif_dataset_id is None
    with pytest.raises(ValueError,match="No model assignment"):
        bind(cfg,instrument="HB2A",metadata=metadata,selected_sample_ids=None,acquisition_id="HB2A_EXP853_SCAN0023")
    b["acquisition_ids"]=["HB2A_EXP853_SCAN0021"]
    with pytest.raises(ValueError,match="must not overlap"):
        AnalysisConfig(sample_elements=["O"],sample_models=value)
    b["sample_id"]="EXP853:NAME:different"
    with pytest.raises(ValueError,match="must not overlap"):
        AnalysisConfig(sample_elements=["O"],sample_models=value)
    with pytest.raises(ValueError,match="Name-only"):
        single_sample_cohort([metadata])
    value["assignments"]=[{k:v for k,v in a.items() if k!="acquisition_ids"}]
    cfg=AnalysisConfig(sample_elements=["O"],sample_models=value)
    with pytest.raises(ValueError,match="name-only"):
        bind(cfg,instrument="HB2A",metadata=metadata,selected_sample_ids=None,acquisition_id="HB2A_EXP853_SCAN0021")


class State(SimpleNamespace):
    def flush(self): pass


def editor():
    value=LiveSetupMixin()
    value.server=SimpleNamespace(state=State(live_sample_models=routing(),live_model_main_id="main-A",
        live_model_library_id="library-A",powgen_ipts="IPTS-123"))
    value._live_adapter=lambda:SimpleNamespace(instrument="NOM")
    value._normalize_powgen_ipts=lambda text:text
    return value


def test_editor_sample_switch_restores_only_its_assignment():
    value=editor()
    value.select_live_model_sample("B")
    assert value.server.state.live_model_main_id is None
    assert value.server.state.live_model_library_id == "library-B"
    value.select_live_model_sample("C")
    assert value.server.state.live_model_main_id is None
    assert value.server.state.live_model_library_id is None


def test_processing_sample_switch_clears_shared_main():
    value=editor()
    state=value.server.state
    state.live_beamline="NOM";state.powgen_processing_scope="selected"
    state.powgen_processing_sample_ids=["A"];state.powgen_monitoring=False
    value._clear_shared_main_on_sample_change()
    state.powgen_main_cif_source="galaxy";state.powgen_main_cif_dataset_id="main-A"
    state.powgen_main_cif_path="old.cif";state.powgen_processing_sample_ids=["B"]
    value._clear_shared_main_on_sample_change()
    assert state.powgen_main_cif_source == "none"
    assert not state.powgen_main_cif_dataset_id and not state.powgen_main_cif_path


def test_first_unprimed_sample_change_cannot_keep_a_main_from_unknown_scope():
    value=editor()
    state=value.server.state
    state.live_beamline="NOM";state.powgen_processing_scope="selected"
    state.powgen_processing_sample_ids=["B"];state.powgen_monitoring=False
    state.powgen_main_cif_source="galaxy";state.powgen_main_cif_dataset_id="main-A"
    value._clear_shared_main_on_sample_change()
    assert state.powgen_main_cif_source=="none"
    assert not state.powgen_main_cif_dataset_id
