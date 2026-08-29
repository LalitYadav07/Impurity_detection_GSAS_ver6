from pathlib import Path

import pytest
import yaml

from radar_pd_nova.configuration import delivery_from_contract, dump_configuration, load_configuration
from radar_pd_nova.models import AnalysisConfig, AnalysisMode, InputSelection, InputSource


def test_full_and_rapid_share_portable_contract(tmp_path: Path) -> None:
    rapid = AnalysisConfig(
        mode="rapid",
        radiation="neutron",
        sample_elements="Cu, P, Pb, O, S",
        environment_elements="Al",
        limits=(0.5, 8.0),
        exclude_regions=[(2.0, 2.2)],
        rapid_gsas_validation_limit=0,
    )
    path = dump_configuration(rapid, tmp_path / "config.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["$schema"] == "radar-pd-config/v1"
    assert payload["analysis"]["mode"] == "rapid"
    assert payload["chemistry"]["sample_elements"] == ["Cu", "P", "Pb", "O", "S"]
    assert payload["full"]["profile"] == "balanced"
    assert payload["rapid"]["gsas_validation_limit"] == 0
    restored = load_configuration(path)
    assert restored.mode is AnalysisMode.RAPID
    assert restored.exclude_regions == [(2.0, 2.2)]


def test_submission_configuration_persists_safe_delivery_context(tmp_path: Path) -> None:
    config = AnalysisConfig(run_name="export-run", sample_elements=["Tb", "Be", "Ge", "O"])
    inputs = InputSelection(
        source=InputSource.IPTS_BROWSER,
        data_path="C:/temporary/upload.dat",
        instrument_path="C:/temporary/profile.instprm",
        facility_root="/HFIR",
        instrument="HB2A",
        ipts="IPTS-28749",
        data_relative_path="shared/Lalit_radarpd/Data/HB2A.dat",
        publish_results_to_ipts=True,
        publish_directory="shared/Lalit_radarpd",
        publish_subfolder="radar-pd-results",
    )

    path = dump_configuration(config, tmp_path / "submission.yaml", inputs=inputs)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    delivery = delivery_from_contract(payload)

    assert delivery["facility_root"] == "/HFIR"
    assert delivery["publish_results_to_ipts"] is True
    assert delivery["publish_directory"] == "shared/Lalit_radarpd"
    assert delivery["publish_subfolder"] == "radar-pd-results"
    assert "C:/temporary" not in path.read_text(encoding="utf-8")


def test_reusable_configuration_excludes_single_pattern_inputs(tmp_path: Path) -> None:
    config = AnalysisConfig(run_name="reusable", sample_elements=["Fe", "O"])

    path = dump_configuration(config, tmp_path / "reusable.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert "ndip_delivery" not in payload
    serialized = path.read_text(encoding="utf-8")
    for input_field in (
        "data_path",
        "instrument_path",
        "main_cif_path",
        "database_archive_path",
    ):
        assert input_field not in serialized


def test_input_source_validation() -> None:
    with pytest.raises(ValueError, match="diffraction pattern"):
        InputSelection(source=InputSource.UPLOAD, instrument_path="profile.instprm")
    with pytest.raises(ValueError, match="instrument profile"):
        InputSelection(source=InputSource.GALAXY, data_dataset_id="data-id")
    valid = InputSelection(
        source=InputSource.GALAXY,
        data_dataset_id="data-id",
        instrument_dataset_id="instrument-id",
    )
    assert valid.data_dataset_id == "data-id"


def test_galaxy_remote_source_requires_and_accepts_authorized_uris() -> None:
    with pytest.raises(ValueError, match="Galaxy remote file source"):
        InputSelection(
            source=InputSource.GALAXY_REMOTE,
            data_dataset_id="history-data-is-not-a-remote-uri",
            instrument_remote_uri="gxfiles://sns/HB2A/profile.instprm",
        )

    selection = InputSelection(
        source=InputSource.GALAXY_REMOTE,
        instrument_source="galaxy_remote",
        data_remote_uri="gxfiles://sns/HB2A/IPTS-123/shared/scan.dat",
        instrument_remote_uri="gxfiles://sns/HB2A/IPTS-123/shared/profile.instprm",
        main_cif_remote_uri="gxfiles://sns/HB2A/IPTS-123/shared/main.cif",
    )

    assert selection.data_remote_uri.endswith("scan.dat")
    assert selection.instrument_source == "galaxy_remote"


def test_ipts_pattern_can_use_independent_uploaded_instrument() -> None:
    selection = InputSelection(
        source=InputSource.IPTS_BROWSER,
        instrument_source="upload",
        data_path="/SNS/HB2A/IPTS-123/shared/reduced/scan.dat",
        data_relative_path="shared/reduced/scan.dat",
        instrument_path="/tmp/profile.instprm",
        instrument="HB2A",
        ipts="IPTS-123",
    )

    assert selection.instrument_source == "upload"
    assert selection.instrument_relative_path is None


def test_uploaded_pattern_can_use_ipts_instrument_and_cif() -> None:
    selection = InputSelection(
        source=InputSource.UPLOAD,
        instrument_source="ipts",
        data_path="/tmp/scan.dat",
        instrument_path="/SNS/HB2A/IPTS-123/shared/profiles/HB2A.instprm",
        instrument_relative_path="shared/profiles/HB2A.instprm",
        main_cif_path="/SNS/HB2A/IPTS-123/shared/structures/main.cif",
        main_cif_relative_path="shared/structures/main.cif",
        instrument="HB2A",
        ipts="IPTS-123",
    )

    assert selection.instrument == "HB2A"
    assert selection.main_cif_relative_path.endswith("main.cif")


def test_invalid_ranges_are_rejected() -> None:
    with pytest.raises(ValueError, match="less than"):
        AnalysisConfig(sample_elements=["Fe"], limits=(8.0, 2.0))


def test_advanced_controls_round_trip_through_portable_contract(tmp_path: Path) -> None:
    config = AnalysisConfig(
        mode="full",
        radiation="xray",
        sample_elements=["Fe", "O"],
        reference_masks_enabled=True,
        reference_mask_presets=["Al_fcc"],
        reference_window_mode="fixed",
        reference_fixed_half_width=0.42,
        reference_fwhm_factor=7.0,
        reference_fractional_d_tolerance=0.004,
        reference_zero_tolerance=0.08,
        reference_min_half_width=0.2,
        reference_max_half_width=1.6,
        light_calibration_enabled=True,
        full_profile="custom",
        full_dedup_threshold=0.91,
        full_score_q_max=9.5,
        full_pearson_cell_min_r=0.35,
        full_lattice_tiebreak_score_tol=0.0012,
        full_candidate_pruning=False,
        full_knee_min_points_hist=7,
        full_knee_min_relative_span=0.08,
        full_knee_keep_if_no_knee=4,
        full_knee_keep_at_most=11,
        excluded_space_groups=[1, 2, 15],
    )

    path = dump_configuration(config, tmp_path / "advanced.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    masks = payload["pattern"]["reference_phase_exclusions"]
    assert masks["half_width"] == 0.42
    assert masks["fractional_d_tolerance"] == 0.004
    assert payload["light_calibration"]["enabled"] is True
    assert payload["full"]["candidate_pruning"] is False
    assert payload["full"]["excluded_space_groups"] == [1, 2, 15]

    restored = load_configuration(path)
    assert restored.reference_fixed_half_width == config.reference_fixed_half_width
    assert restored.reference_max_half_width == config.reference_max_half_width
    assert restored.light_calibration_enabled is True
    assert restored.full_dedup_threshold == config.full_dedup_threshold
    assert restored.full_candidate_pruning is False
    assert restored.excluded_space_groups == config.excluded_space_groups


def test_advanced_reference_ranges_and_space_groups_are_validated() -> None:
    with pytest.raises(ValueError, match="minimum half-width"):
        AnalysisConfig(
            sample_elements=["Fe"],
            reference_min_half_width=5.0,
            reference_max_half_width=2.0,
        )
    with pytest.raises(ValueError, match="1 to 230"):
        AnalysisConfig(sample_elements=["Fe"], excluded_space_groups=[231])
