from pathlib import Path

import pytest
import yaml

from radar_pd_nova.configuration import dump_configuration, load_configuration
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
