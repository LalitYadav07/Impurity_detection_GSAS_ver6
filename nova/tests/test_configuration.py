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


def test_invalid_ranges_are_rejected() -> None:
    with pytest.raises(ValueError, match="less than"):
        AnalysisConfig(sample_elements=["Fe"], limits=(8.0, 2.0))
