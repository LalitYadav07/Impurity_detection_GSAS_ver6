"""The batch artifact must remain small, normalized and independently readable."""

import hashlib
import json
from pathlib import Path

from scripts.ndip_lattice import lattice_from_records, lattice_quality
from scripts.ndip_observables import observations_from_project


def test_observation_packet_hashes_source_and_initial_instrument_parameters(tmp_path):
    path = tmp_path / "trusted-pipeline-output.gpx"
    path.write_bytes(b"test bytes, not unpickled")
    records = [[["PWDR test", {}], ["Instrument Parameters", [{"difC": [100.0, 101.0, True]}, {}]]]]
    packet = observations_from_project(
        path, [], records, mode="full", input_provenance={"resolver": "resolver-v1"}
    )
    assert packet["source_gpx_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    original_profile = packet["context"]["profile_sha256"]
    records[0][1][1][0]["difC"][1] = 105.0
    assert (
        observations_from_project(path, [], records, mode="full", input_provenance={})["context"][
            "profile_sha256"
        ]
        == original_profile
    )
    records[0][1][1][0]["difC"][0] = 200.0
    assert (
        observations_from_project(path, [], records, mode="full", input_provenance={})["context"][
            "profile_sha256"
        ]
        != original_profile
    )
    json.dumps(packet, allow_nan=False)


def test_fallback_extraction_never_certifies_a_fixed_or_unknown_parameter():
    phase = [
        "phase",
        {
            "General": {
                "Cell": [False, 4, 4, 4, 90, 90, 90, 64],
                "Mass": 10,
                "SGData": {"SpGrp": "P 1"},
            },
            "Histograms": {"PWDR x": {"Scale": [1.0, True]}},
        },
    ]
    rows, _ = lattice_from_records([[["Phases", {}], phase]], source="trusted.gpx")
    assert set(rows[0]["metric_states"].values()) == {"unknown"}
    assert rows[0]["cell_covariance"] is None and rows[0]["fraction_state"] == "unknown"
    assert rows[0]["phase_key"] and rows[0]["basis_key"]
    rows[0]["covariance_status"] = "gsas_esd_mismatch"
    assert any(w["code"] == "cell_covariance_review" for w in lattice_quality(rows))
