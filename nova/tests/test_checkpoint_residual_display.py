from radar_pd_nova.results import _full_model_rows


def test_same_final_project_hash_displays_its_known_residual_only_for_matching_models():
    final_hash = "a" * 64
    rows = _full_model_rows({
        "final_rwp": 9.2921236,
        "observables": {"source_gpx_sha256": final_hash},
        "gpx_projects": [
            {"label": "anchor", "sha256": final_hash},
            {"label": "final", "sha256": final_hash},
            {"label": "different", "sha256": "b" * 64},
            {"label": "unidentified"},
            {"label": "explicit", "sha256": final_hash, "rwp": 10.5},
        ],
    })
    assert [row["rwp"] for row in rows] == ["9.292", "9.292", "-", "-", "10.500"]


def test_missing_final_hash_does_not_assign_a_final_residual_to_other_checkpoints():
    rows = _full_model_rows({
        "final_rwp": 4.2,
        "gpx_projects": [{"label": "final", "stage": "final_refinement"}],
    })
    assert rows[0]["rwp"] == "-"
