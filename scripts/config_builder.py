#!/usr/bin/env python3
"""
Pipeline Configuration Builder

This script dynamically generates the default `pipeline_config.yaml` configuration file.
It defines:
- Default paths for data, databases, and output directories.
- Tunable parameters for all pipeline stages (Stage-0 to Stage-4).
- Instrument parameter defaults.
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    from .db_pack import build_db_config, validate_db_config
except ImportError:
    from db_pack import build_db_config, validate_db_config

def build_pipeline_config(
    run_name: str,
    data_file: str,
    instprm_file: str,
    allowed_elements: List[str],
    main_cif: str = None,
    work_root: str = None,
    project_root: str = None,
    db_root: str = None,
    original_json_override: str = None,
    cif_map_json_override: str = None,
    db_config_override: Dict[str, Any] = None,
    min_impurity_percent: float = 0.5,
    max_passes: int = 3,
    sample_env_elements: List[str] = None,
    instrument_mode: str = "auto",
    advanced_params: Dict[str, Any] = None,
    limits: Optional[List[float]] = None,
    exclude_regions: Optional[List[List[float]]] = None,
    reference_phase_exclusions: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Builds a pipeline_config.yaml content and returns it as a string.
    
    original_json_override: If set, use this path for db.original_json instead of
    the default (db_root/highsymm_metadata.json).
    cif_map_json_override: Optional CIF path overlay map for custom/augmented DB packs.
    db_config_override: Optional explicit DB config entries to merge over the generated
    pack layout. Intended for custom pack selection.
    """
    if not run_name or not str(run_name).strip():
        raise ValueError("run_name is required")
    if not data_file:
        raise ValueError("data_file is required")
    if not instprm_file:
        raise ValueError("instprm_file is required")

    if project_root is None:
        project_root = str(Path(__file__).resolve().parent.parent)
    
    if work_root is None:
        work_root = str(Path(project_root) / "runs")
    
    if db_root is None:
        db_root = str(Path(project_root) / "data" / "database_aug")

    db_config = build_db_config(
        db_root,
        original_json=original_json_override,
        cif_map_json=cif_map_json_override,
    )
    if db_config_override:
        for key, value in db_config_override.items():
            if value is None:
                db_config.pop(key, None)
            else:
                db_config[key] = value
    validate_db_config(db_config)

    resolved_work_root = Path(work_root)
    run_parent_root = resolved_work_root.parent if resolved_work_root.name == run_name else resolved_work_root

    # Global section
    config = {
        "PROJECT_ROOT": project_root,
        "WORK_ROOT": str(run_parent_root),
        "DATA_ROOT": str(Path(project_root) / "data"),
        "work_root": work_root,
        "ml_components_dir": str(Path(project_root) / "ML_components"),
        "instrument_map": {
            "cw": str(Path(project_root) / "examples" / "tbssl" / "hb2a_si_ge113.instprm"),
            "hb2a": str(Path(project_root) / "examples" / "tbssl" / "hb2a_si_ge113.instprm"),
            "tof": str(Path(project_root) / "examples" / "lk99" / "2023A_June_HighRes_60HzB3_CWL2p665.instprm"),
            "pg3": str(Path(project_root) / "examples" / "lk99" / "2023A_June_HighRes_60HzB3_CWL2p665.instprm"),
        },
        "db": db_config,
        "allowed_elements": allowed_elements,
        "min_impurity_percent": min_impurity_percent,
        "max_passes": max_passes,
        "instrument_mode": instrument_mode,
    }

    # Add advanced defaults
    config.update({
        "top_candidates": 10,
        "hap_init": 0.05,
        "max_joint_cycles": 8,
        "polish_strategy": "adaptive",
        "polish_refine_cell": True,
        "polish_refine_background": True,
        "polish_refine_main_cell": True,
        "polish_defer_main_cell": True,
        "polish_refine_existing_cells": False,
        "polish_escalate_on_failure": True,
        "polish_stabilization_cycles": 0,
        "polish_cell_trial_cycles": 1,
        "polish_final_cycles": 0,
        "polish_skip_fresh_lst_regen": True,
        "rwp_improve_eps": 0.05,
        "knee_filter": {
            "enable_hist": True,
            "min_points_hist": 5,
            "min_rel_span": 0.03,
            "guard_frac": 0.05,
        },
        "stage4": {
            "radiation": "neutron",
            "wavelength": 1.54,
            "two_theta_range": [5.0, 160.0],
            "samples": 5000,
            "reps": 50,
            "len_tol_pct": 1.0,  # User-requested default
            "ang_tol_deg": 3.0,  # User-requested default
            "seed": 0,
            "score_q_max": 8.0,
            "pearson_q_max": 8.0,
            "pearson_min_points": 25,
            "lattice_tiebreak_score_tol": 5e-4,
            "pearson_cell_refine_min_r": 0.50,
        },
        "hist_filter": {
            "min_active_bins": 4,
            "min_sum_residual": 0.0,
            "topN": 50,
        },
        "corr_threshold": 0.95,
        "exclude_sg": [1, 2],
        "background": {
            "mode": "auto_fixed_points",
            "type": "chebyschev-1",
            "terms": 6,
        },
        "light_calibration": {
            "enabled": False,
            "zero_cycles": 1,
            "profile_cycles": 2,
            "accept_rwp_worsen": 0.15,
            "terms": ["Zero", "U", "V", "W"],
        },
        "main_phase_prenudge": {
            "enabled": True,
            "apply_only_user_main": True,
            "trigger_rwp": 18.0,
            "hard_rwp": 35.0,
            "min_peak_support": 0.50,
            "min_rwp_for_peak_support_trigger": 8.0,
            "strongest_barely_supported_fraction": 0.75,
            "peak_match_tolerance_q": 0.035,
            "frac_window": 0.01,
            "angle_window_deg": 1.0,
            "fail_unresolved_main": False,
        },
        "main_phase_guard": {
            "enabled": True,
            "apply_only_user_main": True,
            "min_weight_pct": 20.0,
        },
        "main_phase_shadow": {
            "enabled": True,
            "top_main_peaks": 8,
            "top_candidate_peaks": 10,
            "peak_match_tolerance_q": 0.040,
            "min_target_prominence_fraction": 0.03,
            "nudge_filter_enabled": True,
            "filter_top_main_peaks": 5,
            "filter_top_candidate_peaks": 5,
            "filter_min_overlap_count": 3,
            "filter_min_overlap_fraction": 0.60,
            "filter_min_shadow_intensity_fraction": 0.60,
            "filter_max_unique_supported_count": 1,
            "filter_max_unique_supported_fraction": 0.25,
            "refill_attempts": 2,
            "require_reliable_main_rwp_max": 25.0,
            "require_reliable_main_peak_support": 0.55,
        },
        "main_phase_cleanup": {
            "enabled": False,
            "apply_only_user_main": True,
            "refine_u_iso": False,
            "refine_positions": False,
            "cycles": 1,
            "accept_rwp_worsen": 0.15,
            "min_rwp_improvement": 0.05,
            "max_position_shift": 0.15,
            "min_u_iso": 0.0,
            "max_u_iso": 0.20,
        },
        "magnetic_precheck": {
            "enabled": False,
            "q_max": 6.0,
            "max_hkl": 8,
            "denominators": [2, 3],
            "width_grid": [0.025, 0.04, 0.065],
            "pseudo_voigt_eta": 0.35,
            "max_k_vectors": 80,
            "max_positions_per_k": 260,
            "top_residual_peaks": 8,
            "null_trials": 48,
            "random_seed": 13,
            "include_gamma": False,
        },
        "xray_doublet": {
            "enabled": "auto",
            "default_intensity_ratio": 0.5,
            "apply_to_64_ml_input": True,
            "apply_to_64_similarity": True,
            "apply_to_512": True,
            "apply_to_lattice_nudge": True,
        },
        "reference_phase_exclusions": {
            "enabled": False,
            "presets": [],
            "window_mode": "auto",
            "fwhm_factor": 6.0,
            "fractional_d_tolerance": 0.003,
            "zero_tolerance_deg": 0.05,
            "zero_tolerance_tof": 25.0,
            "min_half_width_deg": 0.35,
            "max_half_width_deg": 2.00,
            "fallback_half_width_deg": 0.75,
            "min_half_width_tof": 75.0,
            "max_half_width_tof": 750.0,
            "fallback_half_width_tof": 200.0,
            "include_cu_kbeta": False,
        },
        "element_filter": {
            "max_offlist_elements": 0,
            "wildcard_relation": "same_family",
            "require_base": True,
            "ignore_elements": [],
            "disallow_offlist": [],
            "sample_env": {
                "elements": sample_env_elements if sample_env_elements else [],
                "allow_pure": True,
                "allow_with": [],
                "ban_cross_with_base": True,
                "ignore_in_budget": True
            },
            "disallow_pure": ["O", "C"]
        }
    })

    if reference_phase_exclusions is not None:
        config["reference_phase_exclusions"].update(reference_phase_exclusions)

    # Override with advanced_params if provided
    if advanced_params:
        for k, v in advanced_params.items():
            if isinstance(v, dict) and k in config and isinstance(config[k], dict):
                config[k].update(v)
            else:
                config[k] = v

    # Dataset entry
    dataset = {
        "name": run_name,
        "data_path": data_file,
        "instprm_path": instprm_file,
        "mode": instrument_mode,
    }
    if main_cif:
        dataset["main_cif"] = main_cif
    if limits:
        dataset["limits"] = [float(limits[0]), float(limits[1])]
    if exclude_regions:
        dataset["exclude_regions"] = [
            [float(pair[0]), float(pair[1])]
            for pair in exclude_regions
            if pair is not None and len(pair) == 2
        ]

    config["datasets"] = [dataset]

    return yaml.dump(config, sort_keys=False)
