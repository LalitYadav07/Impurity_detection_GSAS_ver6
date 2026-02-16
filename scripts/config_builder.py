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
from typing import Dict, Any, List

def build_pipeline_config(
    run_name: str,
    data_file: str,
    instprm_file: str,
    allowed_elements: List[str],
    main_cif: str = None,
    work_root: str = None,
    project_root: str = None,
    db_root: str = None,
    min_impurity_percent: float = 0.5,
    max_passes: int = 3,
    sample_env_elements: List[str] = None,
    instrument_mode: str = "auto",
    advanced_params: Dict[str, Any] = None
) -> str:
    """
    Builds a pipeline_config.yaml content and returns it as a string.
    """
    if project_root is None:
        project_root = str(Path(__file__).resolve().parent.parent)
    
    if work_root is None:
        work_root = str(Path(project_root) / "runs")
    
    if db_root is None:
        db_root = str(Path(project_root) / "data" / "database_aug")

    # Global section
    config = {
        "PROJECT_ROOT": project_root,
        "WORK_ROOT": str(Path(project_root) / "runs"),
        "DATA_ROOT": str(Path(project_root) / "data"),
        "work_root": work_root,
        "ml_components_dir": str(Path(project_root) / "ML_components"),
        "db": {
            "catalog_csv": str(Path(db_root) / "catalog_deduplicated.csv"),
            "original_json": str(Path(db_root) / "highsymm_metadata.json"),
            "profiles_dir": str(Path(db_root) / "profiles64"),
            "stable_csv": str(Path(db_root) / "mp_experimental_stable.csv"),
        },
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
        "rwp_improve_eps": 0.05,
        "knee_filter": {
            "enable_hist": True,
            "min_points_hist": 5,
            "min_rel_span": 0.03,
            "guard_frac": 0.05,
        },
        "stage4": {
            "wavelength": 1.54,
            "two_theta_range": [5.0, 160.0],
            "samples": 5000,
            "reps": 50,
            "len_tol_pct": 1.0,  # User-requested default
            "ang_tol_deg": 3.0,  # User-requested default
        },
        "hist_filter": {
            "min_active_bins": 4,
            "min_sum_residual": 0.0,
            "topN": 50,
        },
        "corr_threshold": 0.95,
        "exclude_sg": [1, 2],
        "background": {
            "type": "chebyschev-1",
            "terms": 12,
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
                "allow_with": ["O"],
                "ban_cross_with_base": True,
                "ignore_in_budget": True
            },
            "disallow_pure": ["O", "C"]
        }
    })

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

    config["datasets"] = [dataset]

    return yaml.dump(config, sort_keys=False)
