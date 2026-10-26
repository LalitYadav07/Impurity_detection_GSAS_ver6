"""Shared scientific setup conversion for Single pattern and Live drafts."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from .models import AnalysisConfig


def _optional_float(value):
    return None if value in (None, "") else float(value)


def _list_value(value):
    if isinstance(value, list):
        return [str(item) for item in value]
    return [] if value in (None, "") else [str(value)]


def parse_regions(value):
    regions = []
    for line in str(value or "").splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.replace(";", ",").split(",")]
        if len(parts) != 2:
            raise ValueError(f"Ignored region must be 'start,end': {line}")
        regions.append((float(parts[0]), float(parts[1])))
    return regions


def configuration_form_values(config: AnalysisConfig) -> dict[str, Any]:
    """Convert the complete validated model into editable form values."""
    values = {}
    for name in AnalysisConfig.model_fields:
        value = getattr(config, name)
        if name == "limits":
            values.update(
                fit_start=value[0] if value else None,
                fit_end=value[1] if value else None,
            )
            continue
        if name == "exclude_regions":
            values["ignore_regions"] = "\n".join(f"{a}, {b}" for a, b in value)
            continue
        if name in {
            "sample_elements",
            "environment_elements",
            "magnetic_denominators",
            "excluded_space_groups",
        }:
            value = ", ".join(str(item) for item in value)
        elif hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        elif hasattr(value, "value"):
            value = value.value
        values["analysis_mode" if name == "mode" else name] = value
    return deepcopy(values)


CONFIG_FORM_FIELDS = (
    "analysis_mode",
    "background_mode",
    "background_terms",
    "background_type",
    "cleanup_enabled",
    "environment_elements",
    "excluded_space_groups",
    "fit_end",
    "fit_start",
    "full_candidate_pruning",
    "full_cell_angle_tolerance_deg",
    "full_cell_length_tolerance_pct",
    "full_compare_candidates",
    "full_compare_cycles",
    "full_dedup_threshold",
    "full_final_joint_cell_cycles",
    "full_knee_keep_at_most",
    "full_knee_keep_if_no_knee",
    "full_knee_min_points_hist",
    "full_knee_min_relative_span",
    "full_lattice_tiebreak_score_tol",
    "full_max_passes",
    "full_min_phase_percent",
    "full_nudge_candidates",
    "full_nudge_representatives",
    "full_nudge_samples",
    "full_pearson_cell_min_r",
    "full_profile",
    "full_rwp_improvement_threshold",
    "full_score_q_max",
    "full_top_n_ml",
    "ignore_regions",
    "include_cu_kbeta",
    "instrument_mode",
    "light_calibration_enabled",
    "magnetic_denominators",
    "magnetic_precheck",
    "magnetic_q_max",
    "main_prenudge",
    "main_shadow_filter",
    "post_analysis",
    "radiation",
    "rapid_final_polish_enabled",
    "rapid_gsas_validation_limit",
    "rapid_parallel_workers",
    "rapid_phases_per_hypothesis",
    "rapid_show_family_variants",
    "rapid_stage_output_limit",
    "reference_fixed_half_width",
    "reference_fractional_d_tolerance",
    "reference_fwhm_factor",
    "reference_mask_presets",
    "reference_masks_enabled",
    "reference_max_half_width",
    "reference_min_half_width",
    "reference_window_mode",
    "reference_zero_tolerance",
    "refine_positions",
    "refine_u_iso",
    "run_name",
    "sample_elements",
    "scan_interpretation",
)


def configuration_from_form(state: Any) -> AnalysisConfig:
    run_name = str(state.run_name or "").strip()
    fit_start = _optional_float(state.fit_start)
    fit_end = _optional_float(state.fit_end)
    if (fit_start is None) != (fit_end is None):
        raise ValueError("Fit range requires both a start and an end value")
    has_main_phase = (
        (state.main_cif_source == "upload" and bool(state.main_cif_path))
        or (state.main_cif_source == "galaxy" and bool(state.history_main_cif_id))
        or (
            state.main_cif_source == "galaxy_remote" and bool(state.remote_main_cif_uri)
        )
        or (state.main_cif_source == "ipts" and bool(state.facility_main_cif_path))
    )
    presets: dict[str, dict[str, Any]] = {
        "quick": {
            "full_max_passes": 1,
            "full_min_phase_percent": 1.0,
            "full_top_n_ml": 12,
            "full_nudge_candidates": 3,
            "full_nudge_samples": 800,
            "full_nudge_representatives": 10,
            "full_compare_candidates": 1,
            "full_compare_cycles": 4,
            "full_cell_length_tolerance_pct": 0.6,
            "full_cell_angle_tolerance_deg": 1.5,
            "full_rwp_improvement_threshold": 0.12,
            "full_dedup_threshold": 0.95,
            "full_score_q_max": 8.0,
            "full_pearson_cell_min_r": 0.50,
            "full_lattice_tiebreak_score_tol": 0.0005,
            "full_candidate_pruning": True,
            "full_knee_min_points_hist": 5,
            "full_knee_min_relative_span": 0.03,
            "full_knee_keep_if_no_knee": 1,
            "full_knee_keep_at_most": 3,
        },
        "balanced": {
            "full_max_passes": 2,
            "full_min_phase_percent": 0.5,
            "full_top_n_ml": 35,
            "full_nudge_candidates": 7,
            "full_nudge_samples": 5000,
            "full_nudge_representatives": 50,
            "full_compare_candidates": 2,
            "full_compare_cycles": 6,
            "full_cell_length_tolerance_pct": 1.0,
            "full_cell_angle_tolerance_deg": 3.0,
            "full_rwp_improvement_threshold": 0.06,
            "full_dedup_threshold": 0.95,
            "full_score_q_max": 8.0,
            "full_pearson_cell_min_r": 0.50,
            "full_lattice_tiebreak_score_tol": 0.0005,
            "full_candidate_pruning": True,
            "full_knee_min_points_hist": 5,
            "full_knee_min_relative_span": 0.03,
            "full_knee_keep_if_no_knee": 2,
            "full_knee_keep_at_most": 5,
        },
        "thorough": {
            "full_max_passes": 3,
            "full_min_phase_percent": 0.25,
            "full_top_n_ml": 75,
            "full_nudge_candidates": 12,
            "full_nudge_samples": 20000,
            "full_nudge_representatives": 150,
            "full_compare_candidates": 3,
            "full_compare_cycles": 8,
            "full_cell_length_tolerance_pct": 2.0,
            "full_cell_angle_tolerance_deg": 5.0,
            "full_rwp_improvement_threshold": 0.0,
            "full_dedup_threshold": 0.95,
            "full_score_q_max": 8.0,
            "full_pearson_cell_min_r": 0.10,
            "full_lattice_tiebreak_score_tol": 0.0005,
            "full_candidate_pruning": True,
            "full_knee_min_points_hist": 5,
            "full_knee_min_relative_span": 0.03,
            "full_knee_keep_if_no_knee": 0,
            "full_knee_keep_at_most": 10,
        },
    }
    values: dict[str, Any] = {
        "mode": state.analysis_mode,
        "radiation": state.radiation,
        "instrument_mode": state.instrument_mode,
        "sample_elements": state.sample_elements,
        "environment_elements": state.environment_elements,
        "limits": None if fit_start is None else (fit_start, fit_end),
        "exclude_regions": parse_regions(state.ignore_regions),
        "reference_masks_enabled": bool(state.reference_masks_enabled),
        "reference_mask_presets": _list_value(state.reference_mask_presets),
        "reference_window_mode": state.reference_window_mode,
        "reference_fixed_half_width": _optional_float(state.reference_fixed_half_width),
        "reference_fwhm_factor": float(state.reference_fwhm_factor),
        "reference_fractional_d_tolerance": float(
            state.reference_fractional_d_tolerance
        ),
        "reference_zero_tolerance": _optional_float(state.reference_zero_tolerance),
        "reference_min_half_width": _optional_float(state.reference_min_half_width),
        "reference_max_half_width": _optional_float(state.reference_max_half_width),
        "include_cu_kbeta": bool(state.include_cu_kbeta)
        if state.radiation == "xray"
        else False,
        "background_mode": state.background_mode,
        "background_type": state.background_type,
        "background_terms": int(state.background_terms),
        "main_prenudge": bool(state.main_prenudge) if has_main_phase else False,
        "main_shadow_filter": bool(state.main_shadow_filter)
        if has_main_phase
        else False,
        "cleanup_enabled": bool(state.cleanup_enabled) if has_main_phase else False,
        "refine_u_iso": bool(state.refine_u_iso)
        if has_main_phase and state.cleanup_enabled
        else False,
        "refine_positions": bool(state.refine_positions)
        if has_main_phase and state.cleanup_enabled
        else False,
        "light_calibration_enabled": bool(state.light_calibration_enabled)
        if has_main_phase and state.radiation == "xray"
        else False,
        "magnetic_precheck": bool(state.magnetic_precheck)
        if has_main_phase and state.radiation == "neutron"
        else False,
        "magnetic_q_max": float(state.magnetic_q_max),
        "magnetic_denominators": [
            int(value)
            for value in str(state.magnetic_denominators or "")
            .replace(";", ",")
            .split(",")
            if value.strip()
        ],
        "full_profile": state.full_profile,
        "full_max_passes": int(state.full_max_passes),
        "full_min_phase_percent": float(state.full_min_phase_percent),
        "full_top_n_ml": int(state.full_top_n_ml),
        "full_nudge_candidates": int(state.full_nudge_candidates),
        "full_nudge_samples": int(state.full_nudge_samples),
        "full_nudge_representatives": int(state.full_nudge_representatives),
        "full_compare_candidates": int(state.full_compare_candidates),
        "full_compare_cycles": int(state.full_compare_cycles),
        "full_final_joint_cell_cycles": int(
            getattr(state, "full_final_joint_cell_cycles", 12)
        ),
        "full_cell_length_tolerance_pct": float(state.full_cell_length_tolerance_pct),
        "full_cell_angle_tolerance_deg": float(state.full_cell_angle_tolerance_deg),
        "full_rwp_improvement_threshold": float(state.full_rwp_improvement_threshold),
        "full_dedup_threshold": float(state.full_dedup_threshold),
        "full_score_q_max": float(state.full_score_q_max),
        "full_pearson_cell_min_r": float(state.full_pearson_cell_min_r),
        "full_lattice_tiebreak_score_tol": float(state.full_lattice_tiebreak_score_tol),
        "full_candidate_pruning": bool(state.full_candidate_pruning),
        "full_knee_min_points_hist": int(state.full_knee_min_points_hist),
        "full_knee_min_relative_span": float(state.full_knee_min_relative_span),
        "full_knee_keep_if_no_knee": int(state.full_knee_keep_if_no_knee),
        "full_knee_keep_at_most": int(state.full_knee_keep_at_most),
        "excluded_space_groups": [
            int(value)
            for value in re.split(
                r"[,;\s]+", str(state.excluded_space_groups or "").strip()
            )
            if value
        ],
        "rapid_phases_per_hypothesis": int(state.rapid_phases_per_hypothesis),
        "rapid_stage_output_limit": int(state.rapid_stage_output_limit),
        "rapid_gsas_validation_limit": int(state.rapid_gsas_validation_limit),
        "rapid_parallel_workers": int(state.rapid_parallel_workers),
        "rapid_show_family_variants": bool(state.rapid_show_family_variants),
        "rapid_final_polish_enabled": bool(state.rapid_final_polish_enabled),
        "post_analysis": state.post_analysis,
        "scan_interpretation": state.scan_interpretation,
    }
    if state.analysis_mode == "full" and state.full_profile in presets:
        values.update(presets[state.full_profile])
    if run_name:
        values["run_name"] = run_name
    return AnalysisConfig(**values)
