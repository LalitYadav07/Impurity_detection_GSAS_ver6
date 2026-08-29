"""Portable RADAR-PD configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import AnalysisConfig, InputSelection


DELIVERY_SCHEMA = "radar-pd-delivery/v1"
_DELIVERY_FIELDS = (
    "facility_root",
    "instrument",
    "ipts",
    "run_number",
    "bank",
    "data_relative_path",
    "instrument_relative_path",
    "main_cif_relative_path",
    "publish_results_to_ipts",
    "publish_directory",
    "publish_subfolder",
)


def delivery_contract(inputs: InputSelection) -> dict[str, Any]:
    """Return restart-safe facility context without local paths or credentials."""

    payload = {
        field: getattr(inputs, field)
        for field in _DELIVERY_FIELDS
        if getattr(inputs, field) is not None
    }
    payload["$schema"] = DELIVERY_SCHEMA
    payload["original_source"] = inputs.source.value
    return payload


def delivery_from_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract validated run-delivery fields from a submitted configuration."""

    delivery = payload.get("ndip_delivery")
    if not isinstance(delivery, dict) or delivery.get("$schema") != DELIVERY_SCHEMA:
        return {}
    return {field: delivery[field] for field in _DELIVERY_FIELDS if field in delivery}


def dump_configuration(
    config: AnalysisConfig,
    path: str | Path,
    *,
    inputs: InputSelection | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = config.portable_contract()
    if inputs is not None:
        payload["ndip_delivery"] = delivery_contract(inputs)
    destination.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return destination


def load_configuration(path: str | Path) -> AnalysisConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("$schema") != "radar-pd-config/v1":
        raise ValueError("Expected a radar-pd-config/v1 YAML document")
    return config_from_contract(payload)


def config_from_contract(payload: dict[str, Any]) -> AnalysisConfig:
    analysis = payload.get("analysis") or {}
    chemistry = payload.get("chemistry") or {}
    pattern = payload.get("pattern") or {}
    masks = pattern.get("reference_phase_exclusions") or {}
    background = payload.get("background") or {}
    main_phase = payload.get("main_phase") or {}
    cleanup = main_phase.get("cleanup") or {}
    magnetic = payload.get("magnetic_precheck") or {}
    light_calibration = payload.get("light_calibration") or {}
    full = payload.get("full") or {}
    rapid = payload.get("rapid") or {}
    return AnalysisConfig(
        mode=analysis.get("mode", "rapid"),
        radiation=analysis.get("radiation", "neutron"),
        instrument_mode=analysis.get("instrument_mode", "auto"),
        sample_elements=chemistry.get("sample_elements") or [],
        environment_elements=chemistry.get("environment_elements") or [],
        limits=pattern.get("limits"),
        exclude_regions=pattern.get("exclude_regions") or [],
        reference_masks_enabled=bool(masks.get("enabled", False)),
        reference_mask_presets=masks.get("presets") or [],
        reference_window_mode=masks.get("window_mode", "auto"),
        reference_fixed_half_width=masks.get("half_width"),
        reference_fwhm_factor=float(masks.get("fwhm_factor", 6.0)),
        reference_fractional_d_tolerance=float(masks.get("fractional_d_tolerance", 0.003)),
        reference_zero_tolerance=masks.get("zero_tolerance"),
        reference_min_half_width=masks.get("min_half_width"),
        reference_max_half_width=masks.get("max_half_width"),
        include_cu_kbeta=bool(masks.get("include_cu_kbeta", False)),
        background_mode=background.get("mode", "auto_fixed_points"),
        background_type=background.get("type", "chebyschev-1"),
        background_terms=int(background.get("terms", 6)),
        main_prenudge=bool(main_phase.get("prenudge", True)),
        main_shadow_filter=bool(main_phase.get("shadow_filter", True)),
        cleanup_enabled=bool(cleanup.get("enabled", False)),
        refine_u_iso=bool(cleanup.get("refine_u_iso", False)),
        refine_positions=bool(cleanup.get("refine_positions", False)),
        light_calibration_enabled=bool(light_calibration.get("enabled", False)),
        magnetic_precheck=bool(magnetic.get("enabled", False)),
        magnetic_q_max=float(magnetic.get("q_max", 4.0)),
        magnetic_denominators=magnetic.get("denominators") or [2, 3, 4],
        full_profile=full.get("profile", "balanced"),
        full_max_passes=int(full.get("max_passes", 2)),
        full_min_phase_percent=float(full.get("min_phase_percent", 0.5)),
        full_top_n_ml=int(full.get("top_n_ml", 35)),
        full_nudge_candidates=int(full.get("nudge_candidates", 7)),
        full_nudge_samples=int(full.get("nudge_samples", 5000)),
        full_nudge_representatives=int(full.get("nudge_representatives", 50)),
        full_compare_candidates=int(full.get("compare_candidates", 2)),
        full_compare_cycles=int(full.get("compare_cycles", 6)),
        full_cell_length_tolerance_pct=float(full.get("cell_length_tolerance_pct", 1.0)),
        full_cell_angle_tolerance_deg=float(full.get("cell_angle_tolerance_deg", 3.0)),
        full_rwp_improvement_threshold=float(full.get("rwp_improvement_threshold", 0.06)),
        full_dedup_threshold=float(full.get("dedup_threshold", 0.95)),
        full_score_q_max=float(full.get("score_q_max", 8.0)),
        full_pearson_cell_min_r=float(full.get("pearson_cell_min_r", 0.50)),
        full_lattice_tiebreak_score_tol=float(full.get("lattice_tiebreak_score_tol", 0.0005)),
        full_candidate_pruning=bool(full.get("candidate_pruning", True)),
        full_knee_min_points_hist=int(full.get("knee_min_points_hist", 5)),
        full_knee_min_relative_span=float(full.get("knee_min_relative_span", 0.03)),
        full_knee_keep_if_no_knee=int(full.get("knee_keep_if_no_knee", 2)),
        full_knee_keep_at_most=int(full.get("knee_keep_at_most", 5)),
        excluded_space_groups=full.get("excluded_space_groups", [1, 2]),
        rapid_phases_per_hypothesis=int(rapid.get("phases_per_hypothesis", 3)),
        rapid_stage_output_limit=int(rapid.get("stage_output_limit", 10)),
        rapid_gsas_validation_limit=int(rapid.get("gsas_validation_limit", 5)),
        rapid_parallel_workers=int(rapid.get("parallel_workers", 4)),
        rapid_show_family_variants=bool(rapid.get("show_family_variants", True)),
        rapid_final_polish_enabled=bool(rapid.get("final_polish_enabled", False)),
    )
