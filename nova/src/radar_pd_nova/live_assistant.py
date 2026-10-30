"""Deterministic questions and edits for configuring one Live experiment.

The conversation is a view of the existing Live draft. It never submits a
scan, starts a watch, saves a Galaxy configuration, or owns scientific defaults.
Incomplete drafts are allowed; the ordinary Live preflight is the final gate.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
import re
from typing import Any

from .setup_forms import CONFIG_FORM_FIELDS
from .library_setup import LIBRARY_FIELDS
from .uploads import _ELEMENT_SYMBOLS


STEPS = ("experiment", "selection", "structures", "chemistry", "settings", "review")
TITLES = {
    "experiment": "Find your experiment",
    "selection": "Choose samples and banks",
    "structures": "Choose candidate structures",
    "chemistry": "Confirm sample chemistry",
    "settings": "Set the analysis goal",
    "review": "Review the Live setup",
}
PROMPTS = {
    "experiment": "Which beamline and IPTS are you working on? I will look up the samples and reduced-data banks available to this Live session.",
    "selection": "Which samples and banks should share this setup? Select samples with the same chemistry and analysis requirements. Names and run titles help you review the choice.",
    "structures": "Which structures should RADAR-PD search? Add candidate CIFs or a reusable library here. Supply a main-phase CIF only when you know the dominant phase.",
    "chemistry": "Which elements may occur in the sample, including possible impurities? Enter can or environment elements separately. CIF chemistry is a suggestion you can edit.",
    "settings": "What do you want the Live analysis to do? Choose a goal to fill the search settings together, or keep your current settings. Fit regions and background stay editable.",
    "review": "Review the actual configuration and check the experiment. When it is ready, finish setup and use the normal Start monitoring control.",
}

STEP_FIELDS = {
    "experiment": ("live_beamline", "powgen_ipts", "powgen_source_subfolder", "hb2a_normalization", "snap_product_family"),
    "selection": (
        "powgen_processing_scope", "powgen_processing_sample_ids", "powgen_bank_scope",
        "powgen_processing_banks", "nomad_physical_bank", "powgen_backfill_mode",
        "powgen_include_background", "powgen_include_ramps", "powgen_excluded_runs",
        "powgen_wavelength", "powgen_instrument_source", "powgen_instrument_path",
        "powgen_instrument_dataset_id", "nomad_profile_bank",
        "nomad_profile_acknowledged", "nomad_allow_exploratory",
        "hb2a_profile_acknowledged", "hb2a_collimator_override", "hb2a_allow_stopped",
        "hb2a_temperature_channel", "hb2a_temperature_unit", "hb2a_condition_reviewed",
        "hb2a_magnetic_field_channel", "hb2a_magnetic_field_unit", "hb2a_magnetic_field_reviewed",
        "hb2a_pressure_channel", "hb2a_pressure_unit", "hb2a_pressure_reviewed",
        "snap_temperature_channel", "snap_temperature_unit", "snap_temperature_reviewed", "snap_condition_window_reviewed",
    ),
    "structures": (
        "live_database_source", "live_library_archive_source", "live_database_archive_path",
        "powgen_database_dataset_id", "live_library_builder_local_paths",
        "live_library_builder_cif_ids", "live_library_builder_name",
        "powgen_main_cif_source", "powgen_main_cif_path", "powgen_main_cif_dataset_id",
    ),
    "chemistry": ("live_sample_elements", "live_environment_elements", "live_configuration_source", "powgen_configuration_dataset_id"),
    "settings": ("live_configuration_source", "powgen_configuration_dataset_id") + tuple(
        "live_" + name for name in CONFIG_FORM_FIELDS
        if name not in {"sample_elements", "environment_elements"}
    ),
}
EDIT_FIELDS = frozenset(field for fields in STEP_FIELDS.values() for field in fields)
WATCH_FIELDS = tuple(sorted(EDIT_FIELDS | {
    "live_" + name for name in LIBRARY_FIELDS
} | {"powgen_preflight_ready", "powgen_sample_discovery_ipts", "powgen_discovered_sample_options", "powgen_bank_options"}))

HELP = {
    "background": "Background terms are coefficients describing the smooth baseline. More terms are more flexible and can also absorb real scattering. Preview recommendations are starting points; broad features need review.",
    "chemistry": "Sample elements restrict candidate chemistry; they are not a formula or a composition measurement. Include possible impurity elements. A main CIF alone may omit contaminants. Can/environment chemistry is a separate filter from masking peaks.",
    "main": "A main-phase CIF is a known dominant phase used to anchor refinement. A possible or minor phase belongs in the candidate library. The pipeline applies main-phase protection, including a default 20 wt% guard.",
    "library": "Built-in searches the installed MP/COD catalog. Mini searches only your CIFs. Augmented adds your CIFs to the built-in catalog. A ZIP of source CIFs must be built before it can serve as a portable RADAR-PD library.",
    "banks": "Each POWGEN bank is analyzed separately. NOMAD and SNAP require one source histogram and an explicitly matched profile Bank label. HB-2A uses one combined detector-bank pattern. Matching bank numbers alone does not establish a calibration mapping.",
    "profile": "The instrument profile describes coordinate calibration and peak shape. POWGEN and reviewed HB-2A configurations use exact facility mappings when available. NOMAD and SNAP require a supplied .instprm and explicit physical/profile bank mapping. Unmatched configurations need a reviewed profile rather than a guessed calibration.",
    "range": "Fit start/end and ignored regions use the native pattern axis: microseconds for TOF products and degrees 2θ for HB-2A. Displaying d or Q does not change those input units. Say 'exclude Al can peaks' to enable profile-based reference masks, then inspect their widths and overlaps with sample reflections. Can chemistry alone does not enable masks. Negative intensities alone are not invalid data.",
    "goal": "Balanced performs sequential discovery with the current Balanced preset. Rapid ranks hypotheses with focused refinements. Thorough increases the search budget. A minimum phase-fraction stop setting is not a guaranteed detection limit.",
    "magnetic": "The neutron magnetic precheck uses residuals after fitting a supplied main phase. It checks commensurate indexing consistency; it does not determine a magnetic structure.",
}


def signature(values: dict[str, Any]) -> str:
    return sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()


def parse_elements(value: Any, *, optional: bool = False) -> list[str]:
    """Require unambiguous symbols; never reinterpret CO as cobalt."""
    tokens = re.split(r"[,;\s]+", str(value or "").strip())
    tokens = list(dict.fromkeys(token for token in tokens if token))
    if not tokens and not optional:
        raise ValueError("Enter the sample element symbols, for example Fe, O. Include possible impurity elements.")
    bad = [token for token in tokens if token not in _ELEMENT_SYMBOLS]
    if bad:
        raise ValueError("Use separate, case-sensitive element symbols; check: " + ", ".join(bad))
    return tokens


def goal_patch(goal: str) -> dict[str, Any]:
    """Select an existing preset; configuration_from_form resolves its values."""
    if goal in {"balanced", "thorough", "quick"}:
        return {"live_analysis_mode": "full", "live_full_profile": goal}
    if goal == "rapid":
        return {"live_analysis_mode": "rapid"}
    raise ValueError("Choose Balanced, Rapid, Thorough, or keep the current settings.")


def parse_message(text: str) -> tuple[str, Any]:
    """Compatibility entry point; the controller also supplies question context."""
    from .live_assistant_language import interpret_reply
    reply = interpret_reply(text)
    return reply.action, reply.value


@dataclass
class SetupConversation:
    """Receipts over a partial Live draft, separate from an executable config."""
    confirmed: dict[str, str] = field(default_factory=dict)
    messages: list[dict[str, str]] = field(default_factory=list)
    undo: list[dict[str, Any]] = field(default_factory=list)
    pending: dict[str, Any] | None = None

    def invalidate(self, snapshots: dict[str, dict[str, Any]]) -> str | None:
        for index, step in enumerate(STEPS[:-1]):
            if step in self.confirmed and self.confirmed[step] != signature(snapshots[step]):
                for downstream in STEPS[index:]:
                    self.confirmed.pop(downstream, None)
                return step
        return None

    def say(self, role: str, text: str) -> None:
        self.messages.append({"role": role, "text": text})
        self.messages = self.messages[-30:]

    def apply(self, state: Any, values: dict[str, Any], reason: str) -> None:
        if set(values) - EDIT_FIELDS:
            raise ValueError("The assistant can only edit Live setup fields")
        if any(isinstance(value, (float, int)) and not math.isfinite(value) for value in values.values()):
            raise ValueError("Settings must be finite")
        before = {key: deepcopy(getattr(state, key)) for key in values}
        for key, value in values.items():
            setattr(state, key, deepcopy(value))
        self.undo.append({"before": before, "after": deepcopy(values), "reason": reason})
        self.undo = self.undo[-10:]

    def undo_last(self, state: Any) -> str:
        if not self.undo:
            raise ValueError("There is no assistant edit to undo. Your manual entries are preserved.")
        action = self.undo[-1]
        if any(getattr(state, key) != value for key, value in action["after"].items()):
            raise ValueError("Those fields were edited again. Review them directly so undo does not replace newer values.")
        self.undo.pop()
        for key, value in action["before"].items():
            setattr(state, key, deepcopy(value))
        return "Undid: " + action["reason"]
