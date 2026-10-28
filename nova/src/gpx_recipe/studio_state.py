# Adapted from Diffraction Studio (MIT), with documented parameter-identity fixes.
# See studio-provenance.json and STUDIO-LICENSE; no Studio runtime is required.
from __future__ import annotations

import copy
import hashlib
import importlib
import math
import threading
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import numpy as np


STATE_SCHEMA_VERSION = "gsasii-refinement-state/v1"
_GSASII_STATE_LOCK = threading.RLock()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except Exception:
            pass
    return str(value)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _import_gsasii_internal(g2scriptable: ModuleType, module_name: str) -> ModuleType:
    package = str(getattr(g2scriptable, "__package__", "") or "").strip()
    candidates = []
    if package:
        candidates.append(f"{package}.{module_name}")
    candidates.extend((f"GSASII.{module_name}", module_name))
    errors: list[str] = []
    for candidate in dict.fromkeys(candidates):
        try:
            return importlib.import_module(candidate)
        except Exception as exc:
            errors.append(f"{candidate}: {exc!r}")
    raise ImportError(f"Unable to import {module_name}: {'; '.join(errors)}")


def _load_gsasii_model_inputs(
    g2io: ModuleType,
    gpx_path: Path,
) -> tuple[dict[str, Any], Any, Any, Any, Any, Any]:
    """Load GPX model inputs in the identity-registration order GSAS-II requires."""

    path = str(gpx_path)
    controls = g2io.GetControls(path)
    # ReadCheckConstraints uses this exact ordering. GetUsedHistogramsAndPhases
    # registers the random phase/histogram IDs referenced by G2VarObj constraint
    # records; ReadConstraints may otherwise return empty lists without error.
    histograms, phases = g2io.GetUsedHistogramsAndPhases(path)
    constraint_records, fixed_variables = g2io.ReadConstraints(path)
    rigid_bodies = g2io.GetRigidBodies(path)
    return (
        controls,
        histograms,
        phases,
        constraint_records,
        fixed_variables,
        rigid_bodies,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_variable(name: str) -> tuple[int | None, int | None, str, int | None]:
    fields = name.split(":")

    def integer(value: str | None) -> int | None:
        try:
            return int(value) if value not in (None, "", "*") else None
        except (TypeError, ValueError):
            return None

    phase_id = integer(fields[0] if fields else None)
    histogram_id = integer(fields[1] if len(fields) > 1 else None)
    family = fields[2] if len(fields) > 2 else name
    atom_id = integer(fields[3] if len(fields) > 3 else None)
    return phase_id, histogram_id, family, atom_id


def _parameter_group(name: str, source_group: str) -> str:
    _, _, family, _ = _split_variable(name)
    if source_group == "rigid_body":
        return "rigid_body"
    if family.startswith("Back") or family.startswith("Debye") or family.startswith("BkPk"):
        return "background"
    # Histogram displacement variables begin with ``D`` too, so classify them
    # before the reciprocal-cell fallback below. Otherwise Debye-Scherrer
    # DisplaceX/DisplaceY are reported as lattice variables and the agent loses
    # the distinction between specimen geometry and crystal structure.
    if source_group == "histogram" and family in {
        "Shift",
        "DisplaceX",
        "DisplaceY",
    }:
        return "sample_displacement"
    if source_group == "histogram" and family == "Zero":
        return "instrument_zero"
    if source_group == "histogram" and (
        family in {"U", "V", "W", "X", "Y", "Z", "SH/L"}
        or family == "alpha"
        or family.startswith("beta-")
        or family.startswith("sig-")
    ):
        return "instrument_profile"
    if family.startswith("D") and source_group == "hap":
        return "hydrostatic_strain"
    if family in {"A0", "A1", "A2", "A3", "A4", "A5", "Vol"} or family.startswith("D"):
        return "lattice"
    if family.startswith(("Ax", "Ay", "Az", "dAx", "dAy", "dAz")):
        return "atomic_coordinates"
    if family.startswith("AU") or family.startswith("AD"):
        return "atomic_displacement"
    if family == "Afrac":
        return "occupancy"
    if family == "WgtFrac":
        return "phase_fraction"
    if family in {"Size;i", "Size;mx"}:
        return "crystallite_size"
    if family in {"Mustrain;i", "Mustrain;mx"}:
        return "microstrain"
    if family in {"MD", "SHorder", "SHmodel"} or family.startswith("C("):
        return "preferred_orientation"
    if family in {"BabA", "BabU", "Extinction"}:
        return "correction"
    if family == "Scale":
        return "phase_fraction" if source_group == "hap" else "histogram_scale"
    if source_group == "histogram":
        return "histogram_or_instrument"
    return source_group


def _flatten_frozen(controls: dict[str, Any]) -> set[str]:
    frozen: set[str] = set()
    payload = controls.get("parmFrozen", {})
    if not isinstance(payload, dict):
        return frozen
    for values in payload.values():
        if not isinstance(values, (list, tuple, set)):
            continue
        frozen.update(str(value) for value in values)
    return frozen


def _resolved_limit(g2obj: ModuleType, name: str, values: Any) -> dict[str, Any] | None:
    if not isinstance(values, dict) or not values:
        return None
    try:
        matched, value = g2obj.prmLookup(name, values)
    except Exception:
        return None
    number = _finite_float(value)
    if matched is None or number is None:
        return None
    return {"rule": str(matched), "value": number}


def _variable_description(g2obj: ModuleType, name: str) -> tuple[str, str]:
    try:
        location, meaning = g2obj.VarDescr(name)
        return str(location or ""), str(meaning or "")
    except Exception as exc:
        return "", f"Description unavailable: {type(exc).__name__}: {exc}"


def summarize_covariance(
    names: Iterable[Any],
    matrix: Any,
    *,
    descriptions: dict[str, tuple[str, str]] | None = None,
    max_correlations: int = 50,
) -> dict[str, Any]:
    variable_names = [str(name) for name in names]
    try:
        covariance = np.asarray(matrix, dtype=float)
    except Exception:
        covariance = np.empty((0, 0), dtype=float)
    expected = (len(variable_names), len(variable_names))
    if covariance.shape != expected or not variable_names:
        return {
            "available": False,
            "matrix_shape": list(covariance.shape),
            "variable_count": len(variable_names),
            "top_correlations": [],
        }

    diagonal = np.diag(covariance)
    sigmas = np.sqrt(np.clip(diagonal, 0.0, None))
    denominator = np.outer(sigmas, sigmas)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.divide(
            covariance,
            denominator,
            out=np.zeros_like(covariance),
            where=denominator > 0,
        )
    pairs: list[dict[str, Any]] = []
    for left in range(len(variable_names)):
        for right in range(left + 1, len(variable_names)):
            value = _finite_float(correlation[left, right])
            if value is None:
                continue
            left_name = variable_names[left]
            right_name = variable_names[right]
            entry = {
                "left": left_name,
                "right": right_name,
                "correlation": value,
                "absolute_correlation": abs(value),
            }
            if descriptions:
                entry["left_description"] = list(descriptions.get(left_name, ("", "")))
                entry["right_description"] = list(descriptions.get(right_name, ("", "")))
            pairs.append(entry)
    pairs.sort(key=lambda item: float(item["absolute_correlation"]), reverse=True)

    try:
        rank = int(np.linalg.matrix_rank(correlation))
    except Exception:
        rank = None
    try:
        condition = float(np.linalg.cond(correlation))
        if not math.isfinite(condition):
            condition = None
    except Exception:
        condition = None
    summary: dict[str, Any] = {
        "available": True,
        "matrix_shape": list(covariance.shape),
        "variable_count": len(variable_names),
        "rank": rank,
        "rank_basis": "correlation_matrix",
        "condition_number": condition,
        "high_correlation_count_0_9": sum(
            1 for item in pairs if float(item["absolute_correlation"]) >= 0.9
        ),
        "top_correlations": pairs[: max(0, int(max_correlations))],
    }
    if len(variable_names) <= 128:
        summary["variable_names"] = variable_names
        summary["matrix"] = covariance.tolist()
        summary["correlation_matrix"] = correlation.tolist()
    else:
        summary["variable_names_omitted"] = (
            "Covariance variable names exceed the 128-variable inline limit."
        )
        summary["matrix_omitted"] = "Covariance matrix exceeds the 128-variable inline limit."
    return summary


def summarize_shift_over_sigma(covariance_data: dict[str, Any] | None) -> dict[str, Any]:
    """Separate GSAS-II's total-run displacement from its final-cycle shift.

    ``Rvals['Max shft/sig']`` is computed from the difference between the
    starting and final parameter vectors over every cycle in a refinement.
    It is trajectory evidence, not a convergence test. For analytic-Hessian
    refinements GSAS-II also persists ``Rvals.lastShifts``; dividing those
    final-cycle shifts by the covariance sigmas gives the convergence value.
    """
    payload = covariance_data if isinstance(covariance_data, dict) else {}
    raw_names = payload.get("varyList")
    raw_sigmas = payload.get("sig")
    names = [str(value) for value in raw_names] if raw_names is not None else []
    sigmas = list(raw_sigmas) if raw_sigmas is not None else []
    r_values = payload.get("Rvals") if isinstance(payload.get("Rvals"), dict) else {}

    raw_total_values = payload.get("Lastshft")
    total_values = list(raw_total_values) if raw_total_values is not None else []
    total_by_name = {
        name: number
        for name, value in zip(names, total_values)
        if (number := _finite_float(value)) is not None
    }
    raw_final = r_values.get("lastShifts")
    if isinstance(raw_final, dict):
        final_by_name = {
            str(name): number
            for name, value in raw_final.items()
            if (number := _finite_float(value)) is not None
        }
    elif isinstance(raw_final, (list, tuple)):
        final_by_name = {
            name: number
            for name, value in zip(names, raw_final)
            if (number := _finite_float(value)) is not None
        }
    else:
        final_by_name = {}

    sigma_by_name = {
        name: number
        for name, value in zip(names, sigmas)
        if (number := _finite_float(value)) not in (None, 0.0)
    }

    def ratios(shifts: dict[str, float]) -> dict[str, float]:
        return {
            name: shift / sigma_by_name[name]
            for name, shift in shifts.items()
            if name in sigma_by_name
        }

    total_ratios = ratios(total_by_name)
    final_ratios = ratios(final_by_name)

    def maximum(values: dict[str, float]) -> dict[str, Any] | None:
        if not values:
            return None
        name, value = max(values.items(), key=lambda item: abs(item[1]))
        return {"parameter": name, "value": value, "absolute_value": abs(value)}

    reported_total = _finite_float(r_values.get("Max shft/sig"))
    total_maximum = maximum(total_ratios)
    final_maximum = maximum(final_ratios)
    return {
        "available": bool(total_ratios or final_ratios or reported_total is not None),
        "total_run": {
            "reported_maximum": reported_total,
            "computed_maximum": total_maximum,
            "ratios": total_ratios,
        },
        "last_cycle": {
            "available": bool(final_ratios),
            "maximum": final_maximum,
            "ratios": final_ratios,
        },
        "convergence_value": (
            final_maximum["absolute_value"]
            if final_maximum is not None
            else abs(reported_total)
            if reported_total is not None
            else total_maximum["absolute_value"]
            if total_maximum is not None
            else None
        ),
        "convergence_basis": (
            "last_cycle" if final_maximum is not None else "total_run_fallback"
        ),
    }


def current_refinement_variable(name, historical_aliases, configured_now):
    """A past sequential shift alias cannot mask a newly enabled direct variable.

    GPX covariance newCellDict/newAtomDict describe the preceding fit. Current
    native configured names take precedence when inspecting a new recipe stage.
    """
    if name in configured_now:return name
    # Before the first coordinate refinement, there is no newAtomDict history.
    # The native API refines dAx/dAy/dAz while storing physical Ax/Ay/Az. Bind
    # only an exact currently configured same-atom shift; never infer a hold or
    # a different atom from a historical covariance entry.
    fields=name.split(':')
    if len(fields)==4 and fields[2] in {'Ax','Ay','Az'}:
        shifted=':'.join([*fields[:2],'d'+fields[2],fields[3]])
        if shifted in configured_now:return shifted
    return historical_aliases.get(name,name)


def _reconstruct_parameter_space(
    gpx_path: Path,
    gpx: Any,
    g2scriptable: ModuleType,
) -> dict[str, Any]:
    g2io = _import_gsasii_internal(g2scriptable, "GSASIIstrIO")
    g2map = _import_gsasii_internal(g2scriptable, "GSASIImapvars")
    g2obj = _import_gsasii_internal(g2scriptable, "GSASIIobj")

    g2map.InitVars()
    (
        controls,
        histograms,
        phases,
        constraint_records,
        fixed_variables,
        rigid_bodies,
    ) = _load_gsasii_model_inputs(g2io, gpx_path)
    rb_ids = rigid_bodies.get("RBIds", {"Vector": [], "Residue": [], "Spin": []})
    rb_result = g2io.GetRigidBodyModels(rigid_bodies, Print=False)
    if rb_result is None:
        rb_vary, rb_values = [], {}
    else:
        rb_vary, rb_values = rb_result

    phase_result = g2io.GetPhaseData(phases, {}, rb_ids, Print=False)
    (
        atom_count,
        atom_index,
        phase_vary,
        phase_values,
        _,
        ff_tables,
        ef_tables,
        orbital_tables,
        neutron_tables,
        magnetic_tables,
        max_ss_wave,
    ) = phase_result
    calculation_controls = dict(controls)
    calculation_controls.update(
        {
            "atomIndx": atom_index,
            "Natoms": atom_count,
            "FFtables": ff_tables,
            "EFtables": ef_tables,
            "ORBtables": orbital_tables,
            "BLtables": neutron_tables,
            "MFtables": magnetic_tables,
            "maxSSwave": max_ss_wave,
        }
    )
    try:
        hap_vary, hap_values, hap_controls = g2io.GetHistogramPhaseData(
            phases,
            histograms,
            Controls=calculation_controls,
            Print=False,
            resetRefList=False,
        )
    except TypeError:
        hap_vary, hap_values, hap_controls = g2io.GetHistogramPhaseData(
            phases,
            histograms,
            Controls=calculation_controls,
            Print=False,
        )
    calculation_controls.update(hap_controls)
    hist_vary, hist_values, _, hist_controls = g2io.GetHistogramData(histograms, Print=False)
    calculation_controls.update(hist_controls)

    source_values = (
        ("rigid_body", rb_values),
        ("phase", phase_values),
        ("hap", hap_values),
        ("histogram", hist_values),
    )
    parameter_values: dict[str, Any] = {}
    parameter_sources: dict[str, str] = {}
    for source, values in source_values:
        for name, value in values.items():
            text_name = str(name)
            parameter_values[text_name] = value
            parameter_sources[text_name] = source

    configured_vary = [str(value) for value in rb_vary + phase_vary + hap_vary + hist_vary]
    effective_vary_objects = list(rb_vary + phase_vary + hap_vary + hist_vary)
    constraint_records = list(constraint_records)
    fixed_variables = list(fixed_variables)
    if hasattr(g2io, "makeTwinFrConstr"):
        twin_constraints, twin_fixed = g2io.makeTwinFrConstr(phases, histograms, hap_vary)
        constraint_records += list(twin_constraints)
        fixed_variables += list(twin_fixed)

    constraint_values = copy.deepcopy(parameter_values)
    multiplier_error = g2map.EvaluateMultipliers(constraint_records, constraint_values)
    constraint_error = ""
    constraint_warning = ""
    groups: list[Any] = []
    parameter_lists: list[Any] = []
    if not multiplier_error:
        constraint_error, constraint_warning, groups, parameter_lists = g2map.GenerateConstraints(
            effective_vary_objects,
            constraint_records,
            fixed_variables,
            constraint_values,
        )
        if not constraint_error:
            g2map.normParms(constraint_values)
            g2map.Map2Dict(constraint_values, effective_vary_objects)
    effective_vary = [str(value) for value in effective_vary_objects]
    try:
        mapping_text = str(g2map.VarRemapShow([], True) or "")
    except Exception:
        mapping_text = ""

    try:
        covariance_data = gpx["Covariance"]["data"]
    except Exception:
        covariance_data = {}
    last_vary = [str(value) for value in covariance_data.get("varyList", [])]
    configured_last = [str(value) for value in covariance_data.get("varyListStart", last_vary)]
    covariance_values = list(covariance_data.get("variables", []))
    covariance_sigmas = list(covariance_data.get("sig", []))
    last_values = dict(zip(last_vary, covariance_values))
    sigma_by_name = {
        name: float(value)
        for name, value in zip(last_vary, covariance_sigmas)
        if _finite_float(value) is not None
    }
    dependent_values: dict[str, Any] = {}
    dependent_sigmas: dict[str, float] = {}
    for name, payload in (covariance_data.get("depSigDict", {}) or {}).items():
        if not isinstance(payload, (list, tuple)) or len(payload) < 2:
            continue
        text_name = str(name)
        dependent_values[text_name] = payload[0]
        sigma = _finite_float(payload[1])
        if sigma is not None:
            dependent_sigmas[text_name] = sigma
    last_values.update(dependent_values)
    sigma_by_name.update(dependent_sigmas)

    actual_source_variables: dict[str, str] = {}
    for mapping_name in ("newAtomDict", "newCellDict"):
        for shift_name, payload in (covariance_data.get(mapping_name, {}) or {}).items():
            if not isinstance(payload, (list, tuple)) or len(payload) < 2:
                continue
            actual_name = str(payload[0])
            actual_source_variables[actual_name] = str(shift_name)
            last_values[actual_name] = payload[1]
            sigma = sigma_by_name.get(str(shift_name))
            if sigma is not None:
                sigma_by_name[actual_name] = sigma

    for name, value in last_values.items():
        if name not in parameter_values:
            parameter_values[name] = value
            parameter_sources[name] = "derived"

    shift_summary = summarize_shift_over_sigma(covariance_data)
    raw_total_shifts = covariance_data.get("Lastshft")
    total_shifts = list(raw_total_shifts) if raw_total_shifts is not None else []
    total_shift_by_name = {
        name: float(value)
        for name, value in zip(last_vary, total_shifts)
        if _finite_float(value) is not None
    }
    r_values = covariance_data.get("Rvals") if isinstance(covariance_data.get("Rvals"), dict) else {}
    raw_final_shifts = r_values.get("lastShifts")
    final_shift_by_name = (
        {
            str(name): float(value)
            for name, value in raw_final_shifts.items()
            if _finite_float(value) is not None
        }
        if isinstance(raw_final_shifts, dict)
        else {}
    )
    frozen = _flatten_frozen(controls)
    configured_now_set = set(configured_vary)
    effective_now_set = set(effective_vary)
    configured_last_set = set(configured_last)
    last_vary_set = set(last_vary)
    descriptions = {
        name: _variable_description(g2obj, name) for name in parameter_values
    }

    rows: list[dict[str, Any]] = []
    for name, current_value in parameter_values.items():
        refinement_variable = current_refinement_variable(name,actual_source_variables,configured_now_set)
        phase_id, histogram_id, family, atom_id = _split_variable(name)
        source_group = parameter_sources.get(name, "unknown")
        configured_now = refinement_variable in configured_now_set
        effective_now = refinement_variable in effective_now_set
        configured_in_last_cycle = refinement_variable in configured_last_set
        if refinement_variable in last_vary_set:
            status = "refined_independent"
        elif configured_in_last_cycle:
            status = "refined_dependent"
        elif name in dependent_values:
            status = "derived"
        elif configured_now and not effective_now:
            status = "configured_dependent"
        elif configured_now:
            status = "configured"
        else:
            status = "fixed"
        sigma = sigma_by_name.get(name, sigma_by_name.get(refinement_variable))
        value_number = _finite_float(current_value)
        relative_sigma = None
        if sigma is not None and value_number not in (None, 0.0):
            relative_sigma = abs(sigma / value_number)
        total_shift = total_shift_by_name.get(refinement_variable)
        total_shift_over_sigma = None
        if total_shift is not None and sigma not in (None, 0.0):
            total_shift_over_sigma = total_shift / sigma
        final_shift = final_shift_by_name.get(refinement_variable)
        final_shift_over_sigma = None
        if final_shift is not None and sigma not in (None, 0.0):
            final_shift_over_sigma = final_shift / sigma
        lower = _resolved_limit(g2obj, name, controls.get("parmMinDict"))
        upper = _resolved_limit(g2obj, name, controls.get("parmMaxDict"))
        location, meaning = descriptions[name]
        rows.append(
            {
                "name": name,
                "refinement_variable": refinement_variable,
                "family": family,
                "group": _parameter_group(name, source_group),
                "source_group": source_group,
                "phase_id": phase_id,
                "histogram_id": histogram_id,
                "atom_id": atom_id,
                "location": location,
                "meaning": meaning,
                "current_value": _jsonable(current_value),
                "last_cycle_value": _jsonable(last_values.get(name)),
                "sigma": sigma,
                "relative_sigma": relative_sigma,
                "last_shift": final_shift if final_shift is not None else total_shift,
                "last_shift_over_sigma": (
                    final_shift_over_sigma
                    if final_shift_over_sigma is not None
                    else total_shift_over_sigma
                ),
                "last_cycle_shift": final_shift,
                "last_cycle_shift_over_sigma": final_shift_over_sigma,
                "total_run_shift": total_shift,
                "total_run_shift_over_sigma": total_shift_over_sigma,
                "status": status,
                "configured_now": configured_now,
                "effective_now": effective_now,
                "configured_in_last_cycle": configured_in_last_cycle,
                "refined_in_last_cycle": refinement_variable in last_vary_set,
                "frozen_at_limit": refinement_variable in frozen,
                "lower_bound": lower,
                "upper_bound": upper,
            }
        )

    return {
        "controls": _jsonable(controls),
        "constraints": {
            "records": _jsonable(constraint_records),
            "fixed_variables": [str(value) for value in fixed_variables],
            "multiplier_error": str(multiplier_error or ""),
            "error": str(constraint_error or ""),
            "warning": str(constraint_warning or ""),
            "groups": _jsonable(groups),
            "parameter_lists": _jsonable(parameter_lists),
            "mapping_text": mapping_text,
        },
        "parameter_values": parameter_values,
        "parameter_descriptions": descriptions,
        "parameter_space": {
            "parameter_count": len(rows),
            "configured_vary_count": len(configured_vary),
            "effective_vary_count": len(effective_vary),
            "last_vary_count": len(last_vary),
            "configured_vary": configured_vary,
            "effective_vary": effective_vary,
            "configured_last_cycle": configured_last,
            "last_vary": last_vary,
            "frozen_variables": sorted(frozen),
            "parameters": rows,
        },
        "covariance_data": covariance_data,
        "shift_over_sigma": shift_summary,
    }


def extract_gsasii_refinement_state(
    gpx_path: Path | str,
    gpx: Any,
    g2scriptable: ModuleType,
    *,
    max_correlations: int = 50,
) -> dict[str, Any]:
    """Extract a read-only, normalized refinement state from a GSAS-II project.

    GSAS-II does not always persist ``Covariance.data.parmDict``. This adapter
    therefore reconstructs the parameter dictionary with the same
    ``GSASIIstrIO`` loaders used by ``GSASIIstrMain.Refine``. Internal APIs are
    isolated here so generated scripts never need to traverse the GPX tree.
    """

    path = Path(gpx_path).resolve()
    before_hash = _sha256(path)
    warnings: list[str] = []
    public_parameter_api = "available"
    try:
        gpx.get_ParmList()
    except Exception as exc:
        public_parameter_api = "missing_parmDict"
        warnings.append(
            "G2Project.get_ParmList/get_Variable are unavailable for this checkpoint: "
            f"{type(exc).__name__}: {exc}. Parameter values were reconstructed with GSASIIstrIO."
        )

    reconstructed: dict[str, Any] | None = None
    with _GSASII_STATE_LOCK:
        try:
            reconstructed = _reconstruct_parameter_space(path, gpx, g2scriptable)
        except Exception as exc:
            warnings.append(
                "Full parameter reconstruction failed: "
                f"{type(exc).__name__}: {exc}. Covariance-only state is retained."
            )

    covariance_data: dict[str, Any]
    try:
        covariance_data = gpx["Covariance"]["data"]
    except Exception:
        covariance_data = {}
    descriptions = reconstructed.get("parameter_descriptions", {}) if reconstructed else {}
    covariance = summarize_covariance(
        covariance_data.get("varyList", []),
        covariance_data.get("covMatrix", []),
        descriptions=descriptions,
        max_correlations=max_correlations,
    )
    fit_statistics = dict(_jsonable(covariance_data.get("Rvals", {}) or {}))
    histogram_residuals: list[dict[str, Any]] = []
    try:
        phases = list(gpx.phases())
        for histogram in gpx.histograms():
            linked_use_flags: list[bool] = []
            for phase in phases:
                hap = (phase.data.get("Histograms") or {}).get(histogram.name)
                if isinstance(hap, dict):
                    linked_use_flags.append(bool(hap.get("Use", True)))
            histogram_residuals.append(
                {
                    "name": str(histogram.name),
                    "active": (
                        any(linked_use_flags) if linked_use_flags else True
                    ),
                    "residuals": _jsonable(dict(histogram.residuals or {})),
                }
            )
    except Exception as exc:
        warnings.append(
            "Histogram residual extraction failed: "
            f"{type(exc).__name__}: {exc}. Global fit statistics are retained."
        )
    shift_summary = summarize_shift_over_sigma(covariance_data)
    convergence_value = _finite_float(shift_summary.get("convergence_value"))
    reported_total = _finite_float(
        (shift_summary.get("total_run") or {}).get("reported_maximum")
    )
    final_maximum = (shift_summary.get("last_cycle") or {}).get("maximum")
    final_value = _finite_float(
        final_maximum.get("absolute_value") if isinstance(final_maximum, dict) else None
    )
    if reported_total is not None:
        fit_statistics["total_run_max_shift_over_sigma"] = abs(reported_total)
    if final_value is not None:
        fit_statistics["last_cycle_max_shift_over_sigma"] = final_value
    if convergence_value is not None:
        fit_statistics["max_shift_over_sigma"] = convergence_value
        fit_statistics["max_shift_over_sigma_basis"] = shift_summary["convergence_basis"]
    covariance["shift_over_sigma"] = _jsonable(shift_summary)
    state: dict[str, Any] = {
        "schema_version": STATE_SCHEMA_VERSION,
        "source": {
            "gpx_path": str(path),
            "sha256": before_hash,
            "saved_with": _jsonable(
                (reconstructed or {}).get("controls", {}).get("LastSavedUsing")
                if reconstructed
                else None
            ),
        },
        "fit_statistics": fit_statistics,
        "histogram_residuals": histogram_residuals,
        "covariance": covariance,
        "controls": (reconstructed or {}).get("controls", {}),
        "constraints": (reconstructed or {}).get("constraints", {}),
        "parameter_space": (reconstructed or {}).get(
            "parameter_space",
            {
                "parameter_count": 0,
                "configured_vary_count": 0,
                "effective_vary_count": len(covariance_data.get("varyList", []) or []),
                "last_vary_count": len(covariance_data.get("varyList", []) or []),
                "configured_vary": [],
                "effective_vary": [str(value) for value in covariance_data.get("varyList", [])],
                "configured_last_cycle": [
                    str(value) for value in covariance_data.get("varyListStart", [])
                ],
                "last_vary": [str(value) for value in covariance_data.get("varyList", [])],
                "frozen_variables": [],
                "parameters": [],
            },
        ),
        "extraction": {
            "status": "degraded" if reconstructed is None else "complete",
            "public_parameter_api": public_parameter_api,
            "parameter_reconstruction": "GSASIIstrIO" if reconstructed else "unavailable",
            "read_only_hash_verified": before_hash == _sha256(path),
            "warnings": warnings,
        },
    }
    if not state["extraction"]["read_only_hash_verified"]:
        state["extraction"]["status"] = "invalid"
        state["extraction"]["warnings"].append(
            "The GPX hash changed during read-only extraction; do not trust this snapshot."
        )
    return state


def refinement_state_metrics(state: dict[str, Any] | None) -> dict[str, Any]:
    """Flatten authoritative checkpoint convergence evidence into run metrics."""
    state = state if isinstance(state, dict) else {}
    fit = state.get("fit_statistics")
    fit = fit if isinstance(fit, dict) else {}
    covariance = state.get("covariance")
    covariance = covariance if isinstance(covariance, dict) else {}
    metrics: dict[str, Any] = {}

    numeric_aliases = {
        "Rwp": ("Rwp", "wR"),
        "GOF": ("GOF",),
        "chi2": ("chi2", "chisq"),
        "observation_count": ("observation_count", "Nobs"),
        "parameter_count": ("parameter_count", "Nvars"),
        "cycle_count": ("cycle_count", "cycles"),
    }
    for target, aliases in numeric_aliases.items():
        for alias in aliases:
            value = _finite_float(fit.get(alias))
            if value is None:
                continue
            metrics[target] = int(value) if target.endswith("_count") else value
            break

    histogram_rows = state.get("histogram_residuals")
    histogram_rows = histogram_rows if isinstance(histogram_rows, list) else []
    explicitly_active_rows = [
        row
        for row in histogram_rows
        if isinstance(row, dict) and row.get("active") is True
    ]
    metric_histogram_rows = explicitly_active_rows or histogram_rows
    histogram_rwp: list[tuple[str, float]] = []
    for index, row in enumerate(metric_histogram_rows):
        if not isinstance(row, dict):
            continue
        residuals = row.get("residuals")
        residuals = residuals if isinstance(residuals, dict) else {}
        value = _finite_float(residuals.get("wR", residuals.get("Rwp")))
        if value is None:
            continue
        name = str(row.get("name") or row.get("histogram") or f"histogram_{index + 1}")
        histogram_rwp.append((name, value))

    if histogram_rows:
        metrics["histogram_count"] = len(metric_histogram_rows)
        metrics["histogram_total_count"] = len(histogram_rows)
        metrics["inactive_histogram_count"] = (
            len(histogram_rows) - len(metric_histogram_rows)
        )
        metrics["Rwp_histogram_count"] = len(histogram_rwp)
        metrics["Rwp_histogram_coverage_complete"] = (
            len(histogram_rwp) == len(metric_histogram_rows)
        )
    if histogram_rwp:
        metrics["Rwp_by_histogram"] = dict(histogram_rwp)
    if len(metric_histogram_rows) == 1 and len(histogram_rwp) == 1:
        metrics["Rwp"] = histogram_rwp[0][1]
        metrics["wR"] = histogram_rwp[0][1]
        metrics["Rwp_metric_source"] = "single_histogram.residuals.wR"
    elif len(metric_histogram_rows) > 1 and histogram_rwp:
        # A multi-bank GPX does not expose a trustworthy joint scalar Rwp here.
        # Use the worst current bank as the conservative progress signal so an
        # excellent final/selected bank cannot hide a failed companion bank.
        worst_name, worst_rwp = max(histogram_rwp, key=lambda item: item[1])
        best_name, best_rwp = min(histogram_rwp, key=lambda item: item[1])
        metrics["Rwp"] = worst_rwp
        metrics["wR"] = worst_rwp
        metrics["Rwp_metric_source"] = "worst_histogram.residuals.wR"
        metrics["Rwp_worst_histogram"] = worst_name
        metrics["Rwp_best"] = best_rwp
        metrics["Rwp_best_histogram"] = best_name
        metrics["Rwp_mean_across_histograms"] = sum(
            value for _, value in histogram_rwp
        ) / len(histogram_rwp)

    converged = fit.get("converged")
    if isinstance(converged, bool):
        metrics["converged"] = converged

    last_cycle_shift = _finite_float(fit.get("last_cycle_max_shift_over_sigma"))
    convergence_shift = _finite_float(fit.get("max_shift_over_sigma"))
    total_run_shift = _finite_float(fit.get("total_run_max_shift_over_sigma"))
    if last_cycle_shift is not None:
        metrics["last_cycle_max_shift_over_sigma"] = abs(last_cycle_shift)
    if convergence_shift is None:
        convergence_shift = last_cycle_shift
    if convergence_shift is not None:
        convergence_shift = abs(convergence_shift)
        metrics["max_shift_over_sigma"] = convergence_shift
        metrics["max_shift_esd"] = convergence_shift
    if total_run_shift is not None:
        metrics["total_run_max_shift_over_sigma"] = abs(total_run_shift)
    basis = str(fit.get("max_shift_over_sigma_basis") or "").strip()
    if basis:
        metrics["max_shift_over_sigma_basis"] = basis

    rank = covariance.get("rank")
    variable_count = covariance.get("variable_count")
    if isinstance(rank, int) and not isinstance(rank, bool):
        metrics["covariance_rank"] = rank
    if isinstance(variable_count, int) and not isinstance(variable_count, bool):
        metrics["covariance_variable_count"] = variable_count
    condition_number = _finite_float(covariance.get("condition_number"))
    if condition_number is not None:
        metrics["covariance_condition_number"] = condition_number
    high_correlation_count = covariance.get("high_correlation_count_0_9")
    if isinstance(high_correlation_count, int) and not isinstance(high_correlation_count, bool):
        metrics["covariance_high_correlation_count_0_9"] = high_correlation_count
    top_absolute_correlation = max(
        (
            abs(value)
            for item in covariance.get("top_correlations", []) or []
            if isinstance(item, dict)
            and (
                value := _finite_float(
                    item.get("absolute_correlation", item.get("correlation"))
                )
            )
            is not None
        ),
        default=None,
    )
    if top_absolute_correlation is not None:
        metrics["covariance_top_absolute_correlation"] = top_absolute_correlation
        metrics["correlation_recovery_required"] = top_absolute_correlation >= 0.95
    if covariance.get("available") is True and isinstance(rank, int) and isinstance(variable_count, int):
        metrics["rank_deficient"] = rank < variable_count
    return metrics


def refinement_state_overview(state: dict[str, Any] | None, *, max_parameters: int = 120) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    parameter_space = state.get("parameter_space")
    parameter_space = parameter_space if isinstance(parameter_space, dict) else {}
    parameters = parameter_space.get("parameters")
    parameters = parameters if isinstance(parameters, list) else []
    active = [
        row
        for row in parameters
        if isinstance(row, dict)
        and (
            row.get("configured_now")
            or row.get("configured_in_last_cycle")
            or row.get("frozen_at_limit")
            or row.get("status") == "derived"
        )
    ]
    compact_space = {
        key: _jsonable(value)
        for key, value in parameter_space.items()
        if key != "parameters"
    }
    compact_space["active_parameters"] = _jsonable(active[: max(0, int(max_parameters))])
    compact_space["active_parameter_count"] = len(active)
    if len(active) > max_parameters:
        compact_space["active_parameters_truncated"] = len(active) - max_parameters

    covariance = state.get("covariance")
    covariance = covariance if isinstance(covariance, dict) else {}
    compact_covariance = {
        key: _jsonable(value)
        for key, value in covariance.items()
        if key not in {"matrix", "correlation_matrix"}
    }
    return {
        "schema_version": str(state.get("schema_version") or STATE_SCHEMA_VERSION),
        "source": _jsonable(state.get("source") or {}),
        "fit_statistics": _jsonable(state.get("fit_statistics") or {}),
        "histogram_residuals": _jsonable(state.get("histogram_residuals") or []),
        "controls": _jsonable(state.get("controls") or {}),
        "constraints": _jsonable(state.get("constraints") or {}),
        "parameter_space": compact_space,
        "covariance": compact_covariance,
        "extraction": _jsonable(state.get("extraction") or {}),
    }


def diff_parameter_states(
    previous_state: dict[str, Any] | None,
    current_state: dict[str, Any] | None,
    *,
    max_changes: int = 100,
) -> dict[str, Any]:
    def rows(state: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
        if not isinstance(state, dict):
            return {}
        parameter_space = state.get("parameter_space")
        if not isinstance(parameter_space, dict):
            return {}
        values = parameter_space.get("parameters")
        if not isinstance(values, list):
            values = parameter_space.get("active_parameters")
        if not isinstance(values, list):
            return {}
        return {
            str(row.get("name")): row
            for row in values
            if isinstance(row, dict) and str(row.get("name") or "")
        }

    previous_rows = rows(previous_state)
    current_rows = rows(current_state)
    changes: list[dict[str, Any]] = []
    status_changes: list[dict[str, Any]] = []
    changed_groups: set[str] = set()
    value_changed_groups: set[str] = set()
    status_changed_groups: set[str] = set()
    activated_groups: set[str] = set()
    deactivated_groups: set[str] = set()
    for name, current in current_rows.items():
        previous = previous_rows.get(name)
        if previous is None:
            continue
        previous_value = _finite_float(previous.get("current_value"))
        current_value = _finite_float(current.get("current_value"))
        if previous_value is not None and current_value is not None:
            delta = current_value - previous_value
            tolerance = max(1e-12, 1e-12 * max(abs(previous_value), abs(current_value), 1.0))
            if abs(delta) > tolerance:
                previous_sigma = _finite_float(previous.get("sigma")) or 0.0
                current_sigma = _finite_float(current.get("sigma")) or 0.0
                combined_sigma = math.hypot(previous_sigma, current_sigma)
                significance = delta / combined_sigma if combined_sigma > 0 else None
                relative_delta = delta / abs(previous_value) if previous_value != 0 else None
                group = str(current.get("group") or "unknown")
                changed_groups.add(group)
                value_changed_groups.add(group)
                changes.append(
                    {
                        "name": name,
                        "group": group,
                        "location": str(current.get("location") or ""),
                        "meaning": str(current.get("meaning") or ""),
                        "previous": previous_value,
                        "current": current_value,
                        "delta": delta,
                        "relative_delta": relative_delta,
                        "change_over_combined_sigma": significance,
                    }
                )
        previous_status = (
            str(previous.get("status") or ""),
            bool(previous.get("configured_now")),
            bool(previous.get("effective_now")),
            bool(previous.get("frozen_at_limit")),
        )
        current_status = (
            str(current.get("status") or ""),
            bool(current.get("configured_now")),
            bool(current.get("effective_now")),
            bool(current.get("frozen_at_limit")),
        )
        if previous_status != current_status:
            group = str(current.get("group") or "unknown")
            changed_groups.add(group)
            status_changed_groups.add(group)
            previous_active = previous_status[1] or previous_status[2]
            current_active = current_status[1] or current_status[2]
            if current_active and not previous_active:
                activated_groups.add(group)
            elif previous_active and not current_active:
                deactivated_groups.add(group)
            status_changes.append(
                {
                    "name": name,
                    "group": group,
                    "previous_status": previous_status[0],
                    "current_status": current_status[0],
                    "previous_configured": previous_status[1],
                    "current_configured": current_status[1],
                    "previous_effective": previous_status[2],
                    "current_effective": current_status[2],
                    "previous_frozen": previous_status[3],
                    "current_frozen": current_status[3],
                }
            )

    def score(change: dict[str, Any]) -> tuple[int, float]:
        significance = _finite_float(change.get("change_over_combined_sigma"))
        if significance is not None:
            return 1, abs(significance)
        relative = _finite_float(change.get("relative_delta"))
        return 0, abs(relative or _finite_float(change.get("delta")) or 0.0)

    changes.sort(key=score, reverse=True)
    return {
        "parameter_change_count": len(changes),
        "status_change_count": len(status_changes),
        "changed_parameter_groups": sorted(changed_groups),
        "value_changed_parameter_groups": sorted(value_changed_groups),
        "status_changed_parameter_groups": sorted(status_changed_groups),
        "activated_parameter_groups": sorted(activated_groups),
        "deactivated_parameter_groups": sorted(deactivated_groups),
        "parameter_changes": changes[: max(0, int(max_changes))],
        "status_changes": status_changes[: max(0, int(max_changes))],
        "changes_truncated": max(0, len(changes) - max_changes),
    }
