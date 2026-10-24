"""Versioned, declarative post-analysis recipes over normalized GSAS-II results.

No scientific refinement, file loading, or Galaxy submission occurs here.
Presentation changes can therefore be replayed against saved observations.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from .facility_metadata import parse_nexus_timestamp


RECIPES = {
    "phase_evolution": ("Phase evolution", "weight_percent", "absolute"),
    "thermal_expansion": ("Thermal expansion / cell change", "a", "relative"),
    "distortion": ("Lattice shape / distortion", "c/a", "absolute"),
    "lattice_strain": ("Reference lattice strain", "a", "microstrain"),
    "transition_hysteresis": ("Transition / hysteresis", "weight_percent", "absolute"),
    "reaction_progress": ("Reaction progress", "weight_percent", "progress"),
    "operando": ("Operando correlation", "a", "relative"),
    "sample_composition": ("Sample / composition / preparation", "a", "absolute"),
    "broadening": ("Coherent-domain size / microstrain", "", "absolute"),
    "atomic_structure": ("Atomic geometry / site order", "", "absolute"),
}
Kind = Literal[
    "phase_evolution",
    "thermal_expansion",
    "distortion",
    "lattice_strain",
    "transition_hysteresis",
    "reaction_progress",
    "operando",
    "sample_composition",
    "broadening",
    "atomic_structure",
]
CELL_KEYS = ("a", "b", "c", "alpha", "beta", "gamma", "volume")
AXES = (
    "shared",
    "auto",
    "run_number",
    "start_time",
    "elapsed_time",
    "temperature",
    "magnetic_field",
    "pressure",
    "voltage",
    "capacity",
    "sample_state",
)


class ViewSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    kind: Kind
    metric: str = ""
    representation: Literal[
        "auto", "absolute", "difference", "relative", "microstrain", "progress"
    ] = "auto"
    axis: Literal[
        "shared",
        "auto",
        "run_number",
        "start_time",
        "elapsed_time",
        "temperature",
        "magnetic_field",
        "pressure",
        "voltage",
        "capacity",
        "sample_state",
    ] = "shared"
    phases: list[str] = Field(default_factory=list)
    samples: list[str] = Field(default_factory=list)
    banks: list[int] = Field(default_factory=list)
    branch: str = "all"
    reference_run: str | None = None
    connect: bool = False
    show_uncertainty: bool = True
    x_range: tuple[float | str, float | str] | None = None
    y_range: tuple[float, float] | None = None
    include_fixed: bool = False
    excluded_runs: list[str] = Field(default_factory=list)
    hkl: tuple[int, int, int] = (1, 0, 0)
    progress_start: float | None = None
    progress_end: float | None = None
    time_zero: str | None = None
    threshold: float | None = None
    fit_expansion: bool = False
    fit_min: float | None = None
    fit_max: float | None = None
    max_temperature_span: float = Field(default=2.0, ge=0)

    @model_validator(mode="after")
    def validate_definition(self):
        metric = self.metric or RECIPES[self.kind][1]
        if metric and metric not in {
            *CELL_KEYS,
            "weight_percent",
            "c/a",
            "b/a",
            "orthorhombicity",
            "d_hkl",
        }:
            raise ValueError("Unsupported post-analysis metric")
        if self.kind == "phase_evolution" and metric != "weight_percent":
            raise ValueError("Phase evolution requires weight_percent")
        if self.kind == "thermal_expansion" and metric not in CELL_KEYS:
            raise ValueError("Cell change requires a lattice parameter or volume")
        if self.kind == "distortion" and metric not in {"c/a", "b/a", "orthorhombicity"}:
            raise ValueError("Select an explicit lattice distortion definition")
        if self.kind == "lattice_strain" and metric not in {"a", "b", "c", "volume", "d_hkl"}:
            raise ValueError("Select a lattice length, volume, or d_hkl")
        representation = (
            self.representation if self.representation != "auto" else RECIPES[self.kind][2]
        )
        if representation == "progress" and self.kind != "reaction_progress":
            raise ValueError("Progress normalization requires the reaction-progress recipe")
        if self.kind == "reaction_progress" and representation != "progress":
            raise ValueError("Reaction progress requires explicit progress normalization")
        if representation == "microstrain" and self.kind != "lattice_strain":
            raise ValueError("Microstrain representation requires the reference-lattice recipe")
        if self.progress_start is not None and self.progress_end == self.progress_start:
            raise ValueError("Progress endpoints must be different")
        if self.fit_min is not None and self.fit_max is not None and self.fit_min >= self.fit_max:
            raise ValueError("Fit interval start must be below its end")
        if self.fit_expansion and (
            self.kind != "thermal_expansion" or metric not in {"a", "b", "c", "volume"}
        ):
            raise ValueError("Expansion fits require a lattice length or volume")
        if not any(self.hkl):
            raise ValueError("hkl cannot be 0 0 0")
        if any(bank < 1 for bank in self.banks):
            raise ValueError("Banks must be positive")
        return self


class AxisMapping(BaseModel):
    """An explicit reviewed sample/run mapping; never inferred from a title."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    value: float | str
    unit: str
    source: str = Field(min_length=1)
    kind: Literal["measured", "nominal_composition", "preparation_history", "user_supplied"]
    verified: bool = False
    branch: str | None = None
    cycle: str | None = None


class PostAnalysisSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1] = 1
    axis: Literal[
        "auto",
        "run_number",
        "start_time",
        "elapsed_time",
        "temperature",
        "magnetic_field",
        "pressure",
        "voltage",
        "capacity",
        "sample_state",
    ] = "auto"
    views: list[ViewSettings] = Field(
        default_factory=lambda: [
            ViewSettings(id="phases", kind="phase_evolution"),
            ViewSettings(id="cells", kind="thermal_expansion", representation="absolute"),
        ],
        max_length=30,
    )
    # Keys are sample IDs for sample_state, otherwise exact run IDs.
    axis_mappings: dict[str, dict[str, AxisMapping]] = Field(default_factory=dict)
    references: dict[str, str] = Field(default_factory=dict)
    phase_colors: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_views(self):
        if len({v.id for v in self.views}) != len(self.views):
            raise ValueError("Post-analysis view IDs must be unique")
        if any(
            key
            not in {
                "temperature",
                "magnetic_field",
                "pressure",
                "voltage",
                "capacity",
                "sample_state",
            }
            for key in self.axis_mappings
        ):
            raise ValueError("Unsupported mapped axis")
        if any(index < 0 or index >= 10 for index in self.phase_colors.values()):
            raise ValueError("Phase color indices must be between 0 and 9")
        return self


def recipe_options():
    return [{"title": value[0], "value": key} for key, value in RECIPES.items()]


def number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def metric_unit(metric, representation):
    if representation == "relative":
        return "%"
    if representation == "microstrain":
        return "microstrain"
    if representation == "progress":
        return "dimensionless progress proxy"
    if metric == "weight_percent":
        return "modeled-crystalline wt%" if representation == "absolute" else "percentage points"
    if metric == "volume":
        return "angstrom^3"
    if metric in {"alpha", "beta", "gamma"}:
        return "degree"
    if metric in {"a", "b", "c", "d_hkl"}:
        return "angstrom"
    return "dimensionless"


def timestamp(value):
    try:
        parsed = parse_nexus_timestamp(value)
        return parsed.timestamp() if parsed.tzinfo is not None else None
    except (ValueError, OverflowError, OSError):
        return None


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:24]


def _run_id(scan):
    return str(scan.get("run_id") or scan.get("run_name") or scan.get("run_number") or "unknown")


def _sample(scan):
    metadata = scan.get("metadata") or {}
    return str(metadata.get("sample_id") or "")


def _axis(scan, key, settings, origin):
    metadata = scan.get("metadata") or {}
    mapping_key = _sample(scan) if key == "sample_state" else _run_id(scan)
    mapped = settings.axis_mappings.get(key, {}).get(mapping_key)
    if mapped is not None:
        if not mapped.verified:
            return None
        value, unit, source, kind = mapped.value, mapped.unit, mapped.source, mapped.kind
    elif key == "run_number":
        return {
            "value": scan.get("run_number"),
            "unit": "",
            "source": "run record",
            "kind": "identifier",
        }
    elif key in {"start_time", "elapsed_time"}:
        raw = metadata.get("start_time")
        seconds = timestamp(raw)
        if seconds is None:
            return None
        value = (
            seconds
            if key == "start_time"
            else (seconds - origin) / 60 if origin is not None else None
        )
        return {
            "value": value,
            "unit": "UTC" if key == "start_time" else "min",
            "source": "acquisition start",
            "kind": "measured",
        }
    else:
        metric = metadata.get(key)
        if not isinstance(metric, dict) or number(metric.get("value")) is None:
            return None
        if (
            metric.get("status") in {"conflict", "invalid", "unknown"}
            or metric.get("qualified") is False
        ):
            return None
        value, unit = metric["value"], str(metric.get("unit") or "").strip()
        source, kind = str(metric.get("source") or "acquisition metadata"), str(
            metric.get("kind") or "measured"
        )
    if key == "temperature":
        value = number(value)
        if value is None:
            return None
        if unit.lower() in {"c", "°c", "degc", "celsius"}:
            value += 273.15
        elif unit.lower() not in {"k", "kelvin"}:
            return None
        if value <= 0:
            return None
        unit = "K"
    elif key not in {"sample_state"}:
        value = number(value)
        if value is None or not unit:
            return None
    return {
        "value": value,
        "unit": unit,
        "source": source,
        "kind": kind,
        "branch": mapped.branch if mapped is not None else None,
        "cycle": mapped.cycle if mapped is not None else None,
    }


def _d_spacing(values, hkl):
    a, b, c, alpha, beta, gamma = values[:6]
    ca, cb, cg = [math.cos(math.radians(v)) for v in (alpha, beta, gamma)]
    g = [
        [a * a, a * b * cg, a * c * cb],
        [a * b * cg, b * b, b * c * ca],
        [a * c * cb, b * c * ca, c * c],
    ]
    # Cofactors of the symmetric direct metric.
    u, v, w = g[0]
    _, x, y = g[1]
    _, _, z = g[2]
    cof = [
        [x * z - y * y, w * y - v * z, v * y - w * x],
        [w * y - v * z, u * z - w * w, v * w - u * y],
        [v * y - w * x, v * w - u * y, u * x - v * v],
    ]
    det = u * cof[0][0] + v * cof[0][1] + w * cof[0][2]
    if det <= 0:
        raise ValueError("Invalid cell metric")
    reciprocal = sum(hkl[i] * cof[i][j] * hkl[j] for i in range(3) for j in range(3)) / det
    if reciprocal <= 0:
        raise ValueError("Invalid reciprocal spacing")
    return 1 / math.sqrt(reciprocal)


def _metric(row, metric, hkl, include_fixed):
    """Return value/variance/state. Missing covariance is never diagonalized."""
    if metric == "weight_percent":
        state = row.get("fraction_state", "unknown")
        value = number(row.get(metric))
        if value is not None and not 0 <= value <= 100:
            return None, None, "invalid_fraction"
        return value, None, state
    values = [number(row.get(k)) for k in CELL_KEYS]
    if all(value is not None for value in values[:6]):
        if any(value <= 0 for value in values[:3]) or any(
            not 0 < value < 180 for value in values[3:6]
        ):
            return None, None, "invalid_cell"
        try:
            _d_spacing(values, (1, 0, 0))
        except ValueError:
            return None, None, "invalid_cell"
    dependencies = {
        "c/a": [0, 2],
        "b/a": [0, 1],
        "orthorhombicity": [0, 1],
        "d_hkl": list(range(6)),
    }
    indices = dependencies.get(metric, [CELL_KEYS.index(metric)] if metric in CELL_KEYS else [])
    if not indices or any(values[i] is None for i in indices):
        return None, None, "absent"
    states = row.get("metric_states") or {}
    usable = {"refined", "constraint-derived"}
    if not any(states.get(CELL_KEYS[i]) in usable for i in indices):
        if not include_fixed:
            return None, None, "fixed_or_unverified"
        state = "model-constrained"
    else:
        state = (
            "refined"
            if all(states.get(CELL_KEYS[i]) in usable | {"fixed"} for i in indices)
            else "unknown"
        )
        if state == "unknown" and not include_fixed:
            return None, None, state
    if row.get("covariance_fresh") is False:
        return None, None, "stale"

    def evaluate(v):
        if metric == "c/a":
            return v[2] / v[0]
        if metric == "b/a":
            return v[1] / v[0]
        if metric == "orthorhombicity":
            return (v[0] - v[1]) / (v[0] + v[1])
        if metric == "d_hkl":
            return _d_spacing(v, hkl)
        return v[indices[0]]

    try:
        value = evaluate(values)
        gradient = [0.0] * 7
        for i in indices:
            step = max(abs(values[i]) * 1e-6, 1e-7)
            plus, minus = list(values), list(values)
            plus[i] += step
            minus[i] -= step
            gradient[i] = (evaluate(plus) - evaluate(minus)) / (2 * step)
        jacobian = row.get("cell_jacobian")
        if (
            metric in dependencies
            and isinstance(jacobian, list)
            and len(jacobian) == 7
            and jacobian[0]
        ):
            width = len(jacobian[0])
            if all(
                isinstance(r, list) and len(r) == width and all(number(v) is not None for v in r)
                for r in jacobian
            ):
                responses = [
                    sum(gradient[i] * jacobian[i][j] for i in range(7)) for j in range(width)
                ]
                scale = max(
                    1.0,
                    max(
                        sum(abs(gradient[i] * jacobian[i][j]) for i in range(7))
                        for j in range(width)
                    ),
                )
                if max(abs(v) for v in responses) < scale * 1e-8:
                    if not include_fixed:
                        return None, None, "symmetry-constrained"
                    state = "model-constrained"
        covariance = row.get("cell_covariance")
        variance = None
        if (
            isinstance(covariance, list)
            and len(covariance) == 7
            and all(isinstance(r, list) and len(r) == 7 for r in covariance)
        ):
            if all(number(v) is not None for r in covariance for v in r):
                variance = sum(
                    gradient[i] * covariance[i][j] * gradient[j] for i in range(7) for j in range(7)
                )
                variance = max(0.0, variance) if variance >= -1e-12 else None
        elif metric in CELL_KEYS and number(row.get(metric + "_esd")) is not None:
            variance = row[metric + "_esd"] ** 2
        if metric in dependencies and state == "refined":
            state = "constraint-derived"
        return number(value), variance, state
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None, None, "invalid"


def _phase_rows(scan):
    packet = scan.get("observables")
    if isinstance(packet, dict) and packet.get("$schema") == "radar-pd-observables/v1":
        return packet.get("phases") or [], packet.get("context") or {}, packet
    if packet:
        return [], {}, {}  # An unsupported new artifact must not become legacy data.
    # Legacy fractions stay inspectable; legacy cell refinement state is unknown.
    cells = scan.get("lattice_parameters") or []
    if cells:
        return cells, {}, {}
    return (
        [
            dict(row, phase=row.get("phase") or row.get("label"), fraction_state="unknown")
            for row in scan.get("phases") or []
        ],
        {},
        {},
    )


def _group(scan, row, context):
    sample = _sample(scan) or "unassigned:" + _run_id(scan)
    return [
        context.get("instrument") or scan.get("instrument") or _run_id(scan).split("_")[0],
        context.get("ipts")
        or scan.get("ipts")
        or (scan.get("metadata") or {}).get("ipts")
        or "unknown",
        sample,
        scan.get("bank") or context.get("bank"),
        row.get("phase_key") or "legacy:" + str(row.get("phase_id") or row.get("phase")),
        row.get("basis_key") or row.get("space_group"),
        context.get("profile_sha256") or context.get("profile_id") or "unverified-profile",
        [
            context.get("recipe_key") or scan.get("analysis_mode") or "unknown-recipe",
            context.get("configuration_sha256")
            or scan.get("configuration_id")
            or "unverified-configuration",
        ],
        context.get("reduction_key") or "unverified-reduction",
    ]


def _ordered(scans):
    # Revisions replace points by exact run/bank ID; caller marks the active revision.
    active = {}
    for scan in scans:
        if scan.get("active_revision", True):
            key = (_run_id(scan), scan.get("bank"))
            previous = active.get(key)
            if previous is not None:
                old_hash = (previous.get("observables") or {}).get("source_gpx_sha256")
                new_hash = (scan.get("observables") or {}).get("source_gpx_sha256")
                if old_hash != new_hash:
                    scan = {**scan, "post_analysis_revision_conflict": True}
            active[key] = scan
    ordered = list(active.values())
    use_time = bool(ordered) and all(
        timestamp((s.get("metadata") or {}).get("start_time")) is not None for s in ordered
    )
    return sorted(
        ordered,
        key=lambda s: (
            (
                timestamp((s.get("metadata") or {}).get("start_time"))
                if use_time
                else number(s.get("run_number")) or 0
            ),
            _run_id(s),
        ),
    )


def calculate_views(
    scans: list[dict[str, Any]],
    settings: PostAnalysisSettings | dict,
    *,
    pending: list[dict[str, Any]] | None = None,
):
    """Pure replay; returns resolved references for the caller to checkpoint."""
    settings = (
        PostAnalysisSettings.model_validate(settings) if isinstance(settings, dict) else settings
    )
    scans = _ordered(scans)
    starts = [
        t
        for s in scans
        for t in [timestamp((s.get("metadata") or {}).get("start_time"))]
        if t is not None
    ]
    origin = min(starts) if starts else None
    references = dict(settings.references)
    colors = dict(settings.phase_colors)
    for scan in scans:
        for row in _phase_rows(scan)[0]:
            identity = str(row.get("phase_key") or row.get("phase_id") or row.get("phase"))
            if identity not in colors:
                colors[identity] = next(
                    (i for i in range(10) if i not in colors.values()), len(colors) % 10
                )
    output = []
    for view in settings.views:
        view_origin = origin
        definition = RECIPES[view.kind]
        result = {
            "id": view.id,
            "kind": view.kind,
            "title": definition[0],
            "recipe_version": 1,
            "status": "waiting",
            "notices": [],
            "series": [],
            "fits": [],
            "crossings": [],
        }
        output.append(result)
        if view.kind in {"broadening", "atomic_structure"}:
            result.update(
                status="needs_recipe",
                notices=[
                    (
                        "A validated instrument-broadening and size/strain refinement recipe is required; GPX defaults are not measurements."
                        if view.kind == "broadening"
                        else "Select validated atom/site definitions and a coordinate/occupancy refinement recipe; fixed CIF values are not measured site changes."
                    )
                ],
            )
            continue
        metric = view.metric or definition[1]
        representation = definition[2] if view.representation == "auto" else view.representation
        needs_reference = representation in {"difference", "relative", "microstrain"}
        axis = settings.axis if view.axis == "shared" else view.axis
        if axis == "auto":
            axis = (
                "temperature"
                if any(_axis(s, "temperature", settings, origin) for s in scans)
                else "run_number"
            )
        if view.kind == "sample_composition":
            axis = "sample_state"
        if view.kind == "reaction_progress" and view.time_zero is None:
            result.update(
                status="needs_input",
                notices=[
                    "Choose an explicit time-zero acquisition timestamp and fixed progress endpoints."
                ],
            )
            continue
        if view.kind == "reaction_progress":
            view_origin = timestamp(view.time_zero)
            axis = "elapsed_time"
            if view_origin is None or view.progress_start is None or view.progress_end is None:
                result.update(
                    status="needs_input",
                    notices=[
                        "Supply a timezone-qualified time zero and two distinct progress endpoints."
                    ],
                )
                continue
        if view.kind == "operando" and axis not in {
            "voltage",
            "capacity",
            "pressure",
            "magnetic_field",
            "elapsed_time",
            "start_time",
        }:
            result.update(
                status="needs_input",
                notices=[
                    "Choose a measured operando axis or provide a verified run-to-state mapping with source and units."
                ],
            )
            continue
        result.update(
            metric=metric,
            representation=representation,
            axis=axis,
            value_unit=metric_unit(metric, representation),
        )
        review_scans = [
            s
            for s in scans
            if s.get("scientific_status") == "needs_review"
            and (not view.samples or _sample(s) in view.samples)
            and (not view.banks or s.get("bank") in view.banks)
        ]
        result["review_required"] = bool(review_scans)
        if review_scans:
            result["notices"].append(
                f"{len(review_scans)} scan(s) require scientific review. Derived values remain conditional on the accepted model, input profile and recorded metadata."
            )
        groups = {}
        cohorts = {}
        skipped = {}
        for scan in scans:
            run = _run_id(scan)
            if view.samples and _sample(scan) not in view.samples:
                continue
            if view.banks and scan.get("bank") not in view.banks:
                continue
            rows, context, packet = _phase_rows(scan)
            cohort_group = _group(scan, {}, context)
            cohort_group[4:6] = []
            cohorts.setdefault(_hash(cohort_group), []).append(scan)
            for row in rows:
                identity = str(row.get("phase_key") or row.get("phase_id") or row.get("phase"))
                if view.phases and identity not in view.phases:
                    continue
                group = _group(scan, row, context)
                group_key = _hash(group)
                value, variance, state = _metric(row, metric, view.hkl, view.include_fixed)
                x = _axis(scan, axis, settings, view_origin)
                reason = ""
                if value is None:
                    reason = f"{state} {metric}"
                elif x is None:
                    reason = "missing or unqualified horizontal-axis metadata"
                elif scan.get("post_analysis_revision_conflict"):
                    reason = "conflicting active revisions; select one accepted revision"
                elif metric == "weight_percent" and state not in {
                    "refined",
                    "constraint-derived",
                }:
                    # Legacy fractions are exploratory, never silently represented as qualified.
                    state = "unverified-fraction"
                if reason:
                    skipped[reason] = skipped.get(reason, 0) + 1
                point = {
                    "run_id": run,
                    "run_number": scan.get("run_number"),
                    "value": value,
                    "variance": variance,
                    "state": state,
                    "scientific_status": scan.get("scientific_status", "not_assessed"),
                    "x": x["value"] if x else None,
                    "axis_unit": x["unit"] if x else "",
                    "axis_source": x["source"] if x else "",
                    "axis_kind": x["kind"] if x else "",
                    "source_gpx_sha256": packet.get("source_gpx_sha256"),
                    "reason": reason,
                    "covariance_status": row.get("covariance_status"),
                    "excluded": run in view.excluded_runs,
                    "temperature": _axis(scan, "temperature", settings, origin),
                    "temperature_span": (scan.get("metadata") or {}).get("temperature") or {},
                    "acquisition_start": (scan.get("metadata") or {}).get("start_time"),
                    "acquisition_end": (scan.get("metadata") or {}).get("end_time"),
                    "explicit_branch": (x.get("branch") if x else None)
                    or (scan.get("metadata") or {}).get("branch"),
                    "cycle": (x.get("cycle") if x else None)
                    or (scan.get("metadata") or {}).get("cycle"),
                    "process_stage": (scan.get("metadata") or {}).get("process_stage"),
                }
                groups.setdefault(
                    group_key,
                    {
                        "key": group_key,
                        "group": group,
                        "cohort": _hash(cohort_group),
                        "color_index": colors[identity],
                        "phase": str(row.get("phase") or identity),
                        "phase_key": identity,
                        "sample": group[2],
                        "bank": group[3],
                        "points": [],
                    },
                )["points"].append(point)
        for key, series in groups.items():
            points = series["points"]
            by_run = {p["run_id"]: p for p in points}
            for scan in cohorts.get(series["cohort"], []):
                run = _run_id(scan)
                if run in by_run:
                    continue
                x = _axis(scan, axis, settings, view_origin)
                points.append(
                    {
                        "run_id": run,
                        "run_number": scan.get("run_number"),
                        "value": None,
                        "variance": None,
                        "state": "absent",
                        "x": x["value"] if x else None,
                        "axis_unit": x["unit"] if x else "",
                        "axis_source": x["source"] if x else "",
                        "axis_kind": x["kind"] if x else "",
                        "source_gpx_sha256": (scan.get("observables") or {}).get(
                            "source_gpx_sha256"
                        ),
                        "reason": "phase absent from accepted model; presence/absence not established",
                        "excluded": run in view.excluded_runs,
                        "temperature_span": (scan.get("metadata") or {}).get("temperature") or {},
                        "explicit_branch": (x.get("branch") if x else None)
                        or (scan.get("metadata") or {}).get("branch"),
                        "cycle": (x.get("cycle") if x else None)
                        or (scan.get("metadata") or {}).get("cycle"),
                        "process_stage": (scan.get("metadata") or {}).get("process_stage"),
                    }
                )
            acquisition_order = {_run_id(s): i for i, s in enumerate(scans)}
            points.sort(key=lambda p: acquisition_order[p["run_id"]])
            for point in points:
                point["raw_value"] = point["value"]
                point["raw_variance"] = point["variance"]
            # Units and scientific axis kinds are part of the series compatibility.
            variants = {(p["axis_unit"], p["axis_kind"]) for p in points if p["x"] is not None}
            if len(variants) > 1:
                result["notices"].append(
                    f"{series['phase']}: mixed axis units or meanings; series withheld."
                )
                continue
            reference_key = f"{view.id}:{key}"
            reference_id = view.reference_run or references.get(reference_key)
            needs_reference = representation in {
                "difference",
                "relative",
                "microstrain",
            }
            if needs_reference and not reference_id:
                earlier_pending = any(
                    (number(p.get("run_number")) or math.inf)
                    <= (number(points[0].get("run_number")) or 0)
                    for p in pending or []
                    if not _sample(p) or _sample(p) == series["sample"]
                )
                if earlier_pending:
                    result["notices"].append(
                        f"{series['phase']}: waiting for earlier acquisition(s) before pinning the reference."
                    )
                else:
                    first = next(
                        (
                            p
                            for p in points
                            if p["value"] is not None and not p["excluded"] and not p["reason"]
                        ),
                        None,
                    )
                    if first:
                        reference_id = first["run_id"]
                        references[reference_key] = reference_id
            reference = next(
                (
                    p
                    for p in points
                    if p["run_id"] == reference_id and not p["excluded"] and not p["reason"]
                ),
                None,
            )
            # Determine direction from acquisition order. Equal x retains the prior branch.
            previous_x = None
            direction = "unclassified"
            segment = 0
            previous_context = None
            block = 0
            for point in points:
                context = (point.get("cycle"), point.get("process_stage"))
                if previous_context is not None and context != previous_context:
                    previous_x = None
                    direction = "unclassified"
                    segment += 1
                    block += 1
                previous_context = context
                point["branch_block"] = block
                x = point["x"]
                explicit = point.get("explicit_branch")
                if explicit:
                    branch = str(explicit)
                elif (
                    axis
                    in {
                        "temperature",
                        "magnetic_field",
                        "pressure",
                        "voltage",
                        "capacity",
                    }
                    and number(x) is not None
                ):
                    new = (
                        "increasing"
                        if previous_x is not None and x > previous_x
                        else (
                            "decreasing" if previous_x is not None and x < previous_x else direction
                        )
                    )
                    if direction != "unclassified" and new != direction:
                        segment += 1
                    direction = new
                    branch = f"{direction}:{segment}"
                    previous_x = x
                else:
                    branch = "acquisition"
                point["branch"] = branch
                point["reference_run"] = reference_id if needs_reference else None
                if point["excluded"] or point["reason"]:
                    point["value"] = None
                    point["variance"] = None
                elif needs_reference:
                    if reference is None or reference["raw_value"] is None:
                        point.update(
                            value=None,
                            variance=None,
                            reason="reference unavailable; choose an explicit replacement",
                        )
                    else:
                        raw, base = point["raw_value"], reference["raw_value"]
                        va, vb = point["raw_variance"], reference["raw_variance"]
                        if representation != "difference" and base == 0:
                            point.update(
                                value=None,
                                variance=None,
                                reason="zero baseline; use an absolute value or difference",
                            )
                        elif point["run_id"] == reference_id:
                            point.update(
                                value=0.0,
                                variance=0.0,
                                uncertainty_note="self-reference identity; physical uncertainty is not zero",
                            )
                        elif representation == "difference":
                            point.update(
                                value=raw - base,
                                variance=(va + vb if va is not None and vb is not None else None),
                            )
                        else:
                            factor = 1e6 if representation == "microstrain" else 100
                            point.update(
                                value=factor * (raw / base - 1),
                                variance=(
                                    factor**2 * (va / base**2 + raw**2 * vb / base**4)
                                    if va is not None and vb is not None
                                    else None
                                ),
                            )
                elif representation == "progress":
                    if view.progress_start is None or view.progress_end is None:
                        point.update(
                            value=None,
                            variance=None,
                            reason="fixed progress endpoints required",
                        )
                    else:
                        span = view.progress_end - view.progress_start
                        point.update(
                            value=(point["raw_value"] - view.progress_start) / span,
                            variance=(
                                point["raw_variance"] / span**2
                                if point["raw_variance"] is not None
                                else None
                            ),
                        )
                if view.branch != "all" and point["branch"] != view.branch:
                    point["display_hidden"] = True
            first_branches = {}
            for point in points:
                if not point["branch"].startswith("unclassified"):
                    first_branches.setdefault(point["branch_block"], point["branch"])
            for point in points:
                if (
                    point["branch"].startswith("unclassified")
                    and point["branch_block"] in first_branches
                ):
                    point["branch"] = first_branches[point["branch_block"]]
                if point.get("cycle") is not None:
                    point["branch"] = f"cycle {point['cycle']} · {point['branch']}"
                if point.get("process_stage") is not None:
                    point["branch"] = f"{point['branch']} · {point['process_stage']}"
                point["display_hidden"] = view.branch != "all" and point["branch"] != view.branch
            series["reference_run"] = reference_id if needs_reference else None
            series["axis_unit"] = next(iter(variants))[0] if variants else ""
            result["series"].append(series)
            if view.fit_expansion:
                _expansion_fit(result, series, view, axis)
            if view.kind == "transition_hysteresis" and view.threshold is not None:
                for left, right in zip(points, points[1:]):
                    if (
                        left["branch"] != right["branch"]
                        or left["value"] is None
                        or right["value"] is None
                    ):
                        continue
                    if (left["value"] - view.threshold) * (right["value"] - view.threshold) < 0:
                        result["crossings"].append(
                            {
                                "series": key,
                                "branch": right["branch"],
                                "runs": [left["run_id"], right["run_id"]],
                                "interval": [left["x"], right["x"]],
                                "threshold": view.threshold,
                            }
                        )
        result["notices"].extend(
            f"{count} point(s): {reason}." for reason, count in skipped.items()
        )
        result["notices"].extend(
            dict.fromkeys(
                str(warning) for scan in scans for warning in scan.get("scientific_warnings") or []
            )
        )
        if any(
            row.get("covariance_status") == "gsas_esd_mismatch"
            for scan in scans
            for row in _phase_rows(scan)[0]
        ):
            result["notices"].append(
                "Some cell covariance blocks disagree with the GSAS-II API ESDs. Derived uncertainties for those blocks are withheld pending review."
            )
        visible = [
            p
            for s in result["series"]
            for p in s["points"]
            if p["value"] is not None and p["x"] is not None and not p.get("display_hidden")
        ]
        result["status"] = "available" if visible else "needs_input" if scans else "waiting"
        if not visible and scans and not result["notices"]:
            result["notices"].append(
                "No eligible points: review the phase, axis, reference and refinement state."
            )
        if any(p["state"] not in {"refined", "constraint-derived"} for p in visible):
            result["notices"].append(
                "Exploratory values include fixed or unverified model quantities; they are not independently measured changes."
            )
        if metric == "weight_percent":
            result["notices"].append(
                "Fractions use the normalized modeled-crystalline mass basis; omitted or amorphous material is not quantified."
            )
        if metric != "weight_percent" and any(
            not s.get("observables") and s.get("lattice_parameters") for s in scans
        ):
            result["notices"].append(
                "Legacy results lack per-parameter fit state. Use a current Analyze result, or explicitly include model values for exploratory inspection."
            )
        if needs_reference:
            result["notices"].append(
                "Uncertainties are conditional on independent scan fits; shared-reference correlations and calibration systematics are not independent point errors."
            )
    return {
        "$schema": "radar-pd-post-analysis-results/v1",
        "views": output,
        "references": references,
        "phase_colors": colors,
        "settings": {
            **settings.model_dump(mode="json"),
            "references": references,
            "phase_colors": colors,
        },
        "source_run_ids": [_run_id(s) for s in scans],
    }


def _expansion_fit(result, series, view, axis):
    """Fit raw L(T), avoiding correlated shared-reference errors."""
    if axis != "temperature":
        result["notices"].append("Expansion coefficient requires the measured temperature axis.")
        return
    branches = {p["branch"] for p in series["points"]}
    for branch in branches:
        points = []
        for p in series["points"]:
            if (
                p["branch"] != branch
                or p["excluded"]
                or p["state"] not in {"refined", "constraint-derived"}
                or p["reason"]
                or p["raw_value"] is None
                or p["x"] is None
            ):
                continue
            if p["axis_kind"] != "measured":
                continue
            if p.get("covariance_status") == "gsas_esd_mismatch":
                continue
            if view.fit_min is not None and p["x"] < view.fit_min:
                continue
            if view.fit_max is not None and p["x"] > view.fit_max:
                continue
            limits = p["temperature_span"]
            lo, hi = number(limits.get("minimum")), number(limits.get("maximum"))
            if lo is not None and hi is not None and hi - lo > view.max_temperature_span:
                continue
            points.append(p)
        if len({p["x"] for p in points}) < 3:
            if any(p.get("covariance_status") == "gsas_esd_mismatch" for p in series["points"]):
                result["notices"].append(
                    f"{series['phase']}: expansion coefficient is withheld while cell covariance is under review."
                )
            result["notices"].append(
                f"{series['phase']} / {branch}: expansion fit needs three distinct qualified temperatures without unresolved ramps."
            )
            continue
        xs = [p["x"] for p in points]
        ys = [p["raw_value"] for p in points]
        weighted = all(p["raw_variance"] is not None and p["raw_variance"] > 0 for p in points)
        weights = [1 / p["raw_variance"] for p in points] if weighted else [1.0] * len(points)
        n = len(points)
        sw = sum(weights)
        x0 = sum(w * x for w, x in zip(weights, xs)) / sw
        y0 = sum(w * y for w, y in zip(weights, ys)) / sw
        sxx = sum(w * (x - x0) ** 2 for w, x in zip(weights, xs))
        slope = sum(w * (x - x0) * (y - y0) for w, x, y in zip(weights, xs, ys)) / sxx
        if y0 <= 0:
            continue
        residual = sum(w * (y - y0 - slope * (x - x0)) ** 2 for w, x, y in zip(weights, xs, ys))
        noise = max(1.0, residual / (n - 2)) if weighted else residual / (n - 2)
        alpha = slope / y0
        variance = noise / (sxx * y0 * y0) + (slope * slope) * noise / (sw * y0**4)
        result["fits"].append(
            {
                "series": series["key"],
                "branch": branch,
                "temperature": x0,
                "unit": "K^-1",
                "alpha": alpha,
                "alpha_esd": math.sqrt(max(0, variance)) if weighted or noise > 1e-25 else None,
                "slope": slope,
                "cell_at_temperature": y0,
                "fit_interval": [min(xs), max(xs)],
                "run_ids": [p["run_id"] for p in points],
                "method": (
                    "Weighted least squares of raw cells; independent scan covariance, variance not reduced below stated errors"
                    if weighted
                    else "OLS of raw cells; residual-based conditional uncertainty"
                )
                + "; no temperature-error or calibration-systematic model",
            }
        )
