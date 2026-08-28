"""Result discovery and Plotly rendering for normalized RADAR-PD outputs."""

from __future__ import annotations

import csv
import base64
import json
import mimetypes
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import plotly.graph_objects as go
from dateutil.tz import tzstr
from plotly.subplots import make_subplots


_EASTERN_TIME = tzstr("EST5EDT,M3.2.0/2,M11.1.0/2")


_FORMULA_TOKEN = r"[A-Z][a-z]?(?:\d+(?:\.\d*)?|\.\d+)?"


def _canonical_phase_name(value: Any) -> str:
    """Normalize formula typography without altering ordinary phase names."""

    text = " ".join(str(value or "").split())
    compact = re.sub(r"\s+", "", text)
    tokens = re.findall(_FORMULA_TOKEN, compact)
    if tokens and "".join(tokens) == compact:
        return re.sub(r"([A-Z][a-z]?)1(?:\.0+)?(?=[A-Z]|$)", r"\1", compact)
    return text


def _canonical_space_group(value: Any) -> str:
    """Compact a Hermann-Mauguin symbol while retaining its group number."""

    text = " ".join(str(value or "").split())
    if not text:
        return text
    number_match = re.match(r"^(.*?)\s*\((\d{1,3})\)\s*$", text)
    symbol = (number_match.group(1) if number_match else text).strip()
    number = number_match.group(2) if number_match else None
    compact_symbol = re.sub(r"\s+", "", symbol)
    if number and compact_symbol:
        return f"{compact_symbol} ({number})"
    return compact_symbol or (f"({number})" if number else text)


def _split_phase_label(label: str) -> tuple[str, str | None]:
    match = re.match(r"^(.*?)\s*\(SG\s+(.+)\)\s*$", label, flags=re.IGNORECASE)
    if match:
        phase = _canonical_phase_name(match.group(1).strip())
        space_group = _canonical_space_group(match.group(2).strip())
        return phase or label, space_group or None
    separated = re.match(
        r"^(.*?)\s+(?:[\u2013\u2014]|--|\|)\s+(.+?\(\d{1,3}\))\s*$",
        label,
    )
    if not separated:
        return _canonical_phase_name(label), None
    phase = _canonical_phase_name(separated.group(1).strip())
    space_group = _canonical_space_group(separated.group(2).strip())
    return phase or label, space_group or None


def _canonical_phase_label(value: Any) -> str:
    phase, space_group = _split_phase_label(str(value or "").strip())
    return f"{phase} (SG {space_group})" if space_group else phase


def _display_phase_label(phase_id: Any, label: Any = None) -> str:
    """Hide catalog storage identifiers from plot labels without changing data keys."""

    phase_text = str(phase_id or "").strip()
    label_text = str(label or "").strip()
    wrapped_prefix = f"{phase_text} ("
    if phase_text and label_text.startswith(wrapped_prefix) and label_text.endswith(")"):
        scientific_text = label_text[len(wrapped_prefix) : -1].strip()
        if scientific_text:
            return _canonical_phase_label(scientific_text)
    if label_text:
        return _canonical_phase_label(label_text)
    if len(phase_text) > 48:
        return phase_text[:45].rstrip("_-") + "..."
    return _canonical_phase_label(phase_text) or "Phase"


def _phase_axis_label(label: Any) -> str:
    """Compact a scientific phase label for the dedicated Bragg-tick axis."""

    text = _canonical_phase_label(label or "Phase")
    if re.search(r"\(SG\s+.+\)$", text, flags=re.IGNORECASE):
        return text
    number_match = re.search(r"\((\d{1,3})\)\)?$", text)
    if not number_match:
        return text if len(text) <= 48 else text[:45].rstrip(" _-") + "..."
    number = number_match.group(1)
    prefix = text[: number_match.start()].rstrip(" )")
    formula = prefix
    for separator in (" - ", " | ", " / ", " -- "):
        if separator in prefix:
            formula = prefix.split(separator, 1)[0].strip()
            break
    if formula == prefix:
        formula = re.split(r"\s+[\u2013\u2014]\s+", prefix, maxsplit=1)[0].strip()
    if formula and formula != prefix:
        compact = f"{formula} (SG {number})"
        return compact if len(compact) <= 48 else compact[:45].rstrip(" _-") + "..."
    return text if len(text) <= 48 else text[:45].rstrip(" _-") + "..."


def _eastern_iso_timestamp(value: Any) -> str | None:
    """Return a browser-stable ISO timestamp with the Eastern UTC offset."""

    text = str(value or "").strip()
    if not text:
        return None
    try:
        timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(_EASTERN_TIME).isoformat()


def read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def read_plot_payload(path: str | Path) -> dict[str, Any]:
    """Load a plot sidecar and its optional compressed scientific arrays."""

    payload_path = Path(path)
    payload = read_json(payload_path)
    arrays_name = payload.get("arrays_npz")
    if not isinstance(arrays_name, str) or not arrays_name.strip():
        return payload

    arrays_path = _resolve_plot_companion(payload_path, arrays_name, {".npz"})
    if arrays_path is None:
        return payload

    try:
        with np.load(arrays_path, allow_pickle=False) as archive:
            payload["arrays"] = {name: archive[name].tolist() for name in archive.files}
    except Exception:
        pass
    return payload


def _resolve_plot_companion(
    sidecar: Path,
    declared_name: str,
    allowed_suffixes: set[str],
) -> Path | None:
    """Resolve exact or Galaxy-flattened files declared by a plot sidecar."""

    basename = Path(str(declared_name or "")).name
    if not basename or Path(basename).suffix.lower() not in allowed_suffixes:
        return None
    parent = sidecar.resolve().parent
    exact = (parent / basename).resolve()
    try:
        exact.relative_to(parent)
        if exact.is_file() and exact.stat().st_size > 0:
            return exact
    except (ValueError, OSError):
        pass
    try:
        candidates = sorted(
            candidate
            for candidate in parent.iterdir()
            if candidate.is_file()
            and candidate.name.endswith(basename)
            and candidate.suffix.lower() in allowed_suffixes
            and candidate.stat().st_size > 0
        )
    except OSError:
        return None
    return candidates[0] if candidates else None


def read_table(path: str | Path, *, limit: int = 500) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for _, row in zip(range(limit), csv.DictReader(handle))]


@dataclass(frozen=True)
class PlotDescriptor:
    """Presentation metadata for one interactive scientific plot."""

    id: str
    name: str
    path: str
    kind: str
    category: str
    stage: str
    rank: int | None = None
    rwp: float | None = None
    valid: bool = True
    primary: bool = False


@dataclass(frozen=True)
class TableDescriptor:
    """Presentation metadata for one published scientific table."""

    id: str
    name: str
    path: str
    category: str
    stage: str
    primary: bool = False


@dataclass(frozen=True)
class CheckpointDescriptor:
    """A downloadable GSAS-II checkpoint with user-facing provenance."""

    id: str
    name: str
    path: str
    stage: str
    status: str
    handoff_available: bool
    local_available: bool
    galaxy_element_name: str = ""


@dataclass
class ResultView:
    """UI-ready view of the stable ``radar-pd-result/v1`` document."""

    mode: str
    status: str
    result_stage: str
    metrics: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    phases: list[dict[str, Any]] = field(default_factory=list)
    phase_total: str = "-"
    plots: list[PlotDescriptor] = field(default_factory=list)
    tables: list[TableDescriptor] = field(default_factory=list)
    primary_plot_path: str = ""
    rapid_stages: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    top_refinements: list[dict[str, Any]] = field(default_factory=list)
    full_progression: list[dict[str, str]] = field(default_factory=list)
    full_models: list[dict[str, str]] = field(default_factory=list)
    checkpoints: list[CheckpointDescriptor] = field(default_factory=list)
    file_groups: list[dict[str, Any]] = field(default_factory=list)

    def to_state(self) -> dict[str, Any]:
        """Convert dataclasses into Trame-serializable dictionaries."""

        return asdict(self)


def _humanize(value: str) -> str:
    text = re.sub(r"__+", " / ", str(value or ""))
    text = re.sub(r"[_-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1].upper() + text[1:] if text else "Result"


def _table_stage(path: Path) -> tuple[str, str, str, bool]:
    lowered = path.as_posix().lower()
    if "all_gsas_validation_summary" in lowered or "final_refinement_ranking" in lowered:
        return "Final refinement ranking", "Rapid results", "final_refinement", True
    if "reranked_512" in lowered or "lattice_aware_pattern_ranking" in lowered:
        return "Pattern scoring", "Rapid results", "pattern_scoring", False
    if "nudge_results" in lowered:
        return "Lattice nudge", "Rapid results", "lattice_nudge", False
    if "beam64" in lowered or "coarse" in lowered:
        return "Coarse search", "Rapid results", "coarse_search", False
    if "summary_fractions" in lowered or "final_phase_fractions" in lowered:
        return "Final phase fractions", "Scientific result", "phase_fractions", True
    if any(token in lowered for token in ("accepted", "best_combination", "ranking", "candidate")):
        return _humanize(path.stem), "Scientific result", "accepted_models", False
    return _humanize(path.stem), "Technical tables", "technical", False


def discover_tables(root: str | Path) -> list[dict[str, Any]]:
    """Discover tables while assigning stable, human-facing roles."""

    base = Path(root)
    options: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for index, path in enumerate(sorted(base.rglob("*.csv"))):
        try:
            signature = (path.name.lower(), path.stat().st_size)
        except OSError:
            continue
        if signature in seen:
            continue
        seen.add(signature)
        name, category, stage, primary = _table_stage(path)
        options.append(
            asdict(
                TableDescriptor(
                    id=f"table-{index}",
                    name=name,
                    path=str(path),
                    category=category,
                    stage=stage,
                    primary=primary,
                )
            )
        )
    category_order = {"Scientific result": 0, "Rapid results": 1, "Technical tables": 2}
    ordered = sorted(
        options,
        key=lambda item: (category_order.get(item["category"], 9), not item["primary"], item["name"]),
    )
    unique: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for item in ordered:
        name_key = str(item.get("name") or "").strip().casefold()
        if name_key in seen_names:
            continue
        seen_names.add(name_key)
        unique.append(item)
    return unique


def _plot_payload_has_arrays(payload: dict[str, Any]) -> bool:
    """Require the x/y curves needed to render a scientific fit."""

    arrays = payload.get("arrays") if isinstance(payload.get("arrays"), dict) else payload
    kind = str(payload.get("plot_kind") or "").lower()
    is_gsas = "gsas" in kind or any(
        key in payload for key in ("phase_ticks", "phase_weights", "rwp")
    )
    if is_gsas:
        return bool(_array(arrays, "x", "two_theta", "tof")) and bool(
            _array(arrays, "yobs", "observed")
            or _array(arrays, "ycalc", "calculated")
        )

    q = _array(payload, "q", "x") or _nested_array(payload, "data", "q", "x")
    measured = _array(payload, "target", "measured", "y") or _nested_array(
        payload, "data", "target", "measured"
    )
    fitted = _array(payload, "total_fit", "fit", "ycalc") or _nested_array(
        payload, "data", "total_fit", "fit"
    )
    return bool(q) and bool(measured or fitted)


def _plot_arrays_available(path: Path, payload: dict[str, Any]) -> bool:
    """Validate an NPZ reference without eagerly loading its numerical arrays."""

    arrays_name = payload.get("arrays_npz")
    if not isinstance(arrays_name, str) or not arrays_name.strip():
        return _plot_payload_has_arrays(payload) or _static_plot_path(path, payload) is not None
    return _resolve_plot_companion(path, arrays_name, {".npz"}) is not None or _static_plot_path(path, payload) is not None


def _static_plot_path(path: Path, payload: dict[str, Any]) -> Path | None:
    """Resolve a plot sidecar's published image without leaving its directory."""

    source_name = str(payload.get("source_plot") or "").strip()
    if not source_name:
        sidecar_suffix = ".plotdata.json"
        source_name = path.name[: -len(sidecar_suffix)] if path.name.endswith(sidecar_suffix) else ""
    if not source_name:
        return None
    return _resolve_plot_companion(path, source_name, {".png", ".jpg", ".jpeg", ".svg"})


def _static_plot_figure(path: Path, payload: dict[str, Any]) -> go.Figure:
    """Render a published static fit when interactive curve arrays are unavailable."""

    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    figure = go.Figure()
    figure.add_layout_image(
        source=f"data:{mime_type};base64,{encoded}",
        xref="paper",
        yref="paper",
        x=0,
        y=1,
        sizex=1,
        sizey=1,
        sizing="contain",
        xanchor="left",
        yanchor="top",
        layer="above",
    )
    figure.update_xaxes(visible=False, range=[0, 1])
    figure.update_yaxes(visible=False, range=[0, 1], scaleanchor="x")
    rwp = _as_float(payload.get("rwp"))
    title = "Published refinement fit" + (f" / Rwp {rwp:.2f}%" if rwp is not None else "")
    return _finish_figure(figure, title, height=720)


def _looks_like_published_fit(path: Path) -> bool:
    """Identify refinement images from older archives that lack plot sidecars."""

    lowered = path.as_posix().lower()
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".svg"}:
        return False
    if any(
        token in lowered
        for token in (
            "accepted_model",
            "accepted_fit",
            "main_phase_fit",
            "refinement_fit",
            "seq_final",
            "kept_polished",
            "final_polish",
        )
    ):
        return True
    return path.name.lower() in {"curve.png", "fit.png"} and any(
        token in lowered for token in ("/gsas/", "rank512", "refinement")
    )


def _plot_stage(path: Path, payload: dict[str, Any]) -> tuple[str, str, int | None]:
    lowered = f"{path.as_posix()} {payload.get('plot_kind', '')}".lower()
    rank_match = re.search(r"rank(?:512)?[_ -]?0*(\d+)", lowered)
    rank = int(rank_match.group(1)) if rank_match else None
    if "magnetic" in lowered:
        return "Diagnostics", "magnetic_precheck", rank
    if "rapid_refined_pattern" in lowered or "component" in lowered:
        return "Hypothesis fits", "pattern_scoring", rank
    if "main_phase" in lowered:
        return "Refinement fits", "main_phase_anchor", rank
    if "gsas" in lowered or "phase_ticks" in payload or payload.get("rwp") is not None:
        return "Refinement fits", "final_refinement", rank
    return "Diagnostics", "diagnostic", rank


def _plot_display_name(path: Path, payload: dict[str, Any], stage: str, rank: int | None) -> str:
    lowered = path.as_posix().lower()
    if stage == "magnetic_precheck":
        return "Magnetic residual precheck"
    if stage == "main_phase_anchor":
        return "Main-phase refinement fit"
    formulas = payload.get("formulas")
    if stage == "pattern_scoring" and formulas:
        label = " + ".join(part.strip() for part in str(formulas).split("|") if part.strip())
        return f"Pattern fit / {label}" if label else "Pattern fit"
    rwp = _as_float(payload.get("rwp"))
    if stage == "final_refinement":
        pass_match = re.search(r"(?:seq[_ -]?)?pass[_ -]?0*(\d+)", lowered)
        if pass_match:
            pass_number = int(pass_match.group(1))
            if "trial_blend" in lowered or "trial" in lowered:
                prefix = f"Pass {pass_number} trial model"
            elif "accepted_model" in lowered or "accepted" in lowered:
                prefix = f"Pass {pass_number} accepted model"
            else:
                prefix = f"Pass {pass_number} refinement"
        else:
            prefix = f"Refinement rank {rank}" if rank is not None else "Refinement fit"
        return f"{prefix} / Rwp {rwp:.2f}%" if rwp is not None else prefix
    title = str(payload.get("title") or "").strip()
    if title and not any(token in title.lower() for token in ("/mnt/", "\\", ".plotdata")):
        return _humanize(title)
    return _humanize(path.name.replace(".plotdata.json", ""))


def discover_plot_payloads(root: str | Path) -> list[dict[str, Any]]:
    """Index plot metadata; numerical arrays stay unloaded until selection."""

    base = Path(root)
    options: list[dict[str, Any]] = []
    represented_images: set[Path] = set()
    for index, path in enumerate(sorted(base.rglob("*.plotdata.json"))):
        payload = read_json(path)
        if not payload or not _plot_arrays_available(path, payload):
            continue
        static_path = _static_plot_path(path, payload)
        if static_path is not None:
            represented_images.add(static_path.resolve())
        category, stage, rank = _plot_stage(path, payload)
        options.append(
            asdict(
                PlotDescriptor(
                    id=f"plot-{index}",
                    name=_plot_display_name(path, payload, stage, rank),
                    path=str(path),
                    kind=str(payload.get("plot_kind") or "interactive plot"),
                    category=category,
                    stage=stage,
                    rank=rank,
                    rwp=_as_float(payload.get("rwp")),
                )
            )
        )
    for path in sorted(base.rglob("*")):
        if not path.is_file() or not _looks_like_published_fit(path):
            continue
        if path.resolve() in represented_images:
            continue
        payload = {
            "plot_kind": "gsas_static_fit_v1",
            "source_plot": path.name,
            "title": _humanize(path.stem),
        }
        category, stage, rank = _plot_stage(path, payload)
        options.append(
            asdict(
                PlotDescriptor(
                    id=f"plot-{len(options)}",
                    name=_plot_display_name(path, payload, stage, rank),
                    path=str(path),
                    kind="published static fit",
                    category=category,
                    stage=stage,
                    rank=rank,
                )
            )
        )
    return options


def _resolve_manifest_plot(root: Path, artifact: dict[str, Any]) -> Path | None:
    """Resolve one normalized plot artifact inside an extracted result archive."""

    relative_values = [artifact.get("path"), artifact.get("source_path"), artifact.get("name")]
    candidates: list[Path] = []
    for value in relative_values:
        text = str(value or "").strip().replace("\\", "/")
        if not text:
            continue
        relative = Path(text)
        if relative.is_absolute() or ".." in relative.parts:
            continue
        candidates.extend((root / relative, root / "ndip" / relative))
    for candidate in candidates:
        if candidate.suffix.lower() not in {".png", ".jpg", ".jpeg", ".svg"}:
            continue
        try:
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
        except OSError:
            continue

    names = {
        Path(str(value)).name
        for value in relative_values
        if str(value or "").strip()
        and Path(str(value)).suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}
    }
    for name in sorted(names):
        for candidate in sorted(root.rglob(name)):
            try:
                if candidate.is_file() and candidate.stat().st_size > 0:
                    return candidate
            except OSError:
                continue
    return None


def _merge_manifest_plots(
    root: Path,
    result: dict[str, Any],
    discovered: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add every renderable plot declared by ``radar-pd-result/v1``."""

    merged = list(discovered)
    represented: set[Path] = set()
    for item in merged:
        item_path = Path(str(item.get("path") or ""))
        if not item_path.name:
            continue
        represented.add(item_path.resolve())
        if not item_path.name.endswith(".plotdata.json"):
            continue
        payload = read_json(item_path)
        static_path = _static_plot_path(item_path, payload) if payload else None
        if static_path is not None:
            represented.add(static_path.resolve())
    artifacts = result.get("artifacts") or {}
    manifest_plots = artifacts.get("plots") if isinstance(artifacts, dict) else []
    for artifact in manifest_plots or []:
        if not isinstance(artifact, dict):
            continue
        path = _resolve_manifest_plot(root, artifact)
        if path is None or path.resolve() in represented:
            continue
        source_text = str(artifact.get("source_path") or artifact.get("path") or path.name)
        payload = {
            "plot_kind": "gsas_static_fit_v1",
            "source_plot": path.name,
            "title": _humanize(Path(source_text).stem),
        }
        category, stage, rank = _plot_stage(Path(source_text), payload)
        if stage == "diagnostic":
            category, stage = "Refinement fits", "final_refinement"
        merged.append(
            asdict(
                PlotDescriptor(
                    id=f"plot-{len(merged)}",
                    name=_plot_display_name(Path(source_text), payload, stage, rank),
                    path=str(path),
                    kind="published static fit",
                    category=category,
                    stage=stage,
                    rank=rank,
                )
            )
        )
        represented.add(path.resolve())
    return merged


def _deduplicate_named_plots(
    plots: list[dict[str, Any]],
    *,
    preferred_path: str = "",
) -> list[dict[str, Any]]:
    """Hide collection aliases that publish the same scientific plot twice.

    Galaxy may expose one fit through both the complete archive and a named
    output collection. If both copies have the same scientific label, keeping
    both makes the plot selector look broken. Prefer the scientifically ranked
    copy and otherwise keep the first renderable descriptor.
    """

    preferred = str(preferred_path or "")
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for plot in plots:
        name = str(plot.get("name") or "Published plot").strip().casefold()
        if name not in grouped:
            grouped[name] = []
            order.append(name)
        grouped[name].append(plot)

    deduplicated: list[dict[str, Any]] = []
    for name in order:
        candidates = grouped[name]
        selected = next(
            (item for item in candidates if preferred and str(item.get("path") or "") == preferred),
            candidates[0],
        )
        deduplicated.append(selected)
    return deduplicated


def load_plot_with_fallback(
    plots: Iterable[dict[str, Any] | PlotDescriptor],
    preferred_path: str | None = None,
) -> tuple[str, dict[str, Any], go.Figure] | None:
    """Load one renderable plot, falling through ranked descriptors on failure."""

    normalized = [asdict(item) if isinstance(item, PlotDescriptor) else dict(item) for item in plots]
    ordered = sorted(
        normalized,
        key=lambda item: (
            0 if preferred_path and str(item.get("path")) == preferred_path else 1,
            0 if item.get("primary") else 1,
            _rank_value(item.get("rank")),
        ),
    )
    for item in ordered:
        path = str(item.get("path") or "")
        if not path:
            continue
        image_path = Path(path)
        if image_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}:
            try:
                payload = {
                    "plot_kind": "gsas_static_fit_v1",
                    "source_plot": image_path.name,
                }
                return path, payload, _static_plot_figure(image_path, payload)
            except Exception:
                continue
        payload = read_plot_payload(path)
        if not payload:
            continue
        if not _plot_payload_has_arrays(payload):
            static_path = _static_plot_path(Path(path), payload)
            if static_path is not None:
                try:
                    return path, payload, _static_plot_figure(static_path, payload)
                except Exception:
                    pass
            continue
        try:
            figure = figure_for_payload(payload)
        except Exception:
            continue
        def populated(value: Any) -> bool:
            try:
                return value is not None and len(value) > 0
            except TypeError:
                return False

        has_points = any(populated(getattr(trace, "x", None)) or populated(getattr(trace, "y", None)) for trace in figure.data)
        if has_points:
            return path, payload, figure
    return None


def _array(payload: dict[str, Any], *names: str) -> list[float]:
    for name in names:
        value = payload.get(name)
        if isinstance(value, list):
            try:
                return [float(item) for item in value]
            except (TypeError, ValueError):
                continue
    return []


def _nested_array(payload: dict[str, Any], section: str, *names: str) -> list[float]:
    nested = payload.get(section)
    return _array(nested, *names) if isinstance(nested, dict) else []


def _first_value(payload: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload and payload[name] is not None:
            return payload[name]
    return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _component_contribution_percent(component: dict[str, Any]) -> float | None:
    percent = _as_float(
        _first_value(
            component,
            "relative_contribution_pct",
            "contribution_pct",
            "weight_percent",
            "weight_pct",
        )
    )
    if percent is not None:
        return percent
    relative_scale = _as_float(_first_value(component, "relative_scale"))
    if relative_scale is not None:
        return relative_scale * 100.0
    contribution = _as_float(_first_value(component, "contribution"))
    if contribution is None:
        return None
    return contribution * 100.0 if abs(contribution) <= 1.0 else contribution


def component_figure(payload: dict[str, Any]) -> go.Figure:
    """Build the compact rapid 512-bin component-fit inspection plot."""

    q = _array(payload, "q", "x") or _nested_array(payload, "data", "q", "x")
    target = _array(payload, "target", "measured", "y") or _nested_array(payload, "data", "target", "measured")
    total = _array(payload, "total_fit", "fit", "ycalc") or _nested_array(payload, "data", "total_fit", "fit")
    residual = _array(payload, "residual", "difference") or _nested_array(payload, "data", "residual", "difference")
    background = _array(payload, "background", "baseline") or _nested_array(payload, "data", "background", "baseline")
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.74, 0.26], vertical_spacing=0.04)
    figure.add_trace(
        go.Scatter(x=q, y=target, mode="markers", name="Measured", marker={"size": 4, "color": "#27313d", "opacity": 0.58}),
        row=1,
        col=1,
    )
    if background:
        figure.add_trace(go.Scatter(x=q, y=background, mode="lines", name="Background", line={"color": "#9aa7b2", "dash": "dash"}), row=1, col=1)
    components = payload.get("components") or payload.get("phase_components") or payload.get("phases") or []
    colors = ["#0f766e", "#2563eb", "#c2410c", "#7c3aed", "#a16207"]
    if isinstance(components, dict):
        components = [{"label": key, "values": value} for key, value in components.items()]
    for index, component in enumerate(components if isinstance(components, list) else []):
        if not isinstance(component, dict):
            continue
        values = _array(component, "scaled", "component", "values", "intensity", "y")
        label = _canonical_phase_label(
            component.get("label")
            or component.get("phase")
            or component.get("formula")
            or f"Phase {index + 1}"
        )
        contribution = _component_contribution_percent(component)
        if contribution is not None:
            label = f"{label} ({float(contribution):.1f}%)"
        figure.add_trace(
            go.Scatter(x=q[: len(values)], y=values, mode="lines", name=label, line={"width": 1.7, "color": colors[index % len(colors)]}),
            row=1,
            col=1,
        )
    figure.add_trace(go.Scatter(x=q, y=total, mode="lines", name="Total hypothesis fit", line={"color": "#dc2626", "width": 2.5}), row=1, col=1)
    figure.add_trace(go.Scatter(x=q, y=residual, mode="lines", name="Difference", line={"color": "#0f766e", "width": 1.3}), row=2, col=1)
    figure.add_hline(y=0, line={"color": "#94a3b8", "width": 1}, row=2, col=1)
    figure.update_yaxes(title_text="Scaled intensity", row=1, col=1)
    figure.update_yaxes(title_text="Difference", row=2, col=1)
    figure.update_xaxes(title_text="Q (1/Angstrom)", row=2, col=1)
    return _finish_figure(figure, str(payload.get("title") or "Pattern contribution fit"), height=560)


def gsas_figure(payload: dict[str, Any]) -> go.Figure:
    """Build an observed/calculated/difference plot with separated phase tick rows."""

    arrays = payload.get("arrays") if isinstance(payload.get("arrays"), dict) else payload
    x = _array(arrays, "x", "two_theta", "tof")
    yobs = _array(arrays, "yobs", "observed")
    ycalc = _array(arrays, "ycalc", "calculated")
    residual = _array(arrays, "resid", "difference")
    phase_order = payload.get("phase_order") or []
    phase_labels = payload.get("phase_labels") or {}
    ticks = payload.get("phase_ticks") or payload.get("ticks") or {}
    major_ticks = payload.get("phase_major_ticks") or {}
    major_tick_details = payload.get("phase_major_tick_details") or {}
    if not phase_order:
        if isinstance(ticks, dict):
            phase_order = list(ticks)
        elif isinstance(phase_labels, dict):
            phase_order = list(phase_labels)
        else:
            phase_order = list(phase_labels)
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.67, 0.20, 0.13],
        vertical_spacing=0.035,
    )
    figure.add_trace(go.Scatter(x=x, y=yobs, mode="markers", name="Observed", marker={"size": 3, "color": "#27313d", "opacity": 0.55}), row=1, col=1)
    figure.add_trace(go.Scatter(x=x, y=ycalc, mode="lines", name="Calculated", line={"color": "#dc2626", "width": 2}), row=1, col=1)
    figure.add_trace(go.Scatter(x=x, y=residual, mode="lines", name="Difference", line={"color": "#0f766e", "width": 1.2}), row=2, col=1)
    figure.add_hline(y=0, line={"color": "#94a3b8", "width": 1}, row=2, col=1)
    colors = ["#0f766e", "#2563eb", "#c2410c", "#7c3aed", "#a16207"]
    if isinstance(ticks, list):
        ticks = {str(label): values for label, values in zip(phase_order, ticks)}
    axis_labels: list[str] = []
    for index, phase_name in enumerate(phase_order):
        raw_label = phase_labels.get(phase_name) if isinstance(phase_labels, dict) else None
        label = _display_phase_label(phase_name, raw_label)
        axis_labels.append(_phase_axis_label(label))
        positions = ticks.get(phase_name, []) if isinstance(ticks, dict) else []
        if isinstance(positions, dict):
            positions = positions.get("all") or positions.get("positions") or []
        color = colors[index % len(colors)]
        figure.add_trace(
            go.Scatter(
                x=positions,
                y=[index] * len(positions),
                mode="markers",
                name=str(label),
                marker={"symbol": "line-ns", "size": 11, "line": {"width": 2, "color": color}},
                hovertemplate=f"{label}<br>%{{x:.5g}}<extra></extra>",
                showlegend=False,
            ),
            row=3,
            col=1,
        )
        strongest: list[tuple[float, str]] = []
        details = major_tick_details.get(phase_name, []) if isinstance(major_tick_details, dict) else []
        for fallback_rank, item in enumerate(details if isinstance(details, list) else [], start=1):
            if not isinstance(item, dict):
                continue
            x_value = _as_float(item.get("x"))
            if x_value is None:
                continue
            rank = item.get("rank", fallback_rank)
            hkl = str(item.get("hkl") or "").strip()
            strength = _as_float(item.get("relative_strength"))
            hover = f"{label}<br>Top Bragg peak {rank}<br>x={x_value:.5g}"
            if hkl:
                hover += f"<br>HKL={hkl}"
            if strength is not None:
                hover += f"<br>Relative strength={strength:.3f}"
            strongest.append((x_value, hover))
        if not strongest:
            fallback = major_ticks.get(phase_name, []) if isinstance(major_ticks, dict) else []
            for rank, value in enumerate(fallback if isinstance(fallback, list) else [], start=1):
                x_value = _as_float(value)
                if x_value is not None:
                    strongest.append((x_value, f"{label}<br>Top Bragg peak {rank}<br>x={x_value:.5g}"))
        if strongest:
            figure.add_trace(
                go.Scatter(
                    x=[item[0] for item in strongest],
                    y=[index] * len(strongest),
                    mode="markers",
                    name=f"Key peaks: {label}",
                    marker={"symbol": "line-ns", "size": 17, "line": {"width": 3, "color": color}},
                    text=[item[1] for item in strongest],
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False,
                ),
                row=3,
                col=1,
            )
    axis_title = str(payload.get("x_label") or payload.get("xlabel") or "Diffraction coordinate")
    figure.update_yaxes(title_text="Intensity", row=1, col=1)
    figure.update_yaxes(title_text="Difference", row=2, col=1)
    figure.update_yaxes(title_text="Phases", tickmode="array", tickvals=list(range(len(axis_labels))), ticktext=axis_labels, row=3, col=1)
    figure.update_xaxes(title_text=axis_title, row=3, col=1)
    rwp = payload.get("rwp")
    title = "Refinement fit" + (f" / Rwp {float(rwp):.2f}%" if rwp is not None else "")
    longest_label = max((len(label) for label in axis_labels), default=0)
    left_margin = min(250, max(110, 45 + longest_label * 7))
    return _finish_figure(figure, title, height=720, left_margin=left_margin)


def figure_for_payload(payload: dict[str, Any]) -> go.Figure:
    if not payload or not _plot_payload_has_arrays(payload):
        figure = go.Figure()
        figure.add_annotation(
            text="No interactive refinement-fit curves were published for this scan.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 16, "color": "#52635b"},
        )
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False)
        return _finish_figure(figure, "Refinement fit unavailable", height=420)
    kind = str(payload.get("plot_kind") or "").lower()
    if "gsas" in kind or any(key in payload for key in ("phase_ticks", "phase_weights", "rwp")):
        return gsas_figure(payload)
    return component_figure(payload)


def _finish_figure(
    figure: go.Figure,
    title: str,
    *,
    height: int,
    left_margin: int = 70,
) -> go.Figure:
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        height=height,
        margin={"l": left_margin, "r": 25, "t": 60, "b": 55},
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "right", "x": 1},
        font={"family": "Inter, Source Sans 3, Arial, sans-serif", "color": "#17251f"},
    )
    figure.update_xaxes(showgrid=False, zeroline=False, linecolor="#cbd5e1")
    figure.update_yaxes(gridcolor="#e8eef0", zeroline=False, linecolor="#cbd5e1")
    return figure


def phase_fraction_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize phase fractions from full or rapid result summaries."""

    candidates: Iterable[Any] = []
    for key in ("phase_fractions", "weighted_fractions", "phases"):
        value = summary.get(key)
        if value:
            candidates = value
            break
    if not candidates:
        candidates = _rapid_final_fractions(summary)
    rows: list[dict[str, Any]] = []
    if isinstance(candidates, dict):
        candidates = [
            {"phase": key, **value} if isinstance(value, dict) else {"phase": key, "weight_percent": value}
            for key, value in candidates.items()
        ]
    for item in candidates if isinstance(candidates, list) else []:
        if not isinstance(item, dict):
            continue
        phase_label = str(
            _first_value(item, "phase", "formula", "compound_name", "label", "phase_id")
            or "Unknown"
        )
        if phase_label.strip().lower() == "main" and str(item.get("is_main", "")).strip().lower() in {
            "1",
            "true",
            "yes",
        }:
            phase_label = "Known main phase"
        phase, label_space_group = _split_phase_label(phase_label)
        space_group = _first_value(item, "space_group", "sg")
        weight = _first_value(
            item,
            "weight_percent",
            "weight_pct",
            "weight_fraction_pct",
            "fraction",
        )
        rows.append(
            {
                "phase": _canonical_phase_name(phase),
                "space_group": (
                    space_group
                    if isinstance(space_group, (int, float)) and not isinstance(space_group, bool)
                    else _canonical_space_group(space_group)
                    if space_group is not None
                    else (label_space_group or "-")
                ),
                "weight_percent": weight if weight is not None else "-",
            }
        )
    return rows


def complete_experiment_space_groups(scans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill symbol-only SG labels from unambiguous peers in the same experiment."""

    known: dict[str, set[str]] = {}
    parsed: list[tuple[dict[str, Any], str, str | None]] = []
    for scan in scans:
        for phase in scan.get("phases") or []:
            if not isinstance(phase, dict):
                continue
            phase["phase"] = _canonical_phase_name(phase.get("phase") or "Unknown")
            value = _canonical_space_group(phase.get("space_group") or "")
            phase["space_group"] = value or "-"
            match = re.match(r"^(.*?)\s*\((\d{1,3})\)\s*$", value)
            symbol = (match.group(1) if match else value).strip()
            number = match.group(2) if match else None
            key = re.sub(r"\s+", "", symbol).lower()
            if key and number:
                known.setdefault(key, set()).add(number)
            parsed.append((phase, key, number))

    for phase, key, number in parsed:
        candidates = known.get(key, set())
        if number or len(candidates) != 1:
            continue
        symbol = _canonical_space_group(phase.get("space_group") or "")
        completed = f"{symbol} ({next(iter(candidates))})"
        phase["space_group"] = completed
    for scan in scans:
        for phase in scan.get("phases") or []:
            if not isinstance(phase, dict):
                continue
            phase_name = _canonical_phase_name(phase.get("phase") or "Unknown")
            space_group = _canonical_space_group(phase.get("space_group") or "") or "-"
            phase["phase"] = phase_name
            phase["space_group"] = space_group
            phase["label"] = (
                f"{phase_name} (SG {space_group})" if space_group != "-" else phase_name
            )
    return scans


def _rapid_final_fractions(summary: dict[str, Any]) -> list[dict[str, Any]]:
    hypotheses = summary.get("hypotheses")
    if not isinstance(hypotheses, list):
        return []
    for hypothesis in hypotheses:
        if not isinstance(hypothesis, dict):
            continue
        raw = _first_value(hypothesis, "weights_json", "phase_fractions", "weights")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        if isinstance(raw, dict) and raw:
            return [{"phase": phase, "weight_percent": weight} for phase, weight in raw.items()]
        if isinstance(raw, list) and raw:
            return [item for item in raw if isinstance(item, dict)]
    return []


def total_elapsed_seconds(summary: dict[str, Any]) -> float | None:
    """Read total time from a direct result summary or nested Rapid summary."""

    direct = _first_value(summary, "elapsed_seconds", "total_seconds")
    value = _as_float(direct)
    if value is not None:
        return value
    timing = summary.get("timing")
    if isinstance(timing, dict):
        value = _as_float(_first_value(timing, "total", "total_seconds"))
        if value is not None:
            return value
    nested = summary.get("summary")
    if not isinstance(nested, dict):
        nested = summary
    live_run = nested.get("live_run") if isinstance(nested, dict) else None
    timings = live_run.get("timings") if isinstance(live_run, dict) else None
    value = _as_float(_first_value(timings, "total_seconds")) if isinstance(timings, dict) else None
    if value is not None:
        return value

    provenance = summary.get("provenance")
    source_manifest = provenance.get("source_manifest") if isinstance(provenance, dict) else None
    started = source_manifest.get("start_time") if isinstance(source_manifest, dict) else None
    finished = _first_value(summary, "created_utc", "completed_utc", "updated_utc")
    if started and finished:
        try:
            def parse_timestamp(value: Any) -> datetime:
                text = str(value).removesuffix("Z")
                for format_string in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
                    try:
                        return datetime.strptime(text, format_string)
                    except ValueError:
                        continue
                return datetime.fromisoformat(text)

            start_time = parse_timestamp(started)
            finish_time = parse_timestamp(finished)
            start_time = start_time.replace(tzinfo=None)
            finish_time = finish_time.replace(tzinfo=None)
            return max(0.0, (finish_time - start_time).total_seconds())
        except (TypeError, ValueError):
            pass
    return None


def _number_text(value: Any, digits: int = 3) -> str:
    number = _as_float(value)
    return "-" if number is None else f"{number:.{digits}f}"


def _rank_value(value: Any, fallback: int = 999999) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return fallback


def _hypothesis_label(row: dict[str, Any]) -> str:
    raw = _first_value(row, "hypothesis", "formulas", "formula_keys", "phases")
    formulas = [
        _canonical_phase_name(part)
        for part in str(raw or "").replace(" + ", "|").split("|")
        if part.strip()
    ]
    groups = [
        _canonical_space_group(part)
        for part in str(row.get("space_groups") or "").split("|")
        if part.strip()
    ]
    if formulas and len(formulas) == len(groups):
        return " + ".join(f"{formula} (SG {group})" for formula, group in zip(formulas, groups))
    return " + ".join(formulas) if formulas else "-"


def _weights_label(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return value or "-"
    if isinstance(value, dict):
        rendered: list[str] = []
        for label, raw in value.items():
            number = _as_float(raw)
            display_label = _canonical_phase_label(label)
            rendered.append(
                f"{display_label}: {number:.1f}%"
                if number is not None
                else f"{display_label}: {raw}"
            )
        return "; ".join(rendered) if rendered else "-"
    return str(value) if value not in (None, "") else "-"


def _peak_support_plain(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return "-"

    def expand(match: re.Match[str]) -> str:
        label = (match.group("label") or "").strip()
        supported = int(match.group("supported"))
        weak = int(match.group("weak"))
        total = int(match.group("total"))
        missing = max(0, total - supported - weak)
        summary = f"{supported} supported, {weak} weak, {missing} missing/review"
        return f"{label}: {summary}" if label else summary

    return re.sub(
        r"(?:(?P<label>[^:;]+):\s*)?(?P<supported>\d+)\+(?P<weak>\d+)/(?P<total>\d+)",
        expand,
        text,
    )


def _cell_summary(row: dict[str, Any]) -> str:
    lengths = [_number_text(row.get(key), 3) for key in ("a", "b", "c")]
    angles = [_number_text(row.get(key), 2) for key in ("alpha", "beta", "gamma")]
    if all(value == "-" for value in [*lengths, *angles]):
        return "-"
    return f"a, b, c: {', '.join(lengths)}; angles: {', '.join(angles)}"


def curate_rapid_rows(stage: str, rows: Iterable[dict[str, Any]], *, limit: int = 100) -> list[dict[str, Any]]:
    """Project internal Rapid CSV columns into the ORC user-facing vocabulary."""

    source = [dict(row) for row in rows]
    rank_key = {
        "coarse_search": "rank64",
        "pattern_scoring": "rank512",
        "final_refinement": "gsas_rwp_rank",
    }.get(stage)
    if rank_key:
        source.sort(key=lambda row: _rank_value(row.get(rank_key, row.get("rank"))))
    projected: list[dict[str, Any]] = []
    for row in source[:limit]:
        if stage == "coarse_search":
            projected.append(
                {
                    "rank": _rank_value(row.get("rank64", row.get("rank")), len(projected) + 1),
                    "hypothesis": _hypothesis_label(row),
                    "coarse_match": _number_text(row.get("gain64")),
                    "unexplained_signal": _number_text(row.get("sse64")),
                }
            )
        elif stage == "lattice_nudge":
            formula = _canonical_phase_name(
                _first_value(row, "formula", "formula_key", "phase_id") or "-"
            )
            projected.append(
                {
                    "phase": str(formula),
                    "space_group": _canonical_space_group(row.get("space_group") or "-"),
                    "nudge_match": _number_text(row.get("best_score")),
                    "cell_adjustment": _number_text(row.get("distance_from_start"), 4),
                    "best_cell": _cell_summary(row),
                    "time": f"{_number_text(row.get('seconds'), 1)} s" if _as_float(row.get("seconds")) is not None else "-",
                    "status": "Needs review" if str(row.get("error") or "").strip() else "Ready",
                }
            )
        elif stage == "pattern_scoring":
            projected.append(
                {
                    "rank": _rank_value(row.get("rank512", row.get("rank")), len(projected) + 1),
                    "hypothesis": _hypothesis_label(row),
                    "key_peak_support": _peak_support_plain(row.get("peak_support_summary")),
                    "coarse_rank": _rank_value(row.get("rank64")),
                    "pattern_match": _number_text(row.get("score512")),
                    "explained_signal": _number_text(row.get("r2_512")),
                    "unexplained_signal": _number_text(row.get("sse512")),
                }
            )
        elif stage == "final_refinement":
            status = str(row.get("status") or "").strip().lower()
            status_label = {"ok": "Converged", "refine_warning": "Needs review", "error": "Failed"}.get(
                status, status.replace("_", " ").title() or "-"
            )
            projected.append(
                {
                    "rank": _rank_value(row.get("gsas_rwp_rank", row.get("rank")), len(projected) + 1),
                    "hypothesis": _hypothesis_label(row),
                    "rwp": _number_text(_first_value(row, "rwp", "Rwp", "refinement_quality")),
                    "phase_fractions": _weights_label(
                        _first_value(row, "weights_json", "phase_fractions", "weights")
                    ),
                    "pattern_rank": _rank_value(row.get("rank512")),
                    "status": status_label,
                    "time": f"{_number_text(row.get('seconds'), 1)} s" if _as_float(row.get("seconds")) is not None else "-",
                }
            )
    return projected


def _best_rwp(result: dict[str, Any]) -> float | None:
    candidates: list[float] = []
    for row in result.get("hypotheses") or []:
        if not isinstance(row, dict):
            continue
        value = _as_float(_first_value(row, "rwp", "Rwp", "refinement_quality"))
        if value is not None:
            candidates.append(value)
    summary = result.get("summary") or {}
    if isinstance(summary, dict):
        for value in (
            summary.get("best_rwp"),
            (summary.get("live_run") or {}).get("best_rwp") if isinstance(summary.get("live_run"), dict) else None,
            (summary.get("final") or {}).get("final_rwp") if isinstance(summary.get("final"), dict) else None,
        ):
            number = _as_float(value)
            if number is not None:
                candidates.append(number)
    manifest = ((result.get("provenance") or {}).get("source_manifest") or {})
    if isinstance(manifest, dict):
        number = _as_float((manifest.get("metrics") or {}).get("final_rwp"))
        if number is not None:
            candidates.append(number)
    return min(candidates) if candidates else None


def experiment_scan_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Build the small, durable scientific record used by experiment dashboards."""

    hypotheses = [row for row in (result.get("hypotheses") or []) if isinstance(row, dict)]
    successful = [
        row
        for row in hypotheses
        if str(row.get("status") or "ok").strip().lower()
        in {"ok", "complete", "converged", "success"}
    ]
    ranked = successful or hypotheses
    ranked.sort(
        key=lambda row: (
            _rank_value(row.get("gsas_rwp_rank", row.get("rank512"))),
            _as_float(row.get("rwp")) if _as_float(row.get("rwp")) is not None else float("inf"),
        )
    )
    best = ranked[0] if ranked else {}
    fractions_source = dict(result)
    if best:
        fractions_source["hypotheses"] = [best]
    phases: list[dict[str, Any]] = []
    for row in phase_fraction_rows(fractions_source):
        weight = _as_float(row.get("weight_percent"))
        if weight is None:
            continue
        phases.append(
            {
                "phase": str(row.get("phase") or "Unknown"),
                "space_group": str(row.get("space_group") or "-"),
                "weight_percent": round(weight, 6),
            }
        )
    phases.sort(key=lambda row: (-float(row["weight_percent"]), row["phase"]))
    elapsed = total_elapsed_seconds(result)
    best_rwp = _best_rwp(result)
    return {
        "status": str(result.get("status") or "complete"),
        "analysis_mode": str(result.get("analysis_mode") or ""),
        "run_name": str(result.get("run_name") or ""),
        "rwp": round(best_rwp, 6) if best_rwp is not None else None,
        "elapsed_seconds": round(elapsed, 3) if elapsed is not None else None,
        "phases": phases,
        "hypothesis": " + ".join(
            f"{row['phase']} (SG {row['space_group']})" if row["space_group"] != "-" else row["phase"]
            for row in phases
        ),
        "warning_count": len(result.get("warnings") or []),
        "error_count": len(result.get("errors") or []),
    }


def experiment_phase_labels(scans: list[dict[str, Any]]) -> list[str]:
    """Rank phase labels for a readable default display without hiding data."""

    statistics: dict[str, dict[str, float]] = {}
    for scan in scans:
        for phase in scan.get("phases") or []:
            if not isinstance(phase, dict):
                continue
            label = str(phase.get("label") or "Unknown")
            weight = float(phase.get("weight_percent") or 0.0)
            stats = statistics.setdefault(label, {"count": 0.0, "maximum": 0.0, "total": 0.0})
            stats["count"] += 1.0
            stats["maximum"] = max(stats["maximum"], weight)
            stats["total"] += weight
    return sorted(
        statistics,
        key=lambda label: (
            -statistics[label]["maximum"],
            -statistics[label]["total"],
            -statistics[label]["count"],
            label,
        ),
    )


def _metadata_metric(scan: dict[str, Any], key: str) -> float | None:
    metadata = scan.get("metadata") or {}
    value = metadata.get(key) if isinstance(metadata, dict) else None
    if not isinstance(value, dict):
        return None
    return _as_float(value.get("value"))


_EXPERIMENT_METRIC_TITLES = {
    "temperature": "Sample temperature",
    "magnetic_field": "Magnetic field",
    "wavelength": "Wavelength",
    "proton_charge": "Proton charge",
}


def _experiment_metric_unit(scans: list[dict[str, Any]], key: str) -> str:
    units = {
        str(metric.get("unit") or "").strip()
        for scan in scans
        for metric in [((scan.get("metadata") or {}).get(key))]
        if isinstance(metric, dict) and _as_float(metric.get("value")) is not None
        and str(metric.get("unit") or "").strip()
    }
    if len(units) == 1:
        return next(iter(units))
    if len(units) > 1:
        return "mixed units"
    return ""


def _experiment_metric_title(scans: list[dict[str, Any]], key: str) -> str:
    title = _EXPERIMENT_METRIC_TITLES[key]
    unit = _experiment_metric_unit(scans, key)
    return f"{title} ({unit})" if unit else f"{title} (unit not reported)"


def experiment_sample_identity(scan: dict[str, Any]) -> dict[str, str]:
    """Return a stable sample key and a readable label for one experiment row."""

    metadata = scan.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    sample_id = str(metadata.get("sample_id") or "").strip()
    sample_name = str(metadata.get("sample_name") or "").strip()
    sample_formula = str(metadata.get("sample_formula") or "").strip()
    if sample_id:
        detail = " | ".join(value for value in (sample_name, sample_formula) if value)
        return {
            "key": f"id:{sample_id}",
            "label": f"Sample {sample_id}{f' - {detail}' if detail else ''}",
            "id": sample_id,
        }
    if sample_name:
        return {
            "key": f"name:{sample_name}",
            "label": f"{sample_name}{f' - {sample_formula}' if sample_formula else ''}",
            "id": "",
        }
    if sample_formula:
        return {"key": f"formula:{sample_formula}", "label": sample_formula, "id": ""}
    return {"key": "unassigned", "label": "Sample not identified", "id": ""}


def experiment_axis_options(scans: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Return trend axes supported by the metadata available for this experiment."""

    options = [{"title": "Scan number", "value": "run_number"}]
    if any(str((scan.get("metadata") or {}).get("start_time") or "") for scan in scans):
        options.append({"title": "Acquisition time (Eastern)", "value": "start_time"})
    for key in _EXPERIMENT_METRIC_TITLES:
        if any(_metadata_metric(scan, key) is not None for scan in scans):
            options.append({"title": _experiment_metric_title(scans, key), "value": key})
    return options


def _experiment_axis(
    scans: list[dict[str, Any]],
    x_key: str,
) -> tuple[list[Any], str, str]:
    if x_key == "start_time":
        return (
            [_eastern_iso_timestamp((scan.get("metadata") or {}).get("start_time")) for scan in scans],
            "Acquisition time (Eastern)",
            "date",
        )
    if x_key in _EXPERIMENT_METRIC_TITLES:
        return (
            [_metadata_metric(scan, x_key) for scan in scans],
            _experiment_metric_title(scans, x_key),
            "linear",
        )
    # Use categorical strings so Plotly never abbreviates 63714 as 63.714k.
    return ([str(scan.get("run_number") or "-") for scan in scans], "POWGEN scan", "category")


def _scan_condition_text(scan: dict[str, Any]) -> str:
    metadata = scan.get("metadata") or {}
    if not isinstance(metadata, dict):
        return ""
    conditions: list[str] = []
    for label, key in (
        ("T", "temperature"),
        ("Field", "magnetic_field"),
        ("Wavelength", "wavelength"),
    ):
        metric = metadata.get(key)
        if not isinstance(metric, dict) or _as_float(metric.get("value")) is None:
            continue
        value = float(metric["value"])
        unit = str(metric.get("unit") or "").strip()
        conditions.append(
            f"{label}: {value:g} {unit}" if unit else f"{label}: {value:g} [unit not reported]"
        )
    return " | ".join(conditions)


def experiment_phase_fraction_figure(
    scans: list[dict[str, Any]],
    *,
    selected_labels: list[str] | None = None,
    x_key: str = "run_number",
) -> go.Figure:
    """Interactive phase-fraction trend with a deliberately bounded trace set."""

    figure = go.Figure()
    ordered = sorted(scans, key=lambda row: int(row.get("run_number") or 0))
    ranked_labels = experiment_phase_labels(ordered)
    labels = [label for label in (selected_labels or ranked_labels[:5]) if label in ranked_labels]
    x_values, x_title, x_type = _experiment_axis(ordered, x_key)
    chronological_axis = x_key in {"run_number", "start_time"}
    customdata = []
    for scan in ordered:
        sample = experiment_sample_identity(scan)
        customdata.append(
            [
                str(scan.get("run_number") or "-"),
                _scan_condition_text(scan) or "No scan conditions",
                sample["label"],
            ]
        )
    for label in labels:
        values: list[float | None] = []
        for scan in ordered:
            match = next(
                (
                    phase
                    for phase in (scan.get("phases") or [])
                    if isinstance(phase, dict) and str(phase.get("label")) == label
                ),
                None,
            )
            values.append(float(match["weight_percent"]) if match is not None else None)
        trace_x = list(x_values)
        trace_y = list(values)
        trace_customdata = list(customdata)
        if chronological_axis:
            separated_x: list[Any] = []
            separated_y: list[float | None] = []
            separated_customdata: list[list[str]] = []
            previous_sample = ""
            for scan, x_value, y_value, hover_row in zip(
                ordered,
                trace_x,
                trace_y,
                trace_customdata,
            ):
                sample_key = experiment_sample_identity(scan)["key"]
                if previous_sample and sample_key != previous_sample:
                    separated_x.append(None)
                    separated_y.append(None)
                    separated_customdata.append(["", "", ""])
                separated_x.append(x_value)
                separated_y.append(y_value)
                separated_customdata.append(hover_row)
                previous_sample = sample_key
            trace_x = separated_x
            trace_y = separated_y
            trace_customdata = separated_customdata
        figure.add_trace(
            go.Scatter(
                x=trace_x,
                y=trace_y,
                customdata=trace_customdata,
                mode="lines+markers" if chronological_axis else "markers",
                name=label,
                connectgaps=False,
                line=dict(width=1.5),
                marker=dict(size=9, opacity=0.86, line=dict(color="#ffffff", width=1.2)),
                hovertemplate=(
                    "Scan %{customdata[0]}<br>%{fullData.name}: %{y:.2f} wt%"
                    "<br>%{customdata[2]}<br>%{customdata[1]}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title=None,
        xaxis_title=x_title,
        yaxis_title="Weight fraction (wt%)",
        height=430,
        margin=dict(l=62, r=24, t=24, b=104),
        hovermode="x unified" if chronological_axis else "closest",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.20,
            xanchor="left",
            x=0,
            font=dict(size=11),
            itemwidth=34,
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    figure.update_xaxes(showgrid=False, linecolor="#cbd5e1", type=x_type)
    figure.update_yaxes(gridcolor="#e8eef0", linecolor="#cbd5e1", rangemode="tozero")
    return figure


def experiment_phase_heatmap_figure(
    scans: list[dict[str, Any]],
    *,
    x_key: str = "run_number",
) -> go.Figure:
    """Compact scan-by-phase map for experiments with many evolving phases."""

    ordered = sorted(scans, key=lambda row: int(row.get("run_number") or 0))
    labels = experiment_phase_labels(ordered)
    x_values, x_title, x_type = _experiment_axis(ordered, x_key)
    z_values: list[list[float | None]] = []
    hover: list[list[str]] = []
    for label in labels:
        row_values: list[float | None] = []
        row_hover: list[str] = []
        for scan in ordered:
            match = next(
                (
                    phase
                    for phase in (scan.get("phases") or [])
                    if isinstance(phase, dict) and str(phase.get("label")) == label
                ),
                None,
            )
            value = float(match["weight_percent"]) if match is not None else None
            row_values.append(value)
            sample_label = experiment_sample_identity(scan)["label"]
            row_hover.append(
                f"Scan {scan.get('run_number')}<br>{label}<br>"
                + (f"{value:.2f} wt%" if value is not None else "Not reported")
                + f"<br>{sample_label}"
            )
        z_values.append(row_values)
        hover.append(row_hover)

    figure = go.Figure(
        go.Heatmap(
            x=x_values,
            y=labels,
            z=z_values,
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            colorscale=[
                [0.0, "#edf7f2"],
                [0.25, "#b8dfcf"],
                [0.55, "#4da989"],
                [1.0, "#0b5138"],
            ],
            colorbar=dict(title="wt%", thickness=12),
            zmin=0,
            zmax=100,
            hoverongaps=False,
        )
    )
    figure.update_layout(
        title=None,
        xaxis_title=x_title,
        yaxis_title="",
        height=max(300, min(720, 150 + 34 * max(1, len(labels)))),
        margin=dict(l=140, r=28, t=24, b=58),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    figure.update_xaxes(showgrid=False, linecolor="#cbd5e1", type=x_type)
    figure.update_yaxes(showgrid=False, linecolor="#cbd5e1", autorange="reversed")
    return figure


def experiment_fit_diagnostics(scans: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Classify relative Rwp excursions using the experiment's own history.

    The labels are operational diagnostics, not scientific acceptance gates.
    A median/MAD baseline avoids imposing one arbitrary Rwp threshold across
    different instruments, samples, and acquisition conditions.
    """

    usable = [row for row in scans if _as_float(row.get("rwp")) is not None]
    if not usable:
        return {}
    values = np.asarray([float(row["rwp"]) for row in usable], dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, 0.5, 0.05 * max(median, 1.0))
    diagnostics: dict[str, dict[str, Any]] = {}
    for row in usable:
        value = float(row["rwp"])
        score = (value - median) / scale
        if len(usable) < 4:
            label, color = "Collecting baseline", "#64748b"
        elif score >= 3.0:
            label, color = "Rwp outlier", "#b42318"
        elif score >= 2.0:
            label, color = "Elevated Rwp", "#b4690e"
        else:
            label, color = "Within experiment trend", "#14734f"
        diagnostics[str(row.get("run_id") or row.get("run_number"))] = {
            "label": label,
            "color": color,
            "score": round(score, 4),
            "baseline_median": round(median, 6),
        }
    return diagnostics


def experiment_fit_quality_figure(
    scans: list[dict[str, Any]],
    *,
    x_key: str = "run_number",
) -> go.Figure:
    """Interactive Rwp trend for completed scans."""

    ordered = sorted(
        [row for row in scans if _as_float(row.get("rwp")) is not None],
        key=lambda row: int(row.get("run_number") or 0),
    )
    diagnostics = experiment_fit_diagnostics(ordered)
    x_values, x_title, x_type = _experiment_axis(ordered, x_key)
    y_values: list[float | None] = [float(row["rwp"]) for row in ordered]
    text_values: list[str] = [str(row.get("hypothesis") or "No phase summary") for row in ordered]
    customdata: list[list[str]] = [
        [
            str(row.get("run_number") or "-"),
            str(row.get("elapsed_display") or "-"),
            _scan_condition_text(row) or "No scan conditions",
            experiment_sample_identity(row)["label"],
        ]
        for row in ordered
    ]
    colors = [
        diagnostics.get(str(row.get("run_id") or row.get("run_number")), {}).get("color", "#e57c46")
        for row in ordered
    ]
    if x_key in {"run_number", "start_time"}:
        segmented_x: list[Any] = []
        segmented_y: list[float | None] = []
        segmented_text: list[str] = []
        segmented_customdata: list[list[str]] = []
        segmented_colors: list[str] = []
        previous_sample = ""
        for row, x_value, y_value, text_value, hover_row, color in zip(
            ordered,
            x_values,
            y_values,
            text_values,
            customdata,
            colors,
        ):
            sample_key = experiment_sample_identity(row)["key"]
            if previous_sample and sample_key != previous_sample:
                segmented_x.append(None)
                segmented_y.append(None)
                segmented_text.append("")
                segmented_customdata.append(["", "", "", ""])
                segmented_colors.append("rgba(0,0,0,0)")
            segmented_x.append(x_value)
            segmented_y.append(y_value)
            segmented_text.append(text_value)
            segmented_customdata.append(hover_row)
            segmented_colors.append(color)
            previous_sample = sample_key
        x_values = segmented_x
        y_values = segmented_y
        text_values = segmented_text
        customdata = segmented_customdata
        colors = segmented_colors
    figure = go.Figure(
        go.Scatter(
            x=x_values,
            y=y_values,
            text=text_values,
            customdata=customdata,
            mode="lines+markers" if x_key in {"run_number", "start_time"} else "markers",
            name="Rwp",
            line=dict(color="#15543c", width=2),
            marker=dict(color=colors, size=9, line=dict(color="#ffffff", width=1)),
            hovertemplate=(
                "Scan %{customdata[0]}<br>Rwp: %{y:.2f}%<br>%{text}"
                "<br>%{customdata[3]}<br>%{customdata[2]}"
                "<br>Runtime: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    if ordered:
        median = float(np.median([float(row["rwp"]) for row in ordered]))
        figure.add_hline(
            y=median,
            line_dash="dash",
            line_color="#94a3b8",
            annotation_text=f"Experiment median {median:.2f}%",
            annotation_position="top left",
        )
    figure.update_layout(
        title=None,
        xaxis_title=x_title,
        yaxis_title="Rwp (%) - lower is better",
        height=340,
        margin=dict(l=62, r=24, t=24, b=58),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        showlegend=False,
    )
    figure.update_xaxes(showgrid=False, linecolor="#cbd5e1", type=x_type)
    figure.update_yaxes(gridcolor="#e8eef0", linecolor="#cbd5e1", rangemode="tozero")
    return figure


def _message_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("message") or value.get("error") or json.dumps(value, default=str))
    return str(value)


def _primary_plot_index(plots: list[dict[str, Any]], result: dict[str, Any], mode: str) -> int | None:
    if not plots:
        return None
    hypotheses = [row for row in (result.get("hypotheses") or []) if isinstance(row, dict)]
    successful = [row for row in hypotheses if str(row.get("status") or "ok").lower() in {"ok", "complete", "converged", "success"}]
    ranked = successful or hypotheses
    ranked.sort(
        key=lambda row: (
            _rank_value(row.get("gsas_rwp_rank", row.get("rank512"))),
            _as_float(row.get("rwp")) if _as_float(row.get("rwp")) is not None else float("inf"),
        )
    )
    best = ranked[0] if ranked else {}
    curve_path = str(best.get("curve_png") or best.get("curve_csv") or "").replace("\\", "/").lower()
    curve_parts = [part for part in curve_path.split("/") if part][-3:]
    best_pattern_rank = _rank_value(best.get("rank512"))

    def priority(item: dict[str, Any]) -> tuple[Any, ...]:
        lowered = str(item.get("path") or "").replace("\\", "/").lower()
        stage = str(item.get("stage") or "")
        rwp = _as_float(item.get("rwp"))
        rank = item.get("rank")
        if mode == "rapid":
            curve_match = bool(curve_parts and all(part in lowered for part in curve_parts[:-1]))
            rank_match = rank is not None and int(rank) == best_pattern_rank
            stage_score = 0 if curve_match else 1 if stage == "final_refinement" and rank_match else 2 if stage == "final_refinement" else 4 if stage == "pattern_scoring" else 8
            return (stage_score, rwp if rwp is not None else float("inf"), _rank_value(rank))
        final_score = 0 if any(token in lowered for token in ("seq_final", "accepted_model", "final", "polish")) else 2 if stage == "final_refinement" else 4 if stage == "main_phase_anchor" else 8
        pass_match = re.search(r"pass[_ -]?(\d+)", lowered)
        latest_pass = -int(pass_match.group(1)) if pass_match else 0
        return (final_score, latest_pass, rwp if rwp is not None else float("inf"))

    return min(range(len(plots)), key=lambda index: priority(plots[index]))


def _phase_rows_for_view(result: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    rows = phase_fraction_rows(result)
    total = 0.0
    has_weight = False
    for row in rows:
        number = _as_float(row.get("weight_percent"))
        row["weight_value"] = number if number is not None else 0.0
        row["weight_display"] = f"{number:.2f}%" if number is not None else "-"
        row["bar_width"] = min(100.0, max(0.0, number or 0.0))
        if number is not None:
            total += number
            has_weight = True
    return rows, (f"{total:.2f}% total" if has_weight else "-")


def _local_file_groups(root: Path) -> list[dict[str, Any]]:
    group_order = [
        "Report and archive",
        "Scientific tables",
        "Fit plots",
        "Phase CIFs",
        "GSAS-II projects",
        "Reproducibility",
        "Diagnostics and logs",
    ]
    groups: dict[str, list[dict[str, Any]]] = {name: [] for name in group_order}
    seen: set[tuple[str, int]] = set()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name == ".radar-pd-cache.json":
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        signature = (path.name.lower(), size)
        if signature in seen:
            continue
        seen.add(signature)
        name = path.name.lower()
        suffix = path.suffix.lower()
        if name in {"report.html", "results.zip", "overview.tsv"}:
            group = "Report and archive"
        elif suffix in {".csv", ".tsv"}:
            group = "Scientific tables"
        elif suffix in {".png", ".jpg", ".jpeg", ".svg", ".html"} and name != "report.html":
            group = "Fit plots"
        elif suffix == ".cif":
            group = "Phase CIFs"
        elif suffix == ".gpx":
            group = "GSAS-II projects"
        elif name in {"resolved_config.yaml", "input_manifest.json", "summary.json", "state.json"} or "config" in name:
            group = "Reproducibility"
        else:
            group = "Diagnostics and logs"
        display = {
            "report.html": "Integrated result report",
            "results.zip": "Complete results archive",
            "overview.tsv": "Result overview",
            "resolved_config.yaml": "Resolved RADAR-PD configuration",
            "input_manifest.json": "Input provenance manifest",
        }.get(name, _humanize(path.name.replace(".plotdata.json", " interactive data")))
        groups[group].append(
            {
                "id": f"file-{len(seen)}",
                "name": display,
                "filename": path.name,
                "path": str(path),
                "size": f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / (1024 * 1024):.1f} MB",
                "technical": group == "Diagnostics and logs"
                or path.name.endswith(".plotdata.json")
                or suffix == ".npz",
            }
        )
    return [{"name": name, "files": groups[name]} for name in group_order if groups[name]]


def _checkpoint_descriptors(result: dict[str, Any], root: Path) -> list[CheckpointDescriptor]:
    local = list(root.rglob("*.gpx"))
    descriptors: list[CheckpointDescriptor] = []
    for index, item in enumerate(result.get("gpx_projects") or []):
        if not isinstance(item, dict):
            continue
        collection_name = Path(str(item.get("collection_name") or item.get("collection_path") or "")).name
        candidates = [
            Path(str(item.get("path") or "")).name,
            Path(str(item.get("source_path") or "")).name,
        ]
        # Prefer the exact file published as the Galaxy collection element. The
        # archive can also contain a technical source copy with a different name;
        # selecting that copy makes a later GPX handoff impossible to map.
        path = next((candidate for candidate in local if collection_name and candidate.name == collection_name), None)
        if path is None:
            path = next((candidate for candidate in local if candidate.name in candidates), None)
        label = str(item.get("label") or (path.stem if path else "GSAS-II checkpoint"))
        display_name = _humanize(label)
        if "gpx" not in display_name.casefold():
            display_name = f"{display_name} (GPX)"
        descriptors.append(
            CheckpointDescriptor(
                id=f"checkpoint-{index}",
                name=display_name,
                path=str(path) if path else "",
                stage=_humanize(str(item.get("stage") or "Refinement checkpoint")),
                status=_humanize(str(item.get("status") or "Available")),
                handoff_available=bool(collection_name or path),
                local_available=path is not None,
                galaxy_element_name=Path(collection_name).stem if collection_name else (path.stem if path else ""),
            )
        )
    return descriptors


def _full_model_rows(result: dict[str, Any]) -> list[dict[str, str]]:
    """Project Full-mode decisions without exposing pipeline paths or IDs."""

    rows: list[dict[str, str]] = []
    for item in result.get("hypotheses") or []:
        if not isinstance(item, dict):
            continue
        status = _humanize(str(_first_value(item, "decision", "status") or "Reported"))
        stage = _humanize(str(_first_value(item, "stage", "hypothesis_stage", "pipeline_pass") or "Refinement"))
        warning = str(_first_value(item, "warning", "message") or "").strip()
        rows.append(
            {
                "model": _hypothesis_label(item),
                "stage": stage,
                "rwp": _number_text(_first_value(item, "rwp", "Rwp", "refinement_quality")),
                "decision": status,
                "note": warning or "-",
            }
        )
    if rows:
        return rows

    for item in result.get("gpx_projects") or []:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "model": _humanize(str(item.get("label") or "Published model")),
                "stage": _humanize(str(item.get("stage") or "Refinement")),
                "rwp": _number_text(_first_value(item, "rwp", "Rwp")),
                "decision": _humanize(str(item.get("status") or "Published")),
                "note": str(item.get("warning") or "-") or "-",
            }
        )
    return rows


def build_result_view(
    result: dict[str, Any],
    root: str | Path,
    *,
    submitted_mode: str | None = None,
) -> ResultView:
    """Build a deterministic Rapid or Full scientific presentation model."""

    base = Path(root)
    mode = str(result.get("analysis_mode") or "full").lower()
    status = str(result.get("status") or "complete").replace("_", " ").title()
    result_stage = str(result.get("hypothesis_stage") or "phase_refinement")
    if result_stage.strip().lower() in {"", "none", "null", "unknown"}:
        result_stage = "final_refinement" if mode == "full" and status == "Complete" else "phase_refinement"
    phase_rows, phase_total = _phase_rows_for_view(result)
    elapsed = total_elapsed_seconds(result)
    best_rwp = _best_rwp(result)

    plot_dicts = _merge_manifest_plots(base, result, discover_plot_payloads(base))
    original_primary_index = _primary_plot_index(plot_dicts, result, mode)
    preferred_plot_path = (
        str(plot_dicts[original_primary_index].get("path") or "")
        if original_primary_index is not None
        else ""
    )
    plot_dicts = _deduplicate_named_plots(plot_dicts, preferred_path=preferred_plot_path)
    primary_index = _primary_plot_index(plot_dicts, result, mode)
    if primary_index is not None:
        plot_dicts[primary_index]["primary"] = True
        plot_dicts[primary_index]["category"] = "Best refinement"
        original_name = str(plot_dicts[primary_index].get("name") or "Refinement fit")
        plot_dicts[primary_index]["name"] = f"Best refinement - {original_name}"
    plots = [PlotDescriptor(**item) for item in plot_dicts]
    tables = [TableDescriptor(**item) for item in discover_tables(base)]

    table_rows: dict[str, list[dict[str, Any]]] = {}
    for descriptor in tables:
        if descriptor.stage not in table_rows:
            table_rows[descriptor.stage] = read_table(descriptor.path)
    final_source = [row for row in (result.get("hypotheses") or []) if isinstance(row, dict)]
    if not final_source:
        final_source = table_rows.get("final_refinement", [])
    rapid_stages = {
        "coarse_search": curate_rapid_rows("coarse_search", table_rows.get("coarse_search", [])),
        "lattice_nudge": curate_rapid_rows("lattice_nudge", table_rows.get("lattice_nudge", [])),
        "pattern_scoring": curate_rapid_rows("pattern_scoring", table_rows.get("pattern_scoring", [])),
        "final_refinement": curate_rapid_rows("final_refinement", final_source),
    }

    warnings = [_message_text(item) for item in [*(result.get("warnings") or []), *(result.get("errors") or [])] if _message_text(item)]
    if submitted_mode and submitted_mode.lower() != mode:
        warnings.insert(
            0,
            f"Mode mismatch: NOVA submitted {submitted_mode.title()}, but radar-pd-result/v1 reports {mode.title()}. The result mode is authoritative.",
        )
    if mode == "rapid" and result_stage == "pattern_scoring":
        warnings.insert(
            0,
            "Final refinement is not available. Pattern scale coefficients are comparative and are not quantitative phase fractions.",
        )

    progression: list[dict[str, str]] = []
    seen_progression: set[tuple[str, str]] = set()
    for item in result.get("gpx_projects") or []:
        if not isinstance(item, dict):
            continue
        stage = _humanize(str(item.get("stage") or "Refinement"))
        state = _humanize(str(item.get("status") or "Checkpoint"))
        key = (stage, state)
        if key not in seen_progression:
            seen_progression.add(key)
            progression.append({"stage": stage, "status": state})

    metrics = [
        {"label": "Analysis", "value": "Rapid Hypothesis" if mode == "rapid" else "Full RADAR-PD"},
        {"label": "Status", "value": status},
        {"label": "Best Rwp", "value": f"{best_rwp:.3f}%" if best_rwp is not None else "-"},
        {"label": "Reported phases", "value": str(len(phase_rows)) if phase_rows else "-"},
        {"label": "Total time", "value": f"{elapsed:.1f} s" if elapsed is not None else "-"},
        {"label": "Result stage", "value": _humanize(result_stage)},
    ]
    primary_path = plots[primary_index].path if primary_index is not None else ""
    return ResultView(
        mode=mode,
        status=status,
        result_stage=result_stage,
        metrics=metrics,
        warnings=warnings,
        phases=phase_rows,
        phase_total=phase_total,
        plots=plots,
        tables=tables,
        primary_plot_path=primary_path,
        rapid_stages=rapid_stages,
        top_refinements=rapid_stages["final_refinement"][:5],
        full_progression=progression,
        full_models=_full_model_rows(result) if mode == "full" else [],
        checkpoints=_checkpoint_descriptors(result, base),
        file_groups=_local_file_groups(base),
    )
