"""NOVA controls for saved, replayable post-analysis view instances."""

from __future__ import annotations

import asyncio
import csv
from pathlib import Path

from trame.widgets import html, plotly, vuetify3 as vuetify

from .post_analysis import (
    PostAnalysisSettings,
    ViewSettings,
    RECIPES,
    AXES,
    calculate_views,
    recipe_options,
)
from .post_analysis_plot import view_figure, export_bundle
from .uploads import NamedFileUpload


class PostAnalysisMixin:
    def _init_post_analysis(self):
        state = self.server.state
        state.post_analysis = PostAnalysisSettings().model_dump(mode="json")
        state.post_analysis_selected = [v["kind"] for v in state.post_analysis["views"]]
        state.post_analysis_recipes = recipe_options()
        state.post_analysis_active = "phases"
        state.post_analysis_view_options = []
        state.post_analysis_phase_options = []
        state.post_analysis_metric_options = []
        state.post_analysis_branch_options = ["all"]
        state.post_analysis_reference_options = []
        state.post_analysis_editor = {}
        state.post_analysis_status = "Waiting for completed scans"
        state.post_analysis_notices = []
        state.post_analysis_fit_rows = []
        state.post_analysis_panels = []
        state.post_analysis_mapping_path = ""
        state.post_analysis_error = ""
        self._post_analysis_widgets = []
        self._post_analysis_scans = []
        self._post_analysis_result = {"views": []}
        self._select_post_analysis_editor()

    def _post_analysis_setup(self):
        vuetify.VAutocomplete(
            label="Post-analysis views",
            v_model=("post_analysis_selected",),
            items=("post_analysis_recipes",),
            multiple=True,
            chips=True,
            closable_chips=True,
            density="compact",
            variant="outlined",
            hide_details=True,
            update_modelValue=(self.update_post_analysis_selection, "[$event]"),
        )
        html.P(
            "Choose the scientific views to update as scans finish. Requirements are checked per phase and bank.",
            classes="radar-section-help",
        )

    def _post_analysis_dashboard(self):
        with html.Section(classes="radar-result-card radar-post-analysis-card mb-4"):
            html.H3("Selected post-analysis views")
            self._post_analysis_setup()
            with html.Div(classes="radar-experiment-controls"):
                vuetify.VSelect(
                    label="View",
                    v_model=("post_analysis_active",),
                    items=("post_analysis_view_options",),
                    density="compact",
                    variant="outlined",
                    update_modelValue=(self.select_post_analysis_view, "[$event]"),
                )
                vuetify.VBtn(
                    "Duplicate view",
                    click=self.duplicate_post_analysis_view,
                    variant="text",
                    size="small",
                    disabled=("!post_analysis_active",),
                )
                vuetify.VBtn(
                    "Remove this view",
                    click=self.remove_post_analysis_view,
                    variant="text",
                    size="small",
                    disabled=("!post_analysis_active",),
                )
            vuetify.VAlert(
                text=("post_analysis_status",),
                type="info",
                variant="tonal",
                density="compact",
                classes="my-2",
            )
            vuetify.VAlert(
                v_show="!!post_analysis_error",
                text=("post_analysis_error",),
                type="error",
                variant="tonal",
                density="compact",
            )
            with html.Div(v_if="!!post_analysis_active", classes="radar-experiment-controls"):
                vuetify.VSelect(
                    label="Horizontal axis",
                    v_model=("post_analysis_editor.axis",),
                    items=([{"title": a.replace("_", " ").title(), "value": a} for a in AXES],),
                    density="compact",
                    variant="outlined",
                    update_modelValue=(self.update_post_analysis_setting, "['axis', $event]"),
                )
                vuetify.VSelect(
                    label="Metric",
                    v_model=("post_analysis_editor.metric",),
                    items=("post_analysis_metric_options",),
                    density="compact",
                    variant="outlined",
                    update_modelValue=(self.update_post_analysis_setting, "['metric', $event]"),
                )
                vuetify.VSelect(
                    label="Representation",
                    v_model=("post_analysis_editor.representation",),
                    items=("post_analysis_representation_options",),
                    density="compact",
                    variant="outlined",
                    update_modelValue=(
                        self.update_post_analysis_setting,
                        "['representation', $event]",
                    ),
                )
                vuetify.VSelect(
                    label="Phases shown",
                    v_model=("post_analysis_editor.phases",),
                    items=("post_analysis_phase_options",),
                    multiple=True,
                    chips=True,
                    clearable=True,
                    density="compact",
                    variant="outlined",
                    hint="Empty means all phases; hiding does not renormalize fractions.",
                    persistent_hint=True,
                    update_modelValue=(self.update_post_analysis_setting, "['phases', $event]"),
                )
                vuetify.VCombobox(
                    label="Reference scan",
                    v_model=("post_analysis_editor.reference_run",),
                    items=("post_analysis_reference_options",),
                    clearable=True,
                    density="compact",
                    variant="outlined",
                    hint="Blank pins the first eligible acquisition in each compatible series.",
                    persistent_hint=True,
                    update_modelValue=(
                        self.update_post_analysis_setting,
                        "['reference_run', $event]",
                    ),
                )
                vuetify.VSelect(
                    label="Branch / cycle",
                    v_model=("post_analysis_editor.branch",),
                    items=("post_analysis_branch_options",),
                    density="compact",
                    variant="outlined",
                    update_modelValue=(self.update_post_analysis_setting, "['branch', $event]"),
                )
            with html.Div(classes="radar-button-row"):
                vuetify.VCheckbox(
                    label="Connect within each path",
                    v_model=("post_analysis_editor.connect",),
                    density="compact",
                    update_modelValue=(self.update_post_analysis_setting, "['connect', $event]"),
                )
                vuetify.VCheckbox(
                    label="Show available uncertainties",
                    v_model=("post_analysis_editor.show_uncertainty",),
                    density="compact",
                    update_modelValue=(
                        self.update_post_analysis_setting,
                        "['show_uncertainty', $event]",
                    ),
                )
                vuetify.VCheckbox(
                    label="Include fixed model values (exploratory)",
                    v_model=("post_analysis_editor.include_fixed",),
                    density="compact",
                    update_modelValue=(
                        self.update_post_analysis_setting,
                        "['include_fixed', $event]",
                    ),
                )
            with vuetify.VExpansionPanels(variant="accordion", classes="my-2"):
                with vuetify.VExpansionPanel(title="Scientific definitions and additional inputs"):
                    with vuetify.VExpansionPanelText():
                        html.P(
                            "A mapped axis needs a reviewed CSV: axis, run_id, sample_id, value, unit, source, kind, verified. Use sample_id for sample_state; exact run_id for other axes. kind is measured, nominal_composition, preparation_history or user_supplied. verified must be true.",
                            classes="radar-section-help",
                        )
                        NamedFileUpload(
                            "post_analysis_mapping_path",
                            label="Reviewed axis mapping CSV",
                            help_text="Acquisition metadata remains the default. A run mapping must represent that scan's acquisition interval.",
                            extensions=[".csv"],
                            key="post-analysis-axis-mapping",
                        )
                        vuetify.VBtn(
                            "Import axis mapping",
                            click=self.import_post_analysis_mapping,
                            variant="outlined",
                            size="small",
                        )
                        vuetify.VBtn(
                            "Clear imported mappings",
                            click=self.clear_post_analysis_mapping,
                            variant="text",
                            size="small",
                        )
                        with html.Div(
                            v_if="post_analysis_editor.kind === 'reaction_progress'",
                            classes="radar-field-pair",
                        ):
                            for field, label in [
                                ("progress_start", "Progress start value"),
                                ("progress_end", "Progress end value"),
                            ]:
                                self._number_field(
                                    label=label,
                                    v_model=(f"post_analysis_editor.{field}",),
                                    density="compact",
                                    variant="outlined",
                                    update_modelValue=(
                                        self.update_post_analysis_setting,
                                        f"['{field}', $event]",
                                    ),
                                )
                            vuetify.VTextField(
                                label="Time zero (ISO timestamp with timezone)",
                                v_model=("post_analysis_editor.time_zero",),
                                density="compact",
                                variant="outlined",
                                update_modelValue=(
                                    self.update_post_analysis_setting,
                                    "['time_zero', $event]",
                                ),
                            )
                        self._number_field(
                            v_if="post_analysis_editor.kind === 'transition_hysteresis'",
                            label="Candidate crossing threshold",
                            v_model=("post_analysis_editor.threshold",),
                            density="compact",
                            variant="outlined",
                            update_modelValue=(
                                self.update_post_analysis_setting,
                                "['threshold', $event]",
                            ),
                        )
                        with html.Div(v_if="post_analysis_editor.kind === 'lattice_strain'"):
                            vuetify.VTextField(
                                label="h k l (for d_hkl)",
                                v_model=("post_analysis_hkl",),
                                density="compact",
                                variant="outlined",
                                update_modelValue=(self.update_post_analysis_hkl, "[$event]"),
                            )
                        with html.Div(v_if="post_analysis_editor.kind === 'thermal_expansion'"):
                            vuetify.VCheckbox(
                                label="Fit expansion coefficient on each qualified branch",
                                v_model=("post_analysis_editor.fit_expansion",),
                                density="compact",
                                update_modelValue=(
                                    self.update_post_analysis_setting,
                                    "['fit_expansion', $event]",
                                ),
                            )
                            for field, label in [
                                ("fit_min", "Fit minimum temperature (K)"),
                                ("fit_max", "Fit maximum temperature (K)"),
                                (
                                    "max_temperature_span",
                                    "Maximum within-scan temperature range (K)",
                                ),
                            ]:
                                self._number_field(
                                    label=label,
                                    v_model=(f"post_analysis_editor.{field}",),
                                    density="compact",
                                    variant="outlined",
                                    update_modelValue=(
                                        self.update_post_analysis_setting,
                                        f"['{field}', $event]",
                                    ),
                                )
                        vuetify.VCombobox(
                            label="Runs excluded from this calculation",
                            v_model=("post_analysis_editor.excluded_runs",),
                            items=("post_analysis_reference_options",),
                            multiple=True,
                            chips=True,
                            density="compact",
                            variant="outlined",
                            update_modelValue=(
                                self.update_post_analysis_setting,
                                "['excluded_runs', $event]",
                            ),
                        )
            with html.Div(
                v_for="notice in post_analysis_notices", key="notice", classes="radar-section-help"
            ):
                html.Span("{{ notice }}")
            with html.Div(
                v_for="postView in post_analysis_panels", key="postView.id", classes="my-4"
            ):
                html.H3("{{ postView.title }}")
                html.P(
                    "{{ postView.status }} · {{ postView.point_count }} eligible observations",
                    classes="radar-section-help",
                )
                with html.Div(
                    v_for="notice in postView.notices", key="notice", classes="radar-section-help"
                ):
                    html.Span("{{ notice }}")
                vuetify.VBtn(
                    "Edit this view",
                    click=(self.select_post_analysis_view, "[postView.id]"),
                    variant="text",
                    size="small",
                )
                with html.Div(v_show="postView.point_count > 0", classes="radar-plot-frame"):
                    figure = plotly.Figure(
                        display_mode_bar=True,
                        click=(self.select_post_analysis_point, "[$event]"),
                        relayout=(self.relayout_post_analysis, "[postView.id, $event]"),
                    )
                    figure.data = ("postView.figure.data",)
                    figure.layout = ("postView.figure.layout",)
                vuetify.VDataTable(
                    v_if="postView.fits.length",
                    items=("postView.fits",),
                    headers=(
                        [
                            {"title": "Series / branch", "key": "label"},
                            {"title": "α (K⁻¹)", "key": "alpha"},
                            {"title": "Conditional s.u.", "key": "alpha_esd"},
                            {"title": "Fit range (K)", "key": "fit_interval"},
                        ],
                    ),
                    density="compact",
                )
                vuetify.VDataTable(
                    v_if="postView.crossings.length",
                    items=("postView.crossings",),
                    headers=(
                        [
                            {"title": "Branch", "key": "branch"},
                            {"title": "Candidate interval", "key": "interval"},
                            {"title": "Bracketing scans", "key": "runs"},
                            {"title": "Threshold", "key": "threshold"},
                        ],
                    ),
                    density="compact",
                )
            with html.Div(classes="radar-button-row"):
                vuetify.VBtn(
                    "Download views and observations",
                    click=self.download_post_analysis,
                    variant="outlined",
                    size="small",
                )
                vuetify.VBtn(
                    "Save view settings",
                    click=self.save_post_analysis_settings,
                    variant="outlined",
                    size="small",
                )
                vuetify.VBtn(
                    "Choose new automatic reference",
                    click=self.reset_post_analysis_reference,
                    variant="text",
                    size="small",
                )

    def _post_settings(self):
        return PostAnalysisSettings.model_validate(self.server.state.post_analysis)

    def _set_post_settings(self, settings):
        state = self.server.state
        state.post_analysis = settings.model_dump(mode="json")
        state.post_analysis_selected = list(dict.fromkeys(v.kind for v in settings.views))
        if getattr(state, "post_analysis_active", None) not in {v.id for v in settings.views}:
            state.post_analysis_active = settings.views[0].id if settings.views else ""
        self._select_post_analysis_editor()

    def _select_post_analysis_editor(self):
        state = self.server.state
        settings = self._post_settings()
        view = next((v for v in settings.views if v.id == state.post_analysis_active), None)
        state.post_analysis_view_options = [
            {"title": RECIPES[v.kind][0] + f" · {v.id}", "value": v.id} for v in settings.views
        ]
        state.post_analysis_editor = view.model_dump(mode="json") if view else {}
        if view and not view.metric:
            state.post_analysis_editor = {
                **state.post_analysis_editor,
                "metric": RECIPES[view.kind][1],
            }
        state.post_analysis_hkl = " ".join(map(str, view.hkl)) if view else "1 0 0"
        kind = view.kind if view else "phase_evolution"
        state.post_analysis_metric_options = (
            ["weight_percent"]
            if kind == "phase_evolution"
            else (
                ["c/a", "b/a", "orthorhombicity"]
                if kind == "distortion"
                else (
                    ["a", "b", "c", "alpha", "beta", "gamma", "volume"]
                    if kind == "thermal_expansion"
                    else (
                        ["a", "b", "c", "volume", "d_hkl"]
                        if kind == "lattice_strain"
                        else ["weight_percent", "a", "b", "c", "volume", "c/a", "orthorhombicity"]
                    )
                )
            )
        )
        reps = ["auto", "absolute", "difference", "relative"]
        if kind == "reaction_progress":
            reps = ["auto", "progress"]
        if kind == "lattice_strain":
            reps += ["microstrain"]
        state.post_analysis_representation_options = reps

    def update_post_analysis_selection(self, selected=None, **_):
        state = self.server.state
        selected = (
            selected if isinstance(selected, list) else list(state.post_analysis_selected or [])
        )
        try:
            if any(kind not in RECIPES for kind in selected):
                raise ValueError("Unknown post-analysis recipe")
            settings = self._post_settings()
            views = [v for v in settings.views if v.kind in selected]
            for kind in dict.fromkeys(selected):
                if not any(v.kind == kind for v in views):
                    views.append(ViewSettings(id=kind, kind=kind))
            self._set_post_settings(
                PostAnalysisSettings.model_validate(
                    {**settings.model_dump(), "views": [v.model_dump() for v in views]}
                )
            )
            self._refresh_post_analysis()
        except ValueError as exc:
            state.post_analysis_error = str(exc)
        state.flush()

    def select_post_analysis_view(self, value=None, **_):
        self.server.state.post_analysis_active = str(value or "")
        self._select_post_analysis_editor()
        self._refresh_post_analysis()
        self.server.state.flush()

    def duplicate_post_analysis_view(self, **_):
        settings = self._post_settings()
        current = next(
            (v for v in settings.views if v.id == self.server.state.post_analysis_active), None
        )
        if current is None:
            return
        used = {v.id for v in settings.views}
        index = 2
        while f"{current.kind}-{index}" in used:
            index += 1
        duplicate = current.model_copy(update={"id": f"{current.kind}-{index}"})
        payload = settings.model_dump()
        payload["views"].append(duplicate.model_dump())
        try:
            validated = PostAnalysisSettings.model_validate(payload)
            self.server.state.post_analysis_active = duplicate.id
            self._set_post_settings(validated)
            self._refresh_post_analysis()
        except ValueError as exc:
            self.server.state.post_analysis_error = str(exc)
        self.server.state.flush()

    def update_post_analysis_setting(self, field, value, **_):
        state = self.server.state
        try:
            settings = self._post_settings()
            current = next(v for v in settings.views if v.id == state.post_analysis_active)
            if field not in ViewSettings.model_fields or field in {"id", "kind"}:
                raise ValueError("Unsupported setting")
            if value == "" and field in {
                "reference_run",
                "progress_start",
                "progress_end",
                "time_zero",
                "threshold",
                "fit_min",
                "fit_max",
            }:
                value = None
            changed = ViewSettings.model_validate({**current.model_dump(), field: value})
            payload = settings.model_dump()
            payload["views"] = [
                changed.model_dump() if v.id == current.id else v.model_dump()
                for v in settings.views
            ]
            self._set_post_settings(PostAnalysisSettings.model_validate(payload))
            state.post_analysis_error = ""
            self._refresh_post_analysis()
        except (ValueError, StopIteration) as exc:
            state.post_analysis_error = str(exc)
            self._select_post_analysis_editor()
        state.flush()

    def remove_post_analysis_view(self, **_):
        settings = self._post_settings()
        active = self.server.state.post_analysis_active
        payload = settings.model_dump()
        payload["views"] = [v for v in payload["views"] if v["id"] != active]
        payload["references"] = {
            key: value
            for key, value in payload["references"].items()
            if not key.startswith(active + ":")
        }
        self._set_post_settings(PostAnalysisSettings.model_validate(payload))
        self._refresh_post_analysis()
        self.server.state.flush()

    def update_post_analysis_hkl(self, value, **_):
        try:
            hkl = [int(v) for v in str(value).replace(",", " ").split()]
            if len(hkl) != 3:
                raise ValueError("Enter three integer Miller indices")
            self.update_post_analysis_setting("hkl", hkl)
        except ValueError as exc:
            self.server.state.post_analysis_error = str(exc)
            self.server.state.flush()

    def _refresh_post_analysis(self, scans=None, pending=None):
        if not hasattr(self, "_post_analysis_scans"):
            return
        if scans is not None:
            self._post_analysis_scans = list(scans)
        if pending is not None:
            self._post_analysis_pending = list(pending)
        settings = self._post_settings()
        result = calculate_views(
            self._post_analysis_scans, settings, pending=getattr(self, "_post_analysis_pending", [])
        )
        settings.references = result["references"]
        settings.phase_colors = result["phase_colors"]
        self.server.state.post_analysis = settings.model_dump(mode="json")
        result["settings"] = settings.model_dump(mode="json")
        self._post_analysis_result = result
        controller = getattr(self, "_powgen_controller", None)
        if controller is not None and str(getattr(self, "_post_analysis_scope", "")).startswith(
            "watch:"
        ):
            controller.state.post_analysis = settings.model_dump(mode="json")
        state = self.server.state
        by_id = {view.id: view for view in settings.views}
        state.post_analysis_panels = [
            {
                "id": v["id"],
                "title": v["title"],
                "status": v["status"].replace("_", " "),
                "notices": v["notices"],
                "point_count": sum(
                    p["value"] is not None and p["x"] is not None and not p.get("display_hidden")
                    for s in v["series"]
                    for p in s["points"]
                ),
                "figure": plotly.Figure.to_data(view_figure(v, by_id[v["id"]])),
                "fits": [
                    {**fit, "label": fit["branch"] + " · " + fit["series"][:8]} for fit in v["fits"]
                ],
                "crossings": v["crossings"],
            }
            for v in result["views"]
        ]
        selected = next((v for v in result["views"] if v["id"] == state.post_analysis_active), None)
        self._select_post_analysis_editor()
        state.post_analysis_status = (
            selected["status"].replace("_", " ").capitalize()
            if selected
            else "No post-analysis view selected"
        )
        state.post_analysis_notices = selected["notices"] if selected else []
        phases = {s["phase_key"]: s["phase"] for v in result["views"] for s in v["series"]}
        state.post_analysis_phase_options = [
            {"title": name, "value": key} for key, name in phases.items()
        ]
        state.post_analysis_reference_options = [
            str(s.get("run_id") or s.get("run_name") or s.get("run_number"))
            for s in self._post_analysis_scans
        ]
        state.post_analysis_branch_options = [
            "all",
            *dict.fromkeys(
                p["branch"] for s in (selected or {}).get("series", []) for p in s["points"]
            ),
        ]
        state.post_analysis_fit_rows = [
            {**fit, "label": fit["branch"] + " · " + fit["series"][:8]}
            for fit in (selected or {}).get("fits", [])
        ]
        if selected:
            view = next(v for v in settings.views if v.id == selected["id"])
            for widget in self._post_analysis_widgets:
                widget.update(view_figure(selected, view))
        else:
            import plotly.graph_objects as go

            for widget in self._post_analysis_widgets:
                widget.update(go.Figure())

    def import_post_analysis_mapping(self, **_):
        state = self.server.state
        try:
            path = Path(state.post_analysis_mapping_path or "")
            if not path.is_file() or path.stat().st_size > 5_000_000:
                raise ValueError("Choose a mapping CSV smaller than 5 MB")
            settings = self._post_settings()
            payload = settings.model_dump()
            mapping = {axis: dict(values) for axis, values in payload["axis_mappings"].items()}
            with path.open(encoding="utf-8-sig", newline="") as handle:
                for index, row in enumerate(csv.DictReader(handle), 2):
                    if index > 10002:
                        raise ValueError("Mapping exceeds 10,000 rows")
                    axis = str(row.get("axis") or "").strip()
                    key = str(
                        row.get("sample_id" if axis == "sample_state" else "run_id") or ""
                    ).strip()
                    if not key or axis not in AXES:
                        raise ValueError(
                            f"Mapping row {index}: missing identifier or unsupported axis"
                        )
                    if str(row.get("verified") or "").strip().lower() != "true":
                        raise ValueError(
                            f"Mapping row {index}: review the value and set verified=true"
                        )
                    value = str(row.get("value") or "").strip()
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                    entry = {
                        "value": value,
                        "unit": str(row.get("unit") or "").strip(),
                        "source": str(row.get("source") or "").strip(),
                        "kind": row.get("kind"),
                        "verified": True,
                        "branch": str(row.get("branch") or "").strip() or None,
                        "cycle": str(row.get("cycle") or "").strip() or None,
                    }
                    if key in mapping.get(axis, {}) and mapping[axis][key] != entry:
                        raise ValueError(
                            f"Mapping row {index}: conflicting duplicate; remove the previous mapping before replacing it"
                        )
                    mapping.setdefault(axis, {})[key] = entry
            payload["axis_mappings"] = mapping
            self._set_post_settings(PostAnalysisSettings.model_validate(payload))
            state.post_analysis_error = ""
            self._refresh_post_analysis()
        except (ValueError, OSError) as exc:
            state.post_analysis_error = str(exc)
        state.flush()

    def download_post_analysis(self, **_):
        self._refresh_post_analysis()
        self.download_file(
            "radar-pd-post-analysis.zip",
            "application/zip",
            export_bundle(
                self._post_analysis_result, self._post_settings(), self._post_analysis_scans
            ),
        )

    def clear_post_analysis_mapping(self, **_):
        settings = self._post_settings()
        settings.axis_mappings = {}
        self._set_post_settings(settings)
        self._refresh_post_analysis()
        self.server.state.flush()

    def reset_post_analysis_reference(self, **_):
        settings = self._post_settings()
        active = self.server.state.post_analysis_active
        settings.references = {
            key: value
            for key, value in settings.references.items()
            if not key.startswith(active + ":")
        }
        payload = settings.model_dump()
        for view in payload["views"]:
            if view["id"] == active:
                view["reference_run"] = None
        self._set_post_settings(PostAnalysisSettings.model_validate(payload))
        self._refresh_post_analysis()
        self.server.state.flush()

    def select_post_analysis_point(self, payload, **_):
        if not str(getattr(self, "_post_analysis_scope", "")).startswith("watch:"):
            return
        points = payload.get("points") if isinstance(payload, dict) else None
        if points and isinstance(points[0].get("customdata"), (list, tuple)):
            run_id = str(points[0]["customdata"][0])
            if any(str(scan.get("run_id")) == run_id for scan in self._post_analysis_scans):
                self.select_powgen_scan(run_id)

    def relayout_post_analysis(self, view_id, event, **_):
        """Persist visible ranges; link x zoom only for compatible physical axes."""
        if not isinstance(event, dict):
            return
        settings = self._post_settings()
        payload = settings.model_dump()
        current = next((v for v in payload["views"] if v["id"] == view_id), None)
        if current is None:
            return
        changed = False
        for axis, field in [("xaxis", "x_range"), ("yaxis", "y_range")]:
            value = (
                [event[axis + ".range[0]"], event[axis + ".range[1]"]]
                if axis + ".range[0]" in event and axis + ".range[1]" in event
                else event.get(axis + ".range")
            )
            if event.get(axis + ".autorange") is True:
                value = None
            elif value is None:
                continue
            if value is not None and (not isinstance(value, (list, tuple)) or len(value) != 2):
                continue
            if (
                current[field] is not None
                and value is not None
                and list(current[field]) == list(value)
            ):
                continue
            if current[field] is None and value is None:
                continue
            current[field] = value
            changed = True
            if field == "x_range":
                results = {v["id"]: v for v in self._post_analysis_result["views"]}
                source = results.get(view_id, {})
                signature = lambda v: (
                    v.get("axis"),
                    {
                        (s.get("axis_unit"), p.get("axis_kind"))
                        for s in v.get("series", [])
                        for p in s["points"]
                        if p["x"] is not None
                    },
                )
                if source.get("axis") not in {"run_number", "sample_state"}:
                    for other in payload["views"]:
                        if other["id"] != view_id and signature(
                            results.get(other["id"], {})
                        ) == signature(source):
                            other[field] = value
        if changed:
            try:
                self._set_post_settings(PostAnalysisSettings.model_validate(payload))
                self._refresh_post_analysis()
                self.server.state.flush()
            except ValueError as exc:
                self.server.state.post_analysis_error = str(exc)
                self.server.state.flush()

    async def save_post_analysis_settings(self, **_):
        try:
            controller = getattr(self, "_powgen_controller", None)
            if controller is not None and str(getattr(self, "_post_analysis_scope", "")).startswith(
                "watch:"
            ):
                controller.state.post_analysis = self._post_settings().model_dump(mode="json")
                await asyncio.to_thread(controller.persist_state)
                self.server.state.notice = (
                    "Saved post-analysis settings and references to the watch checkpoint."
                )
            else:
                self.save_current_configuration()
        except Exception as exc:
            self.server.state.post_analysis_error = f"View settings were not saved: {exc}"
        self.server.state.flush()
