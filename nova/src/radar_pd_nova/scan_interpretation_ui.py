"""Compatibility readers for archived scan-interpretation reports.

The old standalone widgets are no longer mounted in the live application.
Transition tracking reuses the measured-receipt reader and label helpers.
"""

from __future__ import annotations

import asyncio
import io
import json
import math
from pathlib import Path
import zipfile

from trame.widgets import html, plotly, vuetify3 as vuetify

from .scan_interpretation import InterpretationSettings, ScanInterpreter, digest
from .scan_interpretation_plot import comparison_figure, observation_figure
from .results import phase_fraction_rows


def scan_label(scan):
    """Use acquisition labels in controls; preserve the exact ID in receipts."""
    number = scan.get("run_number")
    bank = scan.get("bank")
    return (
        f"Run {number}" + (f" · bank {bank}" if bank is not None else "")
        if number is not None
        else str(scan.get("run_id", ""))
    )


def phase_labels(*scans):
    """Readable labels never merge distinct candidate identities."""
    names = {}
    for scan in scans:
        phases = [phase for phase in (scan.get("observables") or {}).get("phases", [])
                  if isinstance(phase, dict)]
        normalized = phase_fraction_rows({"phases": phases, "lattice_parameters": phases})
        for phase, label in zip(phases, normalized):
            key = str(phase.get("phase_id", ""))
            group = label["space_group"]
            names[key] = (f"{label['phase']} (SG {group})" if group != "-"
                          else str(phase.get("phase") or key))
    duplicates = {name for name in names.values() if list(names.values()).count(name) > 1}
    return {
        key: f"{name} · {key}" if name in duplicates else name
        for key, name in names.items()
    }


FINDING_LABELS = {
    "peak_movement": "Peak positions",
    "new_features": "Unmatched current features",
    "lost_features": "Unmatched earlier features",
    "profile_shape_change": "Peak shapes",
    "relative_intensity_redistribution": "Relative peak intensities",
    "global_scale_change": "Overall intensity",
    "background_change": "Background difference",
    "persistent_unexplained_features": "Residual features in both fits",
    "unresolved_pattern_change": "Unexplained pattern difference",
    "no_resolved_event": "No change resolved",
    "refined_cell_change": "Lattice parameters",
    "modeled_fraction_change": "Modeled phase fractions",
    "candidate_assignment_changed": "Candidate selection changed",
}
MODEL_FINDINGS = {
    "refined_cell_change", "modeled_fraction_change", "candidate_assignment_changed"
}


def model_change_rows(model, labels, settings):
    """Display equal cell lengths together without changing the stored metrics."""
    groups = []
    for cell in model.get("cell_changes", []):
        if abs(cell["relative_change"]) < settings.cell_fraction:
            continue
        match = next((g for g in groups if
            cell["metric"] in {"a", "b", "c"}
            and set(g["metrics"]) <= {"a", "b", "c"}
            and g["phase_id"] == cell["phase_id"]
            and all(math.isclose(g[k], cell[k], rel_tol=1e-12, abs_tol=1e-12)
                    for k in ("previous", "current"))), None)
        if match:
            match["metrics"].append(cell["metric"])
        else:
            groups.append({**cell, "metrics": [cell["metric"]]})
    rows = [
        {"label": labels.get(g["phase_id"], g["phase_id"]) + " · " + " = ".join(g["metrics"]),
         "change": f"{100*g['relative_change']:+.3g}%"}
        for g in groups
    ]
    rows.extend(
        {"label": labels.get(c["phase_id"], c["phase_id"]) + " · modeled fraction",
         "change": f"{c['percentage_point_difference']:+.3g} pp"}
        for c in model.get("fraction_changes", [])
        if abs(c["percentage_point_difference"]) >= settings.fraction_points
    )
    return rows


def compact_report(report):
    """Checkpoint the decisions and receipt hashes, not redundant curve arrays."""
    result = {k: v for k, v in report.items() if k != "records"}
    result["records"] = []
    for row in report.get("records", []):
        item = {
            k: v
            for k, v in row.items()
            if k not in {"comparisons", "condition_receipt", "conditions"}
        }
        item["condition_metadata_sha256"] = row.get("condition_receipt", {}).get("metadata_sha256")
        item["comparisons"] = []
        for pair in row["comparisons"]:
            features = pair["features"]
            item["comparisons"].append(
                {
                    **{k: v for k, v in pair.items() if k not in {"features", "condition_changes"}},
                    "features": {
                        k: (len(v) if isinstance(v, list) else v) for k, v in features.items()
                    },
                }
            )
        result["records"].append(item)
    result["source_evidence_sha256"] = result.pop("evidence_sha256", None)
    result["source_schema"] = result.get("$schema")
    result["$schema"] = "radar-pd-scan-interpretation-index/v1"
    result["checkpoint_representation"] = (
        "compact evidence index; replay from immutable curve and condition receipts"
    )
    result["evidence_sha256"] = digest(result)
    return result


class ScanInterpretationMixin:
    def _init_scan_interpretation(self):
        self._interpretation_generation = getattr(self, "_interpretation_generation", 0) + 1
        self._interpretation_task = None
        self._scan_interpreter = ScanInterpreter()
        self._interpretation_scans = []
        self._interpretation_all_watch_scans = []
        self._interpretation_report = {"records": []}
        self._interpretation_signature = None
        state = self.server.state
        state.scan_interpretation = InterpretationSettings().model_dump(mode="json")
        for key, value in {
            "scan_change_run": "",
            "scan_change_runs": [],
            "scan_change_pair": "",
            "scan_change_pairs": [],
            "scan_change_summary": "Waiting for measured-curve observations",
            "scan_change_condition_summary": "",
            "scan_change_findings": [],
            "scan_change_notices": [],
            "scan_change_conditions": [],
            "scan_change_qualifications": [],
            "scan_change_metrics": [],
            "scan_change_details": [],
            "scan_change_regions": [],
            "scan_change_region": "full",
            "scan_change_figure": {"data": [], "layout": {}},
            "scan_change_can_plot": False,
            "scan_change_has_curve": False,
            "scan_change_error": "",
            "scan_change_open": False,
            "scan_change_fit": True,
            "scan_change_binding_options": [],
            "scan_change_binding": "",
            "scan_change_binding_confirmed": False,
            "scan_change_revision_options": [],
            "scan_change_revision": "",
            "scan_change_context": "",
            "scan_change_evidence_tab": "pattern",
            "scan_change_items": [],
            "scan_change_finding": "",
            "scan_change_selected_evidence": {},
            "scan_change_model_rows": [],
            "scan_change_model_notice": "",
            "scan_change_review_required": False,
            "experiment_result_view": "results",
        }.items():
            setattr(state, key, value)

    def _scan_change_dashboard(self):
        with html.Section(classes="radar-result-card radar-scan-workspace my-3"):
            with html.Div(classes="radar-scan-heading"):
                with html.Div():
                    html.H3("Scan comparison", classes="radar-card-title")
                    html.P("{{ scan_change_context }}", classes="radar-section-help")
                vuetify.VBtn(
                    "Export comparison", variant="text", size="small",
                    disabled=("!scan_change_can_plot",), click=self.download_scan_change,
                )
            with html.Div(classes="radar-scan-selectors"):
                vuetify.VAutocomplete(
                    label="Scan / bank", v_model=("scan_change_run",),
                    items=("scan_change_runs",), density="compact", variant="outlined",
                    hide_details=True, update_modelValue=(self.select_scan_change, "[$event]"),
                )
                vuetify.VSelect(
                    label="Reference", v_model=("scan_change_pair",),
                    items=("scan_change_pairs",), density="compact", variant="outlined",
                    hide_details=True, disabled=("!scan_change_pairs.length",),
                    update_modelValue=(self.select_scan_change_pair, "[$event]"),
                )
            html.P(
                "{{ scan_change_condition_summary }}", v_if="scan_change_condition_summary",
                classes="radar-scan-conditions-line",
            )
            html.P("{{ scan_change_summary }}", v_if="scan_change_has_curve && !scan_change_can_plot", classes="radar-section-help mb-3")
            with html.Div(v_if="scan_change_has_curve", classes="radar-scan-evidence-grid"):
                self._scan_change_curve_view()
                self._scan_change_evidence_view()
            with html.Div(v_else=True, classes="radar-scan-empty", role="status"):
                html.H4("Comparison unavailable")
                html.P("{{ scan_change_summary }}")
            self._scan_change_settings()

    def _scan_change_curve_view(self, *, expanded=False):
        with html.Div(classes="radar-scan-curve"):
            with html.Div(classes="radar-scan-curve-controls"):
                vuetify.VAutocomplete(
                    label="Pattern region", v_model=("scan_change_region",),
                    items=("scan_change_regions",), variant="outlined", density="compact",
                    hide_details=True,
                    update_modelValue=(self.select_scan_change_region, "[$event]"),
                )
                vuetify.VCheckbox(
                    label="Calculated fit", v_model=("scan_change_fit",), hide_details=True,
                    density="compact",
                    update_modelValue=(self.toggle_scan_change_fit, "[$event]"),
                )
                if not expanded:
                    vuetify.VBtn(
                        "Expand", variant="text", size="small", click=self.open_scan_change,
                        prepend_icon="mdi-arrow-expand",
                    )
            with html.Div(classes="radar-scan-chart" + (" is-expanded" if expanded else "")):
                figure = plotly.Figure(display_mode_bar=True, style="height:100%;width:100%")
                figure.data = ("scan_change_figure.data",)
                figure.layout = ("scan_change_figure.layout",)
            html.P(
                "{{ scan_change_can_plot ? 'Reference intensity is adjusted by one scale and offset. The difference shows current minus adjusted reference.' : 'Measured pattern and current refinement. A compatible reference is not yet available.' }}",
                classes="radar-section-help radar-scan-caption",
            )

    def _scan_change_evidence_view(self):
        with html.Aside(classes="radar-scan-evidence", aria_label="Comparison evidence"):
            with vuetify.VTabs(
                v_model=("scan_change_evidence_tab",), density="compact",
                classes="radar-scan-evidence-tabs", color="#15543c",
            ):
                vuetify.VTab("Pattern", value="pattern")
                vuetify.VTab("Refinement", value="model")
                vuetify.VTab("Conditions", value="conditions")
            with html.Div(v_show="scan_change_evidence_tab === 'pattern'"):
                html.P("Automated observations · select to inspect", v_if="scan_change_can_plot", classes="radar-section-help mb-2")
                with html.Div(classes="radar-scan-findings"):
                    with html.Button(
                        v_for="finding in scan_change_items.filter(f => f.group === 'pattern')",
                        key="finding.code", type="button", classes="radar-scan-finding",
                        aria_pressed=("scan_change_finding === finding.code",),
                        click=(self.select_scan_change_finding, "[finding.code]"),
                    ):
                        html.Strong("{{ finding.label }}")
                        html.Span("{{ finding.summary }}")
                        html.Small("{{ finding.region_count ? finding.region_count + (finding.region_count === 1 ? ' region' : ' regions') + ' · inspect curves' : 'Whole-pattern observation' }}")
                html.P(
                    "No local change was resolved by these checks. This does not establish that the sample is unchanged.",
                    v_if="scan_change_can_plot && !scan_change_items.some(f => f.group === 'pattern')",
                    classes="radar-section-help",
                )
                html.P("Another compatible measured curve is needed for a comparison.", v_if="!scan_change_can_plot", classes="radar-section-help my-2")
            with html.Div(v_show="scan_change_evidence_tab === 'model'", classes="radar-scan-model"):
                html.P("{{ scan_change_model_notice }}", classes="radar-section-help my-2")
                vuetify.VDataTable(
                    items=("scan_change_model_rows",),
                    headers=([{"title":"Candidate / property", "key":"label"},
                              {"title":"Change", "key":"change"}],),
                    density="compact", items_per_page=-1, hide_default_footer=True,
                    no_data_text="No comparable refined change above the display threshold.",
                )
                with html.Div(
                    v_for="finding in scan_change_items.filter(f => f.group === 'model')",
                    key="finding.code", classes="radar-scan-model-note",
                ):
                    html.Strong("{{ finding.label }}")
                    html.P("{{ finding.alternatives.join(' ') }}", classes="radar-section-help")
                    with html.Div(v_if="finding.code === 'candidate_assignment_changed'"):
                        html.P("{{ finding.evidence.join(' · ') }}")
            with html.Div(v_show="scan_change_evidence_tab === 'conditions'"):
                html.P("{{ scan_change_condition_summary }}", classes="radar-section-help my-2")
                with html.Ul(classes="radar-scan-notes"):
                    html.Li("{{ condition }}", v_for="condition in scan_change_conditions", key="condition")
                    html.Li("{{ notice }}", v_for="notice in scan_change_qualifications", key="notice")
                    html.Li("{{ notice }}", v_for="notice in scan_change_notices", key="notice")
                html.P("Sensor sources and device mapping are under Metadata and comparison settings.", classes="radar-section-help")
            with html.Div(
                v_if="scan_change_evidence_tab === 'pattern' && scan_change_selected_evidence.code",
                classes="radar-scan-selected-evidence", aria_live="polite",
            ):
                html.H4("{{ scan_change_selected_evidence.label }}")
                html.P("{{ scan_change_selected_evidence.evidence.join(' · ') }}")
                with html.Ul(classes="radar-scan-notes"):
                    html.Li(
                        "{{ alternative }}", v_for="alternative in scan_change_selected_evidence.alternatives",
                        key="alternative",
                    )
            html.P(
                "Fit or calibration review is pending.", v_if="scan_change_review_required",
                classes="radar-scan-review-note",
            )

    def _scan_change_settings(self):
        with vuetify.VExpansionPanels(variant="accordion", classes="radar-scan-settings mt-3"):
            with vuetify.VExpansionPanel(
                title="Metadata and comparison settings"
            ):
                with vuetify.VExpansionPanelText():
                    html.P(
                        "Recorded conditions describe available sensors. Matching these does not establish that every specimen condition is controlled.",
                        classes="radar-section-help",
                    )
                    with html.Div(
                        v_for="condition in scan_change_conditions",
                        key="condition",
                        classes="mb-1",
                    ):
                        html.Span("{{ condition }}")
                    with html.Div(
                        v_for="notice in scan_change_qualifications",
                        key="notice",
                        classes="mb-1",
                    ):
                        html.Span("{{ notice }}")
                    with html.Div(
                        v_for="notice in scan_change_notices", key="notice", classes="mb-1"
                    ):
                        html.Span("{{ notice }}")
                    with html.Div(
                        v_for="metric in scan_change_metrics", key="metric", classes="mb-1"
                    ):
                        html.Span("{{ metric }}")
                    with html.Div(v_if="scan_change_binding_options.length"):
                        vuetify.VSelect(
                            label="Specimen-connected readback",
                            v_model=("scan_change_binding",),
                            items=("scan_change_binding_options",),
                            density="compact",
                            variant="outlined",
                            hide_details=True,
                        )
                        vuetify.VCheckbox(
                            label="I verified that this device measured this specimen during these runs",
                            v_model=("scan_change_binding_confirmed",),
                            hide_details=True,
                        )
                        vuetify.VBtn(
                            "Apply verified device mapping",
                            size="small",
                            variant="outlined",
                            click=self.bind_scan_change_condition,
                            disabled=(
                                "!scan_change_binding || !scan_change_binding_confirmed",
                            ),
                        )
                    with html.Div(v_if="scan_change_revision_options.length"):
                        vuetify.VSelect(
                            label="Active fit revision",
                            v_model=("scan_change_revision",),
                            items=("scan_change_revision_options",),
                            density="compact",
                            variant="outlined",
                            hide_details=True,
                        )
                        vuetify.VBtn(
                            "Use this fit revision",
                            size="small",
                            variant="outlined",
                            click=self.choose_scan_change_revision,
                            disabled=("!scan_change_revision",),
                        )
                    html.P(
                        "Feature thresholds are engineering settings; event sensitivity and false-alert rates are not certified. No stopping or phase-absence decision is made.",
                        classes="radar-section-help mt-2",
                    )
                    vuetify.VTextField(
                        label="Shape difference threshold",
                        type="text",
                        inputmode="decimal",
                        model_value=("scan_interpretation.shape_effect",),
                        density="compact",
                        variant="outlined",
                        update_modelValue=(
                            self.update_scan_change_setting,
                            "['shape_effect',$event]",
                        ),
                    )
                    with html.Div(style="display:flex;gap:8px;flex-wrap:wrap"):
                        vuetify.VBtn(
                            "Use this scan as reference",
                            size="small",
                            variant="outlined",
                            click=self.pin_scan_change_reference,
                            disabled=("!scan_change_runs.length",),
                        )
                        vuetify.VBtn(
                            "Save interpretation settings",
                            size="small",
                            variant="outlined",
                            click=self.save_scan_change,
                        )
        vuetify.VAlert(
            v_if="scan_change_error",
            text=("scan_change_error",),
            type="error",
            variant="tonal",
            classes="mt-2",
        )

    def _scan_change_dialog(self):
        with vuetify.VDialog(v_model=("scan_change_open",), fullscreen=True, scrollable=True):
            with vuetify.VCard(classes="radar-scan-inspector"):
                with vuetify.VCardTitle(classes="radar-scan-heading"):
                    with html.Div():
                        html.Span("Scan comparison")
                        html.P("{{ scan_change_context }}", classes="radar-section-help")
                    vuetify.VBtn("Close", click="scan_change_open=false", variant="text")
                with vuetify.VCardText():
                    html.P("{{ scan_change_summary }}", classes="radar-section-help mb-3")
                    with html.Div(classes="radar-scan-evidence-grid"):
                        self._scan_change_curve_view(expanded=True)
                        self._scan_change_evidence_view()

    def _refresh_scan_interpretation(self, scans, pending=()):
        if not hasattr(self, "_scan_interpreter"):
            return
        settings = InterpretationSettings.model_validate(self.server.state.scan_interpretation)
        signature = digest(
            {
                "scans": scans,
                "pending": pending,
                "settings": settings.model_dump(),
                "scope": getattr(self, "_post_analysis_scope", ""),
            }
        )
        if signature == self._interpretation_signature:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            # One worker per app, with latest-request coalescing. The server
            # remains responsive while a long historical series is replayed.
            self._interpretation_request = (signature, list(scans), settings, list(pending))
            if getattr(self, "_interpretation_task", None) is None:
                self.server.state.scan_change_summary = "Comparing completed acquisitions…"
                self._interpretation_task = loop.create_task(
                    self._run_scan_interpretation(
                        self._interpretation_generation, self._scan_interpreter
                    )
                )
            return
        try:
            report = self._scan_interpreter.evaluate(scans, settings, pending=pending)
            self._apply_scan_interpretation_report(report, scans, pending)
        except (ValueError, TypeError, OSError) as exc:
            self.server.state.scan_change_error = str(exc)

    async def _run_scan_interpretation(self, generation, engine):
        task = asyncio.current_task()
        try:
            while generation == self._interpretation_generation:
                request = self._interpretation_request
                signature, scans, settings, pending = request
                report = await asyncio.to_thread(engine.evaluate, scans, settings, pending=pending)
                if generation != self._interpretation_generation:
                    return
                if self._interpretation_request[0] == signature:
                    self._apply_scan_interpretation_report(report, scans, pending)
                    self.server.state.flush()
                    return
        except (ValueError, TypeError, OSError) as exc:
            if generation == self._interpretation_generation:
                self.server.state.scan_change_error = str(exc)
                self.server.state.flush()
        finally:
            if getattr(self, "_interpretation_task", None) is task:
                self._interpretation_task = None

    def _apply_scan_interpretation_report(self, report, scans, pending):
        self._interpretation_scans = list(scans)
        self._interpretation_report = report
        self.server.state.scan_interpretation = report["settings"]
        self._interpretation_signature = digest(
            {
                "scans": scans,
                "pending": pending,
                "settings": report["settings"],
                "scope": getattr(self, "_post_analysis_scope", ""),
            }
        )
        choices = []
        by_id = {str(s.get("run_id")): s for s in scans}
        for row in report["records"]:
            scan = by_id.get(row["run_id"], {})
            metadata = scan.get("metadata") or {}
            choices.append(
                {
                    "title": f"{scan_label(scan)} · sample {metadata.get('sample_id','unknown')}",
                    "value": row["run_id"],
                }
            )
        state = self.server.state
        old_latest = (state.scan_change_runs or [{}])[-1].get("value")
        state.scan_change_runs = choices
        ids = [o["value"] for o in choices]
        # Follow arriving data only while the operator is looking at latest.
        if ids and (state.scan_change_run not in ids or state.scan_change_run == old_latest):
            state.scan_change_run = ids[-1]
        if not ids:
            state.scan_change_run = ""
        controller = getattr(self, "_powgen_controller", None)
        if controller is not None and str(getattr(self, "_post_analysis_scope", "")).startswith(
            "watch:"
        ):
            controller.state.scan_interpretation = report["settings"]
            controller.state.interpretation_report = compact_report(report)
            if callable(getattr(controller, "persist_state", None)):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop and not getattr(self, "_interpretation_save_task", None):
                    self._interpretation_save_task = loop.create_task(
                        self._persist_scan_change(controller)
                    )
        self.select_scan_change(state.scan_change_run)

    async def _persist_scan_change(self, controller):
        try:
            while True:
                before = controller.state.interpretation_report.get("evidence_sha256")
                await asyncio.to_thread(controller.persist_state)
                if before == controller.state.interpretation_report.get("evidence_sha256"):
                    break
        except Exception as exc:
            self.server.state.scan_change_error = f"Interpretation checkpoint was not saved: {exc}"
        finally:
            self._interpretation_save_task = None
            self.server.state.flush()

    def select_scan_change(self, value, **_):
        state = self.server.state
        state.scan_change_run = value
        row = next((r for r in self._interpretation_report["records"] if r["run_id"] == value), {})
        changed_acquisition = getattr(self, "_selected_interpretation", {}).get("run_id") != row.get("run_id")
        self._selected_interpretation = row
        from .scan_interpretation import _measurement, _revision

        current_scan = next((s for s in self._interpretation_scans
                             if str(s.get("run_id")) == str(value)), {})
        metadata = current_scan.get("metadata") or {}
        state.scan_change_context = " · ".join(str(v) for v in [
            {"NOM": "NOMAD", "PG3": "POWGEN"}.get(current_scan.get("instrument")),
            current_scan.get("ipts"),
            ("Sample " + str(metadata.get("sample_name") or metadata.get("sample_id")))
                if metadata.get("sample_name") or metadata.get("sample_id") else None,
            f"ID {metadata['sample_id']}" if metadata.get("sample_name") and metadata.get("sample_id") else None,
        ] if v)
        state.scan_change_review_required = bool(row.get("scientific_warnings"))

        revisions = [
            s for s in self._interpretation_scans if _measurement(s) == row.get("measurement_key")
        ]
        state.scan_change_revision_options = (
            [
                {"title": f"{s.get('run_id')} · GPX {_revision(s)[:16]}", "value": _revision(s)}
                for s in revisions
            ]
            if len({_revision(s) for s in revisions}) > 1
            else []
        )
        state.scan_change_revision = ""
        pairs = row.get("comparisons", [])
        state.scan_change_pairs = [
            {
                "title": scan_label(self._scan_by_id(p["reference_run_id"]))
                + " · "
                + ", ".join(r.replace("_", " ") for r in p["roles"]),
                "value": p["reference_run_id"],
            }
            for p in pairs
        ]
        ids = [p["reference_run_id"] for p in pairs]
        if changed_acquisition or state.scan_change_pair not in ids:
            state.scan_change_pair = ids[0] if ids else ""
        state.scan_change_notices = row.get("notices", [])
        state.scan_change_conditions = [
            f"{c['quantity']}: {c.get('value','unavailable'):.6g} {c.get('unit','')} · range {c.get('minimum','?')}–{c.get('maximum','?')} · {c['status']}"
            for c in row.get("conditions", [])
            if isinstance(c.get("value"), (float, int))
        ]
        state.scan_change_qualifications = [
            f"{c.get('channel')}: {c.get('status')} · " + "; ".join(c.get("notices", []))
            for c in row.get("condition_receipt", {}).get("channels", [])
            if c.get("status") not in {"recorded", "duplicate_alias"}
        ]
        state.scan_change_qualifications += [str(w) for w in row.get("scientific_warnings", [])]
        state.scan_change_binding_options = [
            {
                "title": f"{c['quantity']} · {c['channel']} ({c.get('unit','')})",
                "value": c["channel"],
            }
            for c in row.get("condition_receipt", {}).get("channels", [])
            if c.get("role") == "measured"
            and c.get("status") == "needs_mapping"
            and c.get("statistic") == "time_weighted_step_hold"
            and c.get("coverage_fraction", 0) >= 0.95
        ]
        state.scan_change_binding = ""
        state.scan_change_binding_confirmed = False
        self.select_scan_change_pair(state.scan_change_pair)

    def select_scan_change_pair(self, value, **_):
        state = self.server.state
        state.scan_change_pair = value
        row = getattr(self, "_selected_interpretation", {})
        pair = next((p for p in row.get("comparisons", []) if p["reference_run_id"] == value), {})
        self._selected_interpretation_pair = pair
        findings = []
        # Each visible observation must refer to the curves actually displayed.
        # The report also holds history/pinned-reference findings, which must
        # not be presented as measurements of the selected adjacent pair.
        if pair:
            from .scan_interpretation import explain_comparison

            before = self._scan_by_id(value)
            after = self._scan_by_id(state.scan_change_run)
            findings = explain_comparison(
                before,
                after,
                pair["features"],
                InterpretationSettings.model_validate(state.scan_interpretation),
            )
        state.scan_change_details = findings
        state.scan_change_findings = findings[:3]
        state.scan_change_summary = (
            f"{scan_label(after)} compared with {scan_label(before)}"
            if pair
            else (row.get("notices") or ["Waiting for another compatible measured curve"])[0]
        )
        regions = []
        for f in findings:
            for interval in f.get("regions", []):
                if not any(r["interval"] == interval and r["code"] == f["code"] for r in regions):
                    regions.append(
                        {
                            "title": FINDING_LABELS.get(f["code"], f["title"]) + f" · {interval[0]:.5g}–{interval[1]:.5g}",
                            "value": str(len(regions)),
                            "interval": interval,
                            "code": f["code"],
                        }
                    )
        if regions:
            unit = self._scan_interpreter.pattern(after).packet["coordinate"]["unit"]
            unit = {"us": "μs", "angstrom": "Å", "A": "Å"}.get(unit, unit)
            for region in regions:
                region["title"] += f" {unit}"
        self._scan_change_regions = regions
        state.scan_change_regions = [{"title": "Full measured coverage", "value": "full"}] + [
            {k: v for k, v in r.items() if k not in {"interval", "code"}} for r in regions
        ]
        state.scan_change_region = "full"
        state.scan_change_finding = ""
        state.scan_change_selected_evidence = {}
        labels = phase_labels(before, after) if pair else {}

        def readable(text):
            for key in sorted(labels, key=len, reverse=True):
                text = text.replace(key, labels[key])
            return text

        state.scan_change_items = [
            {**f, "label": FINDING_LABELS.get(f["code"], f["title"]),
             "summary": readable((f.get("evidence") or [f["title"]])[0]),
             "evidence": [readable(t) for t in f.get("evidence", [])],
             "region_count": len(f.get("regions", [])),
             "group": "model" if f["code"] in MODEL_FINDINGS else "pattern"}
            for f in findings
        ]
        for item in state.scan_change_items:
            if item["code"] in {"new_features", "lost_features"}:
                count = len(pair["features"][item["code"]])
                side = "current" if item["code"] == "new_features" else "earlier"
                item["summary"] = (
                    f"{count} {side} local feature" + (" was" if count == 1 else "s were")
                    + " not matched."
                )
        state.scan_change_can_plot = pair.get("features", {}).get("status") == "compared"
        features = pair.get("features", {})
        state.scan_change_metrics = (
            [
                f"{features['points']} common valid bins · Pearson {features['pearson']:.6g} · shape difference {features['shape_effect']:.5g}",
                f"Reference scale {features['scale']:.6g} · additive offset {features['offset']:.6g}",
            ]
            if state.scan_change_can_plot
            else []
        )
        state.scan_change_metrics += [
            f"{d['quantity']}: {d['difference']:+.6g} {d.get('unit','')} ({(d['current'] or {}).get('status','unknown')})"
            for d in pair.get("condition_changes", [])
            if d.get("difference") is not None
        ]
        labels = []
        for d in pair.get("condition_changes", []):
            a, b = d.get("previous"), d.get("current")
            if a and b and d.get("difference") is not None:
                labels.append(
                    f"{d['quantity'].replace('_',' ')} {a['value']:.5g} → {b['value']:.5g} {d.get('unit','')} ({b.get('status','unknown')})"
                )
        state.scan_change_condition_summary = " · ".join(labels[:3]) or (
            "Comparable sample conditions are unavailable · see Conditions" if pair else ""
        )
        model = pair.get("model_changes", {})
        state.scan_change_model_rows = model_change_rows(
            model, phase_labels(before, after) if pair else {},
            InterpretationSettings.model_validate(state.scan_interpretation),
        )
        state.scan_change_model_notice = model.get("notice") or (
            "Changes in the refined model; percentages are relative to the reference. Fractions use percentage points (pp). Significance and calibration still require review."
        )
        if model.get("rwp_difference") is not None:
            state.scan_change_metrics.append(
                f"Rwp change {model['rwp_difference']:+.5g} percentage points · same recorded recipe/domain; not a scientific acceptance score"
            )
        if model.get("notice"):
            state.scan_change_metrics.append(model["notice"])
        self._update_scan_change_plot()
        state.flush()

    def _scan_by_id(self, run_id):
        return next(s for s in self._interpretation_scans if str(s.get("run_id")) == str(run_id))

    def choose_scan_change_revision(self, **_):
        state = self.server.state
        row = getattr(self, "_selected_interpretation", {})
        allowed = {s["value"] for s in state.scan_change_revision_options}
        if state.scan_change_revision in allowed:
            payload = dict(state.scan_interpretation)
            payload["active_revisions"] = {
                **payload["active_revisions"],
                row["measurement_key"]: state.scan_change_revision,
            }
            state.scan_interpretation = InterpretationSettings.model_validate(payload).model_dump(
                mode="json"
            )
            self._refresh_scan_interpretation(
                self._interpretation_scans, getattr(self, "_post_analysis_pending", [])
            )
        state.flush()

    def _update_scan_change_plot(self):
        state = self.server.state
        current = next((s for s in self._interpretation_scans if str(s.get("run_id")) == str(state.scan_change_run)), None)
        state.scan_change_has_curve = False
        if current is None:
            state.scan_change_figure = {"data": [], "layout": {}}
            return
        try:
            after = self._scan_interpreter.pattern(current)
        except (ValueError, TypeError, KeyError, OSError):
            state.scan_change_figure = {"data": [], "layout": {}}
            return
        state.scan_change_has_curve = True
        if not state.scan_change_can_plot:
            state.scan_change_figure = plotly.Figure.to_data(observation_figure(
                after, label=state.scan_change_run, show_fit=state.scan_change_fit,
            ))
            return
        region = next(
            (
                r["interval"]
                for r in self._scan_change_regions
                if r["value"] == state.scan_change_region
            ),
            None,
        )
        before = self._scan_interpreter.pattern(self._scan_by_id(state.scan_change_pair))
        state.scan_change_figure = plotly.Figure.to_data(
            comparison_figure(
                before,
                after,
                self._selected_interpretation_pair,
                before_label=state.scan_change_pair,
                after_label=state.scan_change_run,
                region=region,
                show_fit=state.scan_change_fit,
            )
        )

    def open_scan_change(self, **_):
        self._update_scan_change_plot()
        self.server.state.scan_change_open = True
        self.server.state.flush()

    def select_scan_change_region(self, value, **_):
        state = self.server.state
        state.scan_change_region = value
        code = next((r["code"] for r in self._scan_change_regions if r["value"] == value), "")
        state.scan_change_finding = code
        state.scan_change_selected_evidence = next(
            (f for f in state.scan_change_items if f["code"] == code), {}
        )
        self._update_scan_change_plot()
        state.flush()

    def select_scan_change_finding(self, code, **_):
        region = next((r for r in self._scan_change_regions if r["code"] == code), None)
        self.select_scan_change_region(region["value"] if region else "full")
        state = self.server.state
        state.scan_change_finding = code
        state.scan_change_selected_evidence = next(
            (f for f in state.scan_change_items if f["code"] == code), {}
        )
        state.flush()

    def toggle_scan_change_fit(self, value, **_):
        self.server.state.scan_change_fit = bool(value)
        self._update_scan_change_plot()
        self.server.state.flush()

    def update_scan_change_setting(self, key, value, **_):
        try:
            if key != "shape_effect":
                raise ValueError("Unknown interpretation control")
            payload = dict(self.server.state.scan_interpretation)
            payload[key] = float(value)
            settings = InterpretationSettings.model_validate(payload)
            self.server.state.scan_interpretation = settings.model_dump(mode="json")
            self.server.state.scan_change_error = ""
            self._refresh_scan_interpretation(
                self._interpretation_scans, getattr(self, "_post_analysis_pending", [])
            )
        except (ValueError, TypeError) as exc:
            self.server.state.scan_change_error = str(exc)
        self.server.state.flush()

    def pin_scan_change_reference(self, **_):
        row = getattr(self, "_selected_interpretation", {})
        if row.get("group_key"):
            payload = dict(self.server.state.scan_interpretation)
            payload["references"] = {**payload["references"], row["group_key"]: row["run_id"]}
            payload["reference_hashes"] = {
                **payload["reference_hashes"],
                row["group_key"]: row["pattern_sha256"],
            }
            self.server.state.scan_interpretation = payload
            self._refresh_scan_interpretation(
                self._interpretation_scans, getattr(self, "_post_analysis_pending", [])
            )
        self.server.state.flush()

    def bind_scan_change_condition(self, **_):
        from .scan_interpretation import condition_scope

        state = self.server.state
        row = getattr(self, "_selected_interpretation", {})
        try:
            scan = self._scan_by_id(state.scan_change_run)
            if not (scan.get("metadata") or {}).get("sample_id") or not scan.get("ipts"):
                raise ValueError(
                    "A known IPTS and specimen identity are required for a device mapping"
                )
            if not state.scan_change_binding_confirmed:
                raise ValueError(
                    "Verify the specimen/device connection before applying this mapping"
                )
            channel = next(
                c
                for c in row["condition_receipt"]["channels"]
                if c["channel"] == state.scan_change_binding
                and c.get("status") == "needs_mapping"
                and c.get("role") == "measured"
            )
            scope = condition_scope(scan)
            payload = dict(state.scan_interpretation)
            bindings = {
                **payload["condition_bindings"].get(scope, {}),
                channel["quantity"]: channel["channel"],
            }
            payload["condition_bindings"] = {**payload["condition_bindings"], scope: bindings}
            state.scan_interpretation = InterpretationSettings.model_validate(payload).model_dump(
                mode="json"
            )
            state.scan_change_error = ""
            self._refresh_scan_interpretation(
                self._interpretation_scans, getattr(self, "_post_analysis_pending", [])
            )
        except (KeyError, ValueError, StopIteration) as exc:
            state.scan_change_error = str(exc)
        state.flush()

    async def save_scan_change(self, **_):
        try:
            controller = getattr(self, "_powgen_controller", None)
            if controller is not None and str(getattr(self, "_post_analysis_scope", "")).startswith(
                "watch:"
            ):
                controller.state.scan_interpretation = dict(self.server.state.scan_interpretation)
                controller.state.interpretation_report = compact_report(self._interpretation_report)
                await asyncio.to_thread(controller.persist_state)
                self.server.state.notice = "Saved interpretation settings, reference receipts and findings to the watch checkpoint."
            else:
                self.save_current_configuration()
        except Exception as exc:
            self.server.state.scan_change_error = str(exc)
        self.server.state.flush()

    def download_scan_change(self, **_):
        state = self.server.state
        if not state.scan_change_can_plot:
            return
        buffer = io.BytesIO()
        scans = []
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for index, run_id in enumerate([state.scan_change_pair, state.scan_change_run]):
                scan = self._scan_by_id(run_id)
                pattern = self._scan_interpreter.pattern(scan)
                filename = f"pattern-{index}.json"
                raw = json.dumps(pattern.packet, sort_keys=True, allow_nan=False).encode()
                archive.writestr(filename, raw)
                clean = {
                    k: v
                    for k, v in scan.items()
                    if k not in {"pattern", "pattern_path", "pattern_observation", "summary_path"}
                }
                clean["pattern_path"] = filename
                scans.append(clean)
            settings = dict(state.scan_interpretation)
            # The pair replay has its own initial reference; the original
            # full-series comparison and pinned reference are retained below.
            settings.update(references={}, reference_hashes={})
            archive.writestr(
                "manifest.json", json.dumps({"scans": scans}, indent=2, allow_nan=False)
            )
            archive.writestr("settings.json", json.dumps(settings, indent=2, allow_nan=False))
            archive.writestr(
                "comparison.json",
                json.dumps(self._selected_interpretation_pair, indent=2, allow_nan=False),
            )
            archive.writestr(
                "findings.json",
                json.dumps(self._selected_interpretation, indent=2, allow_nan=False),
            )
            archive.writestr(
                "README.txt",
                "Replay: python -m radar_pd_nova.scan_interpretation manifest.json --settings settings.json --output replay.json\nNative-bin curves, masks and errors are retained. Full-series findings require the original history receipts.\n",
            )
        self.download_file("radar-pd-scan-comparison.zip", "application/zip", buffer.getvalue())
