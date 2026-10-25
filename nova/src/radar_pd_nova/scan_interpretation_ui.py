"""Compact live-series interpretation with explicit, inspectable evidence."""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
import zipfile

from trame.widgets import html, plotly, vuetify3 as vuetify

from .scan_interpretation import InterpretationSettings, ScanInterpreter, digest
from .scan_interpretation_plot import comparison_figure


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
            "scan_change_error": "",
            "scan_change_open": False,
            "scan_change_fit": True,
            "scan_change_binding_options": [],
            "scan_change_binding": "",
            "scan_change_binding_confirmed": False,
            "scan_change_revision_options": [],
            "scan_change_revision": "",
        }.items():
            setattr(state, key, value)

    def _scan_change_dashboard(self):
        with html.Section(classes="radar-result-card my-3", style="scroll-margin-top:180px"):
            html.H3("What changed?", classes="radar-card-title")
            html.P("{{ scan_change_summary }}", classes="radar-section-help")
            with html.Div(style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:12px"):
                vuetify.VAutocomplete(
                    label="Acquisition / bank",
                    v_model=("scan_change_run",),
                    items=("scan_change_runs",),
                    density="compact",
                    variant="outlined",
                    hide_details=True,
                    style="flex:1;min-width:230px",
                    update_modelValue=(self.select_scan_change, "[$event]"),
                )
                vuetify.VSelect(
                    label="Compare with",
                    v_model=("scan_change_pair",),
                    items=("scan_change_pairs",),
                    density="compact",
                    variant="outlined",
                    hide_details=True,
                    style="flex:1;min-width:230px",
                    disabled=("!scan_change_pairs.length",),
                    update_modelValue=(self.select_scan_change_pair, "[$event]"),
                )
            html.P(
                "{{ scan_change_condition_summary }}",
                v_if="scan_change_condition_summary",
                classes="radar-section-help",
            )
            with html.Div(
                v_for="finding in scan_change_findings", key="finding.code", classes="mb-2"
            ):
                html.Div("{{ finding.title }}", style="font-weight:600")
                html.Div(
                    "{{ finding.status.replaceAll('_',' ') }} · {{ finding.evidence.slice(0,2).join(' · ') }}{{ finding.evidence.length>2 ? ' · '+(finding.evidence.length-2)+' more measurements in evidence' : '' }}",
                    classes="radar-section-help",
                )
            with html.Div(style="display:flex;gap:8px;flex-wrap:wrap"):
                vuetify.VBtn(
                    "Inspect curves and evidence",
                    variant="outlined",
                    size="small",
                    disabled=("!scan_change_can_plot",),
                    click=self.open_scan_change,
                )
                vuetify.VBtn(
                    "Export this comparison",
                    variant="text",
                    size="small",
                    disabled=("!scan_change_can_plot",),
                    click=self.download_scan_change,
                )
            with vuetify.VExpansionPanels(variant="accordion", classes="mt-2"):
                with vuetify.VExpansionPanel(
                    title="Conditions, qualifications and comparison settings"
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
            with vuetify.VCard():
                with vuetify.VCardTitle(
                    style="display:flex;justify-content:space-between;align-items:center;gap:12px"
                ):
                    html.Span("Scan comparison evidence")
                    vuetify.VBtn("Close", click="scan_change_open=false", variant="text")
                with vuetify.VCardText(style="overflow:auto"):
                    html.P("{{ scan_change_summary }}")
                    with html.Div(style="display:flex;gap:12px;align-items:center;flex-wrap:wrap"):
                        vuetify.VAutocomplete(
                            label="Feature region",
                            v_model=("scan_change_region",),
                            items=("scan_change_regions",),
                            variant="outlined",
                            density="compact",
                            hide_details=True,
                            style="min-width:260px;max-width:600px",
                            update_modelValue=(self.select_scan_change_region, "[$event]"),
                        )
                        vuetify.VCheckbox(
                            label="Show current calculated fit",
                            v_model=("scan_change_fit",),
                            hide_details=True,
                            update_modelValue=(self.toggle_scan_change_fit, "[$event]"),
                        )
                    html.P(
                        "Reference scale and offset are fitted separately; the lower curve retains local differences. Zoom and display choices do not alter the detector inputs.",
                        classes="radar-section-help",
                    )
                    with html.Div(style="min-width:0;width:100%"):
                        figure = plotly.Figure(
                            display_mode_bar=True, style="height:680px;width:100%"
                        )
                        figure.data = ("scan_change_figure.data",)
                        figure.layout = ("scan_change_figure.layout",)
                    with html.Div(
                        v_for="metric in scan_change_metrics", key="metric", classes="mb-1"
                    ):
                        html.Span("{{ metric }}")
                    with html.Div(
                        v_for="finding in scan_change_details", key="finding.code", classes="my-3"
                    ):
                        html.H4("{{ finding.title }}")
                        html.Div("{{ finding.evidence.join(' · ') }}")
                        html.Div(
                            "{{ finding.alternatives.join(' · ') }}", classes="radar-section-help"
                        )

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
                    "title": f"{row['run_id']} · sample {metadata.get('sample_id','unknown')}",
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
        self._selected_interpretation = row
        from .scan_interpretation import _measurement, _revision

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
                "title": p["reference_run_id"]
                + " · "
                + ", ".join(r.replace("_", " ") for r in p["roles"]),
                "value": p["reference_run_id"],
            }
            for p in pairs
        ]
        ids = [p["reference_run_id"] for p in pairs]
        if state.scan_change_pair not in ids:
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
        findings = row.get("findings", [])
        if pair and "previous" not in pair["roles"]:
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
        suffix = f" · {len(findings)-3} more in evidence" if len(findings) > 3 else ""
        state.scan_change_summary = (
            f"{state.scan_change_run} versus {value} · {len(findings)} findings{suffix}"
            if pair
            else (row.get("notices") or ["Waiting for another compatible measured curve"])[0]
        )
        regions = []
        for f in findings:
            for interval in f.get("regions", []):
                if interval not in [r["interval"] for r in regions]:
                    regions.append(
                        {
                            "title": f["title"] + f" · {interval[0]:.5g}–{interval[1]:.5g}",
                            "value": str(len(regions)),
                            "interval": interval,
                        }
                    )
        self._scan_change_regions = regions
        state.scan_change_regions = [{"title": "Full measured coverage", "value": "full"}] + [
            {k: v for k, v in r.items() if k != "interval"} for r in regions
        ]
        state.scan_change_region = "full"
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
            "No qualified condition pair is available" if pair else ""
        )
        model = pair.get("model_changes", {})
        if model.get("rwp_difference") is not None:
            state.scan_change_metrics.append(
                f"Rwp change {model['rwp_difference']:+.5g} percentage points · same recorded recipe/domain; not a scientific acceptance score"
            )
        if model.get("notice"):
            state.scan_change_metrics.append(model["notice"])
        if state.scan_change_open:
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
        if not state.scan_change_can_plot:
            state.scan_change_figure = {"data": [], "layout": {}}
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
        after = self._scan_interpreter.pattern(self._scan_by_id(state.scan_change_run))
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
        self.server.state.scan_change_region = value
        self._update_scan_change_plot()
        self.server.state.flush()

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
