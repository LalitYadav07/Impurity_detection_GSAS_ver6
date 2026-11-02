"""Long saved activity stays available without overwhelming the Live queue."""
import asyncio
from dataclasses import replace

from radar_pd_nova.models import RunStatus
from test_live_restore import app, checkpoint, inspect, restore
from test_live_restore_polish import UiTree


def test_queue_bounds_long_console_preview_and_keeps_full_text_and_lifecycle(app):
    checkpoint(app.service)
    asyncio.run(inspect(app))
    restore(app)
    controller = app._powgen_controller
    failed_id = next(iter(controller.state.failed))
    tail = "Analysis completed successfully\n" + ("/SNS/PG3/IPTS-37869/source.gsa\tstep timing 1.25 seconds\n" * 100)
    failed = replace(controller.state.failed[failed_id], error=tail)
    controller.state.failed[failed_id] = failed
    submitted_id = next(iter(controller.state.submitted))
    record = controller.records[submitted_id]
    record.status = RunStatus.RUNNING
    record.stage = "Processing after an earlier setup error"
    app._sync_powgen_rows(refresh_dependents=False, load_patterns=False)
    rows = {row["run_id"]: row for row in app.server.state.powgen_rows}
    failure = rows[failed.run_id]
    assert failure["status"] == "Failed"  # Console wording cannot reclassify the saved lifecycle.
    assert failure["detail"] == tail and failed.error == tail
    assert len(failure["detail_summary"]) <= 180 and failure["detail_summary"].endswith("...")
    assert failure["detail_summary"].startswith("Analysis completed successfully")
    assert "\n" not in failure["detail_summary"] and "\t" not in failure["detail_summary"]
    active = rows[submitted_id]
    assert active["status"] == "Running"
    assert active["detail"] == active["detail_summary"]
    assert record.stage == "Processing after an earlier setup error"


def test_queue_cell_exposes_full_activity_with_a_keyboard_accessible_disclosure(app):
    nodes = UiTree(app.layout.html).nodes
    slot = next(node for node in nodes if node["attrs"].get("v-slot:item.detail") == "{ item }")
    table = slot["parent"]
    assert table["tag"] == "vdatatable" and table["attrs"]["itemvalue"] == "run_id"
    cell = next(node for node in nodes if node["parent"] is slot)
    assert "radar-queue-activity" in cell["attrs"]["class"]
    preview = next(node for node in nodes if node["parent"] is cell and node["tag"] == "span")
    assert preview["text"].strip() == "{{ item.detail_summary }}"
    disclosure = next(node for node in nodes if node["parent"] is cell and node["tag"] == "details")
    assert "open" not in disclosure["attrs"]
    summary = next(node for node in nodes if node["parent"] is disclosure and node["tag"] == "summary")
    assert summary["text"].strip() == "Show full activity"
    assert "item.scan_label" in summary["attrs"][":aria-label"]
    full_text = next(node for node in nodes if node["parent"] is disclosure and node["tag"] == "pre")
    assert full_text["text"].strip() == "{{ item.detail }}"
    assert full_text["attrs"]["tabindex"] == "0"
    assert "v-html" not in full_text["attrs"] and "title" not in preview["attrs"]
