"""Approved JSON is data; generated executable code is a fixed trusted wrapper."""
from datetime import datetime, timezone
import json
from pathlib import Path

from .contracts import validate_plan, plan_hash


def approve(plan, context, preview, *, approver):
    validate_plan(plan, context)
    if plan["questions"] or plan["unsupported"]:
        raise ValueError("Resolve all plan questions and unsupported requests before approval")
    if not approver.strip() or preview.get("status") != "preview_passed" or preview.get("plan_sha256") != plan_hash(plan):
        raise ValueError("Approval needs an identified user and successful preflight of this exact plan")
    if preview.get("model_signature") != context["model_signature"] or preview.get("source_sha256") != context["source"]["sha256"]:
        raise ValueError("Preview belongs to a different source project")
    return {"schema": "radar-gpx-approval/v1", "approved_plan_sha256": plan_hash(plan),
            "model_signature": context["model_signature"], "preview_source_sha256": context["source"]["sha256"],
            "approver": approver, "approved_utc": datetime.now(timezone.utc).isoformat()}


def compile_script(plan, approval, destination):
    if approval.get("approved_plan_sha256") != plan_hash(plan):
        raise ValueError("Cannot compile a changed or unapproved plan")
    text = '''#!/usr/bin/env python3
"""Generated from an approved RADAR-PD refinement plan. No model code is executed.
Requires the pinned GSAS-II runtime and the shipped gpx_recipe package.
"""
import argparse
import json
from pathlib import Path
from gpx_recipe.execution import run_plan, write_json
from gpx_recipe.contracts import plan_hash

PLAN = json.loads(PLAN_LITERAL)
APPROVAL = json.loads(APPROVAL_LITERAL)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-gpx", required=True)
    parser.add_argument("--output-gpx", required=True)
    args = parser.parse_args()
    try:
        result = run_plan(args.source_gpx, args.output_gpx, PLAN, approved=APPROVAL)
    except Exception as exc:
        # Source/model validation can reject an arrival before an execution
        # receipt exists. Preserve that reason without claiming it executed.
        failure = Path(args.output_gpx).with_suffix(".failure.json")
        if failure.parent.is_dir() and not failure.exists():
            try:
                write_json(failure, {"schema": "radar-gpx-failure/v1", "plan_sha256": plan_hash(PLAN),
                                     "error_type": type(exc).__name__, "error": str(exc)})
            except OSError:
                pass  # Preserve the primary rejection if artifact storage fails.
        raise
    print(json.dumps({"status": result["status"], "stages": len(result["stages"])}))
'''
    prefix, rest = text.split("PLAN_LITERAL", 1)
    middle, suffix = rest.split("APPROVAL_LITERAL", 1)
    text = prefix + repr(json.dumps(plan)) + middle + repr(json.dumps(approval)) + suffix
    compile(text, "<approved_refinement>", "exec")
    Path(destination).write_text(text)
    return text
