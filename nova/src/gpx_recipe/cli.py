#!/usr/bin/env python3
"""NDIP-portable GPX context → plan → preflight → approval → script → execution."""
import argparse
import json
from pathlib import Path

from gpx_recipe.snapshot import snapshot
from gpx_recipe.planner import make_plan, build_payload
from gpx_recipe.compiler import approve, compile_script
from gpx_recipe.execution import run_plan, write_json


def load(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    p = commands.add_parser("inspect"); p.add_argument("gpx"); p.add_argument("output")
    for name in ("context", "plan"):
        p = commands.add_parser(name); p.add_argument("snapshot"); p.add_argument("request_file"); p.add_argument("output"); p.add_argument("--previous")
    p = commands.add_parser("preview"); p.add_argument("gpx"); p.add_argument("plan"); p.add_argument("output_gpx")
    p = commands.add_parser("approve"); p.add_argument("snapshot"); p.add_argument("plan"); p.add_argument("preview"); p.add_argument("output"); p.add_argument("--approver", required=True)
    p = commands.add_parser("compile"); p.add_argument("plan"); p.add_argument("approval"); p.add_argument("output_script")
    p = commands.add_parser("run"); p.add_argument("gpx"); p.add_argument("plan"); p.add_argument("approval"); p.add_argument("output_gpx")
    args = parser.parse_args()
    if args.action == "inspect":
        result = snapshot(args.gpx); write_json(args.output, result)
    elif args.action in {"context", "plan"}:
        context = load(args.snapshot); request = Path(args.request_file).read_text()
        previous = load(args.previous) if args.previous else None
        if previous and "plan" in previous: previous = previous["plan"]
        if args.action == "context":
            result = build_payload(request, context, previous)
        else:
            result = make_plan(request, context, previous=previous)
        write_json(args.output, result)
        if args.action == "plan": write_json(Path(args.output).with_suffix(".plan.json"), result["plan"])
    elif args.action == "preview":
        result = run_plan(args.gpx, args.output_gpx, load(args.plan), preview=True)
    elif args.action == "approve":
        result = approve(load(args.plan), load(args.snapshot), load(args.preview), approver=args.approver); write_json(args.output, result)
    elif args.action == "compile":
        compile_script(load(args.plan), load(args.approval), args.output_script); result = {"script": args.output_script}
    else:
        result = run_plan(args.gpx, args.output_gpx, load(args.plan), approved=load(args.approval))
    print(json.dumps({"action": args.action, "status": result.get("status", "ok")}))


if __name__ == "__main__":
    main()
