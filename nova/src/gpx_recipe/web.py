"""Small local review harness; the planner and worker are NDIP-portable modules.

No Studio import, credential file reader or GSAS-II unpickling occurs in the
HTTP process. Native inspection/preflight/refinement runs in a child runtime.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import threading
from urllib.parse import urlsplit, parse_qs
import uuid

from .compiler import approve, compile_script
from .contracts import validate_plan, plan_hash, stage_table
from .execution import write_json
from .jobs import RecipeQueue
from .planner import make_plan


class Session:
    def __init__(self, args):
        self.root = Path(args.workdir).resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.sources = [str(Path(p).resolve()) for p in args.gpx]
        self.python, self.gsasii_root = args.python, args.gsasii_root
        self.fixture = args.fixture_plan
        self.lock = threading.RLock(); self.worker = ThreadPoolExecutor(max_workers=1)
        self.queue = RecipeQueue(self.root / "jobs", python=args.python, gsasii_root=args.gsasii_root)
        self.state = {"status": "ready", "message": "Inspecting the starting GPX", "plan": None, "preview": None, "approval": None, "origin": None, "revisions": []}
        context_path = self.root / "context.json"
        previous_context = json.loads(context_path.read_text()) if context_path.exists() else None
        previous_session = json.loads((self.root / "session.json").read_text()) if (self.root / "session.json").exists() else None
        self.native("inspect", self.sources[0], str(context_path))
        self.context = json.loads(context_path.read_text())
        self.state["message"] = "Describe a recipe, or load the explicitly labeled execution test fixture."
        if previous_session and previous_session.get("plan"):
            try:
                validate_plan(previous_session["plan"], self.context)
                self.state = previous_session
                if not previous_context or previous_context["source"]["sha256"] != self.context["source"]["sha256"]:
                    self.state.update(preview=None, approval=None, status="ready", message="Saved plan restored; the source changed, so repeat preflight and approval.")
                elif self.state["status"] == "busy":
                    self.state.update(status="error", message="Previous operation was interrupted. Saved plan retained; no scan job was silently repeated.")
            except ValueError:
                self.state["message"] = "The saved plan does not match this GPX model; prepare a new plan. Previous artifacts remain on disk."
        self.save()

    def native(self, *args):
        env = os.environ.copy(); env.pop("SAMBANOVA_API_KEY", None)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        if self.gsasii_root: env["GSASII_ROOT"] = self.gsasii_root
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        command = [self.python, "-m", "gpx_recipe.cli", *args]
        log = self.root / ("native-" + uuid.uuid4().hex[:10] + ".log")
        with log.open("w") as handle:
            result = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=180)
        if result.returncode:
            raise ValueError("Native GSAS-II operation failed. " + log.read_text()[-2500:])

    def save(self):
        write_json(self.root / "session.json", self.state)

    def public(self):
        with self.lock:
            data = copy.deepcopy(self.state)
        data["context"] = self.context
        data["sources"] = [Path(p).name for p in self.sources]
        data["fixture_available"] = bool(self.fixture)
        data["jobs"] = list(json.loads(self.queue.manifest.read_text()).values())
        if data.get("plan"):
            data["table"] = stage_table(data["plan"], self.context)
            data["plan_sha256"] = plan_hash(data["plan"])
        return data

    def schedule(self, action, body):
        with self.lock:
            if self.state["status"] == "busy":
                raise ValueError("Another operation is running; wait before changing the reviewed plan")
            plan = copy.deepcopy(self.state["plan"])
            if action != "plan" and action != "fixture" and (not plan or body.get("plan_sha256") != plan_hash(plan)):
                raise ValueError("The reviewed plan changed; refresh before continuing")
            if action in {"plan", "fixture", "preview"}:
                # A failed revision must never leave the previous recipe
                # executable while the user expects their edit to be active.
                self.state.update(preview=None, approval=None)
            self.state.update(status="busy", message={"plan": "Waiting for ORNL SambaNova 120B…", "preview": "Checking effective parameters in GSAS-II…", "approve": "Recording approval and compiling the script…", "run": "Executing the approved recipe on the supplied scans…", "fixture": "Loading a deterministic execution test fixture…"}[action])
            self.save()
        def work():
            try:
                if action in {"plan", "fixture"}:
                    if action == "fixture":
                        if not self.fixture: raise ValueError("No fixture configured")
                        proposed = json.loads(Path(self.fixture).read_text())
                        validate_plan(proposed, self.context)
                        origin = "Execution test fixture — not generated by the model"
                    else:
                        response = make_plan(body.get("request", ""), self.context, previous=plan)
                        write_json(self.root / ("planning-" + uuid.uuid4().hex[:10] + ".json"), response)
                        proposed, origin = response["plan"], "SambaNova gpt-oss-120b"
                    with self.lock:
                        self.state["revisions"].append({"plan_sha256": plan_hash(proposed), "origin": origin, "plan": proposed})
                        self.state.update(plan=proposed, preview=None, approval=None, origin=origin,
                                          message="Plan ready for review. Check its assumptions and effective-parameter preflight.")
                elif action == "preview":
                    plan_file = self.root / ("plan-" + plan_hash(plan)[:12] + ".json"); write_json(plan_file, plan)
                    output = self.root / ("preview-" + uuid.uuid4().hex[:10] + ".gpx")
                    self.native("preview", self.sources[0], str(plan_file), str(output))
                    preview = json.loads(output.with_suffix(".execution.json").read_text())
                    with self.lock: self.state.update(preview=preview, message="Every stage has a valid GSAS-II parameter preflight. Review the free/fixed table before approving.")
                elif action == "approve":
                    receipt = approve(plan, self.context, self.state["preview"] or {}, approver="interactive-local-user")
                    write_json(self.root / "approved-plan.json", plan)
                    write_json(self.root / "approval.json", receipt)
                    compile_script(plan, receipt, self.root / "refine_project.py")
                    with self.lock: self.state.update(approval=receipt, message="This exact plan is approved. Script compiled; scan jobs can now be submitted.")
                elif action == "run":
                    if not self.state["approval"]: raise ValueError("Approve the plan before running")
                    jobs = [self.queue.enqueue(path, plan, self.state["approval"]) for path in self.sources]
                    for job in jobs:
                        if job["status"] == "queued": self.queue.run_next(job["id"])
                    with self.lock: self.state["message"] = "Scan jobs finished. Completed execution still requires scientific review of the values, residuals and uncertainties."
                with self.lock:
                    self.state["status"] = "ready"; self.save()
            except Exception as exc:
                with self.lock:
                    prefix = "Planning failed. The previous plan is unchanged and its approval has been revoked. " if action == "plan" else ""
                    self.state.update(status="error", message=prefix + str(exc)); self.save()
        self.worker.submit(work)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpx", action="append", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--python", required=True, help="Backend scientific interpreter; never supplied by model")
    parser.add_argument("--gsasii-root")
    parser.add_argument("--fixture-plan", help="Optional, explicitly labeled deterministic test fixture")
    parser.add_argument("--port", type=int, default=8891)
    args = parser.parse_args()
    session = Session(args)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def send_json(self, value, status=200):
            raw = json.dumps(value, allow_nan=False).encode()
            self.send_response(status); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)
        def do_GET(self):
            path = urlsplit(self.path)
            if path.path == "/api/state": return self.send_json(session.public())
            if path.path == "/":
                raw = Path(__file__).with_name("web.html").read_bytes()
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw); return
            if path.path == "/artifact":
                name = parse_qs(path.query).get("name", [""])[0]
                target = (session.root / name).resolve()
                if not target.is_relative_to(session.root) or not target.is_file() or target.suffix not in {".json", ".py", ".gpx", ".log"}:
                    return self.send_json({"error": "Unknown artifact"}, 404)
                raw = target.read_bytes()
                self.send_response(200); self.send_header("Content-Type", "application/octet-stream"); self.send_header("Content-Disposition", 'attachment; filename="' + target.name + '"'); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw); return
            self.send_json({"error": "Not found"}, 404)
        def do_POST(self):
            origin = self.headers.get("Origin")
            if origin and origin != f"http://127.0.0.1:{args.port}": return self.send_json({"error": "Unexpected origin"}, 403)
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json": return self.send_json({"error": "JSON body required"}, 400)
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 100000: return self.send_json({"error": "Invalid request size"}, 400)
            try:
                body = json.loads(self.rfile.read(length)); action = self.path.removeprefix("/api/")
                if action not in {"plan", "fixture", "preview", "approve", "run"}: raise ValueError("Unknown action")
                session.schedule(action, body); self.send_json({"status": "scheduled"}, 202)
            except (ValueError, TypeError) as exc: self.send_json({"error": str(exc)}, 400)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"GPX refinement planner: http://127.0.0.1:{args.port}", flush=True)
    try: server.serve_forever()
    finally: session.worker.shutdown(wait=False); server.server_close()


if __name__ == "__main__": main()
