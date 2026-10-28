"""Session lifecycle tests: stale approvals must not survive an attempted edit."""
import copy
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from gpx_recipe.web import Session
from test_contracts import context, plan


class DeferredWorker:
    def submit(self, work): self.work = work


def test_failed_model_revision_revokes_previous_approval(tmp_path, monkeypatch, context, plan):
    session = object.__new__(Session)
    session.root, session.context = tmp_path, context
    session.lock, session.worker = threading.RLock(), DeferredWorker()
    session.state = {"status": "ready", "plan": copy.deepcopy(plan), "approval": {"previous": True},
                     "preview": {"previous": True}, "revisions": [], "origin": "test"}
    session.schedule("plan", {"request": "Change the background"})
    assert session.state["approval"] is None and session.state["preview"] is None
    def fail(*a, **kw): raise RuntimeError("Model timeout")
    monkeypatch.setattr("gpx_recipe.web.make_plan", fail)
    session.worker.work()
    assert session.state["status"] == "error"
    assert session.state["plan"] == plan
    assert session.state["approval"] is None
    assert "approval has been revoked" in session.state["message"]


def test_restart_restores_saved_plan_without_repeating_work(tmp_path, monkeypatch, context, plan):
    source = tmp_path / "source.gpx"; source.write_bytes(b"trusted fixture")
    (tmp_path / "context.json").write_text(json.dumps(context))
    saved = {"status": "busy", "message": "Was running", "plan": plan, "preview": None,
             "approval": None, "origin": "test", "revisions": []}
    (tmp_path / "session.json").write_text(json.dumps(saved))
    def native(self, *args): Path(args[2]).write_text(json.dumps(context))
    monkeypatch.setattr(Session, "native", native)
    args = SimpleNamespace(workdir=str(tmp_path), gpx=[str(source)], python=sys.executable, gsasii_root=None, fixture_plan=None)
    session = Session(args)
    try:
        assert session.state["plan"] == plan
        assert session.state["status"] == "error"
        assert "interrupted" in session.state["message"]
        assert not session.queue.jobs
    finally:
        session.worker.shutdown()


def test_deterministic_constraint_clarification_does_not_require_a_fake_model_call(tmp_path,monkeypatch,context,plan):
    session=object.__new__(Session);session.root,session.context=tmp_path,context
    session.lock,session.worker=threading.RLock(),DeferredWorker()
    session.state={'status':'ready','plan':None,'approval':None,'preview':None,'revisions':[],'origin':None}
    clarification=copy.deepcopy(plan);clarification.update(stages=[],questions=['Authorize the full existing Uiso group?'])
    monkeypatch.setattr('gpx_recipe.web.make_plan',lambda *a,**k:{'plan':clarification,'calls':[],'origin':'Deterministic source-constraint check'})
    session.schedule('plan',{'request':'Refine a held equality participant'});session.worker.work()
    assert session.state['status']=='ready' and session.state['origin']=='Deterministic source-constraint check'
    assert session.state['approval'] is None and session.state['plan']['questions']
