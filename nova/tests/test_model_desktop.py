"""The desktop opens the clicked audited trial, never a preferred collection element."""
import asyncio
from types import SimpleNamespace
import pytest

from gpx_recipe.symmetry.compiler import file_sha
from gpx_recipe.symmetry.contracts import digest
from radar_pd_nova.symmetry_controller import SymmetryMixin


@pytest.fixture
def desktop(tmp_path,monkeypatch):
    source=tmp_path/'trial/results/model.gpx';source.parent.mkdir(parents=True);source.write_bytes(b'exact selected native checkpoint')
    receipt=dict(id='selected',files={'results/model.gpx':file_sha(source)})
    monkeypatch.setattr('radar_pd_nova.symmetry_controller.verify_receipt',lambda *args:receipt)
    state=SimpleNamespace(gsasii_session_status='',gsasii_launch_url='',sx_selected='selected',sx_error='',flush=lambda:None)
    class App(SymmetryMixin):
        def _symmetry_root(self):return tmp_path/'workspace'
        def _schedule_utility(self,work,name):self.pending.append(work)
        async def _submit_utility_action(self,**kwargs):self.launches.append(kwargs);return object()
    app=App();app.server=SimpleNamespace(state=state);app.pending=[];app.launches=[];app.uploads=[]
    def upload(path,**kwargs):
        app.uploads.append((path.read_bytes(),kwargs));return 'exact-dataset'
    app.service=SimpleNamespace(upload_document=upload)
    app._symmetry_rows=[dict(id='selected',name='Selected model',protocol='protocol',directory=str(tmp_path/'trial'),
        has_gpx=True,has_view=True,gpx_relative='results/model.gpx',receipt_sha256=digest(receipt))]
    return app,receipt


def test_desktop_captures_clicked_trial_and_blocks_duplicate_start(desktop):
    app,_=desktop
    app.open_symmetry_in_gsasii()
    app.server.state.sx_selected='another-trial'
    app.open_symmetry_in_gsasii()
    assert len(app.pending)==1
    asyncio.run(app.pending.pop())
    assert app.uploads[0][0]==b'exact selected native checkpoint'
    assert app.launches[0]['inputs']=={'project_source|source_kind':'single','project_source|gpx_project':{'dataset_id':'exact-dataset'}}
    assert 'selected' in app.launches[0]['name']


def test_changed_receipt_cannot_launch_or_upload(desktop):
    app,receipt=desktop;receipt['changed']=True
    app.open_symmetry_in_gsasii();asyncio.run(app.pending.pop())
    assert not app.uploads and not app.launches
    assert app.server.state.gsasii_session_status=='error'
    assert 'receipt changed' in app.server.state.sx_error


def test_unaudited_checkpoint_cannot_launch_a_desktop(desktop):
    app,_=desktop;app._symmetry_rows[0]['has_view']=False
    app.open_symmetry_in_gsasii()
    assert not app.pending and not app.launches and not app.uploads
    assert 'audited' in app.server.state.sx_error
