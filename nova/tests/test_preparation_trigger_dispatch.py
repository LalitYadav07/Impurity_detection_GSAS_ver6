"""Actual Trame triggers must release the serial message path during saves."""
import asyncio
from copy import deepcopy
from pathlib import Path
import threading
from types import SimpleNamespace
from uuid import uuid4

import pytest
from trame.app import get_server
from trame_server.protocol import CoreServer

from radar_pd_nova.preparation_controller import PreparationMixin


@pytest.mark.parametrize('failure',[False,True])
def test_delayed_save_releases_real_trame_trigger_for_navigation(tmp_path,failure):
    async def check():
        server=get_server('preparation-dispatch-'+uuid4().hex,client_type='vue3')
        state=server.state;published=[]
        state._push_state_fn=lambda delta:published.append(deepcopy(delta))
        app=PreparationMixin();app.server=server
        app._preparation_table_changed=lambda **_:None
        app._prepared_selection_caption=lambda:'Local preparation fixture'
        entered=threading.Event();release=threading.Event();exports=[];uploads=[]
        def export(key,path):
            exports.append(key);Path(path).write_bytes(b'portable snapshot for '+key.encode())
        def upload(path,**kwargs):
            uploads.append((Path(path).read_bytes(),kwargs))
            entered.set()
            if not release.wait(3):raise TimeoutError('Test release was not received')
            if failure:raise OSError('Controlled upload failure')
            return 'saved-dataset'
        service=SimpleNamespace(galaxy_url='https://galaxy.invalid/prefix',upload_document=upload)
        registry=SimpleNamespace(export=export,select=lambda key:None,
            get=lambda key:SimpleNamespace(document=dict(id='a'*24,name=key)))
        store=SimpleNamespace(root=tmp_path,registry=registry)
        app.service=service;app._preparation_store=lambda:store
        app._init_preparation();app._preparation_workspace=store
        app._prep_return=('analysis',False)
        state.workspace_view='preparation';state.setup_collapsed=True
        state.prep_library='original';state.ready()
        protocol=object.__new__(CoreServer);protocol.server=server
        save=server.trigger_name(app.save_preparation_download)
        back=server.trigger_name(app.close_preparation_workspace)
        def select_another():
            state.prep_library='another'
            # A save retains its original service/History as well as library.
            app.service=SimpleNamespace(galaxy_url='https://another.invalid',
                upload_document=lambda *a,**k:pytest.fail('Save switched History'))
        select=server.trigger_name(select_another)
        analysis=server.trigger_name(lambda:setattr(state,'workspace_view','analysis'))

        async def serial_messages():
            assert await protocol.trigger(save,[],{}) is None
            assert await protocol.trigger(save,[],{}) is None  # busy guard
            await protocol.trigger(select,[],{})
            await protocol.trigger(back,[],{})
            await protocol.trigger(analysis,[],{})

        wire=asyncio.create_task(serial_messages())
        try:
            assert await asyncio.to_thread(entered.wait,2)
            # Faithful to serial websocket dispatch: the next trigger cannot
            # run until CoreServer.trigger returns for the previous message.
            await asyncio.wait_for(asyncio.shield(wire),.25)
            assert state.workspace_view=='analysis' and state.prep_busy
            assert len(app._prep_tasks)==1 and exports==['original'] and len(uploads)==1
            assert any(delta.get('workspace_view')=='analysis' for delta in published)
        finally:
            release.set()
            await asyncio.gather(wire,return_exceptions=True)
            await asyncio.gather(*tuple(app._prep_tasks),return_exceptions=True)
            await asyncio.sleep(0)
        assert not state.prep_busy and not app._prep_tasks
        assert exports==['original'] and len(uploads)==1
        assert uploads[0][0]==b'portable snapshot for original'
        assert uploads[0][1]['wait_ready'] is True
        assert state.prep_download_href==''  # never assigned to another library
        assert not list((tmp_path/'exports').glob('*.zip'))
        if failure:assert 'OSError' in state.prep_error
        else:
            assert not state.prep_error
            assert 'previously selected library was saved' in state.prep_message
    asyncio.run(check())


def test_trame_protocol_really_waits_when_callback_returns_a_task():
    """Control for the harness: this is the former blocking return contract."""
    async def check():
        server=get_server('preparation-await-control-'+uuid4().hex,client_type='vue3')
        protocol=object.__new__(CoreServer);protocol.server=server
        entered=asyncio.Event();release=asyncio.Event();navigation=[]
        async def delayed():entered.set();await release.wait()
        first=server.trigger_name(lambda:asyncio.create_task(delayed()))
        second=server.trigger_name(lambda:navigation.append('analysis'))
        async def serial():
            await protocol.trigger(first,[],{})
            await protocol.trigger(second,[],{})
        wire=asyncio.create_task(serial());await entered.wait();await asyncio.sleep(0)
        assert not wire.done() and not navigation
        release.set();await wire
        assert navigation==['analysis']
    asyncio.run(check())
