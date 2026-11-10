"""Navigation dispatch is acknowledged before the existing exit callback runs."""
import asyncio
from copy import deepcopy
import html
from html.parser import HTMLParser
import re
from types import SimpleNamespace
from uuid import uuid4

import pytest
from trame.app import get_server
from trame_server.protocol import CoreServer
from wslink.publish import PublishManager

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.models import RunStatus, UtilityActionRecord


def connected_app(monkeypatch):
    server=get_server('session-nav-'+uuid4().hex,client_type='vue3')
    monkeypatch.setattr('radar_pd_nova.app.get_server',lambda *_a,**_k:server)
    app=RadarPdNovaApp();publisher=PublishManager()
    protocol=object.__new__(CoreServer);protocol.publish=publisher.publish;protocol.server=server
    server._root_protocol=protocol
    server.state._push_state_fn=lambda delta:publisher.publish('trame.state.topic',delta)
    server.state.ready()
    markup=html.unescape(app._leave_session_page.__self__.html)
    trigger=re.search(r"trigger\('([^']+)', \[\$event\]\)",markup).group(1)
    return app,publisher,protocol,trigger


@pytest.mark.parametrize('delay',[0,.25])
def test_real_trame_action_waits_for_current_nonce_ack_after_transport(monkeypatch,delay):
    async def check():
        app,publisher,protocol,trigger=connected_app(monkeypatch)
        events=[];tokens=[]
        class Wire:
            network_monitor=SimpleNamespace(on_enter=lambda:None,on_exit=lambda *_:None)
            async def sendWrappedMessage(self,topic,data,**_):
                if 'trame.actions.topic' not in topic:return
                await asyncio.sleep(delay)
                action=deepcopy(data[0]);token=action['args'][0];tokens.append(token)
                assert action['method']=='exec' and re.fullmatch('[a-f0-9]{32}',token)
                events.append('navigation-dispatched')
                assert await protocol.trigger(trigger,[token],{}) is True
        publisher.registerProtocol(Wire())
        async def checkpoint():events.append('checkpoint')
        async def exit_callback():events.append('exit')
        app._stop_session_enrollment=checkpoint;app.exit_callback=exit_callback
        await app.end_session()
        assert events==['checkpoint','navigation-dispatched','exit']
        assert len(tokens)==1 and not app._acknowledge_session_navigation(tokens[0])
        assert app.server.state.nova_kill_jobs_on_exit is False
    asyncio.run(check())


def test_missing_ack_exits_after_a_small_bounded_wait(monkeypatch):
    async def check():
        app,_,_,_=connected_app(monkeypatch);events=[]
        async def checkpoint():events.append('saved')
        async def exit_callback():events.append('exit')
        app._stop_session_enrollment=checkpoint;app.exit_callback=exit_callback
        app._leave_session_page=lambda nonce:events.append('dispatched')
        loop=asyncio.get_running_loop();start=loop.time()
        await app.end_session();elapsed=loop.time()-start
        assert events==['saved','dispatched','exit']
        assert 1.8<=elapsed<3.0 and app.server.state.session_exit_ready
        assert app._session_navigation_ack is None
    asyncio.run(check())


def test_stale_ack_and_repeated_click_cannot_finish_or_duplicate_close(monkeypatch):
    async def check():
        app,_,protocol,trigger=connected_app(monkeypatch)
        entered=asyncio.Event();tokens=[];exits=[]
        async def checkpoint():pass
        async def exit_callback():exits.append(True)
        def dispatch(token):tokens.append(token);entered.set()
        app._stop_session_enrollment=checkpoint;app.exit_callback=exit_callback;app._leave_session_page=dispatch
        closing=asyncio.create_task(app.end_session());await entered.wait()
        assert await protocol.trigger(trigger,['0'*32],{}) is False
        await asyncio.sleep(0)
        assert not closing.done() and exits==[]
        await app.end_session()
        assert len(tokens)==1
        assert await protocol.trigger(trigger,[tokens[0]],{}) is True
        assert await protocol.trigger(trigger,[tokens[0]],{}) is False
        await closing
        assert exits==[True]
    asyncio.run(check())


def test_failed_checkpoint_never_dispatches_navigation_or_exits(monkeypatch):
    async def check():
        app,_,_,_=connected_app(monkeypatch);events=[]
        async def fail():raise RuntimeError('Checkpoint unavailable')
        async def exit_callback():events.append('exit')
        app._stop_session_enrollment=fail;app.exit_callback=exit_callback
        app._leave_session_page=lambda nonce:events.append('navigate')
        await app.end_session()
        assert events==[] and not app.server.state.session_closing
        assert not app.server.state.session_exit_ready
        assert 'Checkpoint unavailable' in app.server.state.workspace_notice
    asyncio.run(check())


def test_exit_keeps_scientific_job_record_and_disables_framework_kill(monkeypatch):
    async def check():
        from nova.trame import ThemedApp
        app,_,_,_=connected_app(monkeypatch)
        action=UtilityActionRecord(uid='science',tool_id='science-tool',name='Existing scientific job',
            galaxy_job_id='existing-job',status=RunStatus.RUNNING)
        app.utility_actions[action.uid]=action;calls=[]
        app.service=SimpleNamespace(cancel=lambda *_:pytest.fail('Scientific job must not be cancelled'))
        async def checkpoint():calls.append('saved')
        async def exit_callback():
            assert not app.server.state.nova_kill_jobs_on_exit
            with pytest.raises(SystemExit) as stopped:await ThemedApp.exit_callback(app)
            assert stopped.value.code==0;calls.append('exit')
        app._stop_session_enrollment=checkpoint;app.exit_callback=exit_callback
        app._leave_session_page=lambda nonce:app._acknowledge_session_navigation(nonce)
        await app.end_session()
        assert calls==['saved','exit'] and action.status==RunStatus.RUNNING
    asyncio.run(check())


def test_closed_view_has_plain_message_and_native_return_link(monkeypatch):
    app,_,_,_=connected_app(monkeypatch)
    assert app.server.state.trame_error_report_msg==''
    class Tags(HTMLParser):
        def __init__(self):super().__init__();self.anchors=[];self.text=[]
        def handle_starttag(self,tag,attrs):
            if tag=='a':self.anchors.append(dict(attrs))
        def handle_data(self,text):self.text.append(text)
    view=Tags();view.feed(app.server.state.trame__template_error)
    assert view.anchors==[{'href':'/'}]
    assert 'Connection closed' in ''.join(view.text) and 'Return to Galaxy' in ''.join(view.text)
    assert 'trame-reconnect' in app.server.state.trame__template_error_reconnect
    assert 'href="/"' in app.layout.html


def test_rendered_end_session_confirmation_buttons_remain_visible_before_exit_ready(monkeypatch):
    app,_,_,_=connected_app(monkeypatch)
    markup=re.search(r'<VDialog v-model="end_session_dialog".*?</VDialog>',
                     app.layout.html,re.DOTALL).group(0)

    class DialogTree(HTMLParser):
        def __init__(self):super().__init__();self.stack=[];self.nodes=[]
        def handle_starttag(self,tag,attrs):
            node=SimpleNamespace(tag=tag,attrs=dict(attrs),parent=self.stack[-1] if self.stack else None,text='')
            self.nodes.append(node);self.stack.append(node)
        def handle_endtag(self,tag):
            assert self.stack[-1].tag==tag
            self.stack.pop()
        def handle_data(self,value):
            if self.stack:self.stack[-1].text+=value

    tree=DialogTree();tree.feed(markup)
    card=next(node for node in tree.nodes if node.tag=='vcard')
    actions=next(node for node in tree.nodes if node.tag=='vcardactions')
    ready=next(node for node in tree.nodes if node.attrs.get('v-if')=='session_exit_ready')
    # Both status and actions must be children of the same rendered card. A
    # misplaced Python ``with`` otherwise hides the only confirmation buttons.
    assert ready.parent is card
    assert actions.parent is card
    buttons=[node for node in tree.nodes if node.tag=='vbtn']
    assert [button.text.strip() for button in buttons]==['Keep working','End session']
    for button in buttons:
        assert button.parent is actions
        ancestor=button
        while ancestor is not None:
            assert 'v-if' not in ancestor.attrs and 'v-show' not in ancestor.attrs
            ancestor=ancestor.parent
        assert button.attrs[':disabled']=='session_closing'
    assert app.server.state.session_closing is False
    assert app.server.state.session_exit_ready is False
