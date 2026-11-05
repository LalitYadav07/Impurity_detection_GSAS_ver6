import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from radar_pd_nova.feedback import FeedbackDelivery, image_attachment


def png():
    out = io.BytesIO()
    Image.new('RGB', (40, 30), 'white').save(out, format='PNG')
    return out.getvalue()


def test_screenshot_decode_and_metadata_free_name():
    image = image_attachment(png(), '../../screen.png')
    assert image['name'] == 'screen.png'
    with Image.open(io.BytesIO(image['contents'])) as decoded:
        assert decoded.size == (40, 30) and not decoded.info
    for bad in (b'<svg onload=alert(1)>', b'bad', b'', b'x'*(5*1024*1024+1)):
        with pytest.raises(ValueError): image_attachment(bad, 'screen.png')


class API:
    def __init__(self, *, queued=False, failed_share=False):
        self.calls=[];self.title='';self.files=[];self.queued=queued;self.failed_share=failed_share
    def __call__(self, method, url, **kwargs):
        assert kwargs['allow_redirects'] is False
        path=url.split('/api/')[1];self.calls.append((method,path,kwargs))
        if path=='users/current': value={'id':'reporter','email':'reporter@example.org','username':'reporter'}
        elif path=='roles': value=[{'id':'private-role','type':'private','name':'reporter@example.org'}]
        elif path=='histories' and method=='GET': value=[]
        elif path=='histories': self.title=kwargs['json']['name'];value={'id':'report-only'}
        elif path=='histories/report-only': value={'name':self.title,'user_id':'reporter'}
        elif path=='histories/report-only/contents': value=self.files.copy()
        elif path=='tools':
            name=json.loads(kwargs['data']['inputs'])['files_0|NAME']
            self.files.append({'name':name,'id':str(len(self.files))})
            value={'outputs':[self.files[-1]]}
        elif path.startswith('datasets/'): value={'state':'queued' if self.queued else 'ok','permissions':{'access':['private-role']}}
        elif path.endswith('/share_with_users'):
            assert kwargs['json'] in ({'user_ids':['ly6@ornl.gov'],'share_option':'make_accessible_to_shared'}, {'user_ids':[],'share_option':'no_changes'})
            value={}
        elif path.endswith('/sharing'):
            value={'published':False,'importable':False,'users_shared_with':[] if self.failed_share else [{'email':'ly6@ornl.gov'}]}
        else: raise AssertionError(path)
        return SimpleNamespace(status_code=200,raise_for_status=lambda:None,json=lambda:value)


def service():
    return SimpleNamespace(galaxy_url='https://ndip.example.org',_headers={'x-api-key':'test-only'},history_id='PRIVATE-ANALYSIS')


def test_report_shares_only_its_new_history_and_does_not_copy_session_data():
    api=API();delivery=FeedbackDelivery(service(),request=api)
    result=delivery.send('A user message','Bug','', [image_attachment(png(),'screenshot.png')],
                         {'version':'test','view':'results','api_key':'NEVER-SEND','config':'PRIVATE'})
    assert result['history_id']=='report-only' and result['attachments']==1
    assert not any('PRIVATE-ANALYSIS' in path for _,path,_ in api.calls)
    content=next(kwargs['files']['files_0|file_data'][1] for method,path,kwargs in api.calls if path=='tools' and json.loads(kwargs['data']['inputs'])['files_0|NAME']=='message.json')
    assert b'NEVER-SEND' not in content and b'PRIVATE' not in content
    # Confirmed retries reconcile existing attachments instead of duplicating.
    delivery.send('A user message','Bug','',[image_attachment(png(),'screenshot.png')],{'version':'test','view':'results'})
    assert sum(path=='tools' for _,path,_ in api.calls)==3


def test_queued_upload_keeps_attempt_and_retry_does_not_duplicate():
    api=API(queued=True);delivery=FeedbackDelivery(service(),request=api)
    with pytest.raises(ValueError,match='processing'):
        delivery.send('message','Other','',[],{})
    assert not any(path.endswith('/share_with_users') for _,path,_ in api.calls)
    api.queued=False
    delivery.send('message','Other','',[],{})
    assert sum(path=='tools' for _,path,_ in api.calls)==2


def test_missing_sharing_confirmation_is_not_success_and_draft_cannot_change():
    api=API(failed_share=True);delivery=FeedbackDelivery(service(),request=api)
    with pytest.raises(ValueError,match='private access'):delivery.send('message','Other','',[],{})
    with pytest.raises(ValueError,match='pending delivery'):delivery.send('changed','Other','',[],{})


def test_public_upload_default_blocks_message_and_screenshot_before_upload():
    api=API()
    def public_default(method,url,**kwargs):
        response=api(method,url,**kwargs)
        if '/api/datasets/' in url:
            response.json=lambda:{'state':'ok','permissions':{'access':[]}}
        return response
    delivery=FeedbackDelivery(service(),request=public_default)
    with pytest.raises(ValueError,match='Private upload defaults'):
        delivery.send('PRIVATE MESSAGE','Bug','',[image_attachment(png(),'private.png')],{})
    uploaded=[kwargs['files']['files_0|file_data'] for _,path,kwargs in api.calls if path=='tools']
    assert len(uploaded)==1 and uploaded[0][0]=='privacy-check.txt'
    assert b'PRIVATE MESSAGE' not in uploaded[0][1]
    assert not any(path.endswith('/share_with_users') for _,path,_ in api.calls)


def test_static_select_items_survive_real_html_attribute_parsing():
    from ast import literal_eval
    from html.parser import HTMLParser
    from radar_pd_nova.app import RadarPdNovaApp
    class Selects(HTMLParser):
        def __init__(self):super().__init__();self.items=[]
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key==':items' and value.startswith("['"):self.items.append(literal_eval(value))
    app=RadarPdNovaApp();app.create_ui();parser=Selects();parser.feed(app.layout.html)
    assert ['Bug','Feature request','Question','Other'] in parser.items
    assert ['X','Y','Z'] in parser.items


def test_image_transfer_blocks_prepare_until_server_has_decoded_it():
    from radar_pd_nova.app import RadarPdNovaApp
    app=RadarPdNovaApp();s=app.server.state
    app.contact_upload_started()
    assert s.contact_uploading
    app.contact_image(png(),'screenshot.png')
    assert not s.contact_uploading and len(app._feedback_images)==1
    app.contact_upload_started();app.contact_upload_failed()
    assert not s.contact_uploading and 'not transferred' in s.contact_error


def test_contact_button_precedes_end_and_paste_does_not_intercept_text():
    from radar_pd_nova.app import RadarPdNovaApp
    app=RadarPdNovaApp();app.create_ui();template=app.layout.html
    assert template.index('id="radar-contact-developer"') < template.index('id="radar-end-session"')
    assert '@paste="async ($event)' in template
    assert "if (!files.length) return; $event.preventDefault();" in template
    assert 'Answer the questions above' in template
