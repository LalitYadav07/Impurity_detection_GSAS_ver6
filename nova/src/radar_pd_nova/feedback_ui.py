"""Small, session-independent Contact developer workflow."""
import asyncio
import base64
import io
import json
import zipfile

import requests
from trame.widgets import html, vuetify3 as v

from .feedback import FeedbackDelivery, image_attachment, MAX_IMAGES


def browser_private_delivery_js():
    """Use qualified browser APIs inside Vue's component-scoped handler."""
    return ("async () => { const prepared=await trigger('radar_contact_prepare',[]); if(!prepared)return;"
        "try { if(new window.URL(prepared.origin).origin !== window.location.origin)throw new Error();"
        "const who=await window.fetch('/api/users/current',{credentials:'same-origin',redirect:'error',cache:'no-store'});"
        "if(!who.ok || (await who.json()).id !== prepared.owner_id)throw new Error();"
        "const response=await window.fetch('/history/make_private?history_id='+encodeURIComponent(prepared.history_id),"
        "{credentials:'same-origin',redirect:'error',cache:'no-store',headers:{Accept:'application/json'}});"
        "if(!response.ok || !(await response.json()).message)throw new Error();"
        "return trigger('radar_contact_send',[]);"
        "}catch(e){return trigger('radar_contact_privacy_failed',[]);} }")


class FeedbackMixin:
    def _init_feedback(self):
        self._feedback_images = []
        self._feedback_delivery = None
        self._feedback_inflight = False
        self._feedback_uploads = 0
        self.server.state.update(dict(contact_open=False, contact_message='', contact_category='Other',
            contact_reply='', contact_images=[], contact_busy=False, contact_error='',
            contact_status='', contact_receipt='', contact_url='', contact_pending=False,
            contact_download='', contact_file=None, contact_uploading=False))

    def contact_upload_started(self):
        s=self.server.state
        if not s.contact_pending:
            self._feedback_uploads+=1;s.contact_uploading=True;s.flush()

    def contact_upload_failed(self):
        self._feedback_uploads=max(0,self._feedback_uploads-1)
        s=self.server.state;s.contact_uploading=self._feedback_uploads>0
        s.contact_error='The image was not transferred. Use PNG, JPEG or WebP of at most 5 MiB.';s.flush()

    def contact_image(self, contents, name='screenshot.png'):
        s = self.server.state
        if s.contact_busy or s.contact_pending:
            return
        try:
            if len(self._feedback_images) >= MAX_IMAGES:
                raise ValueError('Attach at most four images.')
            image = image_attachment(contents, name)
            if any(im['sha256'] == image['sha256'] for im in self._feedback_images):
                raise ValueError('This screenshot is already attached.')
            self._feedback_images.append(image)
            s.contact_images = [{'name': im['name'], 'preview': 'data:image/png;base64,' +
                base64.b64encode(im['contents']).decode()} for im in self._feedback_images]
            s.contact_error = ''
        except ValueError as exc:
            s.contact_error = str(exc)
        self._feedback_uploads=max(0,self._feedback_uploads-1)
        s.contact_uploading=self._feedback_uploads>0
        s.contact_file = None
        s.flush()

    def remove_contact_image(self, index):
        s = self.server.state
        if s.contact_busy or s.contact_pending:
            return
        if type(index) is int and 0 <= index < len(self._feedback_images):
            self._feedback_images.pop(index)
            s.contact_images = [im for i, im in enumerate(s.contact_images) if i != index]

    def new_contact_report(self):
        if self.server.state.contact_busy or self.server.state.contact_uploading:
            return
        self._init_feedback()
        self.server.state.contact_open = True

    def download_contact_draft(self):
        s = self.server.state
        out = io.BytesIO()
        with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('message.json', json.dumps({'message': s.contact_message,
                'category': s.contact_category, 'reply_contact': s.contact_reply}, indent=2))
            for i, image in enumerate(self._feedback_images):
                archive.writestr(f'{i+1}-{image["name"]}', image['contents'])
        s.contact_download = 'data:application/zip;base64,' + base64.b64encode(out.getvalue()).decode()

    async def prepare_contact_report(self):
        s=self.server.state
        if s.contact_busy or s.contact_uploading or s.contact_receipt or not str(s.contact_message).strip():return None
        s.contact_busy=True;s.contact_pending=True;s.contact_error=''
        s.contact_status='Preparing private report storage…';s.flush()
        if self._feedback_delivery is None:self._feedback_delivery=FeedbackDelivery(self.service)
        try:return await asyncio.to_thread(self._feedback_delivery.prepare)
        except (requests.RequestException,ValueError,KeyError,TypeError):
            self.contact_privacy_failed();return None

    def contact_privacy_failed(self):
        s=self.server.state;s.contact_busy=False;s.contact_status=''
        s.contact_error=('Private delivery could not be prepared. Keep this Galaxy browser signed in as the session owner and retry. '
                         'The draft is retained; no report content is sent until private upload defaults pass a server-side check.')

    async def send_contact_report(self):
        s = self.server.state
        if self._feedback_inflight or s.contact_receipt:
            return
        if not str(s.contact_message).strip():
            s.contact_error = 'Write a message before sending.'
            return
        if self._feedback_delivery is None:
            self._feedback_delivery = FeedbackDelivery(self.service)
        self._feedback_inflight = True
        s.contact_busy = True
        s.contact_pending = True
        s.contact_error = ''
        s.contact_status = 'Saving your report in a private Galaxy History…'
        self.download_contact_draft()
        args = (s.contact_message, s.contact_category, s.contact_reply,
                list(self._feedback_images), {'version': s.application_version,
                                             'view': s.workspace_view})
        s.flush()
        try:
            for attempt in range(30):
                try:
                    receipt = await asyncio.to_thread(self._feedback_delivery.send, *args)
                    break
                except ValueError as exc:
                    if 'still processing an attachment' not in str(exc) or attempt == 29:
                        raise
                    s.contact_status = 'Galaxy is processing the report upload. Your draft is retained…'
                    s.flush()
                    await asyncio.sleep(3)
            s.contact_receipt = receipt['report_id']
            s.contact_url = receipt['url']
            s.contact_status = 'Delivered privately to the RADAR-PD developer. This report remains in Galaxy after End session.'
        except (requests.RequestException, ValueError, KeyError, TypeError):
            # Never interpolate remote responses, URLs or authentication material.
            s.contact_error = ('Delivery is not confirmed. Your message and images are retained. '
                'Retry this report unchanged, or download the draft. Galaxy may still be processing an upload.')
            s.contact_status = ''
        finally:
            self._feedback_inflight = False
            s.contact_busy = False
            s.flush()

    def _contact_ui(self):
        self.server.controller.trigger('radar_contact_image')(self.contact_image)
        self.server.controller.trigger('radar_contact_upload_started')(self.contact_upload_started)
        self.server.controller.trigger('radar_contact_upload_failed')(self.contact_upload_failed)
        self.server.controller.trigger('radar_contact_prepare')(self.prepare_contact_report)
        self.server.controller.trigger('radar_contact_send')(self.send_contact_report)
        self.server.controller.trigger('radar_contact_privacy_failed')(self.contact_privacy_failed)
        send_js = browser_private_delivery_js()
        paste = ("async ($event) => { if(contact_pending || contact_uploading)return; const files=Array.from($event.clipboardData?.items || [])"
                 ".filter(i=>i.kind==='file' && i.type.startsWith('image/')).map(i=>i.getAsFile());"
                 " if (!files.length) return; $event.preventDefault();"
                 " for (const f of files) { if(f.size > 5242880) {await trigger('radar_contact_upload_failed',[]);continue;}"
                 " await trigger('radar_contact_upload_started',[]);try{const b=await f.arrayBuffer(); "
                 "await trigger('radar_contact_image',[b,f.name || 'pasted-screenshot.png']);}"
                 "catch(e){await trigger('radar_contact_upload_failed',[]);} } }")
        with v.VDialog(v_model=('contact_open',),max_width=820,persistent=('contact_busy',)):
            with v.VCard():
                v.VCardTitle('Contact developer')
                with v.VCardText():
                    html.P('Send a bug, question, feature request or any other message to Lalit Yadav. '
                        'Only this message and the images you choose are privately shared. Your Galaxy username/email '
                        'accompany the report; the analysis History, data, logs and keys are not attached.')
                    v.VSelect(label='Category (optional)',v_model=('contact_category',),
                        items=("['Bug','Feature request','Question','Other']",),
                        disabled=('contact_pending',),density='compact',variant='outlined')
                    v.VTextarea(label='Message · you can paste a screenshot here',v_model=('contact_message',),
                        rows=5,auto_grow=True,counter=16000,maxlength=16000,
                        disabled=('contact_pending',),variant='outlined',paste=paste,__events=['paste'])
                    v.VTextField(label='Reply contact (optional)',v_model=('contact_reply',),maxlength=320,
                        disabled=('contact_pending',),density='compact',variant='outlined')
                    v.VFileInput(label='Attach screenshot · PNG, JPEG or WebP · up to 5 MiB each',
                        v_model=('contact_file',),accept='image/png,image/jpeg,image/webp',
                        disabled=('contact_pending || contact_uploading || contact_images.length >= 4',),
                        update_modelValue=("async ($event)=>{ const value=$event; const selected=Array.isArray(value)?value:(value?[value]:[]); const f=selected[0]; if(!f)return;"
                            "if(f.size>5242880){return trigger('radar_contact_upload_failed',[]);}"
                            "await trigger('radar_contact_upload_started',[]);try{const b=await f.arrayBuffer();"
                            "return trigger('radar_contact_image',[b,f.name]);}catch(e){return trigger('radar_contact_upload_failed',[]);} }"),density='compact')
                    with html.Div(v_for='(image,index) in contact_images',key='image.name',classes='d-inline-flex flex-column ma-2'):
                        html.Img(src=('image.preview',),alt=('image.name',),style='max-width:220px;max-height:140px;object-fit:contain')
                        html.Span('{{ image.name }}')
                        v.VBtn('Remove',size='small',variant='text',disabled=('contact_pending',),
                            click=(self.remove_contact_image,'[index]'))
                    v.VAlert(v_if='!!contact_error',text=('contact_error',),type='error',variant='tonal',classes='my-3')
                    html.P('{{ contact_status }}',role='status',v_if='!!contact_status')
                    html.P('Report {{ contact_receipt }}',v_if='!!contact_receipt')
                    html.A('Open saved report',href=('contact_url',),v_if='!!contact_url',target='_blank',rel='noopener noreferrer')
                    v.VBtn('Prepare draft download',click=self.download_contact_draft,variant='text',size='small',disabled=('contact_busy',))
                    html.A('Download report ZIP',href=('contact_download',),download='radar-pd-feedback.zip',v_if='!!contact_download',classes='ml-3')
                with v.VCardActions():
                    v.VBtn('Close',click='contact_open = false',disabled=('contact_busy',))
                    v.VBtn('New report',click=self.new_contact_report,disabled=('contact_busy',),v_if='contact_pending',variant='text')
                    v.VSpacer()
                    v.VBtn("{{ contact_pending ? 'Retry delivery' : 'Send privately' }}",click=send_js,
                        disabled=('contact_busy || contact_uploading || !!contact_receipt || !contact_message.trim()',),loading=('contact_busy',),color='#15543c',variant='flat')
