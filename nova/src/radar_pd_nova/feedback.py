"""Private, report-only Galaxy delivery. Never share the analysis History."""
from __future__ import annotations

import hashlib
import io
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image, UnidentifiedImageError

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGES = 4
MAX_PIXELS = 16_000_000
RECIPIENT = 'ly6@ornl.gov'


def image_attachment(contents, name):
    """Decode then re-encode raster pixels, removing metadata and active content."""
    if not isinstance(contents, bytes) or not 0 < len(contents) <= MAX_IMAGE_BYTES:
        raise ValueError('Choose an image of at most 5 MiB.')
    try:
        with Image.open(io.BytesIO(contents)) as image:
            if image.format not in {'PNG', 'JPEG', 'WEBP'}:
                raise ValueError('Use PNG, JPEG or WebP screenshots.')
            if image.width * image.height > MAX_PIXELS or image.n_frames != 1:
                raise ValueError('Use a single image of at most 16 megapixels.')
            image.load()
            pixels = image.convert('RGB')
            out = io.BytesIO()
            pixels.save(out, format='PNG')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError('The screenshot could not be decoded.') from exc
    data = out.getvalue()
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError('The decoded screenshot exceeds 5 MiB; crop or resize it.')
    label = re.sub(r'[^\w. -]', '_', Path(str(name).replace('\\', '/')).stem)[:80] or 'screenshot'
    return {'name': label + '.png', 'contents': data,
            'sha256': hashlib.sha256(data).hexdigest(), 'mime': 'image/png'}


class FeedbackDelivery:
    """One durable attempt per dialog draft, with retry reconciliation."""
    def __init__(self, service, *, request=None, recipient=None):
        self.base = service.galaxy_url.rstrip('/')
        self.headers = service._headers
        self.request = request or requests.request
        self.recipient = recipient or RECIPIENT
        self.report_id = uuid.uuid4().hex
        self.history_id = None
        self.create_uncertain = False
        self.payload = None
        self.uploads = {}
        self.created_utc = datetime.now(timezone.utc).isoformat()
        self.verified_uploads = {}

    def api(self, method, path, **kwargs):
        response = self.request(method, self.base + '/api/' + path,
                                headers=self.headers, timeout=30, allow_redirects=False, **kwargs)
        response.raise_for_status()
        if response.status_code != 200 and response.status_code != 201:
            raise ValueError('Galaxy did not confirm the feedback operation.')
        return response.json()

    def prepare(self):
        owner = self.api('GET', 'users/current')
        if not owner.get('id'):
            raise ValueError('Sign in to Galaxy before sending feedback.')
        title = 'RADAR-PD feedback · ' + self.report_id
        if self.history_id is None:
            histories = self.api('GET', 'histories', params={'name': title, 'limit': 1000})
            found = [h for h in histories if h.get('name') == title and not h.get('deleted')]
            if len(found) > 1:
                raise ValueError('Multiple report histories were returned; download the draft and contact the developer.')
            if found:
                self.history_id = found[0]['id']
            elif self.create_uncertain:
                raise ValueError('Galaxy has not confirmed the report History yet. Retry later; no duplicate will be created.')
            else:
                self.create_uncertain = True
                self.history_id = self.api('POST', 'histories', json={'name': title})['id']
        hid = self.history_id
        # Protect against accidentally adopting or sharing an existing analysis History.
        status = self.api('GET', f'histories/{hid}')
        if status.get('name') != title or status.get('user_id', owner['id']) != owner['id']:
            raise ValueError('The dedicated report History could not be verified.')
        self.owner = owner
        return {'history_id': hid, 'owner_id': owner['id'], 'origin': self.base}

    def send(self, message, category, reply, images, diagnostics):
        message = str(message).strip()
        if not 1 <= len(message) <= 16000 or len(images) > MAX_IMAGES:
            raise ValueError('Write a message of 1–16,000 characters and attach at most four images.')
        if category not in {'Bug', 'Feature request', 'Question', 'Other'} or len(str(reply)) > 320:
            raise ValueError('Review the category and optional reply contact.')
        # Snapshot once: retries must deliver exactly the reviewed draft.
        reviewed = {'message': message, 'category': category, 'reply_contact': str(reply),
                    'images': [{'name': im['name'], 'sha256': im['sha256']} for im in images],
                    'diagnostics': {k: str(diagnostics.get(k, ''))[:100] for k in ('version', 'view')}}
        if self.payload is not None and self.payload != reviewed:
            raise ValueError('This report already has a pending delivery. Retry it unchanged, or start a new report.')
        self.payload = reviewed
        self.prepare()
        owner = self.owner
        hid = self.history_id
        report = {**reviewed, 'report_id': self.report_id,
                  'created_utc': self.created_utc,
                  'sender': {k: owner.get(k) for k in ('username', 'email')},
                  'recipient': self.recipient}
        roles = self.api('GET', 'roles')
        private = [r['id'] for r in roles if r.get('type') == 'private' and r.get('name') == owner.get('email')]
        if len(private) != 1:
            raise ValueError('The reporting user private role could not be verified.')
        files = [('privacy-check.txt', b'RADAR-PD private feedback permissions check. No report content.', 'txt'), ('message.json', json.dumps(report, indent=2).encode(), 'json')]
        files += [(f'image-{i+1}-{im["sha256"][:12]}.png', im['contents'], 'png') for i, im in enumerate(images)]
        pending = []
        for name, contents, file_type in files:
            existing = self.api('GET', f'histories/{hid}/contents')
            candidates = [row for row in existing if row.get('name') == name and not row.get('deleted')]
            if len(candidates) > 1:
                raise ValueError('Galaxy returned duplicate report attachments; delivery needs review.')
            if candidates:
                self.uploads[name] = candidates[0]['id']
            elif name not in self.uploads:
                # Mark uncertain before POST; a lost response must not re-upload.
                self.uploads[name] = None
                # upload1 uses legacy checkbox values; on Galaxy 26.1 an
                # explicitly false/empty space_to_tab can still enable it.
                # Fetch accepts real JSON booleans and preserves report bytes.
                targets = [{'destination': {'type': 'hdas'}, 'elements': [{
                    'src': 'files', 'name': name, 'ext': file_type, 'dbkey': '?',
                    'space_to_tab': False, 'to_posix_lines': False, 'auto_decompress': False}]}]
                result = self.api('POST', 'tools/fetch',
                    data={'history_id': hid, 'targets': json.dumps(targets)},
                    files={'files_0|file_data': (name, contents, 'application/octet-stream')})
                self.uploads[name] = result['outputs'][0]['id']
            if self.uploads.get(name) is None:
                raise ValueError('An attachment upload is awaiting confirmation. Retry later; the draft is retained.')
            dataset = self.api('GET', 'datasets/' + self.uploads[name])
            if dataset.get('state') in {'error','discarded','failed_metadata'}:
                raise ValueError('A report upload failed in Galaxy; the draft is retained.')
            if set((dataset.get('permissions') or {}).get('access', [])) != set(private):
                raise ValueError('Private upload defaults are not confirmed. No further report content will be uploaded.')
            if dataset.get('state') != 'ok':
                pending.append(name)
            else:
                digest = hashlib.sha256(contents).hexdigest()
                uid = self.uploads[name]
                if self.verified_uploads.get(uid) != digest:
                    response = self.request('GET', self.base+'/api/datasets/'+uid+'/display',
                        headers=self.headers, timeout=30, allow_redirects=False)
                    response.raise_for_status()
                    if response.status_code != 200 or hashlib.sha256(response.content).hexdigest() != digest:
                        raise ValueError('Galaxy altered a report attachment. Download the retained draft; delivery is not confirmed.')
                    self.verified_uploads[uid] = digest
        if pending:
            raise ValueError('Galaxy is still processing an attachment. Retry delivery when the upload completes.')
        # Sharing permissions must include protected contents, never make_public.
        if owner.get('email', '').casefold() != self.recipient.casefold():
            # A previous interrupted delivery may already have a share. Rebuild
            # it after browser-side private reset so new datasets are included.
            self.api('PUT', f'histories/{hid}/share_with_users', json={'user_ids': [], 'share_option': 'no_changes'})
            self.api('PUT', f'histories/{hid}/share_with_users', json={
                'user_ids': [self.recipient], 'share_option': 'make_accessible_to_shared'})
        sharing = self.api('GET', f'histories/{hid}/sharing')
        if sharing.get('published') is not False or sharing.get('importable') is not False:
            raise ValueError('Feedback privacy could not be verified. Delivery has not been marked complete.')
        if owner.get('email', '').casefold() != self.recipient.casefold():
            if not any(u.get('email', '').casefold() == self.recipient.casefold()
                       for u in sharing.get('users_shared_with', [])):
                raise ValueError('Galaxy has not confirmed private access for the developer.')
        return {'report_id': self.report_id, 'history_id': hid,
                'url': self.base + '/histories/view?id=' + hid, 'attachments': len(images)}
