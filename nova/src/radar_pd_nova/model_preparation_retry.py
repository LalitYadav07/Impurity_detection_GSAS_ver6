"""Explicit, witnessed retries of failed native inspection publication.

No scientific protocol is changed. A retry has its own stable dispatch key,
and an unresolved prior submission is never evidence for another submission.
"""
import base64
import copy
import hashlib
import json
import re
import threading

from gpx_recipe.symmetry.compiler import package_signature
from gpx_recipe.symmetry.contracts import digest
from .models import UtilityActionRecord
from .refinement_runtime import RuntimeUnavailable

SCHEMA = 'radar-model-preparation-attempt/v1'
PROOF_SCHEMA = 'radar-model-preparation-failure/v1'
MARKER = 'radar-model-packet.json'
MARKER_SHA = hashlib.sha256(b'{"schema":"radar-model-packet/v1","files":{}}\n').hexdigest()
_SOURCE_LOCKS = {}
_SOURCE_LOCKS_GUARD = threading.Lock()


def inspection_request(source):
    return dict(schema='radar-model-native/v1', operation='inspect', payload={'source':'source.gpx'},
        implementation_sha256=package_signature(), expected_source_sha256=source['source_sha256'],
        input_hashes={'source.gpx':source['source_sha256'], MARKER:MARKER_SHA}, timeout=120.)


def base_key(runtime, source, request):
    return digest([runtime.service.history_id, source['id'], request])[:24]


def existing_job(runtime, key):
    cached = runtime.state['dispatches'].get(key, {}).get('job')
    job = UtilityActionRecord.model_validate(cached) if cached else runtime.find_dispatch(key)
    if job is not None:runtime.verify_job(job, key)
    return job


def no_active_inspection(runtime, source):
    """Check the bounded active queue before replacing a failed inspection."""
    from .native_job_admission import HistoryJobAdmission
    from .galaxy_service import _parameter_value
    for job in HistoryJobAdmission(runtime.service).active_jobs():
        if job['tool_id']!=runtime.tool_id:continue
        details=runtime.service._job_details(job['id'])
        if (details.get('id')!=job['id'] or details.get('history_id')!=runtime.service.history_id
                or details.get('tool_id')!=runtime.tool_id):
            raise RuntimeUnavailable('An active model job has unverified identity; wait before retrying')
        params=details.get('params') or details.get('parameters') or {}
        action=_parameter_value(params,'action')
        if action=='chunk':continue
        if action!='native':raise RuntimeUnavailable('An active model job cannot be identified; wait before retrying')
        try:
            encoded=_parameter_value(params,'model_request_b64')
            if not isinstance(encoded,str) or len(encoded)>6*1024*1024:raise ValueError()
            request=json.loads(base64.b64decode(encoded,validate=True))
            if not isinstance(request,dict) or not request.get('operation') or not request.get('expected_source_sha256'):
                raise ValueError()
        except (ValueError,TypeError):
            raise RuntimeUnavailable('An active native preparation cannot be identified; wait before retrying') from None
        if request.get('operation')=='inspect' and request.get('expected_source_sha256')==source['source_sha256']:
            raise RuntimeUnavailable('An inspection of this source is already active; recover it instead of starting another')


def failure_proof(runtime, source, request, key):
    """Require exact failed-job identity and zero-byte publication failure.

    Do not reinterpret native failures, cancellation, incompatible calibration,
    unreadable metadata, or an unknown acknowledgment as retry permission.
    """
    from .galaxy_service import _parameter_value
    job = existing_job(runtime, key)
    if job is None:raise RuntimeUnavailable('No acknowledged failed inspection was found; no retry was queued')
    details = runtime.service._job_details(str(job.galaxy_job_id))
    params = details.get('params') or details.get('parameters') or {}
    if (details.get('id') != job.galaxy_job_id or details.get('history_id') != runtime.service.history_id
            or details.get('tool_id') != runtime.tool_id or _parameter_value(params,'dispatch_key') != key
            or _parameter_value(params,'action') != 'native'):
        raise RuntimeUnavailable('The failed inspection identity differs from this History and source')
    encoded = _parameter_value(params,'model_request_b64')
    try:
        if not isinstance(encoded,str) or len(encoded)>6*1024*1024:raise ValueError()
        actual = json.loads(base64.b64decode(encoded,validate=True))
    except (ValueError, TypeError):
        raise RuntimeUnavailable('The failed inspection has no verifiable inline request; no retry was queued') from None
    expected = dict(request,dispatch_key=key)
    if actual != expected:
        raise RuntimeUnavailable('The failed inspection request or source revision differs')
    inputs=details.get('inputs') or {}
    source_input=inputs.get('source_gpx') if isinstance(inputs,dict) else None
    expected_dataset=runtime.dataset_id(source)
    if not isinstance(source_input,dict) or source_input.get('id')!=expected_dataset:
        raise RuntimeUnavailable('The failed inspection used a different or unverifiable source dataset')
    stderr = next((str(details[k]) for k in ['stderr','tool_stderr','job_stderr']
                   if details.get(k) and str(details[k]).strip()), '').strip()
    if (details.get('state') != 'error' or type(details.get('exit_code')) is not int
            or details['exit_code'] != 0 or not stderr.startswith('Pulsar metadata processing failed:')):
        raise RuntimeUnavailable('Only a terminal inspection publication failure with native exit code zero can be retried')
    outputs = runtime.service._job_output_ids(details)
    if not all(outputs.get(k) for k in ['result_bundle','console_output']):
        raise RuntimeUnavailable('The failed inspection output identities are incomplete')
    verified = {}
    for name in ['result_bundle','console_output']:
        identifier = outputs[name]
        metadata = runtime.service._dataset_metadata(identifier)
        if (not isinstance(metadata,dict) or metadata.get('id') != identifier
                or metadata.get('state') != 'error' or type(metadata.get('file_size')) is not int
                or metadata['file_size'] != 0 or metadata.get('deleted') or metadata.get('purged')):
            raise RuntimeUnavailable('The failed inspection has a nonempty, active, missing, or altered output')
        verified[name] = dict(id=identifier,state='error',file_size=0)
    if verified['result_bundle']['id'] == verified['console_output']['id']:
        raise RuntimeUnavailable('The inspection outputs are not distinct')
    return dict(schema=PROOF_SCHEMA,history_id=runtime.service.history_id,tool_id=runtime.tool_id,
        source_id=source['id'],source_dataset_id=expected_dataset,
        source_sha256=source['source_sha256'],dispatch_key=key,job_id=job.galaxy_job_id,
        request_sha256=digest(expected),request_body_sha256=digest(request),state='error',exit_code=0,
        failure_kind='pulsar_metadata_publication',stderr_sha256=hashlib.sha256(stderr.encode()).hexdigest(),
        outputs=verified)


def validate_attempt(runtime, source, request, attempt):
    fields={'schema','history_id','campaign','template_sha256','source_id','source_sha256',
            'request_body_sha256','base_dispatch_key','generation','previous_dispatch_key',
            'failure_proof','failure_proof_sha256','dispatch_key'}
    if (not isinstance(attempt,dict) or set(attempt)!=fields or attempt['schema']!=SCHEMA
            or type(attempt['generation']) is not int or attempt['generation']<1
            or not isinstance(attempt['campaign'],str) or not re.fullmatch('campaign-[a-f0-9]{12}',attempt['campaign'])
            or not re.fullmatch('[a-f0-9]{64}',str(attempt['template_sha256']))
            or attempt['history_id'] != runtime.service.history_id or attempt['source_id'] != source['id']
            or attempt['source_sha256'] != source['source_sha256']
            or attempt['request_body_sha256'] != digest(request)
            or attempt['base_dispatch_key'] != base_key(runtime,source,request)
            or digest(attempt['failure_proof']) != attempt['failure_proof_sha256']):
        raise RuntimeUnavailable('The saved preparation attempt no longer matches its source, template, or request')
    proof=attempt['failure_proof']
    if (proof.get('schema')!=PROOF_SCHEMA or proof.get('history_id')!=attempt['history_id']
            or proof.get('tool_id')!=runtime.tool_id or proof.get('source_dataset_id')!=runtime.dataset_id(source)
            or proof.get('source_id')!=source['id'] or proof.get('source_sha256')!=source['source_sha256']
            or proof.get('dispatch_key')!=attempt['previous_dispatch_key']
            or proof.get('request_body_sha256')!=digest(request)):
        raise RuntimeUnavailable('The previous failure does not authorize this preparation attempt')
    identity={k:v for k,v in attempt.items() if k not in {'dispatch_key','failure_proof'}}
    if attempt['dispatch_key'] != digest(identity)[:24]:
        raise RuntimeUnavailable('The preparation attempt dispatch identity changed')
    return attempt


def prepare_attempt(runtime, source, campaign, template_sha256, previous=None):
    """Read-only remote proof, then an in-memory permit for one stable attempt.

    The caller persists the returned record before native() can use its permit.
    A restored record can recover an existing job but cannot obtain a new permit
    merely because its output name has not appeared in Galaxy yet.
    """
    request=inspection_request(source)
    previous_key=base_key(runtime,source,request)
    generation=1
    if previous is not None:
        validate_attempt(runtime,source,request,previous)
        if previous['campaign']!=campaign or previous['template_sha256']!=template_sha256:
            raise RuntimeUnavailable('The saved retry belongs to another campaign or template')
        current=existing_job(runtime,previous['dispatch_key'])
        if current is None:
            entry=runtime.state['dispatches'].get(previous['dispatch_key'],{})
            if previous['dispatch_key'] in runtime._preparation_retry_permits and not entry.get('pending'):
                return copy.deepcopy(previous),False
            raise RuntimeUnavailable('The prior retry acknowledgment is unresolved; recover it before requesting another attempt')
        runtime.service.refresh_utility(current)
        if current.status.value in {'deleted','cancelled'}:
            raise RuntimeUnavailable('A cancelled preparation is not a verified publication failure')
        if current.status.value != 'error':
            return copy.deepcopy(previous),False  # resume collection/wait; never replace an accepted job
        previous_key=previous['dispatch_key'];generation=previous['generation']+1
    proof=failure_proof(runtime,source,request,previous_key)
    no_active_inspection(runtime,source)
    record=dict(schema=SCHEMA,history_id=runtime.service.history_id,campaign=campaign,
        template_sha256=template_sha256,source_id=source['id'],source_sha256=source['source_sha256'],
        request_body_sha256=digest(request),base_dispatch_key=base_key(runtime,source,request),
        generation=generation,previous_dispatch_key=previous_key,
        failure_proof=proof,failure_proof_sha256=digest(proof))
    record['dispatch_key']=digest({k:v for k,v in record.items() if k!='failure_proof'})[:24]
    validate_attempt(runtime,source,request,record)
    runtime._preparation_retry_permits.add(record['dispatch_key'])
    return record,True


class _RetryAdmission:
    """Recheck mutable guards at every existing pre-POST admission boundary."""
    def __init__(self,cancel,guard,proof):
        self.cancel,self.guard,self.proof=cancel,guard,proof
        self.reason=''

    def is_set(self):
        if self.reason or self.cancel.is_set():return True
        try:
            if self.guard() is not True:
                raise RuntimeUnavailable('The campaign owner, scope, or preparation attempt changed before dispatch')
            self.proof()
        except Exception as error:
            self.reason=str(error) or 'Preparation retry admission could not be verified'
            return True
        return False

    def wait(self,seconds):
        self.cancel.wait(seconds)
        return self.is_set()


def submit_attempt(runtime, source, request, asset_id, attempt, before_submit, cancel, guard=None):
    # Coordinate retries of the same scientific input across runtime instances
    # in this viewer. The live active-job check still runs immediately before
    # POST; a second viewer must use the existing single-coordinator contract.
    identity=(runtime.service.galaxy_url,runtime.service.history_id,source['source_sha256'])
    with _SOURCE_LOCKS_GUARD:lock=_SOURCE_LOCKS.setdefault(identity,threading.Lock())
    with lock:
        return _submit_attempt(runtime,source,request,asset_id,attempt,before_submit,cancel,guard)


def _submit_attempt(runtime, source, request, asset_id, attempt, before_submit, cancel, guard):
    key=attempt['dispatch_key']
    body={k:v for k,v in request.items() if k!='dispatch_key'}
    validate_attempt(runtime,source,body,attempt)
    job=existing_job(runtime,key)
    if job is not None:return job
    if (key not in runtime._preparation_retry_permits
            or runtime.state['dispatches'].get(key,{}).get('pending')):
        raise RuntimeUnavailable('The saved retry has an unresolved acknowledgment; no duplicate inspection was submitted')
    if not callable(before_submit) or not callable(guard):
        raise RuntimeUnavailable('A preparation retry requires a durable checkpoint and live admission guard')
    def proof():
        if digest(failure_proof(runtime,source,body,attempt['previous_dispatch_key']))!=attempt['failure_proof_sha256']:
            raise RuntimeUnavailable('The failed inspection changed before retry dispatch')
        no_active_inspection(runtime,source)
    signals=_RetryAdmission(cancel,guard,proof)
    if signals.is_set():raise RuntimeUnavailable(signals.reason or 'Preparation retry cancelled before checkpoint')
    before_submit()
    if signals.is_set():raise RuntimeUnavailable(signals.reason or 'Preparation retry cancelled after checkpoint')
    try:
        return runtime._submit('native',source,request,asset_id,key,cancel=signals)
    except Exception:
        if signals.reason:raise RuntimeUnavailable(signals.reason) from None
        raise
