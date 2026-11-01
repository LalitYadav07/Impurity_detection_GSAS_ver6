"""Authoritative evidence for explicitly requested Live retries."""


class RetrySourceChanged(RuntimeError):
    """A prepared explicit retry no longer has its original authorization."""


def failed_run_proof(service, job_id):
    """Qualify a terminal error for a fresh analysis, never infer execution.

    An ERROR job may have run the full native pipeline and published useful
    artifacts. Its runner/exit fields cannot establish otherwise. Unlike the
    paused-empty-input path below, this path never cancels the old job.
    """
    from .galaxy_service import ANALYZE_TOOL_ID

    def identity(job):
        if (job.get('id') != job_id or job.get('history_id') != service.history_id
                or job.get('tool_id') != ANALYZE_TOOL_ID or job.get('state') != 'error'
                or job.get('deleted') or job.get('purged')):
            return None
        return dict(job_id=job_id, history_id=service.history_id,
                    tool_id=ANALYZE_TOOL_ID, state='error',
                    inputs=service._public_input_dataset_ids(job),
                    outputs=service._job_output_ids(job),
                    update_time=job.get('update_time'))

    proof = identity(service._job_details(job_id))
    if not proof or not proof['inputs']:
        return None
    for identifier in proof['inputs'].values():
        metadata = service._dataset_metadata(identifier)
        if (metadata.get('history_id') != service.history_id or metadata.get('state') != 'ok'
                or metadata.get('deleted') or metadata.get('purged')):
            return None
    # Input metadata reads may be slow. Recheck the authoritative job after
    # them, including its output associations, before saving an intent.
    if identity(service._job_details(job_id)) != proof:
        return None
    return dict(proof, kind='terminal_error',
                reason='Explicit fresh analysis after a terminal Galaxy error; earlier work and artifacts are retained')


def validate_retry_source(service, proof):
    current = (failed_run_proof(service, proof['job_id'])
               if proof.get('kind') == 'terminal_error'
               else failed_input_proof(service, proof['job_id'], retiring=True))
    if current != proof:
        raise RetrySourceChanged('The previous job or its inputs changed; no fresh analysis was submitted. Review the existing job before retrying.')


def _cancelled_empty_upload(service, identifier, metadata):
    """Galaxy soft-deletes fetch outputs when their upload is cancelled.

    Accept that tombstone only with the creating fetch job's matching output
    and History. A deleted user dataset or a cancelled scientific job provides
    no such proof. The dataset remains deleted; retry stages a fresh input.
    """
    creator = metadata.get('creating_job')
    if metadata.get('state') != 'discarded' or not creator:
        return False
    try:
        fetch = service._job_details(creator)
        return (fetch.get('id') == creator
                and fetch.get('history_id') == service.history_id
                and fetch.get('tool_id') == '__DATA_FETCH__'
                and fetch.get('state') in {'deleted', 'deleted_new'}
                and identifier in service._job_output_ids(fetch).values())
    except Exception:
        return False


def failed_input_proof(service, job_id, *, retiring=False):
    job = service._job_details(job_id)
    states = {'paused', 'deleted', 'deleted_new'} if retiring else {'paused'}
    from .galaxy_service import ANALYZE_TOOL_ID
    if (job.get('id') != job_id or job.get('history_id') != service.history_id
            or job.get('tool_id') != ANALYZE_TOOL_ID or job.get('state') not in states):
        return None
    if any(job.get(k) not in (None, '') for k in
           ['external_id', 'job_runner_name', 'command_line', 'command_version', 'exit_code']):
        return None
    refs = service._public_input_dataset_ids(job)
    if not refs:
        return None
    interrupted = []
    for name, identifier in refs.items():
        metadata = service._dataset_metadata(identifier)
        if metadata.get('history_id', service.history_id) != service.history_id:
            return None
        if (metadata.get('state') in {'error', 'discarded', 'failed_metadata'}
                and type(metadata.get('file_size')) is int and metadata['file_size'] == 0
                and not metadata.get('purged')
                and (not metadata.get('deleted') or _cancelled_empty_upload(service, identifier, metadata))):
            interrupted.append(identifier)
        elif metadata.get('state') != 'ok' or metadata.get('deleted') or metadata.get('purged'):
            return None
    if not interrupted:
        return None
    return dict(job_id=job_id, inputs=refs, interrupted_inputs=sorted(interrupted),
        reason='An empty input upload failed before this Analyze job received a runner')


def retire_failed_input(service, proof):
    """Use the saved intent; wait for confirmed cancellation before re-staging."""
    import requests
    job_id = proof['job_id']
    if failed_input_proof(service, job_id, retiring=True) != proof:
        raise RuntimeError('The interrupted-input execution proof changed; review this scan')
    if service._job_details(job_id).get('state') == 'paused':
        response = requests.delete(service.galaxy_url + '/api/jobs/' + job_id,
            headers=service._headers, allow_redirects=False, timeout=30)
        response.raise_for_status()
    if service._job_details(job_id).get('state') not in {'deleted', 'deleted_new'}:
        raise RuntimeError('Waiting for cancellation acknowledgment before re-staging this input')
    if failed_input_proof(service, job_id, retiring=True) != proof:
        raise RuntimeError('The retired input proof changed; no replacement was submitted')
