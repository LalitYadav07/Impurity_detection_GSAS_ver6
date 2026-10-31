"""Prove a Live input was interrupted before retiring its unstarted job."""
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
                and not metadata.get('deleted') and not metadata.get('purged')):
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
