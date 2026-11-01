"""Submit reviewed preparation through the ordinary native recipe worker."""
import json
from pathlib import Path
import shutil
import zipfile

from gpx_recipe.execution import write_json
from .refinement_sources import file_sha, digest


def prepare_on_galaxy(runtime, job):
    job = Path(job)
    packet = job / 'preparation-inputs.zip'
    inputs = job / 'inputs'
    with zipfile.ZipFile(packet, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(inputs.iterdir()):
            if path.is_symlink() or not path.is_file():
                raise ValueError('Preparation inputs must be regular captured files')
            archive.write(path, path.name)
    checksum = file_sha(packet)
    ledger = job / 'preparation-dispatch.json'
    if ledger.exists():
        source = json.loads(ledger.read_text())
        if source['source_sha256'] != checksum:
            raise ValueError('Preparation inputs changed after dispatch; prepare a new revision')
    else:
        dataset = runtime.service.upload_document(packet, label='RADAR-PD reviewed preparation inputs')
        source = dict(source_path=str(packet), source_sha256=checksum, dataset_id=dataset)
        write_json(ledger, source)
    response = runtime.call('prepare', source, job / 'remote',
                            dispatch_key=digest(['prepare', checksum])[:24], recover=True)
    if response.get('status') != 'model_prepared' or not response.get('model_dataset_id'):
        raise ValueError(response.get('error') or 'Native preparation did not publish its verified GPX')
    result = response['preparation']
    if result.get('project_sha256') != response.get('output_sha256'):
        raise ValueError('Preparation receipt differs from the published model')
    # Capture the checked native response beside the local input receipt. The
    # Galaxy IDs, source hash and immutable packet remain recoverable evidence.
    if (job / 'model').exists():
        raise ValueError('A prepared model already exists; use a fresh revision')
    shutil.copytree(job / 'remote' / 'model', job / 'model')
    write_json(job / 'preparation-publication.json', dict(
        dataset_id=response['model_dataset_id'], source_sha256=response['output_sha256'],
        galaxy_job_id=response.get('galaxy_job_id'), response=response))
    return result
