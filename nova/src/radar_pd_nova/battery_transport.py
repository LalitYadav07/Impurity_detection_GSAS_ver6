"""Submit reviewed preparation through the ordinary native recipe worker."""
import json
from pathlib import Path
import shutil
import zipfile

from gpx_recipe.execution import write_json
from .refinement_sources import file_sha, digest


def write_preparation_packet(inputs, packet):
    """Byte-identical across restored paths, mtimes and ZIP library versions."""
    from gpx_recipe.battery.transport import MAX_MEMBERS, MAX_PACKET_BYTES
    paths = sorted(Path(inputs).iterdir())
    if not 1 <= len(paths) <= MAX_MEMBERS or sum(p.stat().st_size for p in paths) > MAX_PACKET_BYTES:
        raise ValueError('Preparation packet exceeds its input budget')
    remaining = MAX_PACKET_BYTES
    with zipfile.ZipFile(packet, 'w', compression=zipfile.ZIP_STORED) as archive:
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise ValueError('Preparation inputs must be regular captured files')
            entry = zipfile.ZipInfo(path.name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100444 << 16
            with path.open('rb') as source:
                data = source.read(remaining + 1)
            if len(data) > remaining:
                raise ValueError('Preparation input grew beyond its size budget')
            remaining -= len(data)
            archive.writestr(entry, data)


def prepare_on_galaxy(runtime, job):
    job = Path(job)
    packet = job / 'preparation-inputs.zip'
    inputs = job / 'inputs'
    write_preparation_packet(inputs, packet)
    checksum = file_sha(packet)
    from gpx_recipe.battery.transport import implementation_signature
    signature = implementation_signature()
    ledger = job / 'preparation-dispatch.json'
    recovering = ledger.exists()
    if recovering:
        record = json.loads(ledger.read_text())
        if record.get('implementation_sha256') != signature:
            raise ValueError('Preparation implementation changed; retain the old receipt and prepare a new revision')
        source = record['source']
        if source['source_sha256'] != checksum:
            raise ValueError('Preparation inputs changed after dispatch; prepare a new revision')
    else:
        dataset = runtime.service.upload_document(packet, label='RADAR-PD reviewed preparation inputs')
        source = dict(source_path=str(packet), source_sha256=checksum, dataset_id=dataset)
        write_json(ledger, dict(source=source, implementation_sha256=signature))
    response = runtime.call('prepare', source, job / 'remote',
                            plan=dict(implementation_sha256=signature),
                            dispatch_key=digest(['prepare', checksum, signature])[:24], recover=recovering)
    if response.get('status') != 'model_prepared' or not response.get('model_dataset_id'):
        raise ValueError(response.get('error') or 'Native preparation did not publish its verified GPX')
    if response.get('implementation_sha256') != signature:
        raise ValueError('Prepared response does not identify the requested native implementation')
    result = response['preparation']
    if result.get('project_sha256') != response.get('output_sha256'):
        raise ValueError('Preparation receipt differs from the published model')
    # Capture the checked native response beside the local input receipt. The
    # Galaxy IDs, source hash and immutable packet remain recoverable evidence.
    if (job / 'model').exists():
        if (file_sha(job/'model/prepared.gpx') != response['output_sha256']
                or json.loads((job/'model/preparation.json').read_text()) != result):
            raise ValueError('The retained prepared model differs from its original receipt; use a fresh revision')
    else:
        shutil.copytree(job / 'remote' / 'model', job / 'model')
    write_json(job / 'preparation-publication.json', dict(
        dataset_id=response['model_dataset_id'], source_sha256=response['output_sha256'],
        galaxy_job_id=response.get('galaxy_job_id'), response=response))
    return result
