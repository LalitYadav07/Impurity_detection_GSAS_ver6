"""Preserve a native model checkpoint as one manually selected recipe input."""
import copy
from pathlib import Path
import shutil
import uuid

from gpx_recipe.symmetry.campaign import verify_receipt
from gpx_recipe.symmetry.compiler import confined, file_sha
from gpx_recipe.symmetry.contracts import digest
from .symmetry_sources import _recorded_source


STRUCTURAL_REQUEST = (
    'Improve this selected structural-model checkpoint using its existing phases. '
    'Refine the existing background, histogram scale and unit-cell parameters in stages. '
    'Keep atomic coordinates, occupancies, displacement parameters, ISODISTORT mode '
    'amplitudes, constraints and instrument calibration fixed. Preserve its fit range '
    'and exclusions. Retain its scientific-review warnings; this request does not '
    'establish a unique space group.'
)


def capture_gpx(row, root):
    """Copy only the exact receipt-bound GPX; never substitute a nearby file."""
    if not row.get('has_gpx') or not row.get('has_view'):
        raise ValueError('Select an independently audited native checkpoint first.')
    receipt = verify_receipt(row['directory'], dict(id=row['id'], protocol_sha256=row['protocol']))
    if digest(receipt) != row.get('receipt_sha256'):
        raise ValueError('The receipt changed; refresh evidence before opening.')
    relative = row['gpx_relative']
    if relative != 'results/' + str(receipt.get('gpx') or ''):
        raise ValueError('The selected GPX does not match the native checkpoint receipt.')
    source = confined(row['directory'], relative)
    expected = receipt.get('files', {}).get(relative)
    if source.suffix.lower() != '.gpx' or not expected:
        raise ValueError('This native GPX is absent from the audited receipt.')
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / (digest(receipt) + '.gpx')
    temporary = root / ('.capture-' + uuid.uuid4().hex)
    try:
        shutil.copyfile(source, temporary)
        if file_sha(temporary) != expected:
            raise ValueError('The selected native GPX changed during capture.')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target, expected


def structural_source(workspace, experiment, row, path, checksum):
    if (row.get('kind') != 'structural' or row.get('status') == 'profile_equivalent'
            or not row.get('native_verified') or not row.get('has_gpx') or not row.get('has_view')):
        raise ValueError('Recipe handoff requires a native structural-model checkpoint.')
    parent = _recorded_source(workspace, experiment, row['source'],
                              expected_sha256=row['source_sha256'])
    if parent is None:
        raise ValueError('Restore this model’s exact published input before opening a recipe.')
    path = Path(path)
    if path.suffix.lower() != '.gpx' or file_sha(path) != checksum:
        raise ValueError('The captured structural GPX changed.')
    source = copy.deepcopy(parent)
    for key in ('dataset_id', 'collection_id', 'collection_name', 'record_uid', 'galaxy_job_id',
                'published_sha256', 'internal_source_sha256', 'origin_source_id',
                'model_parent_id', 'model_parent_sha256', 'recipe_job_id', 'recipe_stage',
                'context', 'context_sha256', 'inspection', 'inspection_revision', 'inspection_error',
                'live_watch', 'observed_utc', 'base_rwp', 'project'):
        source.pop(key, None)
    source.update(
        id='structural-model:' + digest([experiment, row['protocol'], row['id'], checksum])[:24],
        source_path=str(path), source_sha256=checksum, collection_name=path.name,
        scan_key=parent['scan_key'] + ':structural:' + row['id'],
        scan_label=parent['scan_label'] + ' · ' + row['name'],
        project=dict(stage='structural_model_checkpoint', status=row['status']),
        scientific_status=row['status'],
        model_checkpoint=dict(trial_id=row['id'], protocol_sha256=row['protocol'],
                              receipt_sha256=row['receipt_sha256'],
                              input_id=row['source'], input_sha256=row['source_sha256'],
                              worker_sha256=row['worker_revision']),
    )
    metadata = source.setdefault('metadata', {})
    if metadata.get('preparation'):
        metadata['parent_preparation'] = metadata.pop('preparation')
    warnings = list(source.get('scientific_warnings') or [])
    warnings.append('Selected structural-model checkpoint; its fit does not establish a unique space group.')
    if row['status'] != 'valid':
        warnings.append('The structural checkpoint requires scientific review: ' +
                        str(row.get('reason') or row['status'].replace('_', ' ')))
    source['scientific_warnings'] = list(dict.fromkeys(warnings))
    return source
