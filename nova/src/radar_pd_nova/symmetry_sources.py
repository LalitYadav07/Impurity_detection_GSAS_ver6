"""Explicit, verified handoff of RADAR-PD and completed recipe checkpoints.

Recipe outputs are offered only to model exploration. They are never registered
as incoming RADAR-PD sources, preventing recursive recipe enrollment.
"""
from __future__ import annotations

import copy
from pathlib import Path
from math import isfinite

from .refinement_runtime import verify_result_files
from .refinement_sources import digest, file_sha
from .scan_conditions import qualified_metadata
from .scan_conditions import condition_display


def recipe_source(workspace, job):
    """Fill missing acquisition conditions only from the exact original GPX.

    A completed job keeps its captured source immutable. A later catalog may
    recover qualified readbacks that were absent at submission; these may label
    its derived checkpoints only when the original bytes and scan scope agree.
    Explicit old observations, including rejected channels, are never replaced.
    """
    source=copy.deepcopy(job.get('source') or {})
    current=workspace.state.get('sources',{}).get(source.get('id')) or {}
    checksum=source.get('source_sha256')
    if (not isinstance(checksum,str) or len(checksum)!=64 or
            any(c not in '0123456789abcdef' for c in checksum) or
            current.get('source_sha256')!=checksum or
            any(current.get(key)!=source.get(key) for key in
                ['id','experiment','scan_key','sample_key','run_number','bank',
                 'collection_id','collection_name','internal_source_sha256','published_sha256'])):
        return source
    metadata=source.setdefault('metadata',{})
    if not isinstance(metadata,dict):return source
    old_packet=metadata.get('conditions') or {}
    old_channels=(old_packet.get('channels') or []) if isinstance(old_packet,dict) else []
    if not isinstance(old_channels,list):return source
    donor=qualified_metadata(current.get('metadata') or {})
    added=[]
    for quantity in ['temperature','magnetic_field','pressure','voltage']:
        if (quantity in metadata or quantity+'_observation' in metadata or
                any(isinstance(c,dict) and c.get('quantity')==quantity for c in old_channels)):
            continue
        metric=donor.get(quantity) or {}
        if not isinstance(metric,dict):continue
        value=metric.get('value')
        if (metric.get('role')!='measured' or metric.get('status')!='recorded' or
                not isinstance(value,(int,float)) or isinstance(value,bool) or
                not isfinite(value) or not metric.get('unit')):
            continue
        metadata[quantity]=copy.deepcopy(metric);added.append(quantity)
    if added:
        source['acquisition_metadata_recovery']=dict(
            source_id=source['id'],source_sha256=checksum,quantities=added,
            metadata_sha256=digest(current.get('metadata') or {}))
    return source


def _scan_title(source):
    title=source['scan_label']
    condition=qualified_metadata(source.get('metadata') or {}).get('temperature') or {}
    value=condition.get('value')
    if isinstance(value,(int,float)) and not isinstance(value,bool) and isfinite(value) and condition.get('unit'):
        title+=f" · {value:.4g} {condition['unit']}"
    return title


def start_choices(workspace, experiment):
    entries=[]
    for source in workspace.sources(experiment):
        entries.append(dict(id=source['id'],title='RADAR-PD · '+_scan_title(source),source=source))
    for job_id,job in workspace.state.get('jobs',{}).items():
        result=job.get('result') or {};source=recipe_source(workspace,job)
        if job.get('status')!='completed' or result.get('status')!='completed' or source.get('experiment')!=experiment:
            continue
        recipe_id=job.get('recipe_id') or next(iter(job.get('recipes') or []),'')
        version=(workspace.state.get('recipes',{}).get(recipe_id) or {}).get('version')
        recipe_label='Recipe '+str(version) if version is not None else 'Recipe checkpoint'
        checkpoints=[('final','Final recipe output',result.get('final_gpx'))]
        checkpoints.extend((stage['id'],stage.get('title') or stage['id'],stage.get('gpx'))
                           for stage in result.get('stages',[]))
        for stage,label,name in checkpoints:
            if not name or name not in result.get('files',{}) or Path(name).suffix.lower()!='.gpx':continue
            identifier='recipe:'+digest([job_id,stage,name,result['files'][name]])[:24]
            entries.append(dict(id=identifier,title=recipe_label+' · '+_scan_title(source)+' · '+label+' · '+result['files'][name][:8],
                                recipe_label=recipe_label,
                                job_id=job_id,stage=stage,name=name))
    return entries


def acquisition_details(workspace,experiment,source_id):
    """Show the recorded interval summary without loading a GPX or model file."""
    if workspace is None:return []
    entry=next((e for e in start_choices(workspace,experiment) if e['id']==source_id),None)
    if not entry:return []
    source=entry.get('source') or recipe_source(workspace,workspace.state['jobs'][entry['job_id']])
    metadata=qualified_metadata(source.get('metadata') or {})
    return [dict(label='Measured acquisition condition',value=text)
            for quantity,label in [('temperature','Sample temperature'),('magnetic_field','Magnetic field'),
                                   ('pressure','Pressure'),('voltage','Voltage')]
            if (text:=condition_display(metadata.get(quantity),label))]


def resolve_start(workspace, experiment, identifier):
    entries=[entry for entry in start_choices(workspace,experiment) if entry['id']==identifier]
    if len(entries)!=1:raise ValueError('This starting checkpoint changed or is outside the selected experiment; choose it again')
    entry=entries[0]
    if 'source' in entry:return dict(entry['source'])
    job=workspace.state['jobs'][entry['job_id']];result=job['result']
    if not result.get('directory') or not Path(result['directory']).is_dir():
        if not job.get('remote_action'):raise ValueError('The completed recipe artifacts are unavailable locally')
        workspace.recover_job_artifacts(entry['job_id']);result=job['result']
    # Rebuild after recovery; the remote receipt may not describe the same revision.
    current=[x for x in start_choices(workspace,experiment) if x['id']==identifier]
    if len(current)!=1:raise ValueError('The recovered recipe checkpoint is a different revision')
    verify_result_files(result['directory'],result)
    root=Path(result['directory']).resolve();path=(root/entry['name']).resolve()
    if not path.is_relative_to(root) or path.suffix.lower()!='.gpx' or not path.is_file():
        raise ValueError('The selected recipe GPX is not confined to its verified result bundle')
    checksum=file_sha(path)
    if checksum!=result['files'][entry['name']]:raise ValueError('The selected recipe checkpoint changed')
    source=recipe_source(workspace,job)
    # The derived bytes need their own native inspection. Neither an original
    # Galaxy reference nor its cached project context may describe this GPX.
    for key in ['collection_id','dataset_id','collection_name','published_sha256','project','base_rwp',
                'context','inspection','inspection_error','context_sha256']:
        source.pop(key,None)
    dataset=(result.get('final_dataset_id') if entry['stage']=='final' else
             (result.get('stage_dataset_ids') or {}).get(entry['stage']))
    source.update(id=identifier,source_path=str(path),source_sha256=checksum,
                  scan_label=source['scan_label']+' · '+entry['recipe_label']+' · '+('final' if entry['stage']=='final' else entry['stage']),
                  origin_source_id=job['source']['id'],recipe_job_id=entry['job_id'],recipe_stage=entry['stage'],
                  project=dict(stage='verified_recipe_checkpoint',status='completed'))
    if dataset:source['dataset_id']=dataset
    return source
