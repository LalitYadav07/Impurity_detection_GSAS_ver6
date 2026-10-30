"""Explicit, verified handoff of RADAR-PD and completed recipe checkpoints.

Recipe outputs are offered only to model exploration. They are never registered
as incoming RADAR-PD sources, preventing recursive recipe enrollment.
"""
from __future__ import annotations

from pathlib import Path

from .refinement_runtime import verify_result_files
from .refinement_sources import digest, file_sha


def start_choices(workspace, experiment):
    entries=[]
    for source in workspace.sources(experiment):
        entries.append(dict(id=source['id'],title='RADAR-PD · '+source['scan_label'],source=source))
    for job_id,job in workspace.state.get('jobs',{}).items():
        result=job.get('result') or {};source=job.get('source') or {}
        if job.get('status')!='completed' or result.get('status')!='completed' or source.get('experiment')!=experiment:
            continue
        checkpoints=[('final','Final recipe output',result.get('final_gpx'))]
        checkpoints.extend((stage['id'],stage.get('title') or stage['id'],stage.get('gpx'))
                           for stage in result.get('stages',[]))
        for stage,label,name in checkpoints:
            if not name or name not in result.get('files',{}) or Path(name).suffix.lower()!='.gpx':continue
            identifier='recipe:'+digest([job_id,stage,name,result['files'][name]])[:24]
            entries.append(dict(id=identifier,title='Recipe output · '+source['scan_label']+' · '+label,
                                job_id=job_id,stage=stage,name=name))
    return entries


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
    source=dict(job['source'])
    # The derived bytes need their own native inspection. Neither an original
    # Galaxy reference nor its cached project context may describe this GPX.
    for key in ['collection_id','dataset_id','collection_name','published_sha256','project','base_rwp',
                'context','inspection','inspection_error','context_sha256']:
        source.pop(key,None)
    dataset=(result.get('final_dataset_id') if entry['stage']=='final' else
             (result.get('stage_dataset_ids') or {}).get(entry['stage']))
    source.update(id=identifier,source_path=str(path),source_sha256=checksum,
                  scan_label=source['scan_label']+' · '+('final recipe output' if entry['stage']=='final' else 'recipe stage '+entry['stage']),
                  origin_source_id=job['source']['id'],recipe_job_id=entry['job_id'],recipe_stage=entry['stage'],
                  project=dict(stage='verified_recipe_checkpoint',status='completed'))
    if dataset:source['dataset_id']=dataset
    return source
