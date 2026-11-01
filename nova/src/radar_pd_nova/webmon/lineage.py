"""Bind a provisional fitted GPX to its measured snapshot, not its reference."""
import copy
import hashlib
import json
from pathlib import Path
import re


def posterior_source(parent, *, revision, path, checksum, rwp, scope=None):
    match=re.fullmatch(r'PG3_(\d+):([a-f0-9]{64})',str(revision))
    if not match or not re.fullmatch(r'[a-f0-9]{64}',str(checksum)):
        raise ValueError('A posterior requires exact run, snapshot and output identities')
    scope=copy.deepcopy(scope or {})
    if scope.get('run') and str(scope['run'])!=match[1]:
        raise ValueError('Posterior acquisition and reviewed live scope disagree')
    bank=str(parent.get('bank') or '')
    if not bank.isdigit() or int(bank)<1:raise ValueError('A reviewed physical bank is required for handoff')
    source=copy.deepcopy(parent)
    ancestry={k:copy.deepcopy(parent.get(k)) for k in
              ('id','source_sha256','scan_key','run_id','run_number','scan_label','metadata')}
    for key in ('dataset_id','collection_id','collection_name','galaxy_job_id',
                'context','inspection','inspection_revision','live_watch','acquisition'):
        source.pop(key,None)
    identity=hashlib.sha256(json.dumps([parent['experiment'],parent['id'],parent['source_sha256'],revision,checksum,bank],
                                      separators=(',',':')).encode()).hexdigest()[:24]
    sample_id=str(scope.get('sample_id') or '')
    source.update(id='live-posterior-'+identity,record_uid='live-posterior-'+identity,
        scan_key='live-posterior-'+identity,run_id='PG3_'+match[1],run_number=int(match[1]),
        scan_label=f'Run {match[1]} · bank {bank} · live snapshot {match[2][:12]}',
        sample_key='id:'+sample_id if sample_id else 'live-unknown:'+identity,
        sample_label=str(scope.get('sample') or 'Unresolved live sample'),bank=bank,
        source_path=str(path),source_sha256=checksum,internal_source_sha256=checksum,
        published_sha256=checksum,collection_name=Path(path).name,base_rwp=rwp,
        origin='conditional_live_posterior',parent_source_sha256=parent['source_sha256'],
        snapshot_revision=revision,reference_ancestry=ancestry,axis_overrides={},project={},
        metadata={'run_number':int(match[1]),'source_bank':bank,'sample_id':sample_id,
                  'source_kind':'provisional_live_snapshot'},
        scientific_status='conditional',
        scientific_warnings=list(dict.fromkeys([*(parent.get('scientific_warnings') or []),
            'Conditional live fit; reference-run conditions are not target-scan metadata.'])))
    return source
