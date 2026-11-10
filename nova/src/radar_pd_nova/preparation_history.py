"""History recovery using one captured base and small incremental snapshots."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import re
from types import SimpleNamespace
import uuid
import zipfile

from gpx_recipe.execution import write_json
from gpx_recipe.preparation.library import MAX_ARCHIVE_BYTES as MAX_BYTES, validate_manifest
from gpx_recipe.symmetry.compiler import file_sha
from gpx_recipe.symmetry.contracts import digest
from .model_runtime import document_label, identifier
from .preparation_queue import PreparationQueue, MAX_METADATA

SCHEMA='radar-preparation-history/v1'
STATE_LABEL='RADAR-PD preparation state '
INPUT_LABEL='RADAR-PD preparation inputs '


def named(service, label):
    return [r for r in service.search_history_datasets(query=label,limit=200,include_generated=True)
            if document_label(r.get('name'))==label and r.get('state')=='ok'
            and not r.get('deleted') and not r.get('purged')]


def saved_preparations(service):
    """Return the latest ready snapshot of each queue in this History."""
    selected={}
    rows=service.search_history_datasets(query=STATE_LABEL,limit=500,include_generated=True)
    # Galaxy's general picker sorts by update_time. Editing an old HDA must
    # not make an old preparation intent appear newer than its replacement.
    rows=sorted(enumerate(rows),key=lambda pair:(str(pair[1].get('create_time') or ''),-pair[0]),reverse=True)
    for _,row in rows:
        label=document_label(row.get('name'));key=label.removeprefix(STATE_LABEL)
        if (label!=STATE_LABEL+key or not re.fullmatch('[a-f0-9]{32}',key)
                or row.get('state')!='ok' or row.get('deleted') or row.get('purged')):continue
        original=str(row.get('name','')).rsplit('|',1)[-1].strip()
        name=original.removesuffix('.json.gz') if '|' in str(row.get('name','')) else 'Saved preparation'
        selected.setdefault(key,dict(queue_id=key,dataset_id=identifier(row['id']),name=name))
    return list(selected.values())


class PreparationHistory:
    def __init__(self, root, library, runtime, *, queue_id=None):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.library,self.runtime,self.service=library,runtime,runtime.service
        self.path=self.root/'history-reference.json'
        if self.path.exists():
            if self.path.is_symlink() or self.path.stat().st_size>8192:
                raise ValueError('Preparation History reference is absent or oversized')
            self.reference=json.loads(self.path.read_text())
            self._validate_reference(self.reference)
            if queue_id is not None and self.reference['queue_id']!=queue_id:
                raise ValueError('History reference belongs to another preparation queue')
        else:
            key=queue_id or uuid.uuid4().hex
            self.reference=dict(schema=SCHEMA,history_id=self.service.history_id,queue_id=key,
                library_id=library.document['id'],base=None,last_dataset=None,last_fingerprint=None)
            self._validate_reference(self.reference);write_json(self.path,self.reference)

    def _validate_reference(self, value):
        if (not isinstance(value,dict) or set(value)!={'schema','history_id','queue_id','library_id',
                'base','last_dataset','last_fingerprint'} or value['schema']!=SCHEMA
                or value['history_id']!=self.service.history_id
                or value['library_id']!=self.library.document['id']
                or not re.fullmatch('[a-f0-9]{32}',str(value['queue_id']))):
            raise ValueError('Invalid preparation History reference')
        if value['base'] is not None:self._validate_base(value['base'])
        if value['last_dataset'] is not None:identifier(value['last_dataset'])
        if value['last_fingerprint'] is not None and not re.fullmatch('[a-f0-9]{64}',str(value['last_fingerprint'])):
            raise ValueError('Invalid preparation snapshot fingerprint')

    def _validate_base(self, base):
        if (not isinstance(base,dict) or set(base)!={'dataset_id','sha256','library_id'}
                or base['library_id']!=self.library.document['id']
                or not re.fullmatch('[a-f0-9]{64}',str(base['sha256']))):
            raise ValueError('Invalid captured preparation base')
        identifier(base['dataset_id'])

    def _upload(self, path, label, fingerprint):
        bounded=getattr(self.service,'upload_checkpoint_document',None)
        if bounded:return bounded(path,label=label,semantic_sha256=fingerprint)
        return self.service.upload_document(path,label=label,wait_ready=True)

    def _base(self):
        base=self.reference['base']
        if base is not None:
            if not any(r['id']==base['dataset_id'] for r in named(self.service,INPUT_LABEL+base['sha256'])):
                raise ValueError('The captured preparation inputs are no longer ready in this History')
            return base
        path=self.root/'base.zip'
        if not path.exists():self.library.export(path)
        checksum=file_sha(path);label=INPUT_LABEL+checksum
        existing=named(self.service,label)
        dataset=existing[0]['id'] if existing else self._upload(path,label,checksum)
        base=dict(dataset_id=identifier(dataset),sha256=checksum,library_id=self.library.document['id'])
        self._validate_base(base);self.reference['base']=base;write_json(self.path,self.reference)
        return base

    def save(self, state):
        # Reuse the queue's closed validator without constructing another queue
        # or reading/decompressing any unrelated numerical result arrays.
        source=state.get('source') if isinstance(state,dict) else None
        expected=next(a['sha256'] for a in self.library.document['inputs']
                      if a['path']==self.library.document['specification']['source'])
        if (not isinstance(source,dict) or set(source)!={'id','dataset_id','source_sha256'}
                or source['id']!='preparation-'+self.library.document['id']
                or source['source_sha256']!=expected):
            raise ValueError('Snapshot source differs from the captured preparation parent')
        identifier(source['dataset_id'])
        PreparationQueue._validate(SimpleNamespace(library=self.library,runtime=self.runtime,identity=source),state)
        envelope=dict(schema=SCHEMA,history_id=self.service.history_id,queue_id=self.reference['queue_id'],
            base=self._base(),queue=state)
        raw=json.dumps(envelope,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        if len(raw)>MAX_METADATA:raise ValueError('Preparation snapshot exceeds its metadata budget')
        fingerprint=digest(envelope);label=STATE_LABEL+self.reference['queue_id']
        if (fingerprint==self.reference['last_fingerprint'] and any(
                row['id']==self.reference['last_dataset'] for row in named(self.service,label))):
            return self.reference['last_dataset']
        filename=re.sub(r'[^A-Za-z0-9._ -]+','-',self.library.document['name']).strip(' .-')[:100] or 'preparation'
        staging=self.root/'snapshots'/uuid.uuid4().hex;staging.mkdir(parents=True)
        path=staging/(filename+'.json.gz')
        try:
            path.write_bytes(gzip.compress(raw,mtime=0))
            dataset=identifier(self._upload(path,label,fingerprint))
        finally:
            path.unlink(missing_ok=True);staging.rmdir()
        self.reference.update(last_dataset=dataset,last_fingerprint=fingerprint)
        write_json(self.path,self.reference)
        return dataset

    @classmethod
    def recover(cls, root, runtime, registry, dataset_id):
        """Reconstruct only the latest owned queue; active work stays paused."""
        dataset_id=identifier(dataset_id);service=runtime.service
        matches=[r for r in saved_preparations(service) if r['dataset_id']==dataset_id]
        if len(matches)!=1:
            raise ValueError('Select the latest ready preparation checkpoint in this History')
        root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
        path=root/('restore-'+uuid.uuid4().hex+'.json.gz')
        base_path=root/('inputs-'+uuid.uuid4().hex+'.zip')
        try:
            service._download_dataset(dataset_id,path,maximum_bytes=MAX_METADATA)
            with gzip.open(path,'rb') as stream:raw=stream.read(MAX_METADATA+1)
            if len(raw)>MAX_METADATA:raise ValueError('Preparation snapshot is oversized')
            envelope=json.loads(raw)
            if (not isinstance(envelope,dict) or set(envelope)!={'schema','history_id','queue_id','base','queue'}
                    or envelope['schema']!=SCHEMA or envelope['history_id']!=service.history_id
                    or envelope['queue_id']!=matches[0]['queue_id']):
                raise ValueError('Preparation checkpoint belongs to another History or schema')
            base=envelope['base']
            if (not isinstance(base,dict) or set(base)!={'dataset_id','sha256','library_id'}
                    or not re.fullmatch('[a-f0-9]{64}',str(base['sha256']))
                    or not re.fullmatch('[a-f0-9]{24}',str(base['library_id']))
                    or base['library_id']!=envelope['queue'].get('library_id')):
                raise ValueError('Preparation checkpoint has an invalid base')
            if not any(r['id']==base['dataset_id'] for r in named(service,INPUT_LABEL+base['sha256'])):
                raise ValueError('Preparation inputs are not ready in this History')
            service._download_dataset(identifier(base['dataset_id']),base_path,maximum_bytes=MAX_BYTES)
            if file_sha(base_path)!=base['sha256']:raise ValueError('Captured preparation base changed')
            with zipfile.ZipFile(base_path) as archive:
                if archive.getinfo('library.json').file_size>8*1024*1024:
                    raise ValueError('Preparation manifest is oversized')
                manifest=validate_manifest(json.loads(archive.read('library.json')))
            if manifest['id']!=base['library_id']:raise ValueError('Preparation base names another library')
            key=registry.import_archive(base_path);library=registry.get(key)
            history=cls(library.root/'history',library,runtime,queue_id=envelope['queue_id'])
            history.reference.update(base=base,last_dataset=dataset_id,last_fingerprint=digest(envelope))
            write_json(history.path,history.reference)
            source=dict(envelope['queue']['source'],source_path=str(
                library.root/library.document['specification']['source']))
            queue=PreparationQueue(library.root/'queue',library,runtime,source=source,checkpoint=history.save)
            queue.restore(envelope['queue'])
            return key,queue,history
        finally:
            path.unlink(missing_ok=True);base_path.unlink(missing_ok=True)
