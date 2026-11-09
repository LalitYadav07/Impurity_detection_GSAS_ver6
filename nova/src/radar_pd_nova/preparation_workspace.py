"""Shared preparation workspace for setup, Model exploration and Live clients.

This coordinator handles data-only libraries and owned Galaxy job references.
Scientific construction remains in the native preparation worker. It never
loads GSAS-II or treats reference calculations as measured diffraction scans.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from gpx_recipe.execution import write_json
from gpx_recipe.preparation.library import confined,MAX_MODELS
from gpx_recipe.symmetry.chemistry import validate_sites,sites_from_inputs
from gpx_recipe.symmetry.compiler import file_sha, package_signature
from gpx_recipe.symmetry.contracts import digest, validate_candidate
from gpx_recipe.structural_live.defaults import DEFAULT_CELL_RADIUS
from .model_runtime import identifier
from .native_operations import intent_lock
from .preparation_history import PreparationHistory, saved_preparations
from .preparation_queue import PreparationQueue
from .preparation_registry import PreparationRegistry


class PreparationWorkspace:
    def __init__(self,root,runtime):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.runtime=runtime;self.registry=PreparationRegistry(self.root/'registry')
        self._queues={}

    def _source(self,library):
        path=library.root/'galaxy-source.json'
        expected=next(a['sha256'] for a in library.document['inputs']
            if a['path']==library.document['specification']['source'])
        with intent_lock(path):
            if path.exists():
                if path.is_symlink() or path.stat().st_size>8192:raise ValueError('Invalid preparation source reference')
                value=json.loads(path.read_text())
            else:
                # Recovery can restore an acknowledged queue before a local
                # source-reference write was acknowledged. Preserve that HDA.
                queue=library.root/'queue/queue.json'
                if queue.exists():
                    if queue.is_symlink() or queue.stat().st_size>64*1024*1024:
                        raise ValueError('Invalid recovered preparation queue')
                    state=json.loads(queue.read_text())
                    if state.get('history_id')!=self.runtime.service.history_id or state.get('library_id')!=library.document['id']:
                        raise ValueError('Recovered preparation belongs to another History')
                    source=state.get('source',{})
                    if source.get('source_sha256')!=expected:raise ValueError('Recovered parent changed')
                    dataset=identifier(source.get('dataset_id'))
                else:
                    source=confined(library.root,library.document['specification']['source'])
                    dataset=identifier(self.runtime.service.upload_document(source,
                        label='RADAR-PD preparation parent '+expected,wait_ready=True))
                value=dict(history_id=self.runtime.service.history_id,library_id=library.document['id'],
                    dataset_id=dataset,source_sha256=expected)
                write_json(path,value)
            if (not isinstance(value,dict) or set(value)!={'history_id','library_id','dataset_id','source_sha256'}
                    or value['history_id']!=self.runtime.service.history_id
                    or value['library_id']!=library.document['id'] or value['source_sha256']!=expected):
                raise ValueError('Preparation source belongs to another parent or History')
            dataset=identifier(value['dataset_id'])
        return dict(id='preparation-'+library.document['id'],dataset_id=dataset,
            source_sha256=expected,source_path=str(confined(library.root,library.document['specification']['source'])))

    def queue(self,key):
        library=self.registry.get(key)
        if key not in self._queues:
            history=PreparationHistory(library.root/'history',library,self.runtime)
            self._queues[key]=PreparationQueue(library.root/'queue',library,self.runtime,
                source=self._source(library),checkpoint=history.save)
        return self._queues[key]

    def inspect(self,*,name,source,banks):
        """Capture an inspection plan. Starting it is a separate explicit action."""
        models=[dict(kind='reference',bank=label,settings=dict(maximum_denominator=8)) for label in banks]
        return self.registry.create(name=name,source=source,banks=banks,models=models)

    def inspect_from_library(self,key):
        library=self.registry.get(key);banks={}
        for row in library.document['entries']:
            model=row['model'];bank=dict(instrument=str(confined(library.root,model['instrument'])),grid=model['grid'])
            if model['bank'] in banks and banks[model['bank']]!=bank:
                raise ValueError('Disambiguate the different profile contexts sharing one bank label')
            banks[model['bank']]=bank
        library.verify_assets(list(dict.fromkeys([library.document['specification']['source']]+[
            row['model']['instrument'] for row in library.document['entries']])))
        return self.inspect(name=library.document['name'][:160]+' · input inspection',
            source=confined(library.root,library.document['specification']['source']),banks=banks)

    def start(self,key,*,maximum_jobs=2):
        queue=self.queue(key);queue.resume(maximum_jobs=maximum_jobs)
        return queue.snapshot()

    def pause(self,key):
        self.queue(key).pause()

    def tick(self,key):
        return self.queue(key).tick()

    def recover(self,dataset):
        key,queue,_=PreparationHistory.recover(self.root/'recovery',self.runtime,self.registry,dataset)
        self._queues[key]=queue
        # Capture the original parent HDA, rather than uploading an unrelated
        # replacement when this workspace is reopened a second time.
        self._source(queue.library)
        return key

    def saved(self):
        return saved_preparations(self.runtime.service)

    def import_archive(self,path):
        key=self.registry.import_archive(path)
        detached=self.registry.get(key).detach_imported_work()
        return dict(library=key,detached=detached,
            message=('Imported completed assets. Unfinished entries need new jobs; originating jobs were not changed.'
                if detached else 'Imported the verified portable library. No calculations were started.'))

    def references(self,key):
        """Read only receipt-checked native parent/profile inspections."""
        library=self.registry.get(key);rows=library.document['entries'];references=[]
        if any(r['model']['kind']!='reference' for r in rows):
            raise ValueError('Select a parent/profile inspection for candidate discovery')
        if not library.coverage()['all_prepared']:
            raise ValueError('Complete the selected parent and bank inspections first')
        # Portable metadata is not a native signature. Reinspect a foreign
        # reference before allowing it to determine symmetry discovery. Owned
        # History recovery retains the original receipt/attempt lineage.
        if not (library.root/'queue/queue.json').is_file():
            raise ValueError('Inspect this library’s captured inputs before using imported parent metadata for discovery')
        state=self.queue(key).snapshot()
        confirmed={(member['entry'],member['attempt']) for job in state['jobs']
            if job['phase']=='collected' and job['result_sha256']
            for member in job['members']
            if job['outcomes'].get(member['entry'],{}).get('state')=='prepared'
            and job['outcomes'][member['entry']].get('source')=='worker'}
        if any((row['id'],row['attempt']) not in confirmed for row in rows):
            raise ValueError('Inspect the captured inputs or recover their original native inspection receipts before discovery')
        parent=None;identity=None;implementation=package_signature()
        for row in rows:
            assets=library.prepared_assets(row['id'])
            path=next(p for p in assets.values() if p.name=='entry.json')
            if path.stat().st_size>8*1024*1024:raise ValueError('Native reference metadata is oversized')
            manifest=json.loads(path.read_text());model=row['model']
            expected=[file_sha(confined(library.root,name)) for name in
                (library.document['specification']['source'],model['instrument'])]
            if (manifest.get('schema')!='radar-experiment-preparation-entry/v1' or manifest.get('kind')!='reference'
                    or manifest.get('implementation_sha256')!=implementation
                    or manifest.get('input_sha256')!=expected or manifest.get('settings')!=model['settings']
                    or manifest.get('grid')!=model['grid']
                    or manifest.get('observations_included') is not False or manifest.get('scientific_result') is not False):
                raise ValueError('Parent inspection changed or needs the current native implementation')
            from gpx_recipe.preparation.operations import portable_entry_artifacts
            portable_entry_artifacts('reference',manifest)
            detail=manifest.get('detail')
            if (not isinstance(detail,dict) or not isinstance(detail.get('parent'),dict)
                    or not isinstance(detail.get('points'),dict) or not isinstance(manifest.get('native_runtime'),dict)
                    or not {'name','type','setting_sha256','space_group','cell','operations','atoms'}<=set(detail['parent'])
                    or not isinstance(detail['points'].get('candidates'),list)):
                raise ValueError('The native parent inspection is incomplete or malformed')
            current=copy.deepcopy(detail['parent']);current.pop('histograms',None)
            current_identity=digest(current)
            if identity is not None and current_identity!=identity:
                raise ValueError('Bank inspections produced different parent structures or settings')
            parent=current;identity=current_identity
            references.append(dict(bank=model['bank'],instrument=str(confined(library.root,model['instrument'])),
                grid=copy.deepcopy(model['grid']),points=copy.deepcopy(detail['points']),
                native_runtime=copy.deepcopy(manifest['native_runtime'])))
        return dict(library=key,parent=parent,banks=references,
            source=str(confined(library.root,library.document['specification']['source'])),
            source_sha256=file_sha(confined(library.root,library.document['specification']['source'])))

    def create_models(self,reference,*,name,kind,catalogs,bank_labels=None,sites=None,
                      maximum_moment=6.,mode_radius=.05,cell_radius=DEFAULT_CELL_RADIUS,
                      refine_scale=True,refine_background=True):
        context=self.references(reference)
        if kind not in {'magnetic','structural'}:raise ValueError('Choose magnetic or structural model preparation')
        if not isinstance(catalogs,list) or not catalogs:raise ValueError('Discover a finite candidate scope first')
        if kind=='magnetic':
            if not isinstance(sites,list) or not sites:raise ValueError('Review the magnetic sites and ionic form factors')
            sites=sites_from_inputs([v['label'] for v in sites],
                ';'.join(v['label']+'='+v['form_factor'] for v in sites),
                ';'.join(v['label']+'='+str(v['lande_g']) for v in sites))
            validate_sites(context['parent']['atoms'],sites)
        banks={row['bank']:dict(instrument=row['instrument'],grid=row['grid']) for row in context['banks']}
        labels=sorted(banks) if bank_labels is None else list(bank_labels)
        if not labels or len(labels)!=len(set(labels)) or any(label not in banks for label in labels):
            raise ValueError('Select unique banks from the completed native inspection')
        banks={label:banks[label] for label in labels};models=[];searches=[];seen=set()
        for descriptor in catalogs:
            catalog=descriptor['catalog'];vector=catalog['k']
            if (catalog.get('schema')!='radar-symmetry-catalog/v1'
                    or catalog.get('parent_setting_sha256')!=context['parent']['setting_sha256']
                    or catalog.get('coverage',{}).get('discovery_complete') is not True):
                raise ValueError('Resolve incomplete discovery or a changed parent setting before preparation')
            token=digest(vector)
            if token in seen:raise ValueError('A propagation-vector family was declared twice')
            seen.add(token)
            if not catalog.get('candidates'):raise ValueError('The selected family has no supported candidate models')
            if kind=='structural':
                from .preparation_discovery import require_structural_coverage
                require_structural_coverage(catalog)
            if len(models)+len(catalog['candidates'])*len(labels)>MAX_MODELS:
                raise ValueError('The declared scope exceeds one portable library’s model limit; split it into named libraries. No candidates were dropped.')
            candidates=[validate_candidate(c) for c in catalog['candidates']]
            for candidate in candidates:
                if kind=='magnetic':
                    settings=dict(candidate=candidate,k=vector,sites=sites,maximum_moment=maximum_moment,
                        refine_histogram_scale=refine_scale,refine_background=refine_background)
                    child=None
                else:
                    directory=Path(descriptor['directory']).resolve()
                    if not directory.is_relative_to(self.root):raise ValueError('Structural discovery belongs to another workspace')
                    original=directory/candidate['child']['path'];child=original.resolve()
                    if not child.is_relative_to(directory) or original.is_symlink() or file_sha(child)!=candidate['child']['sha256']:
                        raise ValueError('The authoritative structural child changed')
                    settings=dict(candidate=candidate,mode_radius=mode_radius,cell_radius=cell_radius)
                for label in labels:models.append(dict(kind=kind,bank=label,settings=settings,child=child))
            searches.append(dict(kind=kind,k=copy.deepcopy(vector),catalog_sha256=digest(catalog),
                selected_candidates=[c['id'] for c in candidates],coverage=copy.deepcopy(catalog['coverage'])))
        scope=dict(schema='radar-preparation-declared-scope/v1',parent_cif_sha256=context['source_sha256'],
            selected_banks=sorted(banks),searches=searches)
        # A model library carries its completed shared profile dependencies,
        # rather than requiring the user to select a second inspection ZIP.
        inspection=self.registry.get(reference)
        references=[r for r in inspection.document['entries'] if kind=='structural' and r['model']['bank'] in banks
            and {'profile.npz','profile.json'}<={Path(a['path']).name for a in r['files']}]
        if kind=='structural':
            from gpx_recipe.preparation.structural_dependencies import example_models
            models+=example_models(models,bank_grids={label:row['grid'] for label,row in banks.items()})
        models += [dict(kind='reference',bank=r['model']['bank'],settings=r['model']['settings']) for r in references]
        if len(models)>MAX_MODELS:raise ValueError('Model and shared-profile dependencies exceed the library entry bound')
        key=self.registry.create(name=name,source=context['source'],banks=banks,models=models,scope=scope)
        from gpx_recipe.preparation.operations import inherit_reference
        library=self.registry.get(key)
        for row in references:inherit_reference(library,inspection,row['id'])
        return key
