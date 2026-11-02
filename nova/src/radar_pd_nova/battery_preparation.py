"""Battery model preparation controls; scientific work runs in a separate process."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import uuid
import zipfile

from gpx_recipe.battery.preparation import SCHEMA, ROLES, validate_preparation, validate_sample_widths
from gpx_recipe.battery.preparation import file_hash
from gpx_recipe.jobs import scientific_environment
from gpx_recipe.process import run_scientific
from . import nomad
from .battery import powgen


def product_adapter(instrument):
    if instrument not in {'NOM', 'PG3'}:
        raise ValueError('Choose NOMAD or POWGEN for battery preparation')
    return nomad if instrument == 'NOM' else powgen


def invalid_count_error(instrument):
    return 'Nonfinite data or nonpositive uncertainty/TOF in ' + ('NOMAD' if instrument == 'NOM' else 'POWGEN') + ' bank'


def checked_upload(widget, selected, *, maximum=1024*1024):
    path = getattr(widget, "current_path", None)
    if path is None or str(path) != selected or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("Choose a completed file within the displayed upload limit")
    return path


def capture_file(path, destination, *, maximum, expected_signature=None, expected_sha256=None):
    """Freeze one bounded file and detect changes during its read."""
    path = Path(path)
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        if expected_signature and (before.st_size,before.st_mtime_ns)!=expected_signature:
            raise ValueError('Input changed; inspect its source again')
        raw = stream.read(maximum+1)
        after = os.fstat(stream.fileno())
    if len(raw)>maximum:
        raise ValueError('Input exceeds its capture limit')
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or nomad.signature(path)!=(before.st_size,before.st_mtime_ns):
        raise ValueError('Input changed during capture')
    checksum = sha256(raw).hexdigest()
    if expected_sha256 and checksum!=expected_sha256:
        raise ValueError('Input content changed; inspect its source again')
    Path(destination).write_bytes(raw)
    return dict(path=Path(destination).name,sha256=checksum)


def battery_experiment_id(workspace, acquisition, family):
    """Keep an existing unmixed legacy cell recoverable; isolate other cells."""
    from .refinement_sources import digest
    parts=['battery-prepared',workspace.history_id,acquisition['instrument'],
           acquisition['ipts'],acquisition['sample_id']]
    legacy=digest(parts+[family])[:24]
    prior=[s for s in workspace.state['sources'].values() if s.get('experiment')==legacy]
    prior += [j['source'] for j in workspace.state.get('jobs',{}).values()
              if j['source'].get('experiment')==legacy]
    if prior and all(s.get('metadata',{}).get('cell_id')==acquisition['cell_id'] for s in prior):
        return legacy
    return digest(parts+[acquisition['cell_id'],family])[:24]


def run_sources(folder, run, maximum_entries=50_000, *, instrument='NOM'):
    """Filter names before stat calls on a potentially slow archive mount."""
    rows=[]
    with os.scandir(folder) as entries:
        for index,entry in enumerate(entries):
            if index>=maximum_entries:
                raise ValueError('Reduced folder exceeds the bounded run lookup; narrow the source folder')
            if product_adapter(instrument).run_number_from_filename(entry.name)!=run or not entry.is_file(follow_symlinks=False):
                continue
            info=entry.stat(follow_symlinks=False)
            rows.append(dict(path=str(Path(folder)/entry.name),size=info.st_size,modified_ns=info.st_mtime_ns))
    return rows


def preparation_root():
    return Path(os.getenv('RADAR_PD_BATTERY_ROOT') or Path(tempfile.gettempdir())/'radar-battery-models').resolve()


def verify_prepared(job):
    """Reopen only our local complete bundles, preserving their recorded scope."""
    job=Path(job).resolve()
    receipt=job/'model/preparation.json'
    if not (job/'inputs').resolve().is_relative_to(job) or not receipt.resolve().is_relative_to(job) or receipt.is_symlink() or not receipt.is_file() or receipt.stat().st_size>2*1024*1024:
        raise ValueError('Prepared model receipt is unavailable or too large')
    result=json.loads(receipt.read_text())
    if result.get('schema')!='radar-battery-preparation-result/v1' or result.get('status')!='prepared_needs_review':
        raise ValueError('This bundle has no complete prepared model')
    validate_preparation(result['specification'],job/'inputs')
    if 'catalog_context' in result['specification']:
        from gpx_recipe.battery.catalog_evidence import read_catalog, catalog_summary
        spec=result['specification'];reference=spec['catalog_context']
        summary=catalog_summary(read_catalog(reference,job/'inputs',spec['acquisition']),reference)
        if result.get('catalog_context')!=summary:
            raise ValueError('Prepared catalog summary differs from its captured evidence; reopen an unchanged bundle')
    elif result.get('catalog_context') is not None:
        raise ValueError('Prepared catalog summary has no captured evidence')
    project=job/'model/prepared.gpx'
    if project.is_symlink() or not project.resolve().is_relative_to(job) or file_hash(project)!=result['project_sha256']:
        raise ValueError('Prepared model changed; prepare a verified revision again')
    return result


def check_prepared_profiles(job, result):
    """Preserve old bundles for inspection, but do not reuse lossy conversions."""
    from gpx_recipe.battery.profiles import legacy_profile_metadata
    root = Path(job) / 'inputs'
    for bank in result['specification']['banks']:
        path = root / bank['profile']['path']
        if path.suffix.lower() not in {'.prm', '.inst', '.ins'}:
            continue
        try:
            expected = legacy_profile_metadata(path, bank['profile_bank'])['coefficient_mapping']
            recorded = next(row for row in result['banks'] if row['physical_bank'] == bank['physical_bank'])
            actual = (recorded.get('profile_conversion') or {}).get('converted_parameters', {})
            missing = [key for key, value in expected.items() if actual.get(key) != value]
            if missing:
                labels = {'X': 'Lorentzian d width (X)', 'Y': 'Lorentzian d² width (Y)',
                          'Z': 'constant Lorentzian width (Z)', 'sig-0': 'constant Gaussian variance (sig-0)'}
                raise ValueError('converted coefficients differ from the captured source: ' +
                                 ', '.join(labels.get(key, key) for key in missing))
        except (ValueError, StopIteration) as exc:
            raise ValueError(f"Physical bank {bank['physical_bank']} needs a new preparation: {exc}. "
                'Prepare a new revision from the retained inputs with the checked legacy converter. '
                'The previous GPX and fit results remain historical evidence.') from exc


def prepared_source(workspace, job, result):
    """Register one verified preparation without changing the visible selection."""
    from .refinement_sources import digest
    job=Path(job)
    if verify_prepared(job)!=result:
        raise ValueError('Prepared receipt changed; reopen the saved revision')
    check_prepared_profiles(job, result)
    spec=result['specification'];a=spec['acquisition']
    key=digest([result['project_sha256'],spec])[:24]
    source=dict(id=key,experiment=battery_experiment_id(workspace,a,spec['reduction_family']),
        experiment_label=f"Battery · {a['ipts']} · {a['cell_id']}",
        scan_key=key,record_uid='battery-'+key,run_id=f"{a['instrument']}_{a['run']}",run_number=a['run'],
        bank='+'.join(str(b['physical_bank']) for b in spec['banks']),sample_key='id:'+a['sample_id'],
        sample_label=a['sample_id']+' · '+a['cell_id'],scan_label=f"Run {a['run']} · prepared model {key[:8]}",
        metadata={**a,'start_time':a['start_utc'],'preparation':str(job/'model/preparation.json'),
            'electrochemistry':result.get('electrochemistry'),
            'catalog_context':result.get('catalog_context'),
            'physical_banks':{b['histogram']:b['physical_bank'] for b in result['banks']}},axis_overrides={},
        source_path=str(job/'model/prepared.gpx'),source_sha256=result['project_sha256'],
        collection_name='prepared.gpx',base_rwp=None,project=dict(stage='prepared_unqualified'))
    if workspace.runtime.kind=='galaxy':
        publication=job/'preparation-publication.json'
        if not publication.is_file():raise ValueError('Publish this prepared model through the native Galaxy worker first')
        record=json.loads(publication.read_text())
        if record.get('source_sha256')!=result['project_sha256']:raise ValueError('Published preparation changed')
        source['dataset_id']=record['dataset_id']
    workspace.observe([workspace.runtime.resolve(source)])
    return source


class BatteryPreparationMixin:
    def _init_battery_preparation(self):
        s=self.server.state
        self._battery_model_revision=0; self._battery_model_sources={}; self._battery_bank_setup=None
        self._battery_model_phases=[]; self._battery_prepared=None; self._battery_profile_uploads={}; self._battery_model_sample=None
        self._battery_saved_models={}
        self._battery_auto_profiles={}
        s.battery_saved_models=[];s.battery_saved_model='';s.battery_saved_notice=''
        s.battery_model_run='';s.battery_model_runs=[];s.battery_model_data='';s.battery_model_sources=[]
        s.battery_model_banks=[];s.battery_model_bank_options=[];s.battery_model_bank_rows=[]
        s.battery_allow_invalid_edges=False
        s.battery_model_include_echem=True
        s.battery_model_cell='';s.battery_model_review='';s.battery_model_ready=False
        s.battery_model_profile_notes=[]
        s.battery_model_mapping_notes=[]
        s.battery_model_message='Inspect a run, pair its banks with reviewed profiles, then add the phase hypotheses.'
        s.battery_phase_path='';s.battery_phase_name='';s.battery_phase_role='';s.battery_phase_scale='0.1'
        s.battery_phase_size_um='';s.battery_phase_microstrain_ustrain=''
        s.battery_model_phases=[]
        s.battery_native_available=bool(os.getenv('RADAR_PD_REFINEMENT_PYTHON','').strip() or (getattr(getattr(self,'service',None),'galaxy_url',None) and getattr(getattr(self,'service',None),'history_id',None)))
        s.battery_profile_options=[dict(title='Upload a peak profile',value='upload')]
        fields=['battery_model_banks','battery_model_cell','battery_model_review','battery_allow_invalid_edges','battery_model_include_echem']
        for bank in range(1,7):
            for field,value in [('profile','upload'),('path',''),('profile_bank',''),('d_min',''),('d_max','')]:
                name=f'battery_b{bank}_{field}';setattr(s,name,value);fields.append(name)
        s.change(*fields)(self._battery_preparation_edited)
        s.change('battery_model_run')(self._battery_model_run_changed)
        s.change('battery_model_data')(self._battery_model_data_changed)

    def _battery_preparation_edited(self,**_):
        if self._battery_prepared:
            self.server.state.battery_model_message='Inputs changed. Prepare and verify the revised model before continuing.'
        self._battery_model_revision+=1;self._battery_prepared=None
        self.server.state.battery_model_ready=False
        self.server.state.battery_model_profile_notes=[]

    def _battery_reset_preparation(self):
        if not hasattr(self,'_battery_model_revision'):return
        s=self.server.state;self._battery_preparation_edited()
        self._battery_bank_setup=None;self._battery_model_sources={}
        self._battery_auto_profiles={};s.battery_model_mapping_notes=[]
        self._battery_model_phases=[];s.battery_model_phases=[];self._battery_model_sample=None
        s.battery_model_runs=[];s.battery_model_run='';s.battery_model_sources=[];s.battery_model_data=''
        s.battery_model_banks=[];s.battery_model_bank_options=[];s.battery_model_bank_rows=[]
        s.battery_allow_invalid_edges=False
        s.battery_model_cell='';s.battery_model_review=''
        s.battery_phase_size_um='';s.battery_phase_microstrain_ustrain=''
        for bank in range(1,7):
            for field,value in [('profile','upload'),('path',''),('profile_bank',''),('d_min',''),('d_max','')]:
                setattr(s,f'battery_b{bank}_{field}',value)
        s.battery_model_message='Inspect acquisition metadata before preparing a model.'

    def _battery_model_acquisitions(self):
        s=self.server.state
        s.battery_model_runs=[dict(title=f"{r['run']} · {r['identity']['sample_id']} · {r.get('fields',{}).get('run_title',{}).get('value','')}",value=str(r['run']))
            for r in self._battery_acquisitions if r.get('identity',{}).get('identity_status')=='resolved' and r.get('exposure') and r.get('status')!='conflict']
        inventory=self._battery_inventory or {}
        s.battery_profile_options=([dict(title='Match POWGEN reduction metadata', value='powgen_catalog')]
            if inventory.get('instrument')=='PG3' else [])+[dict(title='Upload a peak profile',value='upload')]+[
            dict(title=r['path'],value=str(i)) for i,r in enumerate(inventory.get('profile_candidates',[]))]

    def _battery_model_run_changed(self,**_):
        self._battery_model_data_changed();s=self.server.state
        self._battery_model_sources={};s.battery_model_sources=[];s.battery_model_data=''
        inventory=self._battery_inventory
        if not inventory or not str(s.battery_model_run).isdigit() or inventory['instrument'] not in {'NOM','PG3'}:return
        run=int(s.battery_model_run)
        if run not in {r['run'] for r in self._battery_acquisitions}:return
        acquisition=next(r for r in self._battery_acquisitions if r['run']==run)
        sample=acquisition.get('identity',{}).get('sample_id')
        if self._battery_model_sample!=sample:
            self._battery_model_phases=[];s.battery_model_phases=[]
            s.battery_model_cell='';s.battery_model_review=''
            s.battery_phase_name='';s.battery_phase_path='';s.battery_phase_role=''
            s.battery_phase_size_um='';s.battery_phase_microstrain_ustrain=''
            s.battery_model_message='Review the selected sample and add its phase hypotheses; different sample IDs do not share a cell or model automatically.'
        self._battery_model_sample=sample
        series=self._battery_series
        if series and series.binding.sample_id==sample and not s.battery_model_cell:
            s.battery_model_cell=series.binding.cell_id
        folder=(Path(inventory['root'])/s.battery_family).resolve()
        if not folder.is_relative_to(Path(inventory['root']).resolve()):return
        if s.battery_family not in {f['folder'] for f in inventory['product_families']}:return
        try:
            choices=run_sources(folder,run,instrument=inventory['instrument'])
            self._battery_model_sources={str(i):r for i,r in enumerate(choices)}
            s.battery_model_sources=[dict(title=Path(r['path']).name,value=key) for key,r in self._battery_model_sources.items()]
            if len(choices)==1:s.battery_model_data='0'
        except (OSError,ValueError) as exc:s.battery_error=str(exc)

    def _battery_model_data_changed(self,**_):
        self._battery_preparation_edited();self._battery_bank_setup=None
        self._battery_auto_profiles={}
        s=self.server.state;s.battery_model_banks=[];s.battery_model_bank_options=[];s.battery_model_bank_rows=[]
        s.battery_model_mapping_notes=[]

    def inspect_battery_banks(self,**_):
        if self._battery_running():return
        source=self._battery_model_sources.get(self.server.state.battery_model_data)
        if source is None:self.server.state.battery_error='Choose a reduced run first';return
        self._battery_start(self._inspect_battery_banks_async(deepcopy(source),self._battery_generation,self._battery_model_revision))

    async def _inspect_battery_banks_async(self,source,generation,revision):
        s=self.server.state;s.battery_busy=True;s.battery_error='';s.flush()
        try:
            instrument=self._battery_inventory['instrument']
            setup=await asyncio.to_thread(product_adapter(instrument).inspect_reduced_setup,source['path'],inventory_only=True)
            if generation!=self._battery_generation or revision!=self._battery_model_revision:return
            self._battery_bank_setup=(source,setup)
            s.battery_model_bank_rows=[dict(bank=b.bank,angle=b.angle,points=b.records,
                nominal_d=f'{b.minimum_tof/b.difc:.3g}–{b.maximum_tof/b.difc:.3g}',
                quality='; '.join(b.quality_errors) or 'Numerical records valid') for b in setup.banks]
            s.battery_model_bank_options=[dict(title=f'Bank {b.bank}'+(' · data issues' if b.quality_errors else ''),value=b.bank,
                props=dict(disabled=any(error!=invalid_count_error(instrument) for error in b.quality_errors))) for b in setup.banks]
            self._battery_auto_profiles={};s.battery_model_mapping_notes=[]
            if instrument=='PG3':
                from gpx_recipe.battery.profiles import legacy_profile_metadata
                notes=list(setup.warnings)
                for bank in setup.banks:
                    try:
                        profile=powgen.catalog_profile(setup,bank.bank)
                        if profile.path.suffix.lower()!='.instprm':
                            legacy_profile_metadata(profile.path,profile.profile_bank)
                        self._battery_auto_profiles[bank.bank]=profile
                        setattr(s,f'battery_b{bank.bank}_profile','powgen_catalog')
                        setattr(s,f'battery_b{bank.bank}_profile_bank',str(profile.profile_bank))
                        notes.append(f'Available metadata match for physical bank {bank.bank}: {profile.profile_filename}, profile Bank {profile.profile_bank}; scientific qualification pending.')
                    except ValueError as exc:
                        setattr(s,f'battery_b{bank.bank}_profile','upload')
                        setattr(s,f'battery_b{bank.bank}_profile_bank','')
                        notes.append(f'Physical bank {bank.bank}: {exc}')
                s.battery_model_mapping_notes=notes
            s.battery_model_message='Select complementary valid banks. The displayed d range uses nominal export geometry; the worker checks fit limits against each supplied profile.'
        except Exception as exc:
            if generation==self._battery_generation:s.battery_error=str(exc)
        finally:s.battery_busy=False;s.flush()

    def add_battery_phase(self,**_):
        if self._battery_running():return
        s=self.server.state
        try:
            path=checked_upload(self._battery_phase_upload,s.battery_phase_path)
            name=str(s.battery_phase_name).strip();role=s.battery_phase_role
            scale=float(s.battery_phase_scale)
            import math
            if not name or len(name)>128 or role not in ROLES or not math.isfinite(scale) or scale<=0:
                raise ValueError('Provide a phase name, physical role and positive starting HAP scale')
            widths={}
            for key in ('size_um','microstrain_ustrain'):
                value=getattr(s,'battery_phase_'+key)
                if type(value) not in {str,int,float}:
                    raise ValueError('Enter sample-width starts as numbers, or leave them blank')
                if str(value).strip():widths[key]=float(value)
            widths=validate_sample_widths(widths)
            if len(self._battery_model_phases)>=24 or any(p['name']==name for p in self._battery_model_phases):
                raise ValueError('Use unique phase names and at most 24 hypotheses in one model')
            with path.open('rb') as stream:raw=stream.read(1024*1024+1)
            if len(raw)>1024*1024:raise ValueError('Phase CIF exceeds 1 MiB')
            self._battery_model_phases.append(dict(key='phase_'+uuid.uuid4().hex[:10],name=name,role=role,
                filename=path.name,raw=raw,hap_scale=scale,sample_widths=widths))
            s.battery_model_phases=[{k:v for k,v in p.items() if k!='raw'} for p in self._battery_model_phases]
            self._battery_preparation_edited();s.battery_error=''
            s.battery_phase_name='';s.battery_phase_path='';s.battery_phase_role=''
            s.battery_phase_size_um='';s.battery_phase_microstrain_ustrain=''
        except (OSError,ValueError) as exc:s.battery_error=str(exc)
        s.flush()

    def remove_battery_phase(self,key,**_):
        if self._battery_running():return
        self._battery_model_phases=[p for p in self._battery_model_phases if p['key']!=key]
        self.server.state.battery_model_phases=[{k:v for k,v in p.items() if k!='raw'} for p in self._battery_model_phases]
        self._battery_preparation_edited();self.server.state.flush()

    def _battery_profile_path(self,bank):
        s=self.server.state;choice=getattr(s,f'battery_b{bank}_profile')
        if choice=='upload':
            return checked_upload(self._battery_profile_uploads.get(bank),getattr(s,f'battery_b{bank}_path'))
        if choice=='powgen_catalog':
            profile=self._battery_auto_profiles.get(bank)
            if profile is None:raise ValueError(f'No supported metadata-matched profile for physical bank {bank}; supply and review one explicitly')
            if str(getattr(s,f'battery_b{bank}_profile_bank'))!=str(profile.profile_bank):
                raise ValueError('The Bank label must agree with the metadata-matched POWGEN profile')
            from .powgen_setup import text_hash
            if text_hash(profile.path.read_bytes())!=profile.sha256:raise ValueError('Metadata-matched profile changed; inspect again')
            return profile.path
        if not str(choice).isdigit():raise ValueError(f'Choose a profile for bank {bank}')
        record=self._battery_inventory['profile_candidates'][int(choice)]
        root=Path(self._battery_inventory['root']).resolve();path=(root/record['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():raise ValueError('Profile is no longer within this proposal')
        if nomad.signature(path)!=(record['size_bytes'],record['mtime_ns']):raise ValueError('Profile changed; discover the proposal again')
        return path

    def prepare_battery_model(self,**_):
        if self._battery_running():return
        s=self.server.state
        try:
            python=os.getenv('RADAR_PD_REFINEMENT_PYTHON','').strip()
            if not python and self._refinement_store().runtime.kind!='galaxy':raise ValueError('A native local or Galaxy worker is required for model preparation')
            if not self._battery_bank_setup:raise ValueError('Inspect the selected reduced file first')
            source,setup=self._battery_bank_setup
            instrument=self._battery_inventory['instrument'] if self._battery_inventory else 'NOM'
            selected=s.battery_model_banks
            allow_invalid = s.battery_allow_invalid_edges
            if type(allow_invalid) is not bool:
                raise ValueError('The outside-fit invalid-bin option must be explicitly enabled or disabled')
            valid={b.bank for b in setup.banks if not b.quality_errors or (allow_invalid and
                all(error==invalid_count_error(instrument) for error in b.quality_errors))}
            if not isinstance(selected,list) or not selected or any(type(b) is not int or b not in valid for b in selected) or len(set(selected))!=len(selected):
                raise ValueError('Select distinct, numerically valid physical banks from this run, or explicitly allow invalid count bins outside the selected fit range')
            if nomad.signature(source['path'])!=setup.source_signature:raise ValueError('Reduced data changed; inspect it again')
            record=next(r for r in self._battery_acquisitions if str(r['run'])==s.battery_model_run)
            if record.get('status')=='conflict' or record['identity']['identity_status']!='resolved':raise ValueError('Resolve the sample identity first')
            if type(s.battery_model_include_echem) is not bool:
                raise ValueError('Choose whether to include the reviewed electrochemical record')
            series=self._battery_series if s.battery_model_include_echem else None
            if series and (series.binding.sample_id!=record['identity']['sample_id'] or series.binding.cell_id!=str(s.battery_model_cell).strip()):
                raise ValueError('The reviewed cycler belongs to a different sample or cell. Correct the binding or disable its inclusion for this model.')
            root=preparation_root()
            job=root/uuid.uuid4().hex;inputs=job/'inputs';inputs.mkdir(parents=True)
            data=Path(source['path'])
            data_ref=capture_file(data,inputs/data.name,maximum=64*1024*1024,
                expected_signature=setup.source_signature,expected_sha256=setup.source_sha256)
            evidence=[]
            if instrument=='NOM':
                sidecar=Path(setup.sidecar_path)
                if (nomad.signature(sidecar) if sidecar.is_file() else None)!=setup.sidecar_signature:
                    raise ValueError('Reduction evidence changed; inspect the observation again')
                if setup.sidecar_signature:
                    evidence.append(capture_file(sidecar,(inputs/data.name).with_suffix('.json'),maximum=2*1024*1024,
                        expected_signature=setup.sidecar_signature,expected_sha256=setup.sidecar_sha256))
            else:
                powgen.verify_evidence(data,setup)
                for sidecar in setup.evidence:
                    evidence.append(capture_file(Path(sidecar['path']),inputs/Path(sidecar['path']).name,maximum=1024*1024,
                        expected_signature=sidecar['signature'],expected_sha256=sidecar['sha256']))
                powgen.verify_evidence(data,setup)
            def ref(path):return dict(path=path.name,sha256=sha256(path.read_bytes()).hexdigest())
            spec=dict(schema=SCHEMA,acquisition=dict(instrument=instrument,ipts=self._battery_inventory['ipts'],run=record['run'],
                sample_id=record['identity']['sample_id'],cell_id=str(s.battery_model_cell).strip(),
                start_utc=record['exposure']['start_utc'],end_utc=record['exposure']['end_utc'],metadata_sha256=record['metadata_sha256']),
                reduction_family=s.battery_family,review_note=s.battery_model_review,banks=[],phases=[],reduction_evidence=evidence)
            from gpx_recipe.battery.catalog_evidence import capture_catalog
            catalog=capture_catalog(self._battery_catalog_proposal,record,spec['acquisition'],inputs)
            if catalog:spec['catalog_context']=catalog
            if series:
                from gpx_recipe.battery.electrochemical_evidence import capture_electrochemistry
                spec['electrochemistry']=capture_electrochemistry(series,self._battery_cycler_raw,inputs)
            for bank in s.battery_model_banks:
                profile=self._battery_profile_path(bank)
                if profile.suffix.lower() not in {'.instprm','.prm','.inst','.ins'}:raise ValueError('Choose a GSAS-II or legacy GSAS peak profile')
                copied=inputs/f'bank{bank}{profile.suffix.lower()}'
                choice=getattr(s,f'battery_b{bank}_profile')
                candidate=self._battery_inventory['profile_candidates'][int(choice)] if str(choice).isdigit() else None
                profile_signature=(candidate['size_bytes'],candidate['mtime_ns']) if candidate else nomad.signature(profile)
                profile_ref=capture_file(profile,copied,maximum=1024*1024,expected_signature=profile_signature)
                spec['banks'].append(dict(key=f'b{bank}',physical_bank=bank,data=data_ref,profile=profile_ref,
                    profile_bank=int(getattr(s,f'battery_b{bank}_profile_bank')),
                    profile_origin='powgen_catalog' if choice=='powgen_catalog' else 'explicit',
                    invalid_bin_policy='outside_fit_only' if allow_invalid else 'reject',
                    fit_d=[float(getattr(s,f'battery_b{bank}_d_min')),float(getattr(s,f'battery_b{bank}_d_max'))],profile_review=s.battery_model_review))
            for phase in self._battery_model_phases:
                cif=inputs/(phase['key']+'.cif');cif.write_bytes(phase['raw'])
                spec['phases'].append(dict(key=phase['key'],name=phase['name'],role=phase['role'],cif=ref(cif),
                    links=[dict(bank=b['key'],hap_scale=phase['hap_scale'],sample_widths=deepcopy(phase.get('sample_widths',{}))) for b in spec['banks']],
                    hypothesis_note='User-selected starting hypothesis; native import is checked, scientific acceptance is pending.'))
            validate_preparation(spec,inputs)
            (inputs/'preparation.json').write_text(json.dumps(spec,indent=2))
            (inputs/'acquisition-record.json').write_text(json.dumps(record,indent=2))
            self._battery_preparation_edited()
            self._battery_start(self._prepare_battery_model_async(job,python,self._battery_generation,self._battery_model_revision))
        except (OSError,ValueError,StopIteration,KeyError,IndexError) as exc:
            s.battery_error=str(exc) or 'Select a run with reviewed acquisition metadata';s.flush()

    async def _prepare_battery_model_async(self,job,python,generation,revision):
        s=self.server.state;s.battery_busy=True;s.battery_error='';s.battery_model_message='Preparing and reopening the joint-bank model; no refinement is run.';s.flush()
        def work():
            if not python:
                from .battery_transport import prepare_on_galaxy
                return prepare_on_galaxy(self._refinement_store().runtime, job)
            command=[python,'-m','gpx_recipe.battery.preparation',str(job/'inputs/preparation.json'),str(job/'model')]
            with (job/'native.log').open('w') as log:
                process=run_scientific(command,cwd=job,env=scientific_environment(os.getenv('GSASII_ROOT')),stdout=log,stderr=subprocess.STDOUT,timeout=600)
            receipt=job/'model/preparation.json'
            if not receipt.exists():raise ValueError('Native preparation stopped before producing a receipt; its diagnostic log was retained')
            result=json.loads(receipt.read_text())
            if process.returncode or result.get('status')!='prepared_needs_review':raise ValueError(result.get('error') or 'Native model preparation failed')
            return result
        try:
            result=await asyncio.to_thread(work)
            if generation!=self._battery_generation or revision!=self._battery_model_revision:return
            self._battery_prepared=(job,result);s.battery_model_ready=True
            s.battery_model_profile_notes=[f"Physical bank {b['physical_bank']}: legacy profile title — {b['profile_conversion'].get('legacy_title') or 'not recorded'}. All supported isotropic coefficients were mapped explicitly and checked after native reload. Calibration applicability still requires review."
                for b in result['banks'] if b.get('profile_conversion')]
            ignored=sum(len(b.get('provenance',{}).get('invalid_count_bins',[])) for b in result['banks'])
            banks=len(result['banks']);phases=len(result['phases'])
            s.battery_model_message=f"Prepared {banks} bank{'s' if banks!=1 else ''} and {phases} linked phase{'s' if phases!=1 else ''}. Valid observations, weights and masks survived native save/reopen. Calibration and chemistry remain unqualified."+(f' {ignored} invalid count records were excluded outside the fitted domain or at the native trailing edge; original records remain in the bundle.' if ignored else '')
            if result.get('catalog_context'):
                s.battery_model_message+=' Catalog context and its comparison were captured. '+result['catalog_context']['message']
        except Exception as exc:
            if generation==self._battery_generation and revision==self._battery_model_revision:s.battery_error=str(exc)
        finally:s.battery_busy=False;s.flush()

    def download_battery_model(self,**_):
        if not self._battery_prepared:return
        job,result=self._battery_prepared
        try:
            if verify_prepared(job)!=result:raise ValueError('Prepared receipt changed; reopen the saved revision')
            payload=io.BytesIO()
            with zipfile.ZipFile(payload,'w',zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(job.rglob('*')):
                    if path.is_file() and not path.is_symlink():archive.write(path,path.relative_to(job))
            self.download_file('battery-model.zip','application/zip',payload.getvalue())
        except (OSError,ValueError,KeyError,TypeError) as exc:
            self.server.state.battery_error=str(exc);self.server.state.flush()

    def refresh_saved_battery_models(self,**_):
        if self._battery_running():return
        s=self.server.state;root=preparation_root();self._battery_saved_models={};rows=[];skipped=0
        try:
            if root.exists():
                with os.scandir(root) as entries:
                    for index,entry in enumerate(entries):
                        if index>=500:
                            raise ValueError('Saved-model lookup reached 500 folders; use the retained model bundles to narrow the collection')
                        if not entry.is_dir(follow_symlinks=False):continue
                        receipt=Path(entry.path)/'model/preparation.json'
                        if not receipt.resolve().is_relative_to(root) or receipt.is_symlink() or not receipt.is_file() or receipt.stat().st_size>2*1024*1024:continue
                        try:
                            result=json.loads(receipt.read_text())
                            if result.get('status')!='prepared_needs_review':skipped+=1;continue
                            a=result['specification']['acquisition']
                            key=entry.name;self._battery_saved_models[key]=Path(entry.path)
                            phases=', '.join(p['name'] for p in result['specification']['phases'])
                            rows.append(dict(value=key,title=f"{a['ipts']} · run {a['run']} · {a['sample_id']} · {a['cell_id']} · {phases} · {key}"))
                        except (ValueError,KeyError,TypeError):skipped+=1
            s.battery_saved_models=rows
            s.battery_saved_notice=f'{len(rows)} prepared bundles. Sources and GPX hashes are checked when reopened.'+(f' {skipped} incomplete or unreadable bundles were excluded.' if skipped else '')
        except (OSError,ValueError) as exc:s.battery_error=str(exc)
        s.flush()

    def open_saved_battery_model(self,**_):
        if self._battery_running():return
        s=self.server.state
        try:
            job=self._battery_saved_models.get(s.battery_saved_model)
            if not job or not job.resolve().is_relative_to(preparation_root()):
                raise ValueError('Choose a saved local model from the refreshed list')
            result=verify_prepared(job)
            check_prepared_profiles(job, result)
            self._battery_prepared=(job,result)
            s.battery_error=''
            self.open_battery_refinement()
        except (OSError,ValueError,KeyError) as exc:s.battery_error=str(exc);s.flush()

    def open_battery_refinement(self,**_):
        if not self._battery_prepared:return
        from .refinement_controller import refinement_experiment_options, refinement_transport_label
        job,result=self._battery_prepared
        try:
            workspace=self._refinement_store()
            source=prepared_source(workspace,job,result)
            key,experiment=source['id'],source['experiment']
            s=self.server.state;s.battery_error='';s.rf_experiments=[dict(title=label,value=k) for k,label in refinement_experiment_options(workspace).items()]
            s.rf_transport=refinement_transport_label(workspace.runtime.kind)
            s.rf_experiment=experiment;s.workspace_view='refinement';s.setup_collapsed=True
            # Restoring an experiment may restore its older recipe and result
            # tab. Opening a particular preparation must finish on that model,
            # with only that immutable revision selected for the next recipe.
            s.flush()
            s.rf_scope='scans';s.rf_scans=[key];s.rf_samples=[];s.rf_banks=[]
            s.rf_existing=True;s.rf_future=False;s.rf_mode='recipe';s.rf_source=key
            s.flush();self._refinement_sync();s.flush()
        except Exception as exc:self.server.state.battery_error=str(exc);self.server.state.flush()
