"""HB-2A adapter sharing Galaxy retries/checkpoints without TOF assumptions."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path

from . import hb2a
from .facility_metadata import title_role
from .models import InputSelection, InputSource
from .powgen_controller import PowgenExperimentSettings, PowgenWatchController
from .powgen_watch import WatchDefinition, build_submission_plan, discover_from_listing


class Hb2aSampleIndex:
    def __init__(self, ipts):
        self.ipts = ipts
        self._cache = {}

    def identity(self, run):
        try:
            path = hb2a.raw_path(self.ipts, run.experiment_number, run.run_number)
            sig = hb2a.signature(path)
            cached = self._cache.get(run.run_id)
            if cached and cached[0] == sig:
                return dict(cached[1])
            metadata = hb2a.read_metadata(self.ipts, run.experiment_number, run.run_number, identity_only=True)
            self._cache[run.run_id] = (sig, dict(metadata))
            return metadata
        except (OSError, ValueError, UnicodeError) as exc:
            return {'sample_id':'', 'sample_name':'', 'identity_status':'unidentified',
                'identity_warning':str(exc), 'metadata_notices':[str(exc)]}

    def inventory(self, runs, **_):
        runs = list(runs)
        with ThreadPoolExecutor(max_workers=2) as pool:
            identities = list(pool.map(self.identity, runs))
        groups, options, channels = {}, [], set()
        for run, metadata in zip(runs, identities):
            key = metadata.get('sample_id')
            channels.update(metadata.get('available_channels', []))
            if key:
                group = groups.setdefault(key, {'name':metadata['sample_name'], 'experiment':run.experiment_number, 'count':0})
                group['count'] += 1
        for key, group in sorted(groups.items()):
            unit='scan' if group['count']==1 else 'scans'
            options.append({'value':key, 'title':f"Experiment {group['experiment']} — {group['name']} ({group['count']} {unit})", 'scan_count':group['count']})
        return {'options':options, 'scan_count':len(runs),
            'unidentified_count':sum(not m.get('sample_id') for m in identities),
            'run_options':[{'value':r.run_id, 'sample_id':m.get('sample_id',''),
                'title':f"{r.run_id} — {m.get('run_title') or m.get('identity_warning','Title unavailable')}",
                'run_title':m.get('run_title',''), 'run_role':title_role(m.get('run_title','')),
                'identity_status':m.get('identity_status','unidentified')} for r,m in zip(runs,identities)],
            'metadata_notices':['HB-2A sample labels are scoped to SPICE experiment; repeated labels do not establish identical specimens.',
                'Raw SPICE measured channels have no universal units or selected-sensor marker; review a channel/unit binding before using a temperature axis.'],
            'metadata_resolution':[{'run_number':r.run_number,'run_id':r.run_id,
                'status':m.get('identity_status','unidentified'), 'sample_id':m.get('sample_id',''),
                'sample_display':m.get('sample_name',''),
                'sources':m.get('identity_sources',[]), 'actions':([{'message':m['identity_warning']}] if m.get('identity_warning') else [])}
                for r,m in zip(runs,identities)],
            'condition_channels':sorted(channels)}


@dataclass(frozen=True)
class Hb2aExperimentSettings(PowgenExperimentSettings):
    wavelength_angstrom: str = ''
    instrument: str = 'HB2A'
    normalization: str = 'time'
    profile_acknowledged: bool = False
    condition_bindings_json: str = '{}'
    excluded_acquisitions: tuple[str, ...] = ()
    collimator_override: str = ''
    allow_stopped: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.instrument != 'HB2A' or self.wavelength_angstrom:
            raise ValueError('HB-2A uses the calibrated CW wavelength in its verified profile')
        if self.selected_banks not in (None, (1,)):
            raise ValueError('HB-2A reduced XYE is one combined detector-bank pattern, not 44 independent banks')
        if self.normalization not in hb2a.NORMALIZATIONS:
            raise ValueError('Choose one HB-2A normalization family')
        if self.collimator_override not in {'','IN','OUT'}:
            raise ValueError('Choose measured automatic or explicitly reviewed IN/OUT collimation')
        if len(self.condition_bindings_json)>16384:
            raise ValueError('Condition bindings exceed the supported metadata bound')
        bindings=json.loads(self.condition_bindings_json)
        if not isinstance(bindings, dict) or len(bindings)>4:
            raise ValueError('Condition bindings must be an explicit quantity/channel mapping')
        for quantity,binding in bindings.items():
            if quantity not in {'temperature','magnetic_field','pressure','voltage'} or not isinstance(binding,dict):
                raise ValueError('Choose supported conditions and explicit channel/unit/review mappings')
            if set(binding)-{'channel','unit','reviewed'} or not isinstance(binding.get('channel'),str) or not isinstance(binding.get('unit'),str) or not isinstance(binding.get('reviewed'),bool):
                raise ValueError('Each condition mapping requires a channel, units and boolean review state')
        if any(not __import__('re').fullmatch(r'HB2A_EXP\d+_SCAN\d+', v) for v in self.excluded_acquisitions):
            raise ValueError('Exclude HB-2A acquisitions by full experiment/scan identity')

    def definition(self, instrument_profile_ref=None):
        return WatchDefinition(self.ipts, self.subfolder, self.history_id,
            configuration_ref=self.configuration_dataset_id,
            instrument_profile_ref=instrument_profile_ref or self.instrument_profile_dataset_id or 'hb2a-profile-required',
            instrument='HB2A', facility='HFIR', main_cif_ref=self.main_cif_dataset_id,
            config_refs={'candidate_library':self.database_dataset_id or 'builtin', 'hb2a_normalization':self.normalization,
                'hb2a_conditions':self.condition_bindings_json,
                'hb2a_acquisition_policy':json.dumps({'collimator':self.collimator_override,
                    'allow_stopped':self.allow_stopped},sort_keys=True)})


def _definition(ipts, normalization):
    return Hb2aExperimentSettings(ipts,'inspection','inspection',normalization=normalization).definition()


def discover_hb2a_samples(ipts, sample_index, subfolder=hb2a.FOLDER, *, normalization='time'):
    definition = _definition(ipts, normalization)
    if subfolder != hb2a.FOLDER:
        raise ValueError('Choose the individual shared/autoreduce folder')
    root = hb2a.proposal_root(ipts) / subfolder
    runs = hb2a.discover_listing(definition, hb2a.directory_listing(root, normalization=normalization))
    result = sample_index.inventory(runs)
    result['source_options'] = [{'title':'Individual HB-2A Bragg reductions', 'value':hb2a.FOLDER}]
    result['normalization'] = normalization
    return result


def _selected(runs, settings, index):
    result = []
    for run in runs:
        if run.run_id in settings.excluded_acquisitions:
            continue
        metadata = index.identity(run)
        if settings.selected_sample_ids is not None and metadata.get('sample_id') not in settings.selected_sample_ids:
            continue
        role = title_role(metadata.get('run_title',''))
        if (role == 'background' and not settings.include_background_runs) or (role == 'ramp' and not settings.include_ramp_runs):
            continue
        result.append(run)
    return result


def preflight_hb2a_experiment(ipts, *, selected_sample_ids=None, selected_banks=None,
        sample_index=None, subfolder=hb2a.FOLDER, initial_backfill_limit=None,
        profile_path=None, profile_acknowledged=False, normalization='time',
        condition_bindings=None, excluded_acquisitions=(), excluded_run_numbers=(),
        collimator_override='', allow_stopped=False,
        include_background_runs=False, include_ramp_runs=False):
    if excluded_run_numbers:
        raise ValueError('HB-2A exclusion requires full experiment/scan identifiers, not scan numbers alone')
    settings = Hb2aExperimentSettings(ipts,'inspection','inspection',selected_sample_ids=selected_sample_ids,
        selected_banks=selected_banks, subfolder=subfolder, initial_backfill_limit=initial_backfill_limit,
        normalization=normalization, excluded_acquisitions=tuple(excluded_acquisitions),
        collimator_override=collimator_override, allow_stopped=allow_stopped,
        include_background_runs=include_background_runs,include_ramp_runs=include_ramp_runs)
    index = sample_index or Hb2aSampleIndex(ipts)
    root = hb2a.proposal_root(ipts) / subfolder
    all_runs = hb2a.discover_listing(settings.definition(), hb2a.directory_listing(root, normalization=normalization))
    runs = _selected(all_runs,settings,index)
    if initial_backfill_limit is not None:
        runs = runs[-max(1,initial_backfill_limit):]
    errors, setups, profiles = [], [], []
    for run in runs:
        try:
            setup=hb2a.inspect_reduced_setup(run.source_path,ipts=ipts,condition_bindings=condition_bindings,
                collimator_override=collimator_override,allow_stopped=allow_stopped)
            setups.append(setup)
            if not profile_path:
                raise ValueError('Supply one cycle/Ge/collimator-matched GSAS-II .instprm; an efficiency vcorr file is not a profile')
            profiles.append(hb2a.validate_profile(setup, profile_path, acknowledged=profile_acknowledged))
        except (OSError, ValueError, UnicodeError) as exc:
            errors.append(f'{run.run_id}: {exc}')
    # A manually supplied profile cannot migrate between distinct experiments
    # (which can span years) or instrumental configurations on an IPTS.
    contracts = {(s.experiment,s.metadata['ge_reflection'],s.metadata['collimator'],s.metadata['channel_schema_sha256']) for s in setups}
    if len(contracts)>1:
        errors.append('The selection spans SPICE experiments or instrumental settings; select an experiment-scoped sample and verify its profile separately')
    if not runs:
        errors.append(f'No individual {normalization}-normalization reduced patterns match this sample selection')
    latest = setups[-1].metadata if setups else {}
    return {'ready':not errors,'source_directory':str(root),'scan_count':len(runs),'total_scan_count':len(all_runs),
        'analysis_count':len(profiles),'excluded_count':len(all_runs)-len(runs),'available_banks':[1] if runs else [],
        'first_run':runs[0].run_id if runs else '', 'latest_run':runs[-1].run_id if runs else '',
        'latest_file':Path(runs[-1].source_path).name if runs else '', 'latest_start_time':latest.get('start_time',''),
        'sample_display':latest.get('sample_name',''),
        'temperature_display':f"{latest['temperature']['value']:.5g} K" if latest.get('temperature') else 'Choose a reviewed measured-channel/unit binding',
        'selected_wavelength':profiles[-1]['wavelength_angstrom'] if profiles else '',
        'detected_wavelength':latest.get('nominal_wavelength_angstrom',''),
        'wavelength_source':'Calibrated CW profile; measured m1/Ge reflection is a sanity check',
        'profile_filename':Path(profile_path).name if profile_path else '',
        'profile_source':'User-reviewed cycle, Ge reflection, collimation and sample-environment mapping',
        'profile_error':'; '.join(errors[:5]), 'metadata_notices':latest.get('metadata_notices',[]),
        'quality_notices':['Nuclear phase identification does not explain magnetic-only peaks or solve magnetic order.',
            'Automatic HB-2A peak-profile calibration is not qualified; user review is required.'],
        'metadata_resolution':index.inventory(runs)['metadata_resolution'],
        'message':'; '.join(errors[:5]) if errors else f'Ready for reviewed CW analysis: {len(runs)} individual patterns, one combined detector bank.'}


class Hb2aWatchController(PowgenWatchController):
    resolver_version=hb2a.VERSION
    bank_provenance_key='hb2a_pattern'
    setup_metadata_key='hb2a_setup'

    @property
    def beamline_label(self):
        return 'HB2A'

    def __init__(self,service,settings,*,sample_index=None):
        super().__init__(service,settings,sample_index=sample_index or Hb2aSampleIndex(settings.ipts))
        self._supplied_profile=None
        self.display_profile_error=''

    def restore_latest_state(self):
        restored=super().restore_latest_state()
        if restored:
            try:
                self._profile_path()
                self.display_profile_error=''
            except (OSError,ValueError,RuntimeError) as exc:
                self.display_profile_error=f'Saved HB-2A profile unavailable: {exc}'
        return restored

    def _catalog_digest(self):
        return sha256(hb2a.VERSION.encode()).hexdigest()

    def _watch_extension(self):
        return {'hb2a_normalization':self.settings.normalization,
            'hb2a_profile_acknowledged':self.settings.profile_acknowledged,
            'hb2a_condition_bindings':self.settings.condition_bindings_json,
            'hb2a_collimator_override':self.settings.collimator_override,
            'hb2a_allow_stopped':self.settings.allow_stopped,
            'hb2a_excluded_acquisitions':list(self.settings.excluded_acquisitions)}

    def _list_source_directory(self):
        return hb2a.directory_listing(hb2a.proposal_root(self.settings.ipts)/self.settings.subfolder,
            normalization=self.settings.normalization)

    def _setup(self,run):
        return hb2a.inspect_reduced_setup(run.source_path,ipts=self.settings.ipts,
            condition_bindings=json.loads(self.settings.condition_bindings_json),
            collimator_override=self.settings.collimator_override,allow_stopped=self.settings.allow_stopped)

    def _read_scan_metadata(self,run):
        return hb2a.read_metadata(self.settings.ipts,run.experiment_number,run.run_number,
            condition_bindings=json.loads(self.settings.condition_bindings_json),
            collimator_override=self.settings.collimator_override,allow_stopped=self.settings.allow_stopped)

    def discover(self,listing=None,*,checkpoint=True):
        # Snapshot only files already present at start; no future scan list is
        # needed. New experiments may restart at scan 1, and late reductions of
        # earlier acquisitions are still seen by their full pair identity.
        rows=list(listing) if listing is not None else self._list_source_directory()
        available=discover_from_listing(self.definition,rows)
        current={r.run_id:r for r in available}
        initial=self.state.discovery_cursor.get('hb2a_initial_ids')
        if initial is None:
            initial=[r.run_id for r in available]
            self.state.discovery_cursor['hb2a_initial_ids']=initial
            selected=_selected(available,self.settings,self.sample_index)
            limit=self.settings.initial_backfill_limit
            candidates=[] if limit==0 else selected[-limit:] if limit is not None else selected
            self.state.initial_backfill_complete=True
        else:
            candidates=_selected([r for r in available if r.run_id not in set(initial)],self.settings,self.sample_index)
        # Pending parse failures are retried once a truncated data/raw file is
        # complete. Completed/submitted reductions are never silently replaced.
        for run_id,old in list(self.state.discovered.items()):
            if old.scan_metadata.get('setup_error') and run_id in current:
                del self.state.discovered[run_id]
                candidates.append(current[run_id])
        for phase in (self.state.discovered,self.state.submitted,self.state.completed,self.state.failed):
            for run_id,old in list(phase.items()):
                observed=current.get(run_id)
                setup=old.scan_metadata.get(self.setup_metadata_key) or {}
                try:
                    changed=observed is not None and (observed.fingerprint!=old.fingerprint or
                        list(hb2a.signature(hb2a.raw_path(self.settings.ipts,old.experiment_number,old.run_number)))!=setup.get('metadata',{}).get('spice_signature'))
                except OSError:
                    changed=True
                if changed and setup:
                    warning='HB-2A reduction or SPICE metadata changed after discovery; retained output belongs to its original input. Check a new watch to reprocess.'
                    phase[run_id]=replace(old,scan_metadata={**old.scan_metadata,'source_revision_warning':warning})
        result=[]
        for run in dict((r.run_id,r) for r in candidates).values():
            if self.state.contains(run):
                continue
            try:
                setup=self._setup(run)
                metadata={**setup.metadata, self.setup_metadata_key:setup.as_dict(),'source_bank':1,
                    'run_role':title_role(setup.metadata.get('run_title',''))}
            except (OSError,ValueError,UnicodeError) as exc:
                metadata={'setup_error':str(exc)}
            enriched=replace(run,scan_metadata=metadata)
            self.state.discovered[run.run_id]=enriched;result.append(enriched)
        self._primed=True
        if checkpoint:
            self.persist_state()
        return result

    def _profile_path(self):
        if not self.settings.instrument_profile_dataset_id:
            raise ValueError('HB-2A requires a reviewed GSAS-II CW profile')
        with self._profile_upload_lock:
            if self._supplied_profile is None:
                text=self.service.load_text_document(self.settings.instrument_profile_dataset_id)
                path=Path(self._staging.name)/'supplied.instprm';path.write_text(text,encoding='utf-8')
                self._supplied_profile=path
        return self._supplied_profile

    def _prepare_run(self,run):
        if run.scan_metadata.get('source_revision_warning') or run.scan_metadata.get('setup_error'):
            raise ValueError(run.scan_metadata.get('source_revision_warning') or run.scan_metadata['setup_error'])
        setup=self._setup(run)
        original=run.scan_metadata.get(self.setup_metadata_key) or {}
        if original.get('source_sha256')!=setup.source_sha256 or original.get('metadata',{}).get('spice_sha256')!=setup.metadata['spice_sha256']:
            raise ValueError('HB-2A input changed after discovery; check a new watch')
        profile=self._profile_path()
        contract=(setup.experiment,setup.metadata['ge_reflection'],setup.metadata['collimator'],
            setup.normalization,sha256(profile.read_bytes()).hexdigest(),setup.metadata['channel_schema_sha256'])
        with self._manual_setup_lock:
            if self._manual_setup_signature is not None and self._manual_setup_signature!=contract:
                raise ValueError('Experiment, Ge reflection, collimation or normalization changed; verify a new profile mapping')
            self._manual_setup_signature=contract
        return hb2a.stage_pattern(run.source_path,setup,profile,
            Path(self._staging.name)/run.fingerprint/run.run_id,acknowledged=self.settings.profile_acknowledged)

    def _run_name(self,run):
        setup=run.scan_metadata.get(self.setup_metadata_key) or {}
        key=sha256(json.dumps({'product':setup.get('source_sha256'),
            'spice':setup.get('metadata',{}).get('spice_sha256')},sort_keys=True).encode()).hexdigest()
        return f'{self.settings.ipts}_{run.run_id}_{key[:16]}'

    def launch_submission(self,run):
        plan=build_submission_plan(self.definition,run,self.state)
        if plan is None:
            if run.run_id in self.records:return self.records[run.run_id]
            raise ValueError('Run already acknowledged outside this controller')
        if self._configuration is None:
            with self._configuration_lock:
                if self._configuration is None:
                    self._configuration=self.service.load_configuration_dataset(self.settings.configuration_dataset_id)
        if self._configuration.radiation.value!='neutron' or self._configuration.instrument_mode=='tof':
            raise ValueError('HB-2A requires neutron CW settings, not TOF or Cu K-alpha')
        data,profile,receipt=self._prepare_run(run)
        profile_id=self._profile_dataset_id(profile)
        config=self._configuration.model_copy(update={'run_name':self._run_name(run),'instrument_mode':'cw'})
        inputs=InputSelection(source=InputSource.UPLOAD,data_path=str(data),bank='1',run_number=str(run.run_number),
            instrument_dataset_id=profile_id,main_cif_dataset_id=self.settings.main_cif_dataset_id,
            database_dataset_id=self.settings.database_dataset_id,facility_root='/HFIR',instrument='HB2A',ipts=self.settings.ipts,
            data_relative_path=f'{self.settings.subfolder}/{Path(run.source_path).name}')
        snapshot=self.service.create_submission_snapshot(config,inputs,idempotency_token=plan.idempotency_key,
            output_profile='monitor',prepared_dataset_ids={'configuration':self.settings.configuration_dataset_id})
        record=self.service.submit_snapshot(snapshot)
        if not record.galaxy_job_id:raise RuntimeError(f'Galaxy did not acknowledge {run.run_id}')
        record.provenance[self.bank_provenance_key]=receipt
        return record

    def _profile_matches_watch(self,record,run,candidate_instrument):
        if not candidate_instrument:return False
        try:
            _,profile,_=self._prepare_run(run)
            return self.service.load_text_document(candidate_instrument).replace('\r\n','\n')==profile.read_text().replace('\r\n','\n')
        except (OSError,ValueError,RuntimeError):return False

    def _scientific_summary(self,record):
        summary=super()._scientific_summary(record)
        if summary:
            summary['scientific_status']='needs_review'
            notices=['HB-2A profile requires independent calibration acceptance; magnetic-only peaks are outside nuclear phase identification.']
            summary['scientific_warnings']=list(dict.fromkeys([*summary.get('scientific_warnings',[]),*notices]))
        return summary
