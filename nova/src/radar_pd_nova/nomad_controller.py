"""NOMAD adapter sharing the established live job/retry/checkpoint lifecycle."""
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path

from . import nomad
from .acquisition import acquisition_context
from .models import InputSelection, InputSource
from .nomad_metadata import NomadSampleIndex, read_nomad_scan_metadata
from .powgen_controller import PowgenExperimentSettings, PowgenWatchController
from .powgen_samples import identity_readiness, select_runs
from .powgen_watch import WatchDefinition, discover_from_listing, build_submission_plan


@dataclass(frozen=True)
class NomadExperimentSettings(PowgenExperimentSettings):
    wavelength_angstrom: str = ''
    instrument: str = 'NOM'
    subfolder: str = nomad.MODERN_FOLDER
    profile_bank: int | None = None
    profile_acknowledged: bool = False
    allow_exploratory: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.instrument != 'NOM' or self.wavelength_angstrom:
            raise ValueError('NOMAD uses broadband TOF, not a POWGEN wavelength selection')
        if self.selected_banks is None or len(self.selected_banks)!=1 or self.selected_banks[0] not in range(1,7):
            raise ValueError('Choose one NOMAD physical bank (1-6) for this watch')
        if self.profile_bank is not None and (isinstance(self.profile_bank,bool) or not isinstance(self.profile_bank,int) or self.profile_bank<1):
            raise ValueError('Choose the instrument profile Bank label, not its ordinal block number')

    def definition(self, instrument_profile_ref=None):
        return WatchDefinition(self.ipts,self.subfolder,self.history_id,
            configuration_ref=self.configuration_dataset_id,
            instrument_profile_ref=instrument_profile_ref or self.instrument_profile_dataset_id or 'nomad-profile-required',
            instrument='NOM',main_cif_ref=self.main_cif_dataset_id,
            config_refs={'candidate_library':self.database_dataset_id or 'builtin',
                         'nomad_contract':json.dumps({'bank':self.selected_banks,'profile_bank':self.profile_bank,
                            'profile_acknowledged':self.profile_acknowledged,'allow_exploratory':self.allow_exploratory},sort_keys=True)})


def discover_nomad_samples(ipts, sample_index, subfolder=nomad.MODERN_FOLDER, *, progress=None):
    definition=WatchDefinition(ipts,subfolder,'samples','samples','profile-required',instrument='NOM')
    options=nomad.source_options(ipts)
    root=Path(definition.source_directory)
    runs=discover_from_listing(definition,nomad.directory_listing(root)) if root.is_dir() else []
    kwargs={'progress':progress} if progress is not None else {}
    result=sample_index.inventory(runs,**kwargs) if root.is_dir() else dict(options=[],run_options=[],scan_count=0,unidentified_count=0,source_unavailable=True)
    result['source_options']=options
    return result


def preflight_nomad_experiment(ipts, *, selected_sample_ids=None, selected_banks=None,
        sample_index=None,subfolder=nomad.MODERN_FOLDER,initial_backfill_limit=None,
        profile_path=None,profile_bank=None,profile_acknowledged=False,allow_exploratory=False,
        excluded_run_numbers=(),include_background_runs=False,include_ramp_runs=False):
    definition=WatchDefinition(ipts,subfolder,'preflight','preflight','profile-required',instrument='NOM')
    runs=discover_from_listing(definition,nomad.directory_listing(definition.source_directory))
    total=len(runs);index=sample_index or NomadSampleIndex(ipts)
    stats={}
    runs=select_runs(runs,index,selected_sample_ids,limit=initial_backfill_limit,
        excluded_run_numbers=excluded_run_numbers,include_background_runs=include_background_runs,
        include_ramp_runs=include_ramp_runs,selection_stats=stats)
    excluded=stats.get('excluded_count',0)
    errors=[];setups=[];available=set();notices=[];mapping=[];bank_quality={}
    for run in runs:
        try:
            # Enumerate structure even when the selected bank is invalid, so
            # the user can still choose a valid alternative. validate_profile
            # below enforces the selected bank's numerical checks.
            setup=nomad.inspect_reduced_setup(run.source_path, inventory_only=True);setups.append(setup)
            available.update(b.bank for b in setup.banks);notices.extend(setup.warnings)
            for bank in setup.banks:
                bank_quality.setdefault(bank.bank,set()).update(bank.quality_errors)
            if selected_banks is None or len(selected_banks)!=1:
                raise ValueError('Choose one NOMAD source histogram; each needs its own verified profile mapping')
            if not profile_path or profile_bank is None:
                raise ValueError('Supply a matching .instprm and its Bank label; automatic NOMAD profiles are not yet validated')
            _,receipt=nomad.validate_profile(setup,selected_banks[0],profile_path,profile_bank,acknowledged=profile_acknowledged)
            mapping.append(receipt)
            if setup.warnings and not allow_exploratory:
                raise ValueError('Reduction-quality notices require explicit exploratory-analysis acknowledgement')
        except (OSError,ValueError,UnicodeError) as exc:errors.append(f'{run.run_id}: {exc}')
    if len({(s.calibration,s.environment) for s in setups})>1:
        errors.append('The selected runs use different detector calibrations or sample environments; split them before reusing one supplied profile')
    if not runs:errors.append('No completed individual Bragg reductions match this selection; raw, summed, S(Q) and PDF products do not trigger analysis')
    latest=runs[-1] if runs else None
    metadata=read_nomad_scan_metadata(ipts,latest.run_number) if latest else {}
    notices.insert(0,'Supplied NOMAD profile is not automatically scientifically validated; outputs require scientific review')
    return {'ready':not errors,'source_directory':definition.source_directory,'scan_count':len(runs),
        'total_scan_count':total,'excluded_count':excluded,'analysis_count':len(mapping),
        'available_banks':sorted(available), 'bank_details':[{'bank':b.bank,'angle':b.angle,
            'approx_d_min':b.minimum_tof/b.difc,'approx_d_max':b.maximum_tof/b.difc,
            'quality_errors':sorted(bank_quality.get(b.bank,()))} for b in (setups[-1].banks if setups else [])],
        'first_run':runs[0].run_id if runs else '', 'latest_run':latest.run_id if latest else '',
        'latest_file':Path(latest.source_path).name if latest else '',
        'latest_start_time':metadata.get('start_time',''),'sample_display':f"{metadata.get('sample_id','')} | {metadata.get('sample_name','')}",
        'temperature_display':f"{metadata.get('temperature',{}).get('value','Not reported')} {metadata.get('temperature',{}).get('unit','')}",
        'selected_wavelength':'Broadband TOF','detected_wavelength':'','wavelength_source':'NOMAD: no monochromatic wavelength selection',
        'profile_filename':Path(profile_path).name if profile_path else '',
        'profile_source':'Explicit user-supplied physical/profile bank mapping; not automatically validated',
        'profile_error':'; '.join(errors[:5]),'quality_notices':list(dict.fromkeys(notices)),
        'metadata_resolution':identity_readiness(index,runs),
        'metadata_notices':metadata.get('metadata_notices',[]),
        'message':('; '.join(errors[:5]) if errors else f'Ready for exploratory analysis: {len(runs)} individual runs, physical bank {selected_banks[0]}. Review reduction and profile notices.')}


class NomadWatchController(PowgenWatchController):
    resolver_version=nomad.VERSION
    bank_provenance_key='nomad_bank'
    setup_metadata_key='nomad_setup'

    def __init__(self,service,settings,*,sample_index=None):
        super().__init__(service,settings,sample_index=sample_index or NomadSampleIndex(settings.ipts))
        self._supplied_profile=None

    def _catalog_digest(self):
        # No automatic registry is advertised before scientific acceptance.
        return sha256(nomad.VERSION.encode()).hexdigest()

    def _watch_extension(self):
        return {'nomad_profile_bank':self.settings.profile_bank,
                'nomad_profile_acknowledged':self.settings.profile_acknowledged,
                'nomad_allow_exploratory':self.settings.allow_exploratory,
                'nomad_main_cif_dataset_id':self.settings.main_cif_dataset_id}

    def _submission_identity(self, run):
        plan = build_submission_plan(self.definition, run)
        setup = run.scan_metadata.get('nomad_setup') or {}
        return sha256(json.dumps({'plan': plan.idempotency_key,
            'data': setup.get('source_sha256'), 'reduction': setup.get('sidecar_sha256')},
            sort_keys=True).encode()).hexdigest()

    def _run_name(self, run):
        # Galaxy retries also search by run name/configuration. Include the
        # immutable input identity so a re-reduction cannot recover another job.
        return f'{self.settings.ipts}_{run.run_id}_{self._submission_identity(run)[:16]}'

    def _list_source_directory(self):
        return nomad.directory_listing(self.source_directory)

    def _setup(self,run):
        path=Path(run.source_path);sidecar=nomad.reduction_input_path(path)
        bank=self.settings.selected_banks[0]
        key=(str(path),nomad.signature(path),nomad.signature(sidecar) if sidecar.is_file() else None,bank)
        if key not in self._setup_cache:self._setup_cache[key]=nomad.inspect_reduced_setup(path,selected_bank=bank)
        return self._setup_cache[key]

    def _read_scan_metadata(self,run):
        try:return read_nomad_scan_metadata(self.settings.ipts,run.run_number)
        except (OSError,ValueError,TypeError):return {}

    def discover(self, listing=None, *, checkpoint=True):
        rows = list(listing) if listing is not None else self._list_source_directory()
        # Inspect known paths even when the backfill floor excludes their run
        # numbers. Re-reduction must not silently relabel existing results.
        current = {r.source_path: r for r in discover_from_listing(self.definition, rows)}
        changed = False
        for phase in (self.state.discovered, self.state.submitted, self.state.completed, self.state.failed):
            for run_id, old in list(phase.items()):
                observed = current.get(old.source_path)
                if observed is None:
                    continue
                setup = old.scan_metadata.get('nomad_setup') or {}
                sidecar = nomad.reduction_input_path(old.source_path)
                side_signature = nomad.signature(sidecar) if sidecar.is_file() else None
                expected_side = setup.get('sidecar_signature')
                expected_side = tuple(expected_side) if expected_side is not None else None
                revision = [observed.fingerprint, list(side_signature) if side_signature is not None else None]
                if ((observed.fingerprint == old.fingerprint and side_signature == expected_side)
                        or old.scan_metadata.get('replacement_revision') == revision):
                    continue
                notice = ('Source reduction or reduction input was replaced after discovery; retained results '
                          'belong to the original input. Start a separately checked watch to reprocess.')
                summary = dict(old.scientific_summary)
                if summary:
                    summary['scientific_status'] = 'needs_review'
                    summary['scientific_warnings'] = list(dict.fromkeys([*summary.get('scientific_warnings', []), notice]))
                phase[run_id] = replace(old, scan_metadata={**old.scan_metadata,
                    'replacement_revision': revision, 'source_revision_warning': notice}, scientific_summary=summary)
                changed = True
        if changed and checkpoint:
            self.persist_state()
        return super().discover(rows, checkpoint=checkpoint)

    def _profile_path(self):
        if not self.settings.instrument_profile_dataset_id:
            raise ValueError('NOMAD requires a matching user-supplied GSAS-II profile')
        with self._profile_upload_lock:
            if self._supplied_profile is None:
                text=self.service.load_text_document(self.settings.instrument_profile_dataset_id)
                path=Path(self._staging.name)/'supplied.instprm';path.write_text(text,encoding='utf-8')
                nomad.profile_blocks(path);self._supplied_profile=path
        return self._supplied_profile

    def _prepare_run(self,run):
        setup=self._setup(run)
        discovered_setup = run.scan_metadata.get('nomad_setup') or {}
        if (run.scan_metadata.get('source_revision_warning') or
                discovered_setup.get('source_sha256') != setup.source_sha256 or
                discovered_setup.get('sidecar_sha256') != setup.sidecar_sha256):
            raise ValueError('NOMAD source changed after discovery; check a new watch before reprocessing')
        if setup.warnings and not self.settings.allow_exploratory:
            raise ValueError('NOMAD reduction corrections require exploratory-analysis acknowledgement')
        bank=run.bank or self.settings.selected_banks[0]
        profile=self._profile_path()
        selected=next((b for b in setup.banks if b.bank==bank),None)
        if selected is None:
            raise ValueError('Selected physical bank is absent from this run')
        contract=(setup.calibration,setup.environment,bank,selected.difc,selected.angle,
                  sha256(profile.read_bytes()).hexdigest())
        with self._manual_setup_lock:
            if self._manual_setup_signature is not None and self._manual_setup_signature!=contract:
                raise ValueError('NOMAD detector calibration, environment, bank geometry or supplied profile changed; check a new watch')
            self._manual_setup_signature=contract
        return nomad.stage_bank(Path(run.source_path),setup,bank,profile,self.settings.profile_bank,
            Path(self._staging.name)/run.fingerprint/run.run_id,acknowledged=self.settings.profile_acknowledged,
            acquisition=acquisition_context(run.scan_metadata,instrument='NOM',run_number=run.run_number,source=run.source_path))

    def launch_submission(self,run):
        plan=build_submission_plan(self.definition,run,self.state)
        if plan is None:
            if run.run_id in self.records:return self.records[run.run_id]
            raise ValueError('Run already submitted outside this controller')
        if self._configuration is None:
            with self._configuration_lock:
                if self._configuration is None:self._configuration=self.service.load_configuration_dataset(self.settings.configuration_dataset_id)
        if self._configuration.radiation.value!='neutron' or self._configuration.instrument_mode=='cw':
            raise ValueError('NOMAD requires a neutron TOF configuration')
        data,profile,receipt=self._prepare_run(run)
        profile_id=self._profile_dataset_id(profile)
        config=self._configuration.model_copy(update={'run_name':self._run_name(run),'instrument_mode':'tof'})
        inputs=InputSelection(source=InputSource.UPLOAD,data_path=str(data),bank=str(receipt['source_bank']),
            run_number=str(run.run_number),instrument_dataset_id=profile_id,
            main_cif_dataset_id=self.settings.main_cif_dataset_id,database_dataset_id=self.settings.database_dataset_id,
            facility_root='/SNS',instrument='NOM',ipts=self.settings.ipts,
            data_relative_path=f'{self.settings.subfolder}/{Path(run.source_path).name}')
        snapshot=self.service.create_submission_snapshot(config,inputs,idempotency_token=self._submission_identity(run),
            output_profile='monitor',prepared_dataset_ids={'configuration':self.settings.configuration_dataset_id})
        record=self.service.submit_snapshot(snapshot)
        if not record.galaxy_job_id:raise RuntimeError(f'Galaxy did not acknowledge {run.run_id}')
        receipt['staged_profile_dataset_id']=profile_id
        record.provenance[self.bank_provenance_key]=receipt
        return record

    def _profile_matches_watch(self,record,run,candidate_instrument):
        if not candidate_instrument:return False
        try:
            _,profile,_=self._prepare_run(run)
            actual=self.service.load_text_document(candidate_instrument)
            return actual.replace('\r\n','\n')==profile.read_text().replace('\r\n','\n')
        except (OSError,ValueError,RuntimeError):return False

    def _scientific_summary(self,record):
        summary=super()._scientific_summary(record)
        if summary:
            notices=['NOMAD profile supplied by user; scientific calibration acceptance is pending']
            receipt=record.provenance.get(self.bank_provenance_key) or {}
            notices.extend(receipt.get('warnings') or [])
            for phase in (self.state.submitted, self.state.completed):
                for run in phase.values():
                    if run.galaxy_job_id == record.galaxy_job_id:
                        if not receipt:
                            notices.extend((run.scan_metadata.get('nomad_bank_provenance') or {}).get('warnings') or [])
                        if run.scan_metadata.get('source_revision_warning'):
                            notices.append(run.scan_metadata['source_revision_warning'])
            summary['scientific_status']='needs_review'
            summary['scientific_warnings']=list(dict.fromkeys([*summary.get('scientific_warnings',[]),*notices]))
        return summary
