"""SNAP adapter; the established watch owns retries, jobs and checkpoints."""
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path

from . import snap
from .models import InputSelection, InputSource
from .powgen_controller import PowgenExperimentSettings, PowgenWatchController
from .powgen_samples import identity_readiness
from .facility_metadata import title_role
from .powgen_watch import WatchDefinition, discover_from_listing, build_submission_plan
from .snap_metadata import SnapSampleIndex, read_snap_scan_metadata


@dataclass(frozen=True)
class SnapExperimentSettings(PowgenExperimentSettings):
    wavelength_angstrom: str = ''
    instrument: str = 'SNAP'
    subfolder: str = snap.DEFAULT_FOLDER
    product_family: str = ''
    profile_bank: int | None = None
    profile_acknowledged: bool = False
    allow_exploratory: bool = False

    def __post_init__(self):
        super().__post_init__()
        snap.source_directory(self.ipts,self.subfolder)
        if self.instrument != 'SNAP' or self.wavelength_angstrom:
            raise ValueError('SNAP is broadband neutron TOF; no monochromatic wavelength override applies')
        if self.selected_banks is None or len(self.selected_banks) != 1:
            raise ValueError('Choose one SNAP source histogram per explicit profile mapping')
        if self.profile_bank is not None and (isinstance(self.profile_bank,bool) or not isinstance(self.profile_bank,int) or self.profile_bank < 1):
            raise ValueError('Enter the Bank label declared by the SNAP profile')
        if not self.product_family:
            raise ValueError('Choose a SNAP grouping/export variant; re-reductions are not silently merged')
        try:
            family = json.loads(self.product_family)
            if (not isinstance(family,list) or len(family)!=2 or
                    any(not isinstance(v,str) or len(v)>150 for v in family)):
                raise ValueError()
        except (ValueError,TypeError) as exc:
            raise ValueError('Invalid SNAP export-family selection') from exc
        if '/export/gsa/' in self.subfolder and family[0] != Path(self.subfolder).name:
            raise ValueError('SNAP export family does not match the selected grouping folder')

    def definition(self, instrument_profile_ref=None):
        return WatchDefinition(self.ipts,self.subfolder,self.history_id,
            configuration_ref=self.configuration_dataset_id,
            instrument_profile_ref=instrument_profile_ref or self.instrument_profile_dataset_id or 'snap-profile-required',
            instrument='SNAP',main_cif_ref=self.main_cif_dataset_id,
            config_refs={'candidate_library':self.database_dataset_id or 'builtin',
                         'snap_contract':json.dumps({'family':self.product_family,'banks':self.selected_banks,
                            'profile_bank':self.profile_bank,'acknowledged':self.profile_acknowledged,
                            'exploratory':self.allow_exploratory},sort_keys=True)})


def _listing(ipts, subfolder, product_family=''):
    root = snap.source_directory(ipts,subfolder)
    rows = snap.directory_listing(root) if root.is_dir() else []
    families = snap.product_families(rows)
    family = product_family or (families[0]['value'] if len(families) == 1 else '')
    return [r for r in rows if r['snap_product_family'] == family], families, family


def discover_snap_samples(ipts, sample_index, subfolder=snap.DEFAULT_FOLDER, product_family=''):
    rows, families, selected = _listing(ipts,subfolder,product_family)
    definition = WatchDefinition(ipts,subfolder,'samples','samples','profile-required',instrument='SNAP')
    runs = discover_from_listing(definition,rows)
    result = sample_index.inventory(runs,catalog=True)
    result.update(source_options=snap.source_options(ipts),product_family_options=families,product_family=selected)
    if len(families) > 1 and not selected:
        result['metadata_notices'].append('Choose one SNAP export variant before sample discovery; different re-reductions are not merged')
    return result


def _select_runs(runs, index, selected_ids, *, limit=None, excluded_run_numbers=(),
                 include_background_runs=False, include_ramp_runs=False):
    """Apply actual NeXus/catalog titles before the initial scan limit.

    Modern SNAP GSA headers contain geometry rather than a sample title. Read
    identities only until enough newest eligible runs have been selected;
    selected candidates are verified from source, never a catalog alone.
    """
    runs = list(runs)
    if selected_ids is not None and hasattr(index,'candidate_runs'):
        runs = index.candidate_runs(runs,selected_ids)
    selected=[]
    for run in reversed(list(runs)):
        if run.run_number in excluded_run_numbers:
            continue
        metadata=index.identity(run)
        if selected_ids is not None and metadata.get('sample_id') not in selected_ids:
            continue
        role=title_role(metadata.get('run_title') or '')
        if (role=='background' and not include_background_runs) or (role=='ramp' and not include_ramp_runs):
            continue
        selected.append(replace(run,scan_metadata={**run.scan_metadata,**metadata,'run_role':role}))
        if limit is not None and len(selected)>=max(1,limit):
            break
    return list(reversed(selected))


def preflight_snap_experiment(ipts, *, selected_sample_ids=None, selected_banks=None,
        sample_index=None,subfolder=snap.DEFAULT_FOLDER,initial_backfill_limit=None,
        profile_path=None,profile_bank=None,profile_acknowledged=False,allow_exploratory=False,
        product_family='',excluded_run_numbers=(),include_background_runs=False,include_ramp_runs=False):
    rows,families,family = _listing(ipts,subfolder,product_family)
    definition = WatchDefinition(ipts,subfolder,'preflight','preflight','profile-required',instrument='SNAP')
    all_runs = discover_from_listing(definition,rows)
    index = sample_index or SnapSampleIndex(ipts)
    runs = _select_runs(all_runs,index,selected_sample_ids,limit=initial_backfill_limit,
        excluded_run_numbers=excluded_run_numbers,include_background_runs=include_background_runs,include_ramp_runs=include_ramp_runs)
    errors,notices,setups,mappings = [],[],[],[]
    available = {}
    for run in runs:
        try:
            setup = snap.inspect_reduced_setup(run.source_path,inventory_only=True)
            setups.append(setup);notices.extend(setup.warnings)
            for bank in setup.banks:
                available.setdefault(bank.bank,bank)
            if selected_banks is None or len(selected_banks) != 1:
                raise ValueError('Choose one SNAP source histogram; banks/focusing groups are analyzed separately')
            if not profile_path or profile_bank is None:
                raise ValueError('Supply a reviewed matching SNAP .instprm and profile Bank label; auto-generated adjacent profiles are approximate')
            _,receipt = snap.validate_profile(setup,selected_banks[0],profile_path,profile_bank,acknowledged=profile_acknowledged)
            mappings.append(receipt)
            if setup.warnings and not allow_exploratory:
                raise ValueError('Missing SNAP reduction/state provenance requires explicit exploratory-analysis acknowledgement')
        except (OSError,ValueError,UnicodeError) as exc:
            errors.append(f'{run.run_id}: {exc}')
    contracts = {(s.state_id,s.use_lite_mode,s.calibration,s.normalization,s.corrections,s.grouping,s.product_variant) for s in setups}
    if len(contracts) > 1:
        errors.append('Selected SNAP runs span different instrument states, normalization/calibration versions or export variants; split and review each mapping')
    if not runs:
        errors.append('No unique individual GSAS exports match the selected folder, export variant and sample; raw/reduced NeXus alone cannot trigger a fit')
    latest = runs[-1] if runs else None
    metadata = read_snap_scan_metadata(ipts,latest.run_number) if latest else {}
    notices.insert(0,'SNAP profiles and quantitative fractions require scientific review; an adjacent generated .instprm is not automatically calibrated')
    return {'ready':not errors,'source_directory':definition.source_directory,'scan_count':len(runs),
        'total_scan_count':len(all_runs),'analysis_count':len(mappings),'available_banks':sorted(available),
        'bank_details':[{'bank':b.bank,'angle':b.angle,'approx_d_min':b.minimum_tof/b.difc,
                         'approx_d_max':b.maximum_tof/b.difc,'quality_errors':list(b.quality_errors),
                         'masked_bins':b.masked_bins,'valid_bins':b.valid_bins} for b in available.values()],
        'product_family_options':families,'product_family':family,
        'first_run':runs[0].run_id if runs else '', 'latest_run':latest.run_id if latest else '',
        'latest_file':Path(latest.source_path).name if latest else '',
        'latest_start_time':metadata.get('start_time',''),'sample_display':f"{metadata.get('sample_id','')} | {metadata.get('sample_name','')}",
        'temperature_display':'Select a verified specimen temperature sensor or import a reviewed axis mapping',
        'selected_wavelength':'Broadband TOF','detected_wavelength':'','wavelength_source':'SNAP: no monochromatic wavelength selection',
        'profile_filename':Path(profile_path).name if profile_path else '',
        'profile_source':'Explicit SNAP state/group/profile mapping; calibration acceptance pending',
        'profile_error':'; '.join(errors[:5]),'quality_notices':list(dict.fromkeys(notices)),
        'metadata_resolution':identity_readiness(index,runs),'metadata_notices':metadata.get('metadata_notices',[]),
        'message':'; '.join(errors[:5]) if errors else f'Ready for reviewed exploratory SNAP analysis: {len(runs)} runs, source histogram {selected_banks[0]}'}


class SnapWatchController(PowgenWatchController):
    resolver_version = snap.VERSION
    bank_provenance_key = 'snap_bank'
    setup_metadata_key = 'snap_setup'

    @property
    def beamline_label(self):
        return 'SNAP'

    def __init__(self,service,settings,*,sample_index=None):
        super().__init__(service,settings,sample_index=sample_index or SnapSampleIndex(settings.ipts))
        self._supplied_profile = None

    def _catalog_digest(self):
        return sha256(snap.VERSION.encode()).hexdigest()

    def _watch_extension(self):
        return {'snap_product_family':self.settings.product_family,'snap_profile_bank':self.settings.profile_bank,
                'snap_profile_acknowledged':self.settings.profile_acknowledged,'snap_allow_exploratory':self.settings.allow_exploratory}

    def _list_source_directory(self):
        return snap.directory_listing(self.source_directory,family=self.settings.product_family)

    def _matching_samples(self,runs,*,limit=None):
        return _select_runs(runs,self.sample_index,self.settings.selected_sample_ids,limit=limit,
            excluded_run_numbers=self.settings.excluded_run_numbers,include_background_runs=self.settings.include_background_runs,
            include_ramp_runs=self.settings.include_ramp_runs)

    def _setup(self,run):
        path = Path(run.source_path)
        key = str(path),snap.signature(path)
        previous = self._setup_cache.get(key)
        if previous:
            record = Path(previous.record_path) if previous.record_path else None
            current = snap.signature(record) if record and record.is_file() else None
            if current == previous.record_signature:
                return previous
        setup = snap.inspect_reduced_setup(path,selected_bank=self.settings.selected_banks[0])
        self._setup_cache[key] = setup
        return setup

    def _read_scan_metadata(self,run):
        try:
            return {**read_snap_scan_metadata(self.settings.ipts,run.run_number),'source_family':'SNAP individual GSAS/SLOG'}
        except (OSError,ValueError,TypeError) as exc:
            return {'source_family':'SNAP individual GSAS/SLOG','available':False,
                    'metadata_notices':[f'SNAP acquisition metadata needs review: {exc}']}

    def _submission_identity(self,run):
        plan = build_submission_plan(self.definition,run)
        setup = run.scan_metadata.get(self.setup_metadata_key) or {}
        return sha256(json.dumps({'plan':plan.idempotency_key,'data':setup.get('source_sha256'),
                                 'receipt':setup.get('record_sha256')},sort_keys=True).encode()).hexdigest()

    def _run_name(self,run):
        return f'{self.settings.ipts}_{run.run_id}_{self._submission_identity(run)[:16]}'

    def discover(self,listing=None,*,checkpoint=True):
        rows = list(listing) if listing is not None else self._list_source_directory()
        current = {r.source_path:r for r in discover_from_listing(self.definition,rows)}
        changed = False
        for phase in (self.state.discovered,self.state.submitted,self.state.completed,self.state.failed):
            for key,old in list(phase.items()):
                observed = current.get(old.source_path)
                setup = old.scan_metadata.get(self.setup_metadata_key) or {}
                if not observed or not setup or old.scan_metadata.get('setup_error'):
                    continue
                record = Path(setup['record_path']) if setup.get('record_path') else None
                signature = snap.signature(record) if record and record.is_file() else None
                expected = tuple(setup['record_signature']) if setup.get('record_signature') else None
                revision = [observed.fingerprint,list(signature) if signature else None]
                if ((observed.fingerprint == old.fingerprint and signature == expected)
                        or old.scan_metadata.get('replacement_revision') == revision):
                    continue
                message = 'SNAP reduction/receipt was replaced after discovery; retained results belong to the original input. Check a separate watch to reprocess.'
                summary = dict(old.scientific_summary)
                if summary:
                    summary.update(scientific_status='needs_review',scientific_warnings=list(dict.fromkeys([*summary.get('scientific_warnings',[]),message])))
                phase[key] = replace(old,scan_metadata={**old.scan_metadata,'source_revision_warning':message,
                    'replacement_revision':revision},scientific_summary=summary)
                changed = True
        if changed and checkpoint:
            self.persist_state()
        return super().discover(rows,checkpoint=checkpoint)

    def _profile_path(self):
        if not self.settings.instrument_profile_dataset_id:
            raise ValueError('SNAP requires a matching reviewed GSAS-II profile')
        with self._profile_upload_lock:
            if self._supplied_profile is None:
                text = self.service.load_text_document(self.settings.instrument_profile_dataset_id)
                path = Path(self._staging.name)/'supplied.instprm';path.write_text(text,encoding='utf-8')
                snap.profile_blocks(path);self._supplied_profile = path
        return self._supplied_profile

    def _prepare_run(self,run):
        setup = self._setup(run)
        discovered = run.scan_metadata.get(self.setup_metadata_key) or {}
        if (run.scan_metadata.get('source_revision_warning') or discovered.get('source_sha256') != setup.source_sha256
                or discovered.get('record_sha256') != setup.record_sha256):
            raise ValueError('SNAP source changed after discovery; check a new watch before reprocessing')
        if setup.warnings and not self.settings.allow_exploratory:
            raise ValueError('Missing SNAP reduction provenance requires exploratory acknowledgement')
        bank = run.bank or self.settings.selected_banks[0]
        profile = self._profile_path()
        contract = (setup.state_id,setup.grouping,setup.product_variant,setup.use_lite_mode,
                    setup.calibration,setup.normalization,setup.corrections,bank,sha256(profile.read_bytes()).hexdigest())
        with self._manual_setup_lock:
            if self._manual_setup_signature is not None and self._manual_setup_signature != contract:
                raise ValueError('SNAP state, grouping, calibration, normalization or supplied profile changed; check a new watch')
            self._manual_setup_signature = contract
        if run.scan_metadata.get('run_title'):
            setup=replace(setup,title=run.scan_metadata['run_title'])
        return snap.stage_bank(run.source_path,setup,bank,profile,self.settings.profile_bank,
            Path(self._staging.name)/run.fingerprint/run.run_id,acknowledged=self.settings.profile_acknowledged)

    def launch_submission(self,run):
        if build_submission_plan(self.definition,run,self.state) is None:
            if run.run_id in self.records:
                return self.records[run.run_id]
            raise ValueError('SNAP run already submitted outside this controller')
        if self._configuration is None:
            with self._configuration_lock:
                if self._configuration is None:
                    self._configuration = self.service.load_configuration_dataset(self.settings.configuration_dataset_id)
        if self._configuration.radiation.value != 'neutron' or self._configuration.instrument_mode == 'cw':
            raise ValueError('SNAP requires neutron TOF analysis settings')
        data,profile,receipt = self._prepare_run(run)
        profile_id = self._profile_dataset_id(profile)
        config = self._configuration.model_copy(update={'run_name':self._run_name(run),'instrument_mode':'tof'})
        inputs = InputSelection(source=InputSource.UPLOAD,data_path=str(data),bank=str(receipt['source_bank']),
            run_number=str(run.run_number),instrument_dataset_id=profile_id,main_cif_dataset_id=self.settings.main_cif_dataset_id,
            database_dataset_id=self.settings.database_dataset_id,facility_root='/SNS',instrument='SNAP',ipts=self.settings.ipts,
            data_relative_path=f'{self.settings.subfolder}/{Path(run.source_path).name}')
        snapshot = self.service.create_submission_snapshot(config,inputs,idempotency_token=self._submission_identity(run),
            output_profile='monitor',prepared_dataset_ids={'configuration':self.settings.configuration_dataset_id})
        record = self.service.submit_snapshot(snapshot)
        if not record.galaxy_job_id:
            raise RuntimeError(f'Galaxy did not acknowledge {run.run_id}')
        receipt['staged_profile_dataset_id'] = profile_id
        record.provenance[self.bank_provenance_key] = receipt
        return record

    def _profile_matches_watch(self,record,run,candidate_instrument):
        if not candidate_instrument:
            return False
        try:
            _,profile,_ = self._prepare_run(run)
            return self.service.load_text_document(candidate_instrument).replace('\r\n','\n') == profile.read_text().replace('\r\n','\n')
        except (OSError,ValueError,RuntimeError):
            return False

    def _scientific_summary(self,record):
        summary = super()._scientific_summary(record)
        if summary:
            receipt = record.provenance.get(self.bank_provenance_key) or {}
            summary['scientific_status'] = 'needs_review'
            summary['scientific_warnings'] = list(dict.fromkeys([*summary.get('scientific_warnings',[]),
                *receipt.get('warnings',[]),'SNAP profile mapping is user-reviewed; quantitative scientific acceptance is pending']))
        return summary
