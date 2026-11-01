"""Review and recover a finite battery portfolio in the existing workspace."""
import asyncio
from copy import deepcopy
import json
from gpx_recipe.battery.refinement_policy import validate_policy, policy_description
from gpx_recipe.battery.lithium import SCHEMA as LI_SCHEMA, validate_profile, profile_description

from .battery_campaigns import (MAX_TRIALS, prepare_portfolio, load_portfolio,
                               start_portfolio, portfolio_summary, pause_portfolio)
from .battery_preparation import preparation_root, verify_prepared


def common_width_choices(preparations):
    """Only offer phase/bank links present in every declared native preparation."""
    common = None
    for result in preparations:
        banks = {b['key']:b['physical_bank'] for b in result['banks']}
        links = {(p['name'],banks[link['bank']]) for p in result['phases'] for link in p['links']}
        common = links if common is None else common & links
    return [dict(value=json.dumps([phase,bank]), title=f'{phase} · physical bank {bank}',
                 phase=phase, physical_bank=bank) for phase,bank in sorted(common or [])]


def review_start_values(trial, li_profile=None):
    """Show physical bank labels without changing the sealed numerical input."""
    source=trial['source']
    banks={h['key']:source['metadata']['physical_banks'][h['name']]
           for h in source['context']['histograms']}
    values=deepcopy(trial['starting_values'])
    for phase in values:
        phase['starting_HAP_scales']=[dict(physical_bank=banks[s['histogram']],value=s['value'])
                                     for s in phase.pop('scales')]
        phase['fixed_atoms']=phase.pop('atoms')
        if li_profile:
            phase['atoms_with_occupancy_policy'] = phase.pop('fixed_atoms')
            for atom in phase['atoms_with_occupancy_policy']:
                atom['occupancy_policy'] = ('held grid value' if phase['name'] == li_profile['phase'] and atom['label'] == li_profile['atom'] else
                    'released mixing nuisance' if phase['name'] == li_profile['phase'] and atom['label'] == li_profile['mixing_atom'] else 'fixed')
        widths=[]
        for histogram,models in phase.pop('sample_widths',{}).items():
            row=dict(physical_bank=banks[histogram])
            for native,label,unit in (('Size','sample_size','µm'),('Mustrain','sample_microstrain','µstrain')):
                model_values=models.get(native)
                if model_values:
                    row[label]=dict(model=model_values[0],unit=unit,
                        value=model_values[1][0] if model_values[0]=='isotropic' else model_values[1])
            widths.append(row)
        phase['starting_sample_widths']=widths
    return json.dumps(values,indent=2)


def review_backgrounds(trial):
    banks=trial['source']['metadata']['physical_banks']
    return json.dumps([dict(physical_bank=banks[row['name']],background=row['background'])
                       for row in trial.get('starting_backgrounds',[])],indent=2)


def saved_portfolio_choices(workspace):
    choices=[]
    for token,record in reversed(list(workspace.state.get('battery_portfolios',{}).items())):
        try:
            document=load_portfolio(workspace,token)
            acquisition=document['observation']['acquisition']
            target=document.get('li_profile')
            model=(f"Li {target['phase'][:32]}/{target['atom'][:24]} · mixing {target['mixing_atom'] or 'fixed'}"
                   if target else 'Model comparison')
            completed=sum(workspace.state['jobs'].get(uid,{}).get('status')=='completed' for uid in record['jobs'].values())
            label=f"{acquisition['cell_id'][:40]} · {model} · {completed}/{len(document['trials'])} completed"
        except (OSError,ValueError,KeyError,TypeError):
            label='Saved review unavailable; reopen to inspect the error'
        choices.append(dict(value=token,title=f"{label} · {record['created_utc'][:19]} · {token[:8]}"))
    return choices


class BatteryCampaignMixin:
    def _init_battery_portfolio(self):
        self._battery_portfolio_entries = []
        self._battery_portfolio_token = None
        self._battery_portfolio_revision = 0
        self._battery_width_choices = {}
        s = self.server.state
        s.battery_portfolio_family = ''; s.battery_portfolio_seed = ''
        s.battery_portfolio_note = ''; s.battery_portfolio_band = '0.1'
        s.battery_portfolio_maximum = ''
        s.battery_portfolio_shift_maximum = '0.1'
        s.battery_portfolio_cycles = '6'
        s.battery_portfolio_background = 'preserve'; s.battery_portfolio_bg_coefficients = '6'
        s.battery_portfolio_sizes = []; s.battery_portfolio_strains = []
        s.battery_portfolio_width_options = []
        s.battery_portfolio_entries = []; s.battery_portfolio_review = []
        s.battery_portfolio_ready = False; s.battery_portfolio_started = False
        s.battery_portfolio_options = []; s.battery_portfolio_saved = ''
        s.battery_portfolio_trials = []; s.battery_portfolio_families = []
        s.battery_portfolio_policy = ''
        s.battery_portfolio_acquisition = ''
        s.battery_li_enabled = False; s.battery_li_phase = ''; s.battery_li_atom = ''
        s.battery_li_mixing_atom = ''; s.battery_li_refine_cells = True
        s.battery_li_result_ready = False; s.battery_li_notice = ''; s.battery_li_points = []
        s.battery_li_figure = dict(data=[], layout={})
        s.battery_portfolio_message = 'Add each prepared family/start explicitly, then review the finite trial set.'
        s.battery_portfolio_message_kind = 'info'
        s.change('battery_portfolio_note', 'battery_portfolio_band', 'battery_portfolio_maximum', 'battery_portfolio_shift_maximum',
                 'battery_portfolio_background', 'battery_portfolio_bg_coefficients',
                 'battery_portfolio_sizes', 'battery_portfolio_strains', 'battery_portfolio_cycles',
                 'battery_li_enabled', 'battery_li_phase', 'battery_li_atom', 'battery_li_mixing_atom',
                 'battery_li_refine_cells')(self._battery_portfolio_edited)

    def _battery_li_profile(self):
        s = self.server.state
        if type(s.battery_li_enabled) is not bool:
            raise ValueError('Choose whether to review held-Li sensitivity')
        if not s.battery_li_enabled:
            return None
        if any(not isinstance(getattr(s, key), str) for key in ('battery_li_phase', 'battery_li_atom', 'battery_li_mixing_atom')):
            raise ValueError('Enter exact phase and atom labels for held-Li sensitivity')
        return validate_profile(dict(schema=LI_SCHEMA, phase=s.battery_li_phase.strip(), atom=s.battery_li_atom.strip(),
            mixing_atom=s.battery_li_mixing_atom.strip() or None, refine_cells=s.battery_li_refine_cells))

    def _battery_update_width_choices(self, entries):
        choices = common_width_choices([verify_prepared(row['model']) for row in entries])
        self._battery_width_choices = {r['value']:r for r in choices}
        s = self.server.state
        s.battery_portfolio_width_options = [dict(title=r['title'],value=r['value']) for r in choices]
        for field in ('battery_portfolio_sizes','battery_portfolio_strains'):
            setattr(s,field,[value for value in getattr(s,field) if value in self._battery_width_choices])

    def _battery_refinement_policy(self):
        s = self.server.state
        if s.battery_portfolio_background not in {'preserve','chebyshev'}:
            raise ValueError('Choose the background policy')
        policy = validate_policy()
        if type(s.battery_portfolio_cycles) not in {int,str}:
            raise ValueError('Enter an integer cycle limit')
        policy['cycles_per_stage'] = int(s.battery_portfolio_cycles)
        if s.battery_portfolio_background == 'chebyshev':
            if type(s.battery_portfolio_bg_coefficients) not in (int,str):
                raise ValueError('Enter an integer background coefficient count')
            policy['background'] = dict(function='chebyschev-1',coefficients=int(s.battery_portfolio_bg_coefficients))
        for field,kind in (('battery_portfolio_sizes','sample_size'),('battery_portfolio_strains','sample_microstrain')):
            selected = getattr(s,field)
            if not isinstance(selected,list):raise ValueError('Choose explicit phase/bank sample widths')
            for value in selected:
                if not isinstance(value,str):raise ValueError('Choose explicit phase/bank sample widths')
                row = self._battery_width_choices.get(value)
                if row is None:raise ValueError('A sample-width target is no longer shared by the declared trials')
                policy['sample_widths'].append(dict(phase=row['phase'],physical_bank=row['physical_bank'],kind=kind))
        return validate_policy(policy)

    def _battery_portfolio_edited(self, **_):
        self._battery_portfolio_revision += 1
        self._battery_portfolio_token = None
        s = self.server.state
        s.battery_portfolio_ready = False; s.battery_portfolio_started = False
        s.battery_portfolio_review = []; s.battery_portfolio_trials = []; s.battery_portfolio_families = []
        s.battery_portfolio_policy=''; s.battery_portfolio_acquisition=''
        s.battery_li_result_ready=False; s.battery_li_notice=''; s.battery_li_points=[]
        s.battery_li_figure=dict(data=[],layout={})
        s.battery_portfolio_message = 'Trial set or policy changed. Preflight and review the revised portfolio.'
        s.battery_portfolio_message_kind = 'info'

    def add_battery_trial(self, **_):
        if self._battery_running(): return
        s = self.server.state
        try:
            path = self._battery_saved_models.get(s.battery_saved_model)
            if not path or not path.resolve().is_relative_to(preparation_root()):
                raise ValueError('Find and choose a saved prepared model first')
            family, seed = s.battery_portfolio_family.strip(), s.battery_portfolio_seed.strip()
            if not family or not seed or max(len(family),len(seed))>120:
                raise ValueError('Name the phase family and explicit start, each at most 120 characters')
            if len(self._battery_portfolio_entries)>=MAX_TRIALS:
                raise ValueError(f'A portfolio supports at most {MAX_TRIALS} explicit trials')
            if any((r['family'],r['seed'])==(family,seed) for r in self._battery_portfolio_entries):
                raise ValueError('This family/start label is already declared')
            entries=self._battery_portfolio_entries+[dict(model=str(path),family=family,seed=seed)]
            self._battery_update_width_choices(entries)
            self._battery_portfolio_entries=entries
            self._battery_portfolio_edited()
            s.battery_portfolio_entries=[dict(index=i,family=r['family'],seed=r['seed'],model=r['model'].rsplit('/',1)[-1])
                                        for i,r in enumerate(self._battery_portfolio_entries)]
            s.battery_error=''
        except (OSError,ValueError) as exc: s.battery_error=str(exc)
        s.flush()

    def clear_battery_trials(self, **_):
        if self._battery_running(): return
        self._battery_portfolio_entries=[]
        self._battery_update_width_choices([])
        self.server.state.battery_portfolio_entries=[]
        self.server.state.battery_error=''
        self._battery_portfolio_edited(); self.server.state.flush()

    def remove_battery_trial(self, index, **_):
        if self._battery_running(): return
        try:
            if type(index) is not int or not 0<=index<len(self._battery_portfolio_entries):
                raise ValueError('Choose a current declared trial')
            entries=self._battery_portfolio_entries[:index]+self._battery_portfolio_entries[index+1:]
            self._battery_update_width_choices(entries)
            self._battery_portfolio_entries=entries
            self._battery_portfolio_edited()
            self.server.state.battery_portfolio_entries=[dict(index=i,family=r['family'],seed=r['seed'],model=r['model'].rsplit('/',1)[-1])
                for i,r in enumerate(self._battery_portfolio_entries)]
            self.server.state.battery_error=''
        except (OSError,ValueError) as exc:self.server.state.battery_error=str(exc)
        self.server.state.flush()

    def review_battery_portfolio(self, **_):
        if self._battery_running(): return
        s = self.server.state
        try:
            entries = deepcopy(self._battery_portfolio_entries)
            if not entries: raise ValueError('Declare the family/start models first')
            band = float(s.battery_portfolio_band)
            maximum = float(s.battery_portfolio_maximum) if str(s.battery_portfolio_maximum).strip() else None
            shift_maximum = float(s.battery_portfolio_shift_maximum)
            policy = self._battery_refinement_policy()
            li_profile = self._battery_li_profile()
            workspace = self._refinement_store()
            self._battery_portfolio_edited()
            self._battery_start(self._review_battery_portfolio_async(workspace, entries,
                s.battery_portfolio_note, band, maximum, policy, self._battery_portfolio_revision, shift_maximum, li_profile))
        except (OSError,ValueError) as exc: s.battery_error=str(exc); s.flush()

    async def _review_battery_portfolio_async(self, workspace, entries, note, band, maximum, policy, revision, shift_maximum=0.1, li_profile=None):
        s = self.server.state
        s.battery_busy=True; s.battery_error=''
        s.battery_portfolio_message='Checking the shared observations and native parameter sets for every declared trial.'
        s.flush()
        try:
            token, _ = await asyncio.to_thread(prepare_portfolio, workspace, entries,
                note=note, close_fit_band=band, maximum_wss_per_point=maximum, refinement_policy=policy,
                maximum_last_cycle_shift_over_sigma=shift_maximum, li_profile=li_profile)
            if revision!=self._battery_portfolio_revision: return
            self._battery_portfolio_token=token
            self._battery_portfolio_sync()
        except Exception as exc:
            if revision==self._battery_portfolio_revision:
                s.battery_error=str(exc)
                s.battery_portfolio_message=f'Preflight stopped: {exc}. No trials were queued.'
                s.battery_portfolio_message_kind='error'
        finally: s.battery_busy=False; s.flush()

    def refresh_battery_portfolios(self, **_):
        s = self.server.state
        try:
            workspace=self._refinement_store()
            s.battery_portfolio_options=saved_portfolio_choices(workspace)
            self._battery_portfolio_sync()
        except Exception as exc: s.battery_error=str(exc)
        s.flush()

    def open_battery_portfolio(self, **_):
        if self._battery_running(): return
        s=self.server.state
        try:
            load_portfolio(self._refinement_store(),s.battery_portfolio_saved)
            self._battery_portfolio_token=s.battery_portfolio_saved
            self._battery_portfolio_sync()
            s.battery_error=''
        except Exception as exc: s.battery_error=str(exc)
        s.flush()

    def _battery_portfolio_sync(self):
        from .refinement_controller import recipe_cards
        token=self._battery_portfolio_token
        if not token: return
        workspace=self._refinement_store(); s=self.server.state
        document=load_portfolio(workspace,token)
        a=document['observation']['acquisition']
        s.battery_portfolio_acquisition=f"{a['instrument']} · {a['ipts']} · run {a['run']} · sample {a['sample_id']} · cell {a['cell_id']}"
        summary=portfolio_summary(workspace,token)
        s.battery_portfolio_ready=True
        s.battery_portfolio_started=workspace.state['battery_portfolios'][token]['status']=='started'
        s.battery_portfolio_review=[dict(id=t['id'],family=t['family'],seed=t['seed'],
            phases=', '.join(p['name'] for p in t['starting_values']),parameters=t['independent_parameters'],
            source=t['source']['source_sha256'][:12],
            retained_report=bool(t.get('retained_result_report')),
            stages=[dict(title=stage['title'],variables='; '.join(stage['free']),edits='; '.join(stage['edits']))
                    for stage in recipe_cards(t['plan'],t['source']['context'],
                        physical_banks=t['source']['metadata']['physical_banks'])],
            values=review_start_values(t, document.get('li_profile')),backgrounds=review_backgrounds(t)) for t in document['trials']]
        s.battery_portfolio_trials=[dict(r,
            status=r['status'].replace('_',' '),
            execution_display=r['execution_status'].replace('_',' '),
            score='—' if r['wss_per_point'] is None else f"{r['wss_per_point']:.5g}",
            bank_scores='; '.join(f"Bank {b['bank']}: {b['wss_per_point']:.5g}" for b in r['banks']),
            adequacy=r.get('adequacy','unavailable').replace('_',' '),
            convergence_display=r.get('convergence',{}).get('status','unavailable').replace('_',' '),
            message=' '.join(filter(None,[r.get('message',''),r.get('convergence',{}).get('message','')]))) for r in summary['trials']]
        start_labels={'incomplete':'Incomplete', 'single_start_only':'One start; sensitivity untested',
                      'start_sensitive':'Sensitive to starting values', 'within_screening_band':'Within declared band',
                      'convergence_unresolved':'Convergence unresolved across starts'}
        s.battery_portfolio_families=[dict(f,
            minimum_display='—' if f['minimum_wss_per_point'] is None else f"{f['minimum_wss_per_point']:.5g}",
            spread_display='—' if f['start_spread'] is None else f"{f['start_spread']:.5g}",
            start_check=start_labels[f['start_check']]) for f in summary['families']]
        s.battery_portfolio_message=(f"Verified comparison results: {summary['completed_count']}/{summary['declared_count']} · "
            f"{summary['status'].replace('_',' ')} · revision {token[:12]}. " +
            ('Li is held at each grid value; only the selected mixing occupancy may vary. Calibration and model adequacy require review.'
             if document.get('li_profile') else 'Occupancies and atomic inputs remain fixed. Calibration and model adequacy require review.'))
        s.battery_portfolio_message_kind=('warning' if summary['status'] in
            {'incomplete','optimization_unresolved','no_trial_passes_screen'} else 'info')
        s.battery_portfolio_policy=(policy_description(document.get('refinement_policy'), include_atomic_summary=not document.get('li_profile'))+' '+
            (profile_description(document['li_profile'])+' ' if document.get('li_profile') else '')+
            f"Close-fit screening band: {document['policy']['close_fit_band']:g} weighted residual per point. "
            + ('No adequacy threshold specified. ' if document['policy']['maximum_wss_per_point'] is None else
               f"Per-bank screening ceiling: {document['policy']['maximum_wss_per_point']:g}. ")
            + ('No convergence ceiling was recorded in this review. ' if document['policy'].get('maximum_last_cycle_shift_over_sigma') is None else
               f"Final-cycle shift/sigma ceiling: {document['policy']['maximum_last_cycle_shift_over_sigma']:g}; require all independent parameters and native convergence. ")
            + summary['metric_convention']+' '+document['note'])
        diagnostic = summary.get('li_profile')
        s.battery_li_result_ready = bool(diagnostic)
        s.battery_li_notice = ''; s.battery_li_points = []; s.battery_li_figure = dict(data=[],layout={})
        if diagnostic:
            from .refinement_controller import figure_json
            from .refinement_plots import li_profile_figure
            s.battery_li_notice = diagnostic['status'].replace('_', ' ').capitalize()+'. '+diagnostic['interpretation']
            s.battery_li_points = [dict(p, minimum_display='—' if p['minimum_wss_per_point'] is None else f"{p['minimum_wss_per_point']:.6g}",
                delta_display='—' if p['delta_wss_per_point'] is None else f"{p['delta_wss_per_point']:.6g}",
                reason_display=', '.join(r.replace('_',' ') for r in p['reasons']) or 'within reviewed optimization screens') for p in diagnostic['points']]
            s.battery_li_figure = figure_json(li_profile_figure(diagnostic, summary['trials'],revision=token))

    def start_battery_portfolio(self, **_):
        if self._battery_running(): return
        s=self.server.state
        try:
            if not self._battery_portfolio_token: raise ValueError('Review the current portfolio first')
            workspace=self._refinement_store()
            start_portfolio(workspace,self._battery_portfolio_token,approver='NOVA user reviewed finite battery portfolio')
            self._battery_portfolio_sync(); self._refinement_sync()
            self._refinement_start_pump()
        except Exception as exc: s.battery_error=str(exc)
        s.flush()

    def download_battery_portfolio(self, **_):
        try:
            workspace=self._refinement_store(); token=self._battery_portfolio_token
            document=dict(review=load_portfolio(workspace,token),results=portfolio_summary(workspace,token))
            self.download_file('battery-portfolio-'+token[:12]+'.json','application/json',
                               json.dumps(document,indent=2,allow_nan=False).encode())
        except Exception as exc: self.server.state.battery_error=str(exc); self.server.state.flush()

    def open_battery_trial_result(self, uid, **_):
        from .refinement_controller import refinement_experiment_options
        s=self.server.state
        try:
            workspace=self._refinement_store()
            record=workspace.state['battery_portfolios'][self._battery_portfolio_token]
            if uid not in record['jobs'].values():raise ValueError('Choose a job from the displayed portfolio')
            job=workspace.state['jobs'][uid]
            s.rf_experiments=[dict(title=label,value=key) for key,label in refinement_experiment_options(workspace).items()]
            s.rf_experiment=job['source']['experiment']; s.workspace_view='refinement'
            s.rf_transport='Local scientific worker'; s.setup_collapsed=True; s.flush()
            s.rf_mode='results'; self._refinement_sync()
            self._refinement_schedule(self._select_refinement_job(uid),'battery-trial-result')
        except Exception as exc:s.battery_error=str(exc)
        s.flush()

    def pause_battery_portfolio(self, paused=True, **_):
        try:
            if type(paused) is not bool:raise ValueError('Choose pause or resume')
            pause_portfolio(self._refinement_store(),self._battery_portfolio_token,paused)
            self._battery_portfolio_sync()
            if not paused:self._refinement_start_pump()
        except Exception as exc:self.server.state.battery_error=str(exc)
        self.server.state.flush()

    def retry_battery_trial(self, uid, **_):
        try:
            workspace=self._refinement_store()
            if uid not in workspace.state['battery_portfolios'][self._battery_portfolio_token]['jobs'].values():
                raise ValueError('Choose a trial from this portfolio')
            workspace.retry(uid);self._battery_portfolio_sync();self._refinement_start_pump()
        except Exception as exc:self.server.state.battery_error=str(exc)
        self.server.state.flush()
