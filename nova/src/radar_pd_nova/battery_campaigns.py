"""Finite battery portfolios using the existing refinement workspace queue.

Every start is a separately prepared, immutable model. This adapter does not
guess phases, warm-start a neighbouring exposure, or select a physical model.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import threading
import uuid

from gpx_recipe.battery.refinement_policy import validate_policy, policy_recipe, width_targets, WIDTH_KINDS
from gpx_recipe.battery.lithium import (validate_profile, profile_target, profile_recipe, profile_binding,
                                      verify_profile_preview, verify_profile_result, summarize_profile)
from gpx_recipe.compiler import approve as approve_plan
from gpx_recipe.contracts import plan_hash
from gpx_recipe.convergence import assess_convergence
from gpx_recipe.execution import write_json
from gpx_recipe.fit_domain import DOMAIN_SCHEMA, LEGACY_DOMAIN_SCHEMA, NATIVE_FIT_CONVENTION, fit_domain_digest
from .battery_preparation import prepared_source, verify_prepared
from .refinement_runtime import verify_result_files
from .refinement_sources import digest, file_sha, RefinementScope
from .refinement_workspace import utc, refinement_job_key
from .refinement_messages import sample_width_failure_message

SCHEMA = 'radar-battery-portfolio/v1'
MAX_TRIALS = 24
IDENTITY = ('instrument', 'ipts', 'run', 'sample_id', 'cell_id',
            'start_utc', 'end_utc', 'metadata_sha256')


class ComparisonConventionChanged(ValueError):
    """Valid retained output needs a new review before its metrics are compared."""


def _label(value, what, maximum=120):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f'Provide a {what} with 1–{maximum} characters')
    return value.strip()


def starting_values(context):
    """Scientific values, excluding random native IDs and prior fit outcomes."""
    return [dict(name=p['name'], cell=p['cell'], scales=p['phase_scales'],
                 atoms=[{k: a[k] for k in ('label', 'element', 'coordinates',
                          'occupancy', 'adp_type', 'adp', 'multiplicity')}
                        for a in p['atoms']],
                 sample_widths={h: {k:deepcopy(v) for k,v in values.items() if k in {'Size','Mustrain'}}
                                for h,values in p.get('histogram_phase_models', {}).items()})
            for p in context['phases']]


def fixed_hypothesis(context, physical_banks, policy):
    released = {(t['phase'], t['histogram'], WIDTH_KINDS[t['kind']])
                for t in width_targets(context, physical_banks, policy)}
    fixed = []
    for phase in context['phases']:
        hap = {h: {k:deepcopy(v) for k,v in values.items() if k!='Scale'}
               for h,values in phase.get('histogram_phase_models', {}).items()}
        for histogram, models in hap.items():
            for kind in ('Size', 'Mustrain'):
                if (phase['key'], histogram, kind) in released:
                    if not models.get(kind) or models[kind][0] != 'isotropic':
                        raise ValueError('A released sample width requires an existing isotropic model')
                    models[kind][1][0] = 'released_start'
        fixed.append(dict(name=phase['name'], atoms=starting_values({'phases':[phase]})[0]['atoms'],
                          texture=phase.get('global_texture'), hap=hap))
    return fixed


def observation_contract(view, physical_banks, *, domain_schema=DOMAIN_SCHEMA):
    rows = []
    for histogram in view['histograms']:
        bank = physical_banks.get(histogram['name'])
        sha = fit_domain_digest(histogram, schema=domain_schema)
        if type(bank) is not int or not sha:
            raise ValueError('Every compared histogram needs verified weights and a physical bank')
        instrument = deepcopy(histogram['instrument_parameters'])
        # GSASIIstrIO.GetHistogramData.GetInstParms adds this absent term before
        # refinement. For neutron TOF it is the constant Lorentzian width, not
        # the TOF Zero offset. Only its exact fixed-zero default is equivalent.
        if instrument.get('Type', [''])[0] == 'PNT' and 'Z' not in instrument:
            instrument['Z'] = [0.0, 0.0, False]
        rows.append(dict(bank=bank, observations=sha, instrument=digest(instrument)))
    if not rows or len({r['bank'] for r in rows}) != len(rows):
        raise ValueError('Physical banks must be unique in a portfolio observation')
    return sorted(rows, key=lambda r: r['bank'])


def _read_artifact(directory, response, key):
    verify_result_files(directory, response)
    name = response[key]
    if not isinstance(name, str) or name not in response.get('files', {}):
        raise ValueError('The scientific view is absent from the verified artifact manifest')
    path = (Path(directory)/name).resolve()
    if not path.is_relative_to(Path(directory).resolve()) or path.stat().st_size > 64*1024*1024:
        raise ValueError('The scientific view is outside its bundle or exceeds 64 MiB')
    return json.loads(path.read_text())


def _current_observation_view(workspace, source):
    """Upgrade a cached inspection separately; never rewrite a sealed artifact."""
    directory = Path(source['inspection'])
    response = json.loads((directory/'response.json').read_text())
    if response['source_sha256'] != source['source_sha256']:
        raise ValueError('Inspection belongs to another prepared revision')
    view = _read_artifact(directory, response, 'view')
    if view.get('histograms') and all(fit_domain_digest(h, schema=DOMAIN_SCHEMA) for h in view['histograms']):
        return source, view
    return _native_observation_view(workspace, source)


def _native_observation_view(workspace, source, *, revision='native-domain-v4'):
    key = 'battery-native/'+revision+'/'+source['source_sha256']
    with workspace.lock:
        lock = workspace._inspection_locks.setdefault(key, threading.Lock())
    with lock:
        return _native_observation_view_locked(workspace, source, revision=revision)


def _native_observation_view_locked(workspace, source, *, revision='native-domain-v4'):
    # Generic recipe setup may retain an older, valid cached context. New
    # battery comparisons need the native endpoint convention introduced in
    # v4, without altering any old review, job or its original observation hash.
    directory = workspace.root/'battery-inspections'/source['source_sha256']/revision
    if (directory/'response.json').exists():
        response = json.loads((directory/'response.json').read_text())
    else:
        response = workspace.runtime.call('inspect', source, directory,
            dispatch_key=digest(['battery-inspect', revision, source['source_sha256']])[:24])
    if (response.get('status') != 'inspected' or response.get('source_sha256') != source['source_sha256']
            or response.get('context', {}).get('source', {}).get('sha256') != source['source_sha256']):
        raise ValueError('Native fitted-bin inspection could not be verified')
    view = _read_artifact(directory, response, 'view')
    if not view.get('histograms') or not all(fit_domain_digest(h, schema=DOMAIN_SCHEMA) for h in view['histograms']):
        raise ValueError('Native fitted-bin inspection lacks the corrected observation contract')
    source = dict(source, context=response['context'], inspection=str(directory))
    return source, view


def _retained_result_report(workspace, source, plan, contract):
    """Re-export a reused fit during review, never during summary or execution."""
    uid = refinement_job_key(source, plan)
    with workspace.lock:
        job = deepcopy(workspace.state['jobs'].get(uid, {}))
    if job.get('status') != 'completed':
        return None
    result, directory = job['result'], Path(job['directory'])
    if result.get('source_sha256') != source['source_sha256']:
        raise ValueError('Retained trial belongs to another prepared revision')
    view = _read_artifact(directory, result, 'final_view')
    if all(h.get('fit_range_convention') == NATIVE_FIT_CONVENTION for h in view['histograms']):
        return None
    final_source = dict(source, source_path=str(directory/result['final_gpx']),
                        source_sha256=result['output_sha256'])
    inspected, view = _native_observation_view(workspace, final_source)
    if observation_contract(view, source['metadata']['physical_banks']) != contract:
        raise ValueError('Retained output does not match the reviewed observations and profile')
    return dict(job_id=uid, output_sha256=result['output_sha256'], directory=inspected['inspection'],
                response_sha256=file_sha(Path(inspected['inspection'])/'response.json'))


def _read_retained_report(reference, uid, result):
    directory = Path(reference['directory'])
    response_path = directory/'response.json'
    if (reference['job_id'] != uid or reference['output_sha256'] != result['output_sha256']
            or file_sha(response_path) != reference['response_sha256']):
        raise ValueError('The reviewed re-export no longer matches the verified trial')
    response = json.loads(response_path.read_text())
    if response.get('status') != 'inspected' or response.get('source_sha256') != result['output_sha256']:
        raise ValueError('The re-export belongs to another final GPX')
    return _read_artifact(directory, response, 'view')


def prepare_portfolio(workspace, entries, *, note, close_fit_band=0.1,
                      maximum_wss_per_point=None, refinement_policy=None,
                      maximum_last_cycle_shift_over_sigma=0.1, li_profile=None):
    """Preflight a closed set; preparing never enqueues or runs a refinement."""
    if workspace.runtime.kind != 'local':
        raise ValueError('Battery portfolios currently require the local scientific worker')
    note = _label(note, 'portfolio review note', maximum=2000)
    refinement_policy = validate_policy(refinement_policy)
    li_profile = validate_profile(li_profile)
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_TRIALS:
        raise ValueError(f'Declare one to {MAX_TRIALS} finite family/start pairs')
    for value in (close_fit_band, maximum_wss_per_point, maximum_last_cycle_shift_over_sigma):
        if value is not None and (type(value) not in (int, float) or
                                 not math.isfinite(value) or value < 0):
            raise ValueError('Diagnostic tolerances must be finite and nonnegative')
    if close_fit_band is None:
        raise ValueError('Declare the close-fit screening band')
    if maximum_last_cycle_shift_over_sigma is None:
        raise ValueError('Declare the final-cycle shift/sigma ceiling')
    identities, pairs, scientific_starts, family_topologies = [], set(), set(), {}
    trials = []
    lithium_binding = None
    # Resolve and compare every input before spending native preflight work.
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {'model', 'family', 'seed'}:
            raise ValueError('Each trial needs a prepared model, family and start label')
        family, seed = _label(entry['family'], 'family label'), _label(entry['seed'], 'start label')
        if (family, seed) in pairs:
            raise ValueError('Family/start pairs must be unique')
        pairs.add((family, seed))
        result = verify_prepared(entry['model'])
        source = prepared_source(workspace, entry['model'], result)
        source = workspace.inspect(source['id'])
        source, view = (_native_observation_view(workspace, source, revision='held-li-v2') if li_profile
                        else _current_observation_view(workspace, source))
        spec = result['specification']
        contract = observation_contract(view, source['metadata']['physical_banks'])
        identity = dict(acquisition={k: spec['acquisition'][k] for k in IDENTITY},
                        reduction=spec['reduction_family'], observations=contract, fit_domain_schema=DOMAIN_SCHEMA,
                        reduction_evidence=sorted(record['sha256'] for record in spec.get('reduction_evidence', [])),
                        raw_sources=sorted({b['data']['sha256'] for b in spec['banks']}))
        if identities and identity != identities[0]:
            raise ValueError('Compare the same cell, exposure, reduction, banks, profiles, weights and fit limits')
        identities.append(identity)
        context = source['context']
        plan = (profile_recipe(context, source['metadata']['physical_banks'], refinement_policy, li_profile) if li_profile
                else policy_recipe(context, source['metadata']['physical_banks'], refinement_policy))
        # Ordinary family starts vary only released cells/scales/widths. A Li
        # grid additionally separates its held coordinate and mixing nuisance.
        fixed = fixed_hypothesis(context, source['metadata']['physical_banks'], refinement_policy)
        if li_profile:
            binding = digest(profile_binding(context, fixed, li_profile))
            if lithium_binding is not None and binding != lithium_binding:
                raise ValueError('Held-Li points must retain the same topology, fixed atoms, sample inputs, background model and solver policy; only the selected Li grid and released starts may differ')
            lithium_binding = binding
            topology = binding
        else:
            topology = digest([context['model_signature'], fixed])
        if family in family_topologies and family_topologies[family] != topology:
            raise ValueError('Starts in one family must retain topology and all fixed atomic/shape inputs')
        family_topologies[family] = topology
        values = starting_values(context)
        backgrounds = [dict(name=h['name'], background=h.get('background')) for h in context['histograms']]
        values_hash = digest([topology, values, backgrounds])
        if values_hash in scientific_starts:
            raise ValueError('Duplicate scientific starting values are not independent trials')
        scientific_starts.add(values_hash)
        trials.append(dict(id=digest([family, seed, source['id']])[:24], family=family,
                           seed=seed, source=source, starting_values=values,
                           starting_values_sha256=values_hash, starting_backgrounds=backgrounds, plan=plan))
        if li_profile:
            _, target, mixing, capacity = profile_target(context, li_profile)
            trials[-1]['li_occupancy'] = target['occupancy']
            trials[-1]['mixing_occupancy_bounds'] = [0., capacity] if mixing else None
    if li_profile and not 3 <= len({t['li_occupancy'] for t in trials}) <= 12:
        raise ValueError('A held-Li diagnostic needs three to twelve distinct prepared occupancy values within the 24-trial budget')
    for trial in trials:
        source, plan = trial['source'], trial['plan']
        directory = (workspace.root/'battery-preflights'/
                     digest([source['source_sha256'], plan_hash(plan)])/uuid.uuid4().hex)
        result = workspace.runtime.call('preview', source, directory, plan=plan,
                    dispatch_key=digest(['battery-preview', source['source_sha256'], plan_hash(plan)])[:24])
        if result['status'] != 'preview_passed':
            raise ValueError(result.get('error') or 'A portfolio trial failed native preflight')
        if li_profile:
            verify_profile_preview(source['context'], result['preview'], li_profile)
        trial['preview'] = result['preview']
        trial['independent_parameters'] = len(result['preview']['stages'][-1]['effective_vary'])
        report = _retained_result_report(workspace, source, plan, identities[0]['observations'])
        if report:
            trial['retained_result_report'] = report
    document = dict(schema=SCHEMA, observation=identities[0], note=note, trials=trials,
                    policy=dict(close_fit_band=close_fit_band,
                                maximum_wss_per_point=maximum_wss_per_point,
                                maximum_last_cycle_shift_over_sigma=maximum_last_cycle_shift_over_sigma),
                    recipe='battery_reviewed_policy', refinement_policy=refinement_policy,
                    scientific_assignment='not_established')
    if li_profile:
        document['li_profile'] = li_profile
        document['li_fixed_inputs_sha256'] = lithium_binding
    token = digest(document)
    path = workspace.root/'battery-portfolios'/(token+'.json')
    path.parent.mkdir(exist_ok=True)
    if path.exists() and json.loads(path.read_text()) != document:
        raise ValueError('The saved portfolio revision changed')
    write_json(path, document)
    with workspace.lock:
        workspace.state.setdefault('battery_portfolios', {}).setdefault(token,
            dict(token=token, path=str(path), status='ready', created_utc=utc(), jobs={}))
        workspace.save()
    return token, document


def load_portfolio(workspace, token):
    if not isinstance(token, str) or not re.fullmatch('[0-9a-f]{64}', token):
        raise ValueError('Choose a retained portfolio revision')
    record = workspace.state.get('battery_portfolios', {}).get(token)
    if not record:
        raise ValueError('Choose a retained, preflighted portfolio')
    path = workspace.root/'battery-portfolios'/(token+'.json')
    if path.is_symlink() or not path.resolve().is_relative_to(workspace.root) or path.stat().st_size>64*1024*1024:
        raise ValueError('The portfolio is outside its workspace or exceeds 64 MiB')
    document = json.loads(path.read_text())
    if digest(document) != token or document.get('schema') != SCHEMA:
        raise ValueError('The reviewed portfolio changed; prepare a new revision')
    return document


def start_portfolio(workspace, token, *, approver):
    """Approve the exact reviewed set atomically, then reuse ordinary job dispatch."""
    if workspace.runtime.kind != 'local':
        raise ValueError('Battery portfolios currently require the local scientific worker')
    approver = _label(approver, 'reviewer identity')
    document = load_portfolio(workspace, token)
    approved = []
    # Rehash every input before any recipe or queue mutation.
    for trial in document['trials']:
        source = workspace.runtime.resolve(trial['source'])
        approved.append(approve_plan(trial['plan'], source['context'], trial['preview'], approver=approver))
    with workspace.lock:
        record = workspace.state['battery_portfolios'][token]
        if record['status'] == 'started':
            return deepcopy(record)
        for trial in document['trials']:
            current = workspace.state['sources'].get(trial['source']['id'])
            if not current or current.get('source_sha256') != trial['source']['source_sha256']:
                raise ValueError('A selected model revision changed after review')
        # _enqueue and run_next retain the existing attempts/checkpoint contract.
        before = deepcopy(workspace.state)
        try:
            for trial, approval in zip(document['trials'], approved):
                source = trial['source']
                recipe_id = digest([token, trial['id']])[:24]
                approval['execution_policy_sha256'] = token
                recipe = dict(id=recipe_id, experiment=source['experiment'],
                    version=1+sum(r['experiment']==source['experiment'] for r in workspace.state['recipes'].values()),
                    plan=trial['plan'], approval=approval, context=source['context'], source_id=source['id'],
                    scope=RefinementScope(experiment=source['experiment'], mode='scans', scans=[source['scan_key']]).to_dict(),
                    selection_token=token, existing=True, future=False, approved_utc=utc(),
                    armed=False, started=True, baseline=[source], generation=token,
                    battery_portfolio=token, battery_trial=trial['id'])
                workspace.state['recipes'][recipe_id] = recipe
                record['jobs'][trial['id']] = workspace._enqueue(recipe, source)['id']
            record.update(status='started', approved_utc=utc(), approver=approver)
            workspace.state['paused'] = False
            workspace.save()
        except Exception:
            workspace.state = before
            raise
        return deepcopy(record)


def portfolio_summary(workspace, token):
    document = load_portfolio(workspace, token)
    record = workspace.state['battery_portfolios'][token]
    schema = document['observation'].get('fit_domain_schema', LEGACY_DOMAIN_SCHEMA)
    native_interval = schema == DOMAIN_SCHEMA
    rows, incomplete = [], []
    for trial in document['trials']:
        uid = record['jobs'].get(trial['id'])
        with workspace.lock:
            job = deepcopy(workspace.state['jobs'].get(uid, {}))
        row = dict(id=trial['id'], family=trial['family'], seed=trial['seed'],
                   job_id=uid, status=job.get('status', 'not_started'),
                   execution_status=job.get('status', 'not_started'),
                   parameters=trial['independent_parameters'], banks=[],
                   wss_per_point=None, point_count=None, message=job.get('message', ''))
        if document.get('li_profile'):
            row['li_occupancy'] = trial['li_occupancy']
        if row['status']=='needs_review':
            row['message']=sample_width_failure_message(job.get('result') or {},trial['source']) or row['message']
        if row['status'] == 'completed':
            try:
                result = job['result']; directory = Path(job['directory'])
                if result['source_sha256'] != trial['source']['source_sha256']:
                    raise ValueError('Trial output belongs to another input revision')
                view = _read_artifact(directory, result, 'final_view')
                if trial.get('retained_result_report'):
                    view = _read_retained_report(trial['retained_result_report'], uid, result)
                    row['message'] = ('Reused verified fit; residuals re-exported for this review with the corrected native interval. '
                                      'Refinement was not repeated; its original report remains unchanged.')
                row['convergence_evidence'] = view.get('convergence')
                row['convergence'] = assess_convergence(view.get('convergence'),
                    document['policy'].get('maximum_last_cycle_shift_over_sigma'))
                mapping = trial['source']['metadata']['physical_banks']
                if any((h.get('fit_range_convention') == NATIVE_FIT_CONVENTION) != native_interval
                       for h in view['histograms']):
                    raise ComparisonConventionChanged('This result and its review use different fitted-point reporting conventions. '
                        'Create a new reviewed comparison with the corrected native interval; retained outputs remain available.')
                if observation_contract(view, mapping, domain_schema=schema) != document['observation']['observations']:
                    raise ValueError('The fitted observation or profile changed')
                if document.get('li_profile'):
                    row['li_nuisance'] = verify_profile_result(view, document['li_profile'], trial['li_occupancy'], trial['mixing_occupancy_bounds'])
                for h in view['histograms']:
                    stats = h['fit_statistics']; n, sse = stats['point_count'], stats['weighted_residual_sum']
                    if type(n) is not int or n <= 0 or type(sse) not in (int, float) or not math.isfinite(sse) or sse < 0:
                        raise ValueError('A completed trial lacks finite weighted residual statistics')
                    row['banks'].append(dict(bank=mapping[h['name']], n=n, wss=sse, rwp=h.get('rwp'), wss_per_point=sse/n))
                row['point_count'] = sum(b['n'] for b in row['banks'])
                row['wss_per_point'] = sum(b['wss'] for b in row['banks'])/row['point_count']
                maximum = document['policy']['maximum_wss_per_point']
                row['adequacy'] = ('not_evaluated' if maximum is None else
                    'threshold_exceeded' if any(b['wss_per_point'] > maximum for b in row['banks']) else
                    'within_screening_threshold')
            except (OSError, ValueError, KeyError, TypeError) as exc:
                row.update(status='comparison_unavailable' if isinstance(exc, ComparisonConventionChanged) else 'integrity_failure',
                    message=str(exc), wss_per_point=None, banks=[], point_count=None)
        if row['status'] != 'completed':
            incomplete.append(row['id'])
        rows.append(row)
    families = []
    for family, occupancy in dict.fromkeys((r['family'], r.get('li_occupancy')) for r in rows):
        members = [r for r in rows if r['family'] == family and r.get('li_occupancy') == occupancy]
        values = [r['wss_per_point'] for r in members if r['wss_per_point'] is not None]
        spread = max(values)-min(values) if values else None
        families.append(dict(family=family if occupancy is None else f'{family} · held Li {occupancy:g}', declared_starts=len(members), completed_starts=len(values),
            minimum_wss_per_point=min(values) if values else None, start_spread=spread,
            start_check='incomplete' if len(values)!=len(members) else 'single_start_only' if len(values)==1 else
                'convergence_unresolved' if any(r.get('convergence',{}).get('status')!='passed' for r in members) else
                'start_sensitive' if spread > document['policy']['close_fit_band'] else 'within_screening_band'))
    status = ('awaiting_trials' if record['status']=='ready' else
              'running' if any(r['status'] in {'queued','running','recovering'} for r in rows) else
              'paused' if any(r['status']=='paused' for r in rows) else 'incomplete' if incomplete else
              'optimization_unresolved' if any(r.get('convergence',{}).get('status')!='passed' for r in rows) or
                  any(f['start_check']=='start_sensitive' for f in families) else
              'no_trial_passes_screen' if all(r.get('adequacy')=='threshold_exceeded' for r in rows) else
              'reviewable_unqualified')
    summary = dict(schema='radar-battery-portfolio-summary/v1', token=token, status=status,
                fit_domain_schema=schema,
                metric_convention=('Native fitted interval, including the first bin at or above the upper limit.' if native_interval else
                    'Legacy coordinate interval; the first bin above an upper limit was omitted from reported residuals. '
                    'Retained values are historical. Prepare a new comparison to use the corrected native interval.'),
                declared_count=len(rows), completed_count=sum(r['status']=='completed' for r in rows),
                incomplete=incomplete, trials=rows, families=families,
                scientific_assignment='not_established', composition_status='not_evaluated',
                interpretation='Residual comparisons are conditional on the declared models, fixed inputs and calibration. '
                               'Extra parameters can lower residuals. Screening bands are not confidence intervals or model probabilities.')
    if document.get('li_profile'):
        summary['li_profile'] = summarize_profile(document['li_profile'], rows, document['policy']['close_fit_band'])
        summary['composition_status'] = 'not_identified'
    return summary


def pause_portfolio(workspace, token, paused=True):
    """Pause only this finite set's pending jobs; an active stage can finish."""
    if type(paused) is not bool:
        raise ValueError('Choose a boolean pause or resume action')
    load_portfolio(workspace, token)
    with workspace.lock:
        for uid in workspace.state['battery_portfolios'][token]['jobs'].values():
            job=workspace.state['jobs'][uid]
            if job['status']==('queued' if paused else 'paused'):
                job['status']='paused' if paused else 'queued'
        workspace.save()
