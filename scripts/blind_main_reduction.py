"""One independent, checked alternative when a blind two-phase anchor vanishes.

The collapsed trial is evidence to propose another hypothesis, never an accepted
checkpoint. Supplied main phases and models with more than one survivor are
outside this deliberately narrow policy.
"""
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

# GSASIIstrMath.py and GSASIIstrIO.py clamp phase HAP Scale with
# max(1.e-12, value). This is only a hypothesis trigger, not an acceptance floor.
NATIVE_PHASE_SCALE_FLOOR = 1e-12


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def collapse_evidence(project, histogram, main, parsed, *, label):
    """Require real native scales and every parsed row; never infer missing=0."""
    phases = list(project.phases())
    names = [phase.name for phase in phases]
    if len(names) != 2 or main not in names or set(parsed) != set(names):
        return None
    if project.data.get('Covariance', {}).get('data', {}).get('freshCOV') is not True:
        return None
    rows = {}
    for phase in phases:
        hap = phase.data.get('Histograms', {}).get(histogram)
        if not isinstance(hap, dict) or hap.get('Use') is not True:
            return None
        scale = hap.get('Scale')
        if not isinstance(scale, (list, tuple)) or not scale or not _finite(scale[0]):
            return None
        values = parsed.get(phase.name)
        if not isinstance(values, dict) or any(not _finite(values.get(k)) for k in ('phase_fraction_pct', 'weight_fraction_pct')):
            return None
        rows[phase.name] = dict(hap_scale=float(scale[0]),
                                phase_fraction_pct=float(values['phase_fraction_pct']),
                                weight_fraction_pct=float(values['weight_fraction_pct']))
    other = next(name for name in names if name != main)
    if not (rows[main]['hap_scale'] <= NATIVE_PHASE_SCALE_FLOOR and rows[main]['weight_fraction_pct'] <= 0
            and rows[other]['hap_scale'] > 0 and rows[other]['weight_fraction_pct'] > 0):
        return None
    return dict(schema='radar-blind-main-collapse/v1', main=main, survivor=other,
                histogram=histogram, rows=rows, label=label,
                basis='complete native HAP scales and complete current refinement listing',
                native_covariance_fresh=True, proposal_only=True)


def _valid_evidence(value, main):
    if (not isinstance(value, dict) or value.get('schema')!='radar-blind-main-collapse/v1'
            or value.get('main')!=main or value.get('native_covariance_fresh') is not True
            or value.get('proposal_only') is not True):
        return False
    other=value.get('survivor');rows=value.get('rows')
    if not other or other==main or not isinstance(rows,dict) or set(rows)!={main,other}:
        return False
    if any(not isinstance(row,dict) or any(not _finite(row.get(k)) for k in
           ('hap_scale','phase_fraction_pct','weight_fraction_pct')) for row in rows.values()):
        return False
    return (rows[main]['hap_scale']<=NATIVE_PHASE_SCALE_FLOOR and rows[main]['weight_fraction_pct']<=0
            and rows[other]['hap_scale']>0 and rows[other]['weight_fraction_pct']>0)


def _observation_state(project):
    from gpx_recipe.battery.preparation import array_fingerprints
    from gpx_recipe.snapshot import json_safe
    return {h.name: dict(arrays=array_fingerprints(h), limits=json_safe(h.data['Limits']),
                         instrument=json_safe(h.data['Instrument Parameters']))
            for h in project.histograms()}


def _cell(phase):
    return tuple(float(value) for value in phase.data['General']['Cell'][1:7])


def _require_profile_domain(project):
    from gpx_recipe.tof_domain import assess_tof_domain
    for histogram in project.histograms():
        instrument=histogram.data['Instrument Parameters'][0]
        if 'T' in str(instrument.get('Type', [''])[0]):
            limits=list(histogram.data['Limits'][1])
            domain=assess_tof_domain(instrument, limits)
            if domain['trimmed_lower'] or domain['native_limits']!=limits:
                raise ValueError('Reduced-model profile would require a changed fitted domain')


def promotion_values(receipt, main, accepted, labels):
    """Prepare all final-role bookkeeping without partially mutating its input."""
    after=receipt.get('main_after')
    if (receipt.get('adopted') is not True or receipt.get('removed_phase')!=main
            or after not in accepted or set(receipt.get('fractions',{}))!={after}):
        raise ValueError('Reduced-model promotion does not match the retained phase set')
    return (receipt['output_gpx'], receipt['rwp'], receipt['fractions'], after,
            receipt['main_cif'], [p for p in accepted if p not in {main, after}],
            {key:value for key,value in labels.items() if key!=main})


def _fit_single_model(source, output, directory, *, cycles=20):
    """Native internal fit, using the same active-set/physics checks as recipes.

    This is not a user recipe and does not manufacture a user approval receipt.
    Both deterministic stages are bounded; no model or parameter search occurs.
    """
    from gpx_recipe.snapshot import snapshot, gsas
    from gpx_recipe.execution import configure, check_preflight
    from gpx_recipe.studio_state import extract_gsasii_refinement_state
    from gpx_recipe.convergence import convergence_evidence, require_convergence
    from gpx_recipe.effects import (parameter_changes, require_fixed_values_unchanged,
                                    require_native_covariance, blocking_frozen_variables,
                                    constraint_bound_audit, require_constraint_bounds,
                                    native_storage_audit, require_native_storage)
    from gpx_recipe import advanced
    import importlib
    import shutil
    from gsas_main_phase_refiner import candidate_cell_rejection

    G2 = gsas()
    context = snapshot(source)
    if len(context['phases']) != 1 or len(context['histograms']) != 1:
        raise ValueError('Reduced-model fit requires exactly one phase and histogram')
    if 'Hessian' not in str(context.get('controls', {}).get('deriv type', '')):
        raise ValueError('Reduced-model cycle budget requires the existing native Hessian minimizer')
    project = G2.G2Project(gpxfile=str(source))
    before_domain = _observation_state(project)
    initial_cell = _cell(project.phases()[0])
    _require_profile_domain(project)
    phase_key, hist_key = context['phases'][0]['key'], context['histograms'][0]['key']
    project.save(str(output))
    events = []
    def run_stage(name, with_cell):
        free = [dict(kind='histogram_scale', histogram=hist_key), dict(kind='background', histogram=hist_key)]
        if with_cell:
            free.append(dict(kind='cell', phase=phase_key))
        stage = dict(id=name, title=name, cycles=cycles, free=free, constraints=[],
                     background=None, phase_normalization=[] if with_cell else [hist_key], fit_ranges=[],
                     exclusions=[], bounds=[], texture=[], convergence={'max_shift_over_sigma': 1.0})
        configure(project, stage, context, set())
        project.save()
        before = extract_gsasii_refinement_state(output, project, G2, max_correlations=10)
        check_preflight(before, stage, context)
        project.data.setdefault('Covariance', {})['data'] = {}
        project.save()
        project.do_refinements([{}])
        project.save()
        covariance = project.data.get('Covariance', {}).get('data', {})
        if covariance.get('freshCOV') is not True:
            raise ValueError('Reduced-model stage lacks fresh native covariance')
        native_maps = importlib.import_module((G2.__package__+'.' if G2.__package__ else '')+'GSASIImapvars')
        bounds = constraint_bound_audit(before, covariance, native_maps.Dict2Map)
        after = extract_gsasii_refinement_state(output, project, G2, max_correlations=10)
        if set(after['parameter_space']['last_vary']) != set(before['parameter_space']['effective_vary']):
            raise ValueError('Reduced-model native vary list differs from its checked active set')
        require_native_covariance(covariance, len(after['parameter_space']['last_vary']), after['covariance'])
        require_constraint_bounds(bounds)
        require_native_storage(native_storage_audit(bounds, after))
        require_fixed_values_unchanged(parameter_changes(before, after))
        if blocking_frozen_variables(before, after):
            raise ValueError('Reduced-model stage froze a requested variable')
        advanced.check_site_totals(project, context)
        evidence = convergence_evidence(after)
        require_convergence(evidence, stage['convergence'])
        if _observation_state(project) != before_domain:
            raise ValueError('Reduced-model fit changed observations, fit domain or instrument profile')
        histogram = project.histograms()[0]
        _require_profile_domain(project)
        rwp = float(histogram.get_wR())
        if not math.isfinite(rwp) or rwp < 0:
            raise ValueError('Reduced-model stage has an invalid residual')
        if candidate_cell_rejection(initial_cell, _cell(project.phases()[0]), 0.0, 1.0):
            raise ValueError('Reduced-model stage exceeded the existing cell-change guard')
        return dict(stage=name, status='completed', rwp=rwp, convergence=evidence,
                    actual_vary=after['parameter_space']['last_vary'], cell_refined=with_cell)

    checkpoint=directory/'scale_background_verified.gpx'
    for name, with_cell in (('scale_background', False), ('joint_cell', True)):
        try:
            event=run_stage(name,with_cell)
            if with_cell and event['rwp']>events[0]['rwp']+1e-6:
                raise ValueError('Cell trial worsened the verified scale/background fit')
            events.append(event)
            if not with_cell:
                shutil.copy2(output,checkpoint)
                shutil.copy2(Path(output).with_suffix('.lst'),checkpoint.with_suffix('.lst'))
        except ValueError as exc:
            if not with_cell or not checkpoint.is_file():
                raise
            # Only the completed, converged first stage is eligible for this
            # rollback. The failed cell values and covariance never survive.
            shutil.copy2(output,directory/'cell_trial.rejected.gpx')
            listing=Path(output).with_suffix('.lst')
            if listing.exists():
                shutil.copy2(listing,directory/'cell_trial.rejected.lst')
            shutil.copy2(checkpoint,output)
            shutil.copy2(checkpoint.with_suffix('.lst'),listing)
            if sha256(Path(output).read_bytes()).digest()!=sha256(checkpoint.read_bytes()).digest():
                raise RuntimeError('Reduced-model rollback failed its checkpoint checksum')
            project=G2.G2Project(gpxfile=str(output))
            if _observation_state(project)!=before_domain:
                raise RuntimeError('Reduced-model rollback changed native observations or profile')
            events.append(dict(stage=name,status='rejected',error=str(exc),
                               rollback='verified_scale_background',cell_refined=False))
    retained=next(row for row in reversed(events) if row['status']=='completed')
    retained['retained']=True
    (directory/'native-fit.json').write_text(json.dumps(events, indent=2)+'\n')
    return project, events


def try_reduced_main(base_gpx, output_dir, main, evidence, *, user_supplied_main):
    """Keep the caller's baseline untouched; return a promotion only if qualified."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory/'reduced-main.json'
    receipt = dict(schema='radar-blind-main-reduction/v1', attempted=False, adopted=False,
                   main_before=main, user_supplied_main=bool(user_supplied_main),
                   scope='one two-phase-to-single-phase hypothesis; no search or relaxed guard')
    def finish(reason, **values):
        receipt.update(reason=reason, **values)
        receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
        return receipt
    if user_supplied_main:
        return finish('user_supplied_main_protected')
    valid = [e for e in evidence if _valid_evidence(e, main)]
    if not valid:
        return finish('no_authoritative_zero_scale_proposal')
    if receipt_path.exists():
        # Resume/retry must keep the original checkpoint, not fail its run or
        # re-execute a prior hypothesis attempt. Preserve the original receipt.
        return dict(receipt, reason='prior_attempt_not_repeated', prior_receipt=str(receipt_path))
    source = Path(base_gpx)
    source_hash = sha256(source.read_bytes()).hexdigest()
    receipt.update(attempted=True, source_sha256=source_hash, collapse_evidence=deepcopy(valid))
    # Persist before doing any new native calculation. Failures retain this
    # evidence and the caller continues with its original accepted checkpoint.
    finish('attempt_started')
    try:
        from gpx_recipe.snapshot import snapshot, gsas
        from gpx_recipe.model_edits import edit_model
        from gsas_main_phase_refiner import parse_gsas_lst, require_phase_fractions, candidate_cell_rejection
        G2 = gsas()
        baseline = G2.G2Project(gpxfile=str(source))
        phases = list(baseline.phases())
        if len(phases)!=2 or len(baseline.histograms())!=1 or main not in {p.name for p in phases}:
            raise ValueError('Only a two-phase, single-histogram baseline is supported')
        survivor = next(p for p in phases if p.name != main)
        if any(e['survivor'] != survivor.name for e in valid):
            raise ValueError('Collapse evidence belongs to another survivor')
        before_cell = _cell(survivor)
        baseline_rwp = float(baseline.histograms()[0].get_wR())
        if not math.isfinite(baseline_rwp) or baseline_rwp < 0:
            raise ValueError('Baseline residual is unavailable')
        context = snapshot(source)
        target = next(p['key'] for p in context['phases'] if p['name']==main)
        edit = dict(schema='radar-phase-correction/v1', source_sha256=source_hash,
                    model_signature=context['model_signature'], operation='remove', phase=target,
                    reason='Independent blind-main zero-scale hypothesis; original model remains the fallback')
        edited = directory/'single_model.gpx'
        _, edit_receipt = edit_model(source, edited, edit)
        fitted = directory/'single_model_fitted.gpx'
        project, stages = _fit_single_model(edited, fitted, directory)
        if [p.name for p in project.phases()] != [survivor.name]:
            raise ValueError('Independent fit changed surviving phase identity')
        histogram = project.histograms()[0]
        phase = project.phases()[0]
        if candidate_cell_rejection(before_cell, _cell(phase), 0.0, 1.0):
            raise ValueError('Independent fit exceeded the existing cell-change guard')
        scale = phase.data['Histograms'][histogram.name]['Scale'][0]
        if not _finite(scale) or scale <= 0:
            raise ValueError('Independent fit has no positive surviving native phase scale')
        fractions = parse_gsas_lst(fitted.with_suffix('.lst'), histogram.name)
        require_phase_fractions(fractions, [phase.name], listing=fitted.with_suffix('.lst'))
        if abs(float(fractions[phase.name]['weight_fraction_pct'])-100.0) > 1e-3:
            raise ValueError('Independent single-phase fraction is not normalized')
        rwp = float(histogram.get_wR())
        if not math.isfinite(rwp) or rwp > baseline_rwp + 1e-6:
            raise ValueError('Independent reduced model worsened the same-domain residual')
        if _observation_state(project) != _observation_state(baseline):
            raise ValueError('Independent model differs from the baseline observation/profile/domain')
        promoted_cif = directory/'promoted_main.cif'
        phase.export_CIF(str(promoted_cif))
        if sha256(source.read_bytes()).hexdigest() != source_hash:
            raise RuntimeError('Reduced-main processing modified its baseline')
        return finish('independent_native_model_passed', adopted=True, removed_phase=main,
                      main_after=phase.name, output_gpx=str(fitted), main_cif=str(promoted_cif),
                      output_sha256=sha256(fitted.read_bytes()).hexdigest(), baseline_rwp=baseline_rwp,
                      rwp=rwp, fractions=fractions, native_stages=stages, edit=edit_receipt,
                      retained_stage=next(row['stage'] for row in stages if row.get('retained')),
                      cell_refined=bool(phase.data['General']['Cell'][0]))
    except Exception as exc:
        if sha256(source.read_bytes()).hexdigest() != source_hash:
            raise RuntimeError('Reduced-main processing modified its baseline') from exc
        return finish('independent_native_model_rejected', error_type=type(exc).__name__, error=str(exc))
