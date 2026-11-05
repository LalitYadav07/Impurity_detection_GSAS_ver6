"""One checked profile adjustment before residual/candidate comparisons.

Uses the same native active-set, covariance, convergence and rollback contract
as reviewed recipes. The selected profile is held for every phase candidate.
"""
from pathlib import Path
import json
import uuid


def adjust_main_profile(refiner, config, directory):
    policy = config.get('tof_profile') or {}
    if policy.get('mode', 'fixed') == 'fixed':
        return None
    if policy.get('mode') != 'per_scan' or not policy.get('terms'):
        raise ValueError('TOF profile treatment requires fixed or explicit per_scan terms.')
    from gpx_recipe.snapshot import snapshot, gsas
    from gpx_recipe.contracts import plan_hash
    from gpx_recipe.execution import run_plan
    root = Path(directory) / ('tof-profile-' + uuid.uuid4().hex[:12])
    root.mkdir(parents=True)
    source, output = root/'supplied-profile.gpx', root/'adjusted-profile.gpx'
    old_filename = refiner.project.filename
    refiner.project.save(str(source))
    refiner.project.filename = old_filename
    context = snapshot(source)
    h = next(h for h in context['histograms'] if h['name'] == refiner.histogram.name)
    plan = dict(schema='radar-gpx-recipe/v1', model_signature=context['model_signature'],
        source_sha256=context['source']['sha256'], request='Explicit setup choice: empirical TOF width adjustment before all phase comparisons.',
        assumptions=['The supplied main phase anchors the width. Position calibration and sample widths remain fixed.'],
        questions=[], unsupported=[], failure_policy='stop', stages=[dict(id='profile',
            title='Empirical TOF peak-profile adjustment', cycles=20, constraints=[],
            convergence={'max_shift_over_sigma': 1.0},
            free=[dict(kind='instrument_profile', histogram=h['key'], parameter=name) for name in policy['terms']])])
    report = {'requested': policy, 'status': 'rejected', 'profile_reused_for_candidates': False,
              'source_profile': 'supplied parameters remain preserved in supplied-profile.gpx'}
    try:
        result = run_plan(source, output, plan, approved={
            'approved_plan_sha256': plan_hash(plan), 'model_signature': context['model_signature']})
        if not all(s['status'] == 'completed' for s in result['stages']):
            raise ValueError('Profile stage did not produce a qualified checkpoint.')
        project = gsas().G2Project(gpxfile=str(output))
        profile = root/'empirical-adjusted.instprm'
        project.histogram(h['name']).SaveProfile(str(profile))
        # Apply only the audited profile to the existing main fit. Its following
        # ordinary staged fit recalculates the nuclear baseline with held widths.
        refiner.load_instrument_profile(str(profile))
        report.update(status='adopted', profile_reused_for_candidates=True,
                      exported_instprm=str(profile), audit=result['stages'][0].get('tof_profile_audit'))
    except ValueError as exc:
        report['reason'] = str(exc)
    report['audit_path'] = str(root/'profile-treatment.json')
    Path(report['audit_path']).write_text(json.dumps(report, indent=2)+'\n')
    return report
