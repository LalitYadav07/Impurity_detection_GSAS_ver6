"""Read native campaign evidence without loading GSAS-II in the UI process."""
from __future__ import annotations

import json
import copy
import time
import threading
from collections import OrderedDict
from pathlib import Path
from functools import lru_cache

import numpy as np
import hashlib

from gpx_recipe.symmetry.campaign import Campaign,verify_receipt
from gpx_recipe.symmetry.compiler import confined,assets,job_inputs
from gpx_recipe.symmetry.contracts import digest
from .model_portfolio import portfolio_snapshot

_verified_receipts=OrderedDict()
_evidence_lock=threading.RLock()


def structural_timing_summary(value):
    """Keep missing/older timing receipts distinct from zero-duration work."""
    if (not isinstance(value,dict) or value.get('schema')!='radar-structural-service-timing/v1'
            or type(value.get('scope')) is not str or value['scope'] not in {'model','acquisition'}
            or value.get('queue_startup_transfer_included') is not False):return ''
    values=[value.get(k) for k in ('preparation_seconds','fit_service_seconds','combined_service_seconds')]
    if any(type(v) not in (int,float) or v<0 for v in values):return ''
    try:values=[float(v) for v in values]
    except OverflowError:return ''
    if not np.isfinite(values).all():return ''
    preparation,fitting,total=values
    if not np.isclose(preparation+fitting,total,rtol=1e-8,atol=1e-8):return ''
    return (f'{preparation:.1f} s preparation · {fitting:.1f} s fitting and GPX checks · '
            f'{total:.1f} s combined service time. Other worker tasks, queue, startup and transfer are additional.')


def candidate_display_name(catalog,candidate):
    """Use the authoritative constructed-child label, including its subgroup."""
    if candidate.get('magnetic_space_group'):return candidate['magnetic_space_group']
    # Older prepared catalogs kept the full construction catalog in provenance
    # but replaced its visible labels with only the irrep name. Recover a label
    # only for the same sealed child, modes and embedding, never by ID alone.
    prepared=(catalog.get('provenance') or {}).get('prepared_library') or {}
    original=prepared.get('original_catalog')
    if isinstance(original,dict) and digest(original)==prepared.get('original_catalog_sha256'):
        from gpx_recipe.preparation.structural_hints import candidate_identity
        try:
            matches=[c for c in original.get('candidates',[])
                     if candidate_identity(c)==candidate_identity(candidate)]
        except (KeyError,TypeError,ValueError):matches=[]
        if len(matches)==1:
            for row in (original.get('coverage') or {}).get('candidates',[]):
                if row.get('id')==matches[0]['id'] and row.get('label'):
                    return str(row['label']).split(', basis=',1)[0]
    for row in (catalog.get('coverage') or {}).get('candidates',[]):
        if row.get('id')==candidate['id'] and row.get('label'):
            return str(row['label']).split(', basis=',1)[0]
    return candidate.get('irrep') or candidate['id']


def structural_result_names(root,protocol,protocol_sha):
    """Resolve display labels once per protocol from its sealed input catalog."""
    if protocol.get('kind')!='structural':return {}
    try:
        path=confined(Path(root)/'inputs'/protocol_sha,protocol['catalog']['path'])
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=protocol['catalog']['sha256']:return {}
        catalog=json.loads(raw)
        return {candidate['id']:candidate_display_name(catalog,candidate) for candidate in protocol['candidates']}
    except (OSError,ValueError,KeyError,TypeError):return {}


def structural_discovery_feedback(catalog):
    """Keep failed construction visible without offering an empty fit plan."""
    coverage=catalog.get('coverage') or {}
    rows=[dict(id=str(row.get('id','')),model=str(row.get('label','')).split(', basis=',1)[0],
               state=str(row.get('state','unknown')).replace('_',' '),reason=str(row.get('reason','')))
          for row in coverage.get('candidates',[])]
    ready=len(catalog.get('candidates') or [])
    discovered=coverage.get('discovered_count',len(rows) or ready)
    failed=sum(row['state']=='construction failed' for row in rows)
    deferred=sum(row['state']=='deferred' for row in rows)
    summary=f'{discovered} OPDs discovered · {ready} ready to refine · {failed} construction failures · {deferred} deferred'
    for index,failure in enumerate(coverage.get('irrep_failures') or []):
        rows.append(dict(id=f'irrep-failure-{index+1}',model='Irrep discovery',state='discovery failed',reason=str(failure)))
    error='' if ready else 'No structural models passed native preparation. Review construction details below; no refinement plan is available.'
    return summary,rows,error


def clear_evidence_cache():
    with _evidence_lock:
        _verified_receipts.clear();cached_view.cache_clear();comparison_profile.cache_clear()


def checked_receipt(directory,job,expected_receipt=None):
    """Rehash changed evidence immediately; bound unchanged hash reuse to 5 min.

    Polling hundreds of completed starts must not reread every common GPX and
    output on each UI refresh. Inode/ctime/mtime/size changes invalidate reuse.
    A new viewer always starts with a full cryptographic verification.
    """
    with _evidence_lock:return _checked_receipt(directory,job,expected_receipt)


def _checked_receipt(directory,job,expected_receipt):
    key=str(directory);cached=_verified_receipts.get(key)
    if cached and cached['expected']==expected_receipt and time.monotonic()-cached['at']<300:
        if Campaign._artifact_stamps(directory,cached['paths'])==cached['stamps']:
            _verified_receipts.move_to_end(key);return cached['result']
    raw=json.loads((directory/'result.json').read_text())
    if expected_receipt and digest(raw)!=digest(json.loads(expected_receipt)):raise ValueError('A terminal receipt changed after collection')
    paths=[(directory,p) for p in set(['job.json','result.json',*raw.get('files',{})])]
    if raw.get('status')=='profile_equivalent':
        from gpx_recipe.symmetry.group_receipts import reference_paths
        paths.extend(reference_paths(directory,raw))
    paths.extend((job_inputs(directory,job),a['path']) for a in assets(job['protocol']))
    before=Campaign._artifact_stamps(directory,paths)
    result=verify_receipt(directory,dict(id=job['id'],protocol_sha256=job['protocol_sha256']))
    if expected_receipt and digest(result)!=digest(json.loads(expected_receipt)):raise ValueError('A terminal receipt changed after collection')
    after=Campaign._artifact_stamps(directory,paths)
    if before!=after:raise ValueError('Evidence changed during verification')
    _verified_receipts[key]=dict(expected=expected_receipt,at=time.monotonic(),paths=paths,stamps=after,result=result)
    _verified_receipts.move_to_end(key)
    # Common canted campaigns contain 600--840 starts per acquisition and
    # several acquisitions. A 512-entry LRU thrashes on every ordered refresh.
    # Keep this bounded; changed artifact stamps and the 5-minute TTL still
    # force the same full cryptographic verification.
    while len(_verified_receipts)>4096:_verified_receipts.popitem(last=False)
    return result


def _trial_identity(directory,job,metadata=None):
    p=job['protocol']
    candidate=next(c for c in p['candidates'] if c['id']==job['candidate_id'])
    return dict(id=job['id'],protocol=job['protocol_sha256'],candidate=job['candidate_id'],seed=job['seed_id'],directory=str(directory),
        name=candidate['magnetic_space_group'] or candidate['irrep'] or candidate['id'],bns=candidate.get('bns_number'),
        embedding=dict(basis=candidate['transformation_matrix'],origin=candidate['origin_shift']),kind=p['kind'],source=p.get('acquisition_id') or p['source']['sha256'],source_sha256=p['source']['sha256'],sites=p.get('sites',[]),
        magnetic_function_sha256=p.get('magnetic_function_sha256'),
        metadata=metadata or {},parent=p['parent_phase'],histogram=p['histogram'],k=p['k'],worker_revision=job.get('package_sha256',''),status='pending',rwp=None,nvars=None,reason='',has_view=False,has_gpx=False,
        adequate_rwp_percent=(p.get('policy') or {}).get('adequate_rwp_percent'),
        grouped_surrogate=bool(p.get('policy',{}).get('grouped_surrogate')))


def read_trial(directory,metadata=None,*,expected_receipt=None):
    directory=Path(directory).resolve();job=json.loads((directory/'job.json').read_text());p=job['protocol']
    row=_trial_identity(directory,job,metadata)
    if (directory/'result.json').is_file():
        try:
            result=checked_receipt(directory,job,expected_receipt)
            row.update({k:result.get(k) for k in ['status','rwp','nvars','reason']})
            row['refined_moments']=result.get('moments',[])
            row['resources']=result.get('resources',{})
            row['timings']=result.get('timings',{})
            row['structural_fit']=result.get('structural_fit',{})
            row['screening_only']=result.get('screening_only',False)
            row['native_reference']=(result.get('native_reference') or {}).get('id','')
            row['native_verified']=bool(result.get('effects_verified') and result.get('audit',{}).get('passed'))
            row['grouped_surrogate']=bool(p.get('policy',{}).get('grouped_surrogate'))
            row['receipt_sha256']=digest(result);row['native_version']=result.get('runtime',{}).get('version','unknown')
            row['bns_audited']=result.get('audit',{}).get('checks',{}).get('complete_magnetic_operators_preserved') is True
            gpx='results/'+str(result.get('gpx') or '')
            if gpx in result.get('files',{}):row.update(has_gpx=True,gpx_relative=gpx)
            views=[k for k in result.get('files',{}) if k.endswith('.view.json')]
            row['has_view']=len(views)==1
            if row['has_view']:row['view_path']=str(confined(directory,views[0]))
            if row['status']=='profile_equivalent':
                # The checked reference owns the one full plot/GPX. Alternatives
                # retain compact moment vectors instead of duplicating arrays.
                reference=confined(directory.parent,row['native_reference'])
                source=json.loads((reference/'result.json').read_text())
                row['native_reference_status']=source.get('status')
                linked_views=[k for k in source.get('files',{}) if k.endswith('.view.json')]
                if len(linked_views)!=1:raise ValueError('Verified representative has no unique view')
                row.update(has_view=True,view_path=str(confined(reference,linked_views[0])))
            native='results/'+str(result.get('native_receipt') or '')
            evidence={}
            if native in result.get('files',{}):
                evidence=json.loads(confined(directory,native).read_text())
                row['moment_phase_names']={key:evidence.get(field) for key,field in
                    [('parent','nuclear_phase_name'),('magnetic','magnetic_phase_name')]}
                row['baseline_rwp']=(evidence.get('baseline_residuals',{}).get(p['histogram']) or {}).get('wR')
                row['native_convergence']={key:(evidence.get('refinement_rvals') or {}).get(key)
                                           for key in ['converged','Aborted','SVD0']}
                row['residual_targets']={key:evidence.get('target_peak_audit',{}).get(key)
                                         for key in ['target_count','underfit_peak_count','weighted_residual_improvement']}
            fitted_output=row['status']=='valid' or row['has_view'] or row['has_gpx'] or bool(row['refined_moments'])
            if p.get('policy',{}).get('magnetic_optimizer')=='profiled_quadratic' and fitted_output and not row['screening_only']:
                row['optimizer']=validated_profiled_evidence(evidence,p['policy'],row['status'])
        except Exception as exc:
            row.update(status='integrity_failure',reason=str(exc),has_view=False,has_gpx=False,rwp=None,nvars=None)
            for key in ['receipt_sha256','view_path','gpx_relative','moment_phase_names','baseline_rwp','native_convergence','residual_targets','optimizer']:row.pop(key,None)
    elif expected_receipt:
        row.update(status='integrity_failure',reason='The collected terminal receipt is missing')
    return row


@lru_cache(maxsize=16)
def cached_view(path,mtime_ns,size,ctime_ns=None):
    return json.loads(Path(path).read_text())


def view_for(row):
    path=Path(row['view_path']);stat=path.stat();view=cached_view(str(path),stat.st_mtime_ns,stat.st_size,stat.st_ctime_ns)
    if row.get('status')=='profile_equivalent':
        view=copy.deepcopy(view)
        moments={m['label']:m for m in row.get('refined_moments',[])}
        for phase in view.get('phases',[]):
            for value in phase.get('moments',[]):
                own=moments.get(value['label'])
                if own:value.update(components=own['moment'],magnitude=own['magnitude'])
    return view


@lru_cache(maxsize=1024)
def comparison_profile(path,mtime_ns,size,ctime_ns):
    """Cache compact display comparisons, separately from large full views.

    This is reached only after receipt verification. The complete observed
    domain participates in the identity; calculated curves never merge scans
    or GPX revisions merely because their residuals happen to agree.
    """
    view=cached_view(path,mtime_ns,size,ctime_ns)
    mask=np.asarray(view['fitted'],dtype=bool)
    observed=np.asarray(view['observed'],dtype=float)[mask]
    calculated=np.asarray(view['calculated'],dtype=float)[mask]
    calculated.setflags(write=False)
    return (digest([view['x'],view['observed'],view['weights'],view['fitted']]),
            calculated,max(float(np.max(np.abs(observed))),1e-12))


def choose_group(groups,selected,*,explicit=False):
    """Prefer a qualified fit until a user chooses a particular trial.

    A trial can move from a pending group to its final pattern group. Preserve
    that explicit selection by trial identity rather than the old group ID.
    """
    if not groups:return ''
    if explicit:
        for group in groups:
            if any(row['id']==selected for row in group['members']):return group['id']
    for group in groups:
        if any(row.get('has_view') and row.get('status')=='valid' for row in group['members']):return group['id']
    return groups[0]['id']


def choose_member(members,selected,*,explicit=False):
    if explicit and any(row['id']==selected for row in members):return selected
    qualified=next((row for row in members if row.get('has_view') and row.get('status')=='valid'),None)
    native=next((row for row in members if row.get('has_view') and row.get('native_verified')),None)
    return (qualified or native or (members[0] if members else {})).get('id','')


def display_groups(groups):
    """Keep qualified choices visible without discarding other dispositions."""
    displayed=[]
    for group in groups:
        qualified=[row for row in group['members'] if row.get('has_view') and row.get('status')=='valid']
        if not qualified:qualified=[row for row in group['members'] if row.get('has_view') and row.get('native_verified')]
        representative=min(qualified,key=lambda row:(row['rwp'] is None,row['rwp'] or 0,row['id'])) if qualified else group['representative']
        displayed.append(dict(group,representative=representative))
    return sorted(displayed,key=lambda group:(group['representative'].get('status')!='valid',
        group['representative'].get('rwp') is None,group['representative'].get('rwp') or 0,group['id']))


def disposition_reason(row):
    if row.get('reason'):return row['reason']
    if row['status']=='optimization_unresolved':return 'Native convergence or rank checks did not qualify this checkpoint.'
    if row['status']=='physical_rejection':return 'The refined model violates the declared physical bounds.'
    if row['status']=='valid':return 'Native parameter/data audits and the declared fit criterion passed.'
    return 'Native parameter/data audits retained.' if row['has_view'] else 'Awaiting an independently audited native result.'


def display_status(status):
    return {'prepared':'awaiting dispatch','running':'assigned to a batch',
            'submitting':'submission pending','profile_equivalent':'equivalent screened start',
            'inadequate_fit':'Rwp target not met'}.get(status,status.replace('_',' '))


def model_search_text(row):
    """Search model identities even when a failed start has no curve."""
    status=str(row.get('status') or '')
    parts=[str(row.get(key) or '') for key in ('name','bns','candidate','seed')]
    return ' '.join([*parts,status,display_status(status)]).lower()


def matching_members(members,query):
    query=str(query or '').strip().lower()
    return [row for row in members if not query or query in model_search_text(row)]


def pin_label(row,*,include_start=False):
    """Keep checkpoint quality visible when a curve leaves the main selection."""
    parts=[str(row.get('name') or 'Model')]
    if include_start:
        if row.get('candidate'):parts.append(str(row['candidate'])[-6:])
        if row.get('seed'):parts.append(str(row['seed']))
    rwp=row.get('rwp')
    finite=isinstance(rwp,(int,float)) and not isinstance(rwp,bool) and np.isfinite(rwp) and rwp>=0
    parts.append(f'Rwp {rwp:.3f}%' if finite else 'Rwp unavailable')
    parts.append(display_status(row.get('status') or 'status unavailable'))
    return ' · '.join(parts)


def fit_caption(row,*,temperature=''):
    """Keep the selected acquisition/model identifiable in a full-screen fit."""
    if not row:return ''
    metadata=row.get('metadata') or {}
    parts=[str(metadata.get('scan') or 'Selected scan')]
    if temperature:parts.append(temperature)
    parts.append(str(row.get('name') or 'Selected model'))
    if row.get('bns'):parts.append('BNS '+str(row['bns']))
    rwp=row.get('rwp')
    if isinstance(rwp,(int,float)) and not isinstance(rwp,bool) and np.isfinite(rwp):
        parts.append(f'Rwp {rwp:.3f}%')
    status=row.get('status')
    if status:parts.append((('profiled optimization checks passed' if profiled_optimizer_qualified(row) else 'profiled optimization needs review') if profiled_optimizer_evidence(row) else 'native checks passed') if status=='valid' else display_status(status))
    return ' · '.join(parts)


def fit_diagnostics(row):
    """Explain the declared screen and the audited native result separately.

    Baseline residual windows can include profile or nuclear-model errors. They
    are not indexed magnetic reflections, nor another acceptance rule.
    """
    if not row:return []
    if row.get('status')=='profile_equivalent':
        details=[dict(label='Screened alternative',value='Agreement was checked within the fitted domain. The plot and Bragg ticks show the native-verified representative; the magnetic parameter table retains this screened start. Equality outside masked or excluded regions is not established. Open the representative for its audited GPX.')]
        if row.get('native_reference_status')=='inadequate_fit':
            details.append(dict(label='Fit quality',value='The representative did not meet the declared fit-adequacy criterion. Profile agreement does not make this an adequate fit.'))
        return details
    def finite(value):
        return isinstance(value,(int,float)) and not isinstance(value,bool) and np.isfinite(value)
    details=[];criterion=row.get('adequate_rwp_percent');optimizer=profiled_optimizer_evidence(row)
    if finite(criterion):
        checks='profiled convergence, native-objective, parameter and physicality checks' if optimizer else 'native convergence, parameter and physicality checks'
        details.append(dict(label='Declared fit criterion',value=f'Rwp ≤ {criterion:g}% after {checks}.'))
    if optimizer:
        passed=profiled_optimizer_qualified(row)
        iterations=optimizer.get('iterations');maximum=optimizer.get('maximum_iterations')
        details.append(dict(label='Profiled quadratic optimizer',value=
            f"{'Independent checks passed' if passed else 'Independent checks did not pass'} · {iterations} / {maximum} iterations. "
            'GSAS-II convergence and covariance are not claimed.'))
        null_rwp=(optimizer.get('nuclear_baseline') or {}).get('rwp')
        if finite(null_rwp):
            details.append(dict(label='Matched nuclear-only control',value=
                f'Rwp {null_rwp:.3f}% after optimizing the same allowed scale/background scope, '
                'with identical fixed nuclear geometry, instrument and fit domain.'))
    native=row.get('native_convergence') or {}
    parts=[]
    if type(native.get('converged')) is bool:parts.append('converged' if native['converged'] else 'not converged')
    if native.get('Aborted') is True:parts.append('aborted')
    rank=native.get('SVD0')
    if type(rank) is int and rank>=0:parts.append(f'{rank} rank deficiencies')
    if parts:details.append(dict(label='Native optimization',value=' · '.join(parts)))
    targets=row.get('residual_targets') or {};count=targets.get('target_count');underfit=targets.get('underfit_peak_count')
    if type(count) is int and count>0 and type(underfit) is int and 0<=underfit<=count:
        text=f'{underfit} of {count} fixed nuclear-baseline residual windows still below 50% integrated signal coverage.'
        improvement=targets.get('weighted_residual_improvement')
        if finite(improvement):text+=f' Weighted absolute residual change {(-100*improvement):+.1f}% in those windows.'
        text+=' Windows can contain nuclear or profile mismatch; inspect the measured and calculated curves.'
        details.append(dict(label='Residual-window check',value=text))
    return details


def embedding_evidence(rows,selected):
    """Compare exact model embeddings within one approved source protocol.

    Preserve all native dispositions. Start counts describe optimization and
    coverage, never independent measurements or a structure probability.
    """
    if not selected:return [],''
    keys=('protocol','source','source_sha256')
    if not all(selected.get(key) for key in keys):return [],''
    same_source=[row for row in rows if all(row.get(key)==selected.get(key) for key in keys)]
    partitions={}
    for row in same_source:partitions.setdefault(row['candidate'],[]).append(row)
    table=[];summary=''
    for candidate,members in partitions.items():
        qualified=[row for row in members if row.get('status')=='valid' and row.get('has_view') and
                   isinstance(row.get('rwp'),(int,float)) and np.isfinite(row['rwp']) and row['rwp']>=0]
        unresolved=[row['rwp'] for row in members if row.get('status')=='optimization_unresolved' and
                    row.get('has_view') and isinstance(row.get('rwp'),(int,float)) and np.isfinite(row['rwp']) and row['rwp']>=0]
        terminal=sum(row.get('execution_state')=='terminal' for row in members)
        best=min(qualified,key=lambda row:row['rwp']) if qualified else None
        low,high=(min(row['rwp'] for row in qualified),max(row['rwp'] for row in qualified)) if qualified else (None,None)
        label=members[0]['name']+(' · BNS '+members[0]['bns'] if members[0].get('bns') else '')+' · '+candidate[-6:]
        table.append(dict(id=candidate,model=label,coverage=f'{terminal} / {len(members)}',
            qualified=len(qualified),parameters=best.get('nvars') if best else None,
            best_rwp=round(low,5) if best else None,
            rwp_range=f'{low:.5f}–{high:.5f}%' if best else '—',
            unresolved_rwp=round(min(unresolved),5) if unresolved else None,
            selected='Current' if candidate==selected['candidate'] else '',_sort=(best is None,low if best else float('inf'),candidate)))
        if candidate==selected['candidate']:
            summary=f'{terminal} / {len(members)} starts terminal; {len(qualified)} qualified for this exact embedding.'
            if best:summary+=f' Qualified Rwp range {low:.5f}–{high:.5f}%.'
            if unresolved:summary+=f' Lowest unresolved checkpoint {min(unresolved):.5f}%.'
    table.sort(key=lambda row:row['_sort'])
    for row in table:row.pop('_sort')
    return table,summary


def campaign_can_resume(summaries,enrollments,*,following=False):
    """A queued acquisition is recoverable even before its first trial exists."""
    return (any(s['pending_count'] for s in summaries) or following or
            any(e['status'] in {'queued','preparing'} for e in enrollments))


PROFILED_QUALIFICATION_CHECKS=(
    'raw_optimizer_success','independent_stationarity','no_local_descent',
    'curvature_nonnegative','interior_solution','improves_nuclear_baseline',
    'nuclear_null_identity','serialized_basis_preserved',
    'serialized_native_stationarity','serialized_native_identity')


def validated_profiled_evidence(evidence,policy,status):
    value=evidence.get('optimizer')
    if not isinstance(value,dict) or value.get('schema')!='radar-profiled-magnetic-optimizer/v1' or value.get('name')!='profiled_quadratic':
        raise ValueError('The profiled optimizer receipt is missing or does not match its reviewed protocol')
    qualification=value.get('qualification')
    if (not isinstance(qualification,dict) or type(qualification.get('passed')) is not bool
            or not isinstance(qualification.get('checks'),dict) or not isinstance(qualification.get('reasons'),list)
            or not all(isinstance(reason,str) for reason in qualification['reasons'])):
        raise ValueError('The profiled optimizer qualification evidence is incomplete')
    checks=qualification['checks']
    if any(type(checks.get(key)) is not bool for key in PROFILED_QUALIFICATION_CHECKS):
        raise ValueError('The profiled optimizer qualification checks are incomplete')
    maximum=value.get('maximum_iterations');iterations=value.get('iterations')
    if (type(maximum) is not int or maximum!=policy.get('optimizer_iterations')
            or type(iterations) is not int or not 0<=iterations<=maximum
            or type(value.get('success')) is not bool):
        raise ValueError('The profiled optimizer budget differs from its reviewed protocol')
    if value.get('uncertainty_available') is not False or value.get('native_convergence_declared') is not False:
        raise ValueError('The profiled optimizer cannot claim native covariance or convergence')
    if status=='valid' and not (qualification['passed'] and value['success'] and all(checks[key] for key in PROFILED_QUALIFICATION_CHECKS)):
        raise ValueError('The profiled result is marked valid without its required optimizer checks')
    return {key:value.get(key) for key in
        ['schema','name','iterations','maximum_iterations','success','status','message','qualification','uncertainty_available','native_convergence_declared','tolerances','nuclear_baseline','candidate_applied']}


def profiled_optimizer_evidence(row):
    value=row.get('optimizer')
    return value if isinstance(value,dict) and value.get('name')=='profiled_quadratic' and value.get('schema')=='radar-profiled-magnetic-optimizer/v1' else None


def profiled_optimizer_qualified(row):
    value=profiled_optimizer_evidence(row)
    return value is not None and value.get('success') is True and (value.get('qualification') or {}).get('passed') is True


def unqualified_covariance(row):
    """An aborted or rank-deficient checkpoint has no qualified error bars."""
    native=row.get('native_convergence') or {}
    rank=native.get('SVD0')
    return (row.get('screening_only') or profiled_optimizer_evidence(row) is not None or row.get('status')=='optimization_unresolved' or
            native.get('converged') is False or native.get('Aborted') is True or
            (isinstance(rank,(int,float)) and np.isfinite(rank) and rank>0))


def _native_qualification_note(row):
    """Describe recorded optimization and physicality as separate evidence."""
    native=row.get('native_convergence') or {}
    converged=native.get('converged')
    parts=[('The native solver reported convergence.' if converged is True else
            'The native solver did not converge.' if converged is False else
            'Native convergence evidence is unavailable.')]
    if native.get('Aborted') is True:parts.append('The refinement was aborted.')
    rank=native.get('SVD0')
    if (isinstance(rank,(int,float)) and not isinstance(rank,bool)
            and np.isfinite(rank) and rank>=0 and float(rank).is_integer()):
        if rank>0:parts.append(f'{rank:g} rank '+('deficiency remains.' if rank==1 else 'deficiencies remain.'))
        else:parts.append('No rank deficiencies were reported.')
    else:parts.append('Rank evidence is unavailable.')
    if row.get('status')=='physical_rejection':
        parts.append('Physical rejection: '+disposition_reason(row))
    return ' '.join(parts)+' '


def moment_precision_note(row):
    """Flag weak conditional estimates without turning a screen into a test.

    This does not change a sealed trial's disposition, infer a norm error,
    combine correlated components, or assign a probability of magnetic order.
    """
    if row.get('kind') != 'magnetic':return ''
    if row.get('screening_only'):
        return 'Surrogate-screened magnetic parameters. Their calculated profile agrees with the linked native-verified representative; no separate native refinement or parameter uncertainties are claimed for this start.'
    optimizer=profiled_optimizer_evidence(row)
    if optimizer:
        passed=profiled_optimizer_qualified(row)
        statement='The profiled optimizer and its independent native-objective checks passed.' if passed else 'The profiled optimizer did not pass its declared checks; this is an unqualified checkpoint.'
        if optimizer.get('candidate_applied') is False:
            statement+=' Initial candidate retained; the last optimizer attempt is in the diagnostics.'
        if row.get('status')=='physical_rejection':statement+=' Physical rejection: '+disposition_reason(row)
        elif row.get('status')=='inadequate_fit':statement+=' The declared fit criterion was not met.'
        return (statement+' GSAS-II convergence and covariance were not claimed. '
            'Component, magnitude and direction uncertainties are unavailable and are hidden. '
            'A fitted direction is conditional on this model and does not establish canting or a unique magnetic structure.')
    kernel_note=('Native calculation requires review: this build has verified moment-coordinate '
        'and symmetry-dependent displacement defects for some magnetic models. '
        'Moment directions and model ranking remain provisional. '
        if row.get('magnetic_function_sha256')=='82076dc0571cad6d6e2882c5f7ced9003f08af94449aa293b0fb4c5c2ae7b9f6' else '')
    unqualified=unqualified_covariance(row)
    qualification_note=(_native_qualification_note(row)
        if unqualified or row.get('status')=='physical_rejection' else '')
    if unqualified:
        return kernel_note+qualification_note+(
                'Moment values are retained as a checkpoint; uncertainty estimates '
                'from this attempt are not qualified and are hidden in the table.')
    estimates=row.get('refined_moments') or []
    if not estimates:return kernel_note+qualification_note+'Moment uncertainties are unavailable in this receipt. An audited fit alone does not establish magnetic order.'
    weak=[];missing=[]
    for item in estimates:
        ratios=[];sigmas=item.get('component_esd') or {};values=item.get('moment') or []
        for index,axis in enumerate(('mx','my','mz')):
            if index>=len(values):continue
            value,sigma=values[index],sigmas.get(axis)
            if isinstance(value,(int,float)) and isinstance(sigma,(int,float)) and np.isfinite(value) and np.isfinite(sigma) and sigma>0:
                ratios.append(abs(value)/sigma)
        # A legacy collinear norm may have an actual native uncertainty.
        if not ratios:
            value,sigma=item.get('magnitude'),item.get('magnitude_esd')
            if isinstance(value,(int,float)) and isinstance(sigma,(int,float)) and np.isfinite(value) and np.isfinite(sigma) and sigma>0:
                ratios.append(abs(value)/sigma)
        label=item.get('label','Magnetic atom')
        if not ratios:missing.append(label)
        elif max(ratios)<3:weak.append(f'{label} (largest component / s.u. {max(ratios):.2f})')
    messages=[]
    if weak:messages.append('Weak conditional moment estimates: '+', '.join(weak)+'. No independently varied component exceeds 3 conditional standard uncertainties for these atoms.')
    if missing:messages.append('Moment uncertainty unavailable for '+', '.join(missing)+'.')
    if messages:messages.append('This component screen is not a joint significance test or proof of magnetic order; uncertainties depend on the selected model and fixed nuclear parameters.')
    if row.get('status')=='physical_rejection':
        messages.append('Conditional uncertainties do not qualify a physically rejected model.')
    return kernel_note+qualification_note+' '.join(messages)


def parameter_rows(row,view):
    """Conditional moment uncertainties come from the audited native receipt.

    A nonzero best-fit moment can remain entirely unresolved by the data. Do
    not infer an uncertainty from its magnitude or hide a missing uncertainty.
    """
    estimates={a['label']:a for a in row.get('refined_moments',[])}
    covariance_ok=not unqualified_covariance(row)
    def uncertainty_label(value):
        return round(value,6) if (covariance_ok and isinstance(value,(int,float))
            and not isinstance(value,bool) and np.isfinite(value) and value>=0) else '—'
    rows=[]
    for phase in view['phases']:
        symmetry=phase.get('space_group') or '—'
        if row.get('kind') == 'structural':
            for label, value in zip(('a', 'b', 'c', 'alpha', 'beta', 'gamma'), phase.get('cell', [])):
                if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
                    rows.append(dict(phase=phase['name'], space_group=symmetry, parameter='Cell · '+label,
                        value=round(value, 6), uncertainty='—', unit='Å' if label in ('a','b','c') else '°'))
        for value in phase.get('moments',phase.get('modes',[])):
            magnetic='magnitude' in value
            uncertainty=estimates.get(value['label'],{}).get('magnitude_esd') if magnetic else value.get('esd')
            rows.append(dict(phase=phase['name'],space_group=symmetry,parameter=value['label'],
                             value=round(value.get('magnitude',value.get('value')),6),
                             uncertainty=uncertainty_label(uncertainty),
                             unit='µB' if magnetic else 'Å'))
            if magnetic and 'components' in value:
                component_errors=estimates.get(value['label'],{}).get('component_esd') or {}
                for axis,component in zip(('mx','my','mz'),value['components']):
                    error=component_errors.get(axis)
                    rows.append(dict(phase=phase['name'],space_group=symmetry,parameter=value['label']+' · '+axis,
                        value=round(component,6),uncertainty=uncertainty_label(error),unit='µB'))
    return rows


def parameter_note(row):
    if not row:
        return ''
    if row.get('kind') == 'structural':
        return ('Cell values use the child crystallographic basis shown above. Mode values are normalized '
                'ISODISTORT amplitudes, not individual atomic displacements or the bounded search coordinates. '
                'The space-group column reports the native child symmetry; phase names can retain their original labels. '
                'A dash means uncertainty is unavailable; none is inferred from the fit residual. '
                'The exported GPX contains the atomic coordinates and cell. The fit status above still applies.')
    return ('The first row for each magnetic atom is its magnitude; mx, my and mz use the candidate crystallographic '
            'basis shown in the embedding. Uncertainties are conditional on this model and its fixed nuclear parameters. '
            'A dash means fixed, dependent or unavailable; a magnitude uncertainty is not inferred from component errors '
            'without their covariance. A nonzero moment or qualified fit does not establish ordering.')


def group_models(rows,*,relative_tolerance=1e-6):
    """Only group equivalent calculated profiles on identical observed fit domains.

    Keep every member/disposition visible. This is display compression and does
    not equate space groups, count phase boundaries or choose a physical model.
    """
    groups=[];partitions={};no_curve={}
    for row in sorted(rows,key=lambda r:(r['rwp'] is None,r['rwp'] or 0,r['id'])):
        if not row['has_view']:
            key=(row['status'],row.get('reason',''))
            if key not in no_curve:
                no_curve[key]=dict(id=row['id'],members=[],representative=row);groups.append(no_curve[key])
            no_curve[key]['members'].append(row);continue
        path=Path(row['view_path']);stat=path.stat()
        domain,calc,scale=comparison_profile(str(path),stat.st_mtime_ns,stat.st_size,stat.st_ctime_ns)
        key=(row['source'],row.get('source_sha256'),domain);chosen=None
        for previous,profile in partitions.get(key,[]):
            if len(profile)==len(calc) and np.max(np.abs(profile-calc))<=relative_tolerance*scale:chosen=previous;break
        if chosen is None:
            chosen=dict(id=row['id'],members=[],representative=row);groups.append(chosen);partitions.setdefault(key,[]).append((chosen,calc))
        chosen['members'].append(row)
    return groups


def campaign_rows(root,*,experiment=None,acquisition=None):
    campaign=Campaign(root)
    protocols,trials,summaries=portfolio_snapshot(campaign)
    rows=[];names={}
    for trial in trials:
        # Restrict before opening any GPX/job/result files. Large portfolios in
        # another experiment must not delay the selected experiment's view.
        entry=protocols[trial['protocol']];metadata=entry['metadata'];protocol=entry['protocol']
        if experiment is not None and metadata.get('series')!=experiment:continue
        if acquisition is not None and (protocol.get('acquisition_id') or protocol['source']['sha256'])!=acquisition:continue
        if trial['protocol'] not in names:
            names[trial['protocol']]=structural_result_names(root,protocol,trial['protocol'])
        if trial['state']=='prepared':
            # Pending starts have no scientific evidence to display. Read their
            # exact sealed identity from the ledger once per protocol instead
            # of parsing tens of thousands of identical job.json files on each
            # refresh. Executed results still follow the full artifact audit.
            job=dict(id=trial['id'],protocol=protocol,protocol_sha256=trial['protocol'],
                     candidate_id=trial['candidate'],seed_id=trial['seed'],package_sha256=protocol['implementation_sha256'])
            row=_trial_identity(campaign.trial_directory(trial),job,metadata)
        else:
            row=read_trial(campaign.trial_directory(trial),metadata,expected_receipt=trial.get('receipt'))
        label=names[trial['protocol']].get(trial.get('candidate'))
        if label:row['name']=label
        row['execution_state']=trial['state']
        if trial['state']!='terminal' and row['status']=='pending':row.update(status=trial['state'],reason=trial['error'] or '')
        rows.append(row)
    return rows,summaries


def coverage_rows(rows):
    """One compact row per exact acquisition and declared hypothesis."""
    partitions={}
    for row in rows:partitions.setdefault(row['protocol'],[]).append(row)
    result=[]
    for protocol,members in partitions.items():
        first=members[0];metadata=first.get('metadata') or {};condition=metadata.get('condition')
        received=[r for r in members if r.get('execution_state')=='terminal']
        fits=[r['rwp'] for r in members if r.get('rwp') is not None and r.get('has_view')]
        qualified=[r['rwp'] for r in members if r.get('status')=='valid' and r.get('rwp') is not None]
        result.append(dict(id=protocol,source=first['source'],scan=metadata.get('scan') or first['source'],
            condition=(f"{condition:g} {metadata.get('condition_unit','')}".strip() if isinstance(condition,(int,float)) else '—'),
            bank=str(metadata.get('bank') or '—'),
            sites=', '.join(s['label']+'='+s['form_factor'] for s in first.get('sites',[])) or first.get('kind',''),
            coverage=f'{len(received)} / {len(members)}',qualified=len(qualified),
            best_qualified=f'{min(qualified):.3f}%' if qualified else '—',
            best_audited=f'{min(fits):.3f}%' if fits else '—',
            status='In progress' if len(received)!=len(members) else 'Review qualified alternatives' if qualified else 'No qualified fit'))
    return result


def model_progress(rows):
    """Lead with model coverage; starts and incompatible models are not fits."""
    if any(not r.get('protocol') or not r.get('candidate') for r in rows):
        return 'Model identities are being prepared; checked coverage is not available yet'
    models={}
    for row in rows:
        models.setdefault((row['protocol'],row['candidate']),[]).append(row)
    complete=sum(all(r.get('execution_state')=='terminal' for r in members) for members in models.values())
    evaluated=sum(any(r.get('native_verified') for r in members) for members in models.values())
    fits=[r for r in rows if r.get('native_verified') and r.get('has_view') and
          isinstance(r.get('rwp'),(int,float)) and np.isfinite(r['rwp'])]
    qualified=[r for r in fits if r.get('status')=='valid']
    leader=min(qualified or fits,key=lambda r:r['rwp']) if fits else None
    text=f'{complete} / {len(models)} scan/model embeddings processed · {evaluated} with checked native fits'
    sources={row['source'] for row in rows if row.get('source')}
    if len(sources)>1:
        return text+f' · {len(sources)} scans shown; select a scan to compare its models and Rwp'
    if leader:
        label='Lowest Rwp passing fit checks' if qualified else 'Lowest checked Rwp, needs review'
        text+=f" · {label}: {leader['name']} · Rwp {leader['rwp']:.3f}%"
        baseline=leader.get('baseline_rwp')
        if (leader.get('kind')=='magnetic' and type(baseline) in (int,float)
                and np.isfinite(baseline) and baseline>=0):
            text+=f" · nuclear baseline {baseline:.3f}% (change {leader['rwp']-baseline:+.3f} percentage points)"
    elif complete:
        text+=' · No checked fit returned from the processed models; inspect their failure or review reasons'
    else:
        text+=' · First checked fit pending; assigned batches may still be queued'
    return text


def scope_rows(rows, experiment, scope='experiment'):
    if scope=='campaign':return rows
    if scope!='experiment':raise ValueError('Unknown model-result scope')
    return [row for row in rows if row.get('metadata',{}).get('series')==experiment]
