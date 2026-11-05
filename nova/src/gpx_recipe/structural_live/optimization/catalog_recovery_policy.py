"""Observation-only scheduling with an explicit minimum round for every family.

Ranks allocate additional work; they never exclude an incompletely optimized
model. Restart coverage is recorded, not inferred from a requested budget.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CatalogRecoveryPolicy:
    first_seconds: float = 8.
    second_seconds: float = 20.
    first_starts: int = 1
    second_starts: int = 2

    def __post_init__(self):
        if not np.isfinite([self.first_seconds,self.second_seconds]).all() or min(self.first_seconds,self.second_seconds)<=0:
            raise ValueError('Positive finite per-candidate round budgets required')
        if (type(self.first_starts) is not int or not 1<=self.first_starts<=16
                or type(self.second_starts) is not int or not 2<=self.second_starts<=16):
            raise ValueError('First pass needs a start and the second pass needs at least two bounded starts')

    def fit_options(self,round_index):
        if type(round_index) is not int or round_index not in (0,1):raise ValueError('Unsupported coverage round')
        return dict(seconds=self.first_seconds if round_index==0 else self.second_seconds,
            starts=self.first_starts if round_index==0 else self.second_starts,
            precomputed_proposal_steps=True,balanced_restarts=True)


def round_candidates(catalog_ids,records,round_index):
    """No ranking cut before every catalog entry has a terminal round receipt."""
    ids=list(catalog_ids)
    if not ids or len(set(ids))!=len(ids):raise ValueError('Nonempty unique catalog IDs required')
    if type(round_index) is not int or round_index not in (0,1):raise ValueError('Unsupported coverage round')
    if any(r['candidate_id'] not in ids for r in records):raise ValueError('Unknown candidate in scheduler history')
    completed={r['candidate_id'] for r in records if r.get('round_index')==round_index and r.get('terminal',False)}
    return [cid for cid in ids if cid not in completed]


def prior_start_exclusions(fit):
    return [a['initial_parameters'] for a in fit.get('attempts',[]) if a.get('initial_parameters') is not None]


def optimization_status(rounds):
    attempted=[]
    for record in rounds:
        for attempt in record.get('fit',{}).get('attempts',[]):
            if attempt.get('iterations',0)>0 and attempt.get('initial_parameters') is not None:
                point=np.asarray(attempt['initial_parameters'],float)[:-2]
                if all(np.linalg.norm(point-other)>.2 for other in attempted):attempted.append(point)
    terminal={r['round_index'] for r in rounds if r.get('terminal',False)}
    successful={r['round_index'] for r in rounds if r.get('terminal',False) and r.get('state')=='fitted'}
    failed=sorted({r['round_index'] for r in rounds if r.get('terminal',False) and r.get('state') in {'failed','timed_out'}})
    return dict(coverage_rounds_complete={0,1}.issubset(terminal),coverage_rounds_successful={0,1}.issubset(successful),
        failed_rounds=failed,distinct_starts_with_iterations=len(attempted),
        independent_start_allocation_completed=len(attempted)>=2,
        global_optimum_certified=False,model_exclusion_authorized=False,
        interpretation='Verified numerical fits; unresolved optimization is not evidence against a structural family')


def select_recovery_candidates(rows,top=3,include_flexible=True,include_pareto=True):
    """Allocate extra work without rejecting unselected families.

    The Pareto frontier preserves simple families with promising observed
    fits even when several flexible models precede them in raw fit rank.
    This is an allocation heuristic, not a complexity penalty or evidence.
    Older receipts without a cell count use their recorded mode count only.
    """
    if type(top) is not int or top<1:raise ValueError('Positive shortlist size required')
    ids=[r['candidate_id'] for r in rows]
    if len(set(ids))!=len(ids):raise ValueError('One current record per candidate required')
    fitted=[r for r in rows if r.get('state')=='fitted']
    for r in fitted:
        if not np.isfinite(r['deviance']) or r['deviance']<0:raise ValueError('Finite nonnegative deviance required')
        if any(type(v) is not int or v<0 for v in [r['mode_dimensions'],r.get('cell_dimensions',0)]):
            raise ValueError('Nonnegative integer structural dimensions required')
    ranked=sorted(fitted,key=lambda r:(r['deviance'],r['candidate_id']))
    selected={r['candidate_id'] for r in ranked[:top]}
    if include_pareto:
        best=float('inf')
        for r in sorted(ranked,key=lambda r:(r['mode_dimensions']+r.get('cell_dimensions',0),r['deviance'],r['candidate_id'])):
            if r['deviance']<best:selected.add(r['candidate_id']);best=r['deviance']
    if include_flexible and ranked:
        selected.add(min(ranked,key=lambda r:(-r['mode_dimensions'],r['deviance'],r['candidate_id']))['candidate_id'])
    return [r['candidate_id'] for r in ranked if r['candidate_id'] in selected]
