"""Fixed-total temporal count allocation for a supplied phase history.

This is the conditional multinomial observation layer used when development
diagnostics fix observed block totals. It does not turn those totals into
absolute rate predictions or automatically supply a Poisson-model posterior
over phase paths. An outside category is retained, and overlapping count views
must not be supplied as independent channels.
"""
from dataclasses import dataclass
import numpy as np
from .temporal_counts import PhaseRateLaw, _budget, _readonly, _seed


@dataclass(frozen=True)
class ConditionedTimeAllocation:
    cell_block: np.ndarray
    cell_category: np.ndarray
    probability: np.ndarray
    block_totals: np.ndarray
    metadata: dict

    @property
    def mean(self):
        return self.block_totals[self.cell_block]*self.probability

    @property
    def variance(self):
        return self.mean*(1-self.probability)

    def channel_block_covariance(self, block, channel, *, max_bytes=256*1024**2):
        if (type(block) is not int or not 0 <= block < len(self.block_totals)
                or type(channel) is not int or not 0 <= channel < self.block_totals.shape[1]):
            raise ValueError('An existing block and distinct count channel are required')
        p=self.probability[self.cell_block==block,channel]
        if 24*len(p)**2 > _budget(max_bytes):
            raise ValueError('Full conditional covariance exceeds the working-memory budget')
        return self.block_totals[block,channel]*(np.diag(p)-np.outer(p,p))

    def sample_counts(self, *, seed, max_bytes=256*1024**2):
        """Sequential binomials give exact integer total closure, including zero weights."""
        if self.probability.size*40 > _budget(max_bytes):
            raise ValueError('Conditional count allocation exceeds the working-memory budget')
        rng=np.random.default_rng(_seed(seed))
        counts=np.zeros(self.probability.shape,np.int64)
        for block, totals in enumerate(self.block_totals):
            rows=np.flatnonzero(self.cell_block==block)
            weights=self.probability[rows]
            # Positive suffix sums avoid subtraction of nearly equal remainders.
            suffix=np.cumsum(weights[::-1],axis=0)[::-1]
            remaining=totals.copy()
            for j,row in enumerate(rows):
                p=np.divide(weights[j],suffix[j],out=np.zeros(len(totals)),where=suffix[j]>0)
                assert np.all((p>=0)&(p<=1))
                draw=rng.binomial(remaining,p)
                counts[row]=draw;remaining-=draw
            assert not remaining.any()
        return _readonly(counts)


def conditioned_time_allocation(law, dose, phases_rad, block_index, category_index,
                                block_totals, *, dose_unit, max_bytes=256*1024**2):
    """Allocate fixed block/channel totals over nonoverlapping time categories.

    Every pulse has a block and a category. Use category -1 for pulses outside
    the requested windows; they remain in the normalization and count draw.
    Empty-dose cells remain with probability zero. Positive totals with zero
    predicted intensity fail explicitly. Output rows are sparse block/category
    pairs present in the source clock, sorted lexicographically.

    Conditional on the supplied phase, probabilities are integrated q*g divided
    by the complete block/channel integral. If the caller samples phases from
    a uniform stationary prior, that defines a fixed-total mixture model; it
    is not automatically the posterior obtained by conditioning a known-rate
    Poisson model on those observed totals.
    """
    if not isinstance(law,PhaseRateLaw):
        raise ValueError('An explicit positive phase rate law is required')
    phases=law._phases(phases_rad)
    blocks,categories,totals=map(np.asarray,[block_index,category_index,block_totals])
    n=len(phases)
    if (blocks.shape!=(n,) or categories.shape!=(n,) or blocks.dtype.kind not in 'iu'
            or categories.dtype.kind not in 'iu' or totals.ndim!=2 or totals.dtype.kind not in 'iu'
            or not 1<=len(totals)<=1024 or totals.shape[1]!=law.channels
            or np.any(blocks<0) or np.any(blocks>=len(totals))
            or np.any(categories<-1) or np.any(categories>2**31-1)
            or np.any(totals<0) or np.any(totals>np.iinfo(np.int64).max)):
        raise ValueError('Matching integer block/category identities and nonnegative int64 block totals are required')
    q=np.asarray(dose)
    if q.shape!=(n,) or q.dtype.kind not in 'iuf' or not np.isfinite(q).all() or np.any(q<0):
        raise ValueError('Finite nonnegative pulse doses are required')
    # Bound sorting, phase copies, all possible sparse rows and output arrays.
    if n*(144+16*phases.shape[1]+64*law.channels)>_budget(max_bytes):
        raise ValueError('Conditional temporal cells exceed the working-memory budget')
    meta=law._metadata(dose_unit)
    totals=totals.astype(np.int64)
    order=np.lexsort((categories,blocks));b=blocks[order];c=categories[order]
    boundaries=np.r_[0,np.flatnonzero((np.diff(b)!=0)|(np.diff(c)!=0))+1,n]
    row_blocks=[];row_categories=[];weights=[]
    rates=np.ones(law.channels)
    for first,last in zip(boundaries[:-1],boundaries[1:]):
        ix=order[first:last]
        row_blocks.append(int(b[first]));row_categories.append(int(c[first]))
        weights.append(law.integrated_mean(rates,q[ix],phases[ix],max_bytes=max_bytes))
    weights=np.asarray(weights);row_blocks=np.asarray(row_blocks,np.int64)
    denominator=np.zeros(totals.shape,float);np.add.at(denominator,row_blocks,weights)
    if np.any((denominator==0)&(totals>0)):
        raise ValueError('A positive block total has zero predicted pulse intensity; no dose or count is imputed')
    probability=np.divide(weights,denominator[row_blocks],out=np.zeros_like(weights),where=denominator[row_blocks]>0)
    assert np.all((probability>=0)&(probability<=1))
    check=np.zeros(totals.shape,float);np.add.at(check,row_blocks,probability)
    np.testing.assert_allclose(check[denominator>0],1.,rtol=2e-13,atol=2e-15)
    meta.update(version='positive-phase-fixed-total-allocation/1',
                phase_law_version=meta['version'],given_phase_path_not_marginalized=True,
                observed_block_total_conditioning_not_included=False,
                block_totals_are_inputs_not_absolute_count_rate_predictions=True,
                uniform_prior_path_not_automatically_a_known_rate_Poisson_posterior=True,
                outside_category=-1,outside_dose_and_counts_not_dropped=True,
                exact_conditional_marginal_variance='N*p*(1-p)',
                conditional_channel_independence_not_independence_after_phase_marginalization=True,
                no_probability_count_or_dose_floor=True)
    return ConditionedTimeAllocation(_readonly(row_blocks),_readonly(np.asarray(row_categories,np.int64)),
                                     _readonly(probability),_readonly(totals),meta)
