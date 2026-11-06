"""Trust adaptation survives alternating blocks but never crosses starts."""
import numpy as np
import pytest

from test_structural_live_alternating_native_fit import setup
from gpx_recipe.structural_live.optimization import profiled_native_fit as module
from gpx_recipe.structural_live.optimization.fit_fast_native import residual_and_factor


def test_successful_blocks_retain_trust_adaptation_and_new_starts_reset_it():
    s,t,o,z,c,b,_=setup()
    result,rate=module.fit(s,t,o,b,seconds=3.,starts=2,evaluator=c)
    assert len(result['attempts'])==2
    decreased=False
    for start in range(2):
        for kind in ('cell','modes'):
            rows=[row for row in result['blocks'] if row['start']==start and row['kind']==kind]
            assert len(rows)>1
            assert rows[0]['initial_damping']==1e-3
            decreased|=any(row['final_damping']<row['initial_damping'] for row in rows)
            assert all(left['final_damping']==right['initial_damping']
                       for left,right in zip(rows,rows[1:]))
    assert decreased
    residual,_=residual_and_factor(o,rate)
    assert result['best']['deviance']==pytest.approx(residual@residual)
    assert result['all_allowed_modes_retained']
    assert not result['global_optimum_certified']


def test_unsuccessful_step_increases_damping_without_false_small_progress_exit(monkeypatch):
    s,t,o,z,c,b,_=setup();seen=[]
    def failed_step(jacobian,residual,damping,*args,**kwargs):
        seen.append(damping)
        return np.zeros(jacobian.shape[1]),dict(information_score=1.)
    monkeypatch.setattr(module,'bounded_damped_step',failed_step)
    result,rate=module.fit(s,t,o,b,seconds=3.,starts=1,evaluator=c,
                           force_native_mode_derivatives=True)
    assert max(seen)>=1e7
    assert max(row['final_damping'] for row in result['blocks'])>1e7
    assert result['attempts'][0]['reason']=='native_descent_stalled'
    assert any(row['reason']=='damping_increased' for row in result['blocks'])
    assert not result['global_optimum_certified']
    residual,_=residual_and_factor(o,rate)
    assert result['best']['deviance']==pytest.approx(residual@residual)
