"""Counterbalanced fresh-process timing with the full same fitting workload.

This is local native evidence, not a Galaxy dispatch or NDIP latency test.
No performance threshold changes the scientific acceptance requirements.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_background_native import measured
from gpx_recipe.symmetry.compiler import file_sha


CONSUMER='''import json,sys,time
from pathlib import Path
import numpy as np
from gpx_recipe.live import native_background as background
from gpx_recipe.symmetry.native_profiled import NativeMagneticStartingPoint
path=Path(sys.argv[1]);output=Path(sys.argv[2]);mode=sys.argv[3]
if mode=='original':
    def unavailable(**_):raise background.BackgroundCompilerUnavailable('Counterbalanced original native reference')
    background.compile_columns=unavailable
began=time.monotonic()
native=NativeMagneticStartingPoint(path,refine_scale=True,refine_background=True,maximum_moment=6.)
initialization_seconds=time.monotonic()-began;started=time.monotonic()
model,proof=native.build();compilation_seconds=time.monotonic()-started
assert len(native.background_indices)==6 and len(proof['heldout_relative_errors'])==4
assert proof['native_derivative_calls']==1
assert proof['background_compiler']['method']==('native full powder' if mode=='original' else 'unchanged native background function')
native_build_calls=native.calls
seeds=[np.eye(model.dimension)[i]*m for m in (.4,1.8) for i in range(model.dimension)]
rows=[];residuals=[];optimization_seconds=verification_seconds=0.
for seed in seeds:
    started=time.monotonic();fit=model.fit(seed,max_iterations=250)
    optimization_seconds+=time.monotonic()-started;started=time.monotonic()
    actual=native.residual(fit.coefficients,fit.scale,fit.background)
    error=model.heldout_error(fit.coefficients,fit.scale,fit.background,actual)
    verification_seconds+=time.monotonic()-started
    assert error<1e-8
    cost=float(.5*(actual@actual))
    assert abs(cost-fit.weighted_cost)<=1e-7+1e-8*abs(cost)
    rows.append(dict(cost=cost,native_error=error,optimizer_qualified=fit.success))
    residuals.append(actual)
np.savez_compressed(output.with_suffix('.npz'),residuals=np.array(residuals))
output.write_text(json.dumps(dict(mode=mode,rows=rows,starts=len(seeds),native_build_calls=native_build_calls,
    native_derivative_calls=native.derivative_calls,background_compiler=proof['background_compiler'],
    measured_pipeline_seconds=time.monotonic()-began,stages_seconds=dict(initialization=initialization_seconds,
        compilation=compilation_seconds,optimization=optimization_seconds,verification=verification_seconds))))
'''


@pytest.mark.parametrize('kind',['PNC','PNT'])
def test_counterbalanced_cold_background_compilation_preserves_complete_fits(tmp_path,kind):
    path=measured(tmp_path/'inputs',kind);original_sha=file_sha(path)
    consumer=tmp_path/'consumer.py';consumer.write_text(CONSUMER)
    records=[]
    orders=[['fast','original'],['original','fast'],['fast','original'],['original','fast']]
    for pair,order in enumerate(orders):
        results={}
        for mode in order:
            destination=tmp_path/f'{pair}-{mode}.json';began=time.monotonic()
            with destination.with_suffix('.log').open('w') as log:
                subprocess.run([sys.executable,str(consumer),str(path),str(destination),mode],
                    check=True,timeout=180,stdout=log,stderr=subprocess.STDOUT)
            data=json.loads(destination.read_text());data['fresh_process_seconds']=time.monotonic()-began
            assert data['starts']==6
            results[mode]=data
            assert file_sha(path)==original_sha
        assert results['original']['native_build_calls']==results['fast']['native_build_calls']+6
        for a,b in zip(results['fast']['rows'],results['original']['rows']):
            assert a['cost']==pytest.approx(b['cost'],rel=1e-6,abs=1e-6)
        with np.load(tmp_path/f'{pair}-fast.npz') as a,np.load(tmp_path/f'{pair}-original.npz') as b:
            np.testing.assert_allclose(a['residuals'],b['residuals'],rtol=1e-6,atol=2e-5)
        records.append(dict(pair=pair,order=order,results=results))
        (tmp_path/'progress.json').write_text(json.dumps(dict(kind=kind,complete_pairs=len(records),expected_pairs=4)))
    medians={mode:{stage:float(np.median([r['results'][mode][stage] for r in records]))
        for stage in ('fresh_process_seconds','measured_pipeline_seconds')} for mode in ('fast','original')}
    (tmp_path/'benchmark.json').write_text(json.dumps(dict(kind=kind,pairs=records,medians=medians,
        complete_starts_per_process=6,native_checked_fitted_profiles=48,source_sha256=original_sha,
        scope='Public artificial neutron case; eight independent Python processes in counterbalanced order, '+
            'identical original native checks, six starts and all fitted residual checks; '+
            'includes interpreter/import/setup/fit/output process wall time but excludes Docker/Galaxy startup, '+
            'queue, staging and publication; not NDIP latency or an experimental identification claim'),indent=2))
