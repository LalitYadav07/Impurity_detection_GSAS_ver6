"""Complete native coefficient checks must not consume their cached answer."""
from pathlib import Path

import pytest

pytest.importorskip('GSASII.GSASIIscriptable')

from gpx_recipe.preparation.worker import execute
from gpx_recipe.live.native_profile import NativeReducedProfile
from gpx_recipe.live.native_reflection import compile_profile_reflections
from gpx_recipe.live.reflection_factors import ScatteringFactorCache
from gpx_recipe.symmetry.compiler import file_sha
from test_preparation_worker_native import request


@pytest.mark.parametrize('kind',['PNC','PNT'])
def test_native_coefficient_oracle_bypasses_every_prepared_rendered_response(tmp_path,monkeypatch,kind):
    payload,protocol,candidate=request(tmp_path/'public-inputs',kind)
    payload['settings']=dict(candidate=candidate,k=protocol['k'],sites=protocol['sites'],
        maximum_moment=6.,refine_histogram_scale=True,refine_background=True)
    result=execute('magnetic',payload)
    path=Path(result['output'])/'native-model/magnetic-refinement.gpx';before=file_sha(path)
    native=NativeReducedProfile(path)
    indices=[i for i,key in enumerate(native.vary) if '::AM' in key or key.startswith('::constr')]
    cache=ScatteringFactorCache()
    def forbidden(*args,**kwargs):
        pytest.fail('The independent coefficient oracle read its prepared answer')
    monkeypatch.setattr(cache,'rendered_response',forbidden)
    checked=compile_profile_reflections(native,native.x0,indices,verify_coefficients=True,factor_cache=cache)
    proof=checked['diagnostics'];validation=proof['coefficient_validation']
    assert not proof['prepared_rendered_response_reused']
    assert validation['all_coefficient_directions_checked']
    assert validation['native_structure_factor_calls']>=2*len(indices)**2+1
    assert proof['native_profile_calls']>=1 and file_sha(path)==before
