"""Qualify single-pass native responses against the original construction."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

G = pytest.importorskip('GSASII.GSASIIscriptable')
from GSASII import GSASIIstrMain as MAIN
from gpx_recipe.execution import freeze
from gpx_recipe.structural_live.native import prepare
from gpx_recipe.structural_live.service import make_source, SearchSettings
from gpx_recipe.structural_live.optimization import profile_cache
from gpx_recipe.structural_live.optimization.native_profile_capture import CaptureUnavailable


def native_source(root, kind, *, histogram_name='public control'):
    """Public artificial structure; counts come from independent GSAS-II."""
    root.mkdir()
    if kind == 'PNC':
        parameters = ('Lam:1.54\nZero:0\nU:.12\nV:0\nW:.08\n'
                      'X:0\nY:0\nZ:0\nSH/L:.002\nPolariz.:0\nAzimuth:0\n')
        limits = (12., 125.)
    else:
        parameters = ('difC:6200\ndifA:2\ndifB:.4\nZero:1\n2-theta:105\nfltPath:15.5\nalpha:.99\n'
                      'beta-0:.02\nbeta-1:.03\nbeta-q:0\nsig-0:1\nsig-1:66\nsig-2:0\nsig-q:0\n'
                      'X:0\nY:0\nZ:0\nAzimuth:0\n')
        limits = (3., 40.)  # The simulation API takes milliseconds for TOF.
    instrument = root/'profile.instprm'
    instrument.write_text('#GSAS-II instrument parameter file\nType:'+kind+'\nBank:1\n'+parameters)
    source = root/'source.gpx'
    project = G.G2Project(newgpx=str(source))
    phase = project.add_phase(phasename='Public ErSi', spacegroup='C m c m',
                              cell=[4.195, 10.353, 3.779, 90., 90., 90.])
    phase.add_atom(0., .413, .25, 'Si', 'Si1')
    phase.add_atom(0., .142, .25, 'Er', 'Er1')
    hist = project.add_simulated_powder_histogram(histogram_name, str(instrument),
                *limits, Npoints=1501, scale=120., phases=[phase])
    hist.data['Background'][0][2] = 3
    hist.data['Background'][0][3:] = [3., .2, .1]
    freeze(project)
    hist.data['data'][1][1][:] = 1.
    project.save()
    hist.data['data'][1][1] = np.asarray(MAIN.DoNoFit(str(source), hist.name), float)
    hist.data['data'][1][2][:] = 4.
    project.save()
    child = Path(__file__).with_name('data')/'synthetic-ersi-cmcm-isodistort.cif'
    prepare(source, child, [[-1., 0, 0], [0, -1., 0], [0, 0, 1.]], [0, 0, 0], root/'prepared')
    context = make_source(root/'prepared', root/'evaluation', SearchSettings())
    return source, context


@pytest.mark.parametrize('kind', ['PNC', 'PNT'])
@pytest.mark.parametrize('shifted', [False, True])
def test_capture_matches_original_matrix_at_independent_cells(tmp_path, monkeypatch, kind, shifted):
    source, context = native_source(tmp_path/'input', kind)
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    bank = profile_cache.NativeProfileBank(context)
    cells = np.zeros(context.cell_dimensions)
    if shifted:
        cells = context.radii[-context.cell_dimensions:]*np.array([.13, -.21, .17])
    captured = bank.matrix(cells)
    assert bank.diagnostics()['response_methods'] == {'single native profile pass': 1}
    assert bank.native_profile_calls == 1
    assert bank.matrix(cells) is captured  # Reuse only this exact cell.
    bank.clear()

    def unsupported(*args, **kwargs):
        raise CaptureUnavailable('test of unsupported native driver')

    monkeypatch.setattr(profile_cache, 'capture_response', unsupported)
    reference = bank.matrix(cells)
    np.testing.assert_allclose(captured.toarray(), reference.toarray(), rtol=1e-12, atol=1e-12)
    assert bank.native_profile_calls == 1 + len(bank.hkls)
    assert bank.maximum_reconstruction_error < 1e-9
    assert bank.diagnostics()['capture_fallbacks'] == {'test of unsupported native driver': 1}
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original


def test_native_exception_restores_peak_function_and_capture_ownership(tmp_path, monkeypatch):
    source, context = native_source(tmp_path/'input', 'PNC')
    bank = profile_cache.NativeProfileBank(context)
    powder = bank.n.math.G2pwd
    original_peak = powder.getFCJVoigt3
    original_capture = profile_cache.capture_response

    def broken_capture(*args, **kwargs):
        calls = 0

        def broken_peak(*a, **kw):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('induced native peak failure')
            return original_peak(*a, **kw)

        powder.getFCJVoigt3 = broken_peak
        try:
            with pytest.raises(RuntimeError, match='induced native peak failure'):
                original_capture(*args, **kwargs)
            assert powder.getFCJVoigt3 is broken_peak
        finally:
            powder.getFCJVoigt3 = original_peak
        raise RuntimeError('induced native peak failure')

    monkeypatch.setattr(profile_cache, 'capture_response', broken_capture)
    with pytest.raises(RuntimeError, match='induced native peak failure'):
        bank.matrix(np.zeros(context.cell_dimensions))
    assert powder.getFCJVoigt3 is original_peak
    monkeypatch.setattr(profile_cache, 'capture_response', original_capture)
    matrix = bank.matrix(np.zeros(context.cell_dimensions))
    assert matrix.nnz > 0  # The failed capture released ownership.
    assert bank.builds == 1 and not bank.capture_fallbacks


@pytest.mark.parametrize('kind', ['PNC', 'PNT'])
def test_profile_edit_invalidates_captured_response(tmp_path, kind):
    _, context = native_source(tmp_path/'input', kind)
    bank = profile_cache.NativeProfileBank(context)
    cells = np.zeros(context.cell_dimensions)
    bank.matrix(cells)
    bank.n.parms[bank.hfx+'X'] += .1
    with pytest.raises(ValueError, match='Fixed instrument/profile/grid context changed'):
        bank.matrix(cells)
