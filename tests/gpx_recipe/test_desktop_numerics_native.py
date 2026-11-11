"""The pinned desktop uses reviewed powder math and preserves other paths."""
import inspect
import os
from pathlib import Path

import pytest

pytest.importorskip('GSASII.GSASIIscriptable')
from GSASII import GSASIIspc as S, GSASIIstrMath as M
from gpx_recipe.symmetry import kernel_patch, space_group_patch


@pytest.fixture
def original_source():
    root=Path(os.environ.get('RADAR_GSASII_DESKTOP_SOURCE','/opt/GSAS-II/GSASII'))
    assert (root/'GSASIIstrMath.py').is_file(), 'The pinned original desktop source is required'
    return root


def test_desktop_installs_the_exact_reviewed_corrections(original_source):
    for name,module,patch in [('GSASIIstrMath',M,kernel_patch),('GSASIIspc',S,space_group_patch)]:
        expected,record=patch.corrected_source((original_source/(name+'.py')).read_text())
        assert Path(module.__file__).read_text()==expected
        assert record['corrected_file_sha256']==kernel_patch.sha256(Path(module.__file__).read_bytes())


@pytest.mark.parametrize('name',['MagStructureFactor2','MagStructureFactorDerv'])
def test_desktop_retains_original_non_powder_implementation(original_source,name):
    original=kernel_patch.function_source((original_source/'GSASIIstrMath.py').read_text(),name)
    saved=inspect.getsource(getattr(M,'_radar_original_'+name))
    assert saved.replace('def _radar_original_'+name+'(', 'def '+name+'(', 1)==original


@pytest.mark.parametrize('name',['MagStructureFactor2','MagStructureFactorDerv'])
@pytest.mark.parametrize('kind',['SNC','SNT','PXC','SXC'])
def test_non_powder_calls_delegate_without_entering_powder_math(monkeypatch,name,kind):
    received=[];sentinel=object()
    def original(*args):received.append(args);return sentinel
    monkeypatch.setattr(M,'_radar_original_'+name,original)
    arguments=({},None,':0:','0::',{}, {':0:histType':kind},{})
    assert getattr(M,name)(*arguments) is sentinel
    assert received==[arguments]


@pytest.mark.parametrize('point',[[5e-6,.173,.317],[-5e-6,.173,.317]])
def test_desktop_site_symmetry_respects_periodic_boundaries(point):
    error,group=S.SpcGroup('C m m 2');assert not error
    actual=S.SytSym(point,group);expected=S.SytSym([0.,.173,.317],group)
    assert actual[:2]==expected[:2] and actual[3]==expected[3]
