import hashlib
from pathlib import Path
from unittest.mock import patch
import pytest

pytest.importorskip('GSASII.GSASIIstrMath')
from GSASII import GSASIIstrMath as M
from gpx_recipe.live import validated_backend as backend


def test_version_pinning_and_restore():
    original=M.MagStructureFactor2
    production=hashlib.sha256(Path(M.__file__).read_bytes()).hexdigest()==backend.PINNED_PRODUCTION_STRMATH
    with backend.corrected_backend() as identity:
        assert identity['version']==backend.VERSION
        assert (M.MagStructureFactor2 is original) is production
        assert backend.active_identity()==identity
        with pytest.raises(RuntimeError):
            with backend.corrected_backend():pass
    assert M.MagStructureFactor2 is original
    assert backend.active_identity() is None
    with pytest.raises(ValueError):
        with backend.corrected_backend():raise ValueError('worker failure')
    assert M.MagStructureFactor2 is original


def test_wrong_native_version_refused(monkeypatch):
    for name in ('PINNED_PRODUCTION_STRMATH','PINNED_STRMATH','PINNED_LINUX_STRMATH'):
        monkeypatch.setattr(backend,name,'wrong')
    with pytest.raises(RuntimeError):
        with backend.corrected_backend():pass


def test_legacy_single_crystal_guard():
    if hashlib.sha256(Path(M.__file__).read_bytes()).hexdigest()==backend.PINNED_PRODUCTION_STRMATH:
        pytest.skip('Shared production entry points validate the neutron-powder input before native execution')
    with backend.corrected_backend():
        with pytest.raises(ValueError):M.MagStructureFactor2({},None,':0:','0::',{}, {':0:histType':'SNC'}, {})
