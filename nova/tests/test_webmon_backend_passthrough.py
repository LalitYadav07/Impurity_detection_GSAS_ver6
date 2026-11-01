"""Production uses reviewed build-time code; no runtime replacement is needed."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import pytest
from gpx_recipe.live import validated_backend


def native_function(*_):
    return None


def test_shared_build_is_reported_without_runtime_replacements(tmp_path,monkeypatch):
    source=tmp_path/'native.py';source.write_text('shared immutable runtime')
    monkeypatch.setattr(validated_backend,'PINNED_PRODUCTION_STRMATH',hashlib.sha256(source.read_bytes()).hexdigest())
    module=SimpleNamespace(__file__=str(source),MagStructureFactor2=native_function,
                           MagStructureFactorDerv=native_function,StructureFactor2=native_function)
    functions,hashes=validated_backend.build_functions(module)
    assert functions=={}
    assert set(hashes)=={'MagStructureFactor2','MagStructureFactorDerv','StructureFactor2'}
    assert module.MagStructureFactor2 is native_function


def test_unknown_runtime_cannot_use_local_adapter_without_explicit_opt_in(tmp_path,monkeypatch):
    monkeypatch.delenv('RADAR_PD_LOCAL_EXPERIMENTAL_BACKEND',raising=False)
    source=tmp_path/'native.py';source.write_text('legacy runtime')
    with pytest.raises(RuntimeError,match='explicit local replay'):
        validated_backend.build_functions(SimpleNamespace(__file__=str(source)))
