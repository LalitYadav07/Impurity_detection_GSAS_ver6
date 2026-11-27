"""Built-in readiness must fail before spawning science; custom/dry paths survive."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ndip_runner


def test_missing_parent_root_never_falls_back_to_empty_root(tmp_path, monkeypatch):
    monkeypatch.setenv('RADAR_PD_DB_ROOT', str(tmp_path))
    assert ndip_runner._db_root_for('neutron', None) == tmp_path / 'database_neutron'


def test_complete_flat_layout_stays_supported(tmp_path, monkeypatch):
    (tmp_path / 'catalog_deduplicated.csv').write_text('id,formula\n1,Si\n')
    monkeypatch.setenv('RADAR_PD_DB_ROOT', str(tmp_path))
    assert ndip_runner._db_root_for('neutron', None) == tmp_path


def test_missing_builtin_catalog_never_starts_scientific_subprocess(tmp_path, monkeypatch, capsys):
    data = tmp_path / 'synthetic.xye'
    data.write_text('10 2 1\n11 3 1\n')
    instrument = tmp_path / 'profile.instprm'
    instrument.write_text('#GSAS-II instrument parameter file\nType:PNC\nLam:1.5\n')
    config = tmp_path / 'config.yaml'
    assert ndip_runner.main(['configure', '--allowed-elements', 'Si', '--output', str(config)]) == 0
    monkeypatch.setenv('RADAR_PD_DB_ROOT', str(tmp_path / 'empty'))
    def forbidden(*args, **kwargs):
        pytest.fail('The scientific subprocess ran without its built-in catalog')
    monkeypatch.setattr(ndip_runner, '_pipeline_command', forbidden)
    assert ndip_runner.main(['analyze', '--config', str(config), '--data', str(data),
        '--instrument', str(instrument), '--work-dir', str(tmp_path / 'work'),
        '--output-dir', str(tmp_path / 'portal')]) == 1
    error = capsys.readouterr().err
    assert 'Built-in neutron catalog is unavailable' in error
    assert 'Traceback' not in error and len(error) < 250
    diagnostic = tmp_path / 'portal/diagnostics/catalog-readiness.json'
    assert diagnostic.is_file() and 'profiles64/index.csv' in diagnostic.read_text()


def test_missing_augmented_base_never_starts_builder(tmp_path, monkeypatch):
    from types import ModuleType
    fake = ModuleType('db_pack_builder')
    def forbidden(*args, **kwargs):
        pytest.fail('Builder ran without the requested augmented base')
    fake.build_augmented_db_pack = fake.build_mini_db_pack = forbidden
    monkeypatch.setitem(sys.modules, 'db_pack_builder', fake)
    cif = tmp_path / 'synthetic.cif'
    cif.write_text('data_synthetic\n')
    assert ndip_runner.main(['build-library', '--cif', str(cif), '--library-mode', 'augmented',
        '--base-db-root', str(tmp_path / 'missing'), '--output-root', str(tmp_path / 'output'),
        '--manifest-output', str(tmp_path / 'result.json')]) == 1
