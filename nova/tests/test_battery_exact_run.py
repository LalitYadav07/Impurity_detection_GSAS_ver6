"""Exact acquisition selection reads only verified members of one folder."""
import asyncio
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from radar_pd_nova import battery_controller as module


class State(SimpleNamespace):
    def change(self, *fields):
        return lambda callback: callback

    def flush(self):
        pass


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv('RADAR_PD_ONCAT_ENABLED', 'false')
    app = module.BatteryMixin()
    app.server = SimpleNamespace(state=State())
    app._init_battery()
    folder = tmp_path / 'shared/autoNOM_slowrate/GSAS'
    folder.mkdir(parents=True)
    (folder / 'NOM138504tof.gsa').write_text('Only the filename is needed for this metadata gate.')
    app._battery_inventory = dict(instrument='NOM', ipts='IPTS-42', root=str(tmp_path),
        product_families=[dict(folder='shared/autoNOM_slowrate/GSAS')])
    app.server.state.battery_instrument = 'NOM'
    app.server.state.battery_ipts = 'IPTS-42'
    app.server.state.battery_family = 'shared/autoNOM_slowrate/GSAS'
    app.server.state.battery_exact_run = '138504'
    return app


def record(run):
    return dict(run=run, status='available', fields={}, issues=[],
        identity=dict(sample_id='sample-A', identity_status='resolved'),
        exposure=dict(start_utc='2020-01-01T00:00:00Z', end_utc='2020-01-01T00:01:00Z', duration_seconds=60))


def mock_metadata(monkeypatch, calls):
    def inspect(instrument, ipts, run, **kwargs):
        calls.append(run)
        return record(run)
    monkeypatch.setattr(module, 'inspect_run', inspect)
    monkeypatch.setattr(module, 'ExperimentMetadataIndex', lambda *a, **k:
        SimpleNamespace(enrich=lambda run, identity: identity))


def inspect(app):
    async def run():
        app.inspect_battery_acquisitions()
        if app._battery_task is not None:
            await app._battery_task
    asyncio.run(run())


def test_exact_run_has_one_read_and_no_automatic_preparation(app, monkeypatch):
    calls = []
    mock_metadata(monkeypatch, calls)
    inspect(app)
    assert calls == [138504]
    assert [r['run'] for r in app._battery_acquisitions] == [138504]
    assert [r['value'] for r in app.server.state.battery_model_runs] == ['138504']
    assert 'Inspected exact run 138504 from 1 runs' in app.server.state.battery_message
    assert app.server.state.battery_exact_run == '138504'
    assert app._battery_prepared is None and not app.server.state.battery_model_ready
    assert not app.server.state.battery_busy


@pytest.mark.parametrize('value', [True, None, [], 0, -1, '1.5', 1.5, '1e3', '../138504',
    '138504-138505', 'NOM_138504', '0', '2147483648', '9'*100])
def test_invalid_exact_input_stops_before_background_reads(app, monkeypatch, value):
    app.server.state.battery_exact_run = value
    monkeypatch.setattr(module, 'family_runs', lambda *a: pytest.fail('Invalid input reached filesystem'))
    inspect(app)
    assert app._battery_task is None
    assert 'Exact run number' in app.server.state.battery_error


def test_exact_run_ignores_unused_representative_count(app, monkeypatch):
    calls = []
    mock_metadata(monkeypatch, calls)
    app.server.state.battery_inspection_limit = 'not used'
    inspect(app)
    assert calls == [138504]


def test_blank_keeps_original_even_sampling(app, tmp_path, monkeypatch):
    app.server.state.battery_exact_run = ' '
    app.server.state.battery_inspection_limit = 3
    monkeypatch.setattr(module, 'family_runs', lambda *a: [1, 2, 3, 4, 5])
    calls = []
    mock_metadata(monkeypatch, calls)
    inspect(app)
    assert calls == [1, 3, 5]
    assert app.server.state.battery_total_runs == 5


def test_missing_run_does_not_read_nearby_or_outside_folder(app, tmp_path, monkeypatch):
    calls = []
    mock_metadata(monkeypatch, calls)
    app.server.state.battery_exact_run = '138505'
    (tmp_path / 'NOM138505tof.gsa').write_text('Outside selected folder')
    inspect(app)
    assert calls == [] and app._battery_acquisitions == []
    assert 'not in the selected reduction folder' in app.server.state.battery_error


@pytest.mark.parametrize('name', ['NOM_138505_12.h5.gsa', 'NOM_138505_138506.gsa', 'NOM138505tof.dat'])
def test_derived_or_unpreparable_only_file_cannot_supply_parent_identity(app, tmp_path, monkeypatch, name):
    calls = []
    mock_metadata(monkeypatch, calls)
    app.server.state.battery_exact_run = '138505'
    (tmp_path / app.server.state.battery_family / name).write_text('No contents should be read')
    inspect(app)
    assert calls == [] and app._battery_acquisitions == []
    assert app.server.state.battery_error


def test_known_summed_folder_cannot_become_individual_parent(app, tmp_path, monkeypatch):
    calls = []
    mock_metadata(monkeypatch, calls)
    folder = 'shared/autoreduce/multi_banks_summed/GSAS'
    path = tmp_path / folder
    path.mkdir(parents=True)
    (path / 'NOM_138504.gsa').write_text('Summed product')
    app._battery_inventory['product_families'] = [dict(folder=folder)]
    app.server.state.battery_family = folder
    inspect(app)
    assert calls == [] and 'derived products' in app.server.state.battery_error


@pytest.mark.parametrize('filename,expected_reads', [('PG3_138504-1.gsa', [138504]),
    ('PG3_138504-138505.gsa', [])])
def test_powgen_exact_source_must_be_an_individual_supported_bank(app, tmp_path, monkeypatch, filename, expected_reads):
    folder = tmp_path / 'shared/powgen'
    folder.mkdir()
    (folder / filename).write_text('Filename-only eligibility; native preparation checks body.')
    app._battery_inventory.update(instrument='PG3', product_families=[dict(folder='shared/powgen')])
    app.server.state.battery_instrument = 'PG3'
    app.server.state.battery_family = 'shared/powgen'
    calls = []
    mock_metadata(monkeypatch, calls)
    inspect(app)
    assert calls == expected_reads


def test_unresolved_exact_metadata_stays_visible_but_not_preparable(app, monkeypatch):
    value = record(138504)
    value['status'] = 'conflict'
    value['issues'] = [dict(code='run_conflict', message='Filename and stored run number disagree')]
    monkeypatch.setattr(module, 'inspect_run', lambda *a, **k: deepcopy(value))
    monkeypatch.setattr(module, 'ExperimentMetadataIndex', lambda *a, **k:
        SimpleNamespace(enrich=lambda run, identity: identity))
    inspect(app)
    assert app.server.state.battery_model_runs == []
    assert 'stored run number disagree' in app.server.state.battery_rows[0]['issues']
    assert app._battery_prepared is None


@pytest.mark.parametrize('stage', ['listing', 'membership', 'raw', 'identity', 'catalog'])
@pytest.mark.parametrize('field,value', [('battery_exact_run', '138505'),
    ('battery_family', 'shared/another/GSAS'), ('battery_ipts', 'IPTS-99')])
def test_changed_selection_cannot_apply_an_inflight_result(app, monkeypatch, stage, field, value):
    def changed():
        setattr(app.server.state, field, value)
        app.server.state.battery_rows = [dict(run='new selection')]
    original_listing = module.family_runs
    original_membership = module.require_exact_run_source
    def listing(*args):
        result = original_listing(*args)
        if stage == 'listing': changed()
        return result
    def membership(*args):
        original_membership(*args)
        if stage == 'membership': changed()
    def raw(*args, **kwargs):
        if stage == 'raw': changed()
        return record(138504)
    def identity(run, value):
        if stage == 'identity': changed()
        return value
    def catalog(*args, **kwargs):
        if stage == 'catalog': changed()
        return dict(fields={}, message='Unavailable', status='unavailable')
    monkeypatch.setattr(module, 'family_runs', listing)
    monkeypatch.setattr(module, 'require_exact_run_source', membership)
    monkeypatch.setattr(module, 'inspect_run', raw)
    monkeypatch.setattr(module, 'ExperimentMetadataIndex', lambda *a, **k: SimpleNamespace(enrich=identity))
    app._battery_catalog_client = SimpleNamespace(begin_batch=lambda: None, acquisition=catalog)
    inspect(app)
    assert app.server.state.battery_rows == [dict(run='new selection')]
    assert app._battery_acquisitions == [] and not app.server.state.battery_busy


def test_editing_exact_run_invalidates_old_binding_without_reading(app, monkeypatch):
    app._battery_acquisitions = [record(138504)]
    app._battery_series = object()
    app._battery_prepared = (Path('/old'), {})
    app.server.state.battery_model_ready = True
    monkeypatch.setattr(module, 'family_runs', lambda *a: pytest.fail('Edit started discovery'))
    app._battery_exact_run_changed()
    assert app._battery_acquisitions == [] and app._battery_series is None
    assert app._battery_prepared is None and app._battery_task is None
    assert app.server.state.battery_exact_run == '138504'
    app._battery_family_changed()
    assert app.server.state.battery_exact_run == ''


def test_ui_explains_exact_membership_and_sampling():
    from radar_pd_nova.app import RadarPdNovaApp
    template = RadarPdNovaApp().layout.html
    assert 'Exact run number (optional)' in template
    assert 'individual reduced observation in the selected folder' in template
    assert 'Leave blank for representative sampling' in template
    assert 'does not prepare or submit an analysis' in template
