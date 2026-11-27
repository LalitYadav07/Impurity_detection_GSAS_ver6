import pytest
from types import SimpleNamespace

from radar_pd_nova import powgen_samples
from radar_pd_nova.powgen_samples import include_run, run_role, select_runs
from radar_pd_nova.powgen_watch import WatchedRun


def make_run(tmp_path, number, title):
    p = tmp_path / f"PG3_{number}.gsa"
    p.write_text(f"Sample Run: {number}\n# {title}\n")
    return WatchedRun(number, str(p), "f")


def test_background_and_cooling_need_explicit_inclusion(tmp_path):
    empty = make_run(tmp_path, 1, "Setup empty furnace without sample")
    ramp = make_run(tmp_path, 2, "Mixture Cooling to RT")
    ordinary = make_run(tmp_path, 3, "Temperature calibration at 300")
    assert run_role(empty)["run_role"] == "background"
    assert not include_run(empty)
    assert include_run(empty, include_background_runs=True)
    assert not include_run(ramp)
    assert include_run(ramp, include_ramp_runs=True)
    assert include_run(ordinary)
    assert not include_run(ordinary, excluded_run_numbers=(3,))


def test_missing_title_is_included_without_guessing(tmp_path):
    assert include_run(WatchedRun(1, str(tmp_path / "missing.gsa"), "f"))


@pytest.mark.parametrize('title', [
    'Er2O3 warming to 100K CW 1.5',
    'Er2O3_warming_to_100_K_CW_1.5',
    'sample WARMING TO 25 °C',
    'sample warming to +2.5K',
])
def test_explicit_warming_to_temperature_is_a_queue_ramp(tmp_path, title):
    run = make_run(tmp_path, 62959, title)
    assert run_role(run)['run_role'] == 'ramp'
    assert not include_run(run)
    assert include_run(run, include_ramp_runs=True)


@pytest.mark.parametrize('title', [
    'Er2O3 5K CW 1.5',
    'Er2O3 warm 100K plateau',
    'Er2O3 warming branch, stabilized at 5K',
    'Sample after warming to equilibrium',
    'Warming treatment sample at 100K',
    'warming to 100 Kelvin model name',
])
def test_warm_state_or_branch_is_not_inferred_to_be_a_queue_ramp(tmp_path, title):
    run = make_run(tmp_path, 62958, title)
    assert run_role(run)['run_role'] == 'unclassified'
    assert include_run(run)


def test_latest_stable_er_scan_survives_warming_discovery_and_explicit_exclusion(tmp_path):
    stable = make_run(tmp_path, 62958, 'Er2O3 5K CW 1.5')
    warming = make_run(tmp_path, 62959, 'Er2O3 warming to 100K CW 1.5')

    class Index:
        def identity(self, run):
            return {'sample_id': '116051'}

    def latest(runs, **kwargs):
        return [r.run_number for r in select_runs(runs, Index(), ('116051',), limit=1, **kwargs)]

    assert latest([stable, warming]) == [62958]
    assert latest([stable, warming], include_ramp_runs=True) == [62959]
    assert latest([stable, warming], include_ramp_runs=True,
                  excluded_run_numbers=(62959,)) == [62958]
    # Re-discovery of a newer explicit ramp must not displace the plateau.
    newer = make_run(tmp_path, 62960, 'Er2O3 warming to 150K CW 1.5')
    assert latest([stable, warming, newer], excluded_run_numbers=(62959,)) == [62958]


def test_catalog_inventory_uses_the_same_warming_role_as_queue(tmp_path, monkeypatch):
    runs = [make_run(tmp_path, 62958, 'Er2O3 5K CW 1.5'),
            make_run(tmp_path, 62959, 'Er2O3 warming to 100K CW 1.5')]
    metadata = {r.run_number: {'sample_id': '116051', 'identity_status': 'resolved',
                 'sample_name': 'Er2O3', 'run_title': run_role(r)['run_title']} for r in runs}
    monkeypatch.setattr(powgen_samples, 'load_sample_catalog', lambda *_: metadata)
    monkeypatch.setattr(powgen_samples, 'catalog_created_at', lambda *_: 0)
    index = powgen_samples.PowgenSampleIndex('IPTS-1', metadata_index=SimpleNamespace(
        instrument='PG3', revision=0, generation=0, notices=[],
        refresh=lambda **kw: None, enrich=lambda n, row, **kw: dict(row)))
    result = index.inventory(runs, catalog=True)
    assert [(r['value'], r['run_role']) for r in result['run_options']] == [
        (62958, 'unclassified'), (62959, 'ramp')]
