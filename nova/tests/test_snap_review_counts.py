"""Selection counters must describe real filtering without expanding source reads."""
from radar_pd_nova.powgen_watch import WatchedRun
from radar_pd_nova.snap_controller import _select_runs, preflight_snap_experiment


def runs(numbers):
    return [WatchedRun(n, f'/SNS/SNAP/IPTS-1/shared/SNAP{n}_all.gsa', str(n),
                      instrument='SNAP') for n in numbers]


def test_explicit_and_background_exclusions_are_reported_once():
    calls = []

    class Index:
        def identity(self, run):
            calls.append(run.run_number)
            return {'sample_id': '', 'run_title': 'empty capillary' if run.run_number == 2 else 'Si standard'}

    stats = {}
    selected = _select_runs(runs(range(1, 6)), Index(), None,
                            excluded_run_numbers=(3, 4, 5), selection_stats=stats)
    assert [r.run_number for r in selected] == [1]
    assert calls == [2, 1]
    assert stats == {'excluded_count': 4, 'reviewed_scan_count': 5, 'run_review_complete': True}


def test_bounded_backfill_counts_only_checked_candidates():
    calls = []

    class Index:
        def identity(self, run):
            calls.append(run.run_number)
            return {'sample_id': 'A', 'run_title': 'empty cell' if run.run_number == 4 else 'sample'}

    stats = {}
    selected = _select_runs(runs(range(1, 5)), Index(), ('A',), limit=1, selection_stats=stats)
    assert [r.run_number for r in selected] == [3]
    assert calls == [4, 3]
    assert stats == {'excluded_count': 1, 'reviewed_scan_count': 2, 'run_review_complete': False}


def test_other_samples_are_not_reported_as_role_exclusions():
    class Index:
        def identity(self, run):
            return {'sample_id': 'B' if run.run_number == 2 else 'A', 'run_title': 'sample'}

    stats = {}
    selected = _select_runs(runs([1, 2]), Index(), ('A',), selection_stats=stats)
    assert [r.run_number for r in selected] == [1]
    assert stats == {'excluded_count': 0, 'reviewed_scan_count': 2, 'run_review_complete': True}


def test_preflight_exposes_exclusions_to_the_run_review_summary(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from radar_pd_nova import snap
    from radar_pd_nova.powgen_watch import WatchDefinition
    from test_snap import write_source

    source = write_source(tmp_path / 'SNAP000042_bank_test.gsa')
    monkeypatch.setattr(snap, 'source_directory', lambda *a, **k: tmp_path)
    monkeypatch.setattr(WatchDefinition, 'source_directory', property(lambda self: str(tmp_path)))

    def unexpected_identity_read(run):
        raise AssertionError('Explicitly excluded scans need no metadata read')

    result = preflight_snap_experiment('IPTS-1', selected_banks=(2,),
        sample_index=SimpleNamespace(identity=unexpected_identity_read),
        product_family=snap.parse_filename(source).family, excluded_run_numbers=(42,))
    assert not result['ready'] and result['scan_count'] == 0
    assert result['excluded_count'] == 1 and result['reviewed_scan_count'] == 1
    assert result['run_review_complete'] is True
