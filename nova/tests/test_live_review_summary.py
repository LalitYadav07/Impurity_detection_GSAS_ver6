"""Explain a bounded input selection without claiming a full proposal audit."""
from radar_pd_nova.live_run_selection import run_review_summary
from radar_pd_nova.powgen_samples import select_runs
from radar_pd_nova.powgen_watch import WatchedRun
from radar_pd_nova.hb2a_controller import preflight_hb2a_experiment
from test_hb2a import archive, profile, add


def test_limited_powgen_review_preserves_bounded_reads_and_explains_older_candidates(tmp_path):
    runs = []
    for number in range(1, 6):
        path = tmp_path / f'PG3_{number}.gsa'
        path.write_text('# empty cell\n' if number == 4 else '# sample\n')
        runs.append(WatchedRun(number, str(path), str(number)))
    checked = []

    class Index:
        def identity(self, run):
            checked.append(run.run_number)
            return {'sample_id': 'A'}

    stats = {}
    chosen = select_runs(runs, Index(), ('A',), limit=1,
        excluded_run_numbers=(5,), selection_stats=stats)
    assert [r.run_number for r in chosen] == [3]
    assert checked == [4, 3]
    assert stats['reviewed_scan_count'] == 3
    assert stats['excluded_count'] == 2
    assert stats['run_review_complete'] is False
    text = run_review_summary(stats)
    assert '2 excluded' in text and '3 candidate scans checked for selection' in text
    assert 'Older candidates were not all checked' in text


def test_complete_powgen_review_does_not_imply_additional_unreviewed_candidates(tmp_path):
    path = tmp_path / 'PG3_1.gsa'; path.write_text('# sample\n')
    stats = {}
    chosen = select_runs([WatchedRun(1, str(path), '1')], object(), None, selection_stats=stats)
    assert len(chosen) == 1 and stats['run_review_complete'] is True
    assert stats['reviewed_scan_count'] == 1
    assert 'Older candidates' not in run_review_summary(stats)


def test_hb2a_sample_and_backfill_omissions_are_not_reported_as_role_exclusions(archive, profile):
    for number in range(1, 6):
        add(archive, scan=number)

    class Index:
        def identity(self, run):
            return {'sample_id': 'B' if run.run_number == 3 else 'A',
                    'run_title': 'empty can' if run.run_number == 5 else 'sample'}

        def inventory(self, runs):
            return {'metadata_resolution': []}

    result = preflight_hb2a_experiment('IPTS-1', sample_index=Index(),
        selected_sample_ids=('A',), initial_backfill_limit=1,
        excluded_acquisitions=('HB2A_EXP1000_SCAN0004',),
        profile_path=profile, profile_acknowledged=True)
    assert result['ready'], result['message']
    assert result['scan_count'] == 1 and result['latest_run'] == 'HB2A_EXP1000_SCAN0002'
    assert result['total_scan_count'] == 5
    assert result['excluded_count'] == 2
    assert result['sample_filtered_count'] == 1
    assert result['backfill_omitted_count'] == 1
    assert result['reviewed_scan_count'] == 5 and result['run_review_complete'] is True
    text = run_review_summary(result)
    assert '2 excluded' in text
    assert '1 scan belongs to other samples' in text
    assert '1 additional matching scan is outside this initial batch' in text


def test_older_controller_result_remains_readable_without_inventing_review_scope():
    text = run_review_summary({'excluded_count': 3})
    assert text.startswith('3 excluded by run role or individual selection.')
    assert 'checked for selection' not in text and 'Older candidates' not in text
