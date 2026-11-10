"""Whole-model batching learns only from exact, completed native portfolios."""
import json

import pytest

from gpx_recipe.symmetry.contracts import digest, iter_trials, trial_id
from radar_pd_nova.model_batch_budget import complete_model_measurements, model_group_count
from test_symmetry_contracts import make_protocol


def measured(*, seconds=20., workers=1, group_workers=1, size=1000000, files=20):
    return dict(result=dict(wall_seconds=seconds+2, allocation=dict(workers=workers)),
        group_measurements=[dict(candidate='a', declared=24, wall_seconds=seconds,
            representative_workers=group_workers, output_bytes=size, output_files=files)],
        output_measurements=dict(fixed_bytes=1000, fixed_files=2))


def test_one_complete_model_pilot_then_amortize_dispatch_without_extra_cpus():
    assert model_group_count(1000, []) == 1
    assert model_group_count(1000, [measured()]) == 8
    assert model_group_count(1000, [measured(seconds=120)]) == 1
    assert model_group_count(1000, [measured(seconds=120, workers=4)]) == 5
    assert model_group_count(1000, [measured()], maximum=3) == 3


def test_parallel_checks_inside_a_pilot_are_not_counted_as_free_concurrency():
    # One pilot's four verification workers become one worker/model in a
    # four-model batch. Charge their combined time rather than multiplying
    # the apparent 20-second throughput by four again.
    pilot = measured(seconds=20, workers=4, group_workers=4)
    assert model_group_count(1000, [pilot]) == 8
    slower = measured(seconds=60, workers=4, group_workers=4)
    assert model_group_count(1000, [slower]) == 2  # includes measured process overhead


def test_native_process_and_chunk_overhead_count_against_the_time_target():
    pilot = measured(seconds=1)
    pilot['result']['wall_seconds'] = 100
    assert model_group_count(1000, [pilot]) == 1


def test_large_outputs_and_files_preserve_fourfold_reserve():
    assert model_group_count(1000, [measured(size=20*1024*1024)]) == 1
    assert model_group_count(1000, [measured(files=1000)]) == 1
    assert model_group_count(10*1024*1024, [measured()]) == 6
    pilot = measured(); pilot['output_measurements']['fixed_bytes'] = 128*1024*1024
    assert model_group_count(1000, [pilot]) == 1


def test_legacy_unsupported_and_invalid_measurements_keep_the_pilot():
    assert model_group_count(1000, [dict(result=dict(allocation=dict(workers=16)))]) == 1
    for field, value in [('wall_seconds', float('nan')), ('output_bytes', 0), ('representative_workers', True)]:
        pilot = measured(); pilot['group_measurements'][0][field] = value
        assert model_group_count(1000, [pilot]) == 1
    pilot = measured(); pilot['result']['allocation'] = dict(visible_cpu_count=192)
    assert model_group_count(1000, [pilot]) == 8  # measured single-worker throughput only
    pilot['result']['wall_seconds'] = 60
    assert model_group_count(1000, [pilot]) == 3


@pytest.fixture
def portfolio(tmp_path):
    protocol = make_protocol(tmp_path/'input')
    protocol['policy']['grouped_surrogate'] = True
    protocol['seeds'] = [dict(protocol['seeds'][0], id=str(i)) for i in range(3)]
    ids = [trial_id(protocol, candidate['id'], seed['id']) for candidate, seed in iter_trials(protocol)]
    root = tmp_path/'received'
    for uid in ids:
        path = root/'trials'/uid
        path.mkdir(parents=True)
        (path/'result.json').write_text('{}')
    (root/'trials'/ids[0]/'native.gpx').write_bytes(b'checked native project')
    data = root/'trials'/ids[0]/'group-data'; data.mkdir()
    summary = dict(schema='radar-profiled-group-result/v1', protocol_sha256=digest(protocol),
        candidate_id=protocol['candidates'][0]['id'], declared=3, screened=3,
        native_verified_profiles=1, wall_seconds=20., allocation=dict(representative_workers=1))
    path = data/'summary.json'; path.write_text(json.dumps(summary))
    response = dict(protocol_sha256=digest(protocol), trials=[dict(id=uid,state='finished') for uid in ids])
    return root, protocol, ids, response, path, summary


def test_exact_complete_portfolio_records_all_its_artifacts(portfolio):
    root, protocol, ids, response, path, summary = portfolio
    result = complete_model_measurements(root, protocol, ids, response)
    assert len(result) == 1 and result[0]['declared'] == 3
    assert result[0]['output_files'] == 5
    assert result[0]['output_bytes'] == sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    assert result[0]['wall_seconds'] == 20.


@pytest.mark.parametrize('change', [dict(native_verified_profiles=0), dict(declared=2),
    dict(screened=2), dict(protocol_sha256='f'*64), dict(candidate_id='other'),
    dict(wall_seconds=float('inf')), dict(allocation=dict(representative_workers=True))])
def test_partial_unsupported_or_foreign_group_metrics_cannot_drive_batching(portfolio,change):
    root, protocol, ids, response, path, summary = portfolio
    summary.update(change); path.write_text(json.dumps(summary))
    assert complete_model_measurements(root, protocol, ids, response) == []


def test_pending_portfolio_and_wrong_source_cannot_drive_batching(portfolio):
    root, protocol, ids, response, path, summary = portfolio
    assert complete_model_measurements(root, protocol, ids[:-1], response) == []
    response['trials'].pop()
    assert complete_model_measurements(root, protocol, ids, response) == []
    response['protocol_sha256'] = 'f'*64
    assert complete_model_measurements(root, protocol, ids, response) == []


def test_missing_oversized_or_ambiguous_metrics_do_not_fail_scientific_collection(portfolio):
    root, protocol, ids, response, path, summary = portfolio
    path.write_text('x'*65537)
    assert complete_model_measurements(root, protocol, ids, response) == []
    path.write_text(json.dumps(summary))
    duplicate = root/'trials'/ids[1]/'group-data'; duplicate.mkdir()
    (duplicate/'summary.json').write_text(json.dumps(summary))
    assert complete_model_measurements(root, protocol, ids, response) == []
    path.unlink(); (duplicate/'summary.json').unlink()
    assert complete_model_measurements(root, protocol, ids, response) == []
