from radar_pd_nova.model_batch_budget import batch_size, output_measurements, TARGET_BYTES


def measured(seconds=10.,workers=1,**update):
    values=dict(max_trial_bytes=1300000,max_trial_files=16,fixed_bytes=10000,fixed_files=2,native_artifacts=True)
    values.update(update)
    return dict(result=dict(trials=[dict(wall_seconds=seconds)]*6,allocation=dict(workers=workers)),output_measurements=values)


def test_small_pilot_then_measured_batch_amortizes_dispatch_without_increasing_workers():
    source=2600000
    assert batch_size(source,[])==6
    assert batch_size(source,[measured()])==18
    assert batch_size(source,[measured(workers=2)])==25
    assert batch_size(source,[measured(seconds=900.)])==1


def test_unsupported_only_pilot_and_legacy_receipts_do_not_claim_smaller_native_outputs():
    source=2600000
    assert batch_size(source,[measured(native_artifacts=False)])==6
    legacy=measured();legacy.pop('output_measurements')
    assert batch_size(source,[legacy])==6


def test_expanded_bytes_files_and_large_outlier_bound_the_next_batch():
    source=2600000
    assert batch_size(source,[measured(max_trial_bytes=50*1024*1024)])==1
    assert batch_size(source,[measured(max_trial_files=5000)])==1
    assert batch_size(source,[measured(fixed_bytes=TARGET_BYTES)])==1
    assert batch_size(1,[measured(seconds=.01)],maximum=3)==3


def test_verified_tree_measurement_counts_all_artifacts_and_uses_largest_trial(tmp_path):
    a=tmp_path/'trials/a';a.mkdir(parents=True);(a/'refined.gpx').write_bytes(b'x'*20);(a/'result.json').write_text('{}')
    b=tmp_path/'trials/b';b.mkdir();(b/'result.json').write_text('{}')
    (tmp_path/'response.json').write_text('{}')
    assert output_measurements(tmp_path,{'a','b'})==dict(max_trial_bytes=22,max_trial_files=2,
        fixed_bytes=2,fixed_files=1,native_artifacts=True)
