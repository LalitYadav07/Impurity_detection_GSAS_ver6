from radar_pd_nova.model_batch_budget import batch_size, output_measurements, TARGET_BYTES
from radar_pd_nova.model_batch_budget import concurrency_batch_limit
from gpx_recipe.symmetry.campaign import Campaign
import json
import pytest


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


def trials(campaign,protocol,count,*,state='prepared',dispatch=None):
    with campaign.connect() as db:
        for index in range(count):
            uid=f'{protocol}-{index}'
            handle=json.dumps(dict(dispatch_key=dispatch,key=uid)) if dispatch else None
            db.execute('INSERT INTO trials(id,protocol,candidate,seed,state,handle,updated) VALUES(?,?,?,?,?,?,?)',
                (uid,protocol,'candidate',str(index),state,handle,0.))


def test_single_scan_control_fills_eight_slots_without_changing_its_finite_members(tmp_path):
    c=Campaign(tmp_path/'control');trials(c,'one-scan',24)
    original={row['id'] for row in c.trials()};groups=[]
    for index in range(8):
        size=min(8,concurrency_batch_limit(c,'one-scan',8))
        ready=[row for row in c.trials() if row['state']=='prepared'][:size]
        groups.append([row['id'] for row in ready])
        for row in ready:c.set_state(row['id'],'running',handle=dict(dispatch_key=f'batch-{index}',key=row['id']))
    assert [len(group) for group in groups]==[3]*8
    assert {uid for group in groups for uid in group}==original
    assert len([uid for group in groups for uid in group])==24
    assert not any(row['state']=='prepared' for row in c.trials())


def test_many_ready_scans_keep_the_existing_native_and_output_budget(tmp_path):
    c=Campaign(tmp_path/'many')
    for index in range(8):trials(c,f'scan-{index}',24)
    assert concurrency_batch_limit(c,'scan-0',8)==24
    assert min(batch_size(2600000,[]),concurrency_batch_limit(c,'scan-0',8))==6


def test_inflight_cancellation_and_unacknowledged_intents_keep_their_slots(tmp_path):
    c=Campaign(tmp_path/'busy');trials(c,'next',12)
    trials(c,'accepted',8,state='running',dispatch='one-shared-job')
    trials(c,'cancelling',4,state='cancelling',dispatch='cancelling-job')
    trials(c,'ambiguous',1,state='submitting')
    assert concurrency_batch_limit(c,'next',4)==12
    assert concurrency_batch_limit(c,'next',8)==3
    assert concurrency_batch_limit(c,'next',1)==12


@pytest.mark.parametrize('value',[0,-1,True,1.5])
def test_batch_share_requires_a_configured_positive_job_limit(tmp_path,value):
    with pytest.raises(ValueError,match='positive configured'):
        concurrency_batch_limit(Campaign(tmp_path/'invalid'),'source',value)
