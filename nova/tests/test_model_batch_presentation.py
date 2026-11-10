from datetime import datetime,timezone
from radar_pd_nova.model_batch_presentation import batch_rows, allocation_caption


def test_allocation_caption_distinguishes_model_count_from_native_cpu_budget():
    text=allocation_caption(dict(workers=4,concurrent_model_workers=1,
        representative_workers_per_model=4,memory_mb=8192,allocation_reported=True))
    assert '4 native worker slots' in text
    assert '1 model × up to 4 native checks' in text
    assert '8192 MiB' in text


def test_unreported_allocation_is_visible():
    text=allocation_caption(dict(workers=1,allocation_reported=False,memory_mb=None))
    assert '1 native worker slot (allocation unreported)' in text


def test_dispatch_execution_and_retrieval_remain_distinct():
    chunks={key:dict(members=['a','b'],job=dict(status=key,galaxy_job_id=key,created_utc='2026-11-03T12:00:00Z')) for key in ['queued','running','ok']}
    rows=batch_rows(chunks,now=datetime(2026,11,3,12,2,tzinfo=timezone.utc))
    assert {r['job']:r['state'] for r in rows}=={'queued':'Galaxy queued','running':'Running','ok':'Retrieving and checking results'}
    assert all(r['elapsed']=='2m 00s' and r['starts']==2 for r in rows)
    chunks['ok']['complete']=True
    assert next(r for r in batch_rows(chunks) if r['job']=='ok')['state']=='Results checked'


def test_reviewed_retry_does_not_claim_an_uncertain_attempt_was_never_submitted():
    rows=batch_rows({'uncertain':dict(members=['a'],complete=True,not_dispatched=True,
        explicit_retry={'reason':'No matching job in complete History lookup'}),
        'unposted':dict(members=['b'],complete=True,not_dispatched=True)})
    assert {r['batch']:r['state'] for r in rows}=={
        'uncertai':'Superseded after reviewed retry','unposted':'Not submitted'}
