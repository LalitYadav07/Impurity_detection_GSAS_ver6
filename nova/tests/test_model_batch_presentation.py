from datetime import datetime,timezone
from radar_pd_nova.model_batch_presentation import batch_rows


def test_dispatch_execution_and_retrieval_remain_distinct():
    chunks={key:dict(members=['a','b'],job=dict(status=key,galaxy_job_id=key,created_utc='2026-11-03T12:00:00Z')) for key in ['queued','running','ok']}
    rows=batch_rows(chunks,now=datetime(2026,11,3,12,2,tzinfo=timezone.utc))
    assert {r['job']:r['state'] for r in rows}=={'queued':'Galaxy queued','running':'Running','ok':'Retrieving and checking results'}
    assert all(r['elapsed']=='2m 00s' and r['starts']==2 for r in rows)
    chunks['ok']['complete']=True
    assert next(r for r in batch_rows(chunks) if r['job']=='ok')['state']=='Results checked'
