"""Present the saved Galaxy state without conflating dispatch with execution."""
from datetime import datetime,timezone


def batch_rows(chunks,*,now=None):
    now=now or datetime.now(timezone.utc)
    rows=[]
    for key,chunk in chunks.items():
        job=chunk.get('job') or {};status=job.get('status','')
        if chunk.get('not_dispatched'):state='Not submitted'
        elif chunk.get('complete'):state='Results checked'
        elif not job:state='Preparing submission'
        elif status=='ok':state='Retrieving and checking results'
        elif status=='running':state='Running'
        elif status in {'error','cancelled'}:state=status.capitalize()
        else:state='Galaxy queued'
        created=job.get('created_utc');age='—'
        if created:
            try:
                start=datetime.fromisoformat(created.replace('Z','+00:00'))
                if start.tzinfo is None:start=start.replace(tzinfo=timezone.utc)
                seconds=max(0,int((now-start).total_seconds()))
                age=f'{seconds//60}m {seconds%60:02d}s'
            except (TypeError,ValueError):pass
        rows.append(dict(batch=key[:8],job=job.get('galaxy_job_id') or '—',state=state,
                         starts=len(chunk.get('members') or []),elapsed=age,
                         checked=job.get('updated_utc') or '—',created=created or ''))
    return sorted(rows,key=lambda r:(r['state']=='Results checked',r['created']),reverse=False)
