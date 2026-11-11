"""Present the saved Galaxy state without conflating dispatch with execution."""
from datetime import datetime,timezone


def allocation_caption(allocated):
    workers=allocated.get('workers',1)
    text=f'{workers} native worker slot'+('s' if workers!=1 else '')
    if allocated.get('allocation_reported') is False:
        text+=' (allocation unreported)'
    models=allocated.get('concurrent_model_workers')
    checks=allocated.get('representative_workers_per_model')
    if models and checks:
        text+=f' · {models} model'+('s' if models!=1 else '')+f' × up to {checks} native checks'
    return text+f" · {allocated.get('memory_mb') or 'unreported'} MiB memory"


def batch_rows(chunks,*,now=None):
    now=now or datetime.now(timezone.utc)
    rows=[]
    for key,chunk in chunks.items():
        job=chunk.get('job') or {};status=job.get('status','')
        if chunk.get('explicit_retry'):state='Superseded after reviewed retry'
        elif chunk.get('not_dispatched'):state='Not submitted'
        elif chunk.get('complete'):state='Results checked'
        elif not job:state='Preparing submission'
        elif status=='ok':state='Retrieving and checking results'
        elif status=='running':state='Running'
        elif status in {'error','cancelled'}:state=status.capitalize()
        else:state='Galaxy queued'
        created=job.get('created_utc');age='—'
        terminal=(chunk.get('complete') or chunk.get('explicit_retry')
                  or status in {'ok','error','cancelled'})
        end=job.get('updated_utc') if terminal else now
        if created and end:
            try:
                start=datetime.fromisoformat(created.replace('Z','+00:00'))
                if start.tzinfo is None:start=start.replace(tzinfo=timezone.utc)
                finish=datetime.fromisoformat(end.replace('Z','+00:00')) if isinstance(end,str) else end
                if finish.tzinfo is None:finish=finish.replace(tzinfo=timezone.utc)
                seconds=int((finish-start).total_seconds())
                if seconds>=0:age=f'{seconds//60}m {seconds%60:02d}s'
            except (TypeError,ValueError):pass
        rows.append(dict(batch=key[:8],job=job.get('galaxy_job_id') or '—',state=state,
                         starts=len(chunk.get('members') or []),elapsed=age,
                         checked=job.get('updated_utc') or '—',created=created or ''))
    return sorted(rows,key=lambda r:(r['state']=='Results checked',r['created']),reverse=False)
