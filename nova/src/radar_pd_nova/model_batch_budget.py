"""Host-side batching from verified output sizes and measured native cost."""
from collections import Counter
import json
import math

TARGET_BYTES = 128 * 1024 * 1024
TARGET_FILES = 8000


def concurrency_batch_limit(campaign, protocol, maximum_jobs):
    """Share a small finite tail across available, already approved job slots.

    Native CPU/RAM and output-size caps remain independent. When many scans
    are ready, retain a normal batch per scan; a lone 24-start control can use
    eight three-start jobs instead of occupying only three of eight slots.
    """
    if type(maximum_jobs) is not int or maximum_jobs < 1:
        raise ValueError('Use a positive configured Galaxy batch limit')
    with campaign.connect() as db:
        db.execute('BEGIN')
        prepared={row['protocol']:row['pending'] for row in db.execute(
            "SELECT protocol,COUNT(*) AS pending FROM trials WHERE state='prepared' GROUP BY protocol")}
        running=[dict(row) for row in db.execute(
            "SELECT id,protocol,handle FROM trials WHERE state IN ('running','submitting','cancelling')")]
    active={json.loads(row['handle']).get('dispatch_key',row['id']) if row.get('handle')
            else row['id'] for row in running}
    active_protocols={row['protocol'] for row in running}
    idle=set(prepared)-active_protocols
    competitors=max(1,len(idle or prepared))
    available=max(1,maximum_jobs-len(active))
    shares=max(1,math.ceil(available/competitors))
    return max(1,math.ceil(prepared.get(protocol,0)/shares))


def output_measurements(root, members):
    sizes, counts = Counter(), Counter()
    fixed_bytes = fixed_files = 0
    native = False
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        relative = path.relative_to(root).parts
        size = path.stat().st_size
        if len(relative) > 2 and relative[0] == 'trials' and relative[1] in members:
            sizes[relative[1]] += size
            counts[relative[1]] += 1
            native |= path.suffix.lower() == '.gpx' and size > 0
        else:
            fixed_bytes += size
            fixed_files += 1
    return dict(max_trial_bytes=max(sizes.values(), default=0),
        max_trial_files=max(counts.values(), default=0),
        fixed_bytes=fixed_bytes, fixed_files=fixed_files, native_artifacts=native)


def batch_size(source_bytes, chunks, *, maximum=128, profiled=False, prepared=False):
    """Keep the initial pilot small; learn only from this exact source protocol.

    A fourfold output reserve and a two-source-file floor leave room for other
    embeddings. The existing 512 MiB unpack bound remains independent. Nothing
    here raises worker counts or drops a reviewed scientific start.
    """
    initial = min(maximum, max(1, TARGET_BYTES // max(1, 8 * source_bytes)))
    results = [c for c in chunks if c.get('result')]
    if prepared:
        # A structural unit already owns a complete nonlinear model and all
        # its starts. Publish one model in the unknown-allocation pilot, then
        # at most one concurrent wave per measured native worker allocation.
        # Serial multi-model batches delay every completed fit until the
        # slowest model finishes, without amortizing a shared model kernel.
        workers=results[-1]['result'].get('allocation',{}).get('workers',1) if results else 1
        workers=workers if type(workers) is int and workers>0 else 1
        maximum=min(maximum,workers);initial=min(initial,maximum)
    if not results:
        return min(8, initial)
    measured = [c['output_measurements'] for c in results
        if c.get('output_measurements', {}).get('native_artifacts')]
    limit = initial
    if measured:
        per_trial = max(2 * source_bytes, 4 * max(m['max_trial_bytes'] for m in measured), 1)
        overhead = 4 * max(m['fixed_bytes'] for m in measured)
        file_cost = max(1, 4 * max(m['max_trial_files'] for m in measured))
        file_overhead = 4 * max(m['fixed_files'] for m in measured)
        limit = min(maximum, max(1, (TARGET_BYTES - overhead) // per_trial),
                    max(1, (TARGET_FILES - file_overhead) // file_cost))
    durations = [float(row['wall_seconds']) for c in results for row in c['result']['trials']
        if isinstance(row.get('wall_seconds'), (int, float))
        and not isinstance(row['wall_seconds'], bool)
        and math.isfinite(row['wall_seconds']) and row['wall_seconds'] > 0]
    if profiled:
        # A checked quadratic is transferred into later chunks. Its cold
        # assembly time is not paid again for every subsequent starting point.
        # Learn only from measured warm native runs, retaining output-size and
        # process-allocation limits and the chunk's independent soft deadline.
        warm=[float(row['wall_seconds']) for c in results for row in c['result']['trials']
            if row.get('profiled_setup',{}).get('cache_hit') is True
            and row.get('state')=='finished' and row.get('status')!='timed_out'
            and isinstance(row.get('wall_seconds'),(int,float)) and not isinstance(row['wall_seconds'],bool)
            and math.isfinite(row['wall_seconds']) and row['wall_seconds']>0]
        if warm:durations=warm
    if not durations:
        return limit
    durations.sort()
    p80 = durations[min(len(durations) - 1, math.ceil(.8 * len(durations)) - 1)]
    workers = results[-1]['result']['allocation']['workers']
    return min(limit, max(1, int(180 * workers / max(p80, .1))))
