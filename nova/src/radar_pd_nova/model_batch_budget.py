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
        fixed_bytes=fixed_bytes, fixed_files=fixed_files, native_artifacts=native,
        total_bytes=sum(sizes.values())+fixed_bytes,
        total_files=sum(counts.values())+fixed_files)


def complete_model_measurements(root, protocol, members, response):
    """Read optional timing evidence for complete, natively checked portfolios.

    A cheap unsupported model or a partially returned portfolio cannot teach
    the dispatcher how long a complete scientific calculation takes. These
    measurements affect batching only; missing/invalid evidence never rejects
    an otherwise valid scientific receipt.
    """
    from gpx_recipe.symmetry.compiler import confined
    from gpx_recipe.symmetry.contracts import digest, iter_trials, trial_id
    if not protocol.get('policy', {}).get('grouped_surrogate'):
        return []
    sha = digest(protocol)
    if response.get('protocol_sha256') != sha:
        return []
    groups = {}
    for candidate, seed in iter_trials(protocol):
        groups.setdefault(candidate['id'], set()).add(trial_id(protocol, candidate['id'], seed['id']))
    completed = {row['id'] for row in response.get('trials', []) if row.get('state') == 'finished'}
    result = []
    for candidate, ids in groups.items():
        if not ids <= set(members) or not ids <= completed:
            continue
        try:
            paths = [confined(root, f'trials/{uid}/group-data/summary.json', exists=False) for uid in sorted(ids)]
            summaries = [path for path in paths if path.is_file()]
            if len(summaries) != 1 or summaries[0].stat().st_size > 65536:
                continue
            summary = json.loads(summaries[0].read_text())
            seconds = summary.get('wall_seconds')
            workers = summary.get('allocation', {}).get('representative_workers')
            if (summary.get('schema') != 'radar-profiled-group-result/v1'
                    or summary.get('protocol_sha256') != sha or summary.get('candidate_id') != candidate
                    or summary.get('declared') != len(ids) or summary.get('screened') != len(ids)
                    or type(summary.get('native_verified_profiles')) is not int
                    or summary['native_verified_profiles'] < 1
                    or type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0
                    or type(workers) is not int or not 1 <= workers <= 16):
                continue
            size = count = 0
            native = False
            for uid in ids:
                for path in confined(root, f'trials/{uid}').rglob('*'):
                    if path.is_file():
                        value = path.stat().st_size
                        size += value
                        count += 1
                        native |= path.suffix.lower() == '.gpx' and value > 0
            if native:
                result.append(dict(candidate=candidate, declared=len(ids), wall_seconds=seconds,
                    representative_workers=workers, output_bytes=size, output_files=count))
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return result


def model_group_count(source_bytes, chunks, *, maximum=8):
    """Amortize launch/transfer overhead with complete magnetic models.

    Start with one model. Later batches retain a fourfold artifact reserve and
    target three minutes of measured native work. Account for the workers used
    inside the pilot model before estimating concurrency across several models.
    Visible host cores never increase the granted worker budget.
    """
    results = [c for c in chunks if c.get('result')]
    measured = []
    for chunk in results:
        groups = chunk.get('group_measurements', [])
        if not isinstance(groups, list) or not groups or not isinstance(chunk.get('output_measurements'), dict):
            continue
        wall = chunk['result'].get('wall_seconds')
        workers = chunk['result'].get('allocation', {}).get('workers', 1)
        if (type(workers) is not int or not 1 <= workers <= 16
                or type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0):
            continue
        for row in groups:
            if not isinstance(row, dict):
                continue
            values = [row.get(key) for key in ('wall_seconds', 'representative_workers', 'output_bytes', 'output_files')]
            if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
                continue
            if type(row['representative_workers']) is not int or row['representative_workers'] > workers:
                continue
            used = min(workers, len(groups) * row['representative_workers'])
            measured.append((row, wall * used / len(groups), chunk['output_measurements']))
    if not measured:
        return 1
    granted = results[-1]['result'].get('allocation', {}).get('workers', 1)
    granted = granted if type(granted) is int and 1 <= granted <= 16 else 1
    seconds = max(max(row['wall_seconds'] * row['representative_workers'], overhead)
                  for row, overhead, _ in measured)
    size = max(2 * source_bytes, 4 * max(row['output_bytes'] for row, _, _ in measured), 1)
    files = max(1, 4 * max(row['output_files'] for row, _, _ in measured))
    fixed_size = 4 * max(m.get('fixed_bytes', 0) for _, _, m in measured)
    fixed_files = 4 * max(m.get('fixed_files', 0) for _, _, m in measured)
    return max(1, min(maximum, int(180 * granted / seconds),
        int((TARGET_BYTES - fixed_size) // size), int((TARGET_FILES - fixed_files) // files)))


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
