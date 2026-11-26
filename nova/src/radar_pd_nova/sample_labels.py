"""Display recorded specimen identity without changing comparison grouping."""

from collections import defaultdict


def with_recorded_run_context(options, run_options):
    """Add menu subtitles for reused names using already available run titles.

    These are examples of recorded labels, not inferred material identities.
    Keep the option title and value unchanged: the guided assistant parses the
    title, and the value remains the only specimen-selection/routing key.
    """
    named_ids = defaultdict(set)
    for option in options:
        for name in option.get('sample_names', ()):
            if str(name).strip():
                named_ids[str(name).strip().casefold()].add(option['value'])
    duplicated = set().union(*(ids for ids in named_ids.values() if len(ids) > 1))
    if not duplicated:
        return options
    examples = defaultdict(list)
    for run in run_options:
        sample_id = run.get('sample_id')
        title = ' '.join(str(run.get('run_title') or '').split())
        if sample_id not in duplicated or not title:
            continue
        rows = examples[sample_id]
        if not any(item['run_title'] == title for item in rows):
            rows.append({'run': run['value'], 'run_id': run.get('run_id') or f"Run {run['value']}",
                         'run_title': title})
    result = []
    for option in options:
        rows = examples.get(option['value'], [])
        if not rows:
            result.append(option)
            continue
        # First and last distinct labels can expose a changed acquisition
        # context without expanding a long series into a second run browser.
        shown = rows if len(rows) <= 2 else [rows[0], rows[-1]]
        text = '; '.join(f"{row['run_id']}: {row['run_title']}" for row in shown)
        if len(text) > 440:
            text = text[:437] + '…'
        context = 'Recorded title examples · ' + text
        result.append({**option, 'run_title_examples': shown,
                       'props': {**option.get('props', {}), 'subtitle': context, 'lines': 'three'}})
    return result


def recorded_sample_title(metadata):
    sample_id = str(metadata.get('sample_id') or '').strip()
    sample_name = str(metadata.get('sample_name') or '').strip()
    title = f'Sample {sample_id}' if sample_id else sample_name or 'Sample identity unavailable'
    if sample_id and sample_name and sample_name != sample_id:
        title += ' · ' + sample_name
    return title
