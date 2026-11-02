"""Display recorded specimen identity without changing comparison grouping."""


def recorded_sample_title(metadata):
    sample_id = str(metadata.get('sample_id') or '').strip()
    sample_name = str(metadata.get('sample_name') or '').strip()
    title = f'Sample {sample_id}' if sample_id else sample_name or 'Sample identity unavailable'
    if sample_id and sample_name and sample_name != sample_id:
        title += ' · ' + sample_name
    return title
