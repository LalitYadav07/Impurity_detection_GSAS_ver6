"""Display full source identity evidence without changing enrollment decisions."""
from copy import deepcopy


def identity_review_rows(discovered, selected=(), *, checked=False):
    """Keep excluded identities visible; selected rows only annotate the check.

    This function does not resolve an ID, alter source records, or authorize
    enrollment. Full-discovery conflicts remain visible and unresolved.
    """
    chosen = {str(row.get('run_id') or row.get('run_number')) for row in selected}
    result = []
    for original in discovered:
        row = deepcopy(original)
        key = str(row.get('run_id') or row.get('run_number'))
        if row.get('status') != 'resolved' or not row.get('sample_id'):
            row['selection_note'] = 'Excluded: unresolved identity · not enrolled'
        elif checked and key in chosen:
            row['selection_note'] = 'In checked selection'
        elif checked:
            row['selection_note'] = 'Outside checked selection'
        else:
            row['selection_note'] = 'Source identity evidence · eligibility not checked'
        result.append(row)
    return sorted(result, key=lambda row: (
        row.get('status') == 'resolved' and bool(row.get('sample_id')),
        str(row.get('run_id') or row.get('run_number')),
    ))
