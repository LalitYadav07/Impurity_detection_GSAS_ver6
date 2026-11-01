"""Consistent model-review snapshots without rehashing a protocol per start."""
from collections import defaultdict
import json

from gpx_recipe.symmetry.contracts import digest, iter_trials, trial_id
from gpx_recipe.symmetry_results import summarize_portfolio


def declared_trial_ids(protocol, expected_sha):
    """Reuse one precomputed protocol hash for the native v1 start identities.

    Keep the native worker sealed for running campaigns. Check identity parity
    at both ends so an incompatible native contract fails explicitly.
    """
    if protocol.get('schema') != 'radar-symmetry-protocol/v1':
        raise ValueError('The native trial identity contract changed; use a matching worker release')
    if digest(protocol) != expected_sha:
        raise ValueError('The saved model protocol differs from its declared identity')
    starts = list(iter_trials(protocol))
    ids = [digest(dict(protocol=expected_sha, candidate=candidate['id'], seed=seed['id']))[:24]
           for candidate, seed in starts]
    for index in {0, len(starts)-1} if starts else ():
        candidate, seed = starts[index]
        if ids[index] != trial_id(protocol, candidate['id'], seed['id']):
            raise ValueError('The native trial identity contract changed; use a matching worker release')
    return ids


def portfolio_snapshot(campaign):
    """Protocols, starts and summary denominators share one SQLite read view."""
    with campaign.connect() as db:
        db.execute('BEGIN')
        stored_protocols = [dict(row) for row in db.execute('SELECT * FROM protocols ORDER BY rowid')]
        trials = [dict(row) for row in db.execute('SELECT * FROM trials ORDER BY rowid')]
    by_protocol = defaultdict(list)
    for trial in trials:
        by_protocol[trial['protocol']].append(trial)
    protocols, summaries = {}, []
    for stored in stored_protocols:
        protocol = json.loads(stored['body'])
        metadata = json.loads(stored['metadata'])
        sha = stored['hash']
        expected = declared_trial_ids(protocol, sha)
        members = by_protocol.get(sha, [])
        receipts = [json.loads(row['receipt']) for row in members if row['state'] == 'terminal']
        protocols[sha] = dict(protocol=protocol, metadata=metadata)
        summaries.append(dict(protocol=sha, metadata=metadata,
            pending_count=sum(row['state'] not in {'terminal', 'cancelled'} for row in members),
            **summarize_portfolio(expected, receipts, fit_band_percent=protocol['policy']['fit_band_percent'])))
    if set(by_protocol) - set(protocols):
        raise ValueError('A saved model trial has no declared protocol')
    return protocols, trials, summaries
