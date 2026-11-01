"""Cancelable, read-only recovery for the model-results selector."""
from datetime import datetime, timezone
import json
from gpx_recipe.symmetry.campaign import Campaign
from gpx_recipe.symmetry.contracts import digest
from .model_navigation import scan_label


class ReviewSelectionChanged(RuntimeError):
    pass


class SelectionRecovery:
    """Stop obsolete artifact reads at a trial boundary, never scientific jobs.

    Deliberately exposes no dispatch/cancel methods. The campaign driver keeps
    its independent transport and is unaffected by changing a review selector.
    """
    def __init__(self, transport, cancelled, *, progress=None):
        self.transport, self.cancelled = transport, cancelled
        self.progress, self._seen = progress, set()

    def recover(self, key, directory):
        if self.cancelled.is_set():
            raise ReviewSelectionChanged()
        # One native result archive can contain many starts. Count result
        # groups rather than suggesting that this is the number of fits read.
        state = getattr(self.transport, 'state', {})
        group = state.get('members', {}).get(key, key)
        if group not in self._seen:
            self._seen.add(group)
            if self.progress:
                self.progress(len(self._seen))
        return self.transport.recover(key, directory)


def acquisition_index(campaign,*,experiment=None):
    """Read source identities from sealed plans, before fetching result files.

    These are navigation labels, never verified fit statuses or residuals.
    The full finite ledger and all original trial identities stay intact.
    """
    options={};protocols={}
    with campaign.connect() as db:
        stored=[dict(row) for row in db.execute('SELECT * FROM protocols ORDER BY rowid')]
    for row in stored:
        protocol=json.loads(row['body']);metadata=json.loads(row['metadata'])
        if digest(protocol)!=row['hash']:raise ValueError('The saved acquisition protocol changed')
        if experiment is not None and metadata.get('series')!=experiment:continue
        source=protocol.get('acquisition_id') or protocol['source']['sha256']
        options.setdefault(source,scan_label(metadata,protocol['parent_phase']))
        protocols.setdefault(source,set()).add(row['hash'])
    return options,protocols


class AcquisitionReview:
    """Restrict read-only recovery using the original sealed recovery method.

    No scheduler, submit, resume or cancellation method is exposed. Collection
    retains the same receipt/artifact checks and the same campaign lock.
    """
    def __init__(self,campaign,protocols):
        self.campaign=campaign;self.protocols=frozenset(protocols)
        if not self.protocols:raise ValueError('Select an exact acquisition before recovering its results')

    def trials(self):return [row for row in self.campaign.trials() if row['protocol'] in self.protocols]
    def exclusive(self):return self.campaign.exclusive()
    def trial_directory(self,row):return self.campaign.trial_directory(row)
    def setting(self,*args,**kwargs):return self.campaign.setting(*args,**kwargs)
    def collect(self,row):return self.campaign.collect(row)
    def set_state(self,*args,**kwargs):return self.campaign.set_state(*args,**kwargs)
    def recover(self,transport):return Campaign.recover(self,transport)


def campaign_recency(path):
    """Sort by the saved checkpoint clock, never a random campaign hash."""
    fallback = path.stat().st_mtime
    value = Campaign(path.parent).setting('checkpoint_updated_utc', '')
    try:
        stamp = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp.timestamp(), path.parent.name
    except (TypeError, ValueError, OverflowError):
        return fallback, path.parent.name
