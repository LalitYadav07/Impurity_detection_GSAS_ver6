"""Cancelable, read-only recovery for the model-results selector."""
from datetime import datetime, timezone
from gpx_recipe.symmetry.campaign import Campaign


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
