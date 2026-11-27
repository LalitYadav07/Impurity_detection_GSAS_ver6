"""Share only completed, identical HDA bytes inside one result transaction."""
from concurrent.futures import Future
import hashlib
from pathlib import Path
import shutil
import tempfile
import threading
import uuid

from .result_loading import measure


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class ResultTransfers:
    """No cross-job cache, no native artifact replacement, no partial reads."""

    MAX_FILE_BYTES = 64 * 1024**2
    MAX_TOTAL_BYTES = 96 * 1024**2
    SUBJECTS = frozenset({'summary', 'pattern_observation', 'plot_metadata',
                          'selected_plot_arrays', 'selected_plot_image', 'plots'})

    def __init__(self, root: Path, history_id: str, *, service=None):
        self.root = Path(tempfile.mkdtemp(prefix='.result-transfers-', dir=root))
        self.history_id = history_id
        self.service = service
        self.lock = threading.Lock()
        self.files = {}
        self.bytes = 0
        self.closed = False
        self.close_requested = False
        self.active = 0

    def download(self, identifier, destination, metadata, maximum_bytes, fetch, *, subject):
        """Return False when the ordinary downloader must handle this file."""
        if metadata.get('accessible') is False:
            raise ValueError('Galaxy dataset is not accessible')
        size = metadata.get('file_size')
        if (subject not in self.SUBJECTS or metadata.get('id') != identifier
                or metadata.get('history_id') != self.history_id
                or metadata.get('state') != 'ok' or metadata.get('deleted') or metadata.get('purged')
                or type(size) is not int or size < 0 or size > self.MAX_FILE_BYTES):
            return False
        if maximum_bytes is not None and size > maximum_bytes:
            raise ValueError('Galaxy dataset exceeds its download byte budget')
        # A changed known revision/size must never borrow an earlier transfer.
        identity = (identifier, self.history_id)
        revision = (metadata.get('update_time'), metadata.get('uuid'))
        with self.lock:
            if self.closed or self.close_requested:
                return False
            entry = self.files.get(identity)
            if entry is not None:
                known = entry['revision']
                if entry['size'] != size or any(a is not None and b is not None and a != b for a,b in zip(known,revision)):
                    raise ValueError('Galaxy dataset revision changed during result transfer')
                entry['revision'] = tuple(a if a is not None else b for a,b in zip(known,revision))
                owner = False
            else:
                if self.bytes + size > self.MAX_TOTAL_BYTES:
                    return False
                self.bytes += size
                entry = dict(future=Future(), revision=revision, size=size, path=self.root/uuid.uuid4().hex)
                self.files[identity] = entry
                owner = True
            self.active += 1
        try:
            return self._complete_download(entry, size, destination, fetch, owner)
        finally:
            with self.lock:
                self.active -= 1
                cleanup = self.close_requested and not self.active and not self.closed
                if cleanup:
                    self.closed = True
            if cleanup:
                shutil.rmtree(self.root, ignore_errors=True)

    def _complete_download(self, entry, size, destination, fetch, owner):
        if owner:
            try:
                fetch(entry['path'])
                if entry['path'].stat().st_size != size:
                    raise ValueError('Galaxy dataset size changed during result transfer')
                entry['sha256'] = _sha256(entry['path'])
            except BaseException as exc:
                entry['future'].set_exception(exc)
                try:
                    entry['path'].unlink(missing_ok=True)
                except OSError:
                    pass
                raise
            else:
                entry['future'].set_result(None)
        else:
            # The owner only performs this one download; it never waits for a
            # second shared file. Holding a transfer slot here cannot form a
            # dependency cycle. Failure propagates without a hidden retry.
            with measure('shared_transfer_wait'):
                entry['future'].result()
        target = Path(destination)
        temporary = target.with_name(target.name+'.'+uuid.uuid4().hex+'.partial')
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with measure('shared_transfer_copy') as timing:
                shutil.copyfile(entry['path'], temporary)
                if (temporary.stat().st_size != size
                        or _sha256(temporary) != entry['sha256']):
                    raise ValueError('Shared result transfer changed before copying')
                timing['bytes'] = size
            temporary.replace(target)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        return True

    def close(self):
        """Defer deletion until uncancellable worker threads finish copying."""
        with self.lock:
            self.close_requested = True
            if self.closed or self.active:
                return
            self.closed = True
        shutil.rmtree(self.root, ignore_errors=True)
