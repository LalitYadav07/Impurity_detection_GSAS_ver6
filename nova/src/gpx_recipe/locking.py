"""One process-local and cross-process lock for an owned metadata file."""
from contextlib import contextmanager
import os
from pathlib import Path
import threading
import weakref

_LOCKS=weakref.WeakValueDictionary()
_GUARD=threading.Lock()


def _process_lock(path):
    # Some container-shared filesystems implement flock per process. Distinct
    # objects in one process must still serialize their read/modify/write.
    with _GUARD:
        lock=_LOCKS.get(path)
        if lock is None:lock=threading.RLock();_LOCKS[path]=lock
        return lock


def _after_fork():
    global _LOCKS,_GUARD
    _LOCKS=weakref.WeakValueDictionary();_GUARD=threading.Lock()


if hasattr(os,'register_at_fork'):os.register_at_fork(after_in_child=_after_fork)


@contextmanager
def process_lock(path):
    """Share the writer's reentrant lock with in-process metadata readers.

    A reader may run inside an existing file transaction. Do not acquire a
    second flock descriptor there: only the process-local lock is needed to
    keep the shared Python document from changing underneath its writer.
    """
    with _process_lock(Path(path).resolve()):
        yield


@contextmanager
def file_lock(path):
    import fcntl
    path=Path(path).resolve()
    with process_lock(path),path.open('a+b') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)
