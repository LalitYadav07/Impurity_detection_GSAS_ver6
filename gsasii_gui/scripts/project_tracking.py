"""Preserve the exact project saved by GSAS-II, including Save As.

The native save callback selects the active path. Filesystem recency never
selects a project. pickletools checks GPX records without unpickling them.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickletools
import re
import shutil
import stat
import tempfile
import time
import uuid


def stamp(path):
    value = Path(path).lstat()
    if not stat.S_ISREG(value.st_mode) or value.st_size <= 0:
        raise ValueError("A saved project must be a nonempty regular GPX")
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


@contextmanager
def project_lock(session):
    root = Path(session)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'active-project.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, sort_keys=True)
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_gpx(path):
    count = 0
    with Path(path).open('rb') as stream:
        end = os.fstat(stream.fileno()).st_size
        while stream.tell() < end:
            last = None
            for opcode, _argument, _offset in pickletools.genops(stream):
                last = opcode.name
            if last != 'STOP':
                raise ValueError("Incomplete saved GPX record")
            count += 1
    if not count:
        raise ValueError("No saved GPX records")


def stable_copy(source, destination):
    """Publish only complete, stable regular GPX files."""
    source, destination = Path(source), Path(destination)
    if source.suffix.lower() != '.gpx':
        raise ValueError("The saved project must have a .gpx extension")
    before = stamp(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix='.' + destination.name + '.', dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, 'wb') as target:
            source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(source_fd, 'rb') as stream:
                opened = os.fstat(stream.fileno())
                if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) != before:
                    raise ValueError("The saved GPX changed before copying")
                shutil.copyfileobj(stream, target)
            target.flush()
            os.fsync(target.fileno())
        if stamp(source) != before:
            raise ValueError("The saved GPX changed while copying")
        validate_gpx(temporary)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        temporary.chmod(0o644)
        os.replace(temporary, destination)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


class ProjectTracker:
    def __init__(self, session, output, exports):
        self.session, self.output, self.exports = Path(session), Path(output), Path(exports)
        self.active = self.session / 'active-project.json'
        self.index = self.exports / 'Saved_projects' / 'index.json'
        self._last = None

    @classmethod
    def environment(cls):
        return cls(os.environ['GSASII_SESSION_DIR'], os.environ['GSASII_OUTPUT_PROJECT'],
                   os.environ['GSASII_EXPORT_DIR'])

    def activate(self, path):
        path = Path(os.path.abspath(path))
        stamp(path)
        if path.suffix.lower() != '.gpx':
            raise ValueError("Only a saved GPX can become the Galaxy project")
        validate_gpx(path)
        with project_lock(self.session):
            atomic_json(self.active, dict(schema='radar-gsasii-active-project/v1',
                                         path=str(path), save_id=uuid.uuid4().hex))
            self._sync()

    def sync(self):
        with project_lock(self.session):
            return self._sync()

    def _sync(self):
        document = self.active.read_bytes()
        active = json.loads(document)
        if (active.get('schema') != 'radar-gsasii-active-project/v1'
                or not isinstance(active.get('path'), str) or not Path(active['path']).is_absolute()
                or not isinstance(active.get('save_id'), str)):
            raise ValueError("The active saved-project record is invalid")
        source = Path(active['path'])
        current = (active['save_id'], stamp(source))
        if self._last == current:
            return False
        try:
            index = json.loads(self.index.read_text())
        except FileNotFoundError:
            index = dict(schema='radar-gsasii-saved-projects/v1', projects={})
        if index.get('schema') != 'radar-gsasii-saved-projects/v1' or not isinstance(index.get('projects'), dict):
            raise ValueError("The saved-project archive index is invalid")
        fd, name = tempfile.mkstemp(prefix='saved-project-', suffix='.gpx', dir=self.session)
        os.close(fd)
        snapshot = Path(name)
        try:
            digest = stable_copy(source, snapshot)
            if self.active.read_bytes() != document:
                raise ValueError("The active project changed while copying")
            key = hashlib.sha256(str(source).encode()).hexdigest()[:16]
            label = re.sub(r'[^A-Za-z0-9_.-]+', '_', source.stem)[:80].strip('._') or 'project'
            archive_name = label + '-' + key + '.gpx'
            stable_copy(snapshot, self.index.parent / archive_name)
            stable_copy(snapshot, self.output)
            index['projects'][key] = dict(file=archive_name, name=source.name, sha256=digest)
            index['active'] = key
            atomic_json(self.index, index)
            self._last = current
            print('Saved active GPX ' + source.name + ' (' + digest[:12] + ') to Galaxy', flush=True)
            return True
        finally:
            snapshot.unlink(missing_ok=True)


def install_save_tracking(io_module, tracker, *, report_error=None):
    """Observe the pinned native save function without changing its result."""
    if getattr(io_module.ProjFileSave, '_radar_tracking', False):
        return
    original = io_module.ProjFileSave

    def tracked_save(frame, *args, **kwargs):
        path = Path(os.path.abspath(frame.GSASprojectfile))
        try:
            before = stamp(path)
        except (OSError, ValueError):
            before = None
        result = original(frame, *args, **kwargs)
        try:
            if os.path.abspath(frame.GSASprojectfile) != str(path):
                raise ValueError("The project path changed during its native save")
            # PermissionError is handled inside native GSAS-II and returns
            # normally. An unchanged file does not prove that save succeeded.
            if stamp(path) != before:
                tracker.activate(path)
        except Exception as exc:
            print('Galaxy saved-project preservation failed: ' + str(exc), flush=True)
            if report_error:
                report_error(frame, 'The Galaxy output could not be updated from a verified saved GPX. '
                             'Keep this desktop open and save again. Details are in the session log.')
        return result

    tracked_save._radar_tracking = True
    io_module.ProjFileSave = tracked_save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['initialize', 'sync'])
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    tracker = ProjectTracker.environment()
    if args.operation == 'initialize':
        tracker.activate(os.environ['GSASII_PROJECT_PATH'])
        return
    while True:
        try:
            tracker.sync()
        except Exception as exc:
            if not args.watch:
                raise
            print('Saved-project sync deferred; retrying: ' + str(exc), flush=True)
        if not args.watch:
            return
        time.sleep(0.5)


if __name__ == '__main__':
    main()
