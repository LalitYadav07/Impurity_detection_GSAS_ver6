"""Transport disposable, independently rechecked quadratic models between jobs.

These files are acceleration hints, never scientific results. Their namespace
binds the complete reviewed protocol and native independent basis. A worker
still compares the observed residual, basis maps and held-out native profiles.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import uuid
import zipfile

from .compiler import confined, file_sha
from .contracts import digest

SCHEMA = 'radar-profiled-cache/v1'
MAX_BYTES = 128 * 1024 * 1024
MEMBERS = {'identity.npy', 'offset.npy', 'maps.npy', 'nuclear.npy',
           'quadratic.npy', 'background.npy', 'linear_error.npy'}


def validate_manifest(document, protocol):
    if (not isinstance(document, dict) or set(document) != {'schema', 'protocol_sha256', 'entries'}
            or document['schema'] != SCHEMA or document['protocol_sha256'] != digest(protocol)):
        raise ValueError('Quadratic cache belongs to a different reviewed protocol')
    entries = document['entries']
    candidates = {c['id'] for c in protocol['candidates']}
    if not isinstance(entries, list) or len(entries) > len(candidates):
        raise ValueError('Invalid quadratic cache inventory')
    seen = set()
    for row in entries:
        if (not isinstance(row, dict) or set(row) != {'identity', 'candidate', 'path', 'sha256'}
                or not isinstance(row['identity'], str) or not re.fullmatch('[a-f0-9]{64}', row['identity'])
                or not isinstance(row['sha256'], str) or not re.fullmatch('[a-f0-9]{64}', row['sha256'])
                or not isinstance(row['candidate'],str) or row['candidate'] not in candidates or row['candidate'] in seen
                or row['path'] != '_profiled_cache/' + row['identity'] + '.npz'):
            raise ValueError('Invalid quadratic cache entry')
        seen.add(row['candidate'])
    return entries


def bounded_npz(path, identity):
    """Reject oversized/foreign/pickled archives before native array loading."""
    import numpy as np
    path = Path(path)
    try:
        if not path.is_file() or path.stat().st_size > MAX_BYTES:
            return False
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if ({m.filename for m in members} != MEMBERS or len(members) != len(MEMBERS)
                    or sum(m.file_size for m in members) > MAX_BYTES):
                return False
        with np.load(path, allow_pickle=False) as data:
            return data['identity'].shape == () and str(data['identity']) == identity
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError):
        return False


def restore(document, protocol, source_root, target_root):
    """Return imported entries and discarded hints; never reuse invalid bytes."""
    imported, discarded = [], []
    for row in validate_manifest(document, protocol):
        try:
            source = confined(source_root, row['path'])
            if file_sha(source) != row['sha256'] or not bounded_npz(source, row['identity']):
                raise ValueError('invalid cache bytes')
            target = confined(target_root, row['path'], exists=False)
            target.parent.mkdir(parents=True, exist_ok=True)
            if source != target:
                temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.partial')
                try:
                    shutil.copyfile(source, temporary)
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
            imported.append(dict(row))
        except (OSError, ValueError):
            discarded.append(row['candidate'])
    return imported, discarded


def export(protocol, expected, rows, shared_root, output_root):
    """Publish only models actually exercised by this chunk's native trials."""
    entries = {}
    total = 0
    for row in rows:
        proof = Path(output_root) / 'trials' / row['id'] / 'results' / 'optimizer-receipt.json'
        try:
            identity = json.loads(proof.read_text())['objective_identity']['cache_identity']
            candidate = expected[row['id']][0]
            if candidate in entries or not isinstance(identity, str) or not re.fullmatch('[a-f0-9]{64}', identity):
                continue
            relative = '_profiled_cache/' + identity + '.npz'
            source = confined(shared_root, relative)
            size = source.stat().st_size
            if total + size > MAX_BYTES or not bounded_npz(source, identity):
                continue
            target = confined(output_root, relative, exists=False)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            entries[candidate] = dict(identity=identity, candidate=candidate, path=relative, sha256=file_sha(target))
            total += size
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return dict(schema=SCHEMA, protocol_sha256=digest(protocol), entries=list(entries.values()))


def available(protocol, rows, root, candidates):
    document = dict(schema=SCHEMA, protocol_sha256=digest(protocol),
                    entries=[dict(r) for r in rows if r['candidate'] in candidates])
    good, _ = restore(document, protocol, root, root)
    return dict(document, entries=good)
