"""Bounded structural examples and independent, ambiguity-aware holdouts.

Checksums establish input identity, never scientific qualification. A cohort
is training material; it cannot authorize excluding a native candidate.
"""
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from ..live.array_limits import validate_npz_bytes
from ..symmetry.contracts import validate_candidate
from .tof_kernel_arrays import read_bounded
from .structural_features import pattern_features
from .structural_sampling import partition_trajectories


MAX_BLOCK_BYTES = 160*1024**2
MAX_FEATURE_BYTES = 384*1024**2
ARRAYS = {'q', 'theory_intensity', 'background', 'parameters', 'trajectory',
          'geometry_groups', 'ambiguous_geometry', 'displacement_rms', 'relative_metric_norm'}


@dataclass(frozen=True)
class ExampleBlock:
    manifest: dict
    arrays: dict


def read_block(directory):
    """Validate immutable numerical bytes before allocating their arrays."""
    directory = Path(directory)
    document = json.loads(read_bounded(directory/'examples.json', 8*1024**2))
    raw = read_bounded(directory/'examples.npz', MAX_BLOCK_BYTES)
    if (document.get('schema') != 'radar-structural-ranking-examples/v3'
            or document.get('observations_included') is not False
            or document.get('scientific_result') is not False
            or type(document.get('seed')) is not int or not 0 <= document['seed'] < 2**32
            or document.get('examples_sha256') != hashlib.sha256(raw).hexdigest()):
        raise ValueError('A bound observation-free structural example block is required')
    validate_npz_bytes(raw, maximum_bytes=MAX_BLOCK_BYTES)
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if set(archive.files) != ARRAYS:
            raise ValueError('Unexpected structural example array inventory')
        arrays = {k: archive[k] for k in ARRAYS}
    q, y = arrays['q'], arrays['theory_intensity']
    n = y.shape[0] if y.ndim == 2 else 0
    p = arrays['parameters']
    labels = arrays['geometry_groups']; trajectory = arrays['trajectory']
    groups = document.get('groups')
    if (q.ndim != 1 or not 3 <= len(q) <= 25001 or y.shape != (n, len(q)) or not 2 <= n <= 32769
            or arrays['background'].shape != q.shape or p.ndim != 2 or p.shape[0] != n
            or labels.shape != (n, 2) or labels.dtype.kind not in 'iu'
            or trajectory.shape != (n,) or trajectory.dtype.kind not in 'iu'
            or arrays['ambiguous_geometry'].shape != (n,) or arrays['ambiguous_geometry'].dtype != bool
            or any(arrays[k].shape != (n,) for k in ('displacement_rms', 'relative_metric_norm'))
            or any(a.dtype.kind not in 'fiub' or not np.isfinite(a).all() for a in arrays.values())
            or np.any(q <= 0) or np.any(np.diff(q) <= 0)
            or np.any(labels < 1) or np.any(labels > 230) or trajectory[0] != -1
            or np.any(trajectory[1:] < 0) or np.any(p[0] != 0)
            or any(np.any(arrays[k] < 0) for k in ('displacement_rms', 'relative_metric_norm'))
            or not np.array_equal(arrays['ambiguous_geometry'], labels[:, 0] != labels[:, 1])
            or document.get('examples') != n or document.get('bins') != len(q)
            or not isinstance(groups, list) or len(groups) != n
            or any(not isinstance(g, str) or not 1 <= len(g) <= 500 for g in groups)):
        raise ValueError('Invalid physical structural example dimensions or labels')
    source = document.get('source_sha256'); instrument = document.get('instrument_sha256')
    if any(not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v)
           for v in (source, instrument, document.get('child_sha256'))):
        raise ValueError('Structural examples require exact parent, child and profile identities')
    expected = ['parent:'+source] + [f'parent:{source}:{document["seed"]}:trajectory-{i}' for i in trajectory[1:]]
    if groups != expected:
        raise ValueError('Trajectory groups must be shared across child aliases and bank variants')
    candidate = validate_candidate(document.get('candidate'))
    if not candidate['child'] or candidate['child']['sha256'] != document['child_sha256']:
        raise ValueError('Structural examples must retain their exact native child identity')
    document['candidate'] = candidate
    declared = document.get('declared_child_group')
    if not isinstance(declared, dict) or type(declared.get('number')) is not int or not 1 <= declared['number'] <= 230:
        raise ValueError('Structural examples must retain their declared child symmetry')
    arrays = {k: np.frombuffer(v.tobytes(), dtype=v.dtype).reshape(v.shape) for k, v in arrays.items()}
    return ExampleBlock(document, arrays)


def assemble(blocks, *, bins=2048, split_seed=3917):
    """Keep complete trajectories and all exact aliases on one side of a split.

    A physically ambiguous example has several compatible symmetry labels.
    Identical input features with different labels retain their label union,
    rather than discarding inconvenient examples or inventing a unique label.
    """
    blocks = list(blocks)
    if not blocks or any(not isinstance(b, ExampleBlock) for b in blocks):
        raise TypeError('Validated structural example blocks are required')
    if type(bins) is not int or not 64 <= bins <= 8192:
        raise ValueError('A bounded structural feature grid is required')
    sources = {b.manifest['source_sha256'] for b in blocks}
    profiles = {b.manifest['instrument_sha256'] for b in blocks}
    if len(sources) != 1 or len(profiles) != 1:
        raise ValueError('One exact parent and profile define a training context; fuse banks after ranking')
    count = sum(len(b.arrays['theory_intensity']) for b in blocks)
    if count*4*bins*4 > MAX_FEATURE_BYTES:
        raise ValueError('Assemble structural feature cohorts within the declared memory budget')
    lower = max(b.arrays['q'][0] for b in blocks)
    upper = min(b.arrays['q'][-1] for b in blocks)
    if not lower < upper:
        raise ValueError('Structural examples have no common measured coordinate range')
    edges = np.linspace(lower, upper, bins+1)
    records = []; parents = {}; duplicates = {}

    def find(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]; group = parents[group]
        return group

    def union(first, second):
        first, second = find(first), find(second)
        if first != second:
            parents[max(first, second)] = min(first, second)

    for block_index, block in enumerate(blocks):
        a, d = block.arrays, block.manifest
        for index, curve in enumerate(a['theory_intensity']):
            group = d['groups'][index]; parents.setdefault(group, group)
            feature = pattern_features(a['q'], curve, edges, background=a['background'])
            key = hashlib.sha256(feature.tobytes()).hexdigest()
            labels = set(map(int, a['geometry_groups'][index]))
            ref = (block_index, index)
            if key in duplicates:
                previous = duplicates[key]
                union(previous['group'], group)
                previous['labels'].update(labels); previous['references'].append(ref)
            else:
                record = dict(feature=feature, labels=labels, group=group, references=[ref])
                duplicates[key] = record; records.append(record)
    assignments = partition_trajectories([find(r['group']) for r in records], seed=split_seed)
    classes = sorted(set().union(*(r['labels'] for r in records)))
    indices = {c: i for i, c in enumerate(classes)}
    targets = np.zeros((len(records), len(classes)), bool)
    for i, record in enumerate(records):
        targets[i, [indices[c] for c in record['labels']]] = True
    split = np.asarray([{'train': 0, 'validation': 1, 'test': 2}[assignments[find(r['group'])]]
                        for r in records], np.int8)
    if not np.all(targets[split == 0].any(axis=0)):
        raise ValueError('Every prepared symmetry class needs independent training trajectories')
    definitions = {}; candidate_groups = {}
    for block in blocks:
        candidate = block.manifest['candidate']; key = candidate['id']
        if key in definitions and definitions[key] != candidate:
            raise ValueError('One structural candidate ID cannot describe different child embeddings')
        definitions[key] = candidate
        group = block.manifest['declared_child_group']['number']
        if key in candidate_groups and candidate_groups[key] != group:
            raise ValueError('A structural child cannot acquire conflicting declared symmetry labels')
        candidate_groups[key] = group
    return dict(features=np.stack([r['feature'] for r in records]), compatible_labels=targets,
        classes=np.asarray(classes, np.int32), split=split, q_edges=edges,
        groups=[find(r['group']) for r in records],
        references=[r['references'] for r in records],
        input_sha256=[b.manifest['examples_sha256'] for b in blocks],
        candidates=list(definitions.values()),
        candidate_groups=candidate_groups,
        parent_sha256=next(iter(sources)), instrument_sha256=next(iter(profiles)),
        original_examples=count, unique_examples=len(records),
        ambiguous_examples=int(np.count_nonzero(targets.sum(axis=1) > 1)),
        screening_qualified=False)
