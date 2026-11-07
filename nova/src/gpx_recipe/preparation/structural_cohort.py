"""Bounded structural examples and independent, ambiguity-aware holdouts.

Checksums establish input identity, never scientific qualification. A cohort
is training material; it cannot authorize excluding a native candidate.
"""
from dataclasses import dataclass
import hashlib
import io
import copy
import json
from pathlib import Path

import numpy as np

from ..live.array_limits import validate_npz_bytes
from ..symmetry.contracts import validate_candidate
from .tof_kernel_arrays import read_bounded
from .structural_features import pattern_features, with_parent_reference, three_curve_features
from .structural_paired_blocks import PairedBlock
from .structural_sampling import partition_trajectories
from .structural_training_data import MAX_ROWS


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


class DenseFeatures:
    def __init__(self,channels=8):
        self.channels=channels

    def reserve(self, count, bins):
        if not 1 <= count <= MAX_ROWS or count*self.channels*bins*4 > MAX_FEATURE_BYTES:
            raise ValueError('Assemble structural feature cohorts within the declared memory budget')
        self.values = []

    def append(self, feature):
        self.values.append(feature)

    def finish(self, rows):
        return np.stack(self.values)


def assemble(blocks, *, bins=2048, split_seed=3917, augmenter=None):
    """Keep complete trajectories and exact aliases on one side of a split."""
    blocks = list(blocks)
    return assemble_into(lambda: iter(blocks), DenseFeatures(), bins=bins, split_seed=split_seed, augmenter=augmenter)


def assemble_pairs(blocks, *, bins=4096, split_seed=69271, augmenter=None):
    """Three-curve cohorts use checked parent contexts and class-supported folds."""
    blocks=list(blocks)
    return assemble_into(lambda:iter(blocks),DenseFeatures(3),bins=bins,split_seed=split_seed,
        augmenter=augmenter,paired=True)


def assemble_into(block_factory, sink, *, bins=2048, split_seed=3917, augmenter=None, paired=False):
    """One two-pass assembly algorithm for memory or worker-local file storage.

    Only one validated raw block and one computed feature need reside at a
    time. Exact aliases still merge labels and close whole trajectory groups
    before partitioning; storage must not change the scientific cohort.
    """
    if type(paired) is not bool or type(bins) is not int or not 64 <= bins <= 8192:
        raise ValueError('A bounded structural feature grid is required')
    sources = set(); profiles = set(); metadata = []; count = 0
    lower = -np.inf; upper = np.inf; parent_key = None; parent_arrays = None
    definitions = {}; candidate_groups = {}; generated_groups = {}
    for block in block_factory():
        if not isinstance(block, PairedBlock if paired else ExampleBlock):
            raise TypeError('Validated structural example blocks are required')
        d, a = block.manifest, block.arrays
        metadata.append(copy.deepcopy(d))
        if len(metadata) > 5000:
            raise ValueError('The structural preparation block catalog is bounded')
        sources.add(d['source_sha256']); profiles.add(d['instrument_sha256'])
        count += len(a['theory_intensity'])
        lower = max(lower, a['q'][0]); upper = min(upper, a['q'][-1])
        preferred = d['declared_child_group']['number'] in a['geometry_groups'][0]
        key = (not preferred, d['child_sha256'])
        if parent_key is None or key < parent_key:
            parent_key = key
            parent_arrays = (a['q'].copy(),
                (a['parent_intensity'][0] if paired else a['theory_intensity'][0]).copy(),
                (a['parent_background'][0] if paired else a['background']).copy())
        candidate = d['candidate']; identifier = candidate['id']
        if identifier in definitions and definitions[identifier] != candidate:
            raise ValueError('One structural candidate ID cannot describe different child embeddings')
        definitions[identifier] = copy.deepcopy(candidate)
        group = d['declared_child_group']['number']
        if identifier in candidate_groups and candidate_groups[identifier] != group:
            raise ValueError('A structural child cannot acquire conflicting declared symmetry labels')
        candidate_groups[identifier] = group
        generated_groups.setdefault(identifier,set()).update(map(int,np.unique(a['geometry_groups'])))
    if not metadata:
        raise TypeError('Validated structural example blocks are required')
    if len(sources) != 1 or len(profiles) != 1:
        raise ValueError('One exact parent and profile define a training context; fuse banks after ranking')
    if not lower < upper:
        raise ValueError('Structural examples have no common measured coordinate range')
    if augmenter is not None:
        from .structural_augmentation import ObservationAugmenter
        if (not isinstance(augmenter, ObservationAugmenter)
                or augmenter.metadata['instrument_sha256'] != next(iter(profiles))):
            raise ValueError('Structural augmentation must match the native example profile')
    sink.reserve(count*(1 if augmenter is None else 1+augmenter.controls.variants), bins)
    edges = np.linspace(lower, upper, bins+1)
    q, y, background = parent_arrays
    reference = (three_curve_features(q,y,y,edges,background=background,
        reference_background=background)['features'] if paired else
        pattern_features(q, y, edges, background=background))
    records = []; parents = {}; duplicates = {}; clean_origins = {}; augmented = 0

    def find(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]; group = parents[group]
        return group

    def union(first, second):
        first, second = find(first), find(second)
        if first != second:
            parents[max(first, second)] = min(first, second)

    def put(feature, origin, simulation):
        key = hashlib.sha256(feature.tobytes()).hexdigest()
        if key in duplicates:
            previous = duplicates[key]
            union(previous['group'], origin['group'])
            if not any(o is origin for o in previous['origins']):
                previous['origins'].append(origin)
            previous['simulations'].append(simulation)
        else:
            sink.append(feature)
            record = dict(origins=[origin], group=origin['group'], simulations=[simulation])
            duplicates[key] = record; records.append(record)

    seen = 0
    for block_index, block in enumerate(block_factory()):
        a, d = block.arrays, block.manifest
        if block_index >= len(metadata) or d != metadata[block_index]:
            raise ValueError('Structural inputs changed while building the feature cache')
        seen += 1
        for index, curve in enumerate(a['theory_intensity']):
            group = d['groups'][index]; parents.setdefault(group, group)
            def feature_for(values,background):
                if paired:
                    return three_curve_features(a['q'],values,a['parent_intensity'][index],edges,
                        background=background,reference_background=a['parent_background'][index])['features']
                return with_parent_reference(pattern_features(a['q'],values,edges,background=background),reference)
            feature = feature_for(curve,a['background'])
            key = hashlib.sha256(feature.tobytes()).hexdigest()
            labels = set(map(int, a['geometry_groups'][index])); ref = (block_index, index)
            if key in clean_origins:
                previous = clean_origins[key]
                union(previous['group'], group)
                previous['labels'].update(labels); previous['references'].append(ref)
                previous['metric_norm']=max(previous['metric_norm'],float(a['relative_metric_norm'][index]))
                # The shared clean origin carries every alias label and source
                # to all its noisy siblings, even aliases found in later blocks.
                continue
            origin = dict(labels=labels, group=group, references=[ref],metric_norm=float(a['relative_metric_norm'][index]))
            clean_origins[key] = origin
            put(feature, origin, dict(kind='clean_native'))
            if augmenter is not None:
                for observation in augmenter.observations(a['q'], curve, a['background'], clean_identity=key):
                    augmented += 1
                    augmented_feature = feature_for(observation['intensity'],observation['background'])
                    put(augmented_feature, origin, dict(kind='simulated_observation', **observation['provenance']))
    if seen != len(metadata):
        raise ValueError('Structural inputs changed while building the feature cache')
    classes = sorted(set().union(*(o['labels'] for r in records for o in r['origins'])))
    indices = {c: i for i, c in enumerate(classes)}
    targets = np.zeros((len(records), len(classes)), bool)
    for i, record in enumerate(records):
        targets[i, [indices[c] for o in record['origins'] for c in o['labels']]] = True
    physical_groups=[find(r['group']) for r in records]
    if paired:
        from .structural_partitions import partition
        assignments=partition(physical_groups,targets,seed=split_seed)
    else:
        assignments=partition_trajectories(physical_groups,seed=split_seed)
    split = np.asarray([{'train': 0, 'validation': 1, 'test': 2}[assignments[find(r['group'])]]
                        for r in records], np.int8)
    if not np.all(targets[split == 0].any(axis=0)):
        raise ValueError('Every prepared symmetry class needs independent training trajectories')
    from .structural_domain import envelope
    domain=envelope(max(o['metric_norm'] for i,r in enumerate(records) if split[i]==0 for o in r['origins']))
    return dict(features=sink.finish(len(records)), compatible_labels=targets,
        classes=np.asarray(classes, np.int32), split=split, q_edges=edges,
        groups=[find(r['group']) for r in records],
        references=[[ref for o in r['origins'] for ref in o['references']] for r in records],
        example_simulations=[r['simulations'] for r in records],
        augmentation=None if augmenter is None else augmenter.metadata,
        augmentation_sha256=None if augmenter is None else augmenter.sha256,
        augmented_examples=augmented,
        input_sha256=[d['pairs_sha256' if paired else 'examples_sha256'] for d in metadata],
        candidates=list(definitions.values()),
        candidate_groups=candidate_groups,
        class_candidate_ids={str(group):sorted(identifier for identifier in definitions
            if group in generated_groups[identifier] or candidate_groups[identifier]==group) for group in classes},
        reference_features=reference,
        reference_feature_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
        parent_sha256=next(iter(sources)), instrument_sha256=next(iter(profiles)),
        original_examples=count, unique_examples=len(records),
        ambiguous_examples=int(np.count_nonzero(targets.sum(axis=1) > 1)),
        training_domain=domain,screening_qualified=False,
        **(dict(channels=3,representation='reference-current-signed-difference/shared-Q-asinh/v1',
            reference_kind='calculated_parent_at_shared_parent_compatible_cell',
            partition_method='class-supported-whole-trajectories/v1') if paired else {}))
