"""Compound-specific CNN ranking with data-only portable weights.

Ranking is evidence for ordering native work. This module never grants a
saved model permission to discard candidates or assert a unique space group.
Training runs in its own allocated worker process, not the interactive UI.
"""
import copy
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np

from ..execution import write_json
from ..live.array_limits import validate_npz_bytes
from ..symmetry.resources import allocation
from ..symmetry.contracts import validate_candidate
from .structural_features import pattern_features, with_parent_reference
from .structural_support import fit_batches as fit_support, distances as support_distances, MAX_BYTES as SUPPORT_BYTES, STREAM_METHOD
from .structural_training_data import feature_rows, batches, channel_statistics
from .tof_kernel_arrays import read_bounded, checksum

SCHEMA = 'radar-structural-ranking-model/v4'
ARCHITECTURE = 'parent-q-cnn-8-16-24-32-pool64/v2'
MAX_BYTES = 32*1024**2


def network(class_count):
    from torch import nn
    return nn.Sequential(nn.Conv1d(8, 16, 9, stride=1, padding=4), nn.ReLU(),
        nn.Conv1d(16, 24, 7, stride=2, padding=3), nn.ReLU(),
        nn.Conv1d(24, 32, 5, stride=2, padding=2), nn.ReLU(),
        nn.AdaptiveAvgPool1d(64), nn.Flatten(), nn.Linear(32*64, 64), nn.ReLU(),
        nn.Dropout(.1), nn.Linear(64, class_count))


def _score(logits, compatible):
    """A symmetry-ambiguous example accepts any of its compatible labels."""
    import torch
    order = logits.argsort(dim=1, descending=True)
    top = compatible.gather(1, order)
    loss = torch.logsumexp(logits, 1)-torch.logsumexp(logits.masked_fill(~compatible, -torch.inf), 1)
    return dict(top1=float(top[:, 0].float().mean()),
        top5=float(top[:, :min(5, logits.shape[1])].any(1).float().mean()),
        loss=float(loss.mean()))



def _score_batches(model, rows, targets, indices, mean, scale):
    import torch
    totals = dict(top1=0., top5=0., loss=0.); count = 0
    for selected, values in batches(rows, indices):
        logits = model(torch.from_numpy((values-mean)/scale))
        result = _score(logits, torch.from_numpy(targets[selected]))
        for key in totals:
            totals[key] += result[key]*len(selected)
        count += len(selected)
    if not count:
        raise ValueError('Validation and test partitions must contain independent examples')
    return {key: value/count for key, value in totals.items()}


def _class_routes(classes, groups, supplied=None):
    """A geometry label must lead to a complete nonempty native work set.

    A construction can produce several higher-symmetry special states. Its
    declared SG number alone cannot route all of those generated examples.
    Legacy models may derive routes only when every class has a declared
    construction; new models retain the generated-state evidence explicitly.
    """
    routes=({str(group):sorted(key for key,value in groups.items() if value==group)
             for group in classes} if supplied is None else supplied)
    identifiers=set(groups)
    if (not isinstance(routes,dict) or set(routes)!={str(group) for group in classes}
            or any(not isinstance(value,list) or not value
                   or any(not isinstance(key,str) or key not in identifiers for key in value)
                   or len(value)!=len(set(value))
                   for value in routes.values())
            or set().union(*(set(value) for value in routes.values()))!=identifiers):
        raise ValueError('Every trained symmetry class needs complete nonempty native candidate routes')
    return routes


def train(cohort, destination, *, epochs=60, seed=89113, threads=4):
    import torch
    if type(epochs) is not int or not 1 <= epochs <= 1000 or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Bounded training epochs and random seed are required')
    classes = np.asarray(cohort['classes'])
    features = feature_rows(cohort['features']); targets = np.asarray(cohort['compatible_labels'])
    split = np.asarray(cohort['split'])
    if (not 2 <= len(classes) <= 230 or features.ndim != 3 or features.shape[1] != 8
            or features.dtype != np.float32
            or targets.shape != (len(features), len(classes)) or targets.dtype != bool
            or not targets.any(axis=1).all() or split.shape != (len(features),)
            or set(split) != {0, 1, 2}):
        raise ValueError('A complete finite cohort with independent holdouts is required')
    routes=_class_routes(classes,cohort['candidate_groups'],cohort.get('class_candidate_ids'))
    reference = np.asarray(cohort['reference_features']); edges = np.asarray(cohort['q_edges'])
    if (edges.dtype != np.float64 or edges.shape != (features.shape[2]+1,)
            or not np.isfinite(edges).all() or edges[0] <= 0 or np.any(np.diff(edges) <= 0)
            or reference.shape != (4, features.shape[2]) or reference.dtype != np.float32
            or not np.isfinite(reference).all()
            or hashlib.sha256(reference.tobytes()).hexdigest() != cohort['reference_feature_sha256']):
        raise ValueError('Training needs its exact finite parent reference and coordinate grid')
    with_parent_reference(features.take(np.asarray([0], np.int64))[0, :4], reference)
    # The cohort assembler seals groups after duplicate closure. Verify that
    # no caller subsequently reassigned individual steps or candidate aliases.
    groups = cohort['groups']
    assignments = {}
    for group, code in zip(groups, split):
        assignments.setdefault(group, set()).add(int(code))
    if len(groups) != len(split) or any(len(codes) != 1 for codes in assignments.values()):
        raise ValueError('Physical trajectory groups cannot cross training holdouts')
    budget = allocation(threads)
    torch.set_num_threads(budget['workers'])
    began = time.monotonic(); out = Path(destination); out.mkdir(parents=True, exist_ok=False)
    train_rows, validation_rows, test_rows = (np.flatnonzero(split == code) for code in (0, 1, 2))
    # Global channel normalization preserves the parent peak reference.
    # Per-bin centering would turn a constant reference into zeros and make
    # the residual merely a duplicate of the centered observation channel.
    mean, scale = channel_statistics(features, train_rows)
    y = torch.from_numpy(targets)
    rng = np.random.default_rng(seed); indices = train_rows
    counts = targets[train_rows].sum(0)
    if not counts.all():
        raise ValueError('Each catalog class requires independent training examples')
    weights = counts.sum()/counts
    weights = torch.tensor(weights/weights.mean(), dtype=torch.float32)
    best = None; best_key = None; best_epoch = None; history = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed); model = network(len(classes))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        for epoch in range(epochs):
            model.train(); order = rng.permutation(indices)
            for selected, values in batches(features, order, size=32):
                logits = model(torch.from_numpy((values-mean)/scale)); allowed = y[selected]
                loss = torch.logsumexp(logits, 1)-torch.logsumexp(logits.masked_fill(~allowed, -torch.inf), 1)
                row_weights = (allowed*weights).sum(1)/allowed.sum(1)
                objective = (loss*row_weights).mean()
                if not torch.isfinite(objective):
                    raise ValueError('Structural training produced a nonfinite objective; no model was published')
                optimizer.zero_grad(set_to_none=True); objective.backward(); optimizer.step()
            model.eval()
            with torch.no_grad():
                result = _score_batches(model, features, targets, validation_rows, mean, scale)
            history.append(dict(epoch=epoch+1, validation=result))
            key = (-result['top5'], result['loss'])
            if best_key is None or key < best_key:
                best_key = key; best_epoch = epoch+1; best = copy.deepcopy(model.state_dict())
            write_json(out/'progress.json', dict(epoch=epoch+1, epochs=epochs, validation=result))
        model.load_state_dict(best); model.eval()
        support = fit_support(features, train_rows, validation_rows, targets, mean, scale, seed=seed)
        # This holdout is inspected only after validation selected the frozen
        # checkpoint. Subsequent development must use a fresh external test.
        with torch.no_grad():
            score = _score_batches(model, features, targets, test_rows, mean, scale)
    arrays = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    # Keep the generic sixteen-array transport bound. Eight exactly
    # representable float32 scales and the float64 support limit share one
    # typed numerical plane; no model value or bound is discarded.
    normalization = np.r_[scale.ravel().astype(np.float64), support['limit']]
    arrays.update(q_edges=cohort['q_edges'], feature_mean=mean,
                  feature_scale_and_support_limit=normalization, classes=classes,
                  reference_features=cohort['reference_features'],
                  support_prototypes=support['prototypes'])
    stream = io.BytesIO(); np.savez_compressed(stream, **arrays); raw = stream.getvalue()
    validate_npz_bytes(raw, maximum_bytes=MAX_BYTES); (out/'weights.npz').write_bytes(raw)
    counts = {name: targets[split == code].sum(0).tolist()
              for name, code in [('train', 0), ('validation', 1), ('test', 2)]}
    metadata = dict(schema=SCHEMA, architecture=ARCHITECTURE, parent_sha256=cohort['parent_sha256'],
        instrument_sha256=cohort['instrument_sha256'], input_sha256=cohort['input_sha256'],
        candidates=cohort['candidates'], candidate_groups=cohort['candidate_groups'],
        class_candidate_ids=routes,
        weights_sha256=hashlib.sha256(raw).hexdigest(), classes=classes.tolist(), bins=features.shape[2],
        epochs=epochs, seed=seed, allocation=budget, counts=counts, test=score,
        feature_storage=features.storage, support_training_pool_rows=support['training_pool_rows'],
        augmentation=cohort.get('augmentation'), augmentation_sha256=cohort.get('augmentation_sha256'),
        augmented_examples=cohort.get('augmented_examples', 0),
        checkpoint_selection='validation_top_five_recall_then_loss', selected_epoch=best_epoch,
        reference_feature_sha256=cohort['reference_feature_sha256'],
        support_method=support['method'], support_validation_rows=support['validation_rows'],
        support_coverage_qualified=False,
        ambiguous_examples=cohort['ambiguous_examples'], examples=len(features),
        original_examples=cohort['original_examples'], independent_groups=len(set(groups)),
        background_required=True, screening_qualified=False, scientific_result=False,
        seconds=time.monotonic()-began, history=history)
    write_json(out/'model.json', metadata)
    return metadata


class Ranker:
    """A validated local CNN can order all families; it cannot prune them."""
    def __init__(self, directory, *, threads=1):
        import torch
        self.allocation = allocation(threads)
        torch.set_num_threads(self.allocation['workers'])
        directory = Path(directory)
        d = json.loads(read_bounded(directory/'model.json', 2*1024**2))
        raw = read_bounded(directory/'weights.npz', MAX_BYTES)
        classes = d.get('classes'); bins = d.get('bins')
        if (d.get('schema') not in {SCHEMA,'radar-structural-ranking-model/v3'} or d.get('architecture') != ARCHITECTURE
                or d.get('weights_sha256') != hashlib.sha256(raw).hexdigest()
                or not isinstance(classes, list) or not 2 <= len(classes) <= 230
                or any(type(c) is not int or not 1 <= c <= 230 for c in classes)
                or classes != sorted(set(classes)) or type(bins) is not int or not 64 <= bins <= 8192
                or d.get('background_required') is not True):
            raise ValueError('Unsupported or changed structural ranking model')
        candidates = d.get('candidates'); groups = d.get('candidate_groups')
        if (not isinstance(candidates, list) or not 1 <= len(candidates) <= 5000
                or not isinstance(groups, dict)
                or any(not checksum(d.get(k)) for k in ('parent_sha256', 'instrument_sha256'))):
            raise ValueError('A structural model needs its exact candidate and input identities')
        candidates = [validate_candidate(c) for c in candidates]
        identifiers = [c['id'] for c in candidates]
        if (len(identifiers) != len(set(identifiers)) or set(groups) != set(identifiers)
                or any(type(v) is not int or not 1 <= v <= 230 for v in groups.values())
                or any(not c['child'] or c['magnetic_space_group'] is not None for c in candidates)):
            raise ValueError('A structural model needs unambiguous child-to-family mappings')
        if d['schema']==SCHEMA and 'class_candidate_ids' not in d:
            raise ValueError('Structural ranking models require their native candidate routes')
        routes=_class_routes(classes,groups,d.get('class_candidate_ids'))
        validate_npz_bytes(raw, maximum_bytes=MAX_BYTES)
        model = network(len(classes)); expected = model.state_dict()
        with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
            if set(saved.files) != set(expected)|{'q_edges', 'feature_mean', 'feature_scale_and_support_limit',
                                                 'classes', 'reference_features', 'support_prototypes'}:
                raise ValueError('Unexpected structural model tensor inventory')
            values = {k: saved[k] for k in saved.files}
        for name, tensor in expected.items():
            a = values[name]
            if a.shape != tuple(tensor.shape) or a.dtype != np.float32 or not np.isfinite(a).all():
                raise ValueError('Invalid structural model tensor dimensions or dtype')
        edges, mean = (values[k] for k in ('q_edges', 'feature_mean'))
        normalization = values['feature_scale_and_support_limit']
        if normalization.dtype != np.float64 or normalization.shape != (9,) or not np.isfinite(normalization).all():
            raise ValueError('Invalid structural normalization and support plane')
        with np.errstate(over='ignore', under='ignore'):
            scale = normalization[:8].astype(np.float32).reshape(8, 1)
        reference = values['reference_features']
        if (edges.dtype != np.float64 or mean.dtype != np.float32 or scale.dtype != np.float32
                or values['classes'].dtype != np.int32
                or edges.shape != (bins+1,) or not np.isfinite(edges).all() or edges[0] <= 0
                or np.any(np.diff(edges) <= 0)
                or mean.shape != (8, 1) or scale.shape != mean.shape
                or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0)
                or not np.array_equal(normalization[:8], scale.ravel().astype(np.float64))
                or reference.dtype != np.float32 or reference.shape != (4, bins)
                or not np.isfinite(reference).all()
                or np.any(reference[2] < 0) or np.any(reference[2] > 1)
                or hashlib.sha256(reference.tobytes()).hexdigest() != d.get('reference_feature_sha256')
                or not np.array_equal(values['classes'], classes)):
            raise ValueError('Invalid structural feature normalization')
        coordinate = (((edges[:-1]+edges[1:])*.5-edges[0])/(edges[-1]-edges[0])).astype(np.float32)
        if not np.array_equal(reference[3], coordinate):
            raise ValueError('The parent reference does not use the model coordinate grid')
        prototypes, limit = values['support_prototypes'], normalization[8]
        if (d.get('support_method') not in {'training-k-center/validation-maximum-rms/v1', STREAM_METHOD}
                or prototypes.dtype != np.float32 or prototypes.ndim != 2
                or not 1 <= len(prototypes) <= 256 or prototypes.shape[1] != 8*bins
                or prototypes.nbytes > SUPPORT_BYTES or not np.isfinite(prototypes).all()
                or limit < 0):
            raise ValueError('Invalid bounded structural training-support index')
        model.load_state_dict({k: torch.from_numpy(values[k]) for k in expected}); model.eval()
        self.model, self.metadata = model, d
        self.edges, self.mean, self.scale, self.reference = edges, mean, scale, reference
        self.support_prototypes, self.support_limit = prototypes, float(limit)
        self.routes=routes

    def rank(self, q, intensity, *, background, parent_sha256, instrument_sha256, valid=None, reference=None):
        import torch
        if (parent_sha256 != self.metadata.get('parent_sha256')
                or instrument_sha256 != self.metadata.get('instrument_sha256')):
            raise ValueError('This ranker needs its exact prepared parent and instrument context')
        features = pattern_features(q, intensity, self.edges, background=background, valid=valid)
        parent_features = self.reference
        if reference is not None:
            if (not isinstance(reference, dict) or set(reference)-{'q', 'intensity', 'background', 'valid'}
                    or not {'q', 'intensity', 'background'} <= set(reference)):
                raise ValueError('A parent reference requires its calculated pattern, coordinates and background')
            parent_features = pattern_features(reference['q'], reference['intensity'], self.edges,
                background=reference['background'], valid=reference.get('valid'))
        features = with_parent_reference(features, parent_features)
        # A partial range must not silently acquire the complete-range model's
        # confidence. A future masked-range qualification can explicitly widen
        # this admission rule using independent experimental evidence.
        if np.any(features[2] < .999):
            raise ValueError('This structural ranker is not qualified for missing coordinate intervals')
        features = (features-self.mean)/self.scale
        distance = float(support_distances(self.support_prototypes, features.reshape(1, -1))[0])
        inside = distance <= self.support_limit+np.finfo(np.float32).eps*max(1., self.support_limit)
        support = dict(status='within_empirical_synthetic_support' if inside else 'outside_prepared_support',
            nearest_training_rms=distance, validation_maximum_rms=self.support_limit,
            statistical_coverage_qualified=False)
        if not inside:
            return dict(families=[dict(space_group=group, model_score=None,
                candidate_ids=list(self.routes[str(group)]))
                for group in self.metadata['classes']], ranking_available=False,
                candidate_order=sorted(self.metadata['candidate_groups']),
                support=support, screening_qualified=False, candidate_pruning_authorized=False,
                native_refinement_required=True, complete_native_search_required=True,
                interpretation='This pattern is outside the prepared simulations; retain all families for native refinement')
        with torch.no_grad():
            scores = self.model(torch.from_numpy(features[None])).softmax(1)[0].numpy()
        if not np.isfinite(scores).all():
            raise ValueError('Structural model inference produced nonfinite scores')
        order = np.argsort(-scores, kind='stable')
        families=[dict(space_group=self.metadata['classes'][i], model_score=float(scores[i]),
            candidate_ids=list(self.routes[str(self.metadata['classes'][i])])) for i in order]
        return dict(families=families,
            candidate_order=list(dict.fromkeys(key for row in families for key in row['candidate_ids'])),
            ranking_available=True, support=support, candidate_pruning_authorized=False,
            screening_qualified=False, native_refinement_required=True,
            interpretation='Relative model scores order hypotheses; all candidate families remain available')
