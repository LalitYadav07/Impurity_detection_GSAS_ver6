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
from .structural_features import pattern_features
from .tof_kernel_arrays import read_bounded, checksum

SCHEMA = 'radar-structural-ranking-model/v1'
ARCHITECTURE = 'masked-q-cnn-4-16-32-64-pool32/v1'
MAX_BYTES = 32*1024**2


def network(class_count):
    from torch import nn
    return nn.Sequential(nn.Conv1d(4, 16, 9, stride=2, padding=4), nn.ReLU(),
        nn.Conv1d(16, 32, 7, stride=2, padding=3), nn.ReLU(),
        nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
        nn.AdaptiveAvgPool1d(32), nn.Flatten(), nn.Linear(64*32, 128), nn.ReLU(),
        nn.Dropout(.1), nn.Linear(128, class_count))


def _score(logits, compatible):
    """A symmetry-ambiguous example accepts any of its compatible labels."""
    import torch
    order = logits.argsort(dim=1, descending=True)
    top = compatible.gather(1, order)
    loss = torch.logsumexp(logits, 1)-torch.logsumexp(logits.masked_fill(~compatible, -torch.inf), 1)
    return dict(top1=float(top[:, 0].float().mean()),
        top5=float(top[:, :min(5, logits.shape[1])].any(1).float().mean()),
        loss=float(loss.mean()))


def train(cohort, destination, *, epochs=60, seed=89113, threads=4):
    import torch
    if type(epochs) is not int or not 1 <= epochs <= 1000 or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Bounded training epochs and random seed are required')
    classes = np.asarray(cohort['classes'])
    features = np.asarray(cohort['features']); targets = np.asarray(cohort['compatible_labels'])
    split = np.asarray(cohort['split'])
    if (not 2 <= len(classes) <= 230 or features.ndim != 3 or features.shape[1] != 4
            or features.dtype != np.float32 or not np.isfinite(features).all()
            or features.nbytes > 384*1024**2
            or targets.shape != (len(features), len(classes)) or targets.dtype != bool
            or not targets.any(axis=1).all() or split.shape != (len(features),)
            or set(split) != {0, 1, 2}):
        raise ValueError('A complete finite cohort with independent holdouts is required')
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
    train_rows = split == 0; validation_rows = split == 1; test_rows = split == 2
    mean = features[train_rows, :2].mean(axis=0)
    scale = np.maximum(features[train_rows, :2].std(axis=0), .02)
    normalized = features.copy(); normalized[:, :2] = (normalized[:, :2]-mean)/scale
    x = torch.from_numpy(normalized); y = torch.from_numpy(targets)
    rng = np.random.default_rng(seed); indices = np.flatnonzero(train_rows)
    counts = targets[train_rows].sum(0)
    if not counts.all():
        raise ValueError('Each catalog class requires independent training examples')
    weights = counts.sum()/counts
    weights = torch.tensor(weights/weights.mean(), dtype=torch.float32)
    best = None; best_loss = float('inf'); history = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed); model = network(len(classes))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        for epoch in range(epochs):
            model.train(); order = rng.permutation(indices)
            for start in range(0, len(order), 32):
                selected = order[start:start+32]; logits = model(x[selected]); allowed = y[selected]
                loss = torch.logsumexp(logits, 1)-torch.logsumexp(logits.masked_fill(~allowed, -torch.inf), 1)
                row_weights = (allowed*weights).sum(1)/allowed.sum(1)
                objective = (loss*row_weights).mean()
                if not torch.isfinite(objective):
                    raise ValueError('Structural training produced a nonfinite objective; no model was published')
                optimizer.zero_grad(set_to_none=True); objective.backward(); optimizer.step()
            model.eval()
            with torch.no_grad():
                result = _score(model(x[validation_rows]), y[validation_rows])
            history.append(dict(epoch=epoch+1, validation=result))
            if result['loss'] < best_loss:
                best_loss = result['loss']; best = copy.deepcopy(model.state_dict())
            write_json(out/'progress.json', dict(epoch=epoch+1, epochs=epochs, validation=result))
        model.load_state_dict(best); model.eval()
        # This holdout is inspected only after validation selected the frozen
        # checkpoint. Subsequent development must use a fresh external test.
        with torch.no_grad():
            logits = model(x[test_rows]); score = _score(logits, y[test_rows])
    arrays = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    arrays.update(q_edges=cohort['q_edges'], feature_mean=mean, feature_scale=scale, classes=classes)
    stream = io.BytesIO(); np.savez_compressed(stream, **arrays); raw = stream.getvalue()
    validate_npz_bytes(raw, maximum_bytes=MAX_BYTES); (out/'weights.npz').write_bytes(raw)
    counts = {name: targets[split == code].sum(0).tolist()
              for name, code in [('train', 0), ('validation', 1), ('test', 2)]}
    metadata = dict(schema=SCHEMA, architecture=ARCHITECTURE, parent_sha256=cohort['parent_sha256'],
        instrument_sha256=cohort['instrument_sha256'], input_sha256=cohort['input_sha256'],
        candidates=cohort['candidates'], candidate_groups=cohort['candidate_groups'],
        weights_sha256=hashlib.sha256(raw).hexdigest(), classes=classes.tolist(), bins=features.shape[2],
        epochs=epochs, seed=seed, allocation=budget, counts=counts, test=score,
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
        if (d.get('schema') != SCHEMA or d.get('architecture') != ARCHITECTURE
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
        validate_npz_bytes(raw, maximum_bytes=MAX_BYTES)
        model = network(len(classes)); expected = model.state_dict()
        with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
            if set(saved.files) != set(expected)|{'q_edges', 'feature_mean', 'feature_scale', 'classes'}:
                raise ValueError('Unexpected structural model tensor inventory')
            values = {k: saved[k] for k in saved.files}
        for name, tensor in expected.items():
            a = values[name]
            if a.shape != tuple(tensor.shape) or a.dtype != np.float32 or not np.isfinite(a).all():
                raise ValueError('Invalid structural model tensor dimensions or dtype')
        edges, mean, scale = (values[k] for k in ('q_edges', 'feature_mean', 'feature_scale'))
        if (edges.dtype != np.float64 or mean.dtype != np.float32 or scale.dtype != np.float32
                or values['classes'].dtype != np.int32
                or edges.shape != (bins+1,) or not np.isfinite(edges).all() or edges[0] <= 0
                or np.any(np.diff(edges) <= 0)
                or mean.shape != (2, bins) or scale.shape != mean.shape
                or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0)
                or not np.array_equal(values['classes'], classes)):
            raise ValueError('Invalid structural feature normalization')
        model.load_state_dict({k: torch.from_numpy(values[k]) for k in expected}); model.eval()
        self.model, self.metadata = model, d
        self.edges, self.mean, self.scale = edges, mean, scale

    def rank(self, q, intensity, *, background, parent_sha256, instrument_sha256, valid=None):
        import torch
        if (parent_sha256 != self.metadata.get('parent_sha256')
                or instrument_sha256 != self.metadata.get('instrument_sha256')):
            raise ValueError('This ranker needs its exact prepared parent and instrument context')
        features = pattern_features(q, intensity, self.edges, background=background, valid=valid)
        # A partial range must not silently acquire the complete-range model's
        # confidence. A future masked-range qualification can explicitly widen
        # this admission rule using independent experimental evidence.
        if np.any(features[2] < .999):
            raise ValueError('This structural ranker is not qualified for missing coordinate intervals')
        features[:2] = (features[:2]-self.mean)/self.scale
        with torch.no_grad():
            scores = self.model(torch.from_numpy(features[None])).softmax(1)[0].numpy()
        if not np.isfinite(scores).all():
            raise ValueError('Structural model inference produced nonfinite scores')
        order = np.argsort(-scores, kind='stable')
        return dict(families=[dict(space_group=self.metadata['classes'][i], model_score=float(scores[i]),
            candidate_ids=[key for key, value in self.metadata['candidate_groups'].items()
                           if value == self.metadata['classes'][i]]) for i in order],
            screening_qualified=False, native_refinement_required=True,
            interpretation='Relative model scores order hypotheses; all candidate families remain available')
