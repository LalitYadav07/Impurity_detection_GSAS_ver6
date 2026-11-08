"""Bounded training-only clean replay alongside the complete declared pool.

The strong clean representatives are selected from training trajectories alone.
Replay does not create new independent examples or change split membership.
"""
import numpy as np

from .structural_training_data import feature_rows

REPLAY_ROWS = 16


def replay_batches(rows, indices, representatives, *, split, size=64):
    rows = feature_rows(rows)
    indices = np.asarray(indices); representatives = np.asarray(representatives)
    split = np.asarray(split)
    if (type(size) is not int or not 1 <= size <= 64
            or indices.ndim != 1 or indices.dtype.kind not in 'iu'
            or representatives.ndim != 1 or representatives.dtype.kind not in 'iu'
            or not 1 <= len(representatives) <= 230
            or split.shape != (len(rows),) or split.dtype != np.int8
            or np.any(indices < 0) or np.any(indices >= len(rows))
            or np.any(representatives < 0) or np.any(representatives >= len(rows))):
        raise ValueError('Bounded training indices and clean representatives required')
    if np.any(split[indices] != 0) or np.any(split[representatives] != 0):
        raise ValueError('Clean replay cannot read validation or test trajectories')
    count = min(REPLAY_ROWS, len(representatives))
    for batch, start in enumerate(range(0, len(indices), size)):
        # Rotate through all families when the catalog is larger than one
        # replay batch. Neither the feature tensor nor a GPU activation grows
        # with the number of families: at most 64 + 16 rows enter an update.
        replay = representatives[(batch*count+np.arange(count)) % len(representatives)]
        selected = np.concatenate([indices[start:start+size], replay])
        yield selected, rows.take(selected)
