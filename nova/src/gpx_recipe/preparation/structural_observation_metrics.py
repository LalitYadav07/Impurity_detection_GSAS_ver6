"""Separate clean and simulated recall without extra model evaluations.

The full held-out population still determines checkpoint selection. These
diagnostics do not change labels, remove difficult rows, or qualify screening.
"""
import numpy as np


class ObservationMetrics:
    def __init__(self, clean, rows, classes):
        clean = np.asarray(clean)
        if clean.dtype != bool or clean.shape != (rows,):
            raise ValueError('One clean/native provenance flag per feature row required')
        self.clean = clean.copy()
        self.counts = np.zeros((2, classes), np.int64)
        self.one = np.zeros((2, classes), np.int64)
        self.five = np.zeros((2, classes), np.int64)
        self.rows = np.zeros(2, np.int64)
        self.hits = np.zeros((2, 2), np.int64)

    def add(self, selected, allowed, hit_one, hit_five):
        flags = self.clean[selected]
        for code, mask in enumerate((flags, ~flags)):
            labels = allowed[mask]
            first = hit_one[mask]; five = hit_five[mask]
            self.counts[code] += labels.sum(0)
            self.one[code] += (labels & first[:, None]).sum(0)
            self.five[code] += (labels & five[:, None]).sum(0)
            self.rows[code] += int(mask.sum())
            self.hits[code] += [int(first.sum()), int(five.sum())]

    def finish(self):
        parts = {}
        for code, name in enumerate(('clean_native', 'simulated_observation')):
            n = int(self.rows[code]); present = self.counts[code] > 0
            recall = self.five[code, present] / self.counts[code, present]
            parts[name] = dict(rows=n, represented_classes=int(present.sum()),
                top1=None if not n else float(self.hits[code, 0]/n),
                top5=None if not n else float(self.hits[code, 1]/n),
                macro_top5=None if not n else float(recall.mean()),
                worst_class_top5=None if not n else float(recall.min()))
        return dict(schema='radar-structural-observation-metrics/v1', partitions=parts,
            scope='Diagnostic subsets of the same scored rows; no additional independent pairs or screening qualification')
