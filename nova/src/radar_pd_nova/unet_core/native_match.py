"""Align measured bins and reported errors without creating feature proposals."""

from __future__ import annotations

import numpy as np


def directly_matched(before: dict, after: dict) -> dict:
    xa, xb = np.asarray(before["x"], dtype=float), np.asarray(after["x"], dtype=float)
    if len(xa) < 2 or len(xb) < 2 or not np.all(np.diff(xa) > 0) or not np.all(np.diff(xb) > 0):
        raise ValueError("Native x grids must increase")
    tolerance = min(max(1e-8, 1e-10 * max(float(np.max(np.abs(xa))),
                                           float(np.max(np.abs(xb))))),
                    .01 * min(float(np.min(np.diff(xa))), float(np.min(np.diff(xb)))))
    right = np.searchsorted(xb, xa)
    left = np.clip(right - 1, 0, len(xb) - 1)
    right = np.clip(right, 0, len(xb) - 1)
    bi_all = np.where(np.abs(xb[left] - xa) <= np.abs(xb[right] - xa), left, right)
    ai = np.flatnonzero(np.abs(xb[bi_all] - xa) <= tolerance)
    bi = bi_all[ai]
    if len(ai) < 2 or not np.all(np.diff(bi) > 0):
        return {"x": np.array([]), "good": np.array([], dtype=bool),
                "direct_z": np.array([])}
    x = xa[ai]
    ay, by = np.asarray(before["y"])[ai], np.asarray(after["y"])[bi]
    ae, be = np.asarray(before["sigma"])[ai], np.asarray(after["sigma"])[bi]
    good = (np.asarray(before["valid"])[ai] & np.asarray(after["valid"])[bi] &
            np.isfinite(ay) & np.isfinite(by) & np.isfinite(ae) & np.isfinite(be) &
            (ae > 0) & (be > 0))
    direct_z = np.zeros(len(x), dtype=float)
    direct_z[good] = (by[good] - ay[good]) / np.hypot(ae[good], be[good])
    return {"x": x, "good": good, "before_y": ay, "after_y": by,
            "before_sigma": ae, "after_sigma": be, "direct_z": direct_z}
