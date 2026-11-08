"""Scoped measured-pattern comparisons with the frozen Focus U-Net.

This supplies optional local morphology evidence alongside indicator votes. A scan is compared
only with a selected earlier scan of the same specimen and reduction scope.
The model proposes every displayed box; native measured evidence can reject a
proposal and qualify a label, but cannot invent an event.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import asdict
from pathlib import Path
import math
import re
from threading import RLock
import numpy as np

from .scan_interpretation import _time
from .scan_conditions import _unit
from .transition_input import digest, load_observation
from .unet_core.production import FocusTransitionPredictor
from .unet_core.full_q_production import FullQTransitionPredictor

SCHEMA = "radar-pd-unet-transition-history/v1"
RAW_CACHE_BYTES = 96 * 1024 * 1024
RAW_CACHE_ENTRIES = 16
DISPLAY_CACHE_SOURCES = 102  # Largest map (100) plus selection and reference.


def source_signature(scan):
    """Identity of measured inputs and the provenance used by the adapter."""
    stamps = []
    raw_available = False
    for field in ("source_path", "pattern_path"):
        path = scan.get(field)
        try:
            stat = Path(path).stat() if path else None
            stamps.append((stat.st_size, stat.st_mtime_ns, stat.st_ino) if stat else None)
            if field == "source_path":
                raw_available = stat is not None and bool(scan.get("transition_source_root"))
        except OSError:
            stamps.append("unavailable")
    # load_observation prefers the confined raw product. A later refinement
    # adds a portable observed/fit receipt, but does not replace that input.
    # Invalidate on the receipt only when it is actually the input, including
    # recovery after the original facility mount disappears.
    packet = None if raw_available else scan.get("pattern")
    if isinstance(packet, dict):
        packet = {key: value for key, value in packet.items()
                  if key not in {"ycalc", "fit_valid", "phase_ticks", "phase_order"}}
    fields = ("run_id", "source_path", "metadata", "bank", "instrument", "ipts", "source_family",
              "comparison_setup_id", "transition_source_root", "scientific_warnings",
              "warnings", "notices", "transition_profile_dataset_id")
    if raw_available:
        stamps[1] = None
    else:
        fields += ("pattern_path", "pattern_observation")
    return digest([{field: scan.get(field) for field in fields}, packet, stamps])


def acquisition_key(scan):
    acquired = _time(scan)
    try:
        run = int(scan.get("run_number"))
    except (TypeError, ValueError):
        run = 0
    return (acquired is None, acquired if acquired is not None else run,
            run, str(scan.get("run_id")))


_ORDER_FIELDS = {
    "temperature": "Temperature",
    "magnetic_field": "Magnetic field",
    "pressure": "Pressure",
    "wavelength": "Wavelength",
    "voltage": "Voltage",
    "gas_flow": "Gas flow",
    "proton_charge": "Proton charge",
}


def condition_order_value(scan, field):
    """A physical ordering value with explicit units, or None if unavailable.

    Temperature in an HB-2A title is a nominal setting, not a readback. It is
    useful for arranging those scans and is identified as such in the UI.
    """
    metadata = scan.get("metadata") or {}
    override = (scan.get("axis_overrides") or {}).get(field)
    reviewed = isinstance(override, dict) and (
        override.get("reviewed_mapping") is True
        or str(override.get("source") or "").startswith("Reviewed run mapping:"))
    value = override if reviewed else metadata.get(field)
    if isinstance(value, dict):
        try:
            number = float(value.get("value"))
        except (TypeError, ValueError):
            number = math.nan
        source_unit = str(value.get("unit") or "").strip()
        unit = source_unit.lower()
        conversions = {
            "temperature": {"k": (1, 0), "kelvin": (1, 0), "c": (1, 273.15),
                            "°c": (1, 273.15), "mk": (.001, 0)},
            "magnetic_field": {"t": (1, 0), "mt": (.001, 0)},
            "wavelength": {"å": (1, 0), "a": (1, 0), "angstrom": (1, 0), "nm": (10, 0)},
        }
        if math.isfinite(number):
            if field == "pressure":
                conversion = _unit(field, source_unit)
                if conversion is not None:
                    canonical, scale, offset = conversion
                    if canonical == "Pa":
                        return (number * scale + offset) / 1e9, field, (
                            "reviewed mapping" if reviewed else "measured")
                return None
            if field in conversions:
                if unit in conversions[field]:
                    scale, offset = conversions[field][unit]
                    return number * scale + offset, field, (
                        "reviewed mapping" if reviewed else "measured")
            elif unit:
                return number, unit, "reviewed mapping" if reviewed else "measured"
    if field == "temperature":
        nominal = scan.get("nominal_temperature_K")
        if nominal is None:
            title = str(metadata.get("run_title") or "")
            match = re.search(r"\bT\s*=\s*(\d+(?:\.\d+)?)\s*(mK|K)\b", title, re.I)
            nominal = float(match.group(1)) / (1000 if match.group(2).lower() == "mk" else 1) if match else None
        try:
            number = float(nominal)
            if math.isfinite(number):
                return number, field, "nominal title"
        except (TypeError, ValueError):
            pass
    return None


def available_orders(scans):
    """Only offer a physical sort when every scan has comparable, varying data."""
    options = [{"title": "Acquisition time", "value": "acquired"},
               {"title": "Run number", "value": "run_number"}]
    for field, title in _ORDER_FIELDS.items():
        values = [condition_order_value(scan, field) for scan in scans]
        if not values or any(value is None for value in values):
            continue
        if len({value[1] for value in values}) != 1 or len({value[0] for value in values}) < 2:
            continue
        nominal = any(value[2] == "nominal title" for value in values)
        unit = {"temperature": "K", "magnetic_field": "T", "pressure": "GPa",
                "wavelength": "Å"}.get(field, values[0][1])
        options.append({"title": f"{title} ({unit})" + (" · includes nominal titles" if nominal else ""),
                        "value": field})
    return options


def _order_key(scan, order_by, *, descending=False):
    def absence_rank(missing):
        # Reverse the values, not the placement of unknown metadata.
        return not missing if descending else missing

    if order_by == "run_number":
        try:
            return (absence_rank(False), int(scan.get("run_number")), acquisition_key(scan))
        except (TypeError, ValueError):
            return (absence_rank(True), 0, acquisition_key(scan))
    if order_by in _ORDER_FIELDS:
        value = condition_order_value(scan, order_by)
        return (absence_rank(value is None), value[0] if value else 0, acquisition_key(scan))
    key = acquisition_key(scan)
    return (absence_rank(key[0]), *key[1:])


class ComparisonSuperseded(RuntimeError):
    """An obsolete read-only request yielded to a newer selection."""


class UnetTransitionEngine:
    def __init__(self, *, loader=None, predictor=None,
                 raw_cache_bytes=RAW_CACHE_BYTES):
        if not isinstance(raw_cache_bytes, int) or raw_cache_bytes < 0:
            raise ValueError("Raw-pattern cache budget must be a nonnegative integer")
        self.loader = loader or load_observation
        self.predictor = predictor or FocusTransitionPredictor()
        self._loaded = OrderedDict()
        self._loaded_sizes = {}
        self._loaded_bytes = 0
        self._raw_cache_bytes = raw_cache_bytes
        self._retained_sources = {}
        self._pairs = OrderedDict()
        self._summaries = OrderedDict()
        self._cache_lock = RLock()

    source_signature = staticmethod(source_signature)
    _source_key = source_signature

    def retain_sources(self, scans):
        """Prioritize this display's selection, reference and measured window.

        This does not load or copy arrays. The byte budget always takes
        precedence; earlier sources have priority if a large window exceeds it.
        Ordinary inference keeps its independent 16-entry working cache.
        """
        retained = {}
        for index, scan in enumerate(scans):
            if index >= DISPLAY_CACHE_SOURCES:
                break
            retained.setdefault(self._source_key(scan), index)
        with self._cache_lock:
            self._retained_sources = retained
            self._prune_loaded()

    def _drop_loaded(self, key):
        del self._loaded[key]
        self._loaded_bytes -= self._loaded_sizes.pop(key)

    def _prune_loaded(self):
        ordinary = [key for key in self._loaded if key not in self._retained_sources]
        for key in ordinary[:-RAW_CACHE_ENTRIES]:
            self._drop_loaded(key)
        # Evict ordinary LRU entries first, then the lowest display priority.
        while self._loaded_bytes > self._raw_cache_bytes:
            key = next((key for key in self._loaded if key not in self._retained_sources), None)
            if key is None:
                key = max(self._loaded, key=self._retained_sources.__getitem__)
            self._drop_loaded(key)

    def load(self, scan):
        """Reuse observations within the display priority and raw-byte budget."""
        key = self._source_key(scan)
        with self._cache_lock:
            if key in self._loaded:
                self._loaded.move_to_end(key)
                return self._loaded[key]
        item = self.loader(scan)
        with self._cache_lock:
            if key not in self._loaded:
                size = sum(values.nbytes for values in (
                    item.pattern.x, item.pattern.intensity,
                    item.pattern.sigma, item.pattern.valid))
                if size <= self._raw_cache_bytes:
                    self._loaded[key] = item
                    self._loaded_sizes[key] = size
                    self._loaded_bytes += size
                    self._prune_loaded()
            else:
                item = self._loaded[key]
                self._loaded.move_to_end(key)
        return item

    def _summary(self, scan):
        # Compact source receipts outlive the bounded raw-array cache. A new
        # scan must not reread every earlier file merely to reconstruct scope.
        key = self._source_key(scan)
        with self._cache_lock:
            if key in self._summaries:
                self._summaries.move_to_end(key)
                return self._summaries[key]
        item = self.load(scan)
        values = item.pattern.intensity[item.pattern.valid]
        values = values[np.isfinite(values)]
        positive = values[values > 0]
        summary = {"scope": asdict(item.pattern.scope),
                   "pattern_sha256": item.fingerprint,
                   "source": item.provenance, "notices": [
                       "Absolute normalization is not declared; U-Net preserves relative raw intensities, so counting changes can produce candidate regions."
                       if "comparisons include robust intensity scaling" in notice else notice
                       for notice in item.notices],
                   "color_limits": {
                       "linear": [float(np.min(values)), float(np.max(values))] if len(values) else None,
                       "log": [float(np.min(positive)), float(np.max(positive))] if len(positive) else None}}
        with self._cache_lock:
            self._summaries[key] = summary
            while len(self._summaries) > 5000:
                self._summaries.popitem(last=False)
        return summary

    def _pair_context(self, before_scan, after_scan):
        return None

    def _pair_key(self, before_fingerprint, after_fingerprint, context=None):
        profile_key = tuple(getattr((context or {}).get(name), "profile_sha256", None)
                            for name in ("calibration_before", "calibration_after"))
        return (before_fingerprint, after_fingerprint,
                self.predictor.metadata["checkpoint_sha256"],
                self.predictor.metadata.get("postprocessing_sha256"), profile_key)

    def _compare(self, before, after, context=None):
        key = self._pair_key(before.fingerprint, after.fingerprint, context)
        with self._cache_lock:
            if key in self._pairs:
                self._pairs.move_to_end(key)
                return self._pairs[key]
        result = self.predictor.predict_pair(before, after, **context) if context is not None \
            else self.predictor.predict_pair(before, after)
        # The full inference receipt contains every tile. Keep its counts and
        # timing, but do not retain thousands of tile dictionaries per pair.
        inference = result.get("inference") or {}
        compact = {name: result[name] for name in (
            "schema", "status", "attention", "events", "reasons",
            "coverage_fraction", "supported_intervals", "q_coverage_intervals",
            "causes", "profiles", "postprocessing",
        ) if name in result}
        compact["inference"] = {name: value for name, value in inference.items()
                                if name not in {"inferred_windows", "skipped_windows"}}
        with self._cache_lock:
            if key not in self._pairs:
                self._pairs[key] = compact
            else:
                compact = self._pairs[key]
            while len(self._pairs) > 5000:
                self._pairs.popitem(last=False)
            self._pairs.move_to_end(key)
        return compact

    def evaluate(self, scans, *, order_by="acquired", descending=False,
                 reference_gap=1, progress=None, cancelled=None):
        if len(scans) > 5000:
            raise ValueError("Select a watch with at most 5000 acquisitions")
        if order_by not in {"acquired", "run_number", *_ORDER_FIELDS}:
            raise ValueError("Unsupported scan ordering")
        if reference_gap != "first" and (not isinstance(reference_gap, int)
                                         or not 1 <= reference_gap <= 100):
            raise ValueError("Unsupported comparison spacing")
        def check_current():
            if cancelled is not None and cancelled():
                raise ComparisonSuperseded("A newer comparison request is available")

        check_current()
        ordered = sorted(scans, key=acquisition_key)
        counts = Counter(str(row.get("run_id") or "") for row in ordered)
        groups = OrderedDict()
        by_group = OrderedDict()
        unsupported = []
        total_work = 2 * len(ordered)
        for loaded_count, scan in enumerate(ordered, 1):
            check_current()
            run_id = str(scan.get("run_id") or "")
            row = {"run_id": run_id, "group": None, "previous": None,
                   "status": "unsupported", "attention": "uncertain",
                   "events": [], "reason": "", "pattern_sha256": None,
                   "comparison": None}
            try:
                if not run_id:
                    raise ValueError("Scan has no run ID")
                if counts[run_id] > 1:
                    raise ValueError("Multiple result revisions require review")
                summary = self._summary(scan)
                group = digest(summary["scope"])
                row.update(group=group, **{key: value for key, value in summary.items()
                                         if key != "scope"})
                if group not in groups:
                    groups[group] = {"id": group, "scope": summary["scope"],
                                     "run_ids": []}
                by_group.setdefault(group, []).append((scan, row))
            except (OSError, ValueError, TypeError, UnicodeError, RuntimeError) as error:
                row["reason"] = str(error)
                unsupported.append((scan, row))
            if progress is not None:
                progress(loaded_count, total_work,
                         f"Reading {scan.get('run_number') or run_id}")
        record_entries = []
        for group, entries in by_group.items():
            entries.sort(key=lambda entry: _order_key(entry[0], order_by,
                                                       descending=descending), reverse=descending)
            groups[group]["run_ids"] = [entry[1]["run_id"] for entry in entries]
            source_notices = {row["run_id"]: list(row.get("notices") or [])
                              for _, row in entries}
            for index, (_scan, row) in enumerate(entries):
                check_current()
                if index == 0:
                    row.update(status="initial", reason="First measured pattern in this ordering")
                else:
                    reference_index = 0 if reference_gap == "first" else max(0, index - reference_gap)
                    before_scan, before_row = entries[reference_index]
                    before_id = before_row["run_id"]
                    try:
                        context = self._pair_context(before_scan, _scan)
                        key = self._pair_key(before_row["pattern_sha256"], row["pattern_sha256"], context)
                        with self._cache_lock:
                            comparison = self._pairs.get(key)
                            if comparison is not None:
                                self._pairs.move_to_end(key)
                        if comparison is None:
                            before = self.load(before_scan)
                            item = self.load(_scan)
                            if (before.fingerprint != before_row["pattern_sha256"] or
                                    item.fingerprint != row["pattern_sha256"]):
                                raise ValueError("Measured source changed while comparisons were prepared")
                            comparison = self._compare(before, item, context)
                        row.update(previous=before_id, status=comparison["status"],
                                   reference_steps=index - reference_index,
                                   attention=comparison["attention"],
                                   events=comparison["events"],
                                   comparison=comparison,
                                   notices=list(dict.fromkeys([
                                       *source_notices[before_id],
                                       *source_notices[row["run_id"]]])),
                                   reason="; ".join(comparison.get("reasons") or []))
                    except (OSError, ValueError, TypeError, UnicodeError, RuntimeError) as error:
                        row.update(previous=before_id, reason=str(error))
                record_entries.append((_scan, row))
                if progress is not None:
                    progress(len(ordered) + len(record_entries), total_work,
                             f"Comparing {_scan.get('run_number') or row['run_id']}")
        record_entries.extend(unsupported)
        if progress is not None and unsupported:
            progress(total_work, total_work, "Skipping unavailable scans")
        record_entries.sort(key=lambda entry: _order_key(entry[0], order_by,
                                                         descending=descending), reverse=descending)
        records = [row for _, row in record_entries]
        return {"schema": SCHEMA, "model": self.predictor.metadata,
                "records": records, "groups": list(groups.values()),
                "order_by": order_by, "descending": descending,
                "reference_gap": reference_gap}


class FullQTransitionEngine(UnetTransitionEngine):
    """Use one checked profile for each measured scan before whole-Q inference."""

    def __init__(self, *, calibration_resolver, loader=None,
                 raw_cache_bytes=RAW_CACHE_BYTES):
        super().__init__(loader=loader, predictor=FullQTransitionPredictor(),
                         raw_cache_bytes=raw_cache_bytes)
        self.calibration_resolver = calibration_resolver
        self._calibrations = {}

    def _pair_context(self, before_scan, after_scan):
        def calibration(scan):
            coordinate = self.load(scan).pattern.scope.coordinate
            if coordinate in {"d_angstrom", "q_inv_angstrom"}:
                return None
            key = self._source_key(scan)
            with self._cache_lock:
                if key in self._calibrations:
                    return self._calibrations[key]
            value = self.calibration_resolver(scan)
            with self._cache_lock:
                if len(self._calibrations) >= 5000:
                    self._calibrations.clear()
                self._calibrations[key] = value
            return value
        return {"calibration_before": calibration(before_scan),
                "calibration_after": calibration(after_scan)}
