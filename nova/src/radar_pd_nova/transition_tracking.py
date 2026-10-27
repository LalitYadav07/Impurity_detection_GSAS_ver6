"""Causal, bounded memory of recurring observed-pattern references.

The default legacy API retains all-member compatibility for reproducible old
audits. Passing a BoundaryPolicy enables the online reference-based indicator
stream used by NOVA. Measured-peak matching never receives fitted phases or
refined cells; a separate optional final-Rwp adapter can join the boundary rule.
"""

from __future__ import annotations

import math
import threading
from collections import Counter, OrderedDict
from dataclasses import asdict
from functools import wraps
from pathlib import Path

from .pattern_transition import (
    TransitionSettings,
    compare_observed_patterns,
    prepare_pattern,
)
from .scan_interpretation import _time
from .transition_input import digest, load_observation, observation

SCHEMA = "radar-pd-transition-tracking/v2"


def checkpoint(value):
    """Validate the small persisted settings; never accept arrays or NaN limits."""
    result = {"references": {}, "limits": {}, "styles": {}}
    if not isinstance(value, dict):
        return result
    for field, destination in result.items():
        entries = value.get(field, {})
        if not isinstance(entries, dict) or len(entries) > 5000:
            continue
        for key, item in entries.items():
            if not isinstance(key, str) or len(key) > 128:
                continue
            if (
                field == "references"
                and isinstance(item, str)
                and len(item) <= 256
                or field == "styles"
                and isinstance(item, int)
                and not isinstance(item, bool)
                and 0 <= item < 32
            ):
                destination[key] = item
            elif field == "limits" and isinstance(item, dict):
                values = [item.get(k) for k in ("low", "high", "soft")]
                if (
                    all(
                        isinstance(v, (int, float))
                        and not isinstance(v, bool)
                        and math.isfinite(v)
                        for v in values
                    )
                    and values[0] < values[1]
                    and values[2] > 0
                ):
                    destination[key] = dict(
                        zip(("low", "high", "soft"), values),
                        source=str(item.get("source", "saved reference"))[:256],
                    )
    if isinstance(value.get("policy"), dict):
        from .transition_signals import BoundaryPolicy
        result["policy"] = asdict(BoundaryPolicy.restore(value["policy"]))
    return result


def synchronized(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapped


def acquisition_key(scan):
    time = _time(scan)
    try:
        run = int(scan.get("run_number"))
    except (ValueError, TypeError):
        run = 0
    return (
        time is None,
        time if time is not None else run,
        run,
        str(scan.get("run_id")),
    )


class PatternStateMemory:
    """Keep mutually compatible members; never turn an unknown relation into a change alert."""
    def __init__(self, settings=None, compare=None, style_registry=None, max_states=32):
        self.settings = settings or TransitionSettings()
        self.compare = compare or compare_observed_patterns
        self.styles = style_registry if style_registry is not None else {}
        self.max_states = max_states
        self.states, self.previous, self.anchor, self.last_assignment = [], None, None, None
        self.pending = None

    def status(self, a, b):
        return (self.compare(a, b, self.settings)["status"],
                self.compare(b, a, self.settings)["status"])

    def observe(self, prepared, fingerprint):
        current = prepared.observation
        result = {"run_id": current.observation_id, "pattern_sha256": fingerprint,
                  "state_id": None, "style_index": None, "event": "unassigned",
                  "reason": "comparison_uncertain", "representative": None,
                  "previous": self.previous.observation.observation_id if self.previous else None,
                  "previous_comparison": None, "anchor_comparison": None,
                  "relation_to_other_references": "unresolved"}
        if self.compare(prepared, prepared, self.settings)["status"] != "no_resolved_change":
            result["reason"] = "insufficient_reproducible_feature_support"
            return result
        if self.previous is not None:
            result["previous_comparison"] = self.compare(self.previous, prepared, self.settings)
        if self.anchor is not None:
            result["anchor_comparison"] = self.compare(self.anchor, prepared, self.settings)
        compatible, separated = [], []
        for known in self.states:
            statuses = [s for old in known["members"] for s in self.status(old, prepared)]
            if all(s == "no_resolved_change" for s in statuses):
                compatible.append(known)
            separated.append(all(s == "transition_evidence" for s in statuses))
        chosen = None
        if len(compatible) == 1:
            chosen = compatible[0]
            result["event"] = "continued" if self.last_assignment == chosen["id"] else "returned"
        elif len(compatible) > 1:
            result["reason"] = "ambiguous_between_existing_pattern_references"
        else:
            event = None
            members = [prepared]
            if not self.states:
                event = "initial"
            elif all(separated):
                event = "new"
            elif self.pending is not None and all(s == "no_resolved_change" for s in self.status(self.pending, prepared)):
                event = "reference"
                members = [self.pending, prepared]
            if event is not None:
                points = sum(len(p.observation.x) for s in self.states for p in s["members"])
                if len(self.states) >= self.max_states or points + sum(len(p.observation.x) for p in members) > 2_000_000:
                    result["reason"] = "state_memory_capacity_reached"
                else:
                    first = members[0]
                    first_fingerprint = getattr(first, "_tracking_fingerprint", None) or observation(first.observation).fingerprint
                    state_id = digest([asdict(current.scope), first_fingerprint])
                    prefix = digest(asdict(current.scope))[:32] + ":"
                    style_key = prefix + state_id
                    if style_key not in self.styles:
                        used = {v for k, v in self.styles.items() if k.startswith(prefix)}
                        free = [i for i in range(32) if i not in used]
                        if free:
                            self.styles[style_key] = free[0]
                        else:
                            result["reason"] = "state_memory_capacity_reached"
                    if style_key in self.styles:
                        chosen = {"id": state_id, "first": first, "members": members[:-1],
                                  "style": self.styles[style_key], "established_by": event}
                        self.states.append(chosen)
                        result["event"] = event
            else:
                result["reason"] = "ambiguous_or_nontransitive_state_membership"
        if chosen is not None:
            points = sum(len(p.observation.x) for s in self.states for p in s["members"])
            if points + len(prepared.observation.x) > 2_000_000:
                chosen = None
                result["event"] = "unassigned"
                result["reason"] = "state_memory_point_capacity_reached"
            else:
                chosen["members"].append(prepared)
                result.update(state_id=chosen["id"], style_index=chosen["style"], reason="",
                              representative=chosen["first"].observation.observation_id,
                              relation_to_other_references="separated" if all(separated[i] for i, s in enumerate(self.states[:len(separated)]) if s is not chosen) else "unresolved")
                self.last_assignment = chosen["id"]
                self.pending = None
        if chosen is None:
            self.pending = prepared
        if self.anchor is None:
            self.anchor = prepared
        self.previous = prepared
        return result

class TransitionTrackingEngine:
    """Bounded caches and deterministic replay; presentation never changes inference."""

    def __init__(self, settings=None, *, loader=None, styles=None, policy=None):
        self.settings = settings or TransitionSettings()
        self.loader = loader or load_observation
        self.styles = dict(styles or {})
        self._loaded = OrderedDict()
        self._prepared = OrderedDict()
        self._pairs = OrderedDict()
        self._lock = threading.RLock()
        self.policy = policy
        self._signal_pairs = OrderedDict()
        self._online = None
        self._online_keys = []
        self._online_sources = []
        self._final_fits = {}
        self._final_fit_keys = {}

    @synchronized
    def compare_evidence(self, before, after):
        from .transition_signals import apply_policy, measure_signals
        if self.policy is None:
            return self.pair(before, after)
        key = (getattr(before, "_tracking_fingerprint", None) or observation(before.observation).fingerprint,
               getattr(after, "_tracking_fingerprint", None) or observation(after.observation).fingerprint)
        if key not in self._signal_pairs:
            raw = compare_observed_patterns(before, after, self.settings)
            measurements = measure_signals(before, after, raw)
            compact = {k:v for k,v in raw.items() if k not in {"matches","hints","unmatched","settings"}}
            self._signal_pairs[key] = (compact, measurements)
        self._signal_pairs.move_to_end(key)
        while len(self._signal_pairs) > 256:
            self._signal_pairs.popitem(last=False)
        raw, measurements = self._signal_pairs[key]
        if "rwp" in self.policy.selected:
            from .refinement_indicator import compare_final_fits
            measurements={**measurements,"rwp":compare_final_fits(
                self._final_fits.get(before.observation.observation_id),
                self._final_fits.get(after.observation.observation_id),self.policy)}
        decision = apply_policy(measurements, self.policy)
        return {**raw, **decision, "raw_status": raw["status"], "reasons": [decision["reason"]]}

    def set_policy(self, policy):
        """Explicit user edits replay received data; they never launch refinements."""
        with self._lock:
            if self.policy != policy:
                self.policy = policy
                self._online = None
                self._online_keys = []

    @synchronized
    def evaluate_with_policy(self, scans, policy, *, progress=None):
        self.set_policy(policy)
        return self.evaluate(scans, progress=progress)

    def append(self, scan):
        """Accept a single acquisition; callers need not supply a completed series."""
        if self.policy is None:
            raise ValueError("Choose a boundary policy before appending acquisitions")
        with self._lock:
            previous_ids = {str(s["run_id"]) for s in self._online_sources}
            if (self._online is not None and str(scan["run_id"]) not in previous_ids
                    and (not self._online_sources or acquisition_key(scan) >= acquisition_key(self._online_sources[-1]))):
                if len(self._online_sources) >= 5000:
                    raise ValueError("Tracking supports 5000 received acquisitions per watch")
                self._append_online_scan(scan)
                self._online_sources.append(scan)
                self._online_keys.append(self._online_source_key(scan))
                return self._online.report()
            sources = {str(s["run_id"]): s for s in self._online_sources}
            sources[str(scan["run_id"])] = scan
            return self.evaluate(list(sources.values()))

    @synchronized
    def reclassify(self, policy):
        """Update the rule immediately using only acquisitions already received."""
        self.set_policy(policy)
        return self.evaluate(self._online_sources)

    def _online_source_key(self,scan):
        path = scan.get("source_path") or scan.get("pattern_path")
        try:
            stat = Path(path).stat() if path else None
            revision = [stat.st_size, stat.st_mtime_ns] if stat else scan.get("pattern")
        except OSError:
            revision = "unavailable"
        fit=self._final_fits.get(str(scan.get("run_id"))) if self.policy and "rwp" in self.policy.selected else None
        return digest([scan.get("run_id"), path, revision, scan.get("metadata"),fit,
            scan.get("bank"), scan.get("instrument"), scan.get("ipts"),
            scan.get("source_family"), scan.get("comparison_setup_id"), scan.get("transition_source_root")])

    def _append_online_scan(self, scan):
        try:
            if self.policy and "rwp" in self.policy.selected:
                self._register_final_fits([scan])
            item = self.load(scan)
            self._online.observe(scan, item, self.prepare(item))
        except (OSError, ValueError, TypeError, UnicodeError) as exc:
            self._online.records.append({"run_id": str(scan.get("run_id")), "state_id": None,
                "style_index": None, "event": "unassigned", "reason": str(exc), "group": None,
                "pattern_sha256": None, "boundary": False})

    @synchronized
    def _evaluate_online(self, scans, *, progress=None):
        from .transition_stream import SignalStream
        sources = sorted(scans, key=acquisition_key)
        # Fit updates affect classification only when the user explicitly
        # includes the model-dependent final-Rwp indicator.
        if "rwp" in self.policy.selected:
            self._register_final_fits(sources)
        keys = [self._online_source_key(scan) for scan in sources]
        had_records=bool(self._online and self._online.records)
        replay = self._online is None or keys[:len(self._online_keys)] != self._online_keys
        if replay:
            self.styles = {}
            self._online = SignalStream(self)
            self._online_keys = []
        seen = [str(s.get("run_id")) for s in sources]
        duplicates = {r for r, count in Counter(seen).items() if count > 1}
        for scan in sources[len(self._online_keys):]:
            try:
                if str(scan.get("run_id")) in duplicates:
                    raise ValueError("multiple_result_revisions_require_review")
                item = self.load(scan)
                self._online.observe(scan, item, self.prepare(item))
            except (OSError, ValueError, TypeError, UnicodeError) as exc:
                self._online.records.append({"run_id": str(scan.get("run_id")), "state_id": None,
                    "style_index": None, "event": "unassigned", "reason": str(exc), "group": None,
                    "pattern_sha256": None, "boundary": False})
            if progress is not None:
                progress(len(self._online.records), len(sources), str(scan.get('run_number') or scan.get('run_id')))
        self._online_keys = keys
        self._online_sources = list(sources)
        report = self._online.report()
        report["refinement_evidence_updated"]=bool(replay and had_records and "rwp" in self.policy.selected)
        report["records"] = list({r["run_id"]: r for r in report["records"]}.values())
        return report

    def _register_final_fits(self,scans):
        from .refinement_indicator import final_fit,final_fit_key
        for scan in scans:
            run_id=str(scan["run_id"]);key=final_fit_key(scan)
            if self._final_fit_keys.get(run_id)!=key:
                self._final_fits[run_id]=final_fit(scan)
                self._final_fit_keys[run_id]=key

    def load(self, scan):
        if scan.get("source_path") and scan.get("transition_source_root"):
            path = Path(scan["source_path"])
            try:
                stat = path.stat()
                revision = [str(path), stat.st_size, stat.st_mtime_ns]
            except OSError:
                revision = [str(path), "unavailable"]
        elif scan.get("pattern_path"):
            path = Path(scan["pattern_path"])
            stat = path.stat()
            revision = [str(path), stat.st_size, stat.st_mtime_ns]
        else:
            packet = scan.get("pattern") or {}
            revision = {
                k: packet.get(k)
                for k in (
                    "coordinate",
                    "ordinate",
                    "context",
                    "x",
                    "y",
                    "sigma",
                    "valid",
                )
            }
        key = digest(
            [
                revision,
                scan.get("pattern_observation"),
                scan.get("metadata"),
                scan.get("run_id"),
                scan.get("bank"),
                scan.get("instrument"),
                scan.get("ipts"),
                scan.get("source_family"),
                scan.get("comparison_setup_id"),
                scan.get("transition_source_root"),
            ]
        )
        with self._lock:
            if key not in self._loaded:
                self._loaded[key] = self.loader(scan)
            self._loaded.move_to_end(key)
            while (
                len(self._loaded) > 48
                or sum(len(v.pattern.x) for v in self._loaded.values()) > 500_000
            ):
                if len(self._loaded) == 1:
                    break
                self._loaded.popitem(last=False)
            return self._loaded[key]

    @synchronized
    def prepare(self, item):
        key = (item.fingerprint, item.pattern.observation_id)
        if key not in self._prepared:
            self._prepared[key] = prepare_pattern(item.pattern, self.settings)
            self._prepared[key]._tracking_fingerprint = item.fingerprint
        self._prepared.move_to_end(key)
        while (
            len(self._prepared) > 64
            or sum(len(v.observation.x) for v in self._prepared.values()) > 500_000
        ):
            if len(self._prepared) == 1:
                break
            self._prepared.popitem(last=False)
        return self._prepared[key]

    @synchronized
    def pair(self, before, after, settings=None):
        # Observation fingerprints include scope and only measured data.
        def key(p):
            fingerprint = getattr(p, "_tracking_fingerprint", None)
            if fingerprint is None:
                fingerprint = observation(p.observation).fingerprint
            return fingerprint

        cache_key = (key(before), key(after))
        if cache_key not in self._pairs:
            report = compare_observed_patterns(before, after, self.settings)
            self._pairs[cache_key] = {
                k: v
                for k, v in report.items()
                if k not in {"matches", "hints", "unmatched", "settings"}
            }
        self._pairs.move_to_end(cache_key)
        while len(self._pairs) > 1024:
            self._pairs.popitem(last=False)
        return {
            **self._pairs[cache_key],
            "before": before.observation.observation_id,
            "after": after.observation.observation_id,
        }

    def evaluate(self, scans, *, progress=None):
        if len(scans) > 5000:
            raise ValueError(
                "Tracking supports 5000 acquisitions per watch; select a narrower watch"
            )
        if self.policy is not None:
            return self._evaluate_online(scans, progress=progress)
        sources = sorted(scans, key=acquisition_key)
        records, groups = {}, OrderedDict()
        by_id = {}
        for scan in sources:
            run_id = str(scan.get("run_id") or "")
            if not run_id:
                continue
            if run_id in by_id:
                records[run_id] = {
                    "run_id": run_id,
                    "state_id": None,
                    "event": "unassigned",
                    "reason": "multiple_result_revisions_require_review",
                    "group": None,
                }
                continue
            by_id[run_id] = scan
            try:
                item = self.load(scan)
                group = digest(asdict(item.pattern.scope))
                groups.setdefault(
                    group, {"scope": asdict(item.pattern.scope), "scans": []}
                )["scans"].append(scan)
            except (OSError, ValueError, TypeError, UnicodeError) as exc:
                records[run_id] = {
                    "run_id": run_id,
                    "state_id": None,
                    "event": "unassigned",
                    "reason": str(exc),
                    "group": None,
                }
        for group, definition in groups.items():
            memory = PatternStateMemory(
                self.settings, compare=self.pair, style_registry=self.styles
            )
            for scan in definition["scans"]:
                run_id = str(scan["run_id"])
                if run_id in records:
                    continue
                item = self.load(scan)
                row = memory.observe(self.prepare(item), item.fingerprint)
                row.update(
                    group=group,
                    source=item.provenance,
                    notices=item.notices,
                    order=len(records),
                    provisional=any(
                        not s.get("pattern_path")
                        and not s.get("source_path")
                        and not s.get("pattern")
                        for s in definition["scans"]
                    ),
                )
                records[run_id] = row
        ordered = [
            records[str(s["run_id"])]
            for s in sources
            if str(s.get("run_id")) in records
        ]
        # Duplicate physical IDs appear once and cannot acquire a marker.
        ordered = list({r["run_id"]: r for r in ordered}.values())
        public_groups = [
            {
                "id": key,
                "scope": d["scope"],
                "run_ids": [str(s["run_id"]) for s in d["scans"]],
            }
            for key, d in groups.items()
        ]
        active_styles = {
            r["group"][:32] + ":" + r["state_id"] for r in ordered if r.get("state_id")
        }
        self.styles = {k: v for k, v in self.styles.items() if k in active_styles}
        return {
            "schema": SCHEMA,
            "settings": asdict(self.settings),
            "settings_sha256": self.settings.fingerprint,
            "records": ordered,
            "groups": public_groups,
            "styles": dict(self.styles),
            "physical_transition_confirmed": False,
            "qualifications": [
                "Markers describe observed-pattern compatibility, not chemical phase identities.",
                "State recurrence is an advisory extension of the calibrated pair detector.",
                "Every assigned group member must agree in both directions; ambiguous scans remain unassigned.",
                "A comparison reference can be established after two compatible scans while its relation to earlier references remains unresolved; this is not a transition alert.",
            ],
        }
