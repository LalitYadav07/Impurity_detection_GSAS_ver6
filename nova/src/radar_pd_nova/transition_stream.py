"""Causal reference-based grouping under a user-selected indicator rule."""
from dataclasses import asdict

from .transition_input import digest


class SignalStream:
    def __init__(self, engine):
        self.engine = engine
        self.policy = engine.policy
        self.series = {}
        self.records = []
        self.points = 0

    def observe(self, scan, item, prepared):
        group = digest(asdict(item.pattern.scope))
        series = self.series.setdefault(group, {"scope": asdict(item.pattern.scope),
            "states": [], "active": None, "previous": None, "first": None, "run_ids": []})
        series["run_ids"].append(str(scan["run_id"]))
        previous = series["previous"]
        row = {"run_id": str(scan["run_id"]), "pattern_sha256": item.fingerprint,
            "group": group, "source": item.provenance, "notices": item.notices,
            "state_id": None, "style_index": None, "event": "unassigned",
            "reason": "insufficient_available_selected_indicators", "representative": None,
            "previous": previous.observation.observation_id if previous is not None else None,
            "previous_comparison": None, "anchor_comparison": None,
            "boundary_comparison": None, "boundary": False, "boundary_after": None,
            "relation_to_other_references": "unresolved", "policy_sha256": self.policy.fingerprint,
            "order": len(self.records), "provisional": False}
        self.records.append(row)
        if not self.policy.selected:
            row["reason"] = "no_boundary_indicators_selected"
            return row
        if self.engine.pair(prepared, prepared)["status"] != "no_resolved_change":
            row["reason"] = "insufficient_reproducible_feature_support"
            return row
        active = series["active"]
        chosen = None
        if active is None:
            event = "initial"
        else:
            row["previous_comparison"] = self.engine.compare_evidence(previous, prepared)
            first = series["first"]
            row["anchor_comparison"] = self.engine.compare_evidence(first, prepared)
            baseline = previous if self.policy.comparison == "previous" else active["reference"]
            decision = self.engine.compare_evidence(baseline, prepared)
            row["boundary_comparison"] = decision
            if decision["status"] == "no_resolved_change":
                chosen, event = active, "continued"
            elif decision["status"] == "transition_evidence":
                row["boundary"] = True
                row["boundary_after"] = row["previous"]
                # Adjacent comparisons can accumulate small changes within a
                # group. A jump back may differ from the preceding scan but
                # match the active group's original reference. Include it in
                # recurrence searches as well as all other earlier groups.
                matches = [s for s in series["states"]
                           if self.engine.compare_evidence(s["reference"], prepared)["status"] == "no_resolved_change"]
                if len(matches) == 1:
                    chosen, event = matches[0], "returned"
                elif len(matches) > 1:
                    row["reason"] = "ambiguous_between_existing_pattern_references"
                    series["previous"] = prepared
                    return row
                else:
                    event = "new"
            else:
                row["reason"] = decision.get("reasons", [row["reason"]])[0]
                series["previous"] = prepared
                return row
        if chosen is None:
            if len(series["states"]) >= 32 or self.points + len(prepared.observation.x) > 2_000_000:
                row["reason"] = "state_memory_capacity_reached"
                series["previous"] = prepared
                return row
            # Rwp can distinguish model states of identical measured curves.
            # The originating acquisition makes those states distinct without
            # depending on future scans or the size of the received sequence.
            state_id = digest([group, item.fingerprint, str(scan["run_id"]), self.policy.fingerprint])
            chosen = {"id": state_id, "reference": prepared, "style": len(series["states"])}
            series["states"].append(chosen)
            self.points += len(prepared.observation.x)
        row.update(state_id=chosen["id"], style_index=chosen["style"], event=event, reason="",
                   representative=chosen["reference"].observation.observation_id,
                   relation_to_other_references="selected_reference_rule")
        self.engine.styles[group[:32] + ":" + chosen["id"]] = chosen["style"]
        series["active"] = chosen
        series["previous"] = prepared
        if series["first"] is None:
            series["first"] = prepared
        return row

    def report(self):
        return {"schema": "radar-pd-online-boundaries/v2", "records": list(self.records),
            "groups": [{"id": k, "scope": v["scope"], "run_ids": list(v["run_ids"])} for k, v in self.series.items()],
            "styles": dict(self.engine.styles), "boundary_policy": asdict(self.policy),
            "policy_sha256": self.policy.fingerprint, "physical_transition_confirmed": False,
            "settings": asdict(self.engine.settings), "settings_sha256": self.engine.settings.fingerprint,
            "qualifications": [
                "Assignments use only acquisitions already received, within compatible specimen/bank/setup scopes.",
                "Matching symbols mean agreement with the selected comparison baseline under the displayed rule.",
                "Indicator counts are a user rule, not independent statistical votes or phase probabilities.",
                "Width changes assume fixed instrumental/sample broadening; local profile errors are conditional.",
            ]}
