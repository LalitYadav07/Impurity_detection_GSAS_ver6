"""Summarize reported planner tokens and estimate public Luna list-rate cost.

The provider's usage is authoritative for token counts. This module never
estimates tokens from text length and never treats an absent response as free.
"""
from __future__ import annotations

from decimal import Decimal


LUNA_RATE_SOURCE = "https://developers.openai.com/api/docs/models/gpt-6-luna"
LUNA_RATE_DATE = "2026-10-28"
LUNA_USD_PER_MILLION = {
    "input": Decimal("0.10"),
    "cached_input": Decimal("0.01"),
    "cache_write": Decimal("0.125"),
    "output": Decimal("0.50"),
}


def _count(value):
    return value if type(value) is int and 0 <= value <= 1_000_000_000 else None


def _reported_usage(call):
    usage = call.get("usage") if isinstance(call, dict) else None
    if not isinstance(usage, dict):
        return None
    incoming, outgoing = _count(usage.get("input_tokens")), _count(usage.get("output_tokens"))
    details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    if incoming is None or outgoing is None or not isinstance(details, dict) or not isinstance(output_details, dict):
        return None
    cached = _count(details.get("cached_tokens", 0))
    writes = _count(details.get("cache_write_tokens", 0))
    reasoning = _count(output_details.get("reasoning_tokens", 0))
    total = _count(usage.get("total_tokens", incoming + outgoing))
    if None in (cached, writes, reasoning, total) or cached + writes > incoming or reasoning > outgoing or total != incoming + outgoing:
        return None
    return dict(input_tokens=incoming, cached_input_tokens=cached, cache_write_tokens=writes,
                output_tokens=outgoing, reasoning_output_tokens=reasoning, total_tokens=total)


def _luna_estimate(usage):
    rates = LUNA_USD_PER_MILLION
    normal = usage["input_tokens"] - usage["cached_input_tokens"] - usage["cache_write_tokens"]
    input_multiplier = Decimal(2) if usage["input_tokens"] > 272_000 else Decimal(1)
    output_multiplier = Decimal("1.5") if usage["input_tokens"] > 272_000 else Decimal(1)
    return (input_multiplier * (normal * rates["input"] +
            usage["cached_input_tokens"] * rates["cached_input"] +
            usage["cache_write_tokens"] * rates["cache_write"]) +
            output_multiplier * usage["output_tokens"] * rates["output"]) / 1_000_000


def summarize_calls(calls):
    """Return a safe, JSON-ready aggregate over *all* model attempts."""
    calls = list(calls or [])
    result = dict(schema="radar-gpx-model-usage/v1", calls=len(calls), reported_calls=0,
                  input_tokens=0, cached_input_tokens=0, cache_write_tokens=0,
                  output_tokens=0, reasoning_output_tokens=0, total_tokens=0,
                  estimated_usd=None, price_basis=None)
    estimated = Decimal(0)
    luna_only = bool(calls)
    for call in calls:
        usage = _reported_usage(call)
        luna = isinstance(call, dict) and call.get("provider") == "illumine-luna" and call.get("model") == "gpt-6-luna"
        luna_only &= luna
        if usage is None:
            continue
        result["reported_calls"] += 1
        for key, value in usage.items():
            result[key] += value
        if luna:
            estimated += _luna_estimate(usage)
    if luna_only and result["reported_calls"] == len(calls):
        result["estimated_usd"] = float(estimated)
        result["price_basis"] = dict(model="gpt-6-luna", currency="USD", rate_date=LUNA_RATE_DATE,
                                     source=LUNA_RATE_SOURCE, basis="public Standard API list rates; actual ORNL Azure charges may differ")
    return result


def usage_labels(summary):
    """Compact text for the recipe review pane; never turn unknown usage into zero."""
    if not isinstance(summary, dict) or not summary.get("calls"):
        return "", ""
    calls, reported = summary["calls"], summary.get("reported_calls", 0)
    if not reported:
        return f"{calls} model call(s) · token usage unavailable", ""
    tokens = (f"Model usage · {summary['input_tokens']:,} input"
              f" ({summary['cached_input_tokens']:,} cached, {summary['cache_write_tokens']:,} cache writes)"
              f" + {summary['output_tokens']:,} output = {summary['total_tokens']:,} tokens")
    if reported != calls:
        return tokens + f" · {reported} of {calls} calls reported usage", "Full cost unavailable because one or more calls did not report usage."
    cost = summary.get("estimated_usd")
    if cost is None:
        return tokens, "Cost estimate unavailable for this provider."
    amount = "<$0.000001" if 0 < cost < 0.000001 else f"${cost:.6f}"
    return tokens, f"Approximate Luna list-rate cost: {amount} USD. Actual ORNL Azure charges may differ."
