"""Bounded public POWGEN metadata for reproducible calibration-corpus preparation.

The service uses PG3 as its instrument ID and ranges_q for run selection. Its
limit parameter is not relied upon. Explicit run/proposal filters, byte/record
budgets and returned acquisition identities are checked locally. No credentials
are read or renewed, and detector/monitor counts and powder arrays are excluded.
Authenticated access, when needed, remains with the facility's approved helper.

Example: python -m gpx_recipe.planning.public_catalog --runs 49998-49999 --output NEW_DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .corpus import identity, quantity, summarize_run

ENDPOINT = "https://oncat.ornl.gov/api/datafiles"
CORE_FIELDS = ("id", "location", "indexed", "experiment", "instrument", "size", "modified", "checksum")
MINIMAL_FIELDS = CORE_FIELDS + ("metadata.entry.title", "metadata.entry.start_time",
                               "metadata.entry.sample.name", "metadata.entry.sample.chemical_formula")
DEFAULT_FIELDS = CORE_FIELDS + (
    "metadata.entry.title", "metadata.entry.start_time", "metadata.entry.end_time",
    "metadata.entry.duration", "metadata.entry.duration@units", "metadata.entry.proton_charge",
    "metadata.entry.proton_charge@units", "metadata.entry.sample",
    "metadata.entry.DASlogs.ProtonBeamEnergy", "metadata.entry.DASlogs.LambdaRequest",
    "metadata.entry.DASlogs.Guide_Position", "metadata.entry.DASlogs.SampleTemp",
    "metadata.entry.DASlogs.SampleEnvironment", "metadata.entry.DASlogs.frequency")
ALLOWED_FIELDS = frozenset(DEFAULT_FIELDS + MINIMAL_FIELDS)


class CatalogError(ValueError):
    """A query, response or metadata identity failed the declared contract."""


def _positive_int(value, label):
    if type(value) is not int or value <= 0:
        raise CatalogError(f"A positive integer {label} is required")
    return value


def build_query(*, runs=(), run_range=None, experiment=None, projection=DEFAULT_FIELDS):
    """Build an explicitly bounded, metadata-only query; never an unfiltered instrument scan."""
    runs = tuple(runs)
    if runs and run_range is not None:
        raise CatalogError("Use explicit runs or one inclusive run range")
    if not runs and run_range is None and experiment is None:
        raise CatalogError("Explicit run numbers, run range or proposal is required")
    if len(runs) > 1000:
        raise CatalogError("Split an explicit run list into at most 1000 IDs")
    for run in runs:
        _positive_int(run, "run number")
    if len(set(runs)) != len(runs):
        raise CatalogError("Duplicate requested run numbers")
    query = {"facility": "SNS", "instrument": "PG3", "tags": "type/raw"}
    if runs:
        query["ranges_q"] = "indexed.run_number:" + ",".join(map(str, sorted(runs)))
    if run_range is not None:
        if len(run_range) != 2:
            raise CatalogError("Two inclusive run bounds are required")
        low, high = run_range
        _positive_int(low, "lower run bound"); _positive_int(high, "upper run bound")
        if high < low or high - low >= 10000:
            raise CatalogError("Use an ordered range of at most 10000 run IDs")
        query["ranges_q"] = f"indexed.run_number:{low}-{high}"
    if experiment is not None:
        if not isinstance(experiment, str) or not re.fullmatch(r"IPTS-\d+", experiment):
            raise CatalogError("A literal IPTS-number proposal is required")
        query["experiment"] = experiment
    projection = tuple(projection)
    if not projection or set(projection) - ALLOWED_FIELDS:
        raise CatalogError("Only declared input/provenance metadata fields may be requested")
    if not {"location", "indexed"} <= set(projection):
        raise CatalogError("Location and indexed run identity must be returned")
    query["projection"] = list(projection)
    return query


def record_identity(record):
    if not isinstance(record, dict) or not isinstance(record.get("indexed"), dict):
        raise CatalogError("Returned POWGEN file and indexed identity must be objects")
    location = record.get("location", "")
    if not isinstance(location, str) or any(p in (".", "..") for p in PurePosixPath(location).parts):
        raise CatalogError("An unambiguous archive location is required")
    match = re.fullmatch(r"/SNS/PG3/(IPTS-\d+)/(?:[A-Za-z0-9_-]+/)*PG3_(\d+)(?:_(?:event|histo))?\.nxs(?:\.h5)?", location)
    run = record.get("indexed", {}).get("run_number")
    if match is None or type(run) is not int or int(match[2]) != run:
        raise CatalogError("Returned archive path and indexed POWGEN run disagree")
    if record.get("instrument", "PG3") != "PG3":
        raise CatalogError("Returned instrument is not PG3")
    if record.get("experiment", match[1]) != match[1]:
        raise CatalogError("Returned proposal and archive path disagree")
    return match[1], run


def validate_records(records, query, maximum_records):
    _positive_int(maximum_records, "metadata record budget")
    if not isinstance(records, list) or len(records) > maximum_records:
        raise CatalogError("Metadata record budget or response schema failed")
    selector = query.get("ranges_q", "").removeprefix("indexed.run_number:")
    bounds = tuple(map(int, selector.split("-"))) if "-" in selector else None
    selected = set(map(int, selector.split(","))) if selector and bounds is None else None
    locations = set()
    for record in records:
        proposal, run = record_identity(record)
        if query.get("experiment", proposal) != proposal:
            raise CatalogError("Server returned a proposal outside the requested scope")
        if selected is not None and run not in selected:
            raise CatalogError("Server returned an unrequested run")
        if bounds is not None and not bounds[0] <= run <= bounds[1]:
            raise CatalogError("Server returned a run outside the requested range")
        if record["location"] in locations:
            raise CatalogError("Duplicate returned metadata file identity")
        locations.add(record["location"])
    return records


def fetch_metadata(query, *, maximum_records=10000, maximum_bytes=5*1024**2, timeout=25):
    """Return the checked public records and a receipt, without using a login/cache."""
    _positive_int(maximum_bytes, "response byte budget")
    _positive_int(maximum_records, "metadata record budget")
    # Reconstruct the allowed key/field contract even for programmatic callers.
    if set(query) - {"facility", "instrument", "tags", "ranges_q", "experiment", "projection"}:
        raise CatalogError("Unsupported catalog query fields")
    if query.get("facility") != "SNS" or query.get("instrument") != "PG3" or query.get("tags") != "type/raw":
        raise CatalogError("Only explicit raw-file POWGEN metadata may be queried")
    if not query.get("ranges_q") and not query.get("experiment"):
        raise CatalogError("An explicit run/proposal scope is required")
    selector = query.get("ranges_q")
    selected = ()
    run_range = None
    if selector is not None:
        if not isinstance(selector, str) or not re.fullmatch(r"indexed\.run_number:(?:\d+(?:,\d+)*|\d+-\d+)", selector):
            raise CatalogError("Use an explicit indexed run selector")
        values = selector.removeprefix("indexed.run_number:")
        if "-" in values:run_range = tuple(map(int, values.split("-")))
        else:selected = tuple(map(int, values.split(",")))
    canonical = build_query(runs=selected, run_range=run_range, experiment=query.get("experiment"),
                            projection=query.get("projection", ()))
    if query != canonical:
        raise CatalogError("Use the canonical metadata query contract")
    url = ENDPOINT + "?" + urlencode(query, doseq=True)
    try:
        with urlopen(Request(url, headers={"User-Agent": "RADAR-PD-POWGEN-metadata/1"}), timeout=timeout) as response:
            body = response.read(maximum_bytes + 1)
            if len(body) > maximum_bytes:
                raise CatalogError("Metadata response exceeded the declared byte budget")
        records = json.loads(body)
    except HTTPError as error:
        raise CatalogError(f"Public catalog HTTP status {error.code}") from None
    except (URLError, TimeoutError, json.JSONDecodeError):
        raise CatalogError("Public catalog transport or JSON response failed") from None
    validate_records(records, query, maximum_records)
    acquisitions = {record_identity(r) for r in records}
    requested = set(range(run_range[0],run_range[1]+1)) if run_range else set(selected)
    missing = sorted(requested - {run for _,run in acquisitions})
    return records, {"schema": "powgen-public-metadata-receipt/v1", "query": query, "request_url": url,
        "response_bytes": len(body), "response_sha256": hashlib.sha256(body).hexdigest(),
        "returned_records_sha256": identity(records),
        "metadata_files": len(records), "distinct_acquisition_labels": len(acquisitions),
        "requested_runs_without_returned_metadata": missing,
        "credentials_read_or_renewed": False, "neutron_observation_arrays_requested": False,
        "server_limit_parameter_not_used_as_a_budget": True, "returned_scope_checked": True,
        "physical_specimen_independence_verified": False, "experimental_accuracy_certified": False}


def inventory_record(record):
    """Adapt public input metadata to the existing conservative corpus contract."""
    proposal, run = record_identity(record)
    metadata = record.get("metadata") or {}
    entry = metadata.get("entry") or {}
    sample = entry.get("sample") or {}
    fields = {}
    for key in ("title", "start_time", "end_time", "duration", "proton_charge"):
        if key in entry:
            fields["entry/" + key] = {"value": entry[key], "units": entry.get(key + "@units")}
    for key, value in sample.items():
        if not key.endswith("@units"):
            fields["entry/sample/" + key] = {"value": value, "units": sample.get(key + "@units")}
    logs = {}
    if sample.get("identifier") is not None:
        logs["SampleId"] = {"value": sample["identifier"]}
    for name in ("ProtonBeamEnergy", "LambdaRequest", "Guide_Position", "SampleEnvironment", "frequency"):
        raw = (entry.get("DASlogs") or {}).get(name) or {}
        value = raw.get("average_value")
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            logs[name] = {"value": value, "units": raw.get("average_value@units")}
    raw = (entry.get("DASlogs") or {}).get("SampleTemp") or {}
    temperature_issue = False
    if "minimum_value" in raw and "maximum_value" in raw:
        # An average is not evidence that a cooling/ramping run was isothermal.
        low, high = raw["minimum_value"], raw["maximum_value"]
        low_unit = str(raw.get("minimum_value@units", "")).strip().lower()
        high_unit = str(raw.get("maximum_value@units", "")).strip().lower()
        valid = (type(low) in (int, float) and type(high) in (int, float) and
                 math.isfinite(low) and math.isfinite(high) and 0 < low <= high and
                 low_unit in {"k", "kelvin"} and high_unit in {"k", "kelvin"})
        if valid:
            logs["SampleTemp"] = {"min": low, "max": high, "units": "K", "sampled": True}
        else:
            temperature_issue = True
    issues = ["Public metadata only; exact reduction, illumination and reference identity require review"]
    if temperature_issue:
        issues.append("Catalog temperature extrema or their units are inconsistent")
    if str(entry.get("start_time", "")).startswith(("1989-12-31", "1990-01-01")):
        issues.append("Acquisition timestamp is an epoch placeholder")
    return {"type": "run", "ipts": proposal, "run": run, "metadata": fields, "logs": logs,
        "products": [], "reduction": {"calls": []}, "issues": issues,
        "metadata_sha256": identity(record), "catalog_location": record["location"]}


def metadata_summary(record):
    inventory = inventory_record(record)
    result = summarize_run(inventory)
    energy = inventory["logs"].get("ProtonBeamEnergy")
    result["recorded_proton_energy_gev"] = quantity(energy, {"mev": .001, "gev": 1.})
    result["metadata_source"] = "public_ONCat_input_projection"
    result["mass_independently_verified"] = False
    result["actual_illumination_verified"] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--runs", help="Comma-separated IDs or one inclusive first-last range")
    scope.add_argument("--experiment", help="One literal IPTS-number proposal")
    parser.add_argument("--minimal-index", action="store_true")
    parser.add_argument("--maximum-records", type=int, default=10000)
    parser.add_argument("--maximum-bytes", type=int, default=5*1024**2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runs and "-" in args.runs:
        query = build_query(run_range=tuple(map(int, args.runs.split("-"))),
                            projection=MINIMAL_FIELDS if args.minimal_index else DEFAULT_FIELDS)
    else:
        query = build_query(runs=tuple(map(int, args.runs.split(","))) if args.runs else (),
                            experiment=args.experiment,
                            projection=MINIMAL_FIELDS if args.minimal_index else DEFAULT_FIELDS)
    args.output.mkdir(parents=True, exist_ok=False)
    write = lambda name, data: (args.output/name).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write("request.json", {"query": query, "maximum_records": args.maximum_records,
        "maximum_bytes": args.maximum_bytes, "source_sha256": source_hash})
    try:
        records, receipt = fetch_metadata(query, maximum_records=args.maximum_records, maximum_bytes=args.maximum_bytes)
        write("records.json", records)
        receipt["saved_records_file_sha256"] = hashlib.sha256((args.output/"records.json").read_bytes()).hexdigest()
        summaries = [metadata_summary(r) for r in records]
        write("summaries.json", summaries)
        receipt.update(source_sha256=source_hash, completed=True)
        write("receipt.json", receipt)
        print(json.dumps({k:v for k,v in receipt.items() if k not in ("query", "request_url")}))
    except CatalogError as error:
        write("failure.json", {"completed": False, "reason": str(error), "source_sha256": source_hash})
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
