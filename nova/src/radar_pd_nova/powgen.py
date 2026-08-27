"""Strict POWGEN filename and instrument-profile resolution.

The resolver uses a bounded registry and never guesses a cycle, chooses a
nearby setting, or breaks a tie between equally valid files or profile rules.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from hmac import compare_digest
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


_RUN_RE = re.compile(r"^PG3_(?P<run>\d+)(?=$|[-.])", re.IGNORECASE)
_REDUCED_RE = re.compile(
    r"^PG3_(?P<run>\d+)(?:-(?P<bank>\d+))?"
    r"(?P<suffix>\.gsa|\.xye|\.dat)$",
    re.IGNORECASE,
)

_FORMAT_PRIORITY = {".gsa": 0, ".xye": 1, ".dat": 2}

_PACKAGE_DATA_ROOT = Path(__file__).resolve().parent / "data"
_DEFAULT_REGISTRY_PATH = _PACKAGE_DATA_ROOT / "powgen_instrument_profiles.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PowgenResolutionError(ValueError):
    """Base error for strict POWGEN resolution failures."""


class UnknownPowgenInputError(PowgenResolutionError):
    """Raised when no explicit file or profile rule matches."""


class AmbiguousPowgenInputError(PowgenResolutionError):
    """Raised when more than one equally valid result matches."""


@dataclass(frozen=True)
class PowgenReducedFile:
    """Parsed metadata for one supported POWGEN reduced data file."""

    path: str
    filename: str
    run_number: int
    bank: int | None
    suffix: str


@dataclass(frozen=True)
class PowgenProfileProvenance:
    """The exact registry rule and observations used for a profile match."""

    instrument: str
    rule_id: str
    cycle: str
    run_number: int
    run_min: int
    run_max: int | None
    wavelength_angstrom: str
    frequency_hz: str
    registry_schema_version: int
    registry_source_url: str
    registry_sha256: str


@dataclass(frozen=True)
class PowgenProfileResolution:
    """Resolved instrument profile filename with reproducible provenance."""

    profile_filename: str
    profile_resource_filename: str
    profile_sha256: str
    provenance: PowgenProfileProvenance

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_filename": self.profile_filename,
            "profile_resource_filename": self.profile_resource_filename,
            "profile_sha256": self.profile_sha256,
            "provenance": asdict(self.provenance),
        }


@dataclass(frozen=True)
class _ProfileRule:
    rule_id: str
    cycle: str
    run_min: int
    run_max: int | None
    wavelength: Decimal
    frequency: Decimal
    filename: str
    resource_filename: str
    profile_sha256: str


def _basename(value: str | Path) -> str:
    return str(value).replace("\\", "/").rsplit("/", 1)[-1]


def parse_pg3_run_number(value: str | Path) -> int:
    """Extract a positive PG3 run number from a raw or reduced filename."""

    filename = _basename(value)
    match = _RUN_RE.match(filename)
    if match is None or int(match.group("run")) <= 0:
        raise UnknownPowgenInputError(
            f"Could not parse a PG3 run number from {filename!r}"
        )
    return int(match.group("run"))


def parse_powgen_reduced_filename(value: str | Path) -> PowgenReducedFile:
    """Parse a conventionally named GSA, XYE, or DAT POWGEN reduced file."""

    path = str(value)
    filename = _basename(value)
    match = _REDUCED_RE.fullmatch(filename)
    if match is None:
        raise UnknownPowgenInputError(
            "Unsupported POWGEN reduced filename "
            f"{filename!r}; expected PG3_<run>[-<bank>].gsa/.xye/.dat"
        )

    run_number = int(match.group("run"))
    bank_text = match.group("bank")
    bank = int(bank_text) if bank_text is not None else None
    if run_number <= 0 or bank == 0:
        raise UnknownPowgenInputError(
            f"Unsupported POWGEN reduced filename {filename!r}"
        )

    return PowgenReducedFile(
        path=path,
        filename=filename,
        run_number=run_number,
        bank=bank,
        suffix=match.group("suffix").lower(),
    )


def _exact_decimal(value: Any, field: str) -> Decimal:
    if isinstance(value, bool):
        raise UnknownPowgenInputError(f"{field} must be a numeric value")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise UnknownPowgenInputError(f"{field} must be a numeric value") from exc
    if not result.is_finite():
        raise UnknownPowgenInputError(f"{field} must be finite")
    return result


def _positive_integer(value: Any, field: str) -> int:
    decimal = _exact_decimal(value, field)
    if isinstance(value, bool) or decimal != decimal.to_integral_value():
        raise UnknownPowgenInputError(f"{field} must be an integer")
    result = int(decimal)
    if result <= 0:
        raise UnknownPowgenInputError(f"{field} must be a positive integer")
    return result


def select_preferred_reduced_file(
    candidates: Sequence[str | Path], *, run_number: int | None = None
) -> PowgenReducedFile:
    """Select one reduced file, preferring GSA and rejecting every tie.

    Unsupported filenames are ignored. Without an explicit ``run_number``,
    all supported candidates must refer to exactly one run.
    """

    target_run = (
        _positive_integer(run_number, "run_number")
        if run_number is not None
        else None
    )
    parsed: list[PowgenReducedFile] = []
    for candidate in candidates:
        try:
            item = parse_powgen_reduced_filename(candidate)
        except UnknownPowgenInputError:
            continue
        if target_run is None or item.run_number == target_run:
            parsed.append(item)

    if not parsed:
        target = f" for run {target_run}" if target_run is not None else ""
        raise UnknownPowgenInputError(f"No supported POWGEN reduced files{target}")

    runs = {item.run_number for item in parsed}
    if target_run is None and len(runs) != 1:
        raise AmbiguousPowgenInputError(
            f"Reduced-file candidates contain multiple PG3 runs: {sorted(runs)}"
        )

    best_priority = min(_FORMAT_PRIORITY[item.suffix] for item in parsed)
    matches = [
        item for item in parsed if _FORMAT_PRIORITY[item.suffix] == best_priority
    ]
    if len(matches) != 1:
        names = sorted(item.path for item in matches)
        raise AmbiguousPowgenInputError(
            "Multiple equally preferred POWGEN reduced files matched: "
            + ", ".join(names)
        )
    return matches[0]


def _canonical_json(payload: Mapping[str, Any], location: str) -> bytes:
    try:
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise PowgenResolutionError(
            f"Invalid POWGEN profile registry at {location}"
        ) from exc


def _canonical_text_sha256(raw: bytes) -> str:
    """Hash text resources independently of platform newline conversion."""

    canonical = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return sha256(canonical).hexdigest()


def _registry_payload(
    registry: Mapping[str, Any] | str | Path | None,
) -> tuple[dict[str, Any], bytes, str]:
    if registry is None:
        registry = _DEFAULT_REGISTRY_PATH
    if isinstance(registry, Mapping):
        raw = _canonical_json(registry, "in-memory registry")
        return json.loads(raw), raw, "in-memory registry"

    path = Path(registry)
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PowgenResolutionError(
            f"Invalid POWGEN profile registry at {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise PowgenResolutionError(f"Invalid POWGEN profile registry at {path}")
    return payload, raw, str(path)


def _required_text(value: Any, field: str, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PowgenResolutionError(
            f"Invalid {field} in POWGEN profile registry at {location}"
        )
    return value


def _resource_filename(value: Any, filename: str, location: str) -> str:
    resource = _required_text(value, "resource_filename", location)
    resource_path = PurePosixPath(resource)
    if (
        resource_path.is_absolute()
        or "\\" in resource
        or resource_path.parts != ("powgen_profiles", filename)
    ):
        raise PowgenResolutionError(
            f"Invalid resource_filename in POWGEN profile registry at {location}"
        )
    return resource


def _profile_sha256(value: Any, location: str) -> str:
    digest = _required_text(value, "sha256", location).lower()
    if _SHA256_RE.fullmatch(digest) is None:
        raise PowgenResolutionError(
            f"Invalid sha256 in POWGEN profile registry at {location}"
        )
    return digest


def _registry_decimal(value: Any, field: str, location: str) -> Decimal:
    try:
        result = _exact_decimal(value, field)
    except UnknownPowgenInputError as exc:
        raise PowgenResolutionError(
            f"Invalid {field} in POWGEN profile registry at {location}"
        ) from exc
    if result <= 0:
        raise PowgenResolutionError(
            f"Invalid {field} in POWGEN profile registry at {location}"
        )
    return result


def _registry_integer(value: Any, field: str, location: str) -> int:
    decimal = _registry_decimal(value, field, location)
    if isinstance(value, bool) or decimal != decimal.to_integral_value():
        raise PowgenResolutionError(
            f"Invalid {field} in POWGEN profile registry at {location}"
        )
    return int(decimal)


def _validate_registry(
    payload: Mapping[str, Any], location: str
) -> tuple[int, str, str, list[_ProfileRule]]:
    try:
        schema_version = _registry_integer(
            payload["schema_version"], "schema_version", location
        )
        instrument = _required_text(payload["instrument"], "instrument", location)
        source = payload["source"]
        profiles = payload["profiles"]
    except KeyError as exc:
        raise PowgenResolutionError(
            f"Invalid POWGEN profile registry at {location}"
        ) from exc

    if schema_version != 1 or instrument.upper() != "PG3":
        raise PowgenResolutionError(f"Invalid POWGEN profile registry at {location}")
    if not isinstance(source, Mapping) or not isinstance(profiles, list):
        raise PowgenResolutionError(f"Invalid POWGEN profile registry at {location}")
    try:
        source_url = _required_text(source["url"], "source.url", location)
    except KeyError as exc:
        raise PowgenResolutionError(
            f"Invalid POWGEN profile registry at {location}"
        ) from exc

    rules: list[_ProfileRule] = []
    for rule in profiles:
        if not isinstance(rule, Mapping):
            raise PowgenResolutionError(
                f"Invalid POWGEN profile rule in {location}"
            )
        try:
            rule_id = _required_text(rule["id"], "rule id", location)
            cycle = _required_text(rule["cycle"], "cycle", location)
            run_min = _registry_integer(rule["run_min"], "run_min", location)
            run_max_raw = rule.get("run_max")
            run_max = (
                _registry_integer(run_max_raw, "run_max", location)
                if run_max_raw is not None
                else None
            )
            wavelength = _registry_decimal(
                rule["wavelength_angstrom"], "wavelength_angstrom", location
            )
            frequency = _registry_decimal(
                rule["frequency_hz"], "frequency_hz", location
            )
            filename = _required_text(rule["filename"], "filename", location)
            resource_filename = _resource_filename(
                rule["resource_filename"], filename, location
            )
            profile_sha256 = _profile_sha256(rule["sha256"], location)
        except KeyError as exc:
            raise PowgenResolutionError(
                f"Invalid POWGEN profile rule in {location}"
            ) from exc
        if run_max is not None and run_max < run_min:
            raise PowgenResolutionError(
                f"Invalid POWGEN profile rule in {location}: run_max < run_min"
            )
        if _basename(filename) != filename or not filename.lower().endswith(".instprm"):
            raise PowgenResolutionError(
                f"Invalid POWGEN profile filename in {location}"
            )
        rules.append(
            _ProfileRule(
                rule_id=rule_id,
                cycle=cycle,
                run_min=run_min,
                run_max=run_max,
                wavelength=wavelength,
                frequency=frequency,
                filename=filename,
                resource_filename=resource_filename,
                profile_sha256=profile_sha256,
            )
        )
    return schema_version, instrument.upper(), source_url, rules


def resolve_powgen_profile(
    *,
    run_number: int,
    wavelength_angstrom: float | str | Decimal,
    frequency_hz: float | str | Decimal,
    registry: Mapping[str, Any] | str | Path | None = None,
) -> PowgenProfileResolution:
    """Resolve one exact POWGEN profile registry rule or fail closed."""

    run = _positive_integer(run_number, "run_number")
    observed_wavelength = _exact_decimal(wavelength_angstrom, "wavelength_angstrom")
    observed_frequency = _exact_decimal(frequency_hz, "frequency_hz")
    payload, raw, registry_location = _registry_payload(registry)
    schema_version, instrument, source_url, rules = _validate_registry(
        payload, registry_location
    )

    matches = [
        rule
        for rule in rules
        if run >= rule.run_min
        and (rule.run_max is None or run <= rule.run_max)
        and observed_wavelength == rule.wavelength
        and observed_frequency == rule.frequency
    ]
    if not matches:
        raise UnknownPowgenInputError(
            "No explicit POWGEN profile rule matches "
            f"run={run}, wavelength={observed_wavelength} A, "
            f"frequency={observed_frequency} Hz"
        )
    if len(matches) != 1:
        ids = sorted(rule.rule_id for rule in matches)
        raise AmbiguousPowgenInputError(
            "Multiple POWGEN profile rules match exactly: " + ", ".join(ids)
        )

    rule = matches[0]
    return PowgenProfileResolution(
        profile_filename=rule.filename,
        profile_resource_filename=rule.resource_filename,
        profile_sha256=rule.profile_sha256,
        provenance=PowgenProfileProvenance(
            instrument=instrument,
            rule_id=rule.rule_id,
            cycle=rule.cycle,
            run_number=run,
            run_min=rule.run_min,
            run_max=rule.run_max,
            wavelength_angstrom=str(rule.wavelength),
            frequency_hz=str(rule.frequency),
            registry_schema_version=schema_version,
            registry_source_url=source_url,
            registry_sha256=sha256(raw).hexdigest(),
        ),
    )


def resolve_packaged_powgen_profile_path(
    resolution: PowgenProfileResolution,
    *,
    package_data_root: str | Path | None = None,
) -> Path:
    """Return a verified packaged ``.instprm`` path for a resolution.

    The registry resource is constrained to the package's ``powgen_profiles``
    directory. The file must exist, remain inside that directory after path
    resolution, match the public profile filename, and have the registered
    SHA-256 digest. Profile digests use LF-normalized text so Git checkouts on
    Windows and Linux verify the same scientific profile.
    """

    root = Path(package_data_root) if package_data_root is not None else _PACKAGE_DATA_ROOT
    try:
        root = root.resolve(strict=True)
    except OSError as exc:
        raise PowgenResolutionError(
            f"POWGEN package data directory is unavailable: {root}"
        ) from exc

    resource = resolution.profile_resource_filename
    resource_path = PurePosixPath(resource)
    if (
        resource_path.is_absolute()
        or "\\" in resource
        or resource_path.parts
        != ("powgen_profiles", resolution.profile_filename)
    ):
        raise PowgenResolutionError(
            f"Unsafe packaged POWGEN profile resource: {resource!r}"
        )

    profile_root = (root / "powgen_profiles").resolve()
    candidate = root.joinpath(*resource_path.parts)
    try:
        candidate = candidate.resolve(strict=True)
        candidate.relative_to(profile_root)
    except (OSError, ValueError) as exc:
        raise PowgenResolutionError(
            f"Packaged POWGEN profile is unavailable: {resource}"
        ) from exc
    if not candidate.is_file():
        raise PowgenResolutionError(
            f"Packaged POWGEN profile is not a file: {resource}"
        )

    try:
        digest = _canonical_text_sha256(candidate.read_bytes())
    except OSError as exc:
        raise PowgenResolutionError(
            f"Could not read packaged POWGEN profile: {resource}"
        ) from exc
    if not compare_digest(digest, resolution.profile_sha256.lower()):
        raise PowgenResolutionError(
            "Packaged POWGEN profile checksum mismatch for "
            f"{resolution.profile_filename}: expected "
            f"{resolution.profile_sha256}, got {digest}"
        )
    return candidate


__all__ = [
    "AmbiguousPowgenInputError",
    "PowgenProfileProvenance",
    "PowgenProfileResolution",
    "PowgenReducedFile",
    "PowgenResolutionError",
    "UnknownPowgenInputError",
    "parse_pg3_run_number",
    "parse_powgen_reduced_filename",
    "resolve_packaged_powgen_profile_path",
    "resolve_powgen_profile",
    "select_preferred_reduced_file",
]
