"""Bounded, observed-only inputs for live transition tracking.

Facility products are opened read-only, only below the watch's authorized root.
No project is unpickled and no detector/event data are read. Fitted arrays and
phase identities never enter the statistical detector.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

from .pattern_transition import MAX_POINTS, ObservedPattern, PatternScope
from .scan_interpretation import Pattern, load_pattern


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


@dataclass
class TrackingObservation:
    pattern: ObservedPattern
    fingerprint: str
    provenance: dict
    notices: list[str]


def observation(pattern, provenance=None, notices=()):
    h = sha256(json.dumps(asdict(pattern.scope), sort_keys=True).encode())
    for values in (pattern.x, pattern.intensity, pattern.sigma, pattern.valid):
        h.update(np.asarray(values).tobytes())
    return TrackingObservation(pattern, h.hexdigest(), provenance or {}, list(notices))


def identity(scan):
    metadata = scan.get("metadata") or {}
    specimen = str(metadata.get("sample_id") or "").strip()
    if not specimen or specimen.lower() in {"none", "unknown", "n/a"}:
        raise ValueError("A recorded sample ID is required to link patterns")
    if metadata.get("identity_status") == "conflict":
        raise ValueError("Sample identifiers conflict; pattern linking needs review")
    instrument, ipts = scan.get("instrument"), scan.get("ipts")
    if not instrument or not ipts:
        raise ValueError("Instrument and IPTS provenance are required")
    if metadata.get("source_revision_warning"):
        raise ValueError("The reduction was revised; review the retained acquisition")
    return f"{instrument}/{ipts}/{specimen}/{metadata.get('sample_role', 'sample')}"


def packet_observation(scan):
    """Adapt the existing portable measured-curve receipt, ignoring ycalc/model."""
    packet = scan.get("pattern")
    if not isinstance(packet, dict):
        if not scan.get("pattern_path"):
            raise ValueError(
                "Waiting for a stable reduced file or measured-curve output"
            )
        packet = load_pattern(
            scan["pattern_path"],
            expected_sha256=(scan.get("pattern_observation") or {}).get("sha256"),
        )
    # Pattern.parse also validates fit masks for the old inspector. They are
    # intentionally absent from the detector's input and fingerprint below.
    parsed = Pattern.parse(packet)
    context = packet.get("context") or {}
    metadata = scan.get("metadata") or {}
    bank = scan.get("bank") or metadata.get("source_bank") or context.get("source_bank")
    for field, declared in (
        ("instrument", scan.get("instrument")),
        ("source_bank", bank),
    ):
        if context.get(field) is not None and str(context[field]) != str(declared):
            raise ValueError(
                "Measured receipt disagrees with the selected instrument or bank"
            )
    coordinate = packet.get("coordinate") or {}
    units = {
        ("tof", "us"): "tof_us",
        ("d", "angstrom"): "d_angstrom",
        ("d_spacing", "angstrom"): "d_angstrom",
        ("d", "A"): "d_angstrom",
        ("q", "1/angstrom"): "q_inv_angstrom",
    }
    kind = units.get((coordinate.get("kind"), coordinate.get("unit")))
    geometry = context.get("profile_sha256")
    family = context.get("reduction_family") or scan.get("source_family")
    if not bank or not geometry or not family or kind is None:
        raise ValueError(
            "Bank, geometry, reduction family and supported coordinate units are required"
        )
    basis = (packet.get("ordinate") or {}).get("basis") or "portable observed histogram"
    normalization = context.get("normalization") or "not independently qualified"
    scope = PatternScope(
        identity(scan),
        str(scan["instrument"]),
        str(bank),
        kind,
        str(geometry),
        digest(
            [
                family,
                context.get("calibration"),
                context.get("reduction_setup"),
                scan.get("comparison_setup_id"),
            ]
        ),
        str(normalization),
        str(basis),
    )
    pattern = ObservedPattern(
        scope, parsed.x, parsed.y, parsed.sigma, parsed.valid, str(scan["run_id"])
    )
    return observation(
        pattern,
        {"source": "published measured histogram", "receipt": parsed.key},
        [
            "Reduction normalization is not independently qualified by the project extractor."
        ]
        if "not independently" in normalization
        else [],
    )


def reduced_observation(scan):
    """Read an exact, complete FXYE bank; reject unsupported or changing files."""
    path = Path(scan["source_path"]).resolve()
    root = Path(scan["transition_source_root"]).resolve()
    if not path.is_relative_to(root) or path.suffix.lower() not in {".gsa", ".gss"}:
        raise ValueError("Reduced curve must remain inside the authorized watch folder")
    before = path.stat()
    if not 0 < before.st_size <= 64 * 1024 * 1024:
        raise ValueError("Reduced product exceeds the text-reader bound")
    with path.open("rb") as stream:
        raw = stream.read(64 * 1024 * 1024 + 1)
    if len(raw) > 64 * 1024 * 1024:
        raise ValueError("Reduced product grew beyond the reader bound")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise ValueError("Reduced product is still changing; waiting for a stable file")
    lines = raw.decode("utf-8").splitlines()
    if any(len(line) > 8192 for line in lines):
        raise ValueError("Reduced product contains an oversized line")
    starts = [i for i, line in enumerate(lines) if line.startswith("BANK")]
    if not starts or len(starts) > 64:
        raise ValueError("No bounded explicit GSAS bank inventory")
    for start, end in zip(starts, starts[1:] + [len(lines)]):
        fields = lines[start].split()
        if len(fields) < 6:
            raise ValueError("Bank header is incomplete or still being written")
        declared = int(fields[2])
        if not 32 <= declared <= MAX_POINTS or int(fields[3]) != declared:
            raise ValueError("Unsupported bank record count")
        count = sum(
            bool(line.strip()) and not line.lstrip().startswith("#")
            for line in lines[start + 1 : end]
        )
        if count != declared:
            raise ValueError("Reduced product is truncated or still being written")
    bank = scan.get("bank") or (scan.get("metadata") or {}).get("source_bank")
    if bank is None and len(starts) == 1:
        bank = int(lines[starts[0]].split()[1])
    chosen = [i for i in starts if lines[i].split()[1] == str(bank)]
    if len(chosen) != 1:
        raise ValueError("Choose one explicit physical bank for pattern tracking")
    start = chosen[0]
    end = next((i for i in starts if i > start), len(lines))
    words = lines[start].split()
    if len(words) < 6 or words[-1] != "FXYE":
        raise ValueError("Live pattern tracking supports explicit FXYE products only")
    count = int(words[2])
    if not 32 <= count <= MAX_POINTS or int(words[3]) != count:
        raise ValueError("Unsupported bank record count")
    block = [
        line
        for line in lines[start + 1 : end]
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(block) != count or any(len(line.split()) != 3 for line in block):
        raise ValueError("Reduced bank is truncated or still being written")
    values = np.loadtxt(io.StringIO("\n".join(block)))
    if values.shape != (count, 3) or not np.all(np.isfinite(values[:, 0])):
        raise ValueError("Reduced bank has an invalid numeric grid")
    x, y, sigma = values.T
    if np.any(x <= 0) or np.any(np.diff(x) <= 0):
        raise ValueError("Reduced coordinates must be positive and increasing")
    multiplied = "with Y multiplied by the bin widths" in "\n".join(lines[:40])
    notices = []
    semantics = "native_FXYE_unspecified"
    if multiplied:
        if words[4] != "SLOG":
            raise ValueError("Bin-integrated data require a supported SLOG grid")
        steps = np.diff(x) / x[:-1]
        ratio = float(np.median(steps))
        variation = float(np.max(np.abs(steps[2:-2] - ratio)) / ratio)
        if variation > 0.002:
            raise ValueError("Irregular bin integrals need original bin edges")
        widths = x * (2 * ratio / (2 + ratio)) if variation < 1e-6 else np.gradient(x)
        if variation >= 1e-6:
            notices.append(
                "Density uses approximate midpoint widths; interior step variation is below 0.2%."
            )
        y, sigma = y / widths, sigma / widths
        semantics = "density_from_SLOG_bin_integrals"
    valid = np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
    valid[:2] = valid[-2:] = False
    metadata = scan.get("metadata") or {}
    setup = metadata.get("nomad_setup") or metadata.get("powgen_setup") or {}
    geometry_lines = [
        line.strip()
        for line in lines[max(0, start - 5) : start]
        if "DIFC" in line or "flight path" in line
    ]
    calibration = setup.get("calibration") or setup.get("calibration_files")
    geometry = {
        k: setup.get(k)
        for k in ("cycle", "mode", "wavelength", "frequency", "guide", "environment")
    }
    geometry.update(calibration=calibration, headers=geometry_lines, bank=int(bank))
    if not calibration and not geometry_lines:
        raise ValueError("Reduced-bank calibration or export geometry is unavailable")
    if not calibration:
        notices.append(
            "Geometry is qualified by the recorded bank export header; detector calibration identity is unavailable."
        )
    prefix = "\n".join(lines[:40])
    family = (
        scan.get("source_family")
        or setup.get("source_family")
        or ("Mantid FXYE" if "Mantid" in prefix else "individual FXYE")
    )
    vanadium = re.search(r"Vanadium Run:\s*(\S+)", prefix)
    reduction = {
        "family": family,
        "vanadium": vanadium[1] if vanadium else None,
        "grouping": setup.get("grouping_files"),
        "characterization": setup.get("characterization_files"),
        "setup": scan.get("comparison_setup_id"),
    }
    normalization = (
        "pCharge" if "Normalised to pCharge" in prefix else "not_declared_in_product"
    )
    if normalization.startswith("not_"):
        notices.append(
            "Absolute normalization is not declared; comparisons include robust intensity scaling."
        )
    scope = PatternScope(
        identity(scan),
        str(scan["instrument"]),
        str(bank),
        "tof_us",
        digest(geometry),
        digest(reduction),
        normalization,
        semantics,
    )
    pattern = ObservedPattern(
        scope, x, np.where(np.isfinite(y), y, 0), sigma, valid, str(scan["run_id"])
    )
    return observation(
        pattern,
        {
            "source": "read-only reduced bank",
            "sha256": sha256(raw).hexdigest(),
            "bank": int(bank),
            "intensity_semantics": semantics,
        },
        notices,
    )


def load_observation(scan):
    if scan.get("source_path") and scan.get("transition_source_root"):
        try:
            return reduced_observation(scan)
        except FileNotFoundError:
            # A recovered Galaxy history can outlive its facility mount. A
            # published observation has its own scope and is never mixed with
            # the raw-product basis without explicit provenance agreement.
            pass
    return packet_observation(scan)
