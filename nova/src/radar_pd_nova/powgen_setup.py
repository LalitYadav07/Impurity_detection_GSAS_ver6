"""Resolve each reduced POWGEN bank from its recorded reduction provenance.

This module reads text metadata only. Reduction scripts are parsed as syntax,
never imported or executed. A detector calibration (.h5/.cal) is evidence for
a cycle and grouping; it is not a GSAS instrument parameter file.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from .powgen import PowgenResolutionError, AmbiguousPowgenInputError
from .facility_metadata import reduction_xml

DATA_ROOT = Path(__file__).resolve().parent / "data" / "powgen_calibrations"
CATALOG_PATH = DATA_ROOT / "catalog.json"
RESOLVER_VERSION = "powgen-reduction-provenance/v1"
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_CYCLE = re.compile(r"/SNS/PG3/(?:shared/CALIBRATION/)?(\d{4}[-_]\d(?:_\d)?_11A_CAL)/")
_BANK = re.compile(r"^BANK\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+).*?\s(FXYE|ESD|STD)\s*$")


@dataclass(frozen=True)
class ReducedBank:
    bank: int
    channels: int
    records: int
    binning: str
    encoding: str
    header: str


@dataclass(frozen=True)
class ReducedSetup:
    run_number: int
    wavelength: float
    frequency: float
    guide: float | None
    banks: tuple[ReducedBank, ...]
    calibration_files: tuple[str, ...]
    characterization_files: tuple[str, ...]
    grouping_files: tuple[str, ...]
    cycle: str
    mode: str
    source_signature: tuple[int, int]
    title: str = ""
    reduction_script_signature: tuple[int, int] | None = None
    reduction_script_sha256: str = ""
    reduction_xml_signature: tuple[int, int] | None = None
    reduction_xml_sha256: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BankProfile:
    rule_id: str
    path: Path
    sha256: str
    source_bank: int
    profile_bank: int
    cycle: str
    wavelength: float
    frequency: float
    mode: str
    source: str

    @property
    def profile_filename(self) -> str:
        return self.path.name

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "path": str(self.path), "resolver": RESOLVER_VERSION}


def text_hash(raw: bytes) -> str:
    return sha256(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def _positive(value: str, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise PowgenResolutionError(f"Missing or invalid POWGEN {label}")
    return result


def reduction_arguments(text: str) -> list[dict[str, str]]:
    """Extract literal scientific paths from recognized Mantid calls only."""
    if len(text) > 262144:
        raise PowgenResolutionError("POWGEN reduction script exceeds the metadata size limit")
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise PowgenResolutionError("POWGEN reduction script has unsupported Python syntax") from exc
    accepted = {"SNSPowderReduction", "AlignAndFocusPowder", "LoadCalFile"}
    keys = {"CalibrationFile", "CalFileName", "CalFilename", "CharacterizationRunsFile", "GroupingFile", "SaveAs"}
    result = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
        if name not in accepted:
            continue
        args = {kw.arg: kw.value.value for kw in node.keywords
                if kw.arg in keys and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str)}
        # A second reduction may emit only TOPAS/FullProf with another grouping.
        # It cannot be the producer of this GSA file.
        if name == "SNSPowderReduction" and "SaveAs" in args and "gsas" not in args["SaveAs"].lower():
            continue
        result.append(args)
    return result


def _paths(args: list[dict[str, str]], keys: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted({part.strip() for a in args for k in keys
                         for part in a.get(k, "").split(",") if part.strip()}))


def inspect_reduced_setup(path: str | Path) -> ReducedSetup:
    """Inspect complete bounded structure, including undeclared trailing banks."""
    path = Path(path)
    before = path.stat()
    if before.st_size > 64 * 1024 * 1024:
        raise PowgenResolutionError("POWGEN reduction exceeds the structural inspection size limit")
    header: list[str] = []
    banks: list[ReducedBank] = []
    expected = None
    record_count = 0
    def check_records():
        if banks and banks[-1].encoding == "FXYE" and record_count != banks[-1].records:
            raise PowgenResolutionError("Incomplete POWGEN FXYE bank; wait for autoreduction to finish")
    with path.open(encoding="ascii", errors="strict") as stream:
        for line in stream:
            if len(line) > 8192:
                raise PowgenResolutionError("POWGEN header line exceeds the metadata size limit")
            match = _BANK.fullmatch(line.strip())
            if match:
                check_records()
                banks.append(ReducedBank(int(match[1]), int(match[2]), int(match[3]), match[4], match[5], line.rstrip()))
                record_count = 0
            elif not banks:
                header.append(line)
                count = re.match(r"#\s*(\d+)\s+Histograms", line)
                if count:
                    expected = int(count[1])
                if len(header) > 100:
                    raise PowgenResolutionError("No GSAS BANK header in the bounded POWGEN header")
            elif line.strip() and not line.lstrip().startswith("#"):
                record_count += 1
    check_records()
    if not banks or expected is None or len(banks) != expected or len({b.bank for b in banks}) != len(banks):
        raise PowgenResolutionError("Missing, duplicate, or incomplete POWGEN diffraction banks")
    if any(b.bank <= 0 or b.channels <= 0 or b.records <= 0 for b in banks):
        raise PowgenResolutionError("Invalid POWGEN BANK counts")
    head = "".join(header)
    run = re.search(r"Sample Run:\s*(\d+)", head)
    file_run = re.fullmatch(r"PG3_(\d+)(?:-\d+)?\.(?:gsa|gss)", path.name, re.I)
    if not run or not file_run or int(run[1]) != int(file_run[1]):
        raise PowgenResolutionError("POWGEN filename and Sample Run header do not agree")
    wave = re.search(r"Wavelength:\s*(" + _NUMBER + r")", head)
    freq = re.search(r"Freq:\s*(" + _NUMBER + r")\s*Hz", head)
    guide = re.search(r"Guide:\s*(" + _NUMBER + r")", head)
    if not wave or not freq:
        raise PowgenResolutionError("Reduced POWGEN file does not report wavelength and frequency")
    script = path.with_suffix(".py")
    args: list[dict[str, str]] = []
    script_signature = None
    script_sha256 = ""
    if script.is_file():
        stat = script.stat()
        script_signature = (stat.st_size, stat.st_mtime_ns)
        with script.open("rb") as stream:
            script_raw = stream.read(262145)
        if len(script_raw) > 262144:
            raise PowgenResolutionError("POWGEN reduction script exceeds the metadata size limit")
        args = reduction_arguments(script_raw.decode("utf-8"))
        script_sha256 = sha256(script_raw).hexdigest()
        stat = script.stat()
        if script_signature != (stat.st_size, stat.st_mtime_ns):
            raise PowgenResolutionError("POWGEN reduction script changed while inspecting it")
    calibration = _paths(args, ("CalibrationFile", "CalFileName", "CalFilename"))
    characterization = _paths(args, ("CharacterizationRunsFile",))
    grouping = _paths(args, ("GroupingFile",))
    xml = path.with_suffix('.xml')
    xml_signature, xml_sha256 = None, ''
    if xml.is_file():
        receipt, xml_signature, xml_sha256 = reduction_xml(xml, 'PG3', int(run[1]))
        xml_args = [receipt]
        xml_paths = (_paths(xml_args, ('CalibrationFile',)), _paths(xml_args, ('CharacterizationRunsFile',)),
                     _paths(xml_args, ('GroupingFile',)))
        if script.is_file():
            if (calibration, characterization, grouping) != xml_paths:
                raise AmbiguousPowgenInputError('Saved Python and XML reduction receipts disagree; review the product revision')
        else:
            calibration, characterization, grouping = xml_paths
    cycles = {m[1] for p in (*calibration, *characterization, *grouping) if (m := _CYCLE.search(p))}
    if len(cycles) > 1:
        raise AmbiguousPowgenInputError(f"Conflicting reduction calibration cycles: {sorted(cycles)}")
    modes = set()
    for p in characterization:
        name = Path(p).name
        if re.search(r"HighRes|(?:^|[-_])HR(?:[-_.]|$)", name, re.I):
            modes.add("HR")
        if re.search(r"HighInt|(?:^|[-_])HI(?:[-_.]|$)", name, re.I):
            modes.add("HI")
    if len(modes) > 1:
        raise AmbiguousPowgenInputError("Conflicting HighRes/HighInt reduction evidence")
    after = path.stat()
    signature = (before.st_size, before.st_mtime_ns)
    if signature != (after.st_size, after.st_mtime_ns):
        raise PowgenResolutionError("POWGEN reduction changed while inspecting it; retry after autoreduction completes")
    titles = [line.lstrip("# ").strip() for line in header if line.startswith("# ")]
    return ReducedSetup(int(run[1]), _positive(wave[1], "wavelength"), _positive(freq[1], "frequency"),
                        float(guide[1]) if guide else None, tuple(banks), calibration, characterization,
                        grouping, next(iter(cycles), ""), next(iter(modes), ""), signature,
                        titles[0] if titles else "", script_signature, script_sha256, xml_signature, xml_sha256)


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, Any]:
    catalog = json.loads(CATALOG_PATH.read_text())
    if catalog.get("schema") != RESOLVER_VERSION:
        raise PowgenResolutionError("Unsupported POWGEN bank profile catalog")
    return catalog


def resolve_bank_profile(setup: ReducedSetup, bank: int, *, catalog: Mapping[str, Any] | None = None,
                         data_root: Path = DATA_ROOT) -> BankProfile:
    """Require a unique calibrated cycle/mode/frame/grouping match."""
    if bank not in {b.bank for b in setup.banks}:
        raise PowgenResolutionError(f"BANK {bank} is absent from PG3_{setup.run_number}")
    if not setup.cycle:
        raise PowgenResolutionError("Reduction calibration cycle is missing; supply a verified profile for this run")
    if not setup.mode:
        raise PowgenResolutionError("HighRes/HighInt mode is missing from reduction provenance; no profile was guessed")
    if len(setup.calibration_files) != 1:
        raise PowgenResolutionError("A unique detector calibration is required; several reduction inputs cannot be silently combined")
    # Numeric guide positions drift over time. A broad plausibility check can
    # detect contradicting evidence, but never establishes the mode by itself.
    if setup.guide is not None and ((setup.mode == "HR" and setup.guide > 20) or (setup.mode == "HI" and setup.guide < -20)):
        raise PowgenResolutionError("Guide position conflicts with the recorded HighRes/HighInt reduction mode")
    names = [Path(p).name.upper() for p in setup.calibration_files]
    special = "2BANKS" if any("2BANKS" in p for p in names) else "standard"
    if special == "standard" and any(re.search(r"BANK\d|_OP\.", p) for p in names):
        raise PowgenResolutionError("Partial-detector calibration requires a validated grouping/profile mapping")
    if setup.cycle == "2017_1_2_11A_CAL" and special == "standard":
        if names and all("_ALL." in p for p in names):
            special = "ALL"
        elif names and all("_IP." in p for p in names):
            special = "IP"
        else:
            raise PowgenResolutionError("2017 partial-detector calibration requires a validated grouping/profile mapping")
    if setup.grouping_files:
        raise PowgenResolutionError("Custom detector grouping requires an explicitly validated bank/profile mapping")
    matches = []
    for rule in (catalog if catalog is not None else load_catalog()).get("rules", []):
        if (rule["cycle"] != setup.cycle or rule["mode"] != setup.mode or rule["grouping"] != special
                or rule["source_bank"] != bank or abs(rule["frequency"] - setup.frequency) > 0.001
                or abs(rule["wavelength"] - setup.wavelength) > rule.get("wavelength_tolerance", 0.0005)):
            continue
        # Public interval boundaries are an independent check, not permission
        # to override a different calibration recorded in the reduction.
        if "run_min" in rule and (setup.run_number < rule["run_min"] or
                                  (rule["run_max"] is not None and setup.run_number > rule["run_max"])):
            continue
        matches.append(rule)
    if len(matches) != 1:
        description = f"PG3_{setup.run_number} BANK {bank}, {setup.cycle}, {setup.mode}, {setup.wavelength:g} A, {setup.frequency:g} Hz, {special}"
        if matches:
            raise AmbiguousPowgenInputError(f"Several calibrated profiles match {description}: " + ", ".join(r["id"] for r in matches))
        raise PowgenResolutionError(f"No validated instrument profile matches {description}")
    rule = matches[0]
    name = rule["resource"]
    if Path(name).name != name:
        raise PowgenResolutionError("Unsafe POWGEN profile resource path")
    path = data_root / name
    if text_hash(path.read_bytes()) != rule["sha256"]:
        raise PowgenResolutionError(f"POWGEN profile checksum mismatch: {name}")
    return BankProfile(rule["id"], path, rule["sha256"], bank, rule["profile_bank"], setup.cycle,
                       setup.wavelength, setup.frequency, setup.mode, rule["source"])


def single_bank_profile(profile: BankProfile) -> str:
    """Keep all selected legacy bank records, including PAB interpolation tables."""
    raw = profile.path.read_bytes()
    if text_hash(raw) != profile.sha256:
        raise PowgenResolutionError("Instrument profile changed after resolution")
    text = raw.decode("ascii").replace("\r\n", "\n").replace("\r", "\n")
    if profile.path.suffix.lower() == ".instprm":
        # Packaged modern resources contain one bank only. Bank is a local
        # profile label, not necessarily the GSA bank number (2019A is one).
        if len(re.findall(r"(?:^|;)\s*Bank\s*:", text, re.M)) != 1:
            raise PowgenResolutionError("Expected a single-bank GSAS-II instrument resource")
        return re.sub(r"(^|;)(\s*Bank\s*:)\s*[^;\n]+", r"\g<1>\g<2>1.0", text, flags=re.M)
    result = []
    found = False
    for line in text.splitlines(keepends=True):
        if line.startswith("INS") and line[3:6].strip().isdigit():
            if int(line[3:6]) != profile.profile_bank:
                continue
            found = True
            line = line[:3] + "  1" + line[6:]
        elif line.startswith("INS   BANK"):
            line = "INS   BANK      1\n"
        result.append(line)
    if not found:
        raise PowgenResolutionError("Selected legacy instrument bank is missing")
    return "".join(result)


def stage_bank(source: Path, setup: ReducedSetup, bank: int, profile: BankProfile,
               destination: Path) -> tuple[Path, Path]:
    """Stage one FXYE bank and its profile; preserve numeric data byte-for-byte.

    The current Analyze contract consumes one histogram. BANK labels alone are
    remapped to 1 in both staged files; source bank/provenance are retained in a
    separate receipt. No intensity, error, TOF or profile coefficient changes.
    """
    if source.stat().st_size != setup.source_signature[0] or source.stat().st_mtime_ns != setup.source_signature[1]:
        raise PowgenResolutionError("POWGEN reduction changed after preflight")
    def check_script():
        script = source.with_suffix(".py")
        stat = script.stat() if script.is_file() else None
        signature = (stat.st_size, stat.st_mtime_ns) if stat else None
        if signature != setup.reduction_script_signature:
            raise PowgenResolutionError("POWGEN reduction script changed after preflight")
        xml = source.with_suffix('.xml')
        stat = xml.stat() if xml.is_file() else None
        xml_signature = (stat.st_size,stat.st_mtime_ns) if stat else None
        if xml_signature != setup.reduction_xml_signature:
            raise PowgenResolutionError('POWGEN reduction XML changed after preflight')
        if stat and sha256(xml.read_bytes()).hexdigest() != setup.reduction_xml_sha256:
            raise PowgenResolutionError('POWGEN reduction XML content changed after preflight')
    check_script()
    selected = next((b for b in setup.banks if b.bank == bank), None)
    if selected is None or selected.encoding != "FXYE" or selected.channels != selected.records:
        raise PowgenResolutionError("Bank staging currently requires explicit GSAS FXYE records")
    header = bytearray()
    payload = bytearray()
    active = False
    records = 0
    bank_count = 0
    source_digest = sha256()
    previous_x = None
    with source.open("rb") as stream:
        for raw in stream:
            source_digest.update(raw)
            if raw.startswith(b"BANK"):
                bank_count += 1
                match = _BANK.fullmatch(raw.decode("ascii").strip())
                if match is None:
                    raise PowgenResolutionError("Unsupported GSAS BANK record")
                active = int(match[1]) == bank
                if active:
                    payload.extend(re.sub(rb"^(BANK\s+)\d+", rb"\g<1>1", raw))
            elif bank_count == 0:
                header.extend(re.sub(rb"^(#\s*)\d+(\s+Histograms)", rb"\g<1>1\2", raw))
            elif active:
                if raw.lstrip().startswith(b"#") or not raw.strip():
                    payload.extend(raw)
                    continue
                values = raw.split()
                if len(values) != 3:
                    raise PowgenResolutionError("Malformed GSAS FXYE row")
                x, y, sigma = map(float, values)
                if not all(math.isfinite(v) for v in (x, y, sigma)) or sigma < 0:
                    raise PowgenResolutionError("Nonfinite data or negative uncertainty in GSAS FXYE bank")
                if previous_x is not None and x <= previous_x:
                    raise PowgenResolutionError("POWGEN bank TOF coordinates must increase strictly")
                previous_x = x
                records += 1
                payload.extend(raw)
    stat = source.stat()
    check_script()
    if (stat.st_size, stat.st_mtime_ns) != setup.source_signature or bank_count != len(setup.banks) or records != selected.records:
        raise PowgenResolutionError("Incomplete or changed POWGEN bank; wait for autoreduction to finish")
    instrument_text = single_bank_profile(profile)
    destination.mkdir(parents=True, exist_ok=True)
    data = destination / f"PG3_{setup.run_number}_bank{bank}.gsa"
    instrument = destination / (f"bank{bank}_" + profile.path.name)
    observation_provenance = {
        "schema": "radar-pd-input-provenance/v1", "instrument": "PG3", "source_bank": bank,
        "source_sha256": source_digest.hexdigest(), "reduction_family": "POWGEN individual GSAS/FXYE",
        "reduction_setup": {"cycle": setup.cycle, "mode": setup.mode, "wavelength": setup.wavelength,
            "frequency": setup.frequency, "guide": setup.guide, "calibration_files": setup.calibration_files,
            "characterization_files": setup.characterization_files, "grouping_files": setup.grouping_files},
        "profile_sha256": profile.sha256, "warnings": [],
    }
    marker = ("# RADAR-PD input provenance: " + json.dumps(observation_provenance, separators=(",", ":")) + "\n").encode("ascii")
    if len(marker) > 8192:
        raise PowgenResolutionError("POWGEN observation provenance exceeds its supported header size")
    # Preserve the original title, but keep the receipt inside the batch
    # reader's bounded first-64-line probe even on verbose facility exports.
    insertion = header.find(b"\n") + 1
    header[insertion:insertion] = marker
    data.write_bytes(header + payload)
    instrument.write_text(instrument_text, encoding="ascii")
    receipt = {"resolver": RESOLVER_VERSION, "source_path": str(source), "source_bank": bank,
               "staged_bank": 1, "source_signature": setup.source_signature,
               "source_sha256": source_digest.hexdigest(),
               "data_sha256": sha256(data.read_bytes()).hexdigest(), "profile": profile.as_dict(),
               "staged_profile_sha256": text_hash(instrument.read_bytes()), "setup": setup.as_dict()}
    (destination / f"PG3_{setup.run_number}_bank{bank}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return data, instrument
