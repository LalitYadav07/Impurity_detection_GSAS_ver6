#!/usr/bin/env python3
"""Download public GSAS-II instrument parameter files.

This script crawls public facility pages that currently publish GSAS-II
calibration assets, downloads them, extracts .instprm files from ZIP archives,
and records a manifest that makes newly published files easy to detect on those
same pages.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable


USER_AGENT = "gsasii-instprm-catalog/1.0"
DEFAULT_OUTPUT_DIR = Path("gsasii_instprm_catalog")
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CATALOG_DIR = "catalog"

SOURCE_PAGES = {
    "APS_11BM": {
        "institution": "Argonne National Laboratory",
        "facility": "APS",
        "country": "USA",
        "instrument": "11-BM",
        "beamline": "11-BM",
        "url": "https://wiki-ext.aps.anl.gov/ug11bm/index.php/Standards_Data",
    },
    "DESY_P61B": {
        "institution": "Deutsches Elektronen-Synchrotron DESY",
        "facility": "PETRA III",
        "country": "Germany",
        "instrument": "P61B Large Volume Press",
        "beamline": "P61B",
        "url": "https://photon-science.desy.de/facilities/petra_iii/beamlines/p61_high_energy_wiggler_beamline_lvp/p61b_large_volume_press_desy/software_tools/index_eng.html",
    },
    "HB2A": {
        "institution": "Oak Ridge National Laboratory",
        "facility": "HFIR",
        "country": "USA",
        "instrument": "POWDER",
        "beamline": "HB-2A",
        "url": "https://neutrons.ornl.gov/powder/users",
    },
    "HB2C": {
        "institution": "Oak Ridge National Laboratory",
        "facility": "HFIR",
        "country": "USA",
        "instrument": "WAND2",
        "beamline": "HB-2C",
        "url": "https://neutrons.ornl.gov/wand/users",
    },
    "NOM": {
        "institution": "Oak Ridge National Laboratory",
        "facility": "SNS",
        "country": "USA",
        "instrument": "NOMAD",
        "beamline": "BL-1B",
        "url": "https://neutrons.ornl.gov/nomad/users",
    },
    "PG3": {
        "institution": "Oak Ridge National Laboratory",
        "facility": "SNS",
        "country": "USA",
        "instrument": "POWGEN",
        "beamline": "BL-11A",
        "url": "https://neutrons.ornl.gov/powgen/users",
    },
}

GSASII_TEXT_RE = re.compile(r"gsas[\s_-]*(?:ii|2(?!\d))", re.IGNORECASE)
GSASII_HREF_RE = re.compile(r"gsas[\s_-]*(?:ii|2(?!\d))|gsasii", re.IGNORECASE)
SKIP_TEXT_RE = re.compile(r"tutorial|pdf|joint x-ray|software", re.IGNORECASE)
INSTPRM_SUFFIXES = (".instprm", ".instrprm")
VALID_DOWNLOAD_SUFFIXES = (".zip", ".instprm", ".instrprm", ".prm", ".gpx")
PAGE_ERRORS: list[dict[str, str]] = []
MONTH_RE = re.compile(
    r"(?<![A-Za-z])(Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)(?![A-Za-z])",
    re.IGNORECASE,
)

FILE_CATALOG_FIELDS = [
    "catalog_id",
    "source_code",
    "institution",
    "country",
    "facility",
    "instrument_code",
    "instrument",
    "beamline",
    "format",
    "local_path",
    "filename",
    "source_archive",
    "source_label",
    "source_url",
    "source_sha256",
    "file_sha256",
    "file_size",
    "cycle",
    "run_period",
    "year",
    "month",
    "measurement_type",
    "gsas_type",
    "bank_count",
    "bank_numbers",
    "wavelength",
    "cwl",
    "tof_bank",
    "monochromator",
    "collimation",
    "calibrant",
    "two_theta_min",
    "two_theta_max",
    "difc_min",
    "difc_max",
    "search_text",
]

BANK_CATALOG_FIELDS = [
    "catalog_id",
    "source_code",
    "institution",
    "country",
    "facility",
    "instrument_code",
    "instrument",
    "beamline",
    "local_path",
    "bank_index",
    "bank",
    "gsas_type",
    "measurement_type",
    "wavelength",
    "two_theta",
    "difc",
    "difa",
    "difb",
    "zero",
    "flt_path",
    "alpha",
    "beta_0",
    "beta_1",
    "sig_0",
    "sig_1",
    "sig_2",
    "x",
    "y",
    "z",
]


@dataclass
class Asset:
    source_code: str
    institution: str
    country: str
    facility: str
    instrument: str
    beamline: str
    instrument_code: str
    page_url: str
    label: str
    url: str
    filename: str
    kind: str
    external: bool
    status: str = "unknown"
    sha256: str | None = None
    size: int | None = None
    extracted_files: list[str] | None = None
    error: str | None = None


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href_stack: list[str | None] = []
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            attrs_dict = dict(attrs)
            self._href_stack.append(attrs_dict.get("href"))
            self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._href_stack:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href_stack:
            href = self._href_stack.pop()
            text = html.unescape(" ".join("".join(self._text_parts).split()))
            self._text_parts = []
            if href:
                self.links.append((href, text))


def request(url: str, timeout: int = 45) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def fetch_bytes(url: str, timeout: int = 45) -> bytes:
    with urllib.request.urlopen(request(url), timeout=timeout) as response:
        return response.read()


def fetch_text_or_none(url: str) -> str | None:
    try:
        return fetch_bytes(url).decode("utf-8", errors="replace")
    except (OSError, urllib.error.URLError) as exc:
        PAGE_ERRORS.append({"url": url, "error": str(exc)})
        return None


def normalize_url(page_url: str, href: str) -> str:
    href = html.unescape(href)
    url = urllib.parse.urljoin(page_url, href)
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc.endswith("dropbox.com"):
        query = urllib.parse.parse_qs(parsed.query)
        query["dl"] = ["1"]
        url = urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query, doseq=True)))
    return url


def safe_filename(name: str) -> str:
    name = urllib.parse.unquote(name)
    name = re.sub(r"[^\w.\-+]+", "_", name).strip("._")
    return name or "download"


def filename_from_url(url: str, fallback: str) -> str:
    parsed = urllib.parse.urlparse(url)
    path_name = Path(parsed.path).name
    return safe_filename(path_name or fallback)


def asset_kind(url: str, label: str) -> str:
    lower = (url + " " + label).lower()
    if lower.endswith(".zip") or ".zip" in lower:
        return "zip"
    if any(suffix in lower for suffix in INSTPRM_SUFFIXES):
        return "instprm"
    if ".gpx" in lower:
        return "gpx"
    if ".prm" in lower:
        return "prm"
    return "other"


def is_gsasii_asset(url: str, label: str, include_gpx: bool) -> bool:
    lower_url = url.lower()
    lower_label = label.lower()
    if SKIP_TEXT_RE.search(label) and not lower_url.endswith(VALID_DOWNLOAD_SUFFIXES):
        return False
    if not (GSASII_TEXT_RE.search(label) or GSASII_HREF_RE.search(url)):
        return False
    if lower_url.endswith(".gpx") and not include_gpx:
        return False
    return any(suffix in lower_url for suffix in VALID_DOWNLOAD_SUFFIXES)


def discover_assets(include_gpx: bool) -> list[Asset]:
    PAGE_ERRORS.clear()
    assets: list[Asset] = []
    seen: set[str] = set()
    for code, meta in SOURCE_PAGES.items():
        page_url = meta["url"]
        page_text = fetch_text_or_none(page_url)
        if page_text is None:
            continue
        parser = LinkParser()
        parser.feed(page_text)
        for href, label in parser.links:
            url = normalize_url(page_url, href)
            if not is_gsasii_asset(url, label, include_gpx):
                continue
            if url in seen:
                continue
            seen.add(url)
            kind = asset_kind(url, label)
            filename = filename_from_url(url, f"{code}_{safe_filename(label)}.{kind}")
            parsed = urllib.parse.urlparse(url)
            page_parsed = urllib.parse.urlparse(page_url)
            assets.append(
                Asset(
                    source_code=code,
                    institution=meta["institution"],
                    country=meta["country"],
                    facility=meta["facility"],
                    instrument=meta["instrument"],
                    beamline=meta["beamline"],
                    instrument_code=code,
                    page_url=page_url,
                    label=label,
                    url=url,
                    filename=filename,
                    kind=kind,
                    external=parsed.netloc != page_parsed.netloc,
                )
            )
    return sorted(assets, key=lambda a: (a.facility, a.instrument_code, a.filename))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_previous_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {item["url"]: item for item in data.get("assets", [])}


def download_asset(asset: Asset, root: Path, force: bool, extract: bool) -> Asset:
    instrument_dir = root / asset.facility / asset.instrument_code
    archive_dir = instrument_dir / "downloads"
    archive_dir.mkdir(parents=True, exist_ok=True)
    target = archive_dir / asset.filename
    if not target.exists() or force:
        with urllib.request.urlopen(request(asset.url), timeout=90) as response:
            with target.open("wb") as handle:
                shutil.copyfileobj(response, handle)
    asset.size = target.stat().st_size
    asset.sha256 = sha256_file(target)
    asset.status = "downloaded"
    asset.extracted_files = []

    if extract and asset.kind == "zip":
        extract_root = instrument_dir / "instprm"
        extract_root.mkdir(parents=True, exist_ok=True)
        prefix = f"{safe_filename(Path(asset.filename).stem)}_"
        for suffix in INSTPRM_SUFFIXES:
            for old_file in extract_root.glob(f"{prefix}*{suffix}"):
                old_file.unlink()
        with zipfile.ZipFile(target) as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                name = member.filename.replace("\\", "/")
                basename = Path(name).name
                if name.startswith("__MACOSX/") or basename.startswith("._"):
                    continue
                if not name.lower().endswith(INSTPRM_SUFFIXES):
                    continue
                output_name = safe_filename(f"{Path(asset.filename).stem}_{basename}")
                output_path = extract_root / output_name
                with zf.open(member) as source, output_path.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
                asset.extracted_files.append(str(output_path.relative_to(root)))
    return asset


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def format_number(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value:g}"


def get_param(block: dict[str, str], key: str) -> str | None:
    key_lower = key.lower()
    for candidate, value in block.items():
        if candidate.lower() == key_lower:
            return value
    return None


def read_instprm_blocks(path: Path) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if current:
                blocks.append(current)
            current = {"_comment": line.lstrip("#").strip()}
            bank_match = re.search(r"\bBank\s+(\d+)", line, re.IGNORECASE)
            if bank_match:
                current["Bank"] = bank_match.group(1)
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        current[key.strip()] = value.strip()
    if current:
        blocks.append(current)
    return [block for block in blocks if any(not key.startswith("_") for key in block)]


def first_match(pattern: str, text: str, flags: int = re.IGNORECASE) -> str | None:
    match = re.search(pattern, text, flags)
    return match.group(1) if match else None


def infer_period(text: str) -> str | None:
    match = re.search(r"(20\d{2}[AB])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"(?:^|[_\-\s])(\d{2}[AB])(?:[_\-\s.]|$)", text, re.IGNORECASE)
    if match:
        return f"20{match.group(1).upper()}"
    return None


def infer_metadata_from_names(asset: Asset, path: Path) -> dict[str, str | None]:
    text = " ".join([asset.label, asset.filename, path.name])
    cycle_matches = re.findall(r"cycle\s*[_-]?(\d+)", text, re.IGNORECASE)
    cwl = first_match(r"CWL(\d+(?:p\d+)?)", text)
    month_match = MONTH_RE.search(text)
    monochromator = first_match(r"(Ge\d{3})", text)
    collimation = None
    if monochromator:
        collimation = first_match(rf"{monochromator}_([A-Za-z0-9_]+?)(?:_cycle|\.|$)", text)
    calibrant = None
    if re.search(r"Si[_\-\s]*640e", text, re.IGNORECASE):
        calibrant = "Si 640e"
    elif re.search(r"\bSi\b", text, re.IGNORECASE):
        calibrant = "Si"
    run_period = infer_period(text)
    return {
        "cycle": cycle_matches[-1] if cycle_matches else None,
        "run_period": run_period,
        "year": first_match(r"(20\d{2})", text),
        "month": month_match.group(1) if month_match else None,
        "cwl": cwl.replace("p", ".") if cwl else None,
        "tof_bank": f"B{first_match(r'60HzB(\d+)', text)}" if first_match(r"60HzB(\d+)", text) else None,
        "monochromator": monochromator,
        "collimation": collimation,
        "calibrant": calibrant,
    }


def measurement_type(gsas_type: str | None) -> str | None:
    if gsas_type == "PNT":
        return "TOF"
    if gsas_type == "PNC":
        return "constant_wavelength"
    return None


def block_float_values(blocks: list[dict[str, str]], key: str) -> list[float]:
    values: list[float] = []
    for block in blocks:
        value = parse_float(get_param(block, key))
        if value is not None:
            values.append(value)
    return values


def catalog_file_row(root: Path, asset: Asset, local_path: Path, fmt: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    rel_path = local_path.relative_to(root).as_posix()
    name_metadata = infer_metadata_from_names(asset, local_path)
    blocks = read_instprm_blocks(local_path) if fmt == "instprm" else []
    gsas_types = sorted({get_param(block, "Type") for block in blocks if get_param(block, "Type")})
    primary_type = gsas_types[0] if len(gsas_types) == 1 else None
    bank_numbers = []
    for index, block in enumerate(blocks, start=1):
        bank = parse_float(get_param(block, "Bank"))
        bank_numbers.append(str(int(bank)) if bank is not None and bank.is_integer() else str(index))
    lam_values = block_float_values(blocks, "Lam")
    two_theta_values = block_float_values(blocks, "2-theta")
    difc_values = block_float_values(blocks, "difC")
    catalog_id = safe_filename(f"{asset.instrument_code}_{local_path.stem}")
    row: dict[str, object] = {
        "catalog_id": catalog_id,
        "source_code": asset.source_code,
        "institution": asset.institution,
        "country": asset.country,
        "facility": asset.facility,
        "instrument_code": asset.instrument_code,
        "instrument": asset.instrument,
        "beamline": asset.beamline,
        "format": fmt,
        "local_path": rel_path,
        "filename": local_path.name,
        "source_archive": asset.filename,
        "source_label": asset.label,
        "source_url": asset.url,
        "source_sha256": asset.sha256,
        "file_sha256": sha256_file(local_path) if local_path.exists() else None,
        "file_size": local_path.stat().st_size if local_path.exists() else None,
        "cycle": name_metadata["cycle"],
        "run_period": name_metadata["run_period"],
        "year": name_metadata["year"],
        "month": name_metadata["month"],
        "measurement_type": measurement_type(primary_type),
        "gsas_type": primary_type,
        "bank_count": len(blocks) if blocks else None,
        "bank_numbers": ",".join(bank_numbers) if bank_numbers else None,
        "wavelength": format_number(lam_values[0]) if len(set(lam_values)) == 1 and lam_values else None,
        "cwl": name_metadata["cwl"],
        "tof_bank": name_metadata["tof_bank"],
        "monochromator": name_metadata["monochromator"],
        "collimation": name_metadata["collimation"],
        "calibrant": name_metadata["calibrant"],
        "two_theta_min": format_number(min(two_theta_values)) if two_theta_values else None,
        "two_theta_max": format_number(max(two_theta_values)) if two_theta_values else None,
        "difc_min": format_number(min(difc_values)) if difc_values else None,
        "difc_max": format_number(max(difc_values)) if difc_values else None,
    }
    row["search_text"] = " ".join(str(row.get(field) or "") for field in FILE_CATALOG_FIELDS if field != "search_text")

    bank_rows: list[dict[str, object]] = []
    for index, block in enumerate(blocks, start=1):
        block_type = get_param(block, "Type")
        bank_float = parse_float(get_param(block, "Bank"))
        bank_rows.append(
            {
                "catalog_id": catalog_id,
                "source_code": asset.source_code,
                "institution": asset.institution,
                "country": asset.country,
                "facility": asset.facility,
                "instrument_code": asset.instrument_code,
                "instrument": asset.instrument,
                "beamline": asset.beamline,
                "local_path": rel_path,
                "bank_index": index,
                "bank": int(bank_float) if bank_float is not None and bank_float.is_integer() else get_param(block, "Bank"),
                "gsas_type": block_type,
                "measurement_type": measurement_type(block_type),
                "wavelength": get_param(block, "Lam"),
                "two_theta": get_param(block, "2-theta"),
                "difc": get_param(block, "difC"),
                "difa": get_param(block, "difA"),
                "difb": get_param(block, "difB"),
                "zero": get_param(block, "Zero"),
                "flt_path": get_param(block, "fltPath"),
                "alpha": get_param(block, "alpha"),
                "beta_0": get_param(block, "beta-0"),
                "beta_1": get_param(block, "beta-1"),
                "sig_0": get_param(block, "sig-0"),
                "sig_1": get_param(block, "sig-1"),
                "sig_2": get_param(block, "sig-2"),
                "x": get_param(block, "X"),
                "y": get_param(block, "Y"),
                "z": get_param(block, "Z"),
            }
        )
    return row, bank_rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) if row.get(field) is not None else "" for field in fieldnames})


def write_catalog(root: Path, catalog_dir_name: str, assets: Iterable[Asset]) -> dict[str, object]:
    catalog_dir = root / catalog_dir_name
    catalog_dir.mkdir(parents=True, exist_ok=True)
    file_rows: list[dict[str, object]] = []
    bank_rows: list[dict[str, object]] = []

    for asset in assets:
        local_files: list[tuple[Path, str]] = []
        for rel_path in asset.extracted_files or []:
            path = root / rel_path
            if path.exists():
                local_files.append((path, "instprm"))
        if not local_files and asset.kind in {"gpx", "instprm", "prm"}:
            path = root / asset.facility / asset.instrument_code / "downloads" / asset.filename
            if path.exists():
                local_files.append((path, asset.kind))
        for local_path, fmt in local_files:
            file_row, file_bank_rows = catalog_file_row(root, asset, local_path, fmt)
            file_rows.append(file_row)
            bank_rows.extend(file_bank_rows)

    file_rows.sort(key=lambda row: (str(row.get("instrument_code")), str(row.get("run_period")), str(row.get("cycle")), str(row.get("filename"))))
    bank_rows.sort(key=lambda row: (str(row.get("catalog_id")), int(row.get("bank_index") or 0)))

    files_json = catalog_dir / "files.json"
    banks_json = catalog_dir / "banks.json"
    files_csv = catalog_dir / "files.csv"
    banks_csv = catalog_dir / "banks.csv"
    readme = catalog_dir / "README.md"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(file_rows),
        "bank_count": len(bank_rows),
        "files": file_rows,
    }
    files_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    banks_json.write_text(
        json.dumps({"generated_at": payload["generated_at"], "bank_count": len(bank_rows), "banks": bank_rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(files_csv, file_rows, FILE_CATALOG_FIELDS)
    write_csv(banks_csv, bank_rows, BANK_CATALOG_FIELDS)
    readme.write_text(
        "\n".join(
            [
                "# GSAS-II Instrument Parameter Catalog",
                "",
                "Use `files.csv` or `files.json` to choose a parameter file by instrument, cycle, run period, wavelength/CWL, TOF bank, monochromator, collimation, or calibrant.",
                "",
                "Use `banks.csv` or `banks.json` when a GSAS-II `.instprm` contains multiple banks and you need bank-level values such as `2-theta`, `difC`, `Lam`, `fltPath`, or profile coefficients.",
                "",
                "Important columns:",
                "- `catalog_id`: stable lookup key for a local catalog entry.",
                "- `source_code`: source-specific key such as `HB2A`, `PG3`, or `APS_11BM`.",
                "- `institution`, `country`, `facility`, `instrument_code`: provenance and instrument identity.",
                "- `local_path`: path relative to the catalog root that can be passed to GSAS-II or the refinement app.",
                "- `source_url` and `source_sha256`: provenance for the published archive/file.",
                "- `file_sha256`: checksum for the extracted local parameter file.",
                "- `search_text`: denormalized text field for simple substring search in an app.",
                "",
                "Typical pull-up filters:",
                "- HB-2A: filter `instrument_code=HB2A`, then `cycle`, `monochromator`, and `collimation`.",
                "- POWGEN: filter `instrument_code=PG3`, then `run_period`, `tof_bank`, and `cwl`.",
                "- NOMAD: filter `instrument_code=NOM`, then `run_period`; use `banks.csv` for individual bank angles.",
                "- WAND2: included only when the downloader is run with `--include-gpx`; it is a `.gpx` entry rather than `.instprm`.",
                "- APS 11-BM: filter `source_code=APS_11BM`.",
                "- DESY P61B: filter `source_code=DESY_P61B` for the generic ED-XRD parameter file.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "files_json": str(files_json.relative_to(root)),
        "files_csv": str(files_csv.relative_to(root)),
        "banks_json": str(banks_json.relative_to(root)),
        "banks_csv": str(banks_csv.relative_to(root)),
        "readme": str(readme.relative_to(root)),
        "file_count": len(file_rows),
        "bank_count": len(bank_rows),
    }


def write_manifest(path: Path, assets: Iterable[Asset], previous: dict[str, dict], catalog_summary: dict[str, object] | None) -> None:
    current_urls = {asset.url for asset in assets}
    missing = [
        {"url": url, "label": old.get("label"), "filename": old.get("filename")}
        for url, old in previous.items()
        if url not in current_urls
    ]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "Public facility pages with downloadable GSAS-II assets",
        "new_file_detection": (
            "Run this script on a schedule and compare this manifest to the previous one. "
            "Assets with status=new or status=changed need attention; missing_from_source "
            "records links that disappeared from source pages."
        ),
        "source_pages": SOURCE_PAGES,
        "page_errors": PAGE_ERRORS,
        "catalog": catalog_summary,
        "assets": [asdict(asset) for asset in assets],
        "missing_from_source": missing,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def classify_status(asset: Asset, previous: dict[str, dict]) -> str:
    old = previous.get(asset.url)
    if old is None:
        return "new"
    if asset.sha256 and not old.get("sha256"):
        return "new"
    if asset.sha256 and old.get("sha256") and asset.sha256 != old.get("sha256"):
        return "changed"
    return "unchanged"


def print_summary(assets: list[Asset], previous: dict[str, dict]) -> None:
    counts: dict[str, int] = {}
    for asset in assets:
        counts[asset.status] = counts.get(asset.status, 0) + 1
    missing = len(set(previous) - {asset.url for asset in assets})
    print("GSAS-II calibration assets")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")
    if missing:
        print(f"  missing_from_source: {missing}")
    for asset in assets:
        detail = f"{asset.instrument_code} {asset.label} -> {asset.filename}"
        if asset.extracted_files:
            detail += f" ({len(asset.extracted_files)} .instprm extracted)"
        if asset.error:
            detail += f" ERROR: {asset.error}"
        print(f"  [{asset.status}] {detail}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for downloads, extracted .instprm files, and manifest.",
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="Manifest file name inside --output-dir.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Discover and compare only; do not download.")
    parser.add_argument("--force", action="store_true", help="Re-download files even if already present.")
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Keep ZIP files only; do not extract .instprm files.",
    )
    parser.add_argument(
        "--include-gpx",
        action="store_true",
        help="Include WAND2 GSAS-II .gpx calibration input. It is not an .instprm file.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after individual download errors and record them in the manifest.",
    )
    parser.add_argument(
        "--catalog-dir",
        default=DEFAULT_CATALOG_DIR,
        help="Directory name inside --output-dir for files.csv/files.json/banks.csv/banks.json.",
    )
    parser.add_argument("--no-catalog", action="store_true", help="Do not write the searchable catalog files.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = args.output_dir
    manifest_path = root / args.manifest
    previous = load_previous_manifest(manifest_path)
    assets = discover_assets(include_gpx=args.include_gpx)

    for asset in assets:
        if args.dry_run:
            asset.status = "new" if asset.url not in previous else "known"
            continue
        try:
            download_asset(asset, root, force=args.force, extract=not args.no_extract)
            asset.status = classify_status(asset, previous)
        except (OSError, urllib.error.URLError, zipfile.BadZipFile) as exc:
            asset.status = "error"
            asset.error = str(exc)
            if not args.keep_going:
                raise
        time.sleep(0.1)

    print_summary(assets, previous)
    if args.dry_run:
        return 0
    catalog_summary = None if args.no_catalog else write_catalog(root, args.catalog_dir, assets)
    write_manifest(manifest_path, assets, previous, catalog_summary)
    if catalog_summary:
        print(
            "Catalog written: "
            f"{catalog_summary['file_count']} files, {catalog_summary['bank_count']} banks "
            f"under {root / args.catalog_dir}"
        )
    return 1 if any(asset.status == "error" for asset in assets) else 0


if __name__ == "__main__":
    raise SystemExit(main())
