"""Controlled SNS/HFIR IPTS browsing, publishing, and watch-recipe support.

The NOVA application must never expose an unrestricted server-side file
browser. This module confines every user-selected path to one instrument/IPTS
tree. Reads may use any accessible experiment subdirectory; writes are limited
to a newly named results folder below an explicitly selected writable working
directory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from fnmatch import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any, Iterable, Literal


PATTERN_SUFFIXES = {
    ".csv",
    ".dat",
    ".fxye",
    ".gsa",
    ".gsas",
    ".gss",
    ".txt",
    ".xml",
    ".xrdml",
    ".xy",
    ".xye",
}
INSTRUMENT_SUFFIXES = {".ins", ".inst", ".instprm", ".prm"}
CIF_SUFFIXES = {".cif"}
CONFIG_SUFFIXES = {".json", ".yaml", ".yml"}
ROLE_SUFFIXES = {
    "data": PATTERN_SUFFIXES,
    "instrument": INSTRUMENT_SUFFIXES,
    "cif": CIF_SUFFIXES,
    "config": CONFIG_SUFFIXES,
    "any": PATTERN_SUFFIXES | INSTRUMENT_SUFFIXES | CIF_SUFFIXES | CONFIG_SUFFIXES | {".zip"},
}

_FACILITY_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_NEW_DIRECTORY_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_. -]{0,79}$")
FACILITY_ROOTS = {
    "SNS": "/SNS",
    "HFIR": "/HFIR",
}

# RADAR-PD consumes powder-diffraction patterns, so the interactive browser
# should not expose spectroscopy, reflectometry, imaging, or SANS beamlines.
# Keys are directory names used in the facility filesystems. Aliases cover
# installations where the human-facing and filesystem instrument names differ.
DIFFRACTION_BEAMLINES: dict[str, dict[str, str]] = {
    "SNS": {
        "NOM": "NOMAD (BL-1B) - total scattering / powder diffraction",
        "NOMAD": "NOMAD (BL-1B) - total scattering / powder diffraction",
        "POWGEN": "POWGEN (BL-11A) - powder diffraction",
        "SNAP": "SNAP (BL-3) - high-pressure powder diffraction",
        "VULCAN": "VULCAN (BL-7) - engineering diffraction",
    },
    "HFIR": {
        "HB2A": "POWDER (HB-2A) - powder diffraction",
        "HB2B": "HIDRA (HB-2B) - engineering diffraction",
        "HB2C": "WAND2 (HB-2C) - powder diffraction",
        "WAND": "WAND2 (HB-2C) - powder diffraction",
    },
}


class FacilityPathError(ValueError):
    """Raised when a facility path is outside the permitted IPTS boundary."""


def build_facility_export_path(
    facility_root: str,
    instrument: str,
    ipts: str,
    working_directory: str,
    result_name: str,
    *,
    filename_suffix: str = "results.zip",
) -> str:
    """Build a confined SNS/HFIR destination for NDIP's export tool.

    This function intentionally performs no filesystem access. The NOVA pod
    may browse a read-only mount while NDIP executes ``Export Datasets`` on an
    authenticated analysis-cluster destination. Every user-controlled path
    component is nevertheless validated before the destination is submitted.
    """

    normalized_root = str(facility_root or "").strip().replace("\\", "/").rstrip("/").upper()
    facility = next((name for name, root in FACILITY_ROOTS.items() if normalized_root == root.upper()), None)
    if facility is None:
        raise FacilityPathError("Authenticated export supports only the /SNS and /HFIR facility roots")
    relative = PurePosixPath(_normalize_relative(working_directory))
    if not relative.parts or relative.parts[0].lower() != "shared":
        raise FacilityPathError("Authenticated result export is limited to the selected IPTS shared/ tree")
    clean_result = str(result_name or "").strip()
    if not _NEW_DIRECTORY_NAME.fullmatch(clean_result) or clean_result in {".", ".."}:
        raise FacilityPathError("Invalid result name")
    clean_suffix = str(filename_suffix or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", clean_suffix):
        raise FacilityPathError("Invalid exported filename suffix")
    # NDIP's authenticated exporter copies to one exact file path and does not
    # create missing parent directories. Keep every parent equal to the
    # existing folder selected in NOVA and make the archive filename unique.
    clean_filename = f"{clean_result}-{clean_suffix}".replace(" ", "_")
    destination = (
        PurePosixPath(FACILITY_ROOTS[facility])
        / _component(instrument, "instrument")
        / _ipts_component(ipts)
        / relative
        / clean_filename
    )
    return destination.as_posix()


@dataclass(frozen=True)
class FacilityEntry:
    name: str
    relative_path: str
    kind: Literal["directory", "file"]
    size: int | None = None
    modified_utc: str | None = None
    suffix: str = ""

    def as_item(self) -> dict[str, Any]:
        modified = self.modified_utc.replace("T", " ").replace("+00:00", " UTC") if self.modified_utc else ""
        size = _human_size(self.size) if self.size is not None else ""
        detail = "Folder" if self.kind == "directory" else " | ".join(value for value in (size, modified) if value)
        return {
            **asdict(self),
            "title": self.name,
            "value": self.relative_path,
            "detail": detail,
            "icon": "mdi-folder" if self.kind == "directory" else "mdi-file-document-outline",
        }


@dataclass(frozen=True)
class FacilitySelection:
    facility: str
    facility_root: str
    instrument: str
    ipts: str
    relative_path: str
    absolute_path: str
    size: int
    modified_utc: str
    sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WatchRecipe:
    """Portable contract consumed by the persistent NDIP watcher."""

    instrument: str
    ipts: str
    source_directory: str
    output_directory: str
    configuration: str
    instrument_profile: str | None = None
    use_builtin_cuka: bool = False
    main_cif: str | None = None
    include_patterns: list[str] = field(default_factory=lambda: ["*.dat", "*.xye", "*.fxye", "*.gsa", "*.xrdml"])
    settle_seconds: int = 60
    process_existing: bool = False
    max_attempts: int = 3
    retry_delay_seconds: int = 120
    analysis_mode: Literal["rapid", "full"] = "rapid"
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema: str = "radar-pd-watch/v1"

    def validate(self, browser: "FacilityBrowser") -> "WatchRecipe":
        if self.settle_seconds < 10:
            raise ValueError("Watch settle time must be at least 10 seconds")
        if not 1 <= self.max_attempts <= 10:
            raise ValueError("Watch retry attempts must be between 1 and 10")
        if self.retry_delay_seconds < 10:
            raise ValueError("Watch retry delay must be at least 10 seconds")
        if not self.include_patterns or any("/" in pattern or "\\" in pattern for pattern in self.include_patterns):
            raise ValueError("Watch filename patterns must be simple file globs")
        browser.resolve_directory(self.instrument, self.ipts, self.source_directory)
        browser.resolve_directory(self.instrument, self.ipts, self.output_directory)
        browser.resolve_file(self.instrument, self.ipts, self.configuration, role="config")
        if self.instrument_profile:
            browser.resolve_file(self.instrument, self.ipts, self.instrument_profile, role="instrument")
        elif not self.use_builtin_cuka:
            raise ValueError("A watch recipe requires an instrument profile or the built-in Cu K-alpha profile")
        if self.main_cif:
            browser.resolve_file(self.instrument, self.ipts, self.main_cif, role="cif")
        output = PurePosixPath(_normalize_relative(self.output_directory))
        source = PurePosixPath(_normalize_relative(self.source_directory))
        if output == source:
            raise ValueError("Watch output directory must differ from the source directory")
        return self

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["$schema"] = payload.pop("schema")
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WatchRecipe":
        payload = dict(value)
        payload["schema"] = payload.pop("$schema", "radar-pd-watch/v1")
        return cls(**payload)


@dataclass(frozen=True)
class WatchCandidate:
    relative_path: str
    size: int
    modified_ns: int
    fingerprint: str


class FacilityBrowser:
    """Browse one mounted facility while confining access to a chosen IPTS."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        facility: str = "SNS",
    ) -> None:
        normalized = str(facility or "SNS").strip().upper()
        if normalized not in FACILITY_ROOTS:
            raise FacilityPathError(f"Unknown facility: {facility!r}")
        self.facility = normalized
        if root is None:
            env_name = f"RADAR_PD_{normalized}_ROOT"
            fallback = os.getenv("RADAR_PD_FACILITY_ROOT") if normalized == "SNS" else None
            root = os.getenv(env_name) or fallback or FACILITY_ROOTS[normalized]
        self.root = Path(root).resolve()

    @classmethod
    def for_root(cls, root: str | Path) -> "FacilityBrowser":
        """Create a browser for a persisted absolute facility root."""

        resolved = Path(root).resolve()
        facility = "HFIR" if resolved.as_posix().rstrip("/").upper().endswith("/HFIR") else "SNS"
        return cls(resolved, facility=facility)

    @property
    def available(self) -> bool:
        return self.root.is_dir()

    def list_instruments(self) -> list[dict[str, str]]:
        if not self.available:
            return []
        supported = DIFFRACTION_BEAMLINES[self.facility]
        instruments: list[dict[str, str]] = []
        for path in _visible_directories(self.root):
            name = path.name.upper()
            title = supported.get(name)
            if title and _FACILITY_COMPONENT.fullmatch(path.name) and _has_ipts_child(path):
                instruments.append({"title": title, "value": path.name})
        return instruments

    def list_ipts(self, instrument: str, *, limit: int = 250) -> list[dict[str, str]]:
        """List recent IPTS names without stat'ing every entry on facility NFS.

        Some instruments have a large historical IPTS tree. Calling ``is_dir``
        or ``os.access`` for each child turns one instrument selection into
        hundreds of network metadata requests. Directory names are sufficient
        for suggestions; the selected path is still resolved and validated
        before it can be browsed.
        """

        instrument_root = self._instrument_root(instrument)
        if not instrument_root.is_dir():
            return []
        names: list[tuple[int, str]] = []
        try:
            with os.scandir(instrument_root) as entries:
                for entry in entries:
                    match = re.fullmatch(r"IPTS-(\d+)", entry.name, re.IGNORECASE)
                    if match:
                        names.append((int(match.group(1)), entry.name))
        except OSError:
            return []
        names.sort(key=lambda item: item[0], reverse=True)
        selected = names[: max(1, int(limit))]
        return [{"title": name, "value": name} for _, name in selected]

    def list_directory(
        self,
        instrument: str,
        ipts: str,
        relative_path: str = ".",
        *,
        role: str = "data",
    ) -> list[FacilityEntry]:
        directory = self.resolve_directory(instrument, ipts, relative_path)
        suffixes = ROLE_SUFFIXES.get(role)
        if suffixes is None:
            raise ValueError(f"Unknown facility file role: {role}")
        entries: list[FacilityEntry] = []
        try:
            children = sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.casefold()))
        except OSError as exc:
            raise FacilityPathError(f"Cannot read IPTS directory: {exc}") from exc
        ipts_root = self._ipts_root(instrument, ipts)
        for child in children:
            if child.name.startswith("."):
                continue
            try:
                resolved = child.resolve(strict=True)
                resolved.relative_to(ipts_root)
                stat = resolved.stat()
            except (OSError, ValueError):
                continue
            relative = resolved.relative_to(self._ipts_root(instrument, ipts)).as_posix()
            if resolved.is_dir():
                entries.append(FacilityEntry(child.name, relative, "directory"))
            elif resolved.is_file() and resolved.suffix.lower() in suffixes:
                entries.append(
                    FacilityEntry(
                        child.name,
                        relative,
                        "file",
                        size=stat.st_size,
                        modified_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                        suffix=resolved.suffix.lower(),
                    )
                )
        return entries

    def resolve_directory(self, instrument: str, ipts: str, relative_path: str = ".") -> Path:
        path = self._resolve_ipts_path(instrument, ipts, relative_path)
        if not path.is_dir():
            raise FacilityPathError(f"IPTS folder does not exist: {relative_path}")
        return path

    def resolve_file(self, instrument: str, ipts: str, relative_path: str, *, role: str) -> Path:
        suffixes = ROLE_SUFFIXES.get(role)
        if suffixes is None:
            raise ValueError(f"Unknown facility file role: {role}")
        path = self._resolve_ipts_path(instrument, ipts, relative_path)
        if not path.is_file():
            raise FacilityPathError(f"IPTS file does not exist: {relative_path}")
        if path.suffix.lower() not in suffixes:
            expected = ", ".join(sorted(suffixes))
            raise FacilityPathError(f"Selected {role} file must use one of: {expected}")
        return path

    def select_file(
        self,
        instrument: str,
        ipts: str,
        relative_path: str,
        *,
        role: str,
        checksum: bool = False,
    ) -> FacilitySelection:
        path = self.resolve_file(instrument, ipts, relative_path, role=role)
        stat = path.stat()
        return FacilitySelection(
            facility=self.facility,
            facility_root=str(self.root),
            instrument=_component(instrument, "instrument"),
            ipts=_ipts_component(ipts),
            relative_path=path.relative_to(self._ipts_root(instrument, ipts)).as_posix(),
            absolute_path=str(path),
            size=stat.st_size,
            modified_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            sha256=_sha256(path) if checksum else None,
        )

    def parent_directory(self, relative_path: str) -> str:
        normalized = PurePosixPath(_normalize_relative(relative_path))
        if normalized == PurePosixPath("."):
            return "."
        parent = normalized.parent.as_posix()
        return parent if parent not in {"", "/"} else "."

    def directory_access(self, instrument: str, ipts: str, relative_path: str) -> dict[str, bool]:
        """Return effective access flags without creating or modifying files."""

        directory = self.resolve_directory(instrument, ipts, relative_path)
        return {
            "readable": os.access(directory, os.R_OK),
            "searchable": os.access(directory, os.X_OK),
            "writable": os.access(directory, os.W_OK | os.X_OK),
        }

    def create_directory(self, instrument: str, ipts: str, parent: str, name: str) -> str:
        clean_name = str(name or "").strip()
        if not _NEW_DIRECTORY_NAME.fullmatch(clean_name) or clean_name in {".", ".."}:
            raise FacilityPathError("Folder names may contain letters, numbers, spaces, '.', '_' and '-'")
        parent_path = self.resolve_directory(instrument, ipts, parent)
        if not os.access(parent_path, os.W_OK | os.X_OK):
            raise PermissionError(f"Selected folder is not writable: {parent}")
        target = (parent_path / clean_name).resolve()
        target.relative_to(self._ipts_root(instrument, ipts))
        target.mkdir(mode=0o770, exist_ok=False)
        return target.relative_to(self._ipts_root(instrument, ipts)).as_posix()

    def ensure_output_parent(
        self,
        instrument: str,
        ipts: str,
        working_directory: str,
        subfolder_name: str,
    ) -> str:
        """Create or validate the named result container below a work folder."""

        clean_name = str(subfolder_name or "").strip()
        if not _NEW_DIRECTORY_NAME.fullmatch(clean_name) or clean_name in {".", ".."}:
            raise FacilityPathError("Result subfolder names may contain letters, numbers, spaces, '.', '_' and '-'")
        working = self.resolve_directory(instrument, ipts, working_directory)
        if not os.access(working, os.W_OK | os.X_OK):
            raise PermissionError(f"Selected working folder is not writable: {working_directory}")
        target = (working / clean_name).resolve()
        target.relative_to(self._ipts_root(instrument, ipts))
        if target.exists() and not target.is_dir():
            raise FileExistsError(f"Result destination is not a directory: {target}")
        target.mkdir(mode=0o770, exist_ok=True)
        if not os.access(target, os.W_OK | os.X_OK):
            raise PermissionError(f"Result destination is not writable: {target}")
        return target.relative_to(self._ipts_root(instrument, ipts)).as_posix()

    def write_watch_recipe(self, recipe: WatchRecipe, filename: str = "radar-pd-watch.json") -> Path:
        recipe.validate(self)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.json", filename):
            raise FacilityPathError("Watch recipe filename must end in .json")
        source = self.resolve_directory(recipe.instrument, recipe.ipts, recipe.source_directory)
        destination = source / filename
        _atomic_write_json(destination, recipe.as_dict())
        return destination

    def write_configuration(
        self,
        instrument: str,
        ipts: str,
        directory: str,
        payload: dict[str, Any],
        filename: str = "radar-pd-watch-config.yaml",
    ) -> Path:
        """Atomically save a portable RADAR-PD configuration in an IPTS folder."""

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.ya?ml", filename):
            raise FacilityPathError("Configuration filename must end in .yaml or .yml")
        destination = self.resolve_directory(instrument, ipts, directory) / filename
        import yaml

        _atomic_write_text(
            destination,
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        )
        return destination

    def publish_directory(
        self,
        source: str | Path,
        instrument: str,
        ipts: str,
        destination_parent: str,
        result_name: str,
    ) -> Path:
        source_path = Path(source).resolve()
        if not source_path.is_dir():
            raise FileNotFoundError(f"Result source does not exist: {source_path}")
        parent = self.resolve_directory(instrument, ipts, destination_parent)
        if not os.access(parent, os.W_OK | os.X_OK):
            raise PermissionError(f"Selected result folder is not writable: {destination_parent}")
        clean_name = str(result_name or "").strip()
        if not _NEW_DIRECTORY_NAME.fullmatch(clean_name):
            raise FacilityPathError("Invalid result folder name")
        destination = (parent / clean_name).resolve()
        destination.relative_to(self._ipts_root(instrument, ipts))
        if destination.exists():
            raise FileExistsError(f"Result folder already exists: {destination}")
        staging = Path(tempfile.mkdtemp(prefix=f".{clean_name}-", dir=parent))
        try:
            for child in source_path.iterdir():
                target = staging / child.name
                if child.is_dir():
                    shutil.copytree(child, target)
                elif child.is_file():
                    shutil.copy2(child, target)
            manifest = {
                "$schema": "radar-pd-published-result/v1",
                "published_utc": datetime.now(timezone.utc).isoformat(),
                "destination": destination.relative_to(self._ipts_root(instrument, ipts)).as_posix(),
                "files": [
                    path.relative_to(staging).as_posix()
                    for path in sorted(staging.rglob("*"))
                    if path.is_file()
                ],
            }
            _atomic_write_json(staging / "published_manifest.json", manifest)
            staging.replace(destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return destination

    def discover_watch_candidates(self, recipe: WatchRecipe) -> list[WatchCandidate]:
        recipe.validate(self)
        source = self.resolve_directory(recipe.instrument, recipe.ipts, recipe.source_directory)
        output = self.resolve_directory(recipe.instrument, recipe.ipts, recipe.output_directory)
        result: list[WatchCandidate] = []
        for path in sorted(source.iterdir()):
            if path.name.startswith(".") or not path.is_file():
                continue
            if output == path or output in path.parents:
                continue
            if not any(fnmatch(path.name, pattern) for pattern in recipe.include_patterns):
                continue
            if path.suffix.lower() not in PATTERN_SUFFIXES:
                continue
            stat = path.stat()
            relative = path.resolve().relative_to(self._ipts_root(recipe.instrument, recipe.ipts)).as_posix()
            fingerprint = hashlib.sha256(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()).hexdigest()
            result.append(WatchCandidate(relative, stat.st_size, stat.st_mtime_ns, fingerprint))
        return result

    def _instrument_root(self, instrument: str) -> Path:
        component = _component(instrument, "instrument")
        # SNS/HFIR instrument entries can be facility-managed symlinks into the
        # underlying experiment storage. Keep this lexical root long enough to
        # select the managed IPTS entry, then use the resolved IPTS directory as
        # the containment boundary for every user-controlled descendant path.
        return self.root / component

    def _ipts_root(self, instrument: str, ipts: str) -> Path:
        return (self._instrument_root(instrument) / _ipts_component(ipts)).resolve()

    def _resolve_ipts_path(self, instrument: str, ipts: str, relative_path: str) -> Path:
        relative = _normalize_relative(relative_path)
        candidate = (self._ipts_root(instrument, ipts) / Path(*PurePosixPath(relative).parts)).resolve()
        try:
            candidate.relative_to(self._ipts_root(instrument, ipts))
        except ValueError as exc:
            raise FacilityPathError("Only paths inside the selected IPTS are permitted") from exc
        return candidate


def _component(value: str, label: str) -> str:
    normalized = str(value or "").strip()
    if not _FACILITY_COMPONENT.fullmatch(normalized):
        raise FacilityPathError(f"Invalid {label}: {value!r}")
    return normalized


def _ipts_component(value: str) -> str:
    normalized = str(value or "").strip().upper().replace("_", "-")
    match = re.fullmatch(r"(?:IPTS-?)?(\d+)", normalized)
    if not match:
        raise FacilityPathError(f"Invalid IPTS: {value!r}")
    return f"IPTS-{match.group(1)}"


def _normalize_relative(value: str) -> str:
    text = str(value or ".").strip().replace("\\", "/") or "."
    if text == ".":
        return "."
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise FacilityPathError("Select a path relative to the chosen IPTS")
    return path.as_posix()


def _visible_directories(root: Path) -> Iterable[Path]:
    visible: list[Path] = []
    try:
        for path in root.iterdir():
            if path.name.startswith("."):
                continue
            try:
                if path.is_dir() and os.access(path, os.R_OK | os.X_OK):
                    visible.append(path)
            except OSError:
                # Shared facility trees can expose directory entries whose
                # metadata is intentionally inaccessible to this user.
                continue
    except OSError:
        return []
    return sorted(visible, key=lambda item: item.name.casefold())


def _has_ipts_child(root: Path) -> bool:
    """Return whether a facility directory looks like an instrument tree."""

    return any(
        re.fullmatch(r"IPTS-\d+", path.name, re.IGNORECASE)
        for path in _visible_directories(root)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _human_size(value: int | None) -> str:
    if value is None:
        return ""
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
