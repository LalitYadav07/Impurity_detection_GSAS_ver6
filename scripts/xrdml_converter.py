"""Convert common PANalytical/X'Pert XRDML powder files to simple XYE."""

from __future__ import annotations

from pathlib import Path
import math
import re
import xml.etree.ElementTree as ET

import numpy as np


SUPPORTED_DIFFRACTION_UPLOAD_EXTENSIONS = ["dat", "xye", "gsa", "gss", "gsas", "fxye", "xrdml"]


def _local_name(tag: str) -> str:
    return str(tag).rsplit("}", 1)[-1]


def _text_float(node: ET.Element | None) -> float | None:
    if node is None or node.text is None:
        return None
    try:
        return float(str(node.text).strip())
    except ValueError:
        return None


def _number_list(text: str | None) -> list[float]:
    if not text:
        return []
    values: list[float] = []
    for token in re.split(r"[\s,;]+", str(text).strip()):
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def _iter_children_by_name(node: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(node) if _local_name(child.tag) == name]


def _find_first_descendant(node: ET.Element, name: str) -> ET.Element | None:
    for child in node.iter():
        if _local_name(child.tag) == name:
            return child
    return None


def _axis_positions(data_points: ET.Element, n: int) -> np.ndarray | None:
    positions_nodes = _iter_children_by_name(data_points, "positions")
    selected = None
    for node in positions_nodes:
        axis = str(node.attrib.get("axis", "")).lower()
        if "2theta" in axis or "2-theta" in axis or axis in {"2th", "2t"}:
            selected = node
            break
    if selected is None and positions_nodes:
        selected = positions_nodes[0]
    if selected is None:
        return None

    explicit_positions = _number_list(selected.text)
    if len(explicit_positions) == n:
        x = np.asarray(explicit_positions, dtype=float)
        if np.all(np.isfinite(x)):
            return x

    start = _text_float(_find_first_descendant(selected, "startPosition"))
    end = _text_float(_find_first_descendant(selected, "endPosition"))
    if start is None or end is None or n <= 1:
        return None
    return np.linspace(float(start), float(end), int(n), dtype=float)


def _scan_candidates(root: ET.Element) -> list[tuple[np.ndarray, np.ndarray]]:
    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for data_points in root.iter():
        if _local_name(data_points.tag) != "dataPoints":
            continue

        intensity_node = None
        for name in ("intensities", "counts"):
            intensity_node = _find_first_descendant(data_points, name)
            if intensity_node is not None:
                break
        intensities = _number_list(intensity_node.text if intensity_node is not None else None)
        if len(intensities) < 2:
            continue

        x = _axis_positions(data_points, len(intensities))
        if x is None:
            continue
        y = np.asarray(intensities, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if len(x) < 2:
            continue
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        if np.any(np.diff(x) <= 0):
            unique_x, unique_idx = np.unique(x, return_index=True)
            x = unique_x
            y = y[unique_idx]
        if len(x) >= 2:
            candidates.append((x, y))
    return candidates


def convert_xrdml_to_xye(input_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Convert an XRDML file to a three-column 2theta/intensity/sigma XYE file."""
    input_path = Path(input_path)
    if input_path.suffix.lower() != ".xrdml":
        return input_path
    if not input_path.exists():
        raise FileNotFoundError(f"XRDML file not found: {input_path}")

    try:
        root = ET.parse(input_path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Could not parse XRDML XML file {input_path.name}: {exc}") from exc

    candidates = _scan_candidates(root)
    if not candidates:
        raise ValueError(
            f"Could not find 2theta positions and intensities in XRDML file {input_path.name}"
        )

    x, y = max(candidates, key=lambda item: len(item[0]))
    output_path = Path(output_path) if output_path else input_path.with_suffix(".xye")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# Converted by RADAR-PD from XRDML: {input_path.name}\n")
        handle.write("# 2theta intensity sigma\n")
        for xi, yi in zip(x, y):
            sigma = math.sqrt(max(float(yi), 1.0))
            handle.write(f"{float(xi):.8f} {float(yi):.8f} {sigma:.8f}\n")

    if output_path.stat().st_size <= 0:
        raise RuntimeError(f"Failed to create converted XYE file: {output_path}")
    return output_path


def prepare_powder_data_file(data_path: str | Path) -> Path:
    """Return a GSAS-readable path, converting XRDML to adjacent XYE when needed."""
    path = Path(data_path)
    if path.suffix.lower() == ".xrdml":
        return convert_xrdml_to_xye(path)
    return path
