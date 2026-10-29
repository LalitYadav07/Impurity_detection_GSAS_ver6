"""Prepare a reviewed visual-guide export for the public application image.

Only the page, its embedded screenshots and the listed non-CIF inputs are
included. Licensed ICSD CIFs remain with the user's licensed database access.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re


INPUTS = (
    "ages-analysis-settings.yaml", "reviewed-temperature-axis.csv",
    "llm-refinement-request.txt", "approved-refinement-recipe.json",
)


def package(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    assets = destination / "assets"
    assets.mkdir(exist_ok=True)
    files = {}

    def extract_image(match):
        raw = base64.b64decode(match.group(1), validate=True)
        # Capture exports can carry a .png label around original JPEG bytes.
        # Preserve those bytes and serve their actual media type.
        if raw.startswith(b"\x89PNG\r\n\x1a\n"):
            extension = "png"
        elif raw.startswith(b"\xff\xd8\xff"):
            extension = "jpg"
        else:
            raise ValueError("The guide export contains an unsupported screenshot format")
        digest = hashlib.sha256(raw).hexdigest()
        name = f"assets/{digest[:20]}.{extension}"
        (destination / name).write_bytes(raw)
        files[name] = digest
        return name

    content = (source / "index.html").read_text(encoding="utf-8")
    content = re.sub(r"data:image/png;base64,([A-Za-z0-9+/=]+)", extract_image, content)
    content = content.replace('href="inputs/Al2O3-MgO-cifs.zip"', 'href="#cif-inputs"')
    content = content.replace("Download the two-CIF input", "Prepare the two-CIF input")
    content = content.replace("Two-CIF input ZIP</a>", "Prepare the two-CIF input</a>")
    content = content.replace(
        "NOVA 0.4.34 · 29 October 2026",
        "NOVA 0.4.35 · Screenshots: 0.4.34 · 29 October 2026",
    )
    content = content.replace(
        "Keep this HTML file with its inputs folder, or extract the complete guide bundle before opening it. Screenshots are embedded in the page and work offline.",
        "This guide and its example configuration are served by your RADAR-PD session. Keep that session open while reading. The screenshot version is identified above.",
    )
    content = content.replace(
        '<h3>Start stays disabled</h3>',
        '<h3 id="cif-inputs">Prepare the two-CIF input</h3>'
        '<p>Obtain reviewed Al2O3 and MgO CIFs through your licensed crystallographic '
        'database or another source you may use. Put them in a ZIP named '
        'Al2O3-MgO-cifs.zip, or add the two loose CIF files in the app. The recorded '
        'example used ICSD 9770 for Al2O3 and the supplied MgO model. Licensed '
        'ICSD exports are not redistributed with this guide.</p>'
        '<h3>Start stays disabled</h3>',
    )
    content = re.sub(r'<a href="(inputs/[^"<>]+)"', r'<a download href="\1"', content)
    (destination / "inputs").mkdir(exist_ok=True)
    for name in INPUTS:
        raw = (source / "inputs" / name).read_text(encoding="utf-8").encode("utf-8")
        (destination / "inputs" / name).write_bytes(raw)
        files["inputs/" + name] = hashlib.sha256(raw).hexdigest()
    (destination / "index.html").write_text(content, encoding="utf-8")
    files["index.html"] = hashlib.sha256(content.encode()).hexdigest()
    manifest = {
        "schema": "radar-pd-user-guide/v1", "application_version": "0.4.35",
        "screenshot_version": "0.4.34", "screenshots": len(files) - len(INPUTS) - 1,
        "files": dict(sorted(files.items())),
        "inputs": list(INPUTS), "licensed_cif_files_redistributed": False,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"destination": str(destination), "screenshots": manifest["screenshots"],
                      "downloadable_inputs": len(INPUTS), "html_bytes": len(content.encode())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    package(args.source, args.destination)
