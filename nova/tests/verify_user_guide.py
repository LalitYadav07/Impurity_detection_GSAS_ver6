"""Verify a bundled guide, optionally through real nginx session routes.

--content-only runs without app dependencies, network access or scientific jobs.
The default also checks the app's native guide anchor and nginx at both root and
NDIP-style paths; run that mode inside the NOVA image with its nginx template.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from html.parser import HTMLParser
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = Counter()
        self.refs = set()
        self.guide_link = None
        self.meta = {}
        self.images = []
        self.copy_targets = []
        self.tasks = []
        self.classes = Counter()
        self.screenshot_templates = 0
        self.option_titles = []
        self._title_parts = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if attributes.get("id") == "radar-user-guide":
            self.guide_link = (tag, attributes)
        if attributes.get("id"):
            self.ids[attributes["id"]] += 1
        for attr in ("src", "href", "data-zoom"):
            if attributes.get(attr):
                self.refs.add(attributes[attr])
        if tag == "meta" and attributes.get("name"):
            self.meta[attributes["name"]] = attributes.get("content")
        if tag == "img" and attributes.get("src"):
            self.images.append(attributes)
        if attributes.get("data-copy"):
            self.copy_targets.append(attributes["data-copy"])
        if tag == "article" and attributes.get("data-task"):
            self.tasks.append(attributes["data-task"])
        self.classes.update(attributes.get("class", "").split())
        if tag == "template" and attributes.get("data-original"):
            self.screenshot_templates += 1
        if tag == "h4":
            self._title_parts = []

    def handle_data(self, data):
        if self._title_parts is not None:
            self._title_parts.append(data)

    def handle_endtag(self, tag):
        if tag == "h4" and self._title_parts is not None:
            self.option_titles.append("".join(self._title_parts))
            self._title_parts = None


def expected_application_version(root: Path) -> str:
    project = root.parent / "pyproject.toml"
    if project.is_file():
        match = re.search(r'^version\s*=\s*"([^"]+)"', project.read_text(), re.M)
        if match:
            return match.group(1)
    return importlib.metadata.version("radar-pd-nova")


def verify_content(root: Path, expected_version: str | None = None) -> dict:
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["schema"] == "radar-pd-user-guide/v2"
    expected_version = expected_version or expected_application_version(root)
    assert manifest["application_version"] == expected_version, "Guide and application versions differ"
    assert manifest["licensed_cif_files_redistributed"] is False
    actual_files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"}
    assert actual_files == set(manifest["files"]), "Unlisted or missing guide files"
    for name, digest in manifest["files"].items():
        path = (root / name).resolve()
        assert path.is_relative_to(root), name
        assert path.suffix.lower() not in {".cif", ".zip"}, name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    image_files = {name for name in manifest["files"] if name.startswith("assets/")}
    assert manifest["screenshots"] == len(image_files)
    captures = json.loads((root / "screenshot-sources.json").read_text())
    assert manifest["captures"] == captures
    assert {capture["path"] for capture in captures.values()} == image_files, "Every image needs provenance"
    for capture_id, capture in captures.items():
        assert capture["application_version"] and capture["context"], capture_id
        assert capture["role"] in {"interface", "historical_example"}, capture_id
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", capture["captured"]), capture_id
        path = root / capture["path"]
        raw = path.read_bytes()
        assert ((path.suffix == ".png" and raw.startswith(b"\x89PNG\r\n\x1a\n")) or
                (path.suffix == ".jpg" and raw.startswith(b"\xff\xd8\xff"))), capture_id
    pages = {}
    assert set(manifest['pages']) == {name for name in actual_files if name.endswith('.html')}, \
        'Every bundled guide page must remain registered and checked'
    for name in manifest["pages"]:
        assert name in manifest["files"], name
        text = (root / name).read_text()
        assert "/Users/" not in text and "file://" not in text, name
        assert "data:image/" not in text, name
        references = References()
        references.feed(text)
        assert all(count == 1 for count in references.ids.values()), f"Duplicate IDs in {name}"
        pages[name] = references
    for name, page in pages.items():
        for ref in page.refs:
            url = urllib.parse.urlsplit(ref)
            if url.scheme or url.netloc:
                assert url.scheme in {"http", "https", "mailto"}, ref
                continue
            target = (root / name).parent / urllib.parse.unquote(url.path) if url.path else root / name
            target = target.resolve()
            assert target.is_relative_to(root), (name, ref)
            relative = target.relative_to(root).as_posix()
            assert relative in manifest["files"], (name, ref)
            if url.fragment:
                assert relative in pages, (name, ref)
                assert urllib.parse.unquote(url.fragment) in pages[relative].ids, (name, ref)
        for image in page.images:
            assert image.get("alt", "").strip(), (name, image)
    main = pages["index.html"]
    if 'experiment-planner.html' in pages:
        assert 'experiment-planner.html' in main.refs, 'The planner guide must remain reachable from the main guide'
    assert main.meta["radar-pd-guide-version"] == expected_version
    assert main.tasks == manifest["tasks"] and len(main.tasks) == len(set(main.tasks))
    assert all(target in main.ids for target in main.copy_targets)
    assert main.copy_targets, "The current guide must include adaptable, copyable examples"
    for image in main.images:
        capture = captures[image["data-capture"]]
        assert image["src"] == capture["path"]
        assert image["data-capture-version"] == capture["application_version"]
    archived = pages[f"reference-{manifest['reference_version']}.html"]
    assert archived.meta["radar-pd-reference-version"] == manifest["reference_version"]
    stats = manifest["reference_statistics"]
    assert archived.classes["lesson"] == stats["lessons"]
    assert archived.classes["option"] == stats["rendered_options"]
    assert len(set(archived.option_titles)) == stats["options"]
    assert archived.screenshot_templates == stats["screenshot_templates"]
    assert set(manifest["inputs"]) == {Path(name).name for name in actual_files if name.startswith("inputs/")}
    return {"content": "passed", "application_version": expected_version, "pages": len(pages),
            "tasks": len(main.tasks), "screenshots": len(image_files), "files": len(actual_files),
            "historical_reference": "preserved", "scientific_jobs_submitted": 0}


def verify_ui_button() -> None:
    from radar_pd_nova.app import RadarPdNovaApp
    interface = References()
    interface.feed(RadarPdNovaApp().layout.html)
    assert interface.guide_link is not None
    tag, attrs = interface.guide_link
    assert tag == "a", "Use a native anchor; the pinned VBtn drops target/rel at runtime"
    assert attrs["href"] == "./user-guide/index.html"
    assert attrs["target"] == "_blank"
    assert attrs["rel"] == "noopener noreferrer"


def verify_routes(root: Path, nginx_template: Path) -> list:
    manifest = json.loads((root / "manifest.json").read_text())
    base = Path("/tmp/radar-pd-nginx")
    base.mkdir(exist_ok=True)
    for name in ("client-body", "proxy", "fastcgi", "uwsgi", "scgi"):
        (base / name).mkdir(exist_ok=True)
    checks = []
    with tempfile.TemporaryDirectory(prefix="radar-guide-check-") as scratch:
        config = Path(scratch) / "nginx.conf"
        for prefix in ("/", "/interactivetool/ep/help-smoke/session"):
            config.write_text(nginx_template.read_text().replace("${EP_PATH}", prefix))
            process = subprocess.Popen(["nginx", "-c", str(config), "-g", "daemon off;"],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                url = "http://127.0.0.1:8081" + prefix.rstrip("/") + "/user-guide/"
                deadline = time.monotonic() + 15
                while True:
                    try:
                        with urllib.request.urlopen(url + "index.html", timeout=2) as response:
                            assert response.headers.get_content_type() == "text/html"
                            assert response.read() == (root / "index.html").read_bytes()
                        break
                    except urllib.error.URLError:
                        if process.poll() is not None or time.monotonic() >= deadline:
                            raise RuntimeError("Packaged guide nginx did not start")
                        time.sleep(.1)
                for name, digest in manifest["files"].items():
                    with urllib.request.urlopen(url + name, timeout=3) as response:
                        assert hashlib.sha256(response.read()).hexdigest() == digest, (prefix, name)
                        expected_types = {".png": "image/png", ".jpg": "image/jpeg", ".html": "text/html", ".css": "text/css"}
                        expected_type = expected_types.get(Path(name).suffix)
                        if expected_type:
                            assert response.headers.get_content_type() == expected_type, name
                checks.append({"prefix": prefix, "files": len(manifest["files"]), "status": "passed"})
            finally:
                process.terminate()
                process.wait(timeout=5)
                error = process.stderr.read().decode(errors="replace")
                assert process.returncode == 0, error[-2000:]
    return checks


def verify(root: Path, nginx_template: Path, expected_version: str | None = None):
    result = verify_content(root, expected_version)
    verify_ui_button()
    result.update(ui_button="passed", routes=verify_routes(root, nginx_template))
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/app/www-content/user-guide"))
    parser.add_argument("--nginx-template", type=Path)
    parser.add_argument("--expected-version")
    parser.add_argument("--content-only", action="store_true")
    args = parser.parse_args()
    if args.content_only:
        print(json.dumps(verify_content(args.root, args.expected_version)))
    else:
        if not args.nginx_template:
            parser.error("--nginx-template is required unless --content-only is selected")
        verify(args.root, args.nginx_template, args.expected_version)
