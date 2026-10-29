"""Exercise the real packaged nginx routes at root and NDIP-style prefixes.

Run inside the built NOVA container with this file and nginx.conf.template
mounted read-only. This submits no Galaxy jobs and requires no credentials.
"""
from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.refs = set()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.add(attributes["id"])
        for attr in ("src", "href"):
            if attributes.get(attr):
                self.refs.add(attributes[attr])


def verify(root: Path, nginx_template: Path):
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["application_version"] == "0.4.35"
    assert manifest["screenshots"] == len(list((root / "assets").iterdir()))
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    assert not list(root.rglob("*.cif"))
    assert not list(root.rglob("*.zip"))
    text = (root / "index.html").read_text()
    assert text.count('<template data-original=') == 53
    assert "/Users/" not in text and "file://" not in text
    assert "data:image/png;base64," not in text
    links = References()
    links.feed(text)
    for url in links.refs:
        if url.startswith("#"):
            assert url[1:] in links.ids, url
        elif not urllib.parse.urlsplit(url).scheme:
            assert url in manifest["files"], url

    from radar_pd_nova.app import RadarPdNovaApp
    app = RadarPdNovaApp()
    layout = app.layout.html
    assert 'id="radar-user-guide"' in layout
    assert 'href="./user-guide/index.html"' in layout
    assert 'target="_blank"' in layout
    assert 'rel="noopener noreferrer"' in layout

    base = Path("/tmp/radar-pd-nginx")
    base.mkdir(exist_ok=True)
    for name in ("client-body", "proxy", "fastcgi", "uwsgi", "scgi"):
        (base / name).mkdir(exist_ok=True)
    checks = []
    with tempfile.TemporaryDirectory(prefix="radar-guide-check-") as scratch:
        config = Path(scratch) / "nginx.conf"
        for prefix in ("/", "/interactivetool/ep/help-smoke/session"):
            config.write_text(nginx_template.read_text().replace("${EP_PATH}", prefix))
            process = subprocess.Popen(
                ["nginx", "-c", str(config), "-g", "daemon off;"],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
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
                        raw = response.read()
                        assert hashlib.sha256(raw).hexdigest() == digest, (prefix, name)
                        if name.endswith((".png", ".jpg")):
                            expected = "image/png" if name.endswith(".png") else "image/jpeg"
                            assert response.headers.get_content_type() == expected, name
                checks.append({"prefix": prefix, "files": len(manifest["files"]), "status": "passed"})
            finally:
                process.terminate()
                process.wait(timeout=5)
                error = process.stderr.read().decode(errors="replace")
                assert process.returncode == 0, error[-2000:]
    print(json.dumps({"user_guide": "passed", "ui_button": "passed", "routes": checks}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/app/www-content/user-guide"))
    parser.add_argument("--nginx-template", type=Path, required=True)
    args = parser.parse_args()
    verify(args.root, args.nginx_template)
