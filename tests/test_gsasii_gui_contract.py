from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GUI = ROOT / "gsasii_gui"


def _read(relative_path: str) -> str:
    return (GUI / relative_path).read_text(encoding="utf-8")


def test_gui_image_pins_gsasii_and_provides_native_desktop_stack() -> None:
    dockerfile = _read("Dockerfile")

    assert "ARG GSASII_REF=dcd09eb2a4392b94d22cda3c69021137e6b14620" in dockerfile
    assert "git checkout --detach \"${GSASII_REF}\"" in dockerfile
    assert "wxpython>=4.2" in _read("environment.yml")
    for package in ("nginx", "novnc", "openbox", "websockify", "x11vnc", "xvfb"):
        assert package in dockerfile
    assert "EXPOSE 8080" in dockerfile


def test_desktop_gateway_obeys_ndip_path_prefix() -> None:
    nginx = _read("nginx.conf.template")
    launcher = _read("scripts/run_nginx.sh")

    assert "location ${EP_PATH}/websockify" in nginx
    assert "path=${EP_PATH}/websockify" in nginx
    assert "alias /usr/share/novnc/" in nginx
    assert "envsubst '${EP_PATH}'" in launcher
    assert "^(/[A-Za-z0-9._~-]+)+$" in launcher


def test_gpx_is_edited_as_a_copy_and_continuously_preserved() -> None:
    start = _read("scripts/start.sh")
    sync = _read("scripts/run_project_sync.sh")
    gui = _read("scripts/run_gsasii.sh")

    assert "install -o gsasii -g gsasii -m 0644" in start
    assert '"${session_dir}/radar_pd_project.gpx"' in start
    assert 'source_digest="$(sha256sum "${source_project}"' in sync
    assert 'cat "${snapshot}" > "${output_project}"' in sync
    assert "sleep 0.5" in sync
    assert '/opt/conda/bin/GSAS-II "${project}"' in gui
    assert 'cat "${project}" > "${output_project}"' in gui
