import re
import xml.etree.ElementTree as ET
from pathlib import Path


NOVA_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (NOVA_ROOT / relative_path).read_text(encoding="utf-8")


def test_runtime_files_are_writable_by_ndip_job_user() -> None:
    nginx_script = _read("scripts/run_nginx.sh")
    nginx_config = _read("dockerfiles/nginx.conf.template")
    dockerfile = _read("dockerfiles/Dockerfile")

    assert "/tmp/radar-pd-nginx.conf" in nginx_script
    assert "client_body_temp_path /tmp/radar-pd-nginx/" in nginx_config
    assert "proxy_temp_path /tmp/radar-pd-nginx/" in nginx_config
    assert "fastcgi_temp_path /tmp/radar-pd-nginx/" in nginx_config
    assert "uwsgi_temp_path /tmp/radar-pd-nginx/" in nginx_config
    assert "scgi_temp_path /tmp/radar-pd-nginx/" in nginx_config
    assert "USER 1000:1000" in dockerfile


def test_container_exits_when_either_service_exits() -> None:
    launcher = _read("scripts/run_container.sh")
    dockerfile = _read("dockerfiles/Dockerfile")
    galaxy_xml = _read("galaxy/radar_pd_nova.xml")

    assert "wait -n" in launcher
    assert 'kill -TERM "${trame_pid}"' in launcher
    assert 'kill -TERM "${nginx_pid}"' in launcher
    assert 'exit "${child_status}"' in launcher
    assert "autorestart" not in launcher
    assert "supervisor" not in dockerfile.lower()
    assert "supervisor" not in galaxy_xml.lower()
    assert 'CMD ["/usr/local/bin/run_container.sh"]' in dockerfile
    assert "/usr/local/bin/run_container.sh" in galaxy_xml


def test_nova_prefix_redirects_to_the_canonical_trailing_slash() -> None:
    nginx_script = _read("scripts/run_nginx.sh")
    nginx_config = _read("dockerfiles/nginx.conf.template")

    assert "ep_path=\"${ep_path%/}\"" in nginx_script
    assert "^(/[A-Za-z0-9._~-]+)+$" in nginx_script
    assert "absolute_redirect off;" in nginx_config
    assert "location = ${EP_PATH} {" in nginx_config
    assert "return 308 ${EP_PATH}/$is_args$args;" in nginx_config
    assert "location ^~ ${EP_PATH}/ {" in nginx_config


def test_package_and_galaxy_tool_versions_match() -> None:
    pyproject = _read("pyproject.toml")
    version_match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    assert version_match is not None
    package_version = version_match.group(1)
    tool_version = ET.parse(NOVA_ROOT / "galaxy" / "radar_pd_nova.xml").getroot().get(
        "version"
    )

    assert package_version == tool_version


def test_interactive_startup_keeps_a_console_output_for_ndip_lifecycle() -> None:
    galaxy_xml = _read("galaxy/radar_pd_nova.xml")
    launcher = _read("scripts/run_container.sh")
    root = ET.fromstring(galaxy_xml)
    outputs = root.find("outputs")

    assert outputs is not None
    declared_outputs = list(outputs)
    assert len(declared_outputs) == 1
    assert declared_outputs[0].get("name") == "console_output"
    assert declared_outputs[0].get("format") == "txt"
    assert 'tee -a "$console_output"' in root.findtext("command", default="")
    assert "touch " not in root.findtext("command", default="")
    assert not re.search(r"(^|\s)(echo|printf)\s", launcher)
