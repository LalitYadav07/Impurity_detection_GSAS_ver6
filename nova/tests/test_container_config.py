import re
import xml.etree.ElementTree as ET
from pathlib import Path


NOVA_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (NOVA_ROOT / relative_path).read_text(encoding="utf-8")


def test_container_reuses_supported_nova_supervisor_contract() -> None:
    dockerfile = _read("dockerfiles/Dockerfile")
    supervisor = _read("dockerfiles/supervisord.conf")

    assert dockerfile.startswith(
        "FROM savannah.ornl.gov/radar-pd/radar-pd-nova:nova-0.2.1\n"
    )
    assert "ENV PIXI_ENVIRONMENT_NAME=production" in dockerfile
    assert (
        "python -m pip install --no-cache-dir --no-deps --force-reinstall /src"
        in dockerfile
    )
    assert "USER 1000:1000" not in dockerfile
    assert "nodaemon=true" in supervisor
    assert "command=/bin/bash /run_trame.sh" in supervisor
    assert "command=/bin/bash /run_nginx.sh" in supervisor
    assert "events=PROCESS_STATE_EXITED,PROCESS_STATE_FATAL" in supervisor
    assert "supervisorctl shutdown" in supervisor


def test_nginx_advertises_readiness_only_after_trame_is_ready() -> None:
    trame_script = _read("scripts/run_trame.sh")
    nginx_script = _read("scripts/run_nginx.sh")
    nginx_config = _read("dockerfiles/nginx.conf.template")

    assert "exec python -m radar_pd_nova" in trame_script
    assert "--server" in trame_script
    assert "until wget -q -O /dev/null http://127.0.0.1:8080/" in nginx_script
    assert "envsubst '${EP_PATH}'" in nginx_script
    assert "exec nginx" in nginx_script
    assert "error_page 502 503 504 =200" not in nginx_config
    assert "RADAR-PD is starting" not in nginx_config
    assert "location ${EP_PATH}/ws" in nginx_config
    assert "alias /app/www-content/" in nginx_config
    assert "try_files $uri $uri/ @proxy" in nginx_config
    assert "location @proxy" in nginx_config
    assert "proxy_pass http://127.0.0.1:8080" in nginx_config


def test_galaxy_tool_matches_the_official_interactive_contract() -> None:
    root = ET.parse(NOVA_ROOT / "galaxy" / "radar_pd_nova.xml").getroot()
    xml = _read("galaxy/radar_pd_nova.xml")

    assert root.get("tool_type") == "interactive"
    assert root.get("profile") == "22.01"
    entry_point = root.find("./entry_points/entry_point")
    assert entry_point is not None
    assert entry_point.get("label") == "app_entry"
    assert entry_point.get("requires_path_in_url") == "True"
    assert entry_point.findtext("port") == "8081"
    ep_path = root.find("./environment_variables/environment_variable[@name='EP_PATH']")
    assert ep_path is not None
    assert ep_path.get("inject") == "entry_point_path_for_label"
    assert ep_path.text == "app_entry"
    command = root.findtext("command", default="")
    assert "supervisord -c /etc/supervisord.conf" in command
    assert "/usr/local/bin/run_container.sh" not in command
    output = root.find("./outputs/data")
    assert output is not None
    assert output.get("name") == "output"
    assert output.get("hidden") is None
    assert "detect_errors=" not in xml


def test_package_and_galaxy_tool_versions_match() -> None:
    pyproject = _read("pyproject.toml")
    version_match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    assert version_match is not None
    tool_version = ET.parse(NOVA_ROOT / "galaxy" / "radar_pd_nova.xml").getroot().get(
        "version"
    )

    assert version_match.group(1) == tool_version
