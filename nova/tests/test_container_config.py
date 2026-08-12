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


def test_container_supervises_both_web_services() -> None:
    launcher = _read("scripts/run_container.sh")
    dockerfile = _read("dockerfiles/Dockerfile")
    service_manager = _read("scripts/supervise_services.py")
    galaxy_xml = _read("galaxy/radar_pd_nova.xml")

    assert "wait -n" not in launcher
    assert "exec python /usr/local/bin/supervise_services.py" in launcher
    assert "scripts/supervise_services.py" in dockerfile
    assert 'Service("trame", ("/usr/local/bin/run_trame.sh",))' in service_manager
    assert 'Service("nginx", ("/usr/local/bin/run_nginx.sh",))' in service_manager
    assert "service.restarts += 1" in service_manager
    assert "start_new_session=True" in service_manager
    assert "os.killpg" in service_manager
    assert 'CMD ["/usr/local/bin/run_container.sh"]' in dockerfile
    assert "/usr/local/bin/run_container.sh" in galaxy_xml
    assert "/usr/local/bin/run_trame.sh" not in galaxy_xml
    assert "/usr/local/bin/run_nginx.sh" not in galaxy_xml


def test_nova_prefix_redirects_to_the_canonical_trailing_slash() -> None:
    nginx_script = _read("scripts/run_nginx.sh")
    nginx_config = _read("dockerfiles/nginx.conf.template")

    assert "ep_path=\"${ep_path%/}\"" in nginx_script
    assert "^(/[A-Za-z0-9._~-]+)+$" in nginx_script
    assert "absolute_redirect off;" in nginx_config
    assert "location = ${EP_PATH} {" in nginx_config
    assert "return 308 ${EP_PATH}/$is_args$args;" in nginx_config
    assert "location ^~ ${EP_PATH}/ {" in nginx_config


def test_nginx_answers_readiness_while_trame_is_starting() -> None:
    nginx_config = _read("dockerfiles/nginx.conf.template")

    assert nginx_config.count("proxy_intercept_errors on;") == 2
    assert nginx_config.count("error_page 502 503 504 =200 @radar_pd_starting;") == 2
    assert "location @radar_pd_starting {" in nginx_config
    assert "RADAR-PD is starting" in nginx_config
    assert 'add_header Retry-After "2" always;' in nginx_config


def test_package_and_galaxy_tool_versions_match() -> None:
    pyproject = _read("pyproject.toml")
    version_match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    assert version_match is not None
    package_version = version_match.group(1)
    tool_version = ET.parse(NOVA_ROOT / "galaxy" / "radar_pd_nova.xml").getroot().get(
        "version"
    )

    assert package_version == tool_version


def test_interactive_startup_writes_to_galaxy_managed_output() -> None:
    galaxy_xml = _read("galaxy/radar_pd_nova.xml")
    launcher = _read("scripts/run_container.sh")
    root = ET.fromstring(galaxy_xml)
    outputs = root.find("outputs")

    assert outputs is not None
    declared_outputs = list(outputs)
    assert len(declared_outputs) == 1
    assert declared_outputs[0].get("name") == "output"
    assert declared_outputs[0].get("format") == "txt"
    assert declared_outputs[0].get("hidden") == "true"
    command = root.findtext("command", default="")
    assert "Galaxy command starting" in command
    assert "/usr/local/bin/run_container.sh" in command
    assert "exec /usr/local/bin/run_container.sh" not in command
    assert "/usr/local/bin/run_trame.sh" not in command
    assert "/usr/local/bin/run_nginx.sh" not in command
    assert "$session_log" not in command
    assert "> $output" in command
    assert "tee -a $output" in command
    assert "$console_output" not in command
    assert "touch " not in command
    assert "launcher starting" in launcher
