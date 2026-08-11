from pathlib import Path


NOVA_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_files_are_writable_by_ndip_job_user() -> None:
    nginx_script = (NOVA_ROOT / "scripts" / "run_nginx.sh").read_text(encoding="utf-8")
    nginx_config = (NOVA_ROOT / "dockerfiles" / "nginx.conf.template").read_text(
        encoding="utf-8"
    )
    dockerfile = (NOVA_ROOT / "dockerfiles" / "Dockerfile").read_text(encoding="utf-8")
    galaxy_xml = (NOVA_ROOT / "galaxy" / "radar_pd_nova.xml").read_text(encoding="utf-8")

    assert "/tmp/radar-pd-nginx.conf" in nginx_script
    assert "client_body_temp_path /tmp/radar-pd-nginx/" in nginx_config
    assert "proxy_temp_path /tmp/radar-pd-nginx/" in nginx_config
    assert "fastcgi_temp_path /tmp/radar-pd-nginx/" in nginx_config
    assert "uwsgi_temp_path /tmp/radar-pd-nginx/" in nginx_config
    assert "scgi_temp_path /tmp/radar-pd-nginx/" in nginx_config
    supervisor_config = "/etc/supervisor/conf.d/radar-pd.conf"
    assert supervisor_config in dockerfile
    assert supervisor_config in galaxy_xml
