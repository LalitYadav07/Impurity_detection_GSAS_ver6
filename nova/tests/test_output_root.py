import tempfile
from pathlib import Path

from radar_pd_nova.galaxy_service import GalaxyService


def test_default_output_root_is_scoped_to_runtime_user_and_process() -> None:
    service = GalaxyService("https://example.invalid", "key", "history")

    assert service.output_root.parent == Path(tempfile.gettempdir())
    assert service.output_root.name.startswith("radar-pd-nova-")
    assert service.output_root.is_dir()
