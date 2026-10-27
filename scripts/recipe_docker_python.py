#!/usr/bin/env python3
"""Local scientific-Python shim using the existing RADAR-PD Linux image.

Pass this executable to the review harness's --python. It needs Docker and a
readable local image, not Diffraction Studio or its Python environment.
"""
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    work = os.environ.get("RADAR_GPX_DOCKER_WORKDIR")
    image = os.environ.get("RADAR_GPX_DOCKER_IMAGE")
    if not work or not image:
        raise SystemExit("Set RADAR_GPX_DOCKER_WORKDIR and RADAR_GPX_DOCKER_IMAGE for local validation")
    work = Path(work).resolve(strict=True)
    if not work.is_dir() or work == Path("/"):
        raise SystemExit("Use a specific existing validation directory")
    command = ["docker", "run", "--rm", "--pull", "never", "--platform", "linux/amd64",
               "--user", f"{os.getuid()}:{os.getgid()}", "--entrypoint", "python",
               "-e", "HOME=/tmp", "-e", "MPLBACKEND=Agg",
               "-e", "OPENBLAS_NUM_THREADS=1", "-e", "OMP_NUM_THREADS=1",
               "-v", f"{root / 'scripts'}:/opt/radar-pd/scripts:ro",
               "-v", f"{root / 'nova/src/gpx_recipe'}:/opt/radar-pd/nova/src/gpx_recipe:ro",
               "-v", f"{work}:{work}:rw", image, *sys.argv[1:]]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
