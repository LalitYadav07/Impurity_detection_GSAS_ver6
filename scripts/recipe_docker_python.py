#!/usr/bin/env python3
"""Local scientific-Python shim using the existing RADAR-PD Linux image.

Pass this executable to the review harness's --python. It needs Docker and a
readable local image, not Diffraction Studio or its Python environment.
"""
import os
from pathlib import Path
import subprocess
import sys
import signal
import uuid


def main():
    root = Path(__file__).resolve().parents[1]
    work = os.environ.get("RADAR_GPX_DOCKER_WORKDIR")
    image = os.environ.get("RADAR_GPX_DOCKER_IMAGE")
    if not work or not image:
        raise SystemExit("Set RADAR_GPX_DOCKER_WORKDIR and RADAR_GPX_DOCKER_IMAGE for local validation")
    work = Path(work).resolve(strict=True)
    if not work.is_dir() or work == Path("/"):
        raise SystemExit("Use a specific existing validation directory")
    name='radar-gpx-'+uuid.uuid4().hex
    command = ["docker", "run", "--name",name,"--label","radar-pd.recipe-validation=true","--rm", "--pull", "never", "--platform", "linux/amd64", "--network", "none",
               "--user", f"{os.getuid()}:{os.getgid()}", "--entrypoint", "python",
               "-e", "HOME=/tmp", "-e", "MPLBACKEND=Agg",
               "-e", "OPENBLAS_NUM_THREADS=1", "-e", "OMP_NUM_THREADS=1",
               "-v", f"{root / 'scripts'}:/opt/radar-pd/scripts:ro",
               "-v", f"{root / 'nova/src/gpx_recipe'}:/opt/radar-pd/nova/src/gpx_recipe:ro",
               "-v", f"{root / 'nova/src/gpx_recipe'}:/opt/radar-symmetry/gpx_recipe:ro",
               "-v", f"{work}:{work}:rw", image, *sys.argv[1:]]
    def stop(signum,frame):
        try:subprocess.run(['docker','stop','--time','5',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
        except (OSError,subprocess.TimeoutExpired):pass
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:raise SystemExit(subprocess.call(command))
    finally:
        try:subprocess.run(['docker','rm','--force',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
        except (OSError,subprocess.TimeoutExpired):pass


if __name__ == "__main__":
    main()
