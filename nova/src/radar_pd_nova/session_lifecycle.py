"""Release this supervised interactive container after its recovery checkpoint.

The NOVA app is a child of the web launcher. Exiting that child alone leaves
the launcher and nginx alive. A detached, credential-free helper shuts down
the container's own supervisor after the browser has received its redirect.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

CONFIG = Path('/etc/supervisord.conf')
SOCKET = Path('/tmp/supervisor.sock')
MARKER = 'RADAR_PD_SUPERVISED_SESSION'


def schedule_supervised_shutdown():
    if os.environ.get(MARKER) != '1':
        return False
    command = shutil.which('supervisorctl')
    if not command or not CONFIG.is_file() or not SOCKET.is_socket():
        raise RuntimeError('Container supervisor control is unavailable; this session remains open.')
    subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), command],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        close_fds=True, start_new_session=True, env={MARKER: '1'},
    )
    return True


def main():
    if os.environ.get(MARKER) != '1' or len(sys.argv) != 2:
        raise SystemExit('Only a marked supervised session can request shutdown')
    command = Path(sys.argv[1])
    if not command.is_absolute() or command.name != 'supervisorctl' or not command.is_file():
        raise SystemExit('Supervisor control executable is unavailable')
    if not CONFIG.is_file() or not SOCKET.is_socket():
        raise SystemExit('This process has no session supervisor')
    time.sleep(.5)
    subprocess.run([str(command), '-c', str(CONFIG), 'shutdown'], check=True, timeout=10,
                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == '__main__':
    main()
