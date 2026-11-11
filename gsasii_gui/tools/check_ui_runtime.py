"""Refuse reuse of the qualified image when any native-runtime input changed."""
from pathlib import Path
import re
import subprocess

BASE_COMMIT = 'f3aba94bcca30a1b7e21c36a5a65fc83dc79e217'
BASE_DIGEST = 'sha256:ea610a430850f4addf385c8f55105b7d297e299b4c14e88672e127a6565a67c0'
RUNTIME_INPUTS = (
    'gsasii_gui/Dockerfile',
    'gsasii_gui/environment.yml',
    'nova/src/gpx_recipe/symmetry/kernel_patch.py',
    'nova/src/gpx_recipe/symmetry/space_group_patch.py',
    'tests/gpx_recipe/kernel_geometry_control.py',
    'tests/gpx_recipe/nuclear_powder_control.py',
)


def normalized(path, content):
    if path == 'gsasii_gui/Dockerfile':
        # The image's user-visible version is set by the UI layer. Every
        # other Docker instruction, pinned revision and runtime setting stays
        # byte-identical to the independently qualified base recipe.
        content = re.sub(rb'^ARG RADAR_PD_GSASII_GUI_VERSION=[^\r\n]+$',
                         b'ARG RADAR_PD_GSASII_GUI_VERSION=<UI VERSION>', content, flags=re.M)
    return content


def verify(base, current):
    changed = [path for path in RUNTIME_INPUTS
               if normalized(path, base[path]) != normalized(path, current[path])]
    if changed:
        raise ValueError('Build the complete runtime; qualified-runtime reuse is not allowed after changes to: '
                         + ', '.join(changed))


def verify_ui_image(dockerfile, ui_file):
    version = re.search(rb'^ARG RADAR_PD_GSASII_GUI_VERSION=([^\r\n]+)$', dockerfile, re.M)
    ui_version = re.search(rb'^ENV GSASII_GUI_VERSION=([^\r\n]+)$', ui_file, re.M)
    if not version or not ui_version or version[1] != ui_version[1]:
        raise ValueError('The UI image version must match the source Dockerfile')
    image = ('ghcr.io/lalityadav07/impurity_detection_gsas_ver6:gsasii-gui-'
             + BASE_COMMIT + '@' + BASE_DIGEST).encode()
    declared = re.search(rb'^FROM (\S+)$', ui_file, re.M)
    if not declared or declared[1] != image:
        raise ValueError('The UI image must use the exact qualified runtime digest')


def main():
    root = Path(__file__).resolve().parents[2]
    base = {path: subprocess.run(['git', 'show', BASE_COMMIT + ':' + path], cwd=root,
                                check=True, capture_output=True).stdout for path in RUNTIME_INPUTS}
    current = {path: (root / path).read_bytes() for path in RUNTIME_INPUTS}
    verify(base, current)
    verify_ui_image(current['gsasii_gui/Dockerfile'], (root / 'gsasii_gui/Dockerfile.ui-update').read_bytes())
    print('Qualified desktop native-runtime inputs are unchanged:', BASE_COMMIT)


if __name__ == '__main__':
    main()
