import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('ui_runtime_check',
    Path(__file__).resolve().parents[1] / 'gsasii_gui/tools/check_ui_runtime.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def inputs():
    result = {path: b'qualified runtime input\n' for path in module.RUNTIME_INPUTS}
    result['gsasii_gui/Dockerfile'] = b'FROM qualified-base\nARG RADAR_PD_GSASII_GUI_VERSION=0.1.16\nRUN native-build\n'
    return result


def test_ui_version_change_keeps_the_qualified_runtime():
    base = inputs()
    current = {**base, 'gsasii_gui/Dockerfile': base['gsasii_gui/Dockerfile'].replace(b'0.1.16', b'0.1.17')}
    module.verify(base, current)


@pytest.mark.parametrize('path', module.RUNTIME_INPUTS)
def test_every_changed_runtime_input_requires_a_full_build(path):
    base = inputs()
    with pytest.raises(ValueError, match='complete runtime'):
        module.verify(base, {**base, path: base[path] + b'changed scientific input\n'})


def test_ui_image_requires_the_exact_runtime_and_current_release_version():
    source = b'ARG RADAR_PD_GSASII_GUI_VERSION=0.1.17\n'
    image = ('FROM ghcr.io/lalityadav07/impurity_detection_gsas_ver6:gsasii-gui-'
             + module.BASE_COMMIT + '@' + module.BASE_DIGEST + '\nENV GSASII_GUI_VERSION=0.1.17\n').encode()
    module.verify_ui_image(source, image)
    with pytest.raises(ValueError, match='version'):
        module.verify_ui_image(source, image.replace(b'0.1.17', b'0.1.16'))
    with pytest.raises(ValueError, match='exact qualified runtime'):
        module.verify_ui_image(source, image.replace(module.BASE_DIGEST.encode(), b'latest'))
