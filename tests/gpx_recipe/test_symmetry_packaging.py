from pathlib import Path
import shutil
import subprocess
import sys

from gpx_recipe.symmetry.compiler import package_signature,freeze_runtime


def test_packaged_native_adapters_are_exact_source_mirrors():
    root=Path(__file__).resolve().parents[2];package=root/'nova/src/gpx_recipe/symmetry/adapters'
    for name in ['gsas_magnetic_preflight.py','gsas_isodistort_import.py','gsas_cif_import.py']:
        assert (package/name).read_bytes()==(root/'scripts'/name).read_bytes()


def test_compiler_freezes_all_runtime_data_and_licenses_without_loading_native_modules(tmp_path):
    root=Path(__file__).resolve().parents[2]
    script="""from pathlib import Path
import sys
from gpx_recipe.symmetry.compiler import package_signature,freeze_runtime
assert not any(name.startswith('GSASII') for name in sys.modules)
assert 'gsas_magnetic_preflight' not in sys.modules
freeze_runtime(Path(sys.argv[1]));print(package_signature())
"""
    environment={'PATH':str(Path(sys.executable).parent),'PYTHONPATH':str(root/'nova/src')}
    result=subprocess.run([sys.executable,'-c',script,str(tmp_path/'runtime')],cwd=tmp_path,env=environment,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert len(result.stdout.strip())==64
    assert (tmp_path/'runtime/scripts/gsas_magnetic_preflight.py').is_file()
    assert (tmp_path/'runtime/gpx_recipe/symmetry/vendor/sgtables.json').is_file()
    assert (tmp_path/'runtime/gpx_recipe/symmetry/vendor/GSAS-II-LICENSE').is_file()
    assert (tmp_path/'runtime/gpx_recipe/STUDIO-LICENSE').is_file()
