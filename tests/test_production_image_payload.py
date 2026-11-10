"""Production payload boundaries; optional Docker checks exercise ignore rules.

The command-line check runs against a built image from a read-only /validation
mount. The Docker-context tests use invented sentinel fixtures, never real data.
"""
from pathlib import Path
import argparse
import os
import shlex
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
KERNEL_CONTROLS={'kernel_geometry_control.py','nuclear_powder_control.py'}
CUDA_CONTROLS={
    'test_structural_training_cuda.py','test_structural_paired_training_native.py',
    'test_structural_paired_blocks.py','test_structural_joint_modes_native.py',
    'test_structural_family_targets.py','test_structural_mode_projection.py',
    'test_structural_paired_preparation.py','test_structural_training_jobs.py',
    'test_structural_ranker_cohort.py',
}
ROOT_CONTROLS={f'gpx_recipe/{name}' for name in KERNEL_CONTROLS|CUDA_CONTROLS}
DOCKERFILES=('Dockerfile.ndip','Dockerfile.symmetry','Dockerfile.preparation-cuda','nova/dockerfiles/Dockerfile')
FORBIDDEN=('/opt/radar-pd/nova/tests','/src/tests','/opt/radar-symmetry/tests')


def cleanup_line(name):
    return next(line for line in (ROOT/name).read_text().splitlines() if line.startswith('RUN rm -rf '))


def assert_installed_payload(kind):
    for name in FORBIDDEN:
        assert not Path(name).exists(),f'Inherited development test tree is installed: {name}'
    tests=Path('/opt/radar-pd/tests')
    found={str(p.relative_to(tests)) for p in tests.rglob('*') if p.is_file()}
    assert found<=ROOT_CONTROLS,f'Unapproved tests or fixtures are installed: {sorted(found-ROOT_CONTROLS)}'
    cuda=Path('/opt/radar-cuda-validation')
    if cuda.exists():
        found={str(p.relative_to(cuda)) for p in cuda.rglob('*') if p.is_file()}
        assert found<=CUDA_CONTROLS,f'Unapproved CUDA fixture payload: {sorted(found-CUDA_CONTROLS)}'
    if kind=='ndip':
        for name in KERNEL_CONTROLS:assert (tests/'gpx_recipe'/name).is_file()
        assert Path('/opt/radar-pd/scripts/ndip_runner.py').is_file()
    elif kind=='models':
        for name in KERNEL_CONTROLS:assert (Path('/opt/radar-symmetry')/name).is_file()
        assert Path('/opt/radar-symmetry/ndip_model_campaign.py').is_file()
    elif kind=='preparation-cuda':
        assert {p.name for p in cuda.glob('*.py')}==CUDA_CONTROLS
        assert Path('/opt/radar-symmetry/ndip_preparation_cuda_acceptance.py').is_file()
    elif kind=='nova':
        assert Path('/src/src/radar_pd_nova/app.py').is_file()
    else:raise ValueError('Unknown image kind')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--image-kind',required=True,choices=['ndip','models','preparation-cuda','nova'])
    assert_installed_payload(parser.parse_args().image_kind)
    print('Verified production payload contains no unapproved test fixtures')
    raise SystemExit(0)


import pytest


def test_root_context_limits_tests_to_explicit_synthetic_controls():
    lines=(ROOT/'.dockerignore').read_text().splitlines()
    assert 'nova/tests/' in lines and 'tests/**' in lines
    included={line.removeprefix('!tests/') for line in lines if line.startswith('!tests/') and not line.endswith('/')}
    assert included==ROOT_CONTROLS
    assert all((ROOT/'tests'/name).is_file() for name in included)
    assert 'tests/' in (ROOT/'nova/.dockerignore').read_text().splitlines()


@pytest.mark.parametrize('name',DOCKERFILES)
def test_each_production_image_removes_inherited_tests_before_copy(name):
    text=(ROOT/name).read_text();cleanup=cleanup_line(name)
    paths=set(shlex.split(cleanup)[3:])
    assert set(FORBIDDEN)|{'/opt/radar-pd/tests'}<=paths
    assert not any(path.startswith('/validation') for path in paths)
    assert text.index(cleanup)<text.index('\nCOPY ')


def test_all_explicit_build_copy_sources_exist_and_cuda_closure_is_retained():
    for name in DOCKERFILES:
        context=ROOT/'nova' if name.startswith('nova/') else ROOT
        for line in (ROOT/name).read_text().splitlines():
            if not line.startswith('COPY '):continue
            words=[w for w in shlex.split(line)[1:] if not w.startswith('--')]
            for source in words[:-1]:assert (context/source).exists(),(name,source)
    cuda=(ROOT/'Dockerfile.preparation-cuda').read_text()
    installed=set()
    for line in cuda.splitlines():
        if line.startswith('COPY tests/'):
            words=shlex.split(line)
            assert words[-1]=='/opt/radar-cuda-validation/'
            installed.update(Path(name).name for name in words[1:-1])
    assert installed==CUDA_CONTROLS
    assert '/opt/radar-cuda-validation' in cleanup_line('Dockerfile.preparation-cuda')
    assert '/opt/radar-cuda-validation/test_structural_training_cuda.py' in (ROOT/'scripts/ndip_preparation_cuda_acceptance.py').read_text()


def test_ci_uses_external_tests_and_a_fresh_ndip_cache():
    workflow=(ROOT/'.github/workflows/publish-ndip-to-ghcr.yaml').read_text()
    assert 'ndip-buildcache-v2' in workflow
    assert 'ref=${{ env.IMAGE_NAME }}:ndip-buildcache\n' not in workflow
    assert 'cd /validation/tests' in workflow
    assert 'unittest tests.test_' not in workflow
    for name in ('publish-ndip-to-ghcr.yaml','publish-models-to-ghcr.yaml','publish-nova-to-ghcr.yaml'):
        text=(ROOT/'.github/workflows'/name).read_text()
        assert 'test_production_image_payload.py' in text
        assert '/validation' in text and ':ro' in text


@pytest.mark.skipif(not os.environ.get('RADAR_IMAGE_PACKAGING_DOCKER_BASE'),reason='Set an existing local base for Docker context checks')
@pytest.mark.parametrize('context_kind',['root','nova'])
def test_actual_docker_context_and_inherited_cleanup(tmp_path,context_kind):
    base=os.environ['RADAR_IMAGE_PACKAGING_DOCKER_BASE']
    platform=subprocess.run(['docker','image','inspect','--format','{{.Os}}/{{.Architecture}}',base],
        check=True,capture_output=True,text=True).stdout.strip()
    context=tmp_path/'context';context.mkdir()
    shutil.copyfile(ROOT/('.dockerignore' if context_kind=='root' else 'nova/.dockerignore'),context/'.dockerignore')
    sentinels=(['nova/tests/data/planning-transitions/magnetic-parent.cif','tests/private-measurement.gpx',
        'tests/gpx_recipe/data/observed.npz'] if context_kind=='root' else ['tests/data/private-measurement.cif'])
    for name in sentinels:
        path=context/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('invented exclusion sentinel\n')
    expected=[]
    if context_kind=='root':
        expected=['tests/'+name for name in sorted(ROOT_CONTROLS)]
        expected+=['nova/src/gpx_recipe/marker.py','nova/src/radar_pd_nova/data/powgen_calibrations/public.instprm']
        cleanups=[cleanup_line(name) for name in DOCKERFILES]
    else:
        expected=['src/gpx_recipe/marker.py','src/radar_pd_nova/data/powgen_calibrations/public.instprm']
        cleanups=[cleanup_line('nova/dockerfiles/Dockerfile')]
    for name in expected:
        path=context/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('public synthetic build control\n')
    inherited=[*FORBIDDEN,'/opt/radar-pd/tests','/opt/radar-cuda-validation']
    setup=' && '.join('mkdir -p '+path+' && touch '+path+'/inherited-sentinel' for path in inherited)
    checks=['test ! -e '+path for path in inherited if context_kind=='root' or path!='/opt/radar-cuda-validation']
    checks+=['test ! -e /payload/'+name for name in sentinels]
    checks+=['test -f /payload/'+name for name in expected]
    dockerfile=tmp_path/'Dockerfile'
    dockerfile.write_text('\n'.join(['FROM '+base,'RUN '+setup,*cleanups,'COPY . /payload','RUN set -ex; '+' && '.join(checks)])+'\n')
    result=subprocess.run(['docker','build','--pull=false','--network=none','--progress=plain',
        '--platform',platform,'--file',str(dockerfile),'--iidfile',str(tmp_path/'image.id'),str(context)],capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
