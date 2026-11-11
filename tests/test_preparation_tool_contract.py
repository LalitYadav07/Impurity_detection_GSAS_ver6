"""Rendered launch commands preserve preparation provenance and native inputs."""
import base64
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest
from Cheetah.Template import Template

from gpx_recipe.preparation import authority, certificates

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('preparation_tool_renderer',
    ROOT/'scripts/render_radar_pd_preparation_tools.py')
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


@pytest.fixture
def tools(tmp_path):
    paths = renderer.render('ghcr.io/neutrons/radar-pd:models-'+'a'*40, tmp_path/'tools')
    return {path.stem.removeprefix('radar_pd_'): ET.parse(path).getroot() for path in paths}


def test_canonical_tools_preserve_the_reviewed_producer_and_transport_contract(tools):
    model = tools['model_campaign']
    live = tools['live_snapshot']
    assert model.attrib['version'] in authority.VERSIONS
    assert model.find("outputs/data[@name='preparation_certificate']") is not None
    assert model.find("outputs/collection[@name='verified_projects']") is not None
    assert model.find("outputs/collection[@name='reviewable_projects']") is not None
    for tool in tools.values():
        evidence = tool.find("inputs/param[@name='preparation_evidence']")
        assert evidence.attrib['multiple'] == evidence.attrib['optional'] == 'true'
        assert evidence.attrib['format'] == 'json'
        assert '--preparation-evidence' in tool.findtext('command')
    assert model.find("inputs/param[@name='model_asset_parts']").attrib['multiple'] == 'true'
    assert model.findtext("configfiles/configfile[@name='preparation_provenance']") == live.findtext(
        "configfiles/configfile[@name='preparation_provenance']")


@pytest.mark.parametrize('with_evidence', [False, True])
def test_provenance_comes_from_creating_job_and_omits_unrelated_stdout(tools, with_evidence):
    class Dataset:
        unsanitized = SimpleNamespace(creating_job=SimpleNamespace(
            tool_id='neutrons_radar_pd_model_campaign_prototype', tool_version='0.3.4',
            state='ok', stdout='unrelated private log line\n'+certificates.MARKER+'b'*64+'\n'),
            state='ok', deleted=False, purged=False)

        def __str__(self):
            return '/staged/native-origin.json'

    source = tools['model_campaign'].findtext("configfiles/configfile[@name='preparation_provenance']")
    value = str(Template(source, searchList=[dict(preparation_evidence=[Dataset()] if with_evidence else [])]))
    result = json.loads(value)
    assert 'unrelated private log line' not in value
    assert result['schema'] == authority.SCHEMA
    assert len(result['records']) == int(with_evidence)
    if with_evidence:
        row = result['records'][0]
        assert row['job_stdout'] == certificates.MARKER+'b'*64
        assert row['tool_version'] == '0.3.4' and row['certificate_path'] == '/staged/native-origin.json'


def run_command(command, values, tmp_path, *, folded=False):
    text = str(Template(command, searchList=[values]))
    text = re.sub(r'(?m)^(\s*)python\s', lambda match: match[1]+shlex.quote(sys.executable)+' ', text)
    if folded:
        # Galaxy normalizes command whitespace before dispatch. A shell
        # backslash here can escape a space instead of a newline.
        text = ' '.join(line.strip() for line in text.splitlines())
    # Child commands use the package under test, including in built images.
    env = dict(os.environ, PYTHONPATH=str(Path(certificates.__file__).resolve().parents[2]))
    return subprocess.run(['/bin/sh', '-c', text], cwd=tmp_path, env=env,
                          text=True, capture_output=True, check=True)


@pytest.mark.parametrize('action', ['native', 'admission', 'chunk'])
@pytest.mark.parametrize('source', ['inline', 'dataset'])
@pytest.mark.parametrize('multipart', [False, True])
@pytest.mark.parametrize('folded', [False, True])
def test_real_model_command_preserves_request_parts_and_publishes_certificate(tools, tmp_path, action, source, multipart, folded):
    tool = tools['model_campaign']
    script = tmp_path/'installed_worker.py'
    script.write_text("""import argparse,json,pathlib
from gpx_recipe.preparation.certificates import SCHEMA
from gpx_recipe.symmetry.compiler import package_signature
p=argparse.ArgumentParser()
for name in ('action','source','assets','request','directory','archive'):p.add_argument(name)
p.add_argument('--asset-part',action='append',default=[])
p.add_argument('--preparation-evidence',required=True)
p.add_argument('--preparation-certificate',required=True)
a=p.parse_args();result=vars(a);result['request_content']=json.loads(pathlib.Path(a.request).read_text())
pathlib.Path(a.archive).write_text(json.dumps(result))
pathlib.Path(a.preparation_certificate).write_text(json.dumps(dict(schema=SCHEMA,
implementation_sha256=package_signature(),result_archive_sha256='a'*64,entries=[])))
""")
    decoder = tmp_path/'decode.py'
    decoder.write_text(tool.findtext("configfiles/configfile[@name='request_decoder']"))
    requested = dict(operation='reviewed operation', version=3)
    original = tmp_path/'request.json'
    original.write_text(json.dumps(requested))
    inline = tmp_path/'encoded.txt'
    inline.write_text(base64.b64encode(json.dumps(requested).encode()).decode())
    archive, certificate = tmp_path/'result.json', tmp_path/'certificate.json'
    parts = ['first.zip', 'second.zip'] if multipart else []
    values = dict(action=action, source_gpx='source.gpx', model_assets='inputs.zip',
        model_request=str(original) if source == 'dataset' else None,
        model_request_b64=inline.read_text() if source == 'inline' else '',
        inline_request=str(inline), request_decoder=str(decoder), model_asset_parts=parts,
        preparation_provenance='server-context.json', preparation_certificate=str(certificate),
        result_bundle=str(archive), console_output=str(tmp_path/'console.txt'))
    command = tool.findtext('command').replace('/opt/radar-symmetry/ndip_model_campaign.py', shlex.quote(str(script)))
    completed = run_command(command, values, tmp_path, folded=folded)
    result = json.loads(archive.read_text())
    assert result['action'] == action and result['source'] == 'source.gpx'
    assert result['asset_part'] == parts and result['request_content'] == requested
    assert result['preparation_evidence'] == 'server-context.json'
    assert completed.stdout.strip() == certificates.MARKER+__import__('hashlib').sha256(certificate.read_bytes()).hexdigest()


@pytest.mark.parametrize('with_cache', [False, True])
def test_live_command_preserves_the_cached_response_and_provenance(tools, tmp_path, with_cache):
    worker = tmp_path/'installed_live_worker.py'
    worker.write_text("""import argparse,json,pathlib
p=argparse.ArgumentParser()
for name in ('packet','directory','result','cache_output'):p.add_argument(name)
p.add_argument('--cache');p.add_argument('--preparation-evidence',required=True)
a=p.parse_args();pathlib.Path(a.result).write_text(json.dumps(vars(a)))
pathlib.Path(a.cache_output).write_text('retained response')
""")
    result, cache_output = tmp_path/'live-result.json', tmp_path/'cache.zip'
    command = tools['live_snapshot'].findtext('command').replace('-m gpx_recipe.live.galaxy_worker', shlex.quote(str(worker)))
    run_command(command, dict(packet='scan.zip', response_cache='previous.zip' if with_cache else None,
        result_bundle=str(result), response_cache_output=str(cache_output),
        preparation_provenance='server-context.json', console_output=str(tmp_path/'console.txt')), tmp_path)
    actual = json.loads(result.read_text())
    assert actual['packet'] == 'scan.zip'
    assert actual['cache'] == ('previous.zip' if with_cache else None)
    assert actual['preparation_evidence'] == 'server-context.json'
    assert cache_output.read_text() == 'retained response'


@pytest.mark.parametrize('image', ['ghcr.io/neutrons/radar-pd:latest', 'models-'+40*'a',
    'ghcr.io/neutrons/radar-pd:models-'+40*'a'+"'; false"])
def test_renderer_rejects_unpinned_or_invalid_images(tmp_path, image):
    with pytest.raises(ValueError, match='full-commit'):
        renderer.render(image, tmp_path/'output')
    assert not (tmp_path/'output').exists()
