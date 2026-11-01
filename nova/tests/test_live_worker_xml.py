"""Render real Cheetah, normalize as Galaxy does, and execute both cache forms."""
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import pytest
from Cheetah.Template import Template


@pytest.mark.parametrize('cache',[None,'prior cache.zip'])
@pytest.mark.parametrize('worker_exit',[0,2])
def test_live_tool_command_preserves_arguments_and_redirection(tmp_path,cache,worker_exit):
    root=ET.parse(Path(__file__).resolve().parents[2]/'galaxy/radar_pd_live_snapshot.xml.in').getroot()
    values={'result_bundle':'result bundle.zip','response_cache_output':'cache output.zip',
            'console_output':'console.txt','packet':'input packet.zip','response_cache':cache}
    rendered=str(Template(root.find('command').text,searchList=[values]))
    # Galaxy's ToolEvaluator._build_command_line strips each rendered line,
    # then joins newlines with spaces. A shell continuation becomes a literal
    # space argument under that real deployment normalization.
    # https://docs.galaxyproject.org/en/release_25.1/_modules/galaxy/tools/evaluation.html
    command=' '.join(line.strip() for line in rendered.split('\n')).replace('\r',' ').strip()
    binary=tmp_path/'bin';binary.mkdir()
    recorder=binary/'python'
    recorder.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$RECORDED_ARGS"\nprintf "worker-console\\n"\nexit "$WORKER_EXIT"\n')
    recorder.chmod(0o755)
    output=tmp_path/'argv.txt'
    completed=subprocess.run(['/bin/sh','-ec',command],cwd=tmp_path,capture_output=True,text=True,
                   env={'PATH':str(binary)+':/usr/bin:/bin','RECORDED_ARGS':str(output),'WORKER_EXIT':str(worker_exit)})
    expected=['-m','gpx_recipe.live.galaxy_worker','input packet.zip','worker','result bundle.zip','cache output.zip']
    if cache:expected+=['--cache','prior cache.zip']
    assert output.read_text().splitlines()==expected
    assert completed.returncode==worker_exit
    assert (tmp_path/'console.txt').read_text()=='worker-console\n'
    assert not completed.stdout
    assert root.find("outputs/collection[@name='verified_projects']/discover_datasets").get('directory')=='worker/verified_projects'
