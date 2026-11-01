"""Execute both optional-cache command forms with a recording worker stub."""
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import pytest


@pytest.mark.parametrize('cache',[False,True])
def test_live_tool_command_preserves_arguments_and_redirection(tmp_path,cache):
    root=ET.parse(Path(__file__).resolve().parents[2]/'galaxy/radar_pd_live_snapshot.xml.in').getroot()
    command=root.find('command').text
    lines=[];inside=False
    for line in command.splitlines():
        if line.strip().startswith('#if '):inside=True;continue
        if line.strip()=='#end if':inside=False;continue
        if not inside or cache:lines.append(line)
    command='\n'.join(lines)
    values={'result_bundle':'result bundle.zip','response_cache_output':'cache output.zip',
            'console_output':'console.txt','packet':'input packet.zip','response_cache':'prior cache.zip'}
    for key in sorted(values,key=len,reverse=True):command=command.replace('$'+key,values[key])
    binary=tmp_path/'bin';binary.mkdir()
    recorder=binary/'python'
    recorder.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$RECORDED_ARGS"\n')
    recorder.chmod(0o755)
    output=tmp_path/'argv.txt'
    subprocess.run(['/bin/sh','-ec',command],cwd=tmp_path,check=True,
                   env={'PATH':str(binary)+':/usr/bin:/bin','RECORDED_ARGS':str(output)})
    expected=['-m','gpx_recipe.live.galaxy_worker','input packet.zip','worker','result bundle.zip','cache output.zip']
    if cache:expected+=['--cache','prior cache.zip']
    assert output.read_text().splitlines()==expected
    assert (tmp_path/'console.txt').is_file()
    assert root.find("outputs/collection[@name='verified_projects']/discover_datasets").get('directory')=='worker/verified_projects'
