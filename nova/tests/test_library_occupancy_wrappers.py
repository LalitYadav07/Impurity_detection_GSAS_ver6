from pathlib import Path
import shlex
from xml.etree import ElementTree as ET

from Cheetah.Template import Template


def test_builder_template_exposes_strict_default_and_quotes_review_note():
    source=Path(__file__).resolve().parents[2]/'galaxy/radar_pd_library_builder.xml.in'
    tree=ET.parse(source)
    policy=tree.find("./inputs/param[@name='reviewed_occupancy_excess']")
    assert policy.find("option[@selected='true']").attrib['value']=='0'
    note="Published O'Brien model; $(touch never_execute)"
    variables=dict(library_archive='archive.zip',library_manifest='manifest.json',console_output='console.txt',
        cif_archive='input.zip',library_mode='mini',radiation='neutron',reviewed_occupancy_excess='0.001',occupancy_review_note=note)
    rendered=str(Template(tree.find('command').text,searchList=[variables]))
    assert '--reviewed-occupancy-excess' in rendered
    assert '--occupancy-review-note '+shlex.quote(note) in rendered
    assert rendered.index('cif_source_archive.py')<rendered.index('python -m zipfile -e')
    assert '<container' not in rendered


def test_analyze_patch_has_explicit_file_argument_and_no_manifest_auto_consent():
    source=Path(__file__).resolve().parents[2]/'galaxy/radar_pd_analyze_occupancy.patch'
    text=source.read_text()
    assert '--library-review-activation' in text
    assert 'library_review_payload' in text
    assert 'Archive reviewed:true is never permission' in text
