"""Scientific review notices must survive batch collection and offline export."""
import json
from pathlib import Path
import pytest

from scripts.ndip_input_provenance import read_input_provenance, PREFIX
from scripts.ndip_outputs import collect_outputs


def test_input_quality_notices_survive_normalized_results_and_report(tmp_path):
    source=tmp_path/'bank5.gsa'
    metadata={'schema':'radar-pd-input-provenance/v1','instrument':'NOM','source_bank':5,
              'warnings':[{'code':'unverified_nomad_profile','message':'NOMAD calibration acceptance is pending'},
                          {'code':'nomad_reduction_quality','message':'Placeholder sample corrections'}]}
    source.write_bytes(PREFIX+json.dumps(metadata).encode()+b'\nBANK 1 1 1 SLOG 1 2 .1 0 FXYE\n1 2 3\n')
    run=tmp_path/'run';run.mkdir()
    (run/'input_provenance.json').write_text(json.dumps(read_input_provenance(source)))
    result=collect_outputs(run,tmp_path/'portal',mode='full',include_archive=False)
    assert result['scientific_status']=='needs_review'
    assert result['provenance']['input']['source_bank']==5
    assert {w['code'] for w in result['warnings']}=={'unverified_nomad_profile','nomad_reduction_quality'}
    assert 'Placeholder sample corrections' in (tmp_path/'portal/report.html').read_text()


@pytest.mark.parametrize('payload',[{'schema':'unknown'}, {'schema':'radar-pd-input-provenance/v1','warnings':'verified'}])
def test_invalid_input_notice_contract_fails_explicitly(tmp_path,payload):
    p=tmp_path/'data.gsa';p.write_bytes(PREFIX+json.dumps(payload).encode()+b'\n')
    with pytest.raises(ValueError):read_input_provenance(p)


def test_unmarked_pattern_keeps_existing_behavior_and_marker_is_bounded(tmp_path):
    p=tmp_path/'pattern.dat';p.write_bytes(b'1 2 3\n'*100)
    assert read_input_provenance(p)=={}
    p.write_bytes(b'0'*9000)
    assert read_input_provenance(p)=={}
    p.write_bytes(PREFIX+b'x'*9000)
    with pytest.raises(ValueError,match='size'):read_input_provenance(p)
