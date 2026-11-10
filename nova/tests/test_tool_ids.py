import json
import os
import re
from pathlib import Path
import subprocess
import sys

import pytest

from radar_pd_nova.tool_ids import tool_id,deployment


def test_production_ids_and_legacy_default():
    assert tool_id('model_campaign',environ={})=='neutrons_radar_pd_model_campaign_prototype'
    env={'RADAR_PD_DEPLOYMENT':'production'}
    assert tool_id('model_campaign',environ=env)=='neutrons_radar_pd_model_campaign'
    assert tool_id('interactive',environ=env)=='nova-neutrons_radar_pd_interactive'
    assert tool_id('gsasii_interactive',environ=env)=='nova-neutrons_radar_pd_gsasii_interactive'
    assert tool_id('result_explorer',environ=env)=='nova-neutrons_radar_pd_result_explorer'
    assert tool_id('analyze',environ={'RADAR_PD_ANALYZE_TOOL_ID':'local-analyze'})=='local-analyze'
    with pytest.raises(ValueError):deployment({'RADAR_PD_DEPLOYMENT':'unknown'})
    with pytest.raises(ValueError):tool_id('analyze',environ={'RADAR_PD_ANALYZE_TOOL_ID':'bad/id'})


@pytest.mark.parametrize('mode',['production','prototype'])
def test_every_producer_uses_the_same_deployment_ids(mode):
    code='''
import json
from radar_pd_nova import galaxy_service as s, model_runtime as m, refinement_runtime as r, preparation_limits as p, powgen_watch as w, launch
from radar_pd_nova.webmon.galaxy_runtime import LIVE_TOOL_ID
from radar_pd_nova.tool_ids import is_interactive_tool,is_scientific_tool
values=[s.ANALYZE_TOOL_ID,s.SNS_RESOLVER_TOOL_ID,s.LIBRARY_BUILDER_TOOL_ID,s.GPX_HANDOFF_TOOL_ID,s.COMPARE_SERIES_TOOL_ID,s.RESULT_EXPLORER_TOOL_ID,s.GSASII_INTERACTIVE_TOOL_ID,s.CONFIGURE_TOOL_ID,m.MODEL_TOOL_ID,r.REFINEMENT_TOOL_ID,p.CPU_TRAINING_TOOL_ID,p.CUDA_TRAINING_TOOL_ID,w.ANALYZE_TOOL_ID,LIVE_TOOL_ID,launch.TOOL]
assert s.ANALYZE_TOOL_ID==w.ANALYZE_TOOL_ID
assert m.MODEL_TOOL_ID==p.CPU_TRAINING_TOOL_ID
assert is_interactive_tool(launch.TOOL) and is_interactive_tool(s.GSASII_INTERACTIVE_TOOL_ID)
assert not is_scientific_tool(s.GSASII_INTERACTIVE_TOOL_ID)
assert is_scientific_tool(m.MODEL_TOOL_ID)
print(json.dumps(values))
'''
    env={k:v for k,v in os.environ.items() if not k.startswith('RADAR_PD_')}
    env['RADAR_PD_DEPLOYMENT']=mode
    result=subprocess.run([sys.executable,'-c',code],env=env,text=True,capture_output=True,check=True)
    values=json.loads(result.stdout)
    assert all(('_prototype' in value)==(mode=='prototype') for value in values)


def test_no_hard_coded_deployed_ids_remain_in_producers():
    root=Path(__file__).parents[1]/'src/radar_pd_nova'
    for path in root.rglob('*.py'):
        if 'data' in path.relative_to(root).parts:continue
        assert not re.search(r'neutrons_radar_pd_[a-z_]+_prototype',path.read_text()),str(path)
