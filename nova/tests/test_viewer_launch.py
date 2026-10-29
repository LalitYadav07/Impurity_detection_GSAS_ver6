import json
import pytest
from radar_pd_nova.launch import credential_context


def test_launch_captures_only_selected_vault_references():
    rows=[dict(id='record',name='ornl_illumine',version='1.0',current_group_id='chosen',groups=[
        dict(id='other',name='Unused',secrets=[dict(name='api_key',is_set=True)]),
        dict(id='chosen',name='Personal',secrets=[dict(name='api_key',is_set=True)])])]
    assert credential_context(rows)==[dict(user_credentials_id='record',name='ornl_illumine',version='1.0',
        selected_group=dict(id='chosen',name='Personal'))]
    assert 'secrets' not in json.dumps(credential_context(rows))
    rows[0]['groups'][1]['secrets'][0]['is_set']=False
    with pytest.raises(ValueError,match='this tool version'):credential_context(rows)
    with pytest.raises(ValueError):credential_context([])
