import json
import pytest
from radar_pd_nova.launch import credential_context


def test_two_selected_credentials_pass_references_without_secrets():
    rows=[dict(id=name+'-record',name=name,version='1.0',current_group_id=name+'-group',
        groups=[dict(id=name+'-group',name='Personal '+name,secrets=[dict(name='api_key',is_set=True)])])
        for name in ['ornl_illumine','bilbao']]
    context=credential_context(rows,('ornl_illumine','bilbao'))
    assert [r['name'] for r in context]==['ornl_illumine','bilbao']
    assert 'secrets' not in json.dumps(context) and 'is_set' not in json.dumps(context)
    rows[1]['groups'][0]['secrets'][0]['is_set']=False
    with pytest.raises(ValueError,match='Bilbao'):credential_context(rows,('ornl_illumine','bilbao'))


def test_ambiguous_service_record_cannot_select_an_arbitrary_key():
    row=dict(id='record',name='bilbao',version='1.0',current_group_id='group',groups=[dict(id='group',name='Personal',secrets=[dict(name='api_key',is_set=True)])])
    with pytest.raises(ValueError):credential_context([row,row],('bilbao',))


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
