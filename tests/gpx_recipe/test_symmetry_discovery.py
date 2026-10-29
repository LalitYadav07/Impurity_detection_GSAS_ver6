import json
import pytest
from gpx_recipe.symmetry.discovery import magnetic_request,magnetic_catalog,discover_magnetic,operation_text
from gpx_recipe.symmetry.bilbao_parser import parse_magnetic_subgroups_gsas_output

MACHINE="yesz&1z0z0z0zPn'ma'z{{1,0,0},{0,1,0},{0,0,1}}z{0,0,0}z{0}&"
PARENT=dict(setting_sha256='a'*64,operations=[dict(rotation=[[1,0,0],[0,1,0],[0,0,1]],translation=[0,0,0])])


def test_discovery_uses_exact_operations_and_propagation_basis():
    request=magnetic_request(PARENT,['0','0','1/2'])
    assert request['inputs']['generators']==['x,y,z']
    assert request['inputs']['km1z']=='1/2'
    assert request['inputs']['sub']=='1.1'  # BNS number, not nuclear SG 1 or symbol P1
    assert 'super' not in request['inputs']


def test_changing_origin_changes_the_sent_generator():
    op=dict(rotation=[[-1,0,0],[0,-1,0],[0,0,-1]],translation=[.5,0,0])
    assert operation_text(op)=='-x+1/2,-y,-z'


def test_centered_lattice_k_vector_is_not_reduced_modulo_integer_components():
    assert magnetic_request(PARENT,['1','0','0'])['inputs']['km1x']=='1'


def test_string_false_cannot_silently_change_the_star_policy():
    with pytest.raises(ValueError):magnetic_request(PARENT,['0','0','0'],whole_star='false')


def test_machine_catalog_retains_the_embedding_and_bindings():
    catalog=magnetic_catalog(MACHINE,PARENT,['0','0','0'])
    assert catalog['parent_setting_sha256']=='a'*64
    assert catalog['candidates'][0]['magnetic_space_group']=="Pn'ma'"
    assert catalog['coverage']['discovery_complete']


@pytest.mark.parametrize('bad',[MACHINE.replace('{0,0,0}','{0,0}'),MACHINE.replace('{{1,0,0},{0,1,0},{0,0,1}}',"__import__('os').system('x')"),'not a native catalog',MACHINE.replace('{0,0,0}','{{{{{{0}}}}}}')])
def test_malformed_or_executable_response_is_not_a_catalog(bad):
    with pytest.raises(ValueError):parse_magnetic_subgroups_gsas_output(bad)


def test_service_errors_cannot_leak_the_credential_or_request_headers():
    def broken(*a,**kw):raise RuntimeError('Bearer secret-do-not-log')
    with pytest.raises(ValueError) as exc:discover_magnetic(PARENT,['0','0','0'],api_key='secret-do-not-log',post=broken)
    assert 'secret-do-not-log' not in str(exc.value)


def test_successful_discovery_does_not_persist_account_metadata_or_key():
    class Reply:
        status_code=200
        content=b'small response'
        def json(self):return dict(account={'private':'do-not-copy'},result={'output':MACHINE})
    result=discover_magnetic(PARENT,['0','0','0'],api_key='secret-do-not-log',post=lambda *a,**kw:Reply())
    assert 'do-not-copy' not in json.dumps(result)
    assert 'secret-do-not-log' not in json.dumps(result)
