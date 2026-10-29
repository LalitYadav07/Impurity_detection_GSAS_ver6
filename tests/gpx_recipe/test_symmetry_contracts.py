import copy
import hashlib
import json
from pathlib import Path
import shutil
import pytest

from gpx_recipe.symmetry.contracts import validate_protocol,digest
from gpx_recipe.symmetry.compiler import compile_bundle,verify_inputs


def make_protocol(root):
    candidate=dict(id='mag:fixture',magnetic_space_group="Pn'ma'",bns_lattice='',transformation_matrix=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],origin_shift=[0.,0.,0.],child=None,modes=[],irrep=None)
    root.mkdir(parents=True,exist_ok=True);(root/'source.gpx').write_bytes(b'opaque-native-project')
    source=dict(path='source.gpx',sha256=hashlib.sha256((root/'source.gpx').read_bytes()).hexdigest())
    catalog=dict(schema='radar-symmetry-catalog/v1',parent_setting_sha256='a'*64,k=['0','0','0'],candidates=[candidate])
    (root/'catalog.json').write_text(json.dumps(catalog))
    return validate_protocol(dict(kind='magnetic',source=source,parent_phase='nuclear',histogram='PWDR sample',parent_setting_sha256='a'*64,magnetic_function_sha256='b'*64,implementation_sha256='c'*64,
        catalog=dict(path='catalog.json',sha256=hashlib.sha256((root/'catalog.json').read_bytes()).hexdigest()),k=['0','0','0'],candidates=[candidate],
        seeds=[dict(id='x',axis='x',magnitude=3.)],sites=[dict(label='Mn1',form_factor='Mn',lande_g=2.)],
        policy=dict(cycles=12,timeout_seconds=60,maximum_trials=3,maximum_cell_multiplier=8,maximum_moment=6.,fit_band_percent=.25,maximum_mode=.5,adequate_rwp_percent=12.)))


def test_approved_job_moves_without_the_original_source_directory(tmp_path,monkeypatch):
    original=tmp_path/'original';p=make_protocol(original);job=tmp_path/'job'
    monkeypatch.setattr('gpx_recipe.symmetry.compiler.package_signature',lambda:'c'*64)
    expected=compile_bundle(p,original,job,candidate_id='mag:fixture',seed_id='x',approved_hash=digest(p))
    shutil.rmtree(original);moved=tmp_path/'moved';shutil.move(job,moved)
    verify_inputs(p,moved)
    actual=json.loads((moved/'job.json').read_text());assert actual==expected
    assert str(tmp_path) not in (moved/'worker.py').read_text()


def test_candidate_id_does_not_authorize_a_different_embedding(tmp_path,monkeypatch):
    p=make_protocol(tmp_path/'input');p['candidates'][0]['origin_shift']=[.25,0.,0.]
    with pytest.raises(ValueError,match='differs'):verify_inputs(p,tmp_path/'input')


def test_complete_protocol_is_bound_by_approval(tmp_path):
    p=make_protocol(tmp_path/'input');approved=digest(p);p['policy']['cycles']=13
    with pytest.raises(ValueError,match='Approval'):compile_bundle(p,tmp_path/'input',tmp_path/'job',candidate_id='mag:fixture',seed_id='x',approved_hash=approved)


def test_tampered_catalog_and_source_fail_before_native_loading(tmp_path):
    root=tmp_path/'input';p=make_protocol(root);(root/'source.gpx').write_bytes(b'changed')
    with pytest.raises(ValueError,match='sealed input'):verify_inputs(p,root)


@pytest.mark.parametrize('path',['../source.gpx','/tmp/source.gpx','inputs/../source.gpx','job.json','results/source.gpx','a\\b'])
def test_asset_paths_cannot_escape_or_overwrite_the_runner(tmp_path,path):
    p=make_protocol(tmp_path/'input');p['source']['path']=path
    with pytest.raises(ValueError):validate_protocol(p)


def test_symlink_cannot_escape_an_approved_input_root(tmp_path):
    root=tmp_path/'input';p=make_protocol(root);outside=tmp_path/'outside.gpx';outside.write_bytes((root/'source.gpx').read_bytes());(root/'source.gpx').unlink();(root/'source.gpx').symlink_to(outside)
    with pytest.raises(ValueError,match='escaped'):verify_inputs(p,root)


def test_protocol_cannot_expand_search_or_add_python_at_execution(tmp_path):
    p=make_protocol(tmp_path/'input');p['seeds']*=4
    with pytest.raises(ValueError):validate_protocol(p)
    p=make_protocol(tmp_path/'input');p['script']='print(1)'
    with pytest.raises(ValueError):validate_protocol(p)


def test_same_form_factor_cannot_have_two_native_g_values(tmp_path):
    p=make_protocol(tmp_path/'input');p['sites'].append(dict(label='Mn2',form_factor='Mn',lande_g=3.))
    with pytest.raises(ValueError,match='Lande'):validate_protocol(p)


def test_float_or_multik_input_is_not_rounded_or_simplified(tmp_path):
    p=make_protocol(tmp_path/'input');p['k']=[0.,0.,.49999]
    with pytest.raises(ValueError):validate_protocol(p)


@pytest.mark.parametrize('signs',[{'Mn1':0},{'Mn1':True},{'Mn1':1.0},{'absent':1}])
def test_invalid_or_unselected_parent_site_signs_fail_before_execution(tmp_path,signs):
    p=make_protocol(tmp_path/'input');p['seeds'][0]['site_signs']=signs
    with pytest.raises(ValueError):validate_protocol(p)


def test_candidate_specific_starts_do_not_create_cross_candidate_trials(tmp_path):
    from gpx_recipe.symmetry.contracts import iter_trials
    p=make_protocol(tmp_path/'input');p['candidates'].append(dict(p['candidates'][0],id='mag:second'))
    p['seeds']=[dict(p['seeds'][0],id='a',candidate_ids=['mag:fixture']),dict(p['seeds'][0],id='b',candidate_ids=['mag:second'])]
    p['policy']['maximum_trials']=2;p=validate_protocol(p)
    assert [(c['id'],s['id']) for c,s in iter_trials(p)]==[('mag:fixture','a'),('mag:second','b')]
    p['k']=[['0','0','0'],['0','0','1/2']]
    with pytest.raises(ValueError):validate_protocol(p)
