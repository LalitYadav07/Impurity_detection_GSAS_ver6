import pytest
from gpx_recipe.live.prepare_source import validate_preparation
from gpx_recipe.live.contracts import file_hash


def request(tmp_path):
    paths={}
    for name in ('source_gpx','catalog','observation_source'):
        p=tmp_path/name;p.write_text(name);paths[name]=str(p);paths[name.replace('_gpx','')+'_sha256']=file_hash(p)
    return dict(paths,mapping=dict(mapping_kind='user_reviewed',source_gpx_sha256=paths['source_sha256'],snapshot_sha256=paths['observation_source_sha256'],producer_config_sha256='a'*64),
        parent_phase='Parent',sites=[dict(label='Fe1')],source_revision='PG3_2:'+paths['observation_source_sha256'],reference_revision='PG3_1:'+paths['source_sha256'])


def test_preparation_requires_matching_reviewed_producer_mapping(tmp_path):
    r=request(tmp_path);assert validate_preparation(r)==r
    r['mapping']['snapshot_sha256']='changed'
    with pytest.raises(ValueError,match='mapping'):validate_preparation(r)


def test_preparation_rejects_changed_revision_and_implicit_numeric_switch(tmp_path):
    r=request(tmp_path);r['source_revision']='PG3_2:different'
    with pytest.raises(ValueError,match='revision'):validate_preparation(r)
    r=request(tmp_path);r['refine_transfer']=1
    with pytest.raises(ValueError,match='explicit'):validate_preparation(r)
