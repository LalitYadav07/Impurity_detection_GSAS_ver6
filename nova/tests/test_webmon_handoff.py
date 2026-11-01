from types import SimpleNamespace
import pytest
from radar_pd_nova.webmon.ui import WebmonMixin


@pytest.mark.parametrize('published',[True,False])
def test_posterior_handoff_never_uses_parent_dataset(published):
    obj=WebmonMixin();messages=[];selected={}
    state=SimpleNamespace(webmon_busy=False,webmon_model='posterior',powgen_ipts='1',webmon_target='2',
        webmon_rows=[dict(posterior='posterior',fitted_dataset_id='posterior-hda' if published else None)],flush=lambda:None)
    obj.server=SimpleNamespace(state=state);obj._webmon_is_local=lambda:False
    obj._webmon_request={'reference_revision':'PG3_1:'+'a'*64,'source_revision':'PG3_2:'+'b'*64}
    obj._webmon_post=lambda _:dict(fitted_gpx='/verified/posterior.gpx',gpx_export={'fitted_gpx_sha256':'c'*64},rwp=5.)
    choice=dict(source_sha256='a'*64,refinement_source=dict(id='parent',experiment='PG3:1',bank='2',source_sha256='a'*64,source_path='/parent.gpx',dataset_id='parent-hda',collection_id='parent-collection',galaxy_job_id='parent-job'))
    obj._webmon=SimpleNamespace(selected_checkpoint=lambda *a:choice)
    obj._webmon_handoff_source=lambda source,prompt:selected.update(source)
    obj.webmon_open_posterior_refinement()
    if published:
        assert selected['dataset_id']=='posterior-hda'
        assert selected['source_path']=='/verified/posterior.gpx'
        assert 'collection_id' not in selected
        assert selected['run_id']=='PG3_2'
        assert 'galaxy_job_id' not in selected
    else:
        assert not selected
        assert 'not been mapped' in state.webmon_message


def test_dispatch_identity_uses_content_not_temporary_host_paths():
    from radar_pd_nova.webmon.ui import request_identity
    request=dict(source_gpx='/tmp/old/source.gpx',source_sha256='a'*64,catalog='/tmp/old/catalog.json',catalog_sha256='b'*64,source_revision='PG3_1:c')
    moved=dict(request,source_gpx='/other/source.gpx',catalog='/other/catalog.json')
    assert request_identity(request)==request_identity(moved)
    assert request_identity(request)!=request_identity(dict(moved,source_sha256='d'*64))
