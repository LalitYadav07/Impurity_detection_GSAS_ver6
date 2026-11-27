"""A recovered catalog is data for a new plan, never a recovered approval."""
import asyncio
from copy import deepcopy
import json
import shutil
import threading
from types import SimpleNamespace

import pytest

from gpx_recipe.symmetry.campaign import Campaign
from gpx_recipe.symmetry.compiler import file_sha,prepare_shared_inputs,package_signature
from gpx_recipe.symmetry.contracts import digest,validate_protocol
from gpx_recipe.symmetry.discovery import magnetic_request
from radar_pd_nova import saved_magnetic_setup as saved
from radar_pd_nova import symmetry_controller as controller


@pytest.fixture
def fixture(tmp_path):
    root=tmp_path/'old';root.mkdir();(root/'source.gpx').write_bytes(b'public synthetic opaque source')
    source=dict(id='source',experiment='experiment',sample_key='sample',bank='1',
        source_sha256=file_sha(root/'source.gpx'))
    parent=dict(name='nuclear',type='nuclear',setting_sha256='a'*64,histograms=['PWDR test'],
        operations=[dict(rotation=[[1,0,0],[0,1,0],[0,0,1]],translation=[0,0,0])],
        atoms=[dict(label='Mn1',type='Mn',occupancy=1.,xyz=[0.,0.,0.])])
    context=dict(sha256=source['source_sha256'],phases=[parent],
        histograms=[dict(name='PWDR test',instrument={'Type':['PNC','PNC',False]})],
        runtime={'magnetic_function_sha256':'b'*64})
    candidates=[dict(id='mag:'+str(i),magnetic_space_group='P1',bns_number='1.1',bns_lattice='',
        child=None,modes=[],irrep=None,transformation_matrix=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
        origin_shift=[i*.25,0.,0.]) for i in range(2)]
    catalog=dict(schema='radar-symmetry-catalog/v1',parent_setting_sha256='a'*64,k=['0','0','0'],
        candidates=candidates,coverage=dict(discovery_complete=True,constructed_count=2),
        provenance=dict(request=magnetic_request(parent,['0','0','0']),output_sha256='e'*64))
    (root/'catalog.json').write_text(json.dumps(catalog))
    protocol=validate_protocol(dict(kind='magnetic',source={'path':'source.gpx','sha256':source['source_sha256']},
        parent_phase='nuclear',histogram='PWDR test',parent_setting_sha256='a'*64,
        magnetic_function_sha256='b'*64,implementation_sha256='c'*64,
        catalog={'path':'catalog.json','sha256':file_sha(root/'catalog.json')},k=['0','0','0'],
        candidates=candidates,seeds=[dict(id='x',axis='x',magnitude=2.)],
        sites=[dict(label='Mn1',form_factor='Mn',lande_g=2.)],
        policy=dict(cycles=20,timeout_seconds=120,maximum_trials=2000,maximum_cell_multiplier=8,
            maximum_moment=6.,fit_band_percent=.25,maximum_mode=.5,adequate_rwp_percent=12.,
            moment_model='collinear',magnetic_optimizer='gsas_hessian')))
    campaign=Campaign(tmp_path/'campaigns/campaign-012345abcdef',shared_inputs=True)
    inputs=prepare_shared_inputs(protocol,root,campaign.root)
    template=dict(schema='radar-model-template/v1',protocol=protocol,context=deepcopy(context),
        scope=dict(experiment='experiment',sample_key='sample',bank='1'),origin='radar',
        existing_ids=['source'],seen_ids=['source'],future=False,maximum_scans=1)
    campaign.set_setting('template',template);campaign.set_setting('reference_input_root',str(inputs))
    draft=tmp_path/'draft';draft.mkdir();shutil.copyfile(root/'source.gpx',draft/'source.gpx')
    return SimpleNamespace(root=root,campaign=campaign,inputs=inputs,template=template,protocol=protocol,
        catalog=catalog,context=context,source=source,draft=draft)


def load(f):return saved.load(f.campaign,context=f.context,source=f.source,origin='radar',draft_root=f.draft)


def reseal(f,catalog):
    """Create internally consistent but scientifically wrong recorded data."""
    (f.root/'catalog.json').write_text(json.dumps(catalog))
    f.protocol['catalog']['sha256']=file_sha(f.root/'catalog.json')
    f.inputs=prepare_shared_inputs(f.protocol,f.root,f.campaign.root)
    f.template['protocol']=deepcopy(f.protocol);f.campaign.set_setting('template',f.template)


def test_catalog_bytes_and_distinct_same_bns_embeddings_survive(fixture):
    f=fixture;before=f.campaign.snapshot();result=load(f)
    saved.stage(f.draft,result)
    assert file_sha(f.draft/'catalog.json')==f.protocol['catalog']['sha256']
    assert [c['id'] for c in result['catalog']['candidates']]==['mag:0','mag:1']
    assert result['binding']['reused_approval'] is False and result['binding']['native_operator_audit_required']
    assert f.campaign.snapshot()==before


@pytest.mark.parametrize('field,value',[('id','other'),('sample_key','foreign'),('experiment','foreign'),('bank','2'),('source_sha256','f'*64)])
def test_foreign_source_or_scope_fails_closed(fixture,field,value):
    setattr_source=fixture.source;setattr_source[field]=value
    with pytest.raises(ValueError,match='another source'):load(fixture)


@pytest.mark.parametrize('mutation',['source','protocol','catalog','link'])
def test_changed_or_linked_immutable_assets_are_rejected(fixture,mutation):
    f=fixture
    if mutation=='source':(f.draft/'source.gpx').write_bytes(b'changed')
    if mutation=='protocol':(f.inputs/'protocol.json').write_text('{}')
    if mutation=='catalog':(f.inputs/'catalog.json').write_text('{}')
    if mutation=='link':
        data=(f.inputs/'catalog.json').read_bytes();(f.inputs/'other.json').write_bytes(data)
        (f.inputs/'catalog.json').unlink();(f.inputs/'catalog.json').symlink_to('other.json')
    with pytest.raises(ValueError):load(f)


@pytest.mark.parametrize('mutation',['parent','operations','site','type'])
def test_fresh_native_context_must_match_recorded_parent(fixture,mutation):
    p=fixture.context['phases'][0]
    if mutation=='parent':p['setting_sha256']='d'*64
    if mutation=='operations':p['operations'][0]['translation']=[.5,0.,0.]
    if mutation=='site':p['atoms'][0]['occupancy']=.5
    if mutation=='type':p['type']='magnetic'
    with pytest.raises(ValueError):load(fixture)


@pytest.mark.parametrize('mutation',['incomplete','request','bns','duplicate'])
def test_even_resealed_invalid_catalog_is_rejected(fixture,mutation):
    f=fixture;catalog=deepcopy(f.catalog)
    if mutation=='incomplete':catalog['coverage']['discovery_complete']=False
    if mutation=='request':catalog['provenance']['request']['inputs']['km1z']='1/2'
    if mutation=='bns':catalog['candidates'][0]['bns_number']=None;f.protocol['candidates'][0]['bns_number']=None
    if mutation=='duplicate':catalog['candidates'][1]=deepcopy(catalog['candidates'][0])
    if mutation=='duplicate':
        with pytest.raises(ValueError):reseal(f,catalog)
    else:
        reseal(f,catalog)
        with pytest.raises(ValueError):load(f)


def test_foreign_reference_path_is_never_followed(fixture,tmp_path):
    fixture.campaign.set_setting('reference_input_root',str(tmp_path/'foreign'))
    assert load(fixture)['catalog']==fixture.catalog


@pytest.mark.parametrize('mutation',['sites','k','parent','catalog','source','operations'])
def test_new_plan_rechecks_saved_setup_binding(fixture,mutation):
    f=fixture;result=load(f);saved.stage(f.draft,result)
    d=dict(root=f.draft,context=f.context,saved_magnetic_setup=result['binding'])
    settings=dict(result['settings'],kind='magnetic')
    if mutation=='sites':settings['sites']=[dict(label='Mn1',form_factor='Mn+2',lande_g=2.)]
    if mutation=='k':settings['k']=['0','0','1/2']
    if mutation=='parent':settings['parent']='other'
    if mutation=='catalog':(f.draft/'catalog.json').write_text('{}')
    if mutation=='source':(f.draft/'source.gpx').write_bytes(b'changed after load')
    if mutation=='operations':d['context']['phases'][0]['operations'][0]['translation']=[.5,0.,0.]
    with pytest.raises(ValueError,match='changed'):saved.verify_binding(d,settings)


class State(SimpleNamespace):
    def flush(self):pass


def app_for(f,monkeypatch):
    state=State(**{key:None for key in controller.MODEL_FIELDS})
    state.__dict__.update(rf_experiment='experiment',sx_start_source='source',sx_campaign=f.campaign.root.name,
        sx_has_context=True,sx_busy=False,sx_error='',sx_approved=True,sx_plan_ready=True,
        sx_kind='magnetic',sx_parent='nuclear',sx_histogram='PWDR test',sx_k='0, 0, 0')
    app=controller.SymmetryMixin();app.server=SimpleNamespace(state=state)
    app.service=SimpleNamespace(history_id='history',galaxy_url='https://galaxy.invalid')
    app._symmetry_draft=dict(root=f.draft,context=deepcopy(f.context),source=deepcopy(f.source),
        source_id='source',experiment='experiment')
    app._symmetry_approved='old-approval';app._symmetry_refresh_pending=False
    app._symmetry_root=lambda:f.campaign.root.parent
    workspace=SimpleNamespace();app._refinement_store=lambda:workspace
    app._refresh_symmetry_preparation_notice=lambda:None
    app.tasks=[]
    def schedule(coroutine,*args):
        task=asyncio.create_task(coroutine);app.tasks.append(task);return task
    app._refinement_schedule=schedule
    monkeypatch.setattr(controller,'model_choices',lambda *a:[(f.source,'radar')])
    monkeypatch.setattr(controller,'discover_magnetic',lambda *a,**k:pytest.fail('Saved setup must not discover remotely'))
    return app,state


def test_UI_reuse_requires_new_current_plan_and_approval(fixture,monkeypatch):
    f=fixture;app,s=app_for(f,monkeypatch)
    async def exercise():
        app.use_saved_magnetic_setup();await app.tasks[-1]
    asyncio.run(exercise())
    assert not s.sx_error and not s.sx_busy and s.sx_discovered
    assert s.sx_site_options==[dict(title='Mn1 (Mn)',value='Mn1')]
    assert not s.sx_approved and not s.sx_plan_ready and app._symmetry_approved==''
    assert s.sx_existing=='selected' and not s.sx_follow_new and s.sx_max_scans==1
    assert not f.campaign.trials() and not app._symmetry_draft.get('protocol')
    app.build_symmetry_plan()
    assert not s.sx_error and s.sx_plan_ready and not s.sx_approved
    current=app._symmetry_draft['protocol']
    assert current['implementation_sha256']==package_signature()!='c'*64
    assert digest(current)!=digest(f.protocol)
    app.approve_symmetry()
    assert s.sx_approved and app._symmetry_approved==digest(app._symmetry_draft['template'])
    assert f.campaign.setting('template')==f.template


def test_deferred_parent_callback_preserves_loaded_sites(fixture,monkeypatch):
    app,s=app_for(fixture,monkeypatch)
    original=s.flush
    def flush():
        if getattr(app,'_symmetry_applying_intent',False):app._symmetry_parent_changed()
        original()
    s.flush=flush
    async def exercise():
        app.use_saved_magnetic_setup();await app.tasks[-1]
    asyncio.run(exercise())
    assert not s.sx_error and s.sx_sites==['Mn1']
    assert s.sx_form_factors=='Mn1=Mn' and s.sx_lande=='Mn1=2.0'


@pytest.mark.parametrize('change',['source','campaign','form','history','server'])
def test_selection_change_during_load_cannot_publish_old_setup(fixture,monkeypatch,change):
    app,s=app_for(fixture,monkeypatch);entered=threading.Event();release=threading.Event();original=saved.load
    def delayed(*a,**k):entered.set();assert release.wait(3);return original(*a,**k)
    monkeypatch.setattr(saved,'load',delayed)
    async def exercise():
        app.use_saved_magnetic_setup();task=app.tasks[-1]
        assert await asyncio.to_thread(entered.wait,2)
        if change=='source':s.sx_start_source='changed'
        if change=='campaign':s.sx_campaign='campaign-ffffffffffff'
        if change=='form':s.sx_k='0,0,1/2'
        if change=='history':app.service.history_id='another-history'
        if change=='server':app.service.galaxy_url='https://other.invalid'
        release.set();await task
    asyncio.run(exercise())
    assert 'changed' in s.sx_error and not s.sx_approved and not s.sx_busy
    assert not (fixture.draft/'catalog.json').exists()


@pytest.mark.parametrize('change',['draft','source','history','form'])
def test_callback_cannot_publish_setup_into_changed_selection(fixture,monkeypatch,change):
    app,s=app_for(fixture,monkeypatch);original=s.flush
    def flush():
        if getattr(app,'_symmetry_applying_intent',False):
            if change=='draft':app._symmetry_draft=None
            if change=='source':s.sx_start_source='different'
            if change=='history':app.service.history_id='other'
            if change=='form':s.sx_k='0,0,1/2'
        original()
    s.flush=flush
    async def exercise():
        app.use_saved_magnetic_setup();await app.tasks[-1]
    asyncio.run(exercise())
    assert 'changed' in s.sx_error and not s.sx_approved and not s.sx_busy
    assert not s.sx_plan_ready
    assert not (app._symmetry_draft or {}).get('saved_magnetic_setup')


def test_explicit_rediscovery_clears_saved_binding(fixture,monkeypatch):
    app,s=app_for(fixture,monkeypatch);calls=[]
    async def exercise():
        app.use_saved_magnetic_setup();await app.tasks[-1]
        monkeypatch.setattr(controller,'discover_magnetic',lambda *a,**k:(calls.append(1) or deepcopy(fixture.catalog)))
        app.discover_symmetry_candidates();await app.tasks[-1]
    asyncio.run(exercise())
    assert calls==[1] and 'saved_magnetic_setup' not in app._symmetry_draft
    assert not s.sx_approved
