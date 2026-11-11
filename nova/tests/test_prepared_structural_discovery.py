"""The Model bridge preserves exact trained candidates, not regenerated CIFs."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpx_recipe.symmetry.compiler import file_sha
from radar_pd_nova import prepared_structural_discovery as bridge
from radar_pd_nova.model_admission import SCHEMA, request_identity, validate_receipt
from test_preparation_structural_ranking_hints import prepared
from test_model_runtime import remote
from test_preparation_workspace import workspace, finish_inspection, catalog as catalog_fixture, PARENT


def admitted(payload, candidates):
    from radar_pd_nova import model_admission
    return dict(schema=SCHEMA, identity=request_identity(payload), compatible=True,
        adapter_sha256=file_sha(model_admission.__file__),
        no_refinement=True, inputs_unchanged=True, reason='Native fixture admission',
        k_verified=True, mode_inventory_verified=True,
        parent_setting_sha256='a'*64, candidate_ids=[c['id'] for c in candidates],
        eligible_candidate_ids=[c['id'] for c in candidates],excluded_candidate_ids=[],
        candidate_checks=[dict(id=c['id'], child_sha256=c['child']['sha256'],
            modes_sha256=hashlib.sha256(json.dumps(c['modes'],separators=(',',':')).encode()).hexdigest(),
            mode_k_vectors=[['0','0','0']],eligibility='eligible',eligibility_reason='Matching fixture k') for c in candidates])


@pytest.fixture
def setup(tmp_path):
    library, protocol = prepared(tmp_path/'library')
    root = tmp_path/'fresh-source'; root.mkdir()
    (root/'source.gpx').write_bytes(b'A distinct freshly published scan')
    registry = SimpleNamespace(selected_key=lambda:'selected', get=lambda:library)
    state = SimpleNamespace(sx_opd_policy='all', sx_construct_limit=100,
        sx_parent='Fresh displayed phase', sx_histogram='PWDR new-file.gsa')
    calls=[]
    def native(operation,payload,root):
        calls.append(copy.deepcopy(payload));return admitted(payload,protocol['candidates'])
    app=SimpleNamespace(_preparation_workspace=SimpleNamespace(registry=registry),
        server=SimpleNamespace(state=state),_model_remote=lambda:None,_symmetry_native=native)
    draft=dict(root=root,context=dict(sha256=file_sha(root/'source.gpx')))
    return app,draft,library,protocol,calls


def test_exact_candidate_bytes_and_modes_survive_fresh_gpx_names_and_repeat_discovery(setup):
    app,draft,library,original,calls=setup
    catalog=bridge.discover(app,draft,dict(setting_sha256='a'*64),['0','0','0'],lambda _:None)
    assert len(calls)==1 and calls[0]['phase']=='Fresh displayed phase'
    for old,new in zip(original['candidates'],catalog['candidates']):
        assert old['modes']==new['modes'] and old['id']==new['id']
        assert old['child']['sha256']==file_sha(draft['root']/new['child']['path'])
        assert new['child']['path']!=old['child']['path']
    assert catalog['provenance']['prepared_library']['library_id']==library.document['id']
    assert 'reference_kind' in catalog['provenance']['prepared_library']
    assert 'Reconstructed' in catalog['provenance']['prepared_library']['scope']
    assert bridge.discover(app,draft,dict(setting_sha256='a'*64),['0','0','0'],lambda _:None)==catalog
    assert len(calls)==1
    (draft['root']/'source.gpx').write_bytes(b'Another source revision')
    draft['context']['sha256']=file_sha(draft['root']/'source.gpx')
    bridge.discover(app,draft,dict(setting_sha256='a'*64),['0','0','0'],lambda _:None)
    assert len(calls)==2


@pytest.mark.parametrize('kind,selected,expected',[
    ('fixed_before_transition_parent',True,'selected before-transition measured scan'),
    ('fixed_before_transition_parent',False,'requires a before-transition measured scan'),
    ('calculated_parent_at_shared_parent_compatible_cell',True,'its measured curve is not used'),
    ('calculated_parent_at_shared_parent_compatible_cell',False,'No measured before-transition curve is used'),
    (None,False,'does not record its ML reference type'),
])
def test_reference_notice_describes_only_the_model_reference_actually_used(kind,selected,expected):
    message=bridge.reference_notice(kind,has_reference=selected)
    assert expected in message
    if kind=='calculated_parent_at_shared_parent_compatible_cell':
        assert 'calculated parent pattern' in message


@pytest.mark.parametrize('states', [
    ['preparing'], ['pending'], ['failed'], ['unsupported'], ['cancelled'],
    ['prepared', 'preparing'],
])
def test_incomplete_ranking_scope_stops_before_assets_are_staged(states, tmp_path):
    def unexpected_asset_read(*args, **kwargs):
        pytest.fail('An incomplete ranking scope must not stage assets or start native work')

    library = SimpleNamespace(
        refresh=lambda: None,
        document={'entries': [
            {'model': {'kind': 'structural-training'}, 'state': state}
            for state in states
        ]},
        verify_assets=unexpected_asset_read,
    )
    destination = tmp_path / 'not-created'
    with pytest.raises(bridge.PreparedLibraryUnavailable) as caught:
        bridge.stage_models(library, destination)
    assert f'{states.count("prepared")} / {len(states)} ranking models ready' in str(caught.value)
    assert 'Experiment preparation' in str(caught.value)
    assert not destination.exists()


def test_discovery_retains_the_reference_kind_with_the_bound_model_metadata(setup,monkeypatch):
    app,draft,library,_,_=setup
    original=bridge.stage_models
    def staged(*args):
        models=original(*args)
        for model in models:model['reference_kind']='calculated_parent_at_shared_parent_compatible_cell'
        return models
    monkeypatch.setattr(bridge,'stage_models',staged)
    catalog=bridge.discover(app,draft,dict(setting_sha256='a'*64),['0','0','0'],lambda _:None)
    marker=catalog['provenance']['prepared_library']
    assert marker['reference_kind']=='calculated_parent_at_shared_parent_compatible_cell'
    assert marker['admission']['identity']['files']['model']


def test_incompatible_library_stops_instead_of_regenerating_or_repeating_admission(setup):
    app,draft,_,_,calls=setup
    original=app._symmetry_native
    def incompatible(*args):
        result=original(*args);result.update(compatible=False,reason='Prepared instrument profile differs');return result
    app._symmetry_native=incompatible
    for _ in range(2):
        with pytest.raises(bridge.PreparedLibraryUnavailable,match='instrument profile'):
            bridge.discover(app,draft,dict(setting_sha256='a'*64),['0','0','0'],lambda _:None)
    assert len(calls)==1


def test_distinct_candidates_can_share_exact_child_cif_bytes(setup):
    from radar_pd_nova.model_admission import validate_payload
    app, draft, _, protocol, calls = setup
    assert len(protocol['candidates']) == 2
    assert len({c['child']['sha256'] for c in protocol['candidates']}) == 1
    native = app._symmetry_native
    def checked(operation, payload, root):
        validate_payload(payload)
        return native(operation, payload, root)
    app._symmetry_native = checked
    catalog = bridge.discover(app, draft, dict(setting_sha256='a'*64), ['0','0','0'], lambda _: None)
    children = calls[0]['children']
    assert len(children) == len(set(children)) == 2
    assert Path(children[0]).read_bytes() == Path(children[1]).read_bytes()
    assert [c['id'] for c in catalog['candidates']] == [c['id'] for c in protocol['candidates']]


def test_changed_or_missing_library_bytes_never_reach_admission(setup):
    app,draft,library,_,calls=setup
    child=next(a for a in library.document['inputs'] if a['path'].endswith('.cif'))
    (library.root/child['path']).write_text('changed data')
    with pytest.raises(ValueError,match='changed'):
        bridge.discover(app,draft,dict(setting_sha256='a'*64),['0','0','0'],lambda _:None)
    assert not calls


def test_no_selected_library_preserves_ordinary_discovery(setup):
    app,draft,_,_,calls=setup
    app._preparation_workspace.registry.selected_key=lambda:''
    assert bridge.discover(app,draft,{},['0','0','0'],lambda _:None) is None
    assert not calls


@pytest.mark.parametrize('old',[True,False])
def test_registered_admission_action_is_required(monkeypatch,old):
    options=[dict(value='native'),dict(value='chunk')]+([] if old else [dict(value='admission')])
    class Response:
        def raise_for_status(self):pass
        def json(self):return dict(id='model',inputs=[dict(name='action',options=options)])
    def fetch(*a,**kwargs):
        assert kwargs['allow_redirects'] is False
        return Response()
    monkeypatch.setattr('requests.get',fetch)
    runtime=SimpleNamespace(tool_id='model',service=SimpleNamespace(galaxy_url='https://invalid',_headers={}))
    if old:
        with pytest.raises(bridge.PreparedLibraryUnavailable,match='does not support'):bridge.require_capability(runtime)
    else:bridge.require_capability(runtime)


def test_admission_capability_cannot_use_another_tools_response(monkeypatch):
    class Response:
        def raise_for_status(self):pass
        def json(self):return dict(id='another-tool',inputs=[dict(name='action',options=[dict(value='admission')])])
    monkeypatch.setattr('requests.get',lambda *a,**k:Response())
    runtime=SimpleNamespace(tool_id='model',service=SimpleNamespace(galaxy_url='https://invalid',_headers={}))
    with pytest.raises(bridge.PreparedLibraryUnavailable,match='exact registered'):bridge.require_capability(runtime)


def test_admitted_catalog_requires_same_ranking_attachment(setup):
    app,draft,_,_,_=setup
    catalog=bridge.discover(app,draft,dict(setting_sha256='a'*64),['0','0','0'],lambda _:None)
    (draft['root']/'catalog.json').write_text(json.dumps(catalog))
    protocol=dict(kind='structural',catalog=dict(path='catalog.json'),candidates=catalog['candidates'])
    with pytest.raises(bridge.PreparedLibraryUnavailable,match='could not be attached'):
        bridge.required_attachment(protocol,draft['root'],None)


def test_tampered_receipt_selector_or_candidate_is_rejected(setup):
    app,draft,_,protocol,calls=setup
    bridge.discover(app,draft,dict(setting_sha256='a'*64),['0','0','0'],lambda _:None)
    payload=calls[0];receipt=admitted(payload,protocol['candidates'])
    altered=copy.deepcopy(payload);altered['histogram']='another histogram'
    with pytest.raises(ValueError,match='did not bind'):validate_receipt(receipt,altered)
    receipt['candidate_checks'][0]['child_sha256']='f'*64
    with pytest.raises(ValueError,match='candidate/k'):validate_receipt(receipt,payload)


@pytest.mark.parametrize('field',['k_verified','mode_inventory_verified'])
def test_receipt_cannot_replace_required_native_checks_with_false(setup,field):
    app,draft,_,protocol,calls=setup
    bridge.discover(app,draft,dict(setting_sha256='a'*64),['0','0','0'],lambda _:None)
    payload=calls[0];receipt=admitted(payload,protocol['candidates']);receipt[field]=False
    with pytest.raises(ValueError,match='candidate/k'):validate_receipt(receipt,payload)


@pytest.mark.parametrize('fault',[None,'cancel','lost_ack'])
def test_new_admission_action_uses_existing_durable_submission_guard(remote,tmp_path,fault):
    import threading
    from radar_pd_nova.model_admission import OPERATION
    from radar_pd_nova.native_operations import validate_native_timeout
    from radar_pd_nova.refinement_runtime import RuntimeUnavailable
    runtime,service,source=remote
    root=tmp_path/'inputs';root.mkdir()
    paths={}
    for name in ('parent.cif','profile.instprm','model.json','weights.npz','child.cif'):
        path=root/name;path.write_bytes(b'contract fixture');paths[name]=str(path)
    local=root/'source.gpx';local.write_bytes(Path(source['source_path']).read_bytes())
    payload=dict(source=str(local),parent=paths['parent.cif'],instrument=paths['profile.instprm'],
        model=paths['model.json'],weights=paths['weights.npz'],children=[paths['child.cif']],
        phase='Fresh phase',histogram='Fresh histogram',k=['0','0','0'],
        grid=dict(kind='PNT',lower=3000.,upper=40000.,bins=701,bank=1),
        library_id='a'*24,destination=str(root/'admitted'))
    if fault=='cancel':
        cancel=threading.Event();cancel.set()
        with pytest.raises(RuntimeUnavailable):runtime.submit_native(OPERATION,payload,root,source=source,timeout=300,cancel=cancel)
        assert not service.submissions
    elif fault=='lost_ack':
        service.lose_ack=True
        with pytest.raises(RuntimeUnavailable):runtime.submit_native(OPERATION,payload,root,source=source,timeout=300)
    handle=runtime.submit_native(OPERATION,payload,root,source=source,timeout=300)
    repeated=runtime.submit_native(OPERATION,payload,root,source=source,timeout=300)
    assert repeated==handle and len(service.submissions)==1
    assert service.submissions[0]['inputs']['action']=='admission'
    assert handle['request']['implementation_sha256']
    with pytest.raises(ValueError):validate_native_timeout(OPERATION,float('nan'))


@pytest.mark.parametrize('has_library',[True,False,'mixed'])
def test_actual_discovery_controller_uses_bridge_or_ordinary_path(setup,monkeypatch,has_library):
    import asyncio
    from radar_pd_nova.symmetry_controller import SymmetryMixin
    from radar_pd_nova import symmetry_controller
    helper,draft,library,protocol,calls=setup
    if has_library=='mixed':
        full=helper._symmetry_native
        def subset(*args):
            result=full(*args)
            result['candidate_checks'][0].update(eligibility='excluded',eligibility_reason='Another native k star')
            result['eligible_candidate_ids']=[protocol['candidates'][1]['id']]
            result['excluded_candidate_ids']=[protocol['candidates'][0]['id']]
            return result
        helper._symmetry_native=subset
    class State(SimpleNamespace):
        def flush(self):pass
    app=SymmetryMixin();app.server=SimpleNamespace(state=State(**vars(helper.server.state),
        sx_start_source='fresh',sx_kind='structural',sx_k='0,0,0',sx_parent_reference='',sx_slots=1))
    app._preparation_workspace=helper._preparation_workspace if has_library else None
    app._symmetry_draft=dict(draft,source_id='fresh',context=dict(draft['context'],phases=[dict(
        name=app.server.state.sx_parent,setting_sha256='a'*64)]))
    app._model_remote=lambda:None;app._discovery_identity=lambda:'same'
    app._finish_symmetry_preparation=lambda:None
    app._refinement_schedule=lambda action,*_:asyncio.run(action)
    external=[]
    def native(operation,payload,root):
        if operation=='export':Path(payload['destination']).write_text('data_parent');return {}
        return helper._symmetry_native(operation,payload,root)
    app._symmetry_native=native
    def discover(parent_cif,parent,k,directory,**kwargs):
        external.append(True);directory.mkdir(parents=True)
        c=copy.deepcopy(protocol['candidates'][0]);c['child']['path']='child.cif'
        (directory/'child.cif').write_text('ordinary discovered child')
        c['child']['sha256']=file_sha(directory/'child.cif')
        return dict(schema='radar-symmetry-catalog/v1',candidates=[c],coverage=dict(
            discovery_complete=True,discovered_count=1,constructed_count=1,opd_policy='all',
            candidates=[dict(id=c['id'],state='constructed')]))
    monkeypatch.setattr(symmetry_controller,'discover_structural',discover)
    app.discover_symmetry_candidates()
    assert not app.server.state.sx_error
    assert app.server.state.sx_discovered
    assert bool(external) is (not has_library)
    assert bool(calls) is bool(has_library)
    if has_library=='mixed':
        assert len(app.server.state.sx_candidates)==1
        assert len(app.server.state.sx_discovery_rows)==2
        assert {row['state'] for row in app.server.state.sx_discovery_rows}=={'eligible','excluded'}
        assert '1 / 2 candidates eligible' in app.server.state.sx_preparation_notice


def test_explicit_prepared_library_error_is_not_swallowed_by_optional_transport(remote,tmp_path):
    from gpx_recipe.symmetry.campaign import Campaign
    from radar_pd_nova.model_runtime import GalaxyChunkTransport
    runtime,service,_=remote
    def required(*_):raise bridge.PreparedLibraryUnavailable('Required selected model changed')
    runtime.preparation_provider=required
    transport=GalaxyChunkTransport(runtime,Campaign(tmp_path/'campaign',shared_inputs=True))
    with pytest.raises(bridge.PreparedLibraryUnavailable,match='Required selected model'):
        transport._prepared_inputs({},tmp_path,capture_new=True)
    assert not service.submissions


@pytest.mark.parametrize('damage',[None,'missing-manifest','missing-weights','changed-weights','bad-catalog'])
def test_recovered_required_ranking_cannot_silently_become_unranked(setup,remote,tmp_path,damage):
    from gpx_recipe.preparation import structural_hints
    from gpx_recipe.symmetry.campaign import Campaign
    from radar_pd_nova.model_runtime import GalaxyChunkTransport
    app,draft,library,_,_=setup;runtime,service,_=remote
    catalog=bridge.discover(app,draft,dict(setting_sha256='a'*64),['0','0','0'],lambda _:None)
    root=draft['root'];(root/'catalog.json').write_text(json.dumps(catalog))
    protocol=dict(kind='structural',catalog=dict(path='catalog.json',sha256=file_sha(root/'catalog.json')),
        candidates=catalog['candidates'])
    document=structural_hints.capture(library,protocol,root)
    weights=root/document['ranking_models']['entries'][0]['files']['weights']['path']
    if damage=='missing-manifest':(root/structural_hints.MANIFEST).unlink()
    elif damage=='missing-weights':weights.unlink()
    elif damage=='changed-weights':weights.write_bytes(b'corrupt')
    elif damage=='bad-catalog':(root/'catalog.json').write_text('{broken')
    transport=GalaxyChunkTransport(runtime,Campaign(tmp_path/'recovery-campaign',shared_inputs=True))
    if damage is None:
        restored,paths=transport._prepared_inputs(protocol,root,capture_new=False)
        assert restored==document and str(weights.relative_to(root)) in paths
    else:
        with pytest.raises(bridge.PreparedLibraryUnavailable):
            transport._prepared_inputs(protocol,root,capture_new=False)
    assert not service.submissions


def test_new_preparations_keep_original_authoritative_catalog_in_portable_scope(workspace,tmp_path):
    from gpx_recipe.symmetry.contracts import digest
    store,key,service=workspace
    finish_inspection(store,key,service,tmp_path)
    directory=store.root/'retained-catalog';directory.mkdir()
    child=directory/'child.cif';child.write_text('data_catalog_transport_fixture\n')
    original=catalog_fixture(PARENT,['0','0','0'],structural=True)
    original['candidates'][0]['child']=dict(path='child.cif',sha256=file_sha(child))
    model_key=store.create_models(key,name='Retained catalog',kind='structural',
        catalogs=[dict(catalog=original,directory=str(directory))],bank_labels=['bank 1'])
    library=store.registry.get(model_key)
    scope_path=library.verify_assets([library.document['specification']['scope']])
    scope=json.loads(next(iter(scope_path.values())).read_text())
    assert scope['searches'][0]['catalog']==original
    assert scope['searches'][0]['catalog_sha256']==digest(original)


@pytest.mark.parametrize('wrong_k',[False,True])
def test_retained_authoritative_catalog_is_preserved_and_bound_to_requested_k(setup,monkeypatch,wrong_k):
    from gpx_recipe.symmetry.contracts import digest
    app,draft,_,protocol,_=setup
    original=dict(schema='radar-symmetry-catalog/v1',parent_setting_sha256='a'*64,
        k=['0','0','1/2'] if wrong_k else ['0','0','0'],candidates=copy.deepcopy(protocol['candidates']),
        coverage=dict(discovery_complete=True,discovered_count=2,constructed_count=2,opd_policy='all',
            candidates=[dict(id=c['id'],state='constructed',label='GM3+ · C2/m, basis={(1,0,0)}') for c in protocol['candidates']]))
    stage=bridge.stage_models
    def retained(*args):
        models=stage(*args);models[0]['original_catalogs']=[copy.deepcopy(original)];return models
    monkeypatch.setattr(bridge,'stage_models',retained)
    catalog=bridge.discover(app,draft,dict(setting_sha256='a'*64),['0','0','0'],lambda _:None)
    provenance=catalog['provenance']['prepared_library']
    if wrong_k:assert 'original_catalog' not in provenance and 'Reconstructed' in provenance['scope']
    else:
        assert provenance['original_catalog']==original
        assert provenance['original_catalog_sha256']==digest(original)
        assert {r['label'] for r in catalog['coverage']['candidates']}=={'GM3+ · C2/m'}


def test_multi_k_catalog_filters_only_protocol_and_preserves_complete_model(setup):
    app,draft,library,protocol,calls=setup
    ordinary=app._symmetry_native
    def mixed(*args):
        value=ordinary(*args)
        excluded=value['candidate_checks'][0]
        excluded.update(eligibility='excluded',eligibility_reason='Different native k star')
        value['eligible_candidate_ids']=[protocol['candidates'][1]['id']]
        value['excluded_candidate_ids']=[protocol['candidates'][0]['id']]
        return value
    app._symmetry_native=mixed
    before=json.dumps(library.document,sort_keys=True)
    catalog=bridge.discover(app,draft,dict(setting_sha256='a'*64),['0','0','0'],lambda _:None)
    assert [c['id'] for c in catalog['candidates']]==[protocol['candidates'][1]['id']]
    marker=catalog['provenance']['prepared_library'];receipt=marker['admission']
    assert len(receipt['candidate_checks'])==2 and len(receipt['excluded_candidate_ids'])==1
    assert catalog['coverage']['stored_count']==2 and catalog['coverage']['eligible_count']==1
    assert json.dumps(library.document,sort_keys=True)==before
    from gpx_recipe.preparation import structural_hints
    capture=dict(kind='structural',candidates=catalog['candidates'])
    document=structural_hints.capture(library,capture,draft['root']/'subset-packet')
    model=document['ranking_models']['entries'][0]
    assert model['routes']=={protocol['candidates'][1]['id']:protocol['candidates'][1]['id']}
    metadata=json.loads((draft['root']/'subset-packet'/model['files']['model']['path']).read_text())
    assert len(metadata['candidates'])==2


def test_multi_k_model_selects_matching_retained_catalog(setup,monkeypatch):
    app,draft,_,protocol,_=setup
    def original(candidate,k):
        return dict(schema='radar-symmetry-catalog/v1',parent_setting_sha256='a'*64,k=k,
            candidates=[copy.deepcopy(candidate)],coverage=dict(discovery_complete=True,
                discovered_count=1,constructed_count=1,opd_policy='all',
                candidates=[dict(id=candidate['id'],state='constructed')]))
    catalogs=[original(protocol['candidates'][0],['0','0','1/2']),
              original(protocol['candidates'][1],['0','0','0'])]
    stage=bridge.stage_models
    def staged(*args):
        models=stage(*args);models[0]['original_catalogs']=copy.deepcopy(catalogs);return models
    monkeypatch.setattr(bridge,'stage_models',staged)
    native=app._symmetry_native
    def checked(*args):
        result=native(*args);result['candidate_checks'][0].update(
            eligibility='excluded',eligibility_reason='Different k star')
        result['eligible_candidate_ids']=[protocol['candidates'][1]['id']]
        result['excluded_candidate_ids']=[protocol['candidates'][0]['id']]
        return result
    app._symmetry_native=checked
    catalog=bridge.discover(app,draft,dict(setting_sha256='a'*64),['0','0','0'],lambda _:None)
    assert catalog['provenance']['prepared_library']['original_catalog']==catalogs[1]
    assert len(catalog['candidates'])==1 and catalog['coverage']['excluded_count']==1


def test_no_eligible_candidates_preserves_complete_exclusion_report(setup):
    app,draft,_,protocol,_=setup;native=app._symmetry_native
    def checked(*args):
        result=native(*args)
        for row in result['candidate_checks']:
            row.update(eligibility='excluded',eligibility_reason='Gamma-only candidate')
        result.update(compatible=False,eligible_candidate_ids=[],
            excluded_candidate_ids=[c['id'] for c in protocol['candidates']],reason='No eligible candidates')
        return result
    app._symmetry_native=checked
    with pytest.raises(bridge.PreparedLibraryUnavailable) as caught:
        bridge.discover(app,draft,dict(setting_sha256='a'*64),['0','0','1/2'],lambda _:None)
    assert len(caught.value.candidate_checks)==2
    assert all(row['eligibility']=='excluded' for row in caught.value.candidate_checks)
