"""Before-measurement setup contracts; fake inspections are not physics tests."""
import copy
import json
from pathlib import Path

import pytest

from gpx_recipe.preparation.library import import_library
from gpx_recipe.symmetry.compiler import file_sha,package_signature
from radar_pd_nova.preparation_workspace import PreparationWorkspace
from radar_pd_nova.preparation_discovery import PreparationDiscovery
from radar_pd_nova import preparation_discovery as discovery
from test_model_runtime import remote
from test_preparation_queue import complete


PARENT=dict(name='Parent',type='nuclear',setting_sha256='a'*64,space_group='P 1',
    cell=[4.,5.,6.,90.,90.,90.],operations=[dict(rotation=[[1,0,0],[0,1,0],[0,0,1]],translation=[0,0,0])],
    atoms=[dict(label='Fe1',type='Fe',xyz=[0,0,0],occupancy=1.,adp_type='I',uiso=.005)])


def catalog(parent,k,*,structural=False):
    candidate=dict(id='structural:one' if structural else 'magnetic:one',
        transformation_matrix=[[1,0,0],[0,1,0],[0,0,1]],origin_shift=[0,0,0])
    if structural:candidate.update(irrep='GM1+',child=dict(path='children/child.cif',sha256='0'*64),modes=['mode1'])
    else:candidate.update(magnetic_space_group='P1',bns_number='1.1')
    return dict(schema='radar-symmetry-catalog/v1',parent_setting_sha256=parent['setting_sha256'],k=k,
        candidates=[candidate],coverage=dict(discovery_complete=True,constructed_count=1),provenance={'service':'test'})


@pytest.fixture
def workspace(remote,tmp_path):
    runtime,service,source=remote
    search=service.search_history_datasets
    service.search_history_datasets=lambda query,**kwargs:[dict(r,state='ok',file_size=len(service.data[r['id']]))
        for r in search(query,**kwargs)]
    profile=tmp_path/'profile.instprm';profile.write_text('# contract profile, not native science\n')
    banks={label:dict(instrument=profile,grid=dict(kind='PNT',lower=3000.,upper=40000.,bins=701,bank=n))
           for n,label in enumerate(('bank 1','bank 2'),1)}
    workspace=PreparationWorkspace(tmp_path/'workspace',runtime)
    key=workspace.inspect(name='Parent inspection',source=source['source_path'],banks=banks)
    return workspace,key,service


def finish_inspection(workspace,key,service,tmp_path,*,changed_parent=False,old_implementation=False):
    workspace.start(key);workspace.tick(key);queue=workspace.queue(key);job=queue.snapshot()['jobs'][0]
    def metadata(member):
        parent=copy.deepcopy(PARENT)
        if changed_parent and member['entry']==job['members'][1]['entry']:parent['cell'][0]=7.
        return dict(implementation_sha256='f'*64 if old_implementation else package_signature(),
            native_runtime={'native_stack_sha256':'b'*64},detail=dict(parent=parent,
                histogram='PWDR preparation reference',points={'candidates':[{'k':['0','0','0'],'labels':['Gamma']}]},
                k_scope='Finite test points'))
    complete(queue,service,job,tmp_path,manifest_extra=metadata)
    assert workspace.tick(key)['coverage']['all_prepared']


def test_inspection_and_discovery_need_no_measurement_or_gpx(workspace,tmp_path):
    w,key,service=workspace
    assert not service.submissions
    with pytest.raises(ValueError,match='Complete'):w.references(key)
    finish_inspection(w,key,service,tmp_path)
    context=w.references(key)
    assert context['parent']==PARENT and len(context['banks'])==2
    assert len(service.submissions)==1
    assert all(r['model']['kind']=='reference' for r in w.registry.get(key).document['entries'])


@pytest.mark.parametrize('cause',['parent','implementation'])
def test_different_bank_parents_or_old_reference_implementation_cannot_prepare_models(workspace,tmp_path,cause):
    w,key,service=workspace
    finish_inspection(w,key,service,tmp_path,changed_parent=cause=='parent',old_implementation=cause=='implementation')
    with pytest.raises(ValueError,match='different parent|implementation'):w.references(key)


def test_inspected_magnetic_scope_is_portable_and_keeps_bank_and_k_denominators(workspace,tmp_path):
    w,key,service=workspace;finish_inspection(w,key,service,tmp_path)
    catalogs=[dict(catalog=catalog(PARENT,k)) for k in (['0','0','0'],['0','0','1/2'])]
    model_key=w.create_models(key,name='Two-family preparation',kind='magnetic',catalogs=catalogs,
        sites=[dict(label='Fe1',form_factor='Fe+3',lande_g=2.)])
    library=w.registry.get(model_key)
    assert library.coverage()['total']==4 and library.coverage()['counts']['pending']==4
    assert len(service.submissions)==1  # Capturing a plan never starts calculations.
    scope=json.loads((library.root/'inputs/scope.json').read_text())
    assert scope['selected_banks']==['bank 1','bank 2']
    assert [s['k'] for s in scope['searches']]==[['0','0','0'],['0','0','1/2']]
    path=tmp_path/'portable.zip';w.registry.export(model_key,path)
    restored=import_library(path,tmp_path/'portable-restored')
    assert json.loads((restored.root/'inputs/scope.json').read_text())==scope


@pytest.mark.parametrize('kind',['magnetic','structural'])
def test_model_plan_inherits_only_profile_dependencies_its_engine_consumes(workspace,tmp_path,monkeypatch,kind):
    w,key,service=workspace;finish_inspection(w,key,service,tmp_path)
    context=w.references(key);inspection=w.registry.get(key);row=inspection.document['entries'][0]
    # Coordinator contract only: the real immutable transfer/byte checks have
    # separate tests. No fabricated arrays are passed to a native calculation.
    row['files'] += [dict(path='fixture/'+name,sha256='f'*64,size=1) for name in ('profile.npz','profile.json')]
    lookup=w.registry.get
    monkeypatch.setattr(w.registry,'get',lambda identifier:inspection if identifier==key else lookup(identifier))
    monkeypatch.setattr(w,'references',lambda identifier:context)
    from gpx_recipe.preparation import operations
    transferred=[]
    monkeypatch.setattr(operations,'inherit_reference',lambda target,source,identifier:transferred.append(
        dict(target=target.document['id'],source=source.document['id'],entry=identifier)))
    record=catalog(PARENT,['0','0','0'],structural=kind=='structural')
    descriptor=dict(catalog=record)
    if kind=='structural':
        directory=w.root/'discovery';child=directory/'children/child.cif';child.parent.mkdir(parents=True)
        child.write_text('data_public_child\n');record['candidates'][0]['child']['sha256']=file_sha(child)
        descriptor['directory']=str(directory)
    selected=w.create_models(key,name='Dependent model library',kind=kind,catalogs=[descriptor],
        sites=[dict(label='Fe1',form_factor='Fe+3',lande_g=2.)] if kind=='magnetic' else None)
    library=lookup(selected)
    models=[r for r in library.document['entries'] if r['model']['kind']==kind]
    references=[r for r in library.document['entries'] if r['model']['kind']=='reference']
    assert len(models)==2 and all(r['state']=='pending' for r in models)
    assert len(references)==len(transferred)==(1 if kind=='structural' else 0)
    if kind=='structural':
        examples=[r for r in library.document['entries'] if r['model']['kind']=='structural-examples']
        training=[r for r in library.document['entries'] if r['model']['kind']=='structural-training']
        assert len(examples)==16 and len(training)==2
        from gpx_recipe.preparation.structural_training_blocks import REPRESENTATION
        assert all(e['model']['settings']['representation']==REPRESENTATION for e in examples)
        for trainer in training:
            assert trainer['model']['settings']['representation']==REPRESENTATION
            assert trainer['model']['settings']['bins']==4096
            assert trainer['model']['settings']['dependencies']==sorted(
                e['id'] for e in examples if e['model']['bank']==trainer['model']['bank'])
    if references:assert references[0]['model']['bank']==row['model']['bank']


def test_failed_pattern_dependencies_pause_training_until_explicit_retry(workspace,tmp_path):
    from test_structural_preparation_dependencies import library_inputs
    from gpx_recipe.preparation.structural_dependencies import TRAINING
    w,_,service=workspace
    control,_=library_inputs(tmp_path/'structural-control')
    archive=tmp_path/'control.zip';control.export(archive)
    key=w.import_archive(archive)['library'];w.start(key,maximum_jobs=1)
    initial=w.tick(key);assert initial['active']==1
    job=initial['queue']['jobs'][0]
    complete(w.queue(key),service,job,tmp_path,
             states={m['entry']:'failed' for m in job['members']})
    result=w.tick(key)
    library=w.registry.get(key)
    trainer=next(r for r in library.document['entries'] if r['model']['kind']==TRAINING)
    assert result['active']==0 and result['queue']['paused']
    assert result['coverage']['counts']['failed']==2 and result['coverage']['counts']['pending']==1
    assert result['blocked_training']==[dict(entry=trainer['id'],dependencies=trainer['model']['settings']['dependencies'])]
    assert not result['coverage']['complete']  # No failed block is dropped.
    assert len(service.submissions)==1
    assert w.tick(key)['queue']['paused'] and len(service.submissions)==1
    for row in library.document['entries']:
        if row['state']=='failed':library.retry(row['id'])
    w.start(key,maximum_jobs=1);resumed=w.tick(key)
    assert not resumed['queue']['paused'] and resumed['active']==1
    assert len(service.submissions)==2 and not resumed['blocked_training']


@pytest.mark.parametrize('cause',['incomplete','setting','duplicate','empty','ion','bank'])
def test_unresolved_or_incompatible_discovery_does_not_start_a_hidden_narrower_plan(workspace,tmp_path,cause):
    w,key,service=workspace;finish_inspection(w,key,service,tmp_path)
    c=catalog(PARENT,['0','0','0']);catalogs=[dict(catalog=c)]
    sites=[dict(label='Fe1',form_factor='Fe+3',lande_g=2.)];banks=None
    if cause=='incomplete':c['coverage']['discovery_complete']=False
    if cause=='setting':c['parent_setting_sha256']='c'*64
    if cause=='duplicate':catalogs.append(copy.deepcopy(catalogs[0]))
    if cause=='empty':c['candidates']=[]
    if cause=='ion':sites[0]['form_factor']='Ni+2'
    if cause=='bank':banks=['bank 9']
    with pytest.raises(ValueError):w.create_models(key,name='Invalid',kind='magnetic',catalogs=catalogs,
        bank_labels=banks,sites=sites)
    assert len(w.registry.summaries())==1 and len(service.submissions)==1


def test_workspace_recovery_preserves_parent_dataset_on_second_local_reopen(workspace,tmp_path):
    w,key,service=workspace;finish_inspection(w,key,service,tmp_path)
    dataset=w.saved()[0]['dataset_id'];original=w.queue(key).identity
    fresh=PreparationWorkspace(tmp_path/'fresh-workspace',w.runtime)
    restored_key=fresh.recover(dataset)
    assert fresh.tick(restored_key)['coverage']['all_prepared']
    reopened=PreparationWorkspace(fresh.root,w.runtime)
    assert reopened.queue(restored_key).identity==original
    assert len(service.submissions)==1


def test_portable_parent_receipt_requires_native_reinspection_before_discovery(workspace,tmp_path):
    w,key,service=workspace;finish_inspection(w,key,service,tmp_path)
    packet=tmp_path/'inspection.zip';w.registry.export(key,packet)
    imported=w.import_archive(packet)['library']
    with pytest.raises(ValueError,match='captured inputs'):w.references(imported)
    # Merely attaching a fresh empty queue cannot bless imported metadata.
    w.queue(imported)
    with pytest.raises(ValueError,match='native inspection receipts'):w.references(imported)
    assert w.references(key)['parent']==PARENT
    assert len(service.submissions)==1


def test_scope_limit_is_checked_without_creating_a_silently_reduced_library(workspace,tmp_path,monkeypatch):
    from radar_pd_nova import preparation_workspace as module
    w,key,service=workspace;finish_inspection(w,key,service,tmp_path)
    monkeypatch.setattr(module,'MAX_MODELS',1)
    with pytest.raises(ValueError,match='No candidates were dropped'):
        w.create_models(key,name='Too large for one library',kind='magnetic',
            catalogs=[dict(catalog=catalog(PARENT,['0','0','0']))],
            sites=[dict(label='Fe1',form_factor='Fe+3',lande_g=2.)])
    assert len(w.registry.summaries())==1 and len(service.submissions)==1


def test_discovery_cache_reuses_scoped_catalog_and_never_serializes_service_key(workspace,tmp_path,monkeypatch):
    w,key,service=workspace;finish_inspection(w,key,service,tmp_path)
    monkeypatch.setenv('BCS_API_KEY','PRIVATE-DISCOVERY-KEY-SENTINEL');calls=[]
    def discover(parent,k,**kwargs):
        assert kwargs['api_key']=='PRIVATE-DISCOVERY-KEY-SENTINEL';calls.append(k)
        return catalog(parent,k)
    monkeypatch.setattr(discovery,'discover_magnetic',discover)
    client=PreparationDiscovery(w)
    first=client.discover(key,'magnetic',['0','0','1/2'])
    second=client.discover(key,'magnetic',['0','0','2/4'])
    assert not first['cached'] and second['cached'] and len(calls)==1
    for path in client.root.rglob('*.json'):assert 'PRIVATE-' not in path.read_text()
    receipt=next(client.root.glob('*/receipt.json'));value=json.loads(receipt.read_text())
    value['catalog']['k']=['1/2','0','0'];receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='authoritative catalog'):client.discover(key,'magnetic',['0','0','1/2'])


def test_structural_child_and_full_discovery_coverage_are_retained(workspace,tmp_path,monkeypatch):
    w,key,service=workspace;finish_inspection(w,key,service,tmp_path)
    def discover(cif,parent,k,directory,**kwargs):
        directory=Path(directory);(directory/'children').mkdir(parents=True)
        child=directory/'children/child.cif';child.write_text('data_structure_contract_only\n')
        c=catalog(parent,k,structural=True);c['candidates'][0]['child']['sha256']=file_sha(child)
        c['coverage']['discovered_count']=3
        c['coverage']['candidates']=[{'id':'structural:one','state':'constructed'},
            {'id':'structural:two','state':'deferred'},{'id':'structural:three','state':'deferred'}]
        (directory/'catalog.json').write_text(json.dumps(c));return c
    monkeypatch.setattr(discovery,'discover_structural',discover)
    c=PreparationDiscovery(w).discover(key,'structural',['0','0','0'])
    model_key=w.create_models(key,name='Structural preparation',kind='structural',catalogs=[c],bank_labels=['bank 2'])
    library=w.registry.get(model_key);model=library.document['entries'][0]['model']
    assert model['kind']=='structural' and (library.root/model['child']).read_text()=='data_structure_contract_only\n'
    from gpx_recipe.structural_live.service import SearchSettings
    assert model['settings']['cell_radius']==SearchSettings().cell_radius
    assert model['settings']['mode_radius']==SearchSettings().mode_radius
    custom_key=w.create_models(key,name='Reviewed custom domain',kind='structural',catalogs=[c],
        bank_labels=['bank 2'],cell_radius=.003)
    custom=w.registry.get(custom_key).document['entries'][0]['model']
    assert custom['settings']['cell_radius']==.003
    scope=json.loads((library.root/'inputs/scope.json').read_text())
    assert scope['searches'][0]['coverage']['discovered_count']==3
    child=Path(c['directory'])/'children/child.cif';child.write_text('changed structure')
    with pytest.raises(ValueError,match='authoritative.*changed'):PreparationDiscovery(w).discover(key,'structural',['0','0','0'])
