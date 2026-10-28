"""Intent/scope/lifecycle contracts independent of the GPX scientific engine."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from radar_pd_nova.refinement_sources import (RefinementScope, source_from_record,select_published_project,published_project_path,file_sha)
from radar_pd_nova.refinement_workspace import RefinementWorkspace
from radar_pd_nova.refinement_runtime import LocalRefinementRuntime,verify_result_files


def context(source):
    return dict(model_signature='same-model',source={'name':'source.gpx','sha256':source['source_sha256']},
        histograms=[dict(key='h0',name='h',background={'function':'chebyschev-1','coefficients':6})],
        phases=[dict(key='p0',name='oxide',histograms=['h'],atoms=[])])


def plan(ctx):
    return dict(schema='radar-gpx-recipe/v1',model_signature=ctx['model_signature'],request='background',assumptions=[],questions=[],unsupported=[],
        stages=[dict(id='bg',title='Background only',cycles=3,free=[dict(kind='background',histogram='h0')],constraints=[])])


class Runtime:
    kind='local'
    def __init__(self):self.calls=[];self.response_status='completed';self.callback=None
    def resolve(self,source):return copy.deepcopy(source)
    def call(self,action,source,directory,**kwargs):
        self.calls.append((action,source['id'],kwargs));Path(directory).mkdir(parents=True,exist_ok=True)
        if self.callback:self.callback()
        if action=='inspect':return dict(status='inspected',context=context(source),source_sha256=source['source_sha256'],directory=str(directory))
        if action=='preview':return dict(status='preview_passed',preview={'status':'preview_passed','model_signature':'same-model','source_sha256':source['source_sha256'],
            'plan_sha256':__import__('gpx_recipe.contracts',fromlist=['plan_hash']).plan_hash(kwargs['plan']),'stages':[]})
        return dict(status=self.response_status,directory=str(directory),error='Plan does not match the inspected GPX model' if self.response_status=='needs_review' else None)
    def save_workspace(self,manifest):return None
    def recover_workspace(self):return None


def source(uid,*,exp='exp',sample='id:0012',bank='2',checksum=None):
    return dict(id=uid,experiment=exp,scan_key='scan-'+uid,source_sha256=checksum or uid+'hash',source_path='/not-loaded.gpx',
        sample_key=sample,sample_label=sample,bank=bank,scan_label=uid,run_number=int(''.join(c for c in uid if c.isdigit()) or 1),record_uid=uid)


def create(tmp_path):
    runtime=Runtime();w=RefinementWorkspace(tmp_path,history_id='history',runtime=runtime,
        planner=lambda request,ctx,previous=None:dict(plan=plan(ctx),calls=[dict(label='test planner')]))
    return w,runtime


def approved(w,scope,*,future=False,existing=True):
    base=next(s for s in w.sources(scope.experiment) if scope.matches(s))
    draft=w.prepare(scope.experiment,base['id'],'background only')
    _,token=w.selection(scope,existing=existing,future=future)
    r=w.approve(scope.experiment,draft['generation'],scope,existing=existing,future=future,selection_token=token,approver='test approval')
    return r


def test_exact_sample_and_bank_scope_and_existing_backlog_are_snapshotted(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1'),source('2',sample='id:12'),source('3',bank='3'),source('4',exp='other')])
    scope=RefinementScope('exp','samples',samples=('id:0012',),banks=('2',));recipe=approved(w,scope)
    w.observe([source('5')])
    jobs=w.start(recipe['id'],recipe['selection_token'])
    assert [j['source']['id'] for j in jobs]==['1']
    w.run_next();assert w.run_next() is None
    assert w.public('exp')['jobs'][0]['status']=='completed'
    assert w.state['sources']['1']['source_sha256']=='1hash'


def test_future_only_enrolls_new_matching_arrivals_not_existing_or_other_experiments(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1')]);scope=RefinementScope('exp','samples',samples=('id:0012',),banks=('2',));r=approved(w,scope,future=True,existing=False)
    assert not w.start(r['id'],r['selection_token'])
    w.observe([source('2'),source('3',sample='id:B'),source('4',bank='4'),source('5',exp='other')])
    assert [j['source']['id'] for j in w.state['jobs'].values()]==['2']
    w.observe([source('2')]);assert len(w.state['jobs'])==1


def test_same_source_revision_and_recipe_is_idempotent_but_distinct_acquisitions_are_not(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1',checksum='samebytes'),source('2',checksum='samebytes')]);r=approved(w,RefinementScope('exp'))
    w.start(r['id'],r['selection_token']);w.start(r['id'],r['selection_token']);assert len(w.state['jobs'])==2
    while w.run_next():pass
    w.start(r['id'],r['selection_token']);assert w.run_next() is None
    assert all(j['attempts']==1 for j in w.state['jobs'].values())


def test_approval_rejects_changed_scope_and_changed_draft_generation(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1')]);s=RefinementScope('exp');d=w.prepare('exp','1','background');_,token=w.selection(s)
    w.observe([source('2')])
    with pytest.raises(ValueError,match='scan set changed'):w.approve('exp',d['generation'],s,existing=True,future=False,selection_token=token,approver='test')
    w.discard_draft('exp');_,token=w.selection(s)
    with pytest.raises(ValueError,match='draft changed'):w.approve('exp',d['generation'],s,existing=True,future=False,selection_token=token,approver='test')


def test_restart_holds_interrupted_local_work_and_never_silently_repeats(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1')]);r=approved(w,RefinementScope('exp'));job=w.start(r['id'],r['selection_token'])[0]
    w.state['jobs'][job['id']].update(status='running',attempts=1);w.save()
    restored=RefinementWorkspace(tmp_path,history_id='history',runtime=rt)
    assert restored.state['jobs'][job['id']]['status']=='interrupted'
    assert restored.run_next() is None
    restored.retry(job['id']);restored.run_next();assert restored.state['jobs'][job['id']]['attempts']==2


def test_pause_prevents_new_enrollment_and_queued_execution(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1')]);r=approved(w,RefinementScope('exp'),future=True);w.start(r['id'],r['selection_token']);w.pause('exp')
    w.observe([source('2')]);assert len(w.state['jobs'])==1 and w.run_next() is None
    w.resume_queue('exp');assert w.run_next()['status']=='completed'


def test_changed_models_need_a_recipe_review_not_an_automatic_llm_call(tmp_path):
    w,rt=create(tmp_path);w.observe([source('1')]);r=approved(w,RefinementScope('exp'));w.start(r['id'],r['selection_token']);rt.response_status='needs_review'
    assert w.run_next()['status']=='needs_recipe'
    with pytest.raises(ValueError,match='new recipe'):w.retry(next(iter(w.state['jobs'])))
    assert len([c for c in rt.calls if c[0]=='run'])==1


def test_published_source_prefers_explicit_publication_and_hashes_its_actual_bytes(tmp_path):
    raw=tmp_path/'Technical/model.gpx';raw.parent.mkdir();raw.write_bytes(b'internal')
    pub=tmp_path/'gpx/02_Main.gpx';pub.parent.mkdir();pub.write_bytes(b'enriched publication')
    project=dict(path='Technical/model.gpx',source_path='Technical/model.gpx',collection_path='gpx/02_Main.gpx',collection_name='02_Main.gpx',sha256=file_sha(raw),published_sha256=file_sha(pub),status='accepted',stage='main_phase_anchor')
    record=SimpleNamespace(uid='record',name='run',local_output_dir=tmp_path,output_dataset_ids={'gpx_projects':'collection'},inputs=None)
    payload=dict(summary={'observables':{'source_gpx_sha256':file_sha(raw)},'gpx_projects':[project]})
    item=source_from_record(record,payload,dict(instrument='PG3',ipts='IPTS-1',run_number=1,metadata={'sample_id':'0012'}),'history')
    assert item['source_path']==str(pub) and item['source_sha256']==file_sha(pub) and item['sample_key']=='id:0012'
    pub.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum changed'):source_from_record(record,payload,{},'history')


def test_ambiguous_or_stale_model_index_cannot_fall_back_to_arbitrary_project(tmp_path):
    rows=[dict(status='accepted',stage='final_refinement',sha256='a'),dict(status='accepted',stage='final_refinement',sha256='b')]
    with pytest.raises(ValueError,match='Several accepted'):select_published_project({'gpx_projects':rows},{})
    with pytest.raises(ValueError,match='does not identify'):select_published_project({'gpx_projects':rows,'observables':{'source_gpx_sha256':'missing'}},{})
    assert published_project_path(tmp_path,{'path':'../outside.gpx'}) is None


def test_local_runtime_captures_immutable_original_and_holds_changed_input(tmp_path):
    original=tmp_path/'incoming.gpx';original.write_bytes(b'original')
    rt=LocalRefinementRuntime(tmp_path/'runtime',python='unused');s=dict(source_path=str(original),source_sha256=file_sha(original));capture=rt.resolve(s)
    original.write_bytes(b'revised')
    assert Path(capture['source_path']).read_bytes()==b'original'
    with pytest.raises(ValueError,match='changed'):rt.resolve(s)
    view=tmp_path/'view.json';view.write_text('{}');result={'files':{'view.json':file_sha(view)}};verify_result_files(tmp_path,result);view.write_text('{"tampered":true}')
    with pytest.raises(ValueError,match='changed'):verify_result_files(tmp_path,result)
