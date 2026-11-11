"""An acknowledged run handle must not split one immutable recipe source."""
import copy
import json
from types import SimpleNamespace

import pytest

from radar_pd_nova.refinement_controller import published_refinement_source, refinement_experiment_options
from radar_pd_nova.refinement_sources import RefinementScope, digest, experiment_key, source_from_record
from radar_pd_nova.refinement_workspace import RefinementWorkspace
from test_refinement_integration import create, approved


PENDING = 'pending-original-run-handle'
JOB = 'acknowledged-galaxy-job'


def publication(tmp_path, uid=JOB, *, job=JOB, known_acquisition=True):
    project = dict(collection_name='03_Final_accepted_model.gpx', sha256='a'*64,
                   published_sha256='b'*64, status='accepted', stage='final_refinement')
    summary = dict(gpx_projects=[project])
    if known_acquisition:
        summary.update(acquisition=dict(instrument='NOM', run_number=138502,
                                        sample_id='sample', sample_name='NMC'),
                       provenance={'input': {'source_bank': 2}})
    record = SimpleNamespace(uid=uid, galaxy_job_id=job, name='User_named_NMC_run',
                             local_output_dir=tmp_path, inputs=None,
                             output_dataset_ids={'gpx_projects': 'immutable-collection'})
    return record, {'summary': summary}


def current_source(tmp_path, uid=JOB, **kwargs):
    record, payload = publication(tmp_path, uid, **kwargs)
    return source_from_record(record, payload, None, 'history')


def legacy_source(tmp_path, *, pending=PENDING, known_acquisition=True):
    old = current_source(tmp_path, pending, known_acquisition=known_acquisition)
    old.pop('fallback_record_uid', None)
    old['experiment'] = experiment_key('history', '', '', record_uid=pending)
    old['scan_key'] = digest([old['experiment'], None, old['run_number'] or pending, old['bank']])[:24]
    if not known_acquisition:
        old['sample_key'] = 'scan:' + pending
        old['run_id'] = pending
    return old


@pytest.mark.parametrize('known_acquisition', [True, False])
def test_new_pending_and_acknowledged_handles_share_all_fallback_scope_keys(tmp_path, known_acquisition):
    pending = current_source(tmp_path, PENDING, known_acquisition=known_acquisition)
    recovered = current_source(tmp_path, known_acquisition=known_acquisition)
    for key in ('id', 'experiment', 'scan_key', 'sample_key', 'run_id'):
        assert pending[key] == recovered[key]
    assert pending['record_uid'] == PENDING and recovered['record_uid'] == JOB


@pytest.mark.parametrize('already_drifted', [False, True])
def test_saved_recipe_scope_survives_recovery_and_repeated_refresh(tmp_path, already_drifted):
    w, rt = create(tmp_path/'workspace')
    old = legacy_source(tmp_path)
    w.observe([old])
    recipe = approved(w, RefinementScope(old['experiment']), future=True)
    original_job = w.start(recipe['id'], recipe['selection_token'])[0]
    frozen_jobs, frozen_recipes = copy.deepcopy(w.state['jobs']), copy.deepcopy(w.state['recipes'])
    incoming = current_source(tmp_path)
    if already_drifted:
        # Existing .14 catalogs may already have overwritten the mutable row.
        w.state['sources'][old['id']] = dict(incoming, observed_utc=old.get('observed_utc', 'old'))
    w.observe([incoming])
    recovered = w.state['sources'][old['id']]
    assert recovered['experiment'] == old['experiment']
    assert recovered['scan_key'] == old['scan_key']
    assert recovered['record_uid'] == JOB
    assert [s['id'] for s in w.sources(old['experiment'])] == [old['id']]
    assert refinement_experiment_options(w) == {old['experiment']: old['experiment_label']}
    assert w.state['jobs'] == frozen_jobs and w.state['recipes'] == frozen_recipes
    assert list(w.state['jobs']) == [original_job['id']]
    before_calls = list(rt.calls)
    w.observe([incoming])
    assert w.state['jobs'] == frozen_jobs and rt.calls == before_calls


def test_catalog_returns_the_saved_group_before_ui_selection(tmp_path):
    w, _ = create(tmp_path/'workspace')
    old = legacy_source(tmp_path)
    w.observe([old])
    record, payload = publication(tmp_path)
    (tmp_path/'gpx_index.json').write_text('{}')
    (tmp_path/'summary.json').write_text(json.dumps(payload['summary']))
    service = SimpleNamespace(_download_dataset=lambda *_: pytest.fail('No remote download is needed'))
    found = published_refinement_source(service, w, record, None, {})
    assert found['experiment'] == old['experiment']
    assert found['scan_key'] == old['scan_key']


def test_legacy_group_without_jobs_survives_saved_workspace_and_next_refresh(tmp_path):
    w, rt = create(tmp_path/'workspace')
    old = legacy_source(tmp_path, known_acquisition=False)
    w.observe([old])
    incoming = current_source(tmp_path, known_acquisition=False)
    w.observe([incoming])
    restored = RefinementWorkspace(w.root, history_id='history', runtime=rt)
    restored.observe([incoming])
    assert restored.state['sources'][old['id']]['experiment'] == old['experiment']
    assert restored.state['sources'][old['id']]['sample_key'] == old['sample_key']
    assert restored.state['jobs'] == {} and rt.calls == []


@pytest.mark.parametrize('change', [
    {'id': 'other-source'}, {'galaxy_job_id': 'other-job'},
    {'collection_id': 'other-collection'}, {'collection_name': 'other.gpx'},
    {'internal_source_sha256': 'c'*64}, {'published_sha256': 'c'*64},
    {'source_sha256': 'c'*64}, {'published_sha256': None},
    {'run_number': 138503}, {'bank': '3'}, {'sample_key': 'id:other'},
    {'metadata': {'product_id': 'SLICE_100'}},
])
def test_exact_publication_and_scope_guards_do_not_join_different_science(tmp_path, change):
    w, _ = create(tmp_path/'workspace')
    old = legacy_source(tmp_path)
    w.observe([old])
    incoming = dict(current_source(tmp_path), **change)
    assert w.canonical_source(incoming) == incoming


def test_true_metadata_enrichment_keeps_the_existing_earlier_result_guard(tmp_path):
    w, _ = create(tmp_path/'workspace')
    old = legacy_source(tmp_path)
    w.observe([old])
    w.state['jobs']['old'] = {'source': copy.deepcopy(old)}
    incoming = current_source(tmp_path)
    incoming.update(experiment=experiment_key('history', 'NOM', 'IPTS-21886'),
                    experiment_label='NOM · IPTS-21886')
    w.observe([incoming])
    labels = refinement_experiment_options(w)
    assert labels[incoming['experiment']] == 'NOM · IPTS-21886'
    assert labels[old['experiment']].startswith('Earlier refinement results')


def test_conflicting_saved_pending_groups_are_not_guessed(tmp_path):
    w, _ = create(tmp_path/'workspace')
    first, second = legacy_source(tmp_path), legacy_source(tmp_path, pending='pending-other-viewer')
    w.state['sources'][first['id']] = first
    w.state['jobs'] = {'first': {'source': first}, 'second': {'source': second}}
    incoming = current_source(tmp_path)
    assert w.canonical_source(incoming) == incoming


def test_changed_publication_keeps_old_jobs_in_their_original_group(tmp_path):
    w, _ = create(tmp_path/'workspace')
    old = legacy_source(tmp_path)
    w.observe([old])
    w.state['jobs']['finished'] = {'source': copy.deepcopy(old)}
    w.observe([current_source(tmp_path)])
    frozen_job = copy.deepcopy(w.state['jobs']['finished'])
    record, payload = publication(tmp_path)
    payload['summary']['gpx_projects'][0]['published_sha256'] = 'c'*64
    changed = source_from_record(record, payload, None, 'history')
    assert changed['id'] != old['id']
    w.observe([changed])
    assert w.state['jobs']['finished'] == frozen_job
    assert old['id'] in w.state['superseded_sources']
    assert w.sources(old['experiment']) == []
    assert [s['id'] for s in w.sources(changed['experiment'])] == [changed['id']]
    assert refinement_experiment_options(w)[old['experiment']].startswith('Earlier refinement results')


def test_same_label_distinct_jobs_are_not_one_publication(tmp_path):
    w, _ = create(tmp_path/'workspace')
    old = legacy_source(tmp_path)
    w.observe([old])
    second = current_source(tmp_path, 'different-job', job='different-job')
    w.observe([second])
    assert old['id'] != second['id'] and old['experiment'] != second['experiment']
    assert len(w.state['sources']) == 2
    assert {s['source_sha256'] for s in w.state['sources'].values()} == {'b'*64}


def hb2a_publication(tmp_path, *, experiment=1000, scan=1, name=None, acquisition=None):
    record, payload=publication(tmp_path,uid=f'hb2a-{experiment}-{scan}',known_acquisition=False,
                               job=f'hb2a-{experiment}-{scan}')
    record.name=name or f'IPTS-1_HB2A_EXP{experiment}_SCAN{scan:04d}_0123456789abcdef'
    payload['summary']['provenance']={'input':{'instrument':'HB2A','source_bank':1,
        'experiment_number':experiment,'scan_number':scan}}
    if acquisition:payload['summary']['acquisition']=acquisition
    return source_from_record(record,payload,None,'history')


def test_hb2a_saved_identity_uses_the_verified_experiment_and_scan(tmp_path):
    first=hb2a_publication(tmp_path)
    repeated_number=hb2a_publication(tmp_path,experiment=1001)
    assert first['experiment']==repeated_number['experiment']==experiment_key('history','HB2A','IPTS-1')
    assert first['run_number']==1 and first['metadata']['experiment_number']==1000
    assert first['scan_key']!=repeated_number['scan_key']
    assert first['sample_key']!=repeated_number['sample_key']
    assert first['scan_label']=='Exp 1000 · Scan 1 · bank 1'
    assert first['run_id']=='HB2A_EXP1000_SCAN0001'
    assert repeated_number['run_id']=='HB2A_EXP1001_SCAN0001'
    assert first['sample_label']=='Sample not recorded'


@pytest.mark.parametrize('name',[
    'IPTS-1_HB2A_EXP1001_SCAN0001_0123456789abcdef',
    'IPTS-1_HB2A_EXP1000_SCAN0002_0123456789abcdef',
    'IPTS-1_HB2A_EXP1000_SCAN0001_0123456789abcdef_unverified',
])
def test_hb2a_conflicting_run_names_do_not_invent_a_proposal(tmp_path,name):
    row=hb2a_publication(tmp_path,name=name)
    assert row['experiment']==experiment_key('history','','',record_uid='hb2a-1000-1')


def test_hb2a_published_sample_identity_does_not_require_the_generated_filename(tmp_path):
    row=hb2a_publication(tmp_path,name='User-labelled cold scan',acquisition={
        'instrument':'HB2A','ipts':'IPTS-1','run_number':1,'experiment_number':1000,
        'sample_id':'EXP1000:ID:3','sample_name':'Rare-earth silicate'})
    assert row['experiment_label']=='HB2A · IPTS-1'
    assert row['sample_key']=='id:EXP1000:ID:3'
    assert row['sample_label']=='EXP1000:ID:3 · Rare-earth silicate'
    assert row['scan_label']=='Exp 1000 · Scan 1 · bank 1'


def test_monitored_hb2a_scan_does_not_hide_another_experiments_same_scan_number(tmp_path):
    workspace,_=create(tmp_path/'workspace')
    watched=hb2a_publication(tmp_path,experiment=1000)
    watched['live_watch']=True
    recovered=hb2a_publication(tmp_path,experiment=1001)
    duplicate=dict(watched,id='alternate-publication',record_uid='alternate-job',
                   galaxy_job_id='alternate-job',live_watch=False)
    workspace.observe([watched,recovered,duplicate])
    assert {s['run_id'] for s in workspace.sources(watched['experiment'])}=={
        'HB2A_EXP1000_SCAN0001','HB2A_EXP1001_SCAN0001'}
    assert len(workspace.sources(watched['experiment']))==2
