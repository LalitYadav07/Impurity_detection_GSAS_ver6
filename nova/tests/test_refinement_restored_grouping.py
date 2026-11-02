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
