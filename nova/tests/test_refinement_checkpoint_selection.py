"""A Rapid candidate is usable only after an exact, explicit selection."""
import copy
from types import SimpleNamespace

import pytest

from radar_pd_nova.refinement_sources import (
    CheckpointSelectionRequired, checkpoint_selection_key,
    select_published_project, source_from_record,
)


def projects():
    return [dict(stage='hypothesis_refinement', status='checkpoint',
        sha256=character * 64, published_sha256=published * 64,
        collection_name=name + '.gpx', collection_path='gpx/' + name + '.gpx')
        for character, published, name in [('a', 'c', 'rapid_cleanup'),
                                            ('b', 'd', 'rapid_background')]]


def test_rapid_requires_choice_and_uses_exact_key_independent_of_order():
    rows = projects()
    with pytest.raises(CheckpointSelectionRequired) as caught:
        select_published_project({'gpx_projects': rows}, {})
    assert {choice['value'] for choice in caught.value.choices} == {
        checkpoint_selection_key(row) for row in rows}
    key = checkpoint_selection_key(rows[1])
    assert select_published_project({'gpx_projects': rows[::-1]}, {},
                                    checkpoint_key=key) == rows[1]
    for field in ['sha256', 'published_sha256', 'collection_name', 'status', 'stage']:
        changed = copy.deepcopy(rows)
        changed[1][field] = 'replacement'
        with pytest.raises(ValueError, match='no longer available'):
            select_published_project({'gpx_projects': changed}, {}, checkpoint_key=key)


def test_failed_rollback_unknown_and_unverifiable_projects_are_not_choices():
    rows = projects()
    invalid = []
    for status in ['failed', 'rollback', 'unknown']:
        invalid.append(dict(rows[0], status=status, collection_name=status + '.gpx'))
    invalid += [dict(rows[0], published_sha256=None),
                dict(rows[0], sha256='bad'),
                dict(rows[0], collection_name='../elsewhere.gpx'),
                dict(rows[0], collection_path='gpx/elsewhere.gpx')]
    with pytest.raises(CheckpointSelectionRequired) as caught:
        select_published_project({'gpx_projects': rows + invalid}, {})
    assert len(caught.value.choices) == 2
    for row in invalid:
        with pytest.raises(ValueError):
            select_published_project({'gpx_projects': rows + invalid}, {},
                                      checkpoint_key=checkpoint_selection_key(row))


def test_explicit_choice_cannot_bypass_stale_model_hash_or_new_final():
    rows = projects()
    key = checkpoint_selection_key(rows[0])
    with pytest.raises(ValueError, match='does not identify'):
        select_published_project({'gpx_projects': rows,
            'observables': {'source_gpx_sha256': 'f' * 64}}, {}, checkpoint_key=key)
    accepted = dict(rows[0], stage='final_refinement', status='accepted')
    with pytest.raises(ValueError, match='no longer needs'):
        select_published_project({'gpx_projects': rows + [accepted]}, {}, checkpoint_key=key)


def test_explicit_source_keeps_candidate_review_and_no_unrelated_baseline_rwp(tmp_path):
    rows = projects()
    record = SimpleNamespace(uid='record', galaxy_job_id='job', name='Rapid run',
        local_output_dir=tmp_path, inputs=None,
        output_dataset_ids={'gpx_projects': 'collection'})
    payload = {'summary': {'gpx_projects': rows, 'final_rwp': 5,
        'scientific_status': 'accepted',
        'acquisition': {'instrument': 'NOM', 'ipts': 'IPTS-1', 'run_number': 42}}}
    source = source_from_record(record, payload, {'rwp': 6}, 'history',
        checkpoint_key=checkpoint_selection_key(rows[1]))
    assert source['project'] == rows[1]
    assert source['collection_name'] == 'rapid_background.gpx'
    assert source['source_sha256'] == 'd' * 64
    assert source['scientific_status'] == 'needs_review'
    assert source['base_rwp'] is None
    assert source['explicit_checkpoint'] is True
    assert 'not an accepted final model' in ' '.join(source['scientific_warnings'])
    other = source_from_record(record, payload, None, 'history',
        checkpoint_key=checkpoint_selection_key(rows[0]))
    assert other['id'] != source['id']
