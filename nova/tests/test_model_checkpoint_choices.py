"""A source menu must distinguish retained models for the same measured scan."""
import copy
from types import SimpleNamespace

from radar_pd_nova.symmetry_sources import start_choices


def workspace(sources):
    return SimpleNamespace(
        state={'sources':{s['id']:s for s in sources},'jobs':{}},
        sources=lambda experiment:[s for s in sources if s['experiment']==experiment],
    )


def source(uid, checksum):
    return dict(id=uid,experiment='experiment',scan_label='58827 · Bank 2 · C2/m',
        source_sha256=checksum,metadata={},model_checkpoint={'trial_id':uid},
        scientific_status='optimization_unresolved')


def test_model_choices_show_recorded_gpx_revisions_without_loading_projects():
    sources=[source('structural-model:'+str(i)*24,str(i)*64) for i in range(1,4)]
    before=copy.deepcopy(sources)
    choices=start_choices(workspace(sources),'experiment')
    assert len({c['title'] for c in choices})==3
    for row in choices:
        assert row['title'].startswith('Structural model · ')
        assert 'GPX '+row['source']['source_sha256'][:8] in row['title']
    assert [c['id'] for c in choices]==[s['id'] for s in sources]
    assert sources==before


def test_identical_or_colliding_revision_labels_keep_every_checkpoint_distinct():
    sources=[source('structural-model:'+str(i)*24,'a'*64) for i in range(1,4)]
    choices=start_choices(workspace(sources),'experiment')
    assert len(choices)==len({c['title'] for c in choices})==3
    labels={c['id']:c['title'] for c in choices}
    assert labels=={c['id']:c['title'] for c in start_choices(workspace(list(reversed(sources))),'experiment')}


def test_missing_revision_does_not_masquerade_as_a_gpx_hash():
    sources=[source('structural-model:'+str(i)*24,None) for i in range(1,3)]
    choices=start_choices(workspace(sources),'experiment')
    assert len({c['title'] for c in choices})==2
    assert all('GPX ' not in c['title'] for c in choices)
