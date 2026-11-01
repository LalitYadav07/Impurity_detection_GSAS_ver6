import copy
import json

import pytest

from gpx_recipe.symmetry.campaign import Campaign
from gpx_recipe.symmetry.contracts import digest, iter_trials, trial_id
from radar_pd_nova import model_portfolio as module


def protocol(kind='magnetic', **extra):
    return dict(schema='radar-symmetry-protocol/v1', kind=kind,
        candidates=[dict(id='a', modes=['mode-a']), dict(id='b', modes=['mode-b'])],
        seeds=[dict(id='x'), dict(id='y', candidate_ids=['b']), dict(id='z')],
        policy=dict(fit_band_percent=1.0), **extra)


@pytest.mark.parametrize('kind',['magnetic','structural'])
def test_review_identity_matches_every_native_candidate_and_filtered_start(kind):
    p=protocol(kind)
    expected=[trial_id(p,candidate['id'],seed['id']) for candidate,seed in iter_trials(p)]
    assert module.declared_trial_ids(p,digest(p))==expected
    assert len(expected)==5
    assert len(set(expected))==5


def test_review_does_not_rehash_the_full_protocol_for_every_start(monkeypatch):
    p=protocol(payload='representative protocol details'*100)
    p['candidates']=[dict(id=f'candidate-{i}') for i in range(38)]
    p['seeds']=[dict(id=f'start-{i}') for i in range(24)]
    expected=[trial_id(p,candidate['id'],seed['id']) for candidate,seed in iter_trials(p)]
    calls=[]
    native_digest=module.digest
    def record(value):
        if value is p:calls.append(1)
        return native_digest(value)
    monkeypatch.setattr(module,'digest',record)
    assert module.declared_trial_ids(p,native_digest(p))==expected
    assert len(expected)==912 and len(calls)==1


def test_changed_protocol_or_native_identity_cannot_publish_a_false_denominator(monkeypatch):
    p=protocol()
    with pytest.raises(ValueError,match='declared identity'):
        module.declared_trial_ids(p,'a'*64)
    monkeypatch.setattr(module,'trial_id',lambda *args:'changed contract')
    with pytest.raises(ValueError,match='native trial identity contract changed'):
        module.declared_trial_ids(p,digest(p))
    future=dict(p,schema='radar-symmetry-protocol/v2')
    with pytest.raises(ValueError,match='native trial identity contract changed'):
        module.declared_trial_ids(future,digest(future))


def add(connect,p,*,maximum_rows=None):
    sha=digest(p)
    with connect() as db:
        db.execute('INSERT INTO protocols VALUES(?,?,?,?)',
            (sha,json.dumps(p),'source-sha',json.dumps(dict(series='IPTS-1'))))
        pairs=list(iter_trials(p))
        for candidate,seed in pairs if maximum_rows is None else pairs[:maximum_rows]:
            db.execute('INSERT INTO trials(id,protocol,candidate,seed,state,updated) VALUES(?,?,?,?,?,?)',
                (trial_id(p,candidate['id'],seed['id']),sha,candidate['id'],seed['id'],'prepared',0.0))
    return sha


def test_a_concurrent_acquisition_cannot_mix_protocols_and_starts_from_different_snapshots(tmp_path,monkeypatch):
    campaign=Campaign(tmp_path/'campaign')
    old=protocol();old_sha=add(campaign.connect,old)
    new=dict(copy.deepcopy(old),acquisition_id='new scan')
    original_connect=campaign.connect
    committed=[]

    class Connection:
        def __init__(self):self.inner=original_connect()
        def __enter__(self):self.inner.__enter__();return self
        def __exit__(self,*args):return self.inner.__exit__(*args)
        def execute(self,sql,*args):
            if 'FROM trials' in sql and not committed:
                committed.append(add(original_connect,new))
            return self.inner.execute(sql,*args)

    monkeypatch.setattr(campaign,'connect',Connection)
    protocols,trials,summaries=module.portfolio_snapshot(campaign)
    assert committed
    assert set(protocols)=={old_sha}
    assert {row['protocol'] for row in trials}=={old_sha}
    assert len(summaries)==1 and summaries[0]['declared_count']==5
    assert summaries[0]['pending_count']==5
    protocols,trials,summaries=module.portfolio_snapshot(campaign)
    assert len(protocols)==len(summaries)==2 and len(trials)==10


def test_summary_denominator_comes_from_the_approved_protocol_not_only_present_rows(tmp_path):
    campaign=Campaign(tmp_path/'campaign')
    add(campaign.connect,protocol(),maximum_rows=1)
    _,trials,summaries=module.portfolio_snapshot(campaign)
    assert len(trials)==1
    assert summaries[0]['declared_count']==5
    assert len(summaries[0]['missing'])==5
    assert summaries[0]['received_count']==0
