import copy
import json
from pathlib import Path

import pytest

from gpx_recipe.symmetry.campaign import Campaign
from gpx_recipe.symmetry.compiler import file_sha
from gpx_recipe.symmetry.contracts import digest
from test_symmetry_contracts import make_protocol


class Transport:
    def __init__(self):self.jobs={};self.submissions=[];self.lose_ack=False
    def submit(self,key,directory):
        assert key not in self.jobs
        self.submissions.append(key);self.jobs[key]=dict(state='running',handle=dict(key=key))
        if self.lose_ack:self.lose_ack=False;raise OSError('Acknowledgment lost after remote acceptance')
        return dict(key=key)
    def recover(self,key,directory):return self.jobs.get(key,dict(state='absent'))
    def poll(self,handle,directory):return self.recover(handle['key'],directory)
    def cancel(self,handle,directory):self.jobs[handle['key']]=dict(state='cancelled');return True


@pytest.fixture
def setup(tmp_path,monkeypatch):
    monkeypatch.setattr('gpx_recipe.symmetry.compiler.package_signature',lambda:'c'*64)
    p=make_protocol(tmp_path/'inputs');p['seeds']=[dict(p['seeds'][0],id=v) for v in ['a','b','c']]
    return tmp_path,p,Campaign(tmp_path/'campaign'),Transport()


def finish(c,transport,key,status='valid'):
    root=c.root/'trials'/key;job=json.loads((root/'job.json').read_text());(root/'results').mkdir();artifact=root/'results/checkpoint.gpx';artifact.write_bytes(b'native-test-artifact')
    result=dict(schema='radar-symmetry-trial/v1',id=key,protocol_sha256=job['protocol_sha256'],status=status,rwp=8.,effects_verified=True,inputs_verified=True,audit=dict(passed=True),files={'results/checkpoint.gpx':file_sha(artifact)})
    (root/'result.json').write_text(json.dumps(result));transport.jobs[key]=dict(state='finished')


def test_lost_ack_restart_and_identical_arrivals_do_not_duplicate(setup):
    root,p,c,t=setup;ids=c.enroll(p,root/'inputs',approved_hash=digest(p),metadata=dict(scan='1',sample='A',bank=2))
    t.lose_ack=True;c.tick(t,slots=1);assert len(t.submissions)==1
    c=Campaign(root/'campaign');assert c.enroll(p,root/'inputs',approved_hash=digest(p))==ids
    c.tick(t,slots=1);assert len(t.submissions)==1;assert c.trials()[0]['state']=='running'
    finish(c,t,ids[0]);c.tick(t,slots=1);assert len(t.submissions)==2
    assert c.trials()[0]['state']=='terminal';assert c.summaries()[0]['received_count']==1


def test_unknown_remote_state_stays_visible_and_other_branches_continue(setup):
    root,p,c,t=setup;ids=c.enroll(p,root/'inputs',approved_hash=digest(p));t.lose_ack=True;c.tick(t,slots=2)
    t.jobs[ids[0]]=dict(state='unknown',reason='Remote lookup unavailable');c.tick(t,slots=2)
    assert c.trials()[0]['state']=='needs_recovery'
    assert len(t.submissions)==3;assert c.summaries()[0]['status']=='incomplete'
    assert t.submissions.count(ids[0])==1


def test_resume_rehashes_completed_artifacts_and_detects_receipt_identity_changes(setup):
    root,p,c,t=setup;ids=c.enroll(p,root/'inputs',approved_hash=digest(p));c.tick(t);finish(c,t,ids[0]);c.tick(t)
    (c.root/'trials'/ids[0]/'results/checkpoint.gpx').write_bytes(b'changed')
    Campaign(root/'campaign').tick(t)
    assert c.trials()[0]['state']=='needs_recovery';assert 'artifact changed' in c.trials()[0]['error']


def test_cancellation_never_cancels_other_jobs_or_hides_denominator(setup):
    root,p,c,t=setup;ids=c.enroll(p,root/'inputs',approved_hash=digest(p));c.tick(t,slots=2)
    c.cancel(ids[0],t);c.cancel(ids[2],t)
    assert [r['state'] for r in c.trials()]==['cancelled','running','cancelled']
    assert c.summaries()[0]['declared_count']==3
    assert c.summaries()[0]['missing']==ids


def test_new_source_revision_requires_new_approval_and_separate_trial_keys(setup):
    root,p,c,t=setup;ids=c.enroll(p,root/'inputs',approved_hash=digest(p));new=copy.deepcopy(p)
    (root/'inputs/source.gpx').write_bytes(b'new scan revision');new['source']['sha256']=file_sha(root/'inputs/source.gpx')
    with pytest.raises(ValueError,match='Approval'):c.enroll(new,root/'inputs',approved_hash=digest(p))
    other=c.enroll(new,root/'inputs',approved_hash=digest(new),metadata=dict(scan='2',sample='A',bank=2))
    assert set(ids).isdisjoint(other);assert len(c.trials())==6
    c.tick(t,slots=3);assert len(t.submissions)==3
    # The sealed first scan was not changed when a later input arrived.
    assert (c.root/'trials'/ids[0]/'source.gpx').read_bytes()==b'opaque-native-project'


def test_partial_import_retains_the_whole_approved_denominator(setup):
    root,p,c,t=setup;ids=c.enroll(p,root/'inputs',approved_hash=digest(p));finish(c,t,ids[0])
    imported=Campaign(root/'imported');imported.adopt_completed([c.root/'trials'/ids[0]])
    assert imported.summaries()[0]['declared_count']==3
    assert imported.summaries()[0]['received_count']==1
    assert imported.summaries()[0]['status']=='incomplete'


def test_result_json_cannot_change_the_score_after_collection(setup):
    root,p,c,t=setup;ids=c.enroll(p,root/'inputs',approved_hash=digest(p));finish(c,t,ids[0]);c.tick(t)
    path=c.root/'trials'/ids[0]/'result.json';d=json.loads(path.read_text());d['rwp']=.01;path.write_text(json.dumps(d))
    c.tick(t);assert c.trials()[0]['state']=='needs_recovery';assert 'receipt changed' in c.trials()[0]['error']
    for _ in range(2):
        c=Campaign(root/'campaign');c.tick(t)
        row=c.trials()[0]
        assert row['state']=='needs_recovery'
        assert json.loads(row['receipt'])['rwp']==8.
        assert 'receipt changed' in row['error']


def test_a_late_verified_receipt_resolves_unknown_transport_without_resubmission(setup):
    root,p,c,t=setup;ids=c.enroll(p,root/'inputs',approved_hash=digest(p));t.lose_ack=True;c.tick(t)
    t.jobs[ids[0]]=dict(state='unknown',reason='Remote lookup unavailable');c.tick(t)
    assert c.trials()[0]['state']=='needs_recovery'
    finish(c,t,ids[0]);c=Campaign(root/'campaign');c.tick(t)
    assert c.trials()[0]['state']=='terminal'
    assert t.submissions.count(ids[0])==1


def test_queue_continuation_uses_the_original_sealed_protocol(setup):
    root,p,c,t=setup;ids=c.enroll(p,root/'inputs',approved_hash=digest(p))
    assert c.verify_continuation('c'*64)==ids
    with pytest.raises(ValueError,match='older scientific implementation'):c.verify_continuation('d'*64)
    c.tick(t);finish(c,t,ids[0]);c.tick(t)
    assert c.verify_continuation('c'*64)==ids[1:]


def test_distinct_acquisitions_with_identical_gpx_bytes_remain_distinct(setup):
    root,p,c,t=setup
    a=dict(p,acquisition_id='run-1-bank-2');b=dict(p,acquisition_id='run-2-bank-2')
    first=c.enroll(a,root/'inputs',approved_hash=digest(a));second=c.enroll(b,root/'inputs',approved_hash=digest(b))
    assert set(first).isdisjoint(second)
    assert c.enroll(a,root/'inputs',approved_hash=digest(a))==first
    assert len(c.trials())==6
