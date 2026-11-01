import copy
import json
from pathlib import Path
import pytest
from gpx_recipe.symmetry.campaign import Campaign
from gpx_recipe.symmetry.compiler import file_sha,package_signature
from gpx_recipe.symmetry.contracts import digest,validate_protocol
from radar_pd_nova.model_ledger_recovery import restore_ledger,verify_continuation


@pytest.fixture
def saved(tmp_path):
    inputs=tmp_path/'inputs';inputs.mkdir();(inputs/'source.gpx').write_bytes(b'opaque test project')
    candidate=dict(id='mag:control',magnetic_space_group='P1',transformation_matrix=[[1,0,0],[0,1,0],[0,0,1]],origin_shift=[0,0,0])
    (inputs/'catalog.json').write_text(json.dumps(dict(schema='radar-symmetry-catalog/v1',parent_setting_sha256='a'*64,k=['0','0','0'],candidates=[candidate])))
    p=validate_protocol(dict(kind='magnetic',source=dict(path='source.gpx',sha256=file_sha(inputs/'source.gpx')),
        catalog=dict(path='catalog.json',sha256=file_sha(inputs/'catalog.json')),parent_phase='Parent',histogram='Bank',
        parent_setting_sha256='a'*64,magnetic_function_sha256='b'*64,implementation_sha256=package_signature(),
        k=['0','0','0'],candidates=[candidate],seeds=[dict(id='start'+str(i),axis='x',magnitude=3.) for i in range(3)],
        sites=[dict(label='Fe',form_factor='Fe+3',lande_g=2.)],
        policy=dict(cycles=12,timeout_seconds=60,maximum_trials=3,maximum_cell_multiplier=8,maximum_moment=6.,fit_band_percent=.25,maximum_mode=.5,adequate_rwp_percent=12.)))
    c=Campaign(tmp_path/'original',shared_inputs=True);sha=digest(p)
    c.enroll(p,inputs,approved_hash=sha,metadata=dict(series='NOM:IPTS-1',scan='run1'))
    rows=c.trials();c.set_state(rows[0]['id'],'running',handle=dict(key=rows[0]['id'],dispatch_key='saved-ack'))
    c.set_setting('paused',True)
    return c.snapshot(),{sha:inputs}


def test_bulk_restore_matches_the_original_finite_ledger_and_sealed_jobs(saved,tmp_path,monkeypatch):
    snapshot,inputs=saved
    original=Campaign(tmp_path/'old-reader',shared_inputs=True);original.restore(snapshot,inputs)
    recovered=Campaign(tmp_path/'new-reader',shared_inputs=True)
    monkeypatch.setattr(recovered,'set_state',lambda *a,**k:pytest.fail('Must not sync-commit once per start'))
    restore_ledger(recovered,snapshot,inputs)
    expected=original.snapshot();actual=recovered.snapshot();actual['settings'].pop('restore_in_progress')
    assert actual==expected
    for row in recovered.trials():
        assert (recovered.trial_directory(row)/'job.json').read_bytes()==(original.trial_directory(row)/'job.json').read_bytes()
        assert (recovered.trial_directory(row)/'worker.py').read_bytes()==(original.trial_directory(row)/'worker.py').read_bytes()


@pytest.mark.parametrize('corruption',['missing','duplicate','identity','state','handle','metadata','schema'])
def test_corrupt_snapshot_is_rejected_before_any_trial_is_restored(saved,tmp_path,corruption):
    snapshot,inputs=saved;snapshot=copy.deepcopy(snapshot)
    if corruption=='missing':snapshot['trials'].pop()
    elif corruption=='duplicate':snapshot['trials'].append(copy.deepcopy(snapshot['trials'][0]))
    elif corruption=='identity':snapshot['trials'][0]['seed']='different'
    elif corruption=='state':snapshot['trials'][0]['state']='invented'
    elif corruption=='handle':snapshot['trials'][0]['handle']='[]'
    elif corruption=='metadata':snapshot['protocols'][0]['metadata']='{"secret_path":"/elsewhere"}'
    else:snapshot['schema']='wrong'
    recovered=Campaign(tmp_path/'rejected',shared_inputs=True)
    with pytest.raises(ValueError):restore_ledger(recovered,snapshot,inputs)
    assert not recovered.trials() and not recovered.setting('restore_in_progress',False)


def test_interrupted_reconstruction_can_resume_without_duplicating_starts(saved,tmp_path,monkeypatch):
    snapshot,inputs=saved;recovered=Campaign(tmp_path/'interrupted',shared_inputs=True)
    enroll=recovered.enroll
    def lost(*args,**kwargs):enroll(*args,**kwargs);raise TimeoutError('viewer lost after input reconstruction')
    monkeypatch.setattr(recovered,'enroll',lost)
    with pytest.raises(TimeoutError):restore_ledger(recovered,snapshot,inputs)
    assert recovered.setting('restore_in_progress') and len(recovered.trials())==3
    monkeypatch.setattr(recovered,'enroll',enroll);restore_ledger(recovered,snapshot,inputs)
    assert len(recovered.trials())==3 and {r['id'] for r in recovered.trials()}=={r['id'] for r in snapshot['trials']}
    assert sum(row['state']=='running' for row in recovered.trials())==1


def test_interrupted_ledger_with_a_missing_declared_start_cannot_be_marked_restored(saved,tmp_path):
    snapshot,inputs=saved;recovered=Campaign(tmp_path/'incomplete',shared_inputs=True)
    restore_ledger(recovered,snapshot,inputs)
    with recovered.connect() as db:db.execute('DELETE FROM trials WHERE id=?',(snapshot['trials'][0]['id'],))
    with pytest.raises(ValueError,match='omits or changes declared starts'):restore_ledger(recovered,snapshot,inputs)
    assert recovered.setting('restore_in_progress') is True


def test_pending_plot_rows_need_no_per_start_job_reads_but_are_never_qualified(saved,tmp_path,monkeypatch):
    from radar_pd_nova import symmetry_workspace as module
    snapshot,inputs=saved;recovered=Campaign(tmp_path/'pending',shared_inputs=True)
    restore_ledger(recovered,snapshot,inputs)
    for row in recovered.trials():recovered.set_state(row['id'],'prepared')
    original=Path.read_text
    def read(path,*args,**kwargs):
        if path.name=='job.json':pytest.fail('Pending view must read the common ledger, not every job')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    rows,summaries=module.campaign_rows(recovered.root,experiment='NOM:IPTS-1')
    assert len(rows)==3 and len(summaries)==1
    assert {r['status'] for r in rows}=={'prepared'}
    assert all(not r['has_view'] and not r['has_gpx'] and r['rwp'] is None for r in rows)


def test_continuation_matches_core_and_rehashes_shared_inputs_once(saved,tmp_path,monkeypatch):
    import radar_pd_nova.model_ledger_recovery as module
    snapshot,inputs=saved;campaign=Campaign(tmp_path/'verify',shared_inputs=True);restore_ledger(campaign,snapshot,inputs)
    expected=campaign.verify_sealed_continuation();calls=[];verify=module.verify_inputs
    def counted(protocol,root):calls.append((digest(protocol),str(root)));return verify(protocol,root)
    monkeypatch.setattr(module,'verify_inputs',counted)
    assert verify_continuation(campaign)==expected and len(calls)==1
    assert campaign.setting('paused') is True


@pytest.mark.parametrize('change',['candidate','seed','worker','source','implementation','numeric_encoding'])
def test_continuation_rejects_a_changed_job_worker_or_shared_input(saved,tmp_path,change):
    snapshot,inputs=saved;campaign=Campaign(tmp_path/'changed',shared_inputs=True);restore_ledger(campaign,snapshot,inputs)
    row=campaign.trials()[-1];directory=campaign.trial_directory(row);job=json.loads((directory/'job.json').read_text())
    if change=='worker':(directory/'worker.py').write_text('changed entry point')
    elif change=='source':(campaign.root/'inputs'/row['protocol']/'source.gpx').write_bytes(b'changed source')
    else:
        if change=='candidate':job['candidate_id']='other'
        elif change=='seed':job['seed_id']='other'
        elif change=='implementation':job['package_sha256']='0'*64
        else:
            matrix=job['protocol']['candidates'][0]['transformation_matrix']
            matrix[0][0]=int(matrix[0][0]) if type(matrix[0][0]) is float else float(matrix[0][0])
        (directory/'job.json').write_text(json.dumps(job))
    with pytest.raises(ValueError):verify_continuation(campaign)


def test_input_changed_while_verifying_other_starts_is_rejected(saved,tmp_path,monkeypatch):
    import gpx_recipe.symmetry.compiler as compiler
    snapshot,inputs=saved;campaign=Campaign(tmp_path/'race',shared_inputs=True);restore_ledger(campaign,snapshot,inputs)
    count=[];original=compiler.file_sha;sha=campaign.trials()[0]['protocol']
    def reading(path):
        if Path(path).name=='worker.py' and Path(path).parent.parent==campaign.root/'trials':
            count.append(1)
            if len(count)==3:(campaign.root/'inputs'/sha/'source.gpx').write_bytes(b'changed during verification')
        return original(path)
    monkeypatch.setattr(compiler,'file_sha',reading)
    with pytest.raises(ValueError,match='changed during verification'):verify_continuation(campaign)
