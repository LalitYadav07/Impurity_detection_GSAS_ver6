import threading
from types import SimpleNamespace
import pytest
from gpx_recipe.symmetry.campaign import Campaign
from radar_pd_nova.model_review_recovery import SelectionRecovery, ReviewSelectionChanged, campaign_recency,AcquisitionReview,acquisition_index
from radar_pd_nova.symmetry_controller import SymmetryMixin


def test_selection_cancels_only_obsolete_read_not_the_campaign_driver():
    event=threading.Event();calls=[]
    native=SimpleNamespace(recover=lambda key,directory:calls.append(key) or {'state':'running'})
    view=SelectionRecovery(native,event)
    assert view.recover('first',None)=={'state':'running'}
    event.set()
    with pytest.raises(ReviewSelectionChanged):view.recover('obsolete',None)
    assert calls==['first'] and not hasattr(view,'submit') and not hasattr(view,'cancel')
    assert native.recover('still-owned-by-driver',None)=={'state':'running'}


def test_selection_while_loading_requests_latest_review_without_pausing_jobs():
    event=threading.Event();state=SimpleNamespace(sx_campaign='new',rf_experiment='PG3',sx_result_scope='experiment',sx_results_current=True)
    app=SymmetryMixin();app.server=SimpleNamespace(state=state)
    app._symmetry_loading=True;app._symmetry_loading_identity=('old','PG3','experiment')
    app._symmetry_review_cancel=event;app._symmetry_refresh_pending=False
    app._symmetry_campaign_changed()
    assert event.is_set() and app._symmetry_refresh_pending and not state.sx_results_current


def test_recovery_progress_counts_archives_before_reading_and_stops_after_cancellation():
    event=threading.Event();calls=[];progress=[]
    native=SimpleNamespace(state={'members':{'one':'batch-a','two':'batch-a','three':'batch-b'}},
        recover=lambda key,directory:calls.append(key) or {'state':'running'})
    view=SelectionRecovery(native,event,progress=lambda count:progress.append((count,len(calls))))
    for key in ['one','two','three']:
        assert view.recover(key,None)=={'state':'running'}
    assert progress==[(1,0),(2,2)] and calls==['one','two','three']
    event.set()
    with pytest.raises(ReviewSelectionChanged):view.recover('four',None)
    assert progress==[(1,0),(2,2)]


def test_latest_saved_campaign_is_chosen_by_checkpoint_clock_not_random_id(tmp_path):
    older=Campaign(tmp_path/'campaign-ffffffffffff');older.set_setting('checkpoint_updated_utc','2026-10-30T12:00:00')
    newer=Campaign(tmp_path/'campaign-000000000000');newer.set_setting('checkpoint_updated_utc','2026-10-31T12:00:00Z')
    paths=[older.root/'campaign.sqlite',newer.root/'campaign.sqlite']
    assert sorted(paths,key=campaign_recency)[-1]==paths[1]
    newer.set_setting('checkpoint_updated_utc','invalid')
    assert isinstance(campaign_recency(paths[1])[0],float)


def test_exact_scan_recovery_does_not_fetch_other_acquisitions_or_change_their_states(tmp_path):
    import json
    from gpx_recipe.symmetry.contracts import digest
    campaign=Campaign(tmp_path/'campaign');protocols={}
    with campaign.connect() as db:
        for name in ['cold','warm','another-experiment']:
            protocol=dict(acquisition_id=name,source={'sha256':name},parent_phase='Hematite')
            sha=digest(protocol);protocols[name]=sha
            metadata=dict(scan=name,bank=2,series='other' if name=='another-experiment' else 'NOM')
            db.execute('INSERT INTO protocols VALUES(?,?,?,?)',(sha,json.dumps(protocol),name,json.dumps(metadata)))
            for i in range(4):
                db.execute('INSERT INTO trials(id,protocol,candidate,seed,state,updated) VALUES(?,?,?,?,?,?)',
                    (name+str(i),sha,'candidate','seed','running',0.))
    options,index=acquisition_index(campaign,experiment='NOM')
    assert list(options)==['cold','warm'] and set(index)=={'cold','warm'}
    before={r['id']:r for r in campaign.trials()};calls=[]
    native=SimpleNamespace(recover=lambda key,directory:calls.append(key) or dict(state='running',handle={'key':key}))
    recovery=SelectionRecovery(native,threading.Event())
    view=AcquisitionReview(campaign,index['warm'])
    assert not any(hasattr(view,name) for name in ['tick','submit','cancel','verify_continuation'])
    received=view.recover(recovery)
    assert calls==['warm0','warm1','warm2','warm3']
    assert {r['id'] for r in received}==set(calls)
    for row in campaign.trials():
        if not row['id'].startswith('warm'):assert row==before[row['id']]
    assert len(campaign.trials())==12


def test_selected_review_reuses_original_collection_and_does_not_trust_navigation_as_fit_evidence(tmp_path,monkeypatch):
    import json
    campaign=Campaign(tmp_path/'campaign')
    with campaign.connect() as db:
        for uid,protocol in [('selected','one'),('other','two')]:
            db.execute('INSERT INTO trials(id,protocol,candidate,seed,state,updated) VALUES(?,?,?,?,?,?)',
                (uid,protocol,'candidate','seed','running',0.))
    path=campaign.root/'trials/selected';path.mkdir(parents=True);(path/'result.json').write_text('{}')
    collected=[];monkeypatch.setattr(campaign,'collect',lambda row:collected.append(row['id']))
    transport=SimpleNamespace(recover=lambda *args:pytest.fail('A present artifact must use the original integrity collector'))
    AcquisitionReview(campaign,{'one'}).recover(transport)
    assert collected==['selected']
    assert all(row['state']=='running' for row in campaign.trials())
    with pytest.raises(ValueError,match='exact acquisition'):AcquisitionReview(campaign,set())


def test_changed_acquisition_index_fails_before_any_result_download(tmp_path):
    import json
    campaign=Campaign(tmp_path/'campaign')
    with campaign.connect() as db:
        db.execute('INSERT INTO protocols VALUES(?,?,?,?)',('changed',json.dumps({'source':{'sha256':'x'}}),'x','{}'))
    with pytest.raises(ValueError,match='protocol changed'):acquisition_index(campaign)


def test_switching_scan_during_read_cancels_only_obsolete_review():
    event=threading.Event();state=SimpleNamespace(sx_source='new',sx_results_current=True)
    app=SymmetryMixin();app.server=SimpleNamespace(state=state)
    app._symmetry_loading=True;app._symmetry_recovery_source='old';app._symmetry_review_cancel=event
    app._symmetry_refresh_pending=False
    app._symmetry_source_changed()
    assert event.is_set() and app._symmetry_refresh_pending and not state.sx_results_current


def test_switching_scan_after_loaded_view_requests_its_own_verified_snapshot():
    state=SimpleNamespace(sx_source='new',sx_results_current=True)
    app=SymmetryMixin();app.server=SimpleNamespace(state=state)
    app._symmetry_loading=False;app._symmetry_review_acquisition='old'
    calls=[];app.refresh_symmetry=lambda:calls.append('new snapshot')
    app._symmetry_source_changed()
    assert calls==['new snapshot'] and not state.sx_results_current
