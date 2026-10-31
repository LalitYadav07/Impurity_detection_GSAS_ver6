import threading
from types import SimpleNamespace
import pytest
from gpx_recipe.symmetry.campaign import Campaign
from radar_pd_nova.model_review_recovery import SelectionRecovery, ReviewSelectionChanged, campaign_recency
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


def test_latest_saved_campaign_is_chosen_by_checkpoint_clock_not_random_id(tmp_path):
    older=Campaign(tmp_path/'campaign-ffffffffffff');older.set_setting('checkpoint_updated_utc','2026-10-30T12:00:00')
    newer=Campaign(tmp_path/'campaign-000000000000');newer.set_setting('checkpoint_updated_utc','2026-10-31T12:00:00Z')
    paths=[older.root/'campaign.sqlite',newer.root/'campaign.sqlite']
    assert sorted(paths,key=campaign_recency)[-1]==paths[1]
    newer.set_setting('checkpoint_updated_utc','invalid')
    assert isinstance(campaign_recency(paths[1])[0],float)
