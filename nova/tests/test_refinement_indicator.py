import copy

from radar_pd_nova.refinement_indicator import final_fit,compare_final_fits
from radar_pd_nova.transition_signals import BoundaryPolicy
from radar_pd_nova.transition_tracking import TransitionTrackingEngine
from test_scan_interpretation import scan


def fitted(run,rwp=8):
    return scan(run,final_rwp=rwp,rwp=rwp,analysis_mode="full",analysis_status="completed",
        candidate_library_id="same-library")


def test_only_explicit_completed_final_project_rwp_is_accepted():
    row=fitted(1)
    assert final_fit(row)["available"]
    for changed in [dict(final_rwp=None,rwp=20),dict(analysis_status="submitted"),dict(analysis_mode="rapid"),dict(result_status="failed")]:
        assert not final_fit({**row,**changed})["available"]
    bad=copy.deepcopy(row);bad["observables"]["source_gpx_sha256"]="different"
    assert not final_fit(bad)["available"]


def test_rwp_requires_both_thresholds_and_never_localizes_a_region():
    policy=BoundaryPolicy(selected=("rwp",))
    a=final_fit(fitted(1,8))
    assert compare_final_fits(a,final_fit(fitted(2,10)),policy)["detected"]
    assert not compare_final_fits(a,final_fit(fitted(2,8.9)),policy)["detected"]
    assert not compare_final_fits(final_fit(fitted(1,20)),final_fit(fitted(2,21.2)),policy)["detected"]
    report=TransitionTrackingEngine(policy=policy).evaluate([fitted(1,8),fitted(2,10)])
    signal=report["records"][1]["boundary_comparison"]["indicators"]["rwp"]
    assert signal["detected"] and not signal["localizable"] and not signal["regions"]
    assert report["records"][1]["boundary"]


def test_different_models_recipes_or_fit_windows_withhold_rwp_comparison():
    a=fitted(1);policy=BoundaryPolicy(selected=("rwp",))
    for field in ["configuration_id","candidate_library_id"]:
        b=fitted(2,12);b[field]="different"
        assert not compare_final_fits(final_fit(a),final_fit(b),policy)["available"]
    b=fitted(2,12);b["pattern"]["fit_domain"]["limits"]=[2,8]
    assert not compare_final_fits(final_fit(a),final_fit(b),policy)["available"]
    b=fitted(2,12);b["observables"]["context"]["recipe_key"]="different"
    assert not compare_final_fits(final_fit(a),final_fit(b),policy)["available"]


def test_small_missing_point_variations_are_reported_but_large_mask_changes_are_withheld():
    a=fitted(1);b=fitted(2,12);policy=BoundaryPolicy(selected=("rwp",))
    b["pattern"]["fit_valid"][10]=False
    result=compare_final_fits(final_fit(a),final_fit(b),policy)
    assert result["available"] and .98<result["mask_agreement"]<1
    b["pattern"]["fit_valid"][:500]=[False]*500
    assert not compare_final_fits(final_fit(a),final_fit(b),policy)["available"]


def test_rwp_arrives_after_raw_scan_and_updates_only_an_opted_in_rule():
    first=fitted(1,8);pending=fitted(2,None);pending["analysis_status"]="submitted"
    tracker=TransitionTrackingEngine(policy=BoundaryPolicy(selected=("rwp",)))
    tracker.append(first);before=tracker.append(pending)
    assert before["records"][1]["state_id"] is None
    after=tracker.append(fitted(2,10))
    assert after["records"][1]["boundary"] and after["refinement_evidence_updated"]
    native=TransitionTrackingEngine(policy=BoundaryPolicy())
    native.append(first);old=native.append(pending)
    new=native.append(fitted(2,10))
    assert old["records"]==new["records"]


def test_online_rwp_recurrence_uses_only_preceding_completed_results():
    tracker=TransitionTrackingEngine(policy=BoundaryPolicy(selected=("rwp",)))
    previous=[]
    for row in [fitted(1,8),fitted(2,10),fitted(3,8)]:
        report=tracker.append(row)
        assert report["records"][:-1]==previous
        previous=copy.deepcopy(report["records"])
    assert previous[2]["event"]=="returned"
    assert previous[2]["state_id"]==previous[0]["state_id"]


def test_new_arrivals_and_rule_edits_reuse_qualified_old_final_fit_receipts(monkeypatch):
    import radar_pd_nova.refinement_indicator as module
    calls=[];extract=module.final_fit
    def counted(row):
        calls.append(row["run_id"]);return extract(row)
    monkeypatch.setattr(module,"final_fit",counted)
    tracker=TransitionTrackingEngine(policy=BoundaryPolicy(selected=("rwp",)))
    rows=[fitted(1),fitted(2),fitted(3)]
    tracker.evaluate(rows[:2]);tracker.evaluate(rows)
    tracker.reclassify(BoundaryPolicy(selected=("rwp",),rwp_delta_pp=2))
    assert calls==[r["run_id"] for r in rows]
