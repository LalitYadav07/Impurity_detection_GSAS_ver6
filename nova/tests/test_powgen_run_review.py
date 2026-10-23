from radar_pd_nova.powgen_samples import include_run, run_role
from radar_pd_nova.powgen_watch import WatchedRun


def make_run(tmp_path, number, title):
    p = tmp_path / f"PG3_{number}.gsa"
    p.write_text(f"Sample Run: {number}\n# {title}\n")
    return WatchedRun(number, str(p), "f")


def test_background_and_cooling_need_explicit_inclusion(tmp_path):
    empty = make_run(tmp_path, 1, "Setup empty furnace without sample")
    ramp = make_run(tmp_path, 2, "Mixture Cooling to RT")
    ordinary = make_run(tmp_path, 3, "Temperature calibration at 300")
    assert run_role(empty)["run_role"] == "background"
    assert not include_run(empty)
    assert include_run(empty, include_background_runs=True)
    assert not include_run(ramp)
    assert include_run(ramp, include_ramp_runs=True)
    assert include_run(ordinary)
    assert not include_run(ordinary, excluded_run_numbers=(3,))


def test_missing_title_is_included_without_guessing(tmp_path):
    assert include_run(WatchedRun(1, str(tmp_path / "missing.gsa"), "f"))
