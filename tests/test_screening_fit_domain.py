"""Screening scores must use the same measured bins as native refinement."""
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import gsas_main_phase_refiner as refiner
from gpx_recipe.fit_domain import native_powder_fit_mask


def histogram():
    x = np.arange(14, dtype=float)
    observed = np.ma.array(np.arange(14, dtype=float), mask=False)
    observed[6] = np.nan
    observed.mask[7] = True
    weights = np.ones(14)
    weights[8] = 0
    calculated = np.arange(14, dtype=float)
    calculated[[0, 1, 2, 4, 5, 12, 13]] = 1000
    data = {"data": [{}, [x, observed, weights, calculated]],
            "Limits": [[0, 13], [2.5, 10.5], [4, 5]]}

    def getdata(kind):
        if kind in {"x", "yobs", "ycalc"}:
            return raw_arrays[{"x": 0, "yobs": 1, "ycalc": 3}[kind]]
        if kind in {"Q", "d"}:
            # Native's inclusive upper bin is authoritative, even when a
            # derived coordinate getter applies a strict coordinate limit.
            return np.ma.array(x + 1, mask=x > 10.5)
        raise KeyError(kind)

    raw_arrays = data["data"][1]
    return SimpleNamespace(data=data, getdata=getdata)


def test_fitted_arrays_match_native_domain_and_keep_upper_endpoint_bin():
    hist = histogram()
    arrays = refiner.GSASDataExtractor.get_fitted_arrays(hist)
    raw = hist.data["data"][1]
    expected = native_powder_fit_mask(raw[0], [2.5, 10.5], [[4, 5]])
    expected &= np.isfinite(raw[1].data) & ~raw[1].mask & (raw[2] > 0)
    np.testing.assert_array_equal(arrays["x_native"], raw[0][expected])
    np.testing.assert_array_equal(arrays["x_native"], [3, 9, 10, 11])
    np.testing.assert_array_equal(arrays["Q"], [4, 10, 11, 12])
    np.testing.assert_array_equal(arrays["residual"], np.zeros(4))
    assert refiner._safe_pearson(arrays["yobs"], arrays["ycalc"]) == pytest.approx(1)
    # This is the previous calculation; out-of-domain placeholders reverse it.
    assert refiner._safe_pearson(raw[1], raw[3]) < 0


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_calculation_in_fitted_domain_fails_closed(bad):
    hist = histogram()
    hist.data["data"][1][3][9] = bad
    with pytest.raises(ValueError, match="missing calculated values"):
        refiner.GSASDataExtractor.get_fitted_arrays(hist)


def test_masked_calculation_in_fitted_domain_fails_closed():
    hist = histogram()
    hist.data["data"][1][3] = np.ma.array(hist.data["data"][1][3], mask=False)
    hist.data["data"][1][3].mask[9] = True
    with pytest.raises(ValueError, match="masked calculated values"):
        refiner.GSASDataExtractor.get_fitted_arrays(hist)


def test_pearson_refinement_scores_only_native_fitted_bins(monkeypatch, tmp_path):
    hist = histogram()
    hist.set_refinements = lambda *args, **kwargs: None
    hist.clear_refinements = lambda *args, **kwargs: None
    phase = SimpleNamespace(name="candidate", set_HAP_refinements=lambda *a, **kw: None,
                            HAPvalue=lambda *a, **kw: None,
                            set_refinements=lambda *a, **kw: None)
    project = SimpleNamespace(data={"Controls": {"data": {}}},
                              phases=lambda: [phase], do_refinements=lambda *a: None)
    manager = SimpleNamespace(project=project, main_histogram=hist, main_phase=phase,
                              create_project=lambda **kw: True,
                              add_histogram=lambda *a, **kw: True,
                              add_phase_from_cif=lambda *a: True)
    import gsas_core_infrastructure
    monkeypatch.setattr(gsas_core_infrastructure, "GSASProjectManager", lambda *a: manager)
    monkeypatch.setattr(refiner, "_init_gsas_process", lambda: None)
    monkeypatch.setattr(refiner, "apply_safe_limits", lambda *a: None)
    monkeypatch.setattr(refiner, "_screening_fixed_model", lambda *a: ())
    result = refiner.compute_gsas_pearson_for_cif(
        "pattern", "profile", None, "candidate.cif", str(tmp_path), None, None,
        "screen", cell_refine_min_r=2, export_refined_cif=False, return_diagnostics=True)
    assert result["r_scale"] == pytest.approx(1)
    assert result["pearson"] == pytest.approx(1)
    assert result["cell_refined"] is False


@pytest.mark.parametrize("kind", ["PNC", "PNT"])
def test_fitted_observation_count_matches_native_refinement(tmp_path, kind):
    """Run native GSAS-II with between-bin limits and an internal exclusion."""
    gsas = pytest.importorskip("GSASII.GSASIIscriptable")
    parameters = ("Lam:1.54\nZero:0\nU:.12\nV:0\nW:.08\nX:0\nY:0\nZ:0\nSH/L:.002\nPolariz.:0\nAzimuth:0\n"
                  if kind == "PNC" else
                  "difC:6200\ndifA:2\ndifB:.4\nZero:1\n2-theta:105\nfltPath:15.5\nalpha:.99\n"
                  "beta-0:.02\nbeta-1:.03\nbeta-q:0\nsig-0:1\nsig-1:66\nsig-2:0\nsig-q:0\nX:0\nY:0\nZ:0\nAzimuth:0\n")
    instrument = tmp_path / "profile.instprm"
    instrument.write_text("#GSAS-II instrument parameter file\nType:" + kind + "\nBank:1\n" + parameters)
    project = gsas.G2Project(newgpx=str(tmp_path / "synthetic.gpx"))
    phase = project.add_phase(phasename="Synthetic Si", spacegroup="P 1",
                              cell=[4, 4, 4, 90, 90, 90])
    phase.add_atom(0, 0, 0, "Si", "Si1")
    lower, upper = (12, 125) if kind == "PNC" else (3, 40)
    hist = project.add_simulated_powder_histogram("synthetic", str(instrument),
                                                lower, upper, Npoints=201, phases=[phase])
    raw = hist.data["data"][1]
    x = np.asarray(raw[0]).copy()
    raw[1][:] = 1
    raw[2][:] = 4
    # The upper limit is between bins; native includes the following bin.
    hist.data["Limits"] = [[float(x[0]), float(x[-1])],
                           [float((x[10] + x[11]) / 2), float((x[170] + x[171]) / 2)],
                           [float(x[40]), float(x[70])]]
    project.data["Controls"]["data"]["max cyc"] = 0
    project.do_refinements([{"set": {}}])
    arrays = refiner.GSASDataExtractor.get_fitted_arrays(hist)
    listing = (tmp_path / "synthetic.lst").read_text()
    import re
    counts = re.findall(r"No\. of observations: (\d+)", listing)
    assert counts, listing[-2500:]
    assert len(arrays["yobs"]) == int(counts[-1]) == 130
    assert arrays["x_native"][-1] == x[171]
