import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from scripts.auto_background_points import estimate_background
from scripts import gsas_main_phase_refiner as refiner_mod
from scripts.gsas_main_phase_refiner import (
    GSASDataExtractor,
    GSASMainPhaseRefiner,
    RefinementResults,
    normalize_excluded_regions,
    normalize_background_config,
    parse_gsas_lst,
)


def _mk_results(rwp: float) -> RefinementResults:
    return RefinementResults(
        success=True,
        rwp=rwp,
        chi2=1.0,
        scale=1.0,
        background_params={},
        cell_params={},
        convergence_cycles=1,
    )


class _FakeProject:
    def __init__(self, behavior):
        self.behavior = list(behavior)
        self.data = {"Controls": {"data": {"max cyc": 5}}}
        self.refine_calls = 0

    def refine(self):
        action = self.behavior[self.refine_calls]
        self.refine_calls += 1
        if isinstance(action, Exception):
            raise action
        if action:
            print(action)


class RefineStageCellTests(unittest.TestCase):
    def _make_refiner(self, project):
        refiner = GSASMainPhaseRefiner.__new__(GSASMainPhaseRefiner)
        refiner.project = project
        refiner._enable_scale_refinement = MagicMock()
        refiner._enable_background_refinement = MagicMock()
        refiner._enable_cell_refinement = MagicMock()
        refiner._disable_cell_refinement = MagicMock()
        refiner._read_cell_from_data = MagicMock(return_value=(1.0, 2.0, 3.0, 90.0, 90.0, 90.0))
        refiner._write_cell_to_data = MagicMock()
        return refiner

    def test_refine_stage_cell_retries_on_silent_metric_failure(self):
        project = _FakeProject([
            "Invalid cell metric tensor",
            "",
        ])
        refiner = self._make_refiner(project)
        refiner._extract_refinement_results = MagicMock(side_effect=[
            _mk_results(100.0),
            _mk_results(12.3),
        ])

        result = refiner.refine_stage_cell()

        self.assertAlmostEqual(result.rwp, 12.3)
        self.assertEqual(project.refine_calls, 2)
        refiner._write_cell_to_data.assert_called_once_with(
            (1.0, 2.0, 3.0, 90.0, 90.0, 90.0),
            perturb_pct=0.0,
            seed=1,
        )
        self.assertEqual(project.data["Controls"]["data"]["max cyc"], 5)

    def test_refine_stage_cell_falls_back_on_non_metric_exception(self):
        project = _FakeProject([
            RuntimeError("ordinary refinement failure"),
            "",
        ])
        refiner = self._make_refiner(project)
        refiner._extract_refinement_results = MagicMock(return_value=_mk_results(22.0))

        result = refiner.refine_stage_cell()

        self.assertAlmostEqual(result.rwp, 22.0)
        self.assertEqual(project.refine_calls, 2)
        self.assertIn("ordinary refinement failure", result.error_message)


class ParseLstFallbackTests(unittest.TestCase):
    def test_parse_gsas_lst_single_phase_fraction_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            lst = Path(td) / "single.lst"
            lst.write_text(
                " Phase: TbSSL_Main in histogram: PWDR test\n"
                "  Phase fraction : 1 Refine? False\n"
            )

            result = parse_gsas_lst(lst, "PWDR test")

            self.assertEqual(result["TbSSL_Main"]["phase_fraction_pct"], 100.0)
            self.assertEqual(result["TbSSL_Main"]["weight_fraction_pct"], 100.0)


class ExcludedRegionNormalizationTests(unittest.TestCase):
    def test_normalize_excluded_regions_sorts_clips_and_merges(self):
        result = normalize_excluded_regions(
            [(12.0, 10.0), (11.5, 14.0), (30.0, 40.0), (-5.0, 2.0)],
            lo=0.0,
            hi=35.0,
        )

        self.assertEqual(result, [[0.0, 2.0], [10.0, 14.0], [30.0, 35.0]])


class BackgroundConfigTests(unittest.TestCase):
    def test_normalize_background_config_includes_mode_and_overrides(self):
        cfg = normalize_background_config(
            {"mode": "auto_fixed_points", "terms": 9},
            bg_type="log interpolate",
            bg_coeffs=[1.0, 2.0],
        )

        self.assertEqual(cfg["mode"], "auto_fixed_points")
        self.assertEqual(cfg["type"], "log interpolate")
        self.assertEqual(cfg["terms"], 9)
        self.assertEqual(cfg["coeffs"], [1.0, 2.0])

    def test_configure_background_auto_fixed_points_uses_gsas_fixed_points(self):
        class _FakeHistogram:
            def __init__(self):
                self.payloads = []
                self._background = [
                    ["chebyschev-1", False, 3, 0.0, 0.0, 0.0],
                    {"nDebye": 0, "debyeTerms": [], "nPeaks": 0, "peaksList": [], "background PWDR": ["", 1.0, False]},
                ]

            def set_refinements(self, payload):
                self.payloads.append(payload)
                bg = payload.get("Background")
                if isinstance(bg, dict):
                    if "type" in bg:
                        self._background[0][0] = bg["type"]
                    if "no. coeffs" in bg:
                        n = int(bg["no. coeffs"])
                        coeffs = self._background[0][3:]
                        coeffs = (coeffs + [0.0] * n)[:n]
                        self._background[0] = [self._background[0][0], False, n, *coeffs]
                    if "FixedPoints" in bg:
                        self._background[1]["FixedPoints"] = list(bg["FixedPoints"])

            def clear_refinements(self, payload):
                self.payloads.append({"clear": payload})
                self._background[1].pop("FixedPoints", None)

            def getHistEntryValue(self, path):
                if path == ["Background"]:
                    return self._background
                raise KeyError(path)

            def setHistEntryValue(self, path, value):
                if path == ["Background"]:
                    self._background = value
                    return
                raise KeyError(path)

            def getdata(self, kind):
                if kind == "x":
                    return np.ma.array(np.linspace(5.0, 120.0, 500))
                if kind == "yobs":
                    x = np.linspace(5.0, 120.0, 500)
                    y = 50.0 + 0.1 * x + 20.0 * np.exp(-0.5 * ((x - 40.0) / 0.4) ** 2)
                    return np.ma.array(y)
                raise KeyError(kind)

        refiner = GSASMainPhaseRefiner.__new__(GSASMainPhaseRefiner)
        refiner.histogram = _FakeHistogram()
        refiner.instrument_type = "CW"

        refiner._configure_background(
            background_config={
                "mode": "auto_fixed_points",
                "type": "chebyschev-1",
                "terms": 6,
            }
        )

        bg_payloads = [p["Background"] for p in refiner.histogram.payloads if "Background" in p]
        self.assertEqual(bg_payloads[0]["type"], "chebyschev-1")
        self.assertEqual(bg_payloads[0]["no. coeffs"], 6)
        self.assertTrue(any("FixedPoints" in payload for payload in bg_payloads))
        fixed_payload = next(payload for payload in bg_payloads if "FixedPoints" in payload)
        self.assertTrue(fixed_payload["fit fixed points"])
        self.assertGreater(len(fixed_payload["FixedPoints"]), 10)


class LimitPreservationTests(unittest.TestCase):
    def test_set_limits_to_data_preserves_existing_user_window(self):
        class _FakeHistogram:
            def __init__(self):
                self._lower = 12.5
                self._upper = 98.5
                self._excluded = [[20.0, 22.0]]
                self.refinement_payloads = []

            def Limits(self, which):
                return self._lower if which == "lower" else self._upper

            def Excluded(self, value=None):
                if value is None:
                    return list(self._excluded)
                self._excluded = list(value)

            def set_refinements(self, payload):
                self.refinement_payloads.append(payload)

            def getdata(self, kind):
                if kind != "x":
                    raise AssertionError(f"unexpected data kind: {kind}")
                return [0.0, 100.0]

        refiner = GSASMainPhaseRefiner.__new__(GSASMainPhaseRefiner)
        refiner.histogram = _FakeHistogram()
        refiner.project = MagicMock()

        with patch.object(refiner_mod, "apply_safe_limits", autospec=True) as safe_limits:
            refiner._set_limits_to_data()

        self.assertEqual(
            refiner.histogram.refinement_payloads[-1],
            {"Limits": {"low": 12.5, "high": 98.5}},
        )
        self.assertEqual(refiner.histogram.Excluded(), [[20.0, 22.0]])
        safe_limits.assert_called_once_with(refiner.project)


class DataExtractorAlignmentTests(unittest.TestCase):
    def test_get_all_arrays_uses_shared_mask_for_excluded_regions(self):
        class _FakeHistogram:
            def __init__(self):
                self._mask = np.array([False, True, False, True, False])

            def getdata(self, kind):
                if kind == "x":
                    return np.ma.array([1, 2, 3, 4, 5], mask=self._mask)
                if kind == "Q":
                    return np.ma.array([10, 20, 30, 40, 50], mask=self._mask)
                if kind == "d":
                    return np.ma.array([0.1, 0.2, 0.3, 0.4, 0.5], mask=self._mask)
                if kind == "yobs":
                    return np.ma.array([100, 200, 300, 400, 500], mask=[False] * 5)
                if kind == "ycalc":
                    return np.ma.array([90, 190, 290, 390, 490], mask=self._mask)
                if kind == "background":
                    return np.ma.array([5, 5, 5, 5, 5], mask=self._mask)
                raise KeyError(kind)

        arrays = GSASDataExtractor.get_all_arrays(_FakeHistogram())

        self.assertTrue(np.array_equal(arrays["x_native"], np.array([1, 3, 5])))
        self.assertTrue(np.array_equal(arrays["Q"], np.array([10, 30, 50])))
        self.assertTrue(np.array_equal(arrays["yobs"], np.array([100, 300, 500])))
        self.assertTrue(np.array_equal(arrays["ycalc"], np.array([90, 290, 490])))
        self.assertTrue(np.array_equal(arrays["residual"], np.array([10, 10, 10])))


class AutoBackgroundPointTests(unittest.TestCase):
    def test_estimate_background_keeps_positive_tail_support_with_sparse_zero_tail(self):
        n = 1200
        x = np.linspace(3000.0, 463000.0, n)
        y = 8.0 + 0.00002 * (x.max() - x)
        y += 12.0 * np.exp(-0.5 * ((x - 75000.0) / 2500.0) ** 2)
        y += 6.0 * np.exp(-0.5 * ((x - 220000.0) / 8000.0) ** 2)
        tail_mask = x > (x.min() + 0.8 * (x.max() - x.min()))
        y[tail_mask] = 0.0
        sparse_tail = np.flatnonzero(tail_mask)[::18]
        y[sparse_tail] = np.linspace(0.4, 2.0, sparse_tail.size)

        background, points, _resolved = estimate_background(x, y)

        self.assertGreater(float(np.median(background[tail_mask])), 0.05)
        tail_points = points[points[:, 0] >= x[tail_mask][0]]
        self.assertGreater(tail_points.shape[0], 5)
        self.assertGreater(float(np.median(tail_points[:, 1])), 0.05)


if __name__ == "__main__":
    unittest.main()
