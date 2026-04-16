import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from scripts.gsas_main_phase_refiner import GSASMainPhaseRefiner, RefinementResults, parse_gsas_lst


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


if __name__ == "__main__":
    unittest.main()
