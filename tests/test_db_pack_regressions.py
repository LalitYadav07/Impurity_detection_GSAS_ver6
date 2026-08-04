import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

import yaml
from pymatgen.core import Lattice, Structure

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.aniso_db_loader import CatalogPaths, DBLoader, build_mask
from scripts.config_builder import build_pipeline_config
from scripts.db_pack import build_db_config, get_db_pack_layout
from scripts.db_pack_builder import _emit_progress, build_mini_db_pack
from scripts.gsas_complete_pipeline_nomain import BenchTimer, UnifiedPipeline, _crop_native_arrays_by_q


def _write_catalog(path: Path, phase_id: str) -> None:
    hi, lo = build_mask(["Li", "O"])
    path.write_text(
        "id,pretty_formula,space_group,SG_symbol,elements_mask_hi,elements_mask_lo,npz,n_reflections\n"
        f"{phase_id},LiO,1,P1,{int(hi)},{int(lo)},phases/{phase_id}.npz,4\n",
        encoding="utf-8",
    )


def _make_structure() -> Structure:
    return Structure(
        Lattice.cubic(4.2),
        ["Li", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )


class DBPackLayoutTests(unittest.TestCase):
    def test_get_db_pack_layout_matches_runtime_directory_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            layout = get_db_pack_layout(tmpdir)
            self.assertEqual(layout.catalog_csv.name, "catalog_deduplicated.csv")
            self.assertEqual(layout.stable_csv.name, "mp_experimental_stable.csv")
            self.assertEqual(layout.profiles_dir.name, "profiles64")
            self.assertEqual(layout.profiles_npz.name, "profiles64.npz")
            self.assertEqual(layout.profiles_index_csv.name, "index.csv")
            self.assertEqual(layout.original_json.name, "highsymm_metadata.json")
            self.assertEqual(layout.cif_map_json.name, "cif_map.json")

    def test_build_pipeline_config_accepts_custom_db_pack_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pack_root = Path(tmpdir) / "user_db_pack"
            pack_root.mkdir()
            db_cfg = build_db_config(
                pack_root,
                cif_map_json=pack_root / "cif_map.json",
            )
            db_cfg["original_json"] = None

            cfg_text = build_pipeline_config(
                run_name="dbpack_test",
                data_file="/tmp/example.dat",
                instprm_file="/tmp/example.instprm",
                allowed_elements=["Li", "O"],
                db_root=str(pack_root),
                db_config_override=db_cfg,
            )
            cfg = yaml.safe_load(cfg_text)

            self.assertEqual(cfg["db"]["catalog_csv"], str(pack_root / "catalog_deduplicated.csv"))
            self.assertEqual(cfg["db"]["stable_csv"], str(pack_root / "mp_experimental_stable.csv"))
            self.assertEqual(cfg["db"]["profiles_dir"], str(pack_root / "profiles64"))
            self.assertEqual(cfg["db"]["cif_map_json"], str(pack_root / "cif_map.json"))
            self.assertNotIn("original_json", cfg["db"])
            self.assertEqual(cfg["stage4"]["seed"], 0)
            self.assertEqual(cfg["stage4"]["pearson_q_max"], 8.0)
            self.assertTrue(cfg["polish_defer_main_cell"])

    def test_build_pipeline_config_persists_dataset_excluded_regions(self):
        cfg_text = build_pipeline_config(
            run_name="exclude_test",
            data_file="/tmp/example.dat",
            instprm_file="/tmp/example.instprm",
            allowed_elements=["Al"],
            limits=[5.0, 120.0],
            exclude_regions=[[12.5, 13.1], [44.0, 45.5]],
        )
        cfg = yaml.safe_load(cfg_text)

        self.assertEqual(cfg["datasets"][0]["limits"], [5.0, 120.0])
        self.assertEqual(
            cfg["datasets"][0]["exclude_regions"],
            [[12.5, 13.1], [44.0, 45.5]],
        )

    def test_sample_environment_does_not_allow_oxides_by_default(self):
        cfg_text = build_pipeline_config(
            run_name="env_policy_test",
            data_file="/tmp/example.dat",
            instprm_file="/tmp/example.instprm",
            allowed_elements=["Tb", "Be", "Ge", "O"],
            sample_env_elements=["Al"],
        )
        cfg = yaml.safe_load(cfg_text)
        sample_env = cfg["element_filter"]["sample_env"]

        self.assertEqual(sample_env["elements"], ["Al"])
        self.assertTrue(sample_env["allow_pure"])
        self.assertEqual(sample_env["allow_with"], [])
        self.assertTrue(sample_env["ban_cross_with_base"])

    def test_db_pack_progress_events_include_elapsed_time_and_counters(self):
        events = []
        now = time.perf_counter()

        _emit_progress(
            events.append,
            step="precheck",
            message="Checking custom CIFs",
            fraction=0.5,
            current=2,
            total=4,
            source_name="phase.cif",
            started_at=now - 2.0,
            stage_started_at=now - 0.5,
            checked_count=2,
            queued_count=1,
            skipped_count=1,
        )

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["step"], "precheck")
        self.assertEqual(event["fraction"], 0.5)
        self.assertEqual(event["current"], 2)
        self.assertEqual(event["total"], 4)
        self.assertEqual(event["source_name"], "phase.cif")
        self.assertGreaterEqual(event["elapsed_s"], 1.0)
        self.assertGreaterEqual(event["stage_elapsed_s"], 0.1)
        self.assertEqual(event["checked_count"], 2)
        self.assertEqual(event["queued_count"], 1)
        self.assertEqual(event["skipped_count"], 1)


class PipelineDeterminismTests(unittest.TestCase):
    def test_low_q_pearson_crop_uses_q_mask_but_returns_native_x(self):
        x_native = [100.0, 200.0, 300.0, 400.0]
        residual = [1.0, 2.0, 3.0, 4.0]
        q_values = [10.0, 7.5, 4.0, 12.0]

        x_crop, y_crop, meta = _crop_native_arrays_by_q(
            x_native,
            residual,
            q_values,
            q_max=8.0,
            min_points=2,
        )

        self.assertTrue(meta["enabled"])
        self.assertEqual(list(x_crop), [200.0, 300.0])
        self.assertEqual(list(y_crop), [2.0, 3.0])
        self.assertEqual(meta["output_points"], 2)

    def test_benchmark_report_writes_json_and_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            timer = BenchTimer("bench_test")
            with timer.block("stage"):
                pass

            json_path, csv_path = timer.write_report(
                str(Path(tmpdir) / "benchmark_report.json"),
                {"passes": [{"pass": 1, "compare_candidate_count": 2}]},
            )

            payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["run_name"], "bench_test")
            self.assertEqual(payload["passes"][0]["compare_candidate_count"], 2)
            self.assertTrue(Path(csv_path).exists())

    def test_choose_top_new_uses_pearson_then_phase_id_only_for_ties(self):
        fractions = {
            "phase_b": {"weight_fraction_pct": 2.0},
            "phase_a": {"weight_fraction_pct": 2.0},
            "phase_c": {"weight_fraction_pct": 1.9},
        }
        pearson = {"phase_a": 0.80, "phase_b": 0.80, "phase_c": 0.99}

        best = UnifiedPipeline._choose_top_new_by_wf(
            fractions,
            ["phase_b", "phase_a", "phase_c"],
            pearson,
        )

        self.assertEqual(best, "phase_a")

    def test_copy_gpx_with_lst_preserves_matching_lst(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src = root / "source.gpx"
            dst = root / "nested" / "dest.gpx"
            src.write_text("gpx", encoding="utf-8")
            src.with_suffix(".lst").write_text("lst", encoding="utf-8")

            UnifiedPipeline._copy_gpx_with_lst(str(src), str(dst))

            self.assertEqual(dst.read_text(encoding="utf-8"), "gpx")
            self.assertEqual(dst.with_suffix(".lst").read_text(encoding="utf-8"), "lst")


class DBLoaderPathResolutionTests(unittest.TestCase):
    def test_loader_resolves_relative_cif_map_and_relative_cif_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pack_root = Path(tmpdir) / "pack"
            cifs_dir = pack_root / "cifs"
            cifs_dir.mkdir(parents=True)
            phase_id = "user_phase"
            _write_catalog(pack_root / "catalog_deduplicated.csv", phase_id)

            structure = _make_structure()
            cif_path = cifs_dir / f"{phase_id}.cif"
            structure.to(filename=str(cif_path), fmt="cif")

            (pack_root / "cif_map.json").write_text(
                json.dumps({phase_id: f"cifs/{phase_id}.cif"}),
                encoding="utf-8",
            )

            loader = DBLoader(
                CatalogPaths(
                    catalog_csv=str(pack_root / "catalog_deduplicated.csv"),
                    cif_map_json="cif_map.json",
                )
            )

            self.assertEqual(loader.ensure_cif_on_disk(phase_id), str(cif_path.resolve()))
            loaded = loader.load_structure(phase_id)
            self.assertEqual(loaded.composition.reduced_formula, structure.composition.reduced_formula)


class MainPhaseExclusionTests(unittest.TestCase):
    def test_pipeline_detects_db_entry_matching_main_cif(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            cif_path = tmpdir / "main_phase.cif"
            structure = _make_structure()
            structure.to(filename=str(cif_path), fmt="cif")

            pack = build_mini_db_pack(
                [cif_path],
                tmpdir / "mini_pack",
                source_type="neutron",
                overwrite=True,
            )

            pipe = UnifiedPipeline({})
            ok = pipe.initialize_database(pack.db_config)
            self.assertTrue(ok)

            matches = pipe._matching_db_ids_for_main_phase(str(cif_path))
            self.assertIn(pack.phase_ids[0], matches)

    def test_loader_resolves_relative_original_json_against_catalog_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pack_root = Path(tmpdir) / "pack"
            pack_root.mkdir(parents=True)
            phase_id = "json_phase"
            _write_catalog(pack_root / "catalog_deduplicated.csv", phase_id)

            structure = _make_structure()
            scratch_cif = pack_root / "scratch.cif"
            structure.to(filename=str(scratch_cif), fmt="cif")
            cif_text = scratch_cif.read_text(encoding="utf-8")

            (pack_root / "highsymm_metadata.json").write_text(
                json.dumps({phase_id: {"cif_content": cif_text}}),
                encoding="utf-8",
            )

            loader = DBLoader(
                CatalogPaths(
                    catalog_csv=str(pack_root / "catalog_deduplicated.csv"),
                    original_json="highsymm_metadata.json",
                )
            )

            with tempfile.TemporaryDirectory() as outdir:
                out_path = Path(loader.ensure_cif_on_disk(phase_id, out_dir=outdir))
                self.assertTrue(out_path.exists())
                self.assertIn("_symmetry_space_group_name_H-M", out_path.read_text(encoding="utf-8"))

            loaded = loader.load_structure(phase_id)
            self.assertEqual(loaded.composition.reduced_formula, structure.composition.reduced_formula)


if __name__ == "__main__":
    unittest.main()
