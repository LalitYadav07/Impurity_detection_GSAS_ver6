import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.aniso_db_loader import CatalogPaths, DBLoader
from scripts.db_pack_builder import (
    NEUTRON_DEFAULT_WAVELENGTH,
    NEUTRON_DEFAULT_TOP_M,
    NEUTRON_DEFAULT_TWO_THETA_MAX,
    XRAY_DEFAULT_WAVELENGTH,
    XRAY_DEFAULT_TOP_M,
    XRAY_DEFAULT_TWO_THETA_MAX,
    _optimized_xray_two_theta_max,
    _normalize_catalog_df,
    _build_base_duplicate_index,
    _find_matching_base_phase_ids,
    infer_phase_display_name,
    build_augmented_db_pack,
    collect_phase_inputs,
    build_mini_db_pack,
    resolve_simulation_settings,
)


def _write_cif(path: Path, species, frac_coords, a: float = 4.2) -> None:
    structure = Structure(Lattice.cubic(a), species, frac_coords)
    structure.to(filename=str(path), fmt="cif")


class SimulationSettingsTests(unittest.TestCase):
    def test_xray_builtin_pack_contract_is_preserved(self):
        db_root = REPO_ROOT / "data" / "database_xray"
        settings = resolve_simulation_settings("xray", base_db_root=db_root)

        self.assertEqual(settings.source_type, "xray")
        self.assertEqual(settings.radiation, "xray")
        self.assertAlmostEqual(settings.wavelength, XRAY_DEFAULT_WAVELENGTH, places=6)
        self.assertEqual(settings.topM, XRAY_DEFAULT_TOP_M)
        self.assertEqual(settings.two_theta_min, 0.0)
        self.assertAlmostEqual(
            settings.two_theta_max,
            _optimized_xray_two_theta_max(settings.q_max, settings.wavelength),
            places=6,
        )
        self.assertLess(settings.two_theta_max, XRAY_DEFAULT_TWO_THETA_MAX)
        self.assertEqual((settings.q_min, settings.q_max, settings.n_bins), (0.5, 6.0, 64))
        self.assertAlmostEqual(settings.sigma_bins, 0.7, places=6)

    def test_neutron_pack_contract_uses_explicit_default_wavelength(self):
        db_root = REPO_ROOT / "data" / "database_neutron"
        settings = resolve_simulation_settings("neutron", base_db_root=db_root)

        self.assertEqual(settings.source_type, "neutron")
        self.assertEqual(settings.radiation, "neutron")
        self.assertAlmostEqual(settings.wavelength, NEUTRON_DEFAULT_WAVELENGTH, places=6)
        self.assertEqual(settings.topM, NEUTRON_DEFAULT_TOP_M)
        self.assertEqual(settings.topM, XRAY_DEFAULT_TOP_M)
        self.assertEqual((settings.two_theta_min, settings.two_theta_max), (0.0, NEUTRON_DEFAULT_TWO_THETA_MAX))
        self.assertEqual((settings.q_min, settings.q_max, settings.n_bins), (0.5, 6.0, 64))
        self.assertAlmostEqual(settings.sigma_bins, 0.7, places=6)


class DBPackBuilderTests(unittest.TestCase):
    def test_phase_display_name_uses_formula_then_declared_name_then_filename(self):
        cif = "_chemical_name_common 'rock salt reference'\n"

        self.assertEqual(
            infer_phase_display_name("collcode123.cif", cif, "NaCl"),
            "NaCl - rock salt reference",
        )
        self.assertEqual(
            infer_phase_display_name("collcode123.cif", "data_collcode123\n", "NaCl"),
            "NaCl",
        )
        self.assertEqual(
            infer_phase_display_name("Fe2VAl_L21.cif", "data_unknown\n", ""),
            "Fe2VAl L21",
        )
        self.assertEqual(
            infer_phase_display_name("iron.cif", "_pd_phase_id 1\n", "Fe"),
            "Fe",
        )
        self.assertEqual(
            infer_phase_display_name("collcode258024.cif", "data_collcode258024\n", "Al Fe2 V"),
            "AlFe2V",
        )

    def test_normalize_catalog_df_preserves_uint64_element_masks(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "id": ["phase_a"],
                "pretty_formula": ["XeUuo"],
                "space_group": ["225"],
                "SG_symbol": ["Fm-3m"],
                "elements_list": ["Xe,Uuo"],
                "elements_mask_hi": ["1"],
                "elements_mask_lo": ["17221765249978874268"],
                "npz": ["phases/phase_a.npz"],
                "n_reflections": ["160"],
            }
        )

        out = _normalize_catalog_df(df)

        self.assertEqual(str(out["elements_mask_lo"].dtype), "UInt64")
        self.assertEqual(int(out.loc[0, "elements_mask_lo"]), 17221765249978874268)

    def test_loader_compacts_formula_style_catalog_names(self):
        import pandas as pd

        loader = DBLoader.__new__(DBLoader)
        loader.catalog = pd.DataFrame(
            [
                {
                    "id": "user_phase",
                    "display_name": "Al Fe2 V",
                    "pretty_formula": "Al Fe2 V",
                }
            ]
        )
        loader._row_index = {"user_phase": 0}

        self.assertEqual(loader.get_pretty_name("user_phase"), "AlFe2V")

    def test_build_mini_db_pack_writes_runtime_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            cif1 = tmpdir / "phase_a.cif"
            cif2 = tmpdir / "phase_b.cif"
            _write_cif(cif1, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]], a=5.63)
            _write_cif(cif2, ["Li", "F"], [[0, 0, 0], [0.5, 0.5, 0.5]], a=4.03)
            # A formula/name scalar is optional. The builder must derive the
            # scientific label from atom sites instead of exposing phase_a.
            cif1.write_text(
                "\n".join(
                    line
                    for line in cif1.read_text(encoding="utf-8").splitlines()
                    if not line.lstrip().startswith("_chemical_formula")
                )
                + "\n",
                encoding="utf-8",
            )

            out_root = tmpdir / "mini_xray_pack"
            result = build_mini_db_pack(
                [cif1, cif2],
                out_root,
                source_type="xray",
                overwrite=True,
            )

            self.assertEqual(len(result.phase_ids), 2)
            self.assertTrue(result.layout.catalog_csv.exists())
            self.assertTrue(result.layout.stable_csv.exists())
            self.assertTrue(result.layout.profiles_npz.exists())
            self.assertTrue(result.layout.profiles_index_csv.exists())
            self.assertTrue(result.layout.original_json.exists())
            self.assertTrue(result.layout.cif_map_json.exists())
            self.assertTrue(result.layout.manifest_json.exists())

            with np.load(result.layout.profiles_npz) as z:
                self.assertEqual(z["profiles"].shape, (2, 64))
                self.assertAlmostEqual(float(z["q_min"]), 0.5)
                self.assertAlmostEqual(float(z["q_max"]), 6.0)

            loader = DBLoader(CatalogPaths(
                catalog_csv=str(result.db_config["catalog_csv"]),
                cif_map_json=str(result.db_config["cif_map_json"]),
                original_json=str(result.db_config["original_json"]),
            ))
            self.assertEqual(len(loader.catalog), 2)
            self.assertEqual(loader.get_pretty_name(result.phase_ids[0]), "NaCl")
            s0 = loader.load_structure(result.phase_ids[0])
            self.assertGreater(len(s0), 0)

    def test_build_mini_db_pack_dedupes_identical_cif_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            cif1 = tmpdir / "phase_dup_a.cif"
            cif2 = tmpdir / "phase_dup_b.cif"
            _write_cif(cif1, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]], a=5.63)
            cif2.write_text(cif1.read_text(encoding="utf-8"), encoding="utf-8")

            out_root = tmpdir / "mini_pack"
            result = build_mini_db_pack(
                [cif1, cif2],
                out_root,
                source_type="xray",
                overwrite=True,
            )

            self.assertEqual(len(result.phase_ids), 1)
            with np.load(result.layout.profiles_npz) as z:
                self.assertEqual(z["profiles"].shape, (1, 64))

    def test_build_augmented_db_pack_merges_base_and_overlay_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            base_cif = tmpdir / "base_phase.cif"
            extra_cif = tmpdir / "extra_phase.cif"
            _write_cif(base_cif, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]], a=5.63)
            _write_cif(extra_cif, ["Li", "F"], [[0, 0, 0], [0.5, 0.5, 0.5]], a=4.03)

            base_root = tmpdir / "base_pack"
            base_result = build_mini_db_pack(
                [base_cif],
                base_root,
                source_type="neutron",
                overwrite=True,
            )

            aug_root = tmpdir / "aug_pack"
            aug_result = build_augmented_db_pack(
                [extra_cif],
                aug_root,
                source_type="neutron",
                base_db_root=base_root,
                overwrite=True,
            )

            with np.load(aug_result.layout.profiles_npz) as z:
                self.assertEqual(z["profiles"].shape, (2, 64))

            manifest = json.loads(aug_result.layout.manifest_json.read_text(encoding="utf-8"))
            self.assertEqual(manifest["kind"], "augmented")
            self.assertEqual(manifest["n_added_phases"], 1)
            self.assertEqual(manifest["n_base_phases"], 1)

            loader = DBLoader(CatalogPaths(
                catalog_csv=str(aug_result.db_config["catalog_csv"]),
                cif_map_json=str(aug_result.db_config["cif_map_json"]),
                original_json=str(aug_result.db_config["original_json"]),
            ))

            self.assertEqual(len(loader.catalog), 2)
            base_structure = loader.load_structure(base_result.phase_ids[0])
            extra_structure = loader.load_structure(aug_result.phase_ids[0])
            self.assertGreater(len(base_structure), 0)
            self.assertGreater(len(extra_structure), 0)

    def test_build_augmented_db_pack_skips_phase_already_in_base_by_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            base_cif = tmpdir / "base_phase.cif"
            base_dup_cif = tmpdir / "base_phase_uploaded_again.cif"
            extra_cif = tmpdir / "extra_phase.cif"
            _write_cif(base_cif, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]], a=5.63)
            base_dup_cif.write_text(base_cif.read_text(encoding="utf-8"), encoding="utf-8")
            _write_cif(extra_cif, ["Li", "F"], [[0, 0, 0], [0.5, 0.5, 0.5]], a=4.03)

            base_root = tmpdir / "base_pack"
            build_mini_db_pack(
                [base_cif],
                base_root,
                source_type="neutron",
                overwrite=True,
            )

            aug_root = tmpdir / "aug_pack"
            aug_result = build_augmented_db_pack(
                [base_dup_cif, extra_cif],
                aug_root,
                source_type="neutron",
                base_db_root=base_root,
                overwrite=True,
            )

            manifest = json.loads(aug_result.layout.manifest_json.read_text(encoding="utf-8"))
            self.assertEqual(manifest["n_added_phases"], 1)
            self.assertEqual(len(aug_result.failures), 1)
            self.assertIn("already exists in base database", aug_result.failures[0]["error"])

            loader = DBLoader(CatalogPaths(
                catalog_csv=str(aug_result.db_config["catalog_csv"]),
                cif_map_json=str(aug_result.db_config["cif_map_json"]),
                original_json=str(aug_result.db_config["original_json"]),
            ))
            self.assertEqual(len(loader.catalog), 2)

    def test_base_duplicate_index_reuses_candidate_structures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            base_cif = tmpdir / "base_phase.cif"
            base_dup_cif = tmpdir / "base_phase_uploaded_again.cif"
            _write_cif(base_cif, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]], a=5.63)
            base_dup_cif.write_text(base_cif.read_text(encoding="utf-8"), encoding="utf-8")

            base_result = build_mini_db_pack(
                [base_cif],
                tmpdir / "base_pack",
                source_type="neutron",
                overwrite=True,
            )
            loader = DBLoader(CatalogPaths(
                catalog_csv=str(base_result.db_config["catalog_csv"]),
                cif_map_json=str(base_result.db_config["cif_map_json"]),
                original_json=str(base_result.db_config["original_json"]),
            ))
            duplicate_index = _build_base_duplicate_index(loader)
            self.assertIsNotNone(duplicate_index)

            phase = collect_phase_inputs([base_dup_cif])[0]
            matches = _find_matching_base_phase_ids(phase, base_loader=loader, duplicate_index=duplicate_index)
            self.assertEqual(matches, base_result.phase_ids)
            cache_size = len(duplicate_index.structure_cache)

            matches_again = _find_matching_base_phase_ids(phase, base_loader=loader, duplicate_index=duplicate_index)
            self.assertEqual(matches_again, base_result.phase_ids)
            self.assertEqual(len(duplicate_index.structure_cache), cache_size)


if __name__ == "__main__":
    unittest.main()
