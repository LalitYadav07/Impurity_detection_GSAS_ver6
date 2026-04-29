import json
import sys
import tempfile
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
from scripts.db_pack_builder import build_mini_db_pack
from scripts.gsas_complete_pipeline_nomain import UnifiedPipeline


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
