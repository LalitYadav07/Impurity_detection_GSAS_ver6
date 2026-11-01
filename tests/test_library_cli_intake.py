"""CLI intake contract: no scientific compute or pytest dependency."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ndip_runner
from cif_source_archive import validate_cif_member_names


class LibraryCliIntakeContract(unittest.TestCase):
    def run_intake(self, names, *, mode="mini", failed=()):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs = root / "inputs"
            inputs.mkdir()
            for name in names:
                (inputs / name).write_bytes(("data_" + name + "\n").encode())
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
            seen = []
            fake = ModuleType("db_pack_builder")

            def build(paths, output_root, **kwargs):
                seen.extend(paths)
                self.assertEqual(kwargs["reviewed_occupancy_excess"], 0.0)
                self.assertEqual(kwargs["occupancy_review_note"], "")
                # Read every selected source as the real builder does. Bad
                # paths must raise, not vanish from the source accounting.
                for path in paths:
                    path.read_bytes()
                output_root.mkdir()
                manifest_path = output_root / "manifest.json"
                manifest_path.write_text(json.dumps({"source_names": [p.name for p in paths]}))
                return SimpleNamespace(pack_root=output_root, manifest_path=manifest_path,
                    db_config={}, phase_ids=[p.stem for p in paths if p.name not in failed],
                    failures=[{"source_name": p.name, "error": "test invalid source"}
                              for p in paths if p.name in failed])

            fake.build_mini_db_pack = fake.build_augmented_db_pack = build
            output = root / "published.json"
            argv = ["build-library", "--cif-dir", str(inputs), "--library-mode", mode,
                    "--output-root", str(root / "output"), "--manifest-output", str(output)]
            if mode == "augmented":
                argv.extend(["--base-db-root", str(root / "builtin")])
            args = ndip_runner.build_parser().parse_args(argv)
            with patch.dict(sys.modules, {"db_pack_builder": fake}):
                self.assertEqual(args.handler(args), 0)
            after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
            self.assertEqual(before, after)
            return [p.name for p in seen], json.loads(output.read_text())

    def test_mixed_case_sources_reach_each_library_mode(self):
        for mode in ("mini", "augmented"):
            with self.subTest(mode=mode):
                seen, manifest = self.run_intake(["lower.cif", "UPPER.CIF", "mixed.CiF", "notes.txt"], mode=mode)
                self.assertEqual(seen, ["UPPER.CIF", "lower.cif", "mixed.CiF"])
                self.assertEqual(manifest["phase_count"], 3)
                self.assertEqual(manifest["source_names"], seen)
                self.assertEqual(manifest["failures"], [])

    def test_all_uppercase_sources_are_not_reported_as_empty(self):
        seen, manifest = self.run_intake(["A.CIF", "B.CIF"])
        self.assertEqual(seen, ["A.CIF", "B.CIF"])
        self.assertEqual(manifest["phase_count"], 2)

    def test_uppercase_failure_remains_in_published_accounting(self):
        seen, manifest = self.run_intake(["good.cif", "BAD.CIF"], failed=("BAD.CIF",))
        self.assertEqual(seen, ["BAD.CIF", "good.cif"])
        self.assertEqual(manifest["phase_count"], 1)
        self.assertEqual(manifest["failures"], [{"source_name": "BAD.CIF", "error": "test invalid source"}])

    def test_existing_flat_archive_collision_rule_is_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "sources.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("a/Model.cif", "data_first\n")
                handle.writestr("b/model.CIF", "data_second\n")
            with self.assertRaisesRegex(ValueError, "basename collision"):
                validate_cif_member_names(archive)

    def test_cli_preserves_case_distinct_names_on_supported_filesystems(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "Case.cif").touch()
            (root / "case.cif").touch()
            if len(list(root.iterdir())) != 2:
                self.skipTest("Filesystem cannot represent case-distinct filenames")
        seen, manifest = self.run_intake(["Case.cif", "case.cif"])
        self.assertEqual(seen, ["Case.cif", "case.cif"])
        self.assertEqual(manifest["phase_count"], 2)

    def test_non_cif_only_directory_fails_before_builder(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "notes.txt").write_text("not a source")
            args = ndip_runner.build_parser().parse_args(["build-library", "--cif-dir", str(root),
                "--output-root", str(root / "output"), "--manifest-output", str(root / "manifest.json")])
            fake = ModuleType("db_pack_builder")
            def forbidden(*_, **__):
                self.fail("Unknown files must not reach the scientific builder")
            fake.build_mini_db_pack = fake.build_augmented_db_pack = forbidden
            with patch.dict(sys.modules, {"db_pack_builder": fake}):
                with self.assertRaisesRegex(ValueError, "At least one CIF"):
                    args.handler(args)
            self.assertFalse((root / "manifest.json").exists())

    def test_bad_matching_path_is_not_silently_filtered(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "invalid.CIF").mkdir()
            args = ndip_runner.build_parser().parse_args(["build-library", "--cif-dir", str(root),
                "--output-root", str(root / "output"), "--manifest-output", str(root / "manifest.json")])
            fake = ModuleType("db_pack_builder")
            def build(paths, *_, **__):
                for path in paths:
                    path.read_bytes()
            fake.build_mini_db_pack = fake.build_augmented_db_pack = build
            with patch.dict(sys.modules, {"db_pack_builder": fake}):
                with self.assertRaises(IsADirectoryError):
                    args.handler(args)
            self.assertFalse((root / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
