"""Read-only, bounded audit of exact public CIFs in the pinned catalog image.

Run only against the packaged built-in catalog. No measured patterns, user
libraries, credentials or facility data are inputs to this diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def audit(ids, output):
    from aniso_db_loader import CatalogPaths, DBLoader
    from cif_occupancy import load_authorized_structure
    from gsas_cif_import import add_verified_phase
    from ndip_catalog_bundle import verify_installed_catalog
    from pymatgen.io.cif import CifFile
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from GSASII import GSASIIscriptable as G2

    root = Path(os.environ["RADAR_PD_DB_ROOT"]).resolve()
    if root != Path("/opt/radar-pd/catalog"):
        raise ValueError("This diagnostic accepts only the pinned built-in catalog")
    package = verify_installed_catalog(root)
    database = root / "database_neutron"
    mapping = database / "cif_path_map.json"
    loader = DBLoader(CatalogPaths(str(database / "catalog_deduplicated.csv"),
                                  str(mapping) if mapping.exists() else None,
                                  str(database / "highsymm_metadata.json")))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for pid in ids:
        if not re.fullmatch(r"(?:mp|cod)-[0-9]+", pid):
            raise ValueError("Only exact public MP/COD identifiers are allowed")
        original = Path(loader.ensure_cif_on_disk(pid, out_dir=str(output / "sources")))
        raw = original.read_bytes()
        if len(raw) > 1024 * 1024:
            raise ValueError("Unexpectedly large source CIF")
        saved = output / (pid + ".cif")
        saved.write_bytes(raw)
        cif = CifFile.from_str(raw.decode("utf-8"))
        blocks = []
        for name, block in cif.data.items():
            blocks.append({"name": name, "data": plain(block.data)})
        structure = load_authorized_structure(original)
        detected = []
        for tolerance in (0.001, 0.01, 0.1):
            symmetry = SpacegroupAnalyzer(structure, symprec=tolerance)
            detected.append({"symprec": tolerance,
                             "number": symmetry.get_space_group_number(),
                             "symbol": symmetry.get_space_group_symbol()})
        project = G2.G2Project(newgpx=str(output / (pid + ".gpx")))
        phase = add_verified_phase(project, str(saved), phasename=pid, histograms=[])
        project.save()
        general = phase.data["General"]
        sg = general["SGData"]
        row = loader.catalog.iloc[loader._row_index[pid]]
        rows.append({"id": pid, "source_sha256": hashlib.sha256(raw).hexdigest(),
                     "source_bytes": len(raw), "saved_cif": saved.name,
                     "catalog": {k: plain(row.get(k)) for k in
                                 ("pretty_formula", "space_group", "SG_symbol", "npz")},
                     "source_blocks": blocks,
                     "pymatgen": {"formula": structure.composition.formula,
                                  "cell": list(structure.lattice.parameters),
                                  "sites": len(structure), "detected_symmetry": detected},
                     "native": {"symbol": sg["SpGrp"],
                                "cell": plain(general["Cell"]),
                                "ops": plain(sg["SGOps"]),
                                "centering": plain(sg["SGCen"]),
                                "inversion": plain(sg["SGInv"]),
                                "atoms": plain(phase.data["Atoms"]),
                                "atom_pointers": plain(general["AtomPtrs"]),
                                "counts": plain(general["NoAtoms"])}})
    result = {"schema": "radar-public-catalog-source-audit/v1", "catalog": package,
              "build_source": os.environ.get("RADAR_PD_GIT_COMMIT"), "records": rows}
    (output / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"ids": ids, "records": len(rows), "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    ids = args.ids.split(",")
    if not 1 <= len(ids) <= 8 or len(set(ids)) != len(ids):
        raise ValueError("Request between one and eight unique public record IDs")
    audit(ids, args.output)
