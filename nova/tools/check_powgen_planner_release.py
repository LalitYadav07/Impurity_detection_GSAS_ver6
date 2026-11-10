"""Offline native installation checks, not experimental validation.

Run with the scientific Python packaged in nova/dockerfiles/Dockerfile. Each campaign
uses a fresh bounded process; all requests, logs and results are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from gpx_recipe.jobs import scientific_environment
from gpx_recipe.planning import instrument
from gpx_recipe.planning.io import write_json


NATIVE_SHA256 = "6fea6023fa33a2585d091b04a479194c37f6efac4336ee29c7bef7af31e930af"
BANDS = (
    ("band1", "2026B_HighRes_60HzB1_CWL0p8.instprm", .3, 3.1),
    ("band2", "2026B_HighRes_60HzB2_CWL1p5.instprm", .55, 7.24),
    ("band3", "2026B_HighRes_60HzB3_CWL2p665.instprm", 1.1, 7.24),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as value:
        return {key: value[key] for key in value.files}


def check_campaign(output):
    receipt = json.loads((output / "result.json").read_text())
    require(receipt["synthetic"], "A release check must be synthetic")
    require(not receipt["experimental_accuracy_certified"], "Software checks cannot certify experimental accuracy")
    require(receipt["calculation_backend"]["native_sources"]["GSASIIstrMath.py"] == NATIVE_SHA256,
            "The reviewed native backend is required without a legacy opt-in")
    for name, expected in receipt["artifacts"].items():
        require(digest(output / name) == expected, "Artifact checksum mismatch: " + name)
    curves = arrays(output / "curves.npz")
    require(np.all(np.isfinite(curves["raw_rate"])) and np.all(curves["raw_rate"] >= 0),
            "Raw expected rates must be finite and nonnegative")
    previous = np.zeros_like(curves["d"], dtype=np.int64)
    count_scale = reference_variance = None
    for look in receipt["looks"]:
        data = arrays(output / "reduced" / look["sufficient_statistics"])
        charge = look["cumulative_charge_uah"]
        require(np.issubdtype(data["counts"].dtype, np.integer), "Counts must be integers")
        require(np.all(data["counts"] >= 0), "Counts must be nonnegative")
        np.testing.assert_array_equal(previous + data["increment"], data["counts"])
        np.testing.assert_allclose(data["expected_counts"], charge * curves["raw_rate"], rtol=2e-14)
        np.testing.assert_allclose(data["sigma"] ** 2,
                                   data["counting_variance"] + data["shared_reference_variance"], rtol=2e-14)
        if count_scale is not None:
            np.testing.assert_allclose(data["counting_variance"] * charge, count_scale, rtol=2e-14)
            np.testing.assert_array_equal(data["shared_reference_variance"], reference_variance)
        count_scale = data["counting_variance"] * charge
        reference_variance = data["shared_reference_variance"]
        previous = data["counts"]
    return receipt, curves


def run(output):
    require(not output.exists() or not any(output.iterdir()), "Choose an empty release-check directory")
    output.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "powgen-planner-installation-check/v1", "status": "running",
               "experimental_validation": False, "campaigns": [], "properties": [],
               "python": sys.version, "platform": platform.platform(),
               "dependencies": {name: importlib.metadata.version(name) for name in
                                ("numpy", "scipy", "h5py", "PyCifRW")}}
    summary_path = output / "installation-check.json"
    write_json(summary_path, summary)
    env = scientific_environment(os.environ.get("GSASII_ROOT"))
    env["MPLCONFIGDIR"] = str(output / "matplotlib")
    started = time.monotonic()

    def campaign(name, request=None, source=None):
        directory = output / name
        directory.mkdir()
        if source is None:
            source = directory / "request.json"
            write_json(source, request)
        destination = directory / "simulation"
        with (directory / "worker.log").open("w") as log:
            process = subprocess.run(
                [sys.executable, "-m", "gpx_recipe.planning.worker", str(source), str(destination)],
                cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        require(process.returncode == 0, name + " failed; inspect " + str(directory / "worker.log"))
        receipt, curves = check_campaign(destination)
        summary["campaigns"].append({"name": name, "result_sha256": digest(destination / "result.json"),
                                     "configuration": receipt["configuration"], "bins": len(curves["d"]),
                                     "expected_counts": [row["total_expected_counts"] for row in receipt["looks"]],
                                     "artifacts_verified": len(receipt["artifacts"])})
        write_json(summary_path, summary)
        print(json.dumps({"passed": name, "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
        return destination, receipt, curves

    try:
        example = Path(instrument.__file__).parent / "data" / "examples" / "silicon.cif"
        base = {"schema": "powgen-planning-scenario/v1", "cif": str(example),
                "sample": {"mass_g": .950, "radius_cm": .295, "packing_fraction": .3730165664701342,
                           "beam_width_cm": 1., "beam_height_cm": 4.},
                "response_mode": "current_cycle", "resolution_model": "cycle_table", "container": "PAC06",
                "relative_step": .0004, "seed": 42, "seconds": [60., 300., 1800.],
                "beam": {"average_current_ua": 1500., "proton_energy_gev": 1.3}}
        band2 = None
        for label, configuration, lower, upper in BANDS:
            request = dict(base, configuration=configuration, d_min=lower, d_max=upper)
            time_output, time_receipt, time_curves = campaign(label + "-time", request)
            charge_request = {key: value for key, value in request.items() if key not in {"seconds", "beam"}}
            charge_request["charge_uah"] = [25., 125., 750.]
            charge_output, _, charge_curves = campaign(label + "-charge", charge_request)
            for key in time_curves:
                np.testing.assert_array_equal(time_curves[key], charge_curves[key])
            for look in time_receipt["looks"]:
                filename = "reduced/" + look["sufficient_statistics"]
                a, b = arrays(time_output / filename), arrays(charge_output / filename)
                for key in a:
                    np.testing.assert_array_equal(a[key], b[key])
            summary["properties"].append(label + ": exact time/charge equivalence; cumulative integer counts; Poisson charge scaling; shared reference floor")
            if label == "band2":
                band2 = request, time_output, time_receipt, time_curves

        request, time_output, time_receipt, time_curves = band2
        replay_output, _, replay_curves = campaign("portable-replay", source=time_output / "scenario.json")
        for key in time_curves:
            np.testing.assert_array_equal(time_curves[key], replay_curves[key])
        for look in time_receipt["looks"]:
            filename = "reduced/" + look["sufficient_statistics"]
            np.testing.assert_array_equal(arrays(time_output / filename)["counts"], arrays(replay_output / filename)["counts"])
        summary["properties"].append("Portable scenario reproduces means and seeded counts exactly")

        _, isotope_receipt, isotope_curves = campaign("silicon-30", dict(request, isotopes={"Si": "30"}))
        require(isotope_receipt["cell_mass_da"] > time_receipt["cell_mass_da"], "Si-30 mass was not applied")
        require(not np.allclose(isotope_curves["raw_nuclear_rate"], time_curves["raw_nuclear_rate"]), "Isotope change did not alter scattering")
        summary["properties"].append("Si-30 changes scattering and cell mass consistently")

        changed = output / "expanded-silicon.cif"
        changed.write_text(example.read_text().replace("5.4312", "5.46"))
        structural_output, structural_receipt, structural_curves = campaign("structural-comparison", dict(request, comparison_cif=str(changed)))
        np.testing.assert_array_equal(time_curves["raw_rate"], structural_curves["raw_rate"])
        require(not np.allclose(arrays(structural_output / "comparison-curves.npz")["raw_rate"], time_curves["raw_rate"]),
                "Changed structural cell did not alter the comparison pattern")
        require(structural_receipt["inputs"]["comparison_cif"]["sha256"] == digest(changed), "Comparison input identity was lost")
        summary["properties"].append("Structural comparison changes the pattern and preserves the original baseline")

        # Artificial P1 controls test the machinery without implying a measured
        # magnetic material, transition, or subgroup-discovery validation.
        cell = """_cell_length_a 4
_cell_length_b 5
_cell_length_c 6
_cell_angle_alpha 80
_cell_angle_beta 75
_cell_angle_gamma 70
"""
        atoms = """loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_U_iso_or_equiv
Fe1 Fe2+ .13 .27 .38 1 .003
"""
        iron = output / "synthetic-fe.cif"
        iron.write_text("data_synthetic_Fe\n" + cell + "_symmetry_space_group_name_H-M 'P 1'\nloop_\n_space_group_symop_operation_xyz\n'x,y,z'\n" + atoms)
        magnetic = output / "synthetic-fe.mcif"
        magnetic.write_text("data_synthetic_Fe_magnetic\n" + cell + """_space_group_magn.name_BNS 'P 1'
_space_group_magn.number_BNS '1.1'
loop_
_space_group_symop_magn_operation.id
_space_group_symop_magn_operation.xyz
1 'x,y,z,+1'
loop_
_space_group_symop_magn_centering.id
_space_group_symop_magn_centering.xyz
1 'x,y,z,+1'
""" + atoms + """loop_
_atom_site_moment.label
_atom_site_moment.crystalaxis_x
_atom_site_moment.crystalaxis_y
_atom_site_moment.crystalaxis_z
Fe1 .6 -1.1 2.2
""")
        magnetic_request = dict(request, cif=str(iron), mcif=str(magnetic))
        _, _, unit = campaign("magnetic-unit", magnetic_request)
        _, _, doubled = campaign("magnetic-double", dict(magnetic_request, moment_multiplier=2.))
        require(np.max(unit["raw_magnetic_rate"]) > 0, "The magnetic control has no signal")
        np.testing.assert_allclose(doubled["raw_magnetic_rate"], 4 * unit["raw_magnetic_rate"], rtol=2e-12, atol=1e-12)
        np.testing.assert_array_equal(doubled["raw_nuclear_rate"], unit["raw_nuclear_rate"])
        summary["properties"].append("Doubling supplied magnetic moments quadruples magnetic intensity and preserves nuclear intensity")

        _, mixed, _ = campaign("crystalline-mixture", dict(request, additional_phases=[{"cif": str(iron), "mass_fraction": .2}]))
        require(mixed["mixture"], "Mixture was not retained")
        np.testing.assert_allclose(sorted(mixed["phase_mass_fractions"]), [.2, .8], rtol=0, atol=1e-15)
        summary["properties"].append("Declared 80/20 crystalline mass fractions survive native simulation and export")
        summary["status"] = "passed"
        summary["limits"] = ["Installation and analytic controls do not establish agreement with experiments.",
                             "Online Bilbao and ISODISTORT discovery are outside this offline check.",
                             "Experimental intensity and noise discrepancies remain in the qualification reports."]
    except Exception as error:
        summary.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        write_json(summary_path, summary)
    print(json.dumps({"status": summary["status"], "report": str(summary_path), "campaigns": len(summary["campaigns"])}))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output.resolve())
