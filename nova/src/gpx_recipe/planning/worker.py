"""A fresh native process owns one immutable POWGEN simulation campaign."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from .contracts import validate_scenario
from .instrument import selected_profile, selected_resolution_table
from .exposure_inputs import cumulative_exposures
from .io import write_json, export_snapshot, export_count_look


def run(scenario, output):
    from gpx_recipe.snapshot import gsas
    gsas()  # Initialize the explicit native installation before dependent imports.
    from .gsas import calculate
    from .response import powgen_response
    from .counting import Accumulator

    request = validate_scenario(scenario)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new campaign directory")
    output.mkdir(parents=True, exist_ok=True)
    inputs = {}
    input_directory = output / 'inputs';input_directory.mkdir()
    def capture(key, filename):
        source=Path(filename);before=source.stat()
        if not 0 < before.st_size <= 20*1024*1024:
            raise ValueError('A structure exceeds the bounded input size')
        raw=source.read_bytes();after=source.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise ValueError('A structure changed while being captured')
        staged=input_directory/(key+source.suffix.lower());staged.write_bytes(raw)
        inputs[key]={'sha256':hashlib.sha256(raw).hexdigest(),'name':source.name,'captured':str(staged.relative_to(output))}
        return str(staged)
    for key in ('cif','mcif','comparison_cif'):
        if request.get(key):request[key]=capture(key,request[key])
    for index,phase in enumerate(request['additional_phases'],1):
        phase['cif']=capture('additional_phase_'+str(index),phase['cif'])
    profile = selected_profile(request.get("configuration"), request.get("custom_profile"))
    local_profile = output / ("instrument" + Path(profile["profile"]).suffix)
    shutil.copyfile(profile["profile"], local_profile)
    config = {"id": profile["id"], "kind": "PNT", "profile": str(local_profile),
              "profile_sha256": profile["sha256"],
              "profile_bank": (request.get("profile_bank") if request.get("custom_profile") else profile.get("profile_bank")),
              "source_bank": profile.get("source_bank") or 1,
              "grid": {"d_min": request["d_min"], "d_max": request["d_max"],
                       "relative_step": request["relative_step"]}}
    if request['resolution_model']=='cycle_table':
        table=selected_resolution_table(profile)
        if request['d_min']<table['d_range_a'][0] or request['d_max']>table['d_range_a'][1]:
            raise ValueError('Selected numerical table requires d limits within '
                +str(table['d_range_a'][0])+'–'+str(table['d_range_a'][1])+' Å; extrapolation is unavailable')
        destination=output/'resolution.irf';shutil.copyfile(table['path'],destination)
        config.update(resolution_table=str(destination),resolution_table_sha256=table['sha256'])
    material = {key: request[key] for key in ("cif", "mcif", "isotopes", "moment_multiplier",
                 "crystallite_size_um", "microstrain", "march_dollase", "texture_axis", "additional_phases") if key in request}
    if request.get('prepared_magnetic') or request.get('prepared_structural'):
        prepared = request.get('prepared_magnetic') or request['prepared_structural'];source = Path(prepared['gpx']).resolve(strict=True)
        if (hashlib.sha256(source.read_bytes()).hexdigest() != prepared['gpx_sha256'] or
            inputs['cif']['sha256'] != prepared['parent_cif_sha256'] or profile['sha256'] != prepared['profile_sha256']):
            raise ValueError('Prepared magnetic model no longer matches the parent/configuration')
        staged=input_directory/'prepared.gpx';shutil.copyfile(source,staged)
        if hashlib.sha256(staged.read_bytes()).hexdigest()!=prepared['gpx_sha256']:
            raise ValueError('Prepared structure changed while being captured')
        material['gpx'] = str(staged)
        material['nuclear_cell_multiplier']=prepared.get('nuclear_cell_multiplier',1.)
    metadata, arrays = calculate(config, material, output / "model", output)
    comparisons = {}
    if request.get("comparison_cif"):
        other = dict(material, cif=request["comparison_cif"])
        other.pop("mcif", None)
        other.pop("gpx", None)
        other.pop("nuclear_cell_multiplier", None)
        other_meta, other_arrays = calculate(config, other, output / "comparison-model", output)
        # Both previews describe the same total sample mass, including mixtures.
        # A changed primary cell description must not change the plot's basis.
        comparison_scale=metadata['cell_mass_da']/other_meta['cell_mass_da']
        comparisons = {"comparison_nuclear": other_arrays["nuclear"]*comparison_scale,
                       "comparison_magnetic": other_arrays["magnetic"]*comparison_scale}
    calculated = arrays["nuclear"] + arrays["magnetic"]
    receipt = {"schema": "powgen-planning-result/v1", "synthetic": True,
               "configuration": profile["id"], "profile_sha256": profile["sha256"],
               "inputs": inputs, "calculation_backend": metadata["calculation_backend"],
               "tof_profile_kernel":metadata['tof_profile_kernel'],
               "legacy_tof_profile_translation":metadata.get('legacy_tof_profile_translation'),
               "isotope_selection": metadata["isotope_selection"], "isotope_options": metadata["isotope_options"],
               "cif_adp_normalizations": metadata.get('cif_adp_normalizations', []),
               "phases": metadata["phases"], "cell_mass_da": metadata["cell_mass_da"],
               "intensity_basis": metadata['intensity_basis'], "mixture": metadata['mixture'],
               "phase_mass_fractions": metadata['phase_mass_fractions'], "volume_rule": metadata['volume_rule'],
               "parent_model_artifact": 'model/'+metadata['reference_parent_artifact'],
               "mode": request["response_mode"], "looks": [], "warnings": [],
               "experimental_accuracy_certified": False, "counting_time_recommendation_available": False}
    if metadata.get('resolution_table'):receipt['resolution_table']=metadata['resolution_table']
    if metadata.get('neutron_amplitude_model'):
        receipt['neutron_amplitude_model'] = metadata['neutron_amplitude_model']
    if metadata['tof_profile_kernel']['evaluations']:
        receipt['tof_replay_note']='TOF-REPLAY.txt'
        (output/'TOF-REPLAY.txt').write_text(
            'TOF simulation arrays use the normalized RADAR-PD planning convolution recorded in result.json.\n'
            'The GPX stores model inputs; recalculating it with the installed native GSAS-II peak kernel produces a different curve.\n'
            'Replay the complete scenario with the matching planning runtime and source hashes:\n'
            'python -m gpx_recipe.planning.worker scenario.json NEW_OUTPUT_DIRECTORY\n'
            'Instrument response parameters and absolute count/noise/exposure accuracy require separate experimental qualification.\n')
    if any(isinstance(value,dict) for value in request['isotopes'].values()):
        receipt['warnings'].append('Isotope atom fractions assume independent random occupation and unpolarized nuclei. Atomic parameters and thermal spin incoherence are recorded model inputs; ordering, isotope-dependent structure/dynamics and spin-resolved resonances need separate evidence. The forward GPX requires the RADAR-PD planning isotope adapter; the portable scenario replays these selections.')
        (output/'ISOTOPE-REPLAY.txt').write_text(
            'This archive contains simulation inputs and outputs, not a standalone native GSAS-II isotope-mixture implementation.\n'
            'Replay with the matching RADAR-PD planning runtime and source hashes in result.json:\n'
            'python -m gpx_recipe.planning.worker scenario.json NEW_OUTPUT_DIRECTORY\n'
            'The forward GPX uses temporary isotope records that require the planning adapter. Do not use it directly as a native GSAS-II isotope-mixture model.\n'
            'The reference-parent GPX uses native isotope labels for symmetry discovery. Its natural/default stock for mixed elements is not the declared forward composition.\n'
            'Mixtures assume fixed structure, independent random isotope occupation and unpolarized nuclei. Thermal spin incoherence is not a spin-resolved resonance model.\n')
    if request.get('mcif'):
        receipt['warnings'].append('Magnetic structure is taken as supplied; its parent-subgroup relationship has not been independently established by this import.')
    if request.get('prepared_magnetic'):
        receipt['magnetic_state'] = request['prepared_magnetic']
    if request.get('prepared_structural'):
        receipt['structural_state'] = request['prepared_structural']
    if metadata['mixture']:
        receipt['warnings'].append('Crystalline phases use declared total-sample mass fractions and additive occupied solid volumes. Magnetic scattering belongs to the primary phase; phase fractions and specimen homogeneity require experimental verification.')
    if request["response_mode"] == "current_cycle":
        from .current_response import current_response, bundle_directory
        if request['beam'].get('proton_energy_gev', 1.3) != 1.3 or 'neutron_yield_ratio' in request['beam']:
            raise ValueError('Current response uses the measured 1.3 GeV cycle; other proton energies need their own response')
        campaign, components, response = current_response(config, metadata, arrays, request['sample'], request['seed'], container=request['container'],intensity_calibration=request['intensity_calibration'])
        shutil.copytree(bundle_directory(config['id']), output/'calibration')
        campaign.save_reference(output/'reference-model.npz')
        receipt.update(response=response,calibration_status=response['calibration_status'],intensity_unit=response['y_unit'])
        receipt['warnings'].extend(response['warnings'])
        looks, receipt['exposure_assumptions'] = cumulative_exposures(request)
        for i, look in enumerate(looks, 1):
            snapshot=campaign.advance(look['cumulative_charge_uah'])
            receipt['looks'].append(export_count_look(output/'reduced',i,arrays['x'],snapshot,config,look,campaign))
        np.savez_compressed(output/'curves.npz',x=arrays['x'],d=arrays['d'],expected=campaign.expected,
            nuclear=components['nuclear']/components['normalization'],
            magnetic=components['magnetic']/components['normalization'],
            incoherent=components['incoherent']/components['normalization'],
            residual_background=(components['blank']-components['background_subtraction'])/components['normalization'],
            raw_rate=campaign.rate,raw_nuclear_rate=components['nuclear'],raw_magnetic_rate=components['magnetic'],
            raw_background_rate=components['incoherent']+components['blank'],
            tof_edges_us=components['tof_edges_us'],d_edges_a=components['d_edges_a'])
        if comparisons:
            other_campaign, _, _ = current_response(config,other_meta,other_arrays,request['sample'],request['seed'],container=request['container'],intensity_calibration=request['intensity_calibration'])
            np.savez_compressed(output/'comparison-curves.npz',d=arrays['d'],expected=other_campaign.expected,
                                raw_rate=other_campaign.rate)
    elif request["response_mode"] == "historical_estimate":
        # The available historical response is explicitly restricted. It must
        # never silently provide rates for another guide, band or grouping.
        if profile.get("mode") != "HR" or profile.get("frequency") != 60 or profile.get("wavelength") not in (.8, 1.5, 2.665):
            raise ValueError("No historical count response is available for this exact configuration; use a profile preview")
        band = { .8: [0.267, 1.333], 1.5: [0.967, 2.033], 2.665: [2.132, 3.198]}[profile["wavelength"]]
        source_bank = {.8: 1, 1.5: 2, 2.665: 3}[profile["wavelength"]]
        if config["source_bank"] != source_bank or profile.get("grouping") != "standard":
            raise ValueError("Historical response does not map to the selected grouping/bank")
        config["wavelength_band_a"] = band
        settings = {"calibration": f"calibration/pg3-{53997 + source_bank}/calibration.json"}
        beam = request["beam"]
        if beam.get("proton_energy_gev", 1.0) != 1.0:
            if "neutron_yield_ratio" not in beam:
                raise ValueError("The historical response was measured at 1 GeV. A changed proton energy requires an independently supplied neutron-yield ratio")
            settings["neutron_yield_ratio"] = beam["neutron_yield_ratio"]
        rates, response = powgen_response(config, metadata, arrays, request["sample"], settings, Path(__file__).parent / "data")
        receipt.update(response=response, calibration_status=response["calibration_status"],
                       intensity_unit=response["y_unit"])
        receipt['warnings'].extend(response.get('warnings',[]))
        acc = Accumulator(rates["rate"], rates["response"], request["seed"], background_rate=rates["background_rate"])
        looks, receipt['exposure_assumptions'] = cumulative_exposures(request)
        for i, look in enumerate(looks, 1):
            snap = acc.advance(look['cumulative_charge_uah'])
            receipt["looks"].append(export_snapshot(output / "reduced", i, arrays["x"], snap, config,
                {**look, "exposure_unit": "microampere_hour",
                 "calibration_status": response["calibration_status"],
                 "noise_interpretation": "Effective reduced-count approximation; these are not reconstructed raw neutron counts"}, acc))
        np.savez_compressed(output / "curves.npz", x=arrays["x"], d=arrays["d"],
                            expected=rates["nuclear"] + rates["magnetic"] + rates["background_rate"] / rates["response"],
                            nuclear=rates["nuclear"], magnetic=rates["magnetic"])
        if comparisons:
            other_rates, _ = powgen_response(config, other_meta, other_arrays, request["sample"], settings, Path(__file__).parent / "data")
            np.savez_compressed(output / "comparison-curves.npz", d=arrays["d"], expected=other_rates["nuclear"] + other_rates["magnetic"] + other_rates["background_rate"] / other_rates["response"])
    else:
        receipt.update(calibration_status="profile_only_no_count_response", intensity_unit=
            "GSAS-II intensity per primary reference-cell mass equivalent" if metadata['mixture'] else "GSAS-II intensity per reference nuclear cell")
        receipt['warnings'].append("The selected profile predicts peak shapes. Absolute intensity and counting-time noise require a qualified response calibration.")
        np.savez_compressed(output / "curves.npz", x=arrays["x"], d=arrays["d"], expected=calculated,
                            nuclear=arrays["nuclear"], magnetic=arrays["magnetic"], **comparisons)
    portable=copy.deepcopy(request)
    if portable.get('custom_profile'):portable['custom_profile']=local_profile.name
    for key in ('cif','mcif','comparison_cif'):
        if key in inputs:portable[key]=inputs[key]['captured']
    for index,phase in enumerate(portable['additional_phases'],1):
        phase['cif']=inputs['additional_phase_'+str(index)]['captured']
    for key in ('prepared_magnetic','prepared_structural'):
        if portable.get(key):portable[key]['gpx']='inputs/prepared.gpx'
    write_json(output / "scenario.json", portable)
    receipt['planning_source_sha256']={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
        for name in ('worker.py','gsas.py','instrument.py','fullprof_lookup.py','legacy_tof.py','tof_profile.py','tof_backend.py','response.py','physics.py','counting.py','contracts.py','exposure_inputs.py','current_response.py','empirical_counts.py','standard_counts.py','focused_reduction.py','count_response.py','io.py','mixture.py','isotopes.py','cif_adps.py','nuclear.py')}
    artifacts = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in output.rglob("*") if p.is_file() and p.suffix in {".npz", ".gpx", ".gsa", ".instprm", ".prm", ".irf", ".json", '.cif', '.mcif', '.txt', '.h5'}}
    receipt["artifacts"] = artifacts
    write_json(output / "result.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Existing simulation campaigns are immutable; choose a new output directory')
    try:
        request=json.loads(args.scenario.read_text())
        for key in ('cif','mcif','comparison_cif','custom_profile'):
            if request.get(key) and not Path(request[key]).is_absolute():request[key]=str((args.scenario.parent/request[key]).resolve())
        for phase in request.get('additional_phases',[]):
            if isinstance(phase,dict) and isinstance(phase.get('cif'),str) and not Path(phase['cif']).is_absolute():
                phase['cif']=str((args.scenario.parent/phase['cif']).resolve())
        for key in ('prepared_magnetic','prepared_structural'):
            if request.get(key) and not Path(request[key]['gpx']).is_absolute():request[key]['gpx']=str((args.scenario.parent/request[key]['gpx']).resolve())
        result = run(request, args.output)
        print(json.dumps({"result": str(args.output / "result.json"), "status": result["calibration_status"]}))
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "failure.json", {"error_type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
