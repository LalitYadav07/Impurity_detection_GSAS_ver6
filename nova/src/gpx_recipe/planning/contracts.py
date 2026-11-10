"""Bounded, explicit scenarios for a pre-experiment POWGEN calculation."""
from __future__ import annotations

import math
from pathlib import Path

from .physics import Cylinder
from .mixture import additional_phases
from .exposure_inputs import validate_exposure


def positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be finite and positive")
    return float(value)


def validate_scenario(value):
    if not isinstance(value, dict) or value.get("schema") != "powgen-planning-scenario/v1":
        raise ValueError("A POWGEN planning scenario is required")
    allowed = {"schema", "configuration", "custom_profile", "profile_bank", "cif", "mcif", "comparison_cif",
               "sample", "container", "intensity_calibration", "seconds", "charge_uah", "beam", "d_min", "d_max", "relative_step", "seed", "moment_multiplier",
               "isotopes", "crystallite_size_um", "microstrain", "march_dollase", "texture_axis", "response_mode", "resolution_model", "prepared_magnetic", "prepared_structural", "additional_phases"}
    if set(value) - allowed:
        raise ValueError("Unknown planning fields: " + ", ".join(sorted(set(value) - allowed)))
    result = dict(value)
    for key in ("cif", "mcif", "comparison_cif", "custom_profile"):
        if result.get(key):
            path = Path(result[key]).resolve(strict=True)
            if not path.is_file() or not 0 < path.stat().st_size <= 20 * 1024 * 1024:
                raise ValueError("A bounded nonempty " + key + " file is required")
            result[key] = str(path)
    if not result.get("cif"):
        raise ValueError("Upload the nuclear CIF")
    result['additional_phases'] = additional_phases(result.get('additional_phases', []))
    for phase in result['additional_phases']:
        path = Path(phase['cif']).resolve(strict=True)
        if not path.is_file() or not 0 < path.stat().st_size <= 20 * 1024 * 1024:
            raise ValueError('A bounded nonempty additional phase CIF is required')
        phase['cif'] = str(path)
    if result.get('prepared_magnetic') and result.get('prepared_structural'):
        raise ValueError('Compose magnetic ordering on a separately verified distorted parent')
    if result.get('prepared_magnetic') or result.get('prepared_structural'):
        prepared = result.get('prepared_magnetic') or result['prepared_structural']
        if result.get('mcif'): raise ValueError('Choose a supplied magnetic CIF or a prepared subgroup')
        if not isinstance(prepared,dict) or not {'gpx','gpx_sha256','parent_cif_sha256','profile_sha256'} <= set(prepared):
            raise ValueError('A provenance-bound prepared planning state is required')
        multiplier = prepared.get('nuclear_cell_multiplier',1.)
        if type(multiplier) not in (int,float) or not math.isfinite(multiplier) or not 1/64 <= multiplier <= 64:
            raise ValueError('Prepared nuclear cell multiplier must be between 1/64 and 64')
        if result.get('prepared_structural') and 'nuclear_cell_multiplier' not in prepared:
            raise ValueError('A structural state must declare its original nuclear cell reference')
    result["d_min"] = positive(result.get("d_min", .55), "Minimum d")
    result["d_max"] = positive(result.get("d_max", 10.), "Maximum d")
    result["relative_step"] = positive(result.get("relative_step", .0008), "Relative bin width")
    if result["d_max"] <= result["d_min"]:
        raise ValueError("Maximum d must exceed minimum d")
    bins = math.ceil(math.log(result["d_max"] / result["d_min"]) / result["relative_step"])
    if not 3 <= bins <= 50000:
        raise ValueError("Choose a d range and binning containing 3–50,000 bins")
    result.update(validate_exposure(result))
    seed = result.get("seed", 42)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Use an integer seed in [0, 2^32)")
    result["seed"] = seed
    sample = result.get("sample", {})
    if not isinstance(sample, dict):
        raise ValueError("Declare sample geometry")
    for key, number in sample.items():
        if type(number) not in (int, float) or not math.isfinite(number):
            raise ValueError("Sample geometry must contain finite numeric values")
    geometry = Cylinder(**sample)
    result["sample"] = dict(vars(geometry))
    if type(result.get('moment_multiplier',1.)) not in (int,float):
        raise ValueError('Moment multiplier must be numeric')
    result["moment_multiplier"] = float(result.get("moment_multiplier", 1.))
    if not math.isfinite(result["moment_multiplier"]) or not 0 <= result["moment_multiplier"] <= 10:
        raise ValueError("Moment multiplier must be between 0 and 10")
    from .isotopes import selections
    result["isotopes"] = selections(result.get("isotopes", {}))
    if result.get("profile_bank") is not None:
        if type(result["profile_bank"]) is not int or not 1 <= result["profile_bank"] <= 100:
            raise ValueError("Declare an integer profile bank between 1 and 100")
    result["response_mode"] = result.get("response_mode", "profile_only")
    if result["response_mode"] not in {"profile_only", "historical_estimate", "current_cycle"}:
        raise ValueError("Choose a profile preview, current-cycle conditional simulation or historical estimate")
    result['resolution_model']=result.get('resolution_model','profile')
    if (result['resolution_model'] not in {'profile','cycle_table'} or
            (result['resolution_model']=='cycle_table' and result['response_mode']=='historical_estimate')):
        raise ValueError('Choose a registered profile or a current-cycle numerical table; historical count estimates use their original profile')
    result['container']=result.get('container','PAC06')
    if result['container'] not in {'PAC06','none'}:
        raise ValueError('Choose the measured PAC06 container or no sample can')
    result['intensity_calibration']=result.get('intensity_calibration','vanadium')
    if (result['intensity_calibration'] not in {'vanadium','silicon_63584'} or
            (result['response_mode']!='current_cycle' and result['intensity_calibration']!='vanadium')):
        raise ValueError('Standard intensity normalization is available only for the current-cycle count model')
    if result['resolution_model']=='cycle_table' and result['intensity_calibration']!='vanadium':
        raise ValueError('The numerical table currently uses the vanadium response; the Si normalization belongs to its original profile')
    for key in ("crystallite_size_um", "march_dollase"):
        if key in result: result[key] = positive(result[key], key)
    if "microstrain" in result:
        number = result["microstrain"]
        if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
            raise ValueError("Microstrain must be finite and nonnegative")
    if "texture_axis" in result:
        axis=result['texture_axis']
        if not isinstance(axis,list) or len(axis)!=3 or any(type(v) is not int or abs(v)>100 for v in axis) or not any(axis):
            raise ValueError('Texture axis must contain three bounded integer Miller indices and cannot be zero')
    return result
