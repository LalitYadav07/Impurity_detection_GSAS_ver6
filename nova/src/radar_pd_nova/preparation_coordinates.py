"""Checked range conversion; preparation still uses its native reference grid."""
import math

import numpy as np

from .diffraction_coordinates import CwCalibration, TofCalibration, coordinate_values

AXES={'native','d_angstrom','q_inv_angstrom'}
NATIVE={'PNT':'tof_us','PNC':'two_theta_deg'}
UNITS={'tof_us':'µs','two_theta_deg':'° 2θ','d_angstrom':'Å','q_inv_angstrom':'Å⁻¹'}


def convert_limits(path,*,kind,bank,source,target,lower,upper,expected_sha256):
    """Convert only endpoints using the explicitly selected profile block.

    Q runs opposite to d and native coordinates. Returning sorted endpoints
    changes no intensities, measured arrays or reference binning convention.
    """
    if kind not in NATIVE or source not in AXES or target not in AXES:
        raise ValueError('Choose a supported neutron profile and range coordinate')
    if (any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v)
            for v in (lower,upper)) or not 0<lower<upper):
        raise ValueError('Enter positive, increasing range limits')
    source=NATIVE[kind] if source=='native' else source
    target=NATIVE[kind] if target=='native' else target
    calibration=None
    if source!=target and NATIVE[kind] in {source,target}:
        reader=TofCalibration if kind=='PNT' else CwCalibration
        try:
            calibration=reader.from_profile(path,source_bank=bank,profile_bank=bank,
                                            expected_sha256=expected_sha256)
        except (KeyError,TypeError,OverflowError) as exc:
            raise ValueError('The selected profile block lacks complete numeric coordinate parameters') from exc
    converted=np.sort(coordinate_values([lower,upper],source,target,calibration))
    if not np.isfinite(converted).all() or not 0<converted[0]<converted[1]:
        raise ValueError('This range does not map to positive, distinct native limits')
    return converted.tolist()


def native_range_label(grid):
    return f"{grid['lower']:g}–{grid['upper']:g} {UNITS[NATIVE[grid['kind']]]}"
