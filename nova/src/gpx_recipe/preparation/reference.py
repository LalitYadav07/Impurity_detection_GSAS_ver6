"""Create a theory-only native context from a CIF and bank instrument profile."""
from pathlib import Path
import hashlib
import math

import numpy as np

MAXIMUM_TOF_MICROSECONDS = 2_000_000.


def validate_grid(*, kind, lower, upper, bins=2500, bank=1):
    """Validate the same finite reference context before dispatch or native use."""
    if (kind not in {'PNC', 'PNT'} or type(bins) is not int or not 3 <= bins <= 25000
            or type(bank) is not int or not 1 <= bank <= 10000
            or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
                for v in (lower, upper)) or not 0 < lower < upper
            or (kind == 'PNC' and upper >= 180)
            or (kind == 'PNT' and upper > MAXIMUM_TOF_MICROSECONDS)):
        raise ValueError('An explicit supported bank, histogram type and bounded physical grid are required')
    return dict(kind=kind,lower=float(lower),upper=float(upper),bins=bins,bank=bank)


def create_reference(cif, instrument, output, *, kind, lower, upper, bins=2500, bank=1):
    """Ranges are degrees 2theta (PNC) or microseconds (PNT).

    These empty histogram arrays specify a planned rendering grid. They are
    never passed to the measured refinement adapter or published as scans.
    """
    validate_grid(kind=kind,lower=lower,upper=upper,bins=bins,bank=bank)
    cif, instrument, output = Path(cif), Path(instrument), Path(output)
    for path, limit in ((cif, 10*1024*1024), (instrument, 1024*1024)):
        if not path.is_file() or not 0 < path.stat().st_size <= limit:
            raise ValueError('Missing or oversized preparation input')
    checksums = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (cif, instrument)]
    from GSASII import GSASIIscriptable as G
    from ..execution import freeze
    from ..live.gpx_index import install_fresh_gpx_index
    install_fresh_gpx_index()
    output.mkdir(parents=True, exist_ok=False)
    target = output/'reference.gpx'
    project = G.G2Project(newgpx=str(target))
    phase = project.add_phase(str(cif))
    if len(project.phases()) != 1 or phase.data['General']['Type'] != 'nuclear' or phase.data['General'].get('Modulated'):
        raise ValueError('Select one unmodulated nuclear parent CIF')
    unit = 1000. if kind == 'PNT' else 1.
    # The native simulation convenience API rejects Tmax > 200 ms as a
    # historical units-error guard. Real POWGEN reduced ranges can exceed it.
    # Use that API for its reviewed instrument/HAP initialization, then set
    # the requested empty native logarithmic grid before any calculation.
    # No measured arrays are resampled, and native scientific functions stay
    # unchanged. Existing ranges retain their original API-generated arrays.
    extended_tof = kind == 'PNT' and upper > 200000.
    seed_upper = min(upper, 200000.) if extended_tof else upper
    seed_lower = min(lower, seed_upper/2.) if extended_tof else lower
    histogram = project.add_simulated_powder_histogram('Preparation reference', str(instrument),
        seed_lower/unit, seed_upper/unit, Npoints=bins, ibank=bank, scale=1., phases=[phase])
    if histogram.InstrumentParameters['Type'][0] != kind:
        raise ValueError('Instrument profile type differs from the planned grid')
    data = histogram.data['data'][1]
    if extended_tof:
        # Match the native TOF convention: logarithmic bin centers beginning
        # at lower, with upper the exclusive edge, not the last center.
        data[0] = np.exp(np.arange(bins)*((np.log(upper/unit)-np.log(lower/unit))/bins)
                         + np.log(lower))
        for i in range(1, 6):
            data[i] = np.zeros_like(data[0])
        actual_range = [float(data[0][0]), float(data[0][-1])]
        histogram.data['Limits'] = [actual_range.copy(), actual_range.copy()]
    if len(data[0]) > bins+1:
        raise ValueError('Native simulation grid exceeded its declared size')
    for i in (1, 3, 4, 5):
        data[i] = np.zeros_like(data[0])
    data[2] = np.ones_like(data[0])
    histogram.data['data'][0]['Dummy'] = True
    freeze(project)
    project.data.setdefault('Covariance', {})['data'] = {}
    project.save()
    if checksums != [hashlib.sha256(path.read_bytes()).hexdigest() for path in (cif, instrument)]:
        raise ValueError('Preparation input changed during native import')
    return dict(path=str(target), parent_phase=phase.name, histogram=histogram.name,
        cif_sha256=checksums[0], instrument_sha256=checksums[1], kind=kind, bank=bank,
        bins=len(data[0]), native_range=[float(data[0][0]), float(data[0][-1])],
        observations_included=False, scientific_result=False)
