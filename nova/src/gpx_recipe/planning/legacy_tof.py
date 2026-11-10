"""Preserve documented isotropic GSAS TOF profile3 coefficients on import.

The installed GSAS-II legacy reader omits sigma0 and the Lorentzian terms.
GSAS technical manual (2004), pp146-148, defines the Gaussian variance and
Lorentzian FWHM polynomials. Native .instprm files already express these fields.
This translation restores serialized values; it fits no response parameters.
"""
import hashlib
import math
from pathlib import Path


VERSION='powgen-documented-legacy-TOF-profile3/1'
MANUAL='https://subversion.xray.aps.anl.gov/EXPGUI/gsas/all/GSAS%20Manual.pdf'
FIELDS=('alpha','beta-0','beta-1','sig-0','sig-1','sig-2','Z','X','Y')


def restore_profile3(path,bank,instrument):
    """Restore all nine isotropic coefficients and record their exact source.

Nonzero anisotropic/strain/fault terms need a separate native interpretation
and fail before any mutation. Tabulated tails retain the existing calculation
adapter's independent lookup-domain/semantics checks.
    """
    path=Path(path)
    if path.suffix.lower() not in {'.prm','.inst','.ins'} or not str(instrument['Type'][0]).startswith('PNT'):
        return None
    if isinstance(bank,bool) or not isinstance(bank,int) or not 1<=bank<=99:
        raise ValueError('An explicit legacy TOF bank between 1 and 99 is required')
    if path.stat().st_size>2*1024**2:
        raise ValueError('Legacy instrument profile exceeds the bounded input size')
    raw=path.read_bytes();cards={}
    for line in raw.decode('utf-8-sig').splitlines():
        if not line.startswith('INS'):continue
        if len(line)<12:raise ValueError('Truncated legacy instrument record')
        key=line[:12].rstrip()
        if key in cards:raise ValueError('Duplicate legacy instrument record: '+key)
        cards[key]=line[12:].strip()
    prefix=f'INS {bank:2}'
    if prefix+'PRCF1' in cards:
        header=cards[prefix+'PRCF1'];keys=[prefix+f'PRCF1{n}' for n in range(1,7)]
    elif prefix+'PRCF' in cards:
        header=cards[prefix+'PRCF'];keys=[prefix+f'PRCF{n:2}' for n in range(1,7)]
    else:raise ValueError('Selected legacy TOF bank has no profile descriptor')
    try:
        descriptor=header.split();function,count=int(descriptor[0]),int(descriptor[1])
        if abs(function)!=3 or count!=21:
            raise ValueError('Only the documented 21-coefficient legacy TOF profile 3 is supported')
        coefficients=[float(value) for key in keys for value in cards[key].split()]
    except (KeyError,IndexError,TypeError) as error:
        raise ValueError('Incomplete legacy TOF profile3 records') from error
    if len(coefficients)!=21 or not all(math.isfinite(v) for v in coefficients):
        raise ValueError('Exactly 21 finite legacy TOF profile 3 coefficients are required')
    unsupported=[{'coefficient':i+1,'value':v} for i,v in enumerate(coefficients) if i>=9 and v!=0.]
    if unsupported:
        raise ValueError('Nonzero legacy anisotropic/strain/fault profile terms require a separate supported model: '+str(unsupported))
    if any(key not in instrument or not isinstance(instrument[key],(list,tuple)) or len(instrument[key])<2 for key in FIELDS):
        raise ValueError('Native TOF instrument fields are incomplete')
    before={key:[float(instrument[key][0]),float(instrument[key][1])] for key in FIELDS}
    after=dict(zip(FIELDS,coefficients[:9]))
    # The original and native kernels both use Gaussian variance and Lorentzian
    # FWHM: gamma0 -> Z, gamma1 -> X, gamma2 -> Y. There is no fitted factor.
    for key,value in after.items():instrument[key]=[value,value,False]
    return {'schema':VERSION,'source':str(path),'source_sha256':hashlib.sha256(raw).hexdigest(),
        'adapter_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'bank':bank,'function':function,'coefficient_count':21,'source_coefficients':coefficients,
        'native_before_initial_and_current':before,'native_after':after,
        'Gaussian_variance_law':'sig-0+sig-1*d**2+sig-2*d**4',
        'Lorentzian_FWHM_law':'Z+X*d+Y*d**2','Lorentzian_unit_factor':1.,
        'parameter_values_fitted':False,'manual_source':MANUAL,'manual_printed_pages':[146,147,148],
        'experimental_response_qualified':False}
