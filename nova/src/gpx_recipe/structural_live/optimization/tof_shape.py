"""Cell-independent inputs to the reviewed isotropic native TOF peak shape.

This is a native calculation specification, not a saved response or a native
origin certificate. It contains no observations, lattice, atomic coordinates,
space group, background or executable object. Imported specifications can only
request fresh calculations with the installed, checked native functions.
"""
from __future__ import annotations

import copy
import json
import numpy as np

from .native_cell_proposals import ProposalUnavailable, _hash
from .native_profile_capture import RIETVELD_DRIVERS
from .native_tof_scalars import _nested, FUNCTION_SOURCES, WIDTH_SOURCE
from ...symmetry.contracts import digest

SCHEMA='radar-isotropic-native-tof-shape/v1'
BANK_PARAMETERS=('difC','alpha','beta-0','beta-1','beta-q',
                 'sig-0','sig-1','sig-2','sig-q','X','Y','Z')
SAMPLE_PARAMETERS=('Size;i','Size;mx','Mustrain;i','Mustrain;mx')
MAXIMUM_POINTS=25000


def validate(value):
    if (not isinstance(value,dict) or set(value)!={'schema','bank','sample'}
            or value['schema']!=SCHEMA or not isinstance(value['bank'],dict)
            or set(value['bank'])!=set(BANK_PARAMETERS)
            or not isinstance(value['sample'],dict)
            or set(value['sample'])!=set(SAMPLE_PARAMETERS)
            or len(json.dumps(value,allow_nan=False))>8192):
        raise ValueError('A bounded isotropic TOF shape specification is required')
    if any(type(x) not in (int,float) or not np.isfinite(x)
           for group in ('bank','sample') for x in value[group].values()):
        raise ValueError('TOF shape inputs must be finite native scalars')
    sample=value['sample']
    if (value['bank']['difC']<=0 or sample['Size;i']<=0
            or sample['Mustrain;i']<0
            or any(not 0<=sample[k]<=1 for k in ('Size;mx','Mustrain;mx'))):
        raise ValueError('TOF shape size, strain and mixing inputs are not physical')
    return value


class IsotropicTOFShape:
    """The same native shape for all cells, sites and compatible subgroups.

    The supported GSAS-II branches depend only on d, instrument coefficients
    and four isotropic sample coefficients. Zero/TOF calibration still places
    each current reflection in the caller; it is not a peak-shape input.
    """
    def __init__(self,native,histogram_name,phase_name):
        native._check()
        histogram=native.histo_phases[0][histogram_name]
        phase=native.histo_phases[1][phase_name]
        hfx=f":{histogram['hId']}:";phfx=f"{phase['pId']}:{histogram['hId']}:"
        if (native.controls.get(hfx+'histType')!='PNT'
                or native.controls.get(phfx+'SizeType')!='isotropic'
                or native.controls.get(phfx+'MustrainType')!='isotropic'
                or native.histdict.get(hfx+'pdabc') is not None):
            raise ProposalUnavailable('Cell-independent TOF shapes require the reviewed isotropic profile without a lookup table')
        value=dict(schema=SCHEMA,
            bank={k:float(native.parms[hfx+k]) for k in BANK_PARAMETERS},
            sample={k:float(native.parms[phfx+k]) for k in SAMPLE_PARAMETERS})
        self._initialize(value,native.math)
        native._check()

    @classmethod
    def from_specification(cls,value,math):
        """Recalculate with installed native code; never deserialize a function."""
        instance=cls.__new__(cls);instance._initialize(value,math);return instance

    def _initialize(self,value,math):
        self._spec=copy.deepcopy(validate(value));self.math=math
        driver=math.getPowderProfile;sample=math.GetSampleSigGam;widths=math.G2pwd.getWidthsTOF
        if (_hash(driver) not in RIETVELD_DRIVERS or driver.__globals__ is not math.__dict__
                or _hash(sample)!=FUNCTION_SOURCES['GetSampleSigGam']
                or _hash(widths)!=WIDTH_SOURCE):
            raise ProposalUnavailable('Installed TOF shape functions are not qualified')
        self._functions=(driver,sample,widths)
        # The only closure variable is SGData. It is unused by both isotropic
        # branches; no sample-specific symmetry or cell is retained here.
        self._widths=_nested(driver,'GetReflSigGamTOF',{'SGData':{}})
        self._exponentials=_nested(driver,'GetReflAlpBet',{})
        self._parms={':0:'+k:v for k,v in self._spec['bank'].items()}
        self._parms.update({'0:0:'+k:v for k,v in self._spec['sample'].items()})
        self._controls={':0:histType':'PNT','0:0:SizeType':'isotropic',
                        '0:0:MustrainType':'isotropic'}

    @property
    def specification(self):return copy.deepcopy(self._spec)

    @property
    def identity(self):return digest(self._spec)

    def _check(self):
        if (self.math.getPowderProfile,self.math.GetSampleSigGam,
                self.math.G2pwd.getWidthsTOF)!=self._functions:
            raise ProposalUnavailable('Native TOF shape functions changed')

    def at(self,d):
        """Return native alpha, beta, variance and Lorentzian width at d."""
        self._check();d=np.asarray(d,float)
        if (d.ndim!=1 or not 1<=len(d)<=MAXIMUM_POINTS or not np.isfinite(d).all()
                or np.any(d<=0)):
            raise ValueError('A bounded positive one-dimensional d-spacing grid is required')
        rows=np.zeros((18,len(d)));rows[4]=d
        metric=np.eye(3)
        rows[6:8]=self._widths(rows,0,metric,metric,'0:0:',self._controls,self._parms,{},':0:')
        rows[12:14]=self._exponentials(rows,0,':0:',self._parms)
        result=np.asarray([rows[i] for i in (12,13,6,7)]).T.copy()
        self._check()
        if not np.isfinite(result).all():raise ProposalUnavailable('Native TOF shape returned nonfinite coefficients')
        return result

    def support(self,d):
        parameters=self.at(d)
        _,low,high=self._functions[2](np.zeros(len(parameters)),*parameters.T)
        return parameters,np.asarray(low,float),np.asarray(high,float)
