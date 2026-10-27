"""Display coordinates from explicitly mapped, checksum-verified profiles.

GSAS-II Dsp2pos: t = Zero + difC*d + difA*d**2 + difB/d.
CW: 2theta = 2 asin(Lam/(2d)) + Zero, in degrees.
https://github.com/AdvancedPhotonSource/GSAS-II/blob/main/GSASII/GSASIIlattice.py
These functions only relabel/reorder measured samples. They do not resample,
change intensity units, or supply data to the statistical detector.
"""
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
import math
import re

import numpy as np


@dataclass(frozen=True)
class CwCalibration:
    """Neutron CW Bragg conversion using the mapped calibrated Lam and Zero."""
    wavelength: float
    zero: float = 0.0
    source_bank: int = 1
    profile_bank: int = 1
    profile_sha256: str = ''
    label: str = 'Mapped neutron CW profile'

    def __post_init__(self):
        if not math.isfinite(self.wavelength) or self.wavelength <= 0 or not math.isfinite(self.zero):
            raise ValueError('CW conversion requires finite calibrated wavelength and zero')
        if any(isinstance(b,bool) or int(b)!=b or b<1 for b in (self.source_bank,self.profile_bank)):
            raise ValueError('CW conversion requires explicit positive bank labels')

    @classmethod
    def from_profile(cls,path,*,source_bank=1,profile_bank=1,expected_sha256):
        path=Path(path)
        if path.suffix.lower()!='.instprm' or not 0<path.stat().st_size<=1024*1024:
            raise ValueError('CW coordinate conversion needs one bounded modern .instprm')
        before=path.stat();raw=path.read_bytes();after=path.stat()
        checksum=sha256(raw).hexdigest()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or not expected_sha256 or checksum!=expected_sha256:
            raise ValueError('CW profile changed or does not match the analysis checksum')
        fields={}
        for line in raw.decode('utf-8-sig').splitlines():
            if line.lstrip().startswith('#'):continue
            for term in line.split(';'):
                if ':' not in term:continue
                key,value=(v.strip() for v in term.split(':',1))
                if key in fields:raise ValueError('Select one unambiguous CW profile block')
                fields[key]=value
        bank=float(fields.get('Bank',1))
        if fields.get('Type')!='PNC' or not math.isfinite(bank) or bank!=profile_bank:
            raise ValueError('The mapped neutron CW profile bank is unavailable')
        return cls(float(fields['Lam']),float(fields.get('Zero',0)),source_bank,profile_bank,checksum,path.name)

    def spacing(self,two_theta):
        angle=np.asarray(two_theta,dtype=float)-self.zero
        if not np.all(np.isfinite(angle)) or np.any((angle<=0)|(angle>=180)):
            raise ValueError('Measured 2theta is outside the calibrated physical CW range')
        return self.wavelength/(2*np.sin(np.radians(angle/2)))

    def two_theta(self,d):
        d=np.asarray(d,dtype=float)
        ratio=self.wavelength/(2*d)
        if not np.all(np.isfinite(ratio)) or np.any((ratio<=0)|(ratio>=1)):
            raise ValueError('d-spacing is outside the CW Bragg range')
        return np.degrees(2*np.arcsin(ratio))+self.zero


@dataclass(frozen=True)
class TofCalibration:
    difc: float
    difa: float = 0.0
    difb: float = 0.0
    zero: float = 0.0
    source_bank: int = 1
    profile_bank: int = 1
    profile_sha256: str = ""
    label: str = "Mapped TOF profile"

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.difc,self.difa,self.difb,self.zero)) or self.difc <= 0:
            raise ValueError("Invalid TOF coordinate coefficients")
        if any(isinstance(b,bool) or int(b)!=b or b<1 for b in (self.source_bank,self.profile_bank)):
            raise ValueError("Explicit physical and profile Bank labels are required")

    @classmethod
    def from_profile(cls,path,*,source_bank,profile_bank,expected_sha256):
        path=Path(path)
        if path.suffix.lower() not in {".instprm",".prm",".inst",".ins"}:
            raise ValueError("Coordinate conversion needs a supported neutron TOF profile")
        before=path.stat()
        if not 0 < before.st_size <= 1024*1024:
            raise ValueError("Instrument profile exceeds the coordinate-reader bound")
        raw=path.read_bytes();after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise ValueError("Instrument profile changed during coordinate inspection")
        checksum=sha256(raw).hexdigest()
        if not expected_sha256 or checksum != expected_sha256:
            raise ValueError("Instrument profile checksum does not match its bank mapping")
        text=raw.decode("utf-8-sig")
        if path.suffix.lower()!=".instprm":
            # GSASIIfiles.SetPowderInstParms reads legacy TOF ICONS as
            # difC, difA, Zero; difB is zero. Bank labels are fixed columns.
            lines=text.splitlines()
            kinds=[line[12:].strip() for line in lines if line[:12].strip()=="INS   HTYPE"]
            icons=[line[12:].split() for line in lines if line.startswith("INS")
                and line[3:6].strip()==str(profile_bank) and line[6:12]==" ICONS"]
            if len(kinds)!=1 or not kinds[0].startswith("PNT") or len(icons)!=1 or len(icons[0])<3:
                raise ValueError("A unique mapped legacy neutron TOF ICONS record is required")
            c,a,z=map(float,icons[0][:3])
            return cls(c,a,0,z,int(source_bank),int(profile_bank),checksum,path.name)
        blocks={}
        for chunk in re.split(r"(?mi)^(?=#\s*Bank\s+\d+)",text):
            fields={}
            for line in chunk.splitlines():
                if line.lstrip().startswith("#"):
                    continue
                for field in line.split(";"):
                    if ":" not in field:
                        continue
                    key,value=field.split(":",1);key=key.strip()
                    if key not in {"Type","Bank","difC","difA","difB","Zero"}:
                        continue
                    if key in fields:
                        raise ValueError("Duplicate or ambiguous coordinate profile fields")
                    fields[key]=value.strip()
            if not fields:
                continue
            if fields.get("Type") != "PNT" or not {"Bank","difC"}.issubset(fields):
                raise ValueError("Coordinate conversion requires a neutron TOF profile block")
            bank=float(fields["Bank"])
            if not math.isfinite(bank) or bank<1 or int(bank)!=bank or int(bank) in blocks:
                raise ValueError("Ambiguous profile Bank labels")
            blocks[int(bank)]=fields
        if profile_bank not in blocks:
            raise ValueError("Mapped instrument Bank label is absent")
        fields=blocks[profile_bank]
        return cls(float(fields["difC"]),float(fields.get("difA",0)),float(fields.get("difB",0)),
            float(fields.get("Zero",0)),int(source_bank),int(profile_bank),checksum,path.name)

    def tof(self,d):
        d=np.asarray(d,dtype=float)
        return self.zero+self.difc*d+self.difa*d*d+self.difb/d

    def spacing(self,tof):
        """Invert only the positive, increasing physical branch; reject invalid ranges."""
        tof=np.asarray(tof,dtype=float)
        if not np.all(np.isfinite(tof)) or not tof.size:
            raise ValueError("Finite TOF coordinates are required")
        roots=np.roots([2*self.difa,self.difc,0,-self.difb])
        stationary=sorted(r.real for r in roots if abs(r.imag)<1e-8 and r.real>1e-12)
        bounds=[1e-12,*stationary,float("inf")]
        branches=[]
        for lo,hi in zip(bounds,bounds[1:]):
            test=math.sqrt(lo*hi) if math.isfinite(hi) else max(1,lo*2)
            if self.difc+2*self.difa*test-self.difb/test**2 > 0:
                branches.append((lo,hi))
        if len(branches)!=1:
            raise ValueError("TOF calibration has no unique increasing physical branch")
        lo,branch_end=branches[0];lo*=1+1e-10
        if math.isfinite(branch_end):
            branch_end*=1-1e-10
        hi=min(branch_end,max(lo*2,float(np.max(tof)-self.zero)/self.difc*2,1))
        for _ in range(80):
            if self.tof(hi)>=np.max(tof) or hi>=branch_end:
                break
            hi=min(branch_end,hi*2)
        if np.min(tof)<self.tof(lo) or np.max(tof)>self.tof(hi):
            raise ValueError("Measured TOF range is outside the profile's increasing physical branch")
        left=np.full(tof.shape,lo);right=np.full(tof.shape,hi)
        for _ in range(64):
            middle=(left+right)/2
            low=self.tof(middle)<tof
            left=np.where(low,middle,left);right=np.where(low,right,middle)
        result=(left+right)/2
        if not np.allclose(self.tof(result),tof,rtol=1e-10,atol=1e-7):
            raise ValueError("TOF coordinate inversion did not converge")
        return result


def coordinate_values(values,source,target,calibration=None):
    if source not in {"tof_us","two_theta_deg","d_angstrom","q_inv_angstrom"} or target not in {"tof_us","two_theta_deg","d_angstrom","q_inv_angstrom"}:
        raise ValueError("Unsupported diffraction coordinate")
    x=np.asarray(values,dtype=float)
    if source==target:
        return x.copy()
    if {'tof_us','two_theta_deg'}.intersection({source,target}) and calibration is None:
        raise ValueError("Diffraction conversion needs the exact mapped instrument profile")
    if 'tof_us' in {source,target} and isinstance(calibration,CwCalibration) or 'two_theta_deg' in {source,target} and isinstance(calibration,TofCalibration):
        raise ValueError('Cannot infer TOF from a CW profile or 2theta from a TOF profile')
    if source in {'tof_us','two_theta_deg'}:
        d=calibration.spacing(x)
    elif source=="d_angstrom":
        d=x
    elif source=="q_inv_angstrom":
        d=2*np.pi/x
    else:
        raise ValueError("Unsupported diffraction coordinate")
    if not np.all(np.isfinite(d)) or np.any(d<=0):
        raise ValueError("Positive d-spacings are required")
    if target=="d_angstrom":
        return d
    if target=="q_inv_angstrom":
        return 2*np.pi/d
    if target=='two_theta_deg':
        return calibration.two_theta(d)
    if target=="tof_us":
        result=calibration.tof(d)
        if not np.allclose(calibration.spacing(result),d,rtol=1e-9,atol=1e-10):
            raise ValueError("d-spacing is outside the physical TOF branch")
        return result
    raise ValueError("Unsupported diffraction coordinate")


def display_pattern(pattern,target,calibration=None):
    if pattern.scope.coordinate==target:
        return pattern
    if calibration is not None and str(calibration.source_bank)!=str(pattern.scope.bank):
        raise ValueError("Instrument profile belongs to another physical bank")
    x=coordinate_values(pattern.x,pattern.scope.coordinate,target,calibration)
    order=np.argsort(x)
    if np.any(np.diff(x[order])<=0):
        raise ValueError("Converted coordinates are not strictly monotonic")
    return replace(pattern,scope=replace(pattern.scope,coordinate=target),x=x[order],
        intensity=pattern.intensity[order],sigma=pattern.sigma[order],valid=pattern.valid[order])
