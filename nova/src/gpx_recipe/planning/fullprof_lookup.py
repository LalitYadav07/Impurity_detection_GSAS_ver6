"""Bounded POWGEN FullProf NPROF=9 numerical resolution tables.

The current ORNL IRF header supplies d, Gaussian variance, Lorentzian FWHM,
alpha, beta and a TOF shift. Interpolation here is explicitly piecewise linear;
this module does not claim native FullProf interpolation/replay equivalence.
The four D2TOF terms follow the 2021 FullProf convention documented in ORNL's
FullProf_CommensurateMagneticOrder tutorial: Zero+C*d+A*d**2+B/d.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from .tof_profile import density

VERSION='powgen-tabulated-TOF-profile/1'
MAX_BYTES=2*1024**2


@dataclass(frozen=True)
class LookupProfile:
    table: np.ndarray
    coefficients: tuple  # C, A, B, Zero; microseconds and Angstroms
    two_theta: float
    source_sha256: str

    def _d(self, d):
        value=np.asarray(d,float)
        if (value.ndim>1 or value.size>50000 or not value.size
                or not np.isfinite(value).all()
                or np.any(value<self.table[0,0]) or np.any(value>self.table[-1,0])):
            raise ValueError('Finite d spacings inside the supplied resolution table required; no extrapolation')
        return value

    def parameters(self, d):
        """Return variance, FWHM, left/right rates and shift in TOF units."""
        d=self._d(d)
        return np.stack([np.interp(d,self.table[:,0],self.table[:,j]) for j in range(1,6)],axis=-1)

    def tof(self, d):
        d=self._d(d);c,a,b,zero=self.coefficients
        with np.errstate(over='ignore',invalid='ignore'):
            value=zero+c*d+a*d*d+b/d+self.parameters(d)[...,4]
        if not np.isfinite(value).all():raise ValueError('TOF coordinates exceed their finite domain')
        return value

    def derivative(self, d):
        """dTOF/dd, including the declared piecewise-linear shift table."""
        d=self._d(d);c,a,b,_=self.coefficients
        indices=np.clip(np.searchsorted(self.table[:,0],d,side='right')-1,0,len(self.table)-2)
        shift_slope=np.diff(self.table[:,5])/np.diff(self.table[:,0])
        with np.errstate(over='ignore',invalid='ignore'):
            value=c+2*a*d-b/d**2+shift_slope[indices]
        if not np.isfinite(value).all():raise ValueError('TOF derivatives exceed their finite domain')
        return value

    def point_pattern(self, x, d, areas, *, specimen_variance=None,
                      specimen_gamma=None, max_evaluations=20000000):
        """Sum normalized reflection densities with explicit integrated areas.

        Areas are supplied by the crystallographic engine, not inferred from
        the resolution file. No detector response, normalization or noise is
        invented here. Positive sample broadening is added to the table widths.
        Full tails are evaluated on x; no finite-window renormalization is used.
        """
        x=np.asarray(x,float);d=self._d(d);areas=np.asarray(areas,float)
        if (x.ndim!=1 or not 1<=len(x)<=50000 or not np.isfinite(x).all()
                or np.any(np.diff(x)<=0) or d.ndim!=1 or areas.shape!=d.shape
                or not np.isfinite(areas).all() or np.any(areas<0)
                or type(max_evaluations) is not int or not 1<=max_evaluations<=100000000
                or len(x)*np.count_nonzero(areas)>max_evaluations):
            raise ValueError('Bounded ordered TOF points and matching nonnegative reflection areas required')
        def broadening(value):
            result=np.zeros_like(d) if value is None else np.asarray(value,float)
            if result.shape!=d.shape or not np.isfinite(result).all() or np.any(result<0):
                raise ValueError('Specimen variance/FWHM must be matching finite nonnegative arrays')
            return result
        sample_sig=broadening(specimen_variance);sample_gam=broadening(specimen_gamma)
        parameters=self.parameters(d);positions=self.tof(d);result=np.zeros_like(x)
        for i in np.flatnonzero(areas):
            sig,gam,alpha,beta,_=parameters[i]
            with np.errstate(over='ignore',invalid='ignore'):
                result+=areas[i]*density(x-positions[i],alpha,beta,sig+sample_sig[i],gam+sample_gam[i])
        if not np.isfinite(result).all():raise ValueError('Pattern intensity exceeds its finite domain')
        return result

    def metadata(self):
        return dict(version=VERSION,source_sha256=self.source_sha256,rows=len(self.table),
            d_range_a=self.table[[0,-1],0].tolist(),two_theta_deg=self.two_theta,
            tof_coefficients=dict(zip(('difC','difA','difB','Zero'),self.coefficients)),
            table_columns=['d_A','variance_us2','Lorentzian_FWHM_us','alpha_per_us','beta_per_us','shift_us'],
            interpolation='piecewise_linear_in_d_no_extrapolation',
            kernel='normalized_two_sided_exponential_pseudoVoigt',
            native_fullprof_equivalence_established=False,experimental_accuracy_certified=False)


def read_lookup(path):
    """Read only the explicitly supported current numerical-table grammar."""
    path=Path(path)
    if not path.is_file() or not 1<=path.stat().st_size<=MAX_BYTES:
        raise ValueError('A bounded nonempty IRF resolution table is required')
    raw=path.read_bytes()
    if len(raw)>MAX_BYTES:raise ValueError('Resolution table exceeds its byte budget')
    try:lines=raw.decode('utf-8-sig').splitlines()
    except UnicodeDecodeError as exc:raise ValueError('Resolution table must be UTF-8 text') from exc
    fields={};rows=None;end=False;index=1
    allowed={'NPROF','TOFRG','D2TOF','TWOTH','LIST_SIG_GAM_ALF_BET_SHIFT','END'}
    while index<len(lines):
        line=lines[index].strip();index+=1
        if not line or line.startswith(('!','#')):continue
        words=line.split();key=words[0]
        if key not in allowed or key in fields or end:
            raise ValueError('Unsupported, duplicate or trailing resolution directive: '+key)
        fields[key]=words[1:]
        if key=='END':
            if len(words)!=1:raise ValueError('Unexpected data following END')
            end=True
        elif key=='LIST_SIG_GAM_ALF_BET_SHIFT':
            try:count=int(words[1])
            except (ValueError,IndexError) as exc:raise ValueError('Explicit table row count required') from exc
            if not 2<=count<=4096 or index+count>len(lines):raise ValueError('Invalid table row count')
            try:rows=np.asarray([[float(v) for v in s.split()] for s in lines[index:index+count]],float)
            except (ValueError,TypeError) as exc:raise ValueError('Malformed numerical profile table') from exc
            index+=count
    try:
        if fields['NPROF']!=['9'] or not end:raise ValueError('Only terminated NPROF=9 tables are supported')
        coeff=tuple(float(v) for v in fields['D2TOF'])
        theta=float(fields['TWOTH'][0])
        if len(fields['TWOTH'])!=1 or len(coeff)!=4:raise ValueError('Explicit current four-term D2TOF and one angle required')
        tof_range=[float(v) for v in fields['TOFRG']]
        if len(tof_range)!=3 or not 0<tof_range[0]<tof_range[2] or tof_range[1]<=0:raise ValueError('Invalid informative TOF range')
    except (KeyError,IndexError,ValueError) as exc:raise ValueError('Incomplete or invalid resolution header') from exc
    if (rows is None or rows.ndim!=2 or rows.shape[1]!=6 or not np.isfinite(rows).all()
            or np.any(rows[:,0]<=0) or np.any(np.diff(rows[:,0])<=0)
            or np.any(rows[:,1:3]<0) or np.any(rows[:,3:5]<=0)
            or not np.isfinite([*coeff,theta,*tof_range]).all() or not 0<theta<=180):
        raise ValueError('Invalid physical profile parameters or d domain')
    rows.setflags(write=False)
    profile=LookupProfile(rows,coeff,theta,hashlib.sha256(raw).hexdigest())
    # The minimum polynomial derivative can be at an endpoint or at a real
    # positive second-derivative root. Check both ends of every shift segment,
    # including distinct left/right slopes at each interpolation knot.
    c,a,b,_=coeff
    critical=np.cbrt(-b/a) if a!=0 else -1.
    slopes=np.diff(rows[:,5])/np.diff(rows[:,0])
    for low,high,slope in zip(rows[:-1,0],rows[1:,0],slopes):
        points=[low,high]+([critical] if low<critical<high else [])
        values=np.asarray([c+2*a*d-b/d**2+slope for d in points])
        if not np.isfinite(values).all() or np.any(values<=0):
            raise ValueError('TOF coordinate mapping is not strictly increasing over the table')
    if np.any(profile.tof(rows[:,0])<=0):raise ValueError('TOF must remain positive')
    return profile


def native_reflection_pattern(profile, x, reflection_lists, instrument):
    """Apply the table to pinned GSAS commensurate TOF reflection areas.

    GSAS supplies multiplicity, structure factor, phase fraction, texture and
    specimen broadening. Only its instrumental profile is replaced. The caller
    must set GSAS TOF coefficients to the table before generating reflections.
    """
    parameters=[];areas=[]
    for block in reflection_lists:
        r=np.asarray(block,float)
        if r.ndim!=2 or r.shape[1]!=18 or not np.isfinite(r).all():
            raise ValueError('Pinned commensurate TOF reflection columns required')
        amplitude=r[:,9]*r[:,11]
        if not np.isfinite(amplitude).all() or np.any(amplitude<0):
            raise ValueError('Nonnegative finite crystallographic reflection areas required')
        r=r[amplitude>0];amplitude=amplitude[amplitude>0]
        if not len(r):continue
        d=r[:,4]
        sig=instrument['sig-0']+instrument['sig-1']*d**2+instrument['sig-2']*d**4+instrument.get('sig-q',0.)*d
        gam=instrument['X']*d+instrument['Y']*d**2+instrument['Z']
        sample=np.column_stack([r[:,6]-sig,r[:,7]-gam])
        tolerance=64*np.finfo(float).eps*np.maximum(1.,np.abs(r[:,6:8])+np.abs(np.column_stack([sig,gam])))
        if not np.isfinite(sample).all() or np.any(sample < -tolerance):
            raise ValueError('Native profile contains unexplained negative specimen broadening')
        if not np.allclose(r[:,5],profile.tof(d),rtol=2e-12,atol=1e-7):
            raise ValueError('Native reflection positions do not use the selected table coordinate')
        parameters.append(np.column_stack([d,np.maximum(sample,0.)]));areas.append(amplitude)
    if not parameters:return np.zeros_like(x,dtype=float)
    grouped,inverse=np.unique(np.concatenate(parameters),axis=0,return_inverse=True)
    integrated=np.bincount(inverse,weights=np.concatenate(areas))
    return profile.point_pattern(x,grouped[:,0],integrated,
        specimen_variance=grouped[:,1],specimen_gamma=grouped[:,2])
