"""Normalized two-sided-exponential/pseudoVoigt TOF density.

This evaluates the convolution itself, including the opposite signs of the
left/right exponential arguments. It does not establish a calibrated beam
response. Rates, Gaussian variance and Lorentzian FWHM require external
instrument/specimen calibration.
"""
import math
import numpy as np
from scipy.special import erfc,erfcx,exp1,expi

VERSION='powgen-normalized-exponential-pseudoVoigt/1'


def scaled_exp1(z):
    z=np.asarray(z,complex)
    if not np.isfinite(z).all() or np.any(z.imag<=0):raise ValueError('Upper-half-plane finite arguments required')
    result=np.empty(z.shape,complex);large=abs(z)>50
    result[~large]=np.exp(z[~large])*exp1(z[~large])
    inv=1/z[large];term=np.ones_like(inv);total=term.copy()
    for n in range(1,31):term*=-n*inv;total+=term
    result[large]=inv*total
    # E1(-a+i0)=-Ei(a)-i*pi. Both the ordinary complex library and a
    # large-argument expansion can lose this exponentially small branch term.
    # Near that axis evaluate the analytic real-Ei part by a convergent Taylor
    # recurrence, retaining the branch term separately. For large arguments
    # the same term supplements the stable inverse-power expansion.
    near=(z.real<0)&(z.imag<=1)&(z.imag<=-.25*z.real)
    central=near&~large
    r=z[central].real;h=1j*z[central].imag
    value=(-np.exp(r)*expi(-r)).astype(complex)
    term=value.copy();power=np.ones_like(value)
    for n in range(1,33):
        power*=-h/r;term=term*h/n+power/n;value+=term
    result[central]=value-1j*np.pi*np.exp(z[central])
    result[near&large]-=1j*np.pi*np.exp(z[near&large])
    # Around the imaginary axis the leading real terms can cancel. Expand in
    # 1/Im(z) using real moments of Exp(1)+Re(z), rather than multiplying two
    # complex sums whose real parts nearly cancel. The dimensionless moment
    # recurrence avoids forming large factorials or coordinate powers.
    vertical=(z.imag>50)&(abs(z.real)<=.25*z.imag)
    a,b=z[vertical].real,z[vertical].imag
    ratio=a/b;power=np.ones_like(a);moment=power.copy()
    real=np.zeros_like(a);imag=-np.ones_like(a)/b
    for n in range(1,31):
        power*=ratio
        moment=(a+1)/b if n==1 else power+(n/b)*moment
        if n%2:real+=(-1.)**((n-1)//2)*moment/b
        else:imag+=(-1.)**(n//2+1)*moment/b
    result[vertical]=real+1j*imag
    return result


def exgaussian_right(x,rate,sigma):
    x=np.asarray(x,float);ratio=rate*sigma;z=(ratio-x/sigma)/np.sqrt(2)
    if not np.isfinite(ratio):raise ValueError('Unrepresentable exponential/Gaussian scale product')
    out=np.empty_like(x);positive=z>=0
    with np.errstate(over='ignore',under='ignore',divide='ignore'):
        logp=math.log(rate)-math.log(2)
        out[positive]=np.exp(logp-.5*(x[positive]/sigma)**2+np.log(erfcx(z[positive])))
        out[~positive]=np.exp(logp+.5*ratio*ratio-rate*x[~positive])*erfc(z[~positive])
    return out


def density(x,alpha,beta,variance,gamma):
    """Convolve a normalized two-sided exponential with the TCH pseudoVoigt.

    Rates have reciprocal-x units; variance has squared-x units; gamma is
    Lorentzian FWHM. No curve/area rescaling or probability floor is applied.
    """
    x=np.asarray(x,float)
    if (x.ndim!=1 or not 1<=len(x)<=1000000 or not np.isfinite(x).all()
            or not np.isfinite([alpha,beta,variance,gamma]).all()
            or min(alpha,beta)<=0 or min(variance,gamma)<0):
        raise ValueError('Bounded finite coordinates, positive rates and nonnegative widths required')
    ratio=min(alpha,beta)/max(alpha,beta);c=min(alpha,beta)/(1+ratio)
    if variance==gamma==0:return c*np.exp(np.where(x<0,alpha*x,-beta*x))
    gaussian=math.sqrt(8*math.log(2)*variance);scale=max(gaussian,gamma)
    g,l=gaussian/scale,gamma/scale
    width=scale*(g**5+2.69269*g**4*l+2.42843*g**3*l*l+4.47163*g*g*l**3+.07842*g*l**4+l**5)**.2
    fraction=gamma/width
    eta=0. if gamma==0 else 1. if variance==0 else 1.36603*fraction-.47719*fraction**2+.11116*fraction**3
    result=np.zeros_like(x)
    if eta<1:
        a,b=alpha/max(alpha,beta),beta/max(alpha,beta)
        normal=(b*exgaussian_right(-x,alpha,width/math.sqrt(8*math.log(2)))+
            a*exgaussian_right(x,beta,width/math.sqrt(8*math.log(2))))/(a+b)
        result+=(1-eta)*normal
    if eta:
        left=alpha*x+1j*alpha*width/2
        right=-beta*x+1j*beta*width/2
        lorentz=-c/math.pi*(scaled_exp1(left).imag+scaled_exp1(right).imag)
        result+=eta*lorentz
    if not np.isfinite(result).all() or np.any(result<0):raise ValueError('Invalid analytic convolution density')
    return result
