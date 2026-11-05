"""Declared native nuisance scope and original-bin objective helpers."""
import time
import numpy as np
from .fit_fast_native import residual_and_factor

def background_reference(table):
    """A declared positive native background response scale is required."""
    value=float(getattr(table,'background_reference',float('nan')))
    if not np.isfinite(value) or value<=0:
        raise ValueError('Positive finite background reference required')
    return value

def background_components(table):
    shape=np.asarray(table.renderer.background).shape
    fixed=np.asarray(getattr(table,'fixed_background',np.zeros(shape)),float)
    response=np.asarray(getattr(table,'background_response',np.ones(shape)),float)
    if fixed.shape!=shape or response.shape!=shape or not np.isfinite(fixed).all() or not np.isfinite(response).all():
        raise ValueError('Fixed native background and coefficient response must match the observation grid')
    return fixed,response

class NativeObjective:
    def __init__(self,source,table,observation,cell_step=.002):
        self.source=source;self.table=table;self.obs=observation;self.cell_step=cell_step
        self.last=None;self.calls=0
        self.background=background_reference(table)
        self.fixed,self.response=background_components(table)

    def calculate(self,z,jacobian=False):
        z=np.asarray(z,float);s=self.source;d=len(s.radii);m=s.mode_dimensions
        if self.last is None or not np.array_equal(z,self.last['z']):
            y,j=s.native_mode_jacobian(z[:d]*s.radii);self.calls+=1
            signal=y-self.table.renderer.background;scale=np.exp(z[-2]);back=self.background*np.exp(z[-1])
            rate=scale*signal+self.fixed+back*self.response;r,factor=residual_and_factor(self.obs,rate)
            self.last=dict(z=z.copy(),rate=rate,residual=r,factor=factor,signal=signal,mode_jac=j,jac=None)
        a=self.last
        if jacobian and a['jac'] is None:
            derivative=np.zeros((d+2,len(a['rate'])))
            derivative[:m]=a['mode_jac']*s.radii[:m,None]*np.exp(z[-2])
            for i in range(m,d):
                lo=z[:d].copy();hi=lo.copy()
                step=float(self.cell_step[i]) if np.ndim(self.cell_step) else float(self.cell_step)
                lo[i]=max(-1.,lo[i]-step);hi[i]=min(1.,hi[i]+step)
                low=s.native(lo*s.radii);high=s.native(hi*s.radii);self.calls+=2
                derivative[i]=(high-low)/(hi[i]-lo[i])*np.exp(z[-2])
            derivative[-2]=a['signal']*np.exp(z[-2]);derivative[-1]=self.background*np.exp(z[-1])*self.response
            a['jac']=(derivative[:,self.obs.mask]*a['factor']).T
        return a
