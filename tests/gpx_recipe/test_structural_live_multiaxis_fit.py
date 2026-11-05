from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
from gpx_recipe.structural_live.observations import Acquisition,CumulativeCounts

def fixture():
    x=np.linspace(0,1,60);radii=np.array([.05,.0008,.0008]);bg=np.full(len(x),.002)
    basis=np.array([np.exp(-((x-c)/.07)**2) for c in [.2,.45,.7]])
    def native(p):
        z=p/radii
        return bg+.008+(z[0]**2*.004)*basis[0]+.002*z[1]*basis[1]+.002*z[2]*basis[2]
    def check(p):
        p=np.asarray(p)
        if p.shape!=(3,) or np.any(abs(p)>radii):raise ValueError('domain')
        return p
    s=SimpleNamespace(radii=radii,mode_dimensions=1,cell_dimensions=2,check=check,native=native,
        native_mode_jacobian=lambda p:(native(p),np.array([2*p[0]/radii[0]**2*.004*basis[0]])))
    table=SimpleNamespace(renderer=SimpleNamespace(background=bg),background_reference=.002,fallback=True,evaluate=native)
    true=np.array([.5,.12,-.1,.03,np.log(1.25)])
    rate=np.exp(true[-2])*(native(true[:3]*radii)-bg)+.002*np.exp(true[-1])
    counts=np.random.default_rng(3344).poisson(rate*1e8)
    acq=Acquisition('sample','run','test','configuration','run','hash','q_inverse_angstrom','counts','unit')
    obs=CumulativeCounts.from_counts(acq,'three-looks',0,1e8,x,counts,1.,np.ones(len(x),bool))
    return s,table,obs,true
