from types import SimpleNamespace
import numpy as np
from test_structural_live_multiaxis_fit import fixture

def setup(offset=0.):
    s,t,o,z=fixture();s.fast=SimpleNamespace(x=o.x);s.chemistry=SimpleNamespace(log_kernel=lambda p:np.where(p[:,0]>=0,0.,-np.inf))
    called=[];native=s.native
    def guarded(p):
        assert np.isfinite(s.chemistry.log_kernel(np.asarray(p)[None,:])[0])
        called.append(p.copy());return native(p)
    s.native=guarded
    cache=SimpleNamespace(evaluate=lambda p:guarded(p)+offset,
        mode_jacobian=lambda p:s.native_mode_jacobian(p),diagnostics=lambda:dict(test_fixture=True))
    points=np.array([[0.,0.,0.],[.4,.05,-.05]])
    bank=SimpleNamespace(points=points,curves=np.array([guarded(p*s.radii) for p in points]),proposal_cell_steps=np.full(3,.002))
    return s,t,o,z,cache,bank,called
