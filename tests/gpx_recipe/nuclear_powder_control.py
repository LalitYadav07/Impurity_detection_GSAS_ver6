"""Public synthetic native controls, independent complex-amplitude sums.

No fitted patterns, GPXs, private data, credentials or network are used.
"""
import copy
import inspect
import json

import numpy as np
from GSASII import GSASIIstrMath as M
from gpx_recipe.symmetry.kernel_patch import corrected_nuclear_function


def control():
    scope = dict(M.StructureFactor2.__globals__)
    exec(compile(corrected_nuclear_function(inspect.getsource(M.StructureFactor2)),
                 '<reviewed-powder-intensities>', 'exec'), scope)
    corrected = scope['StructureFactor2']
    hkl = np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,0], [2,1,1], [-1,2,1]], float)
    cell = np.array([4., 5., 6.])
    metric = np.diag(1 / cell**2)
    positions = np.array([[.11,.21,.32], [.42,.07,.28]])
    occupancy = np.array([.8,1.])
    uiso = np.array([.012,.017])
    rows = []
    for kind in ['PNC', 'PNT']:
        for inversion in [False, True]:
            for imaginary in [False, True]:
                lengths = np.array([.7 + (.24j if imaginary else 0), .58 + .01j])
                if not imaginary: lengths = lengths.real.astype(complex)
                controls = {'Natoms':{'0::':2}, ':0:histType':kind,
                    'FFtables':{}, 'EFtables':{}, 'ORBtables':{},
                    'BLtables':{t:['Nat. Abund.', {'SL':[float(b.real),float(b.imag)]}]
                                for t,b in zip(['Er','O'], lengths)}}
                parameters = {':0:Lam':2., '0:0:BabA':0., '0:0:BabU':0.}
                for i, element in enumerate(['Er','O']):
                    parameters.update({f'0::Atype:{i}':element, f'0::Amul:{i}':2. if inversion else 1.,
                        f'0::Afrac:{i}':occupancy[i], f'0::AI/A:{i}':'I', f'0::AUiso:{i}':uiso[i]})
                    parameters.update({f'0::A{axis}:{i}':v for axis,v in zip('xyz',positions[i])})
                group = dict(SGOps=[(np.eye(3),np.zeros(3))],SGCen=[np.zeros(3)],SGInv=inversion)
                def reference(parms):
                    xyz=np.array([[parms[f'0::A{a}:{i}']+parms.get(f'0::dA{a}:{i}',0.) for a in 'xyz'] for i in range(2)])
                    occ=np.array([parms[f'0::Afrac:{i}'] for i in range(2)])
                    u=np.array([parms[f'0::AUiso:{i}'] for i in range(2)])
                    q2=np.einsum('ni,ij,nj->n',hkl,metric,hkl)
                    dw=np.exp(-2*np.pi**2*q2[:,None]*u)
                    phase=2*np.pi*hkl@xyz.T
                    plus=np.exp(1j*phase)
                    minus=np.exp(-1j*phase)
                    if inversion:plus,minus=plus+minus,plus+minus
                    forward=np.sum(dw*occ*lengths*plus,axis=1)
                    reverse=np.sum(dw*occ*lengths*minus,axis=1)
                    return (abs(forward)**2+abs(reverse)**2)/2
                array=np.zeros((len(hkl),16));array[:,:3]=hkl
                array[:,4]=1/np.sqrt(np.einsum('ni,ij,nj->n',hkl,metric,hkl));array[:,14]=2.
                original={'RefList':array.copy(),'FF':{}}
                revised=copy.deepcopy(original)
                M.StructureFactor2(original,metric,':0:','0::',group,controls,parameters)
                corrected(revised,metric,':0:','0::',group,controls,parameters)
                expected=reference(parameters)
                error=float(np.max(abs(revised['RefList'][:,9]-expected)))
                original_error=float(np.max(abs(original['RefList'][:,9]-expected)))
                assert error<1e-10,(kind,inversion,imaginary,error)
                if imaginary and not inversion:assert original_error>1e-4
                derivatives=M.StructureFactorDerv2(revised,metric,':0:','0::',group,controls,parameters)
                derivative_errors=[]
                for field in ['0::Afrac:0','0::dAx:0','0::AUiso:0']:
                    step=1e-6;plus=dict(parameters);minus=dict(parameters)
                    plus[field]=plus.get(field,0.)+step;minus[field]=minus.get(field,0.)-step
                    numerical=(reference(plus)-reference(minus))/(2*step)
                    err=float(np.max(abs(derivatives[field]-numerical)))
                    assert err<1e-6,(field,err)
                    derivative_errors.append(err)
                rows.append(dict(kind=kind,inversion=inversion,imaginary_scattering=imaginary,
                    corrected_error=error,original_error=original_error,derivative_errors=derivative_errors))
    return dict(passed=True,cases=rows)


if __name__ == '__main__':
    print(json.dumps(control(),sort_keys=True))
