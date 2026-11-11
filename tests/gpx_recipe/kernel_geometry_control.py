"""Independent public/synthetic controls for the exact deployed GSAS-II kernel.

No GPX, experimental arrays, credentials, network, or installed source edits.
Run in the already cached scientific base image. Candidate corrections are
compiled into isolated function objects and are never applied to the module.
"""
import contextlib
import argparse
import copy
import hashlib
import inspect
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
from GSASII import GSASIIstrMath as M, GSASIIspc as S

FILE_SHA = '2c9875e52546547b6ac565d1d0f25aabf5f06b1e1aba43fa8dd4cf359353f559'
FORWARD_SHA = '82076dc0571cad6d6e2882c5f7ced9003f08af94449aa293b0fb4c5c2ae7b9f6'
DERIVATIVE_SHA = 'df959b6cb576c83efbea4117f2ad4b73768253310a6188254dd08033e07b72f7'


def direct_basis(cell):
    """Cartesian columns along a, b, c, derived directly from cell angles."""
    a, b, c, alpha, beta, gamma = cell
    ca, cb, cg = np.cos(np.deg2rad([alpha, beta, gamma]))
    sg = np.sin(np.deg2rad(gamma))
    z2 = 1 - cb**2 - ((ca - cb*cg)/sg)**2
    assert z2 > 0
    unit = np.array([[1., cg, cb], [0., sg, (ca-cb*cg)/sg], [0., 0., np.sqrt(z2)]])
    return unit, unit @ np.diag([a, b, c])


def patched_function(fn, changes):
    source = inspect.getsource(fn)
    for old, new in changes:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    scope = dict(fn.__globals__)
    exec(compile(source, '<synthetic-geometry-control>', 'exec'), scope)
    return scope[fn.__name__], hashlib.sha256(source.encode()).hexdigest()


raw_forward, raw_deriv = M.MagStructureFactor2, M.MagStructureFactorDerv
unit_forward=unit_deriv=None
unit_forward_sha=unit_deriv_sha=transpose_sha=None

HKL = np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1],
                [0,1,1], [1,1,1], [1,-1,1], [2,1,0], [1,2,3], [3,-2,1], [2,2,-1]], dtype=float)
CELLS = {
    'orthogonal': [4.,5.,6.,90.,90.,90.],
    'hexagonal': [5.,5.,13.,90.,90.,120.],
    'monoclinic': [5.,8.,6.,90.,112.,90.],
    'triclinic': [4.,6.,8.,78.,101.,109.],
}
MOMENTS = [[1.1,.7,2.3], [-1.2,.8,.4], [.3,-2.1,1.8], [1e-8,1e-8,3.], [2.,1e-8,1e-8]]
SG = dict(SGOps=[(np.eye(3),np.zeros(3))], SGCen=[np.zeros(3)],
          SGInv=False, SGFixed=True, MagMom=np.ones(1))
PFX, HFX = '0::', ':0:'
CONTROLS = {'Natoms': {PFX:1}, 'MFtables': {'Fe':{}}, HFX+'histType':'PNT'}


def reference(cell, moment):
    unit, direct = direct_basis(cell)
    q = HKL @ np.linalg.inv(direct)
    q /= np.linalg.norm(q,axis=1)[:,None]
    mc = unit @ np.asarray(moment)
    return (.539/2)**2 * (mc @ mc - (q @ mc)**2)


def calculation(fn, cell, moment, *, uiso=.01, group=None, position=None, uij=None):
    unit, direct = direct_basis(cell)
    reciprocal = np.linalg.inv(direct.T @ direct)
    refs = np.zeros((len(HKL),12))
    refs[:,:3] = HKL
    refs[:,3] = 1
    refs[:,4] = 1/np.sqrt(np.einsum('ni,ij,nj->n',HKL,reciprocal,HKL))
    group = SG if group is None else group
    count = len(group['SGOps'])*len(group['SGCen'])*(1+int(group['SGInv'] and not group['SGFixed']))
    params = {PFX+'Atype:0':'Fe', PFX+'Amul:0':float(count), PFX+'Afrac:0':1.,
              PFX+'AI/A:0':'I', PFX+'AUiso:0':uiso}
    if position is not None:
        params.update({PFX+'A'+axis+':0':float(value) for axis,value in zip('xyz',position)})
    params.update({PFX+'AM'+axis+':0': float(value) for axis,value in zip('xyz',moment)})
    if uij is not None:
        params[PFX+'AI/A:0']='A';params[PFX+'AUiso:0']=0.
        params.update({PFX+'AU'+axis+':0':float(value)
                       for axis,value in zip(['11','22','33','12','13','23'],uij)})
    ref = {'RefList':refs, 'FF':{}}
    fn(ref, reciprocal, HFX, PFX, group, CONTROLS, params)
    return refs[:,9].copy(), ref, reciprocal, params


def orbit_reference(cell, moment, group, position, *, uij=None):
    """Explicit Cartesian axial-vector sum, independent of native SF routines."""
    unit,direct=direct_basis(cell)
    q_raw=HKL @ np.linalg.inv(direct)
    q=q_raw.copy()
    q/=np.linalg.norm(q,axis=1)[:,None]
    rotations=list(group['SGOps'])
    if group['SGInv'] and not group['SGFixed']:
        rotations += [(-r,-t) for r,t in rotations]
    all_ops=[(r,t+cen) for cen in group['SGCen'] for r,t in rotations]
    amplitude=np.zeros((len(HKL),3),complex)
    mcart=unit @ np.asarray(moment)
    if uij is not None:
        a,b,c,d,e,f=uij
        u=np.array([[a,d,e],[d,b,f],[e,f,c]])
        ast=np.linalg.norm(np.linalg.inv(direct).T,axis=0)
        ucart=direct @ np.diag(ast) @ u @ np.diag(ast) @ direct.T
    for (rotation,translation),spin in zip(all_ops,group['SpnFlp']):
        rc=unit @ rotation @ np.linalg.inv(unit)
        assert np.allclose(rc.T@rc,np.eye(3),atol=1e-10), 'Incompatible synthetic cell'
        mapped=round(np.linalg.det(rotation))*float(spin)*(rc @ mcart)
        perpendicular=mapped[None,:]-(q@mapped)[:,None]*q
        phase=np.exp(2j*np.pi*(HKL@(rotation@position+translation)))
        thermal=np.ones(len(HKL))
        if uij is not None:
            space_rotation=direct@rotation@np.linalg.inv(direct)
            transformed=space_rotation@ucart@space_rotation.T
            thermal=np.exp(-2*np.pi**2*np.einsum('ni,ij,nj->n',q_raw,transformed,q_raw))
        amplitude+=perpendicular*(phase*thermal)[:,None]
    return (.539/2)**2*np.sum(abs(amplitude)**2,axis=1)


def fixed_c2prime_group():
    error,group=S.SpcGroup('C 2/c')
    assert not error
    operations=list(group['SGOps'])
    if group['SGInv']:
        operations += [(-r,-t) for r,t in operations]
    operations=[(r,t+cen) for cen in group['SGCen'] for r,t in operations]
    # For this conventional monoclinic group, det(R)*Rxx is the character
    # that primes both the twofold and glide, leaving inversion unprimed.
    spins=np.array([round(np.linalg.det(r))*round(r[0,0]) for r,t in operations])
    group=copy.deepcopy(group)
    group.update(SGOps=operations,SGCen=[np.zeros(3)],SGFixed=True,
        SpnFlp=spins,MagMom=spins*np.array([round(np.linalg.det(r)) for r,t in operations]))
    return group


def scaled_error(actual, expected):
    return float(np.max(np.abs(actual-expected))/max(float(np.max(np.abs(expected))),1e-12))


def main():
    global unit_forward,unit_deriv,unit_forward_sha,unit_deriv_sha,transpose_sha
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--kernel-patch',type=Path)
    mode.add_argument('--installed',action='store_true',help='Verify installed functions without constructing replacements')
    parser.add_argument('--output',type=Path,help='Write a JSON receipt separately from GSAS-II startup messages')
    args=parser.parse_args()
    patch_record=None
    if args.installed:
        unit_forward,unit_deriv=raw_forward,raw_deriv
        unit_forward_sha=hashlib.sha256(inspect.getsource(unit_forward).encode()).hexdigest()
        unit_deriv_sha=hashlib.sha256(inspect.getsource(unit_deriv).encode()).hexdigest()
        variants={'unit_basis_consistent':(unit_forward,unit_deriv)}
    else:
        assert hashlib.sha256(Path(M.__file__).read_bytes()).hexdigest() == FILE_SHA
        assert hashlib.sha256(inspect.getsource(raw_forward).encode()).hexdigest() == FORWARD_SHA
        assert hashlib.sha256(inspect.getsource(raw_deriv).encode()).hexdigest() == DERIVATIVE_SHA
        transpose_forward,transpose_sha=patched_function(raw_forward,[
            ('uAmat = G2lat.Gmat2AB(GS)[0] #RVD',
             'uAmat = G2lat.Gmat2AB(nl.inv(Ginv))[0] #synthetic control'),
            ('Kdata = np.inner(Gdata.T,uAmat).T', 'Kdata = np.inner(Gdata.T,uAmat.T).T')])
        spec=importlib.util.spec_from_file_location('reviewed_kernel_patch',args.kernel_patch)
        patch=importlib.util.module_from_spec(spec);spec.loader.exec_module(patch)
        updated,patch_record=patch.corrected_source(Path(M.__file__).read_text())
        compiled={}
        for name in patch.ORIGINAL_FUNCTION_SHA256:
            source=patch.function_source(updated,name)
            scope=dict(vars(M));exec(compile(source,'<reviewed-synthetic-patch>','exec'),scope)
            compiled[name]=scope[name]
        unit_forward=compiled['MagStructureFactor2'];unit_deriv=compiled['MagStructureFactorDerv']
        unit_forward_sha=patch_record['corrected_functions']['MagStructureFactor2']
        unit_deriv_sha=patch_record['corrected_functions']['MagStructureFactorDerv']
        variants={'native':(raw_forward,raw_deriv),'unit_basis_forward_only':(unit_forward,raw_deriv),
            'transpose_forward_matches_derivative':(transpose_forward,raw_deriv),
            'unit_basis_consistent':(unit_forward,unit_deriv)}
    # Constant magnetic form factor isolates the geometric polarization factor.
    saved_form_factor = M.G2el.MagScatFac
    M.G2el.MagScatFac = lambda table, sq: np.ones_like(sq)
    results=[];orbits=[];anisotropic=[];derivative_checks=[]
    try:
        for name,(forward,deriv) in variants.items():
            for cell_name,cell in CELLS.items():
                for moment in MOMENTS:
                    calculated,ref,g,params = calculation(forward,cell,moment)
                    d = ref['RefList'][:,4]
                    expected=reference(cell,moment)*np.exp(-4*np.pi**2*.01/d**2)
                    delta=1e-6
                    plus=calculation(forward,cell,moment,uiso=.01+delta)[0]
                    minus=calculation(forward,cell,moment,uiso=.01-delta)[0]
                    finite=(plus-minus)/(2*delta)
                    with contextlib.redirect_stdout(io.StringIO()):
                        analytic=deriv(ref,g,HFX,PFX,SG,CONTROLS,params)[PFX+'AUiso:0']
                    # Physical intensities must be a homogeneous quadratic form
                    # in all freely varied moment components at fixed geometry.
                    other=np.array([.6,-.4,1.3])
                    a=np.asarray(moment)
                    cross = calculation(forward,cell,a+other)[0]+calculation(forward,cell,a-other)[0]
                    quadratic=2*(calculated+calculation(forward,cell,other)[0])
                    results.append(dict(variant=name,cell=cell_name,moment=moment,
                        cartesian_projection_relative_error=scaled_error(calculated,expected),
                        uiso_derivative_relative_error=scaled_error(analytic,finite),
                        quadratic_identity_relative_error=scaled_error(cross,quadratic)))
        groups=[('P -1','triclinic'),('P 21/c','monoclinic'),('C 2/c','monoclinic'),
                ('R -3 c','hexagonal'),('P n m a','orthogonal'),
                ('P 4/m m m','tetragonal'),('F d -3 m','cubic'),("C 2'/c' explicit",'monoclinic')]
        cells=dict(CELLS,tetragonal=[5.,5.,8.,90.,90.,90.],cubic=[6.,6.,6.,90.,90.,90.])
        position=np.array([.173,.287,.391])
        for symbol,cell_name in groups:
            if symbol.endswith('explicit'):
                group=fixed_c2prime_group()
            else:
                error,group=S.SpcGroup(symbol)
                assert not error,(symbol,error)
                S.SetMagnetic(group)
            orbit_variants=[('unit_basis_consistent',(unit_forward,unit_deriv))]
            if not args.installed:orbit_variants.insert(0,('native',(raw_forward,raw_deriv)))
            for variant,(forward,deriv) in orbit_variants:
                for moment in MOMENTS[:3]:
                    cell=cells[cell_name]
                    calculated,ref,g,params=calculation(forward,cell,moment,group=group,position=position)
                    d=ref['RefList'][:,4]
                    expected=orbit_reference(cell,moment,group,position)*np.exp(-4*np.pi**2*.01/d**2)
                    delta=1e-6
                    plus=calculation(forward,cell,moment,uiso=.01+delta,group=group,position=position)[0]
                    minus=calculation(forward,cell,moment,uiso=.01-delta,group=group,position=position)[0]
                    with contextlib.redirect_stdout(io.StringIO()):
                        analytic=deriv(ref,g,HFX,PFX,group,CONTROLS,params)[PFX+'AUiso:0']
                    orbits.append(dict(variant=variant,group=symbol,cell=cell_name,moment=moment,
                        operations=len(group['MagMom']),
                        cartesian_projection_relative_error=scaled_error(calculated,expected),
                        uiso_derivative_relative_error=scaled_error(analytic,(plus-minus)/(2*delta))))
            if args.kernel_patch or args.installed:
                uij=np.array([.01,.012,.015,.002,.001,-.003])
                for variant,(forward,unused) in orbit_variants:
                    for moment in MOMENTS[:3]:
                        cell=cells[cell_name]
                        calculated,ref,g,params=calculation(forward,cell,moment,group=group,position=position,uij=uij)
                        expected=orbit_reference(cell,moment,group,position,uij=uij)
                        anisotropic.append(dict(variant=variant,group=symbol,moment=moment,
                            cartesian_projection_relative_error=scaled_error(calculated,expected)))
                        if variant=='unit_basis_consistent':
                            with contextlib.redirect_stdout(io.StringIO()):
                                analytic=unit_deriv(ref,g,HFX,PFX,group,CONTROLS,params)
                            for j,axis in enumerate(['11','22','33','12','13','23']):
                                step=np.zeros(6);step[j]=1e-6
                                plus=calculation(unit_forward,cell,moment,group=group,position=position,uij=uij+step)[0]
                                minus=calculation(unit_forward,cell,moment,group=group,position=position,uij=uij-step)[0]
                                derivative_checks.append(dict(group=symbol,moment=moment,variable='AU'+axis,
                                    relative_error=scaled_error(analytic[PFX+'AU'+axis+':0'],(plus-minus)/2e-6)))
                            # Magnetic derivatives are numerical in this GSAS-II
                            # stack. Compare to a larger independent, exactly
                            # quadratic symmetric difference of the Cartesian sum.
                            saved_forward=M.MagStructureFactor2
                            M.MagStructureFactor2=unit_forward
                            try:
                                magnetic=M.MagStructureFactorDerv2(ref,g,HFX,PFX,group,CONTROLS,params)
                            finally:
                                M.MagStructureFactor2=saved_forward
                            for j,axis in enumerate('xyz'):
                                step=np.zeros(3);step[j]=.001
                                plus=orbit_reference(cell,np.asarray(moment)+step,group,position,uij=uij)
                                minus=orbit_reference(cell,np.asarray(moment)-step,group,position,uij=uij)
                                derivative_checks.append(dict(group=symbol,moment=moment,variable='AM'+axis,
                                    relative_error=scaled_error(magnetic[PFX+'AM'+axis+':0'],(plus-minus)/.002)))
                            for j,axis in enumerate('xyz'):
                                step=np.zeros(3);step[j]=1e-6
                                plus=orbit_reference(cell,moment,group,position+step,uij=uij)
                                minus=orbit_reference(cell,moment,group,position-step,uij=uij)
                                derivative_checks.append(dict(group=symbol,moment=moment,variable='dA'+axis,
                                    relative_error=scaled_error(analytic[PFX+'dA'+axis+':0'],(plus-minus)/2e-6)))
    finally:
        M.G2el.MagScatFac=saved_form_factor
    summary=[]
    for variant in dict.fromkeys(r['variant'] for r in results):
        for cell in CELLS:
            rows=[r for r in results if r['variant']==variant and r['cell']==cell]
            summary.append(dict(variant=variant,cell=cell,cases=len(rows),
                **{key:max(r[key] for r in rows) for key in rows[0] if key.endswith('_error')}))
    corrected=[r for r in results if r['variant']=='unit_basis_consistent']
    passed=all(r['cartesian_projection_relative_error']<1e-9 and
        r['uiso_derivative_relative_error']<1e-7 and
        r['quadratic_identity_relative_error']<1e-9 for r in corrected)
    orbit_passed=all(r['cartesian_projection_relative_error']<1e-9 and
        r['uiso_derivative_relative_error']<1e-7 for r in orbits if r['variant']=='unit_basis_consistent')
    anisotropic_passed=all(r['cartesian_projection_relative_error']<1e-9
        for r in anisotropic if r['variant']=='unit_basis_consistent') if anisotropic else None
    derivatives_passed=all(r['relative_error']<1e-5 for r in derivative_checks) if derivative_checks else None
    document=dict(schema='radar-native-moment-geometry-control/v1',
        source_file_sha256=hashlib.sha256(Path(M.__file__).read_bytes()).hexdigest(),
        native_forward_sha256=hashlib.sha256(inspect.getsource(raw_forward).encode()).hexdigest(),
        native_nonmagnetic_derivative_sha256=hashlib.sha256(inspect.getsource(raw_deriv).encode()).hexdigest(),
        synthetic_forward_sha256=unit_forward_sha,synthetic_derivative_sha256=unit_deriv_sha,
        transpose_control_sha256=transpose_sha,experimental_data_used=False,installed_functions=args.installed,
        scope='P1 and seven full general-position orbits, constant form factor; not an experimental validation',
        corrected_synthetic_checks_passed=passed,corrected_full_orbit_checks_passed=orbit_passed,
        corrected_anisotropic_checks_passed=anisotropic_passed,patch=patch_record,
        corrected_derivative_checks_passed=derivatives_passed,
        summary=summary,cases=results,full_orbit_cases=orbits,anisotropic_cases=anisotropic,
        derivative_checks=derivative_checks)
    if args.output:
        args.output.write_text(json.dumps(document,indent=2)+'\n')
    else:
        print(json.dumps(document,indent=2))
    raise SystemExit(0 if passed and orbit_passed and anisotropic_passed is not False and derivatives_passed is not False else 1)


if __name__=='__main__':
    main()
