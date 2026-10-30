"""Independent BNS operator checks before trusting a native MSG construction."""
from __future__ import annotations
import numpy as np
from functools import lru_cache


class GreyGroup(ValueError):pass
class UnverifiedMagneticSetting(ValueError):pass


def operation_key(rotation,translation,spin):
    t=np.asarray(translation,dtype=float)%1
    t[np.isclose(t,1,atol=1e-8,rtol=0)]=0
    r=np.asarray(rotation,dtype=float)
    if not np.allclose(r,np.rint(r),atol=1e-8,rtol=0):raise UnverifiedMagneticSetting('Nonintegral BNS rotation in the requested fractional basis')
    return tuple(int(v) for v in np.rint(r).ravel())+tuple(float(v) for v in np.round(t,8))+(int(spin),)


def require_collinear_line(rotations,cell_matrix,axis_lengths,native_direction):
    """A shared asymmetric-unit axis must stay a line on every symmetry mate.

    Crystal-axis moment components use unit axes. Fractional position rotations
    act through the full cell matrix; spin reversal and axial parity only change
    a sign and therefore cannot repair a rotated, noncollinear direction.
    """
    a=np.asarray(cell_matrix,dtype=float);d=a@np.diag(1/np.asarray(axis_lengths,dtype=float));v=d@np.asarray(native_direction,dtype=float)
    norm=np.linalg.norm(v)
    if not np.isfinite(norm) or norm<1e-12:raise UnverifiedMagneticSetting('A finite nonzero ordered-moment direction is required')
    for rotation in rotations:
        other=a@np.asarray(rotation,dtype=float)@np.linalg.inv(a)@v
        if np.linalg.norm(np.cross(v,other))>1e-8*norm*np.linalg.norm(other):
            raise UnverifiedMagneticSetting('This group and direction generate noncollinear symmetry mates, outside the declared collinear executor')


def verify_collinear_structure(phase):
    from GSASII import GSASIIlattice as L,GSASIIspc as S
    general=phase['General'];cell=general['Cell'][1:7];a=L.cell2AB(cell)[0];d=a@np.diag(1/np.asarray(cell[:3]));cx=general['AtomPtrs'][0]
    directions=[np.asarray(atom[cx+4:cx+7],dtype=float) for atom in phase['Atoms'] if np.linalg.norm(atom[cx+4:cx+7])>1e-10]
    if not directions:return
    reference=d@directions[0]
    for direction in directions:
        vector=d@direction
        if np.linalg.norm(np.cross(reference,vector))>1e-8*np.linalg.norm(reference)*np.linalg.norm(vector):raise UnverifiedMagneticSetting('Different magnetic orbits do not share the declared collinear direction')
    require_collinear_line([r for r,_ in S.AllOps(general['SGData'])[2]],a,cell[:3],directions[0])


def verify_symmetry_moments(phase):
    """Reconstruct each final moment in its site-symmetry invariant subspace."""
    from GSASII import GSASIIspc as S
    from gsas_magnetic_preflight import moment_basis,direction_in_basis
    general=phase['General'];cx,ct,_,_=general['AtomPtrs'];sg=general['SGData']
    for atom in phase['Atoms']:
        vector=np.asarray(atom[cx+4:cx+7],dtype=float)
        if not np.isfinite(vector).all():raise UnverifiedMagneticSetting('A final magnetic moment is nonfinite')
        _,_,_,duplicates=S.SytSym(atom[cx:cx+3],sg)
        indices,coefficients=S.GetCSpqinel(sg['SpnFlp'],duplicates)
        if np.linalg.norm(vector)>1e-10 and not direction_in_basis(moment_basis(indices,coefficients),vector):
            raise UnverifiedMagneticSetting('Final moment violates the exact candidate site symmetry: '+str(atom[ct-1]))


@lru_cache(maxsize=2048)
def _authoritative_operations(identifier):
    from pymatgen.symmetry.maggroups import MagneticSpaceGroup
    identifier=list(identifier) if isinstance(identifier,tuple) else identifier
    try:
        group=MagneticSpaceGroup(identifier)
        operations=frozenset(operation_key(op.rotation_matrix,op.translation_vector,op.time_reversal) for op in group.symmetry_ops)
    except Exception as exc:raise UnverifiedMagneticSetting('The declared magnetic group could not be verified against the BNS database') from exc
    if operation_key(np.eye(3),[0,0,0],-1) in operations:
        raise GreyGroup('A grey magnetic group contains pure time reversal and forbids a nonzero static ordered dipole at every site')
    return operations


def authoritative_operations(candidate):
    identifier=tuple(int(v) for v in candidate['bns_number'].split('.')) if candidate.get('bns_number') else candidate['magnetic_space_group'].replace(' ','')
    return _authoritative_operations(identifier)


def verify_native_operators(candidate,sgdata,*,authority=None):
    from GSASII import GSASIIspc as S
    authority=authoritative_operations(candidate) if authority is None else authority
    operations=S.AllOps(sgdata)[2];spins=sgdata['SpnFlp']
    if len(operations)!=len(spins):raise UnverifiedMagneticSetting('Native magnetic operation/time-reversal inventory is incomplete')
    actual={operation_key(m,t,spin) for (m,t),spin in zip(operations,spins)}
    if actual!=authority:raise UnverifiedMagneticSetting('The native magnetic setting does not reproduce the authoritative BNS operators; an explicit setting transformation is required')
    return dict(complete_bns_operator_inventory_verified=True,operation_count=len(actual),bns_number=candidate.get('bns_number'),pure_time_reversal_absent=True)
