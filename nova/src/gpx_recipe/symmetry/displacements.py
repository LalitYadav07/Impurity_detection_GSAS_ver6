"""Transfer fixed crystallographic Uij without changing its physical ellipsoid.

GSAS-II stores U11,U22,U33,U12,U13,U23 in square angstroms. Its
Debye-Waller quadratic form is h.T @ (D @ U @ D) @ h, where D contains
reciprocal-axis lengths. Candidate basis rows define the new direct axes;
fractional coordinates therefore transform by inv(basis).T. Translations
do not affect a displacement tensor.
"""
import math
import numpy as np


def metric(cell):
    values=np.asarray(cell,dtype=float)
    if values.shape!=(6,) or not np.all(np.isfinite(values)) or np.any(values[:3]<=0) or np.any(values[3:]<=0) or np.any(values[3:]>=180):
        raise ValueError('Fixed displacements need a finite physical unit cell')
    a,b,c,alpha,beta,gamma=values
    ca,cb,cg=np.cos(np.deg2rad([alpha,beta,gamma]))
    g=np.array([[a*a,a*b*cg,a*c*cb],[a*b*cg,b*b,b*c*ca],[a*c*cb,b*c*ca,c*c]])
    if np.linalg.eigvalsh(g).min()<=0:raise ValueError('Fixed displacement cell metric is not positive definite')
    return g


def matrix(six):
    u=np.asarray(six,dtype=float)
    if u.shape!=(6,) or not np.all(np.isfinite(u)):raise ValueError('Anisotropic displacement tensor must contain six finite values')
    result=np.array([[u[0],u[3],u[4]],[u[3],u[1],u[5]],[u[4],u[5],u[2]]])
    if np.linalg.eigvalsh(result).min() < -1e-10:raise ValueError('The source anisotropic displacement tensor is nonphysical')
    return result


def transform_uij(six,parent_cell,child_cell,basis,operation):
    g=metric(parent_cell);new_g=metric(child_cell)
    t=np.asarray(basis,dtype=float);r=np.asarray(operation,dtype=float)
    if t.shape!=(3,3) or r.shape!=(3,3) or not np.all(np.isfinite(t)) or not np.all(np.isfinite(r)) or abs(np.linalg.det(t))<1e-10:
        raise ValueError('Invalid fixed displacement basis/operation')
    if not np.allclose(new_g,t@g@t.T,rtol=1e-8,atol=1e-8):raise ValueError('Candidate cell does not match its displacement basis')
    if not np.allclose(r.T@g@r,g,rtol=1e-8,atol=1e-8):raise ValueError('Parent displacement operation does not preserve its cell metric')
    d=np.sqrt(np.diag(np.linalg.inv(g)));new_d=np.sqrt(np.diag(np.linalg.inv(new_g)))
    covariance=matrix(six)*np.outer(d,d)
    f=np.linalg.inv(t).T@r
    child=f@covariance@f.T/np.outer(new_d,new_d)
    child=(child+child.T)*.5
    return [float(child[i,j]) for i,j in [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]]


def expand_magnetic_phase(parent,child,basis,uvec,vvec,transformer):
    """Use native coordinate expansion without its anisotropic code defect.

    FillUnitCell's A branch adds a lattice vector to the operation number,
    producing codes such as ``0:[1 1 1]``. ADPs do not determine coordinates.
    Expand an internal topology-only copy through its scalar-code path, then
    restore each original ADP type. No calculation or saved physical project
    uses an isotropic approximation: transfer_fixed_displacements installs the
    complete original ellipsoids before the magnetic phase is calculated.
    """
    import copy
    topology=copy.deepcopy(parent);pcia=parent['General']['AtomPtrs'][3]
    for atom in topology['Atoms']:
        if atom[pcia]=='A':atom[pcia]='I'
    result,codes=transformer(topology,child,basis,uvec,vvec,True)
    cia=result['General']['AtomPtrs'][3]
    if len(codes)!=len(result['Atoms']):raise ValueError('Native magnetic coordinate expansion lost its site mapping')
    for atom,code in zip(result['Atoms'],codes):
        try:
            index=int(str(code).split(':',1)[0])
            if index<0:raise ValueError()
            atom[cia]=parent['Atoms'][index][pcia]
        except (ValueError,IndexError,TypeError):raise ValueError('Native magnetic coordinate expansion has an invalid parent identity') from None
    return result,codes


def transfer_fixed_displacements(parent,child,codes,basis):
    """Bind every generated magnetic atom to an exact fixed parent ADP."""
    _pcx,pct,_pcs,pcia=parent['General']['AtomPtrs']
    _cx,ct,_cs,cia=child['General']['AtomPtrs']
    if len(codes)!=len(child['Atoms']):raise ValueError('Fixed displacement site mapping is incomplete')
    rows=[]
    for atom,code in zip(child['Atoms'],codes):
        try:
            source,op=str(code).split('+')[0].split(':');index=int(source);operation=int(op)
            if index<0 or not operation:raise ValueError()
            original=parent['Atoms'][index]
            op_index=abs(operation)%100-1
            if op_index<0:raise ValueError()
            rotation=np.asarray(parent['General']['SGData']['SGOps'][op_index][0],dtype=float)*(1 if operation>0 else -1)
        except (ValueError,IndexError,TypeError):raise ValueError('Invalid parent mapping for fixed magnetic displacements') from None
        mode=original[pcia]
        if mode not in {'I','A'} or atom[cia]!=mode:raise ValueError('Magnetic construction changed the displacement model')
        if 'U' in str(original[pct+1]) or 'U' in str(atom[ct+1]):raise ValueError('Magnetic displacement transfer requires held atomic displacement flags')
        if mode=='I':
            values=[float(original[pcia+1])]
            if not math.isfinite(values[0]) or values[0]<0:raise ValueError('Nonphysical fixed Uiso')
            atom[cia+1]=values[0]
        else:
            values=transform_uij(original[pcia+2:pcia+8],parent['General']['Cell'][1:7],child['General']['Cell'][1:7],basis,rotation)
            atom[cia+2:cia+8]=values
        rows.append(dict(atom_id=str(atom[-1]),label=atom[ct-1],parent_label=original[pct-1],code=code,adp_type=mode,values=values))
    return dict(schema='radar-fixed-magnetic-displacements/v1',policy='preserve_fixed_physical_ellipsoids',atoms=rows)


def verify_fixed_displacements(parent,child,receipt,basis):
    """Recompute from the original parent; never trust only a saved child tensor."""
    import copy
    if receipt.get('schema')!='radar-fixed-magnetic-displacements/v1':raise ValueError('Missing fixed displacement audit')
    rows=receipt['atoms'];ids=[str(a[-1]) for a in child['Atoms']]
    if ids!=[r['atom_id'] for r in rows]:raise ValueError('Magnetic displacement site identities changed')
    expected=transfer_fixed_displacements(parent,copy.deepcopy(child),[r['code'] for r in rows],basis)
    cia=child['General']['AtomPtrs'][3]
    for atom,row,saved in zip(child['Atoms'],expected['atoms'],rows):
        actual=atom[cia+1:cia+2] if atom[cia]=='I' else atom[cia+2:cia+8]
        if atom[cia]!=row['adp_type'] or saved['adp_type']!=row['adp_type'] or not np.allclose(actual,row['values'],rtol=0,atol=1e-10) or not np.allclose(saved['values'],row['values'],rtol=0,atol=1e-10):
            raise ValueError('The fixed magnetic displacement ellipsoid changed')
    return True
