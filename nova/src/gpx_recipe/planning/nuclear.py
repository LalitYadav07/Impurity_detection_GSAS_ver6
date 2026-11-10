"""Independent direct-cell nuclear structure-factor verification.

Expand explicit CIF symmetry operators and rotate displacement tensors without
GSAS symmetry or structure-factor routines. Scattering lengths are supplied in
one explicitly shared unit system; this checks numerical kinematics rather than
atomic-data accuracy, instrument response, physical specimen identity or
magnetism. CIF parsing uses PyCifRW in the scientific validation environment.
No sample/count calibration or fitted intensity scale enters this calculation.
"""
from pathlib import Path
from fractions import Fraction
import ast
import re
import numpy as np

def number(value):
    text=re.sub(r'\(\d+\)','',str(value))
    result=float(text)
    if not np.isfinite(result):raise ValueError('Finite explicit CIF numbers required')
    return result

def affine(expression):
    tree=ast.parse(expression,mode='eval')
    if sum(1 for _ in ast.walk(tree))>64:raise ValueError('Symmetry expression budget')
    zero=lambda:[Fraction(0) for _ in range(4)]
    def parse(node):
        if isinstance(node,ast.Name) and node.id in ('x','y','z'):
            v=zero();v['xyz'.index(node.id)]=Fraction(1);return v
        if isinstance(node,ast.Constant) and type(node.value) is int:
            v=zero();v[3]=Fraction(node.value);return v
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            return [(-x if isinstance(node.op,ast.USub) else x) for x in parse(node.operand)]
        if isinstance(node,ast.BinOp):
            left,right=parse(node.left),parse(node.right)
            if isinstance(node.op,(ast.Add,ast.Sub)):
                return [a+(-b if isinstance(node.op,ast.Sub) else b) for a,b in zip(left,right)]
            if isinstance(node.op,ast.Mult):
                if not any(left[:3]):return [left[3]*x for x in right]
                if not any(right[:3]):return [right[3]*x for x in left]
            if isinstance(node.op,ast.Div) and not any(right[:3]) and right[3]:
                return [x/right[3] for x in left]
        raise ValueError('Only affine rational CIF symmetry expressions are supported')
    result=parse(tree.body)
    if any(abs(x)>16 for x in result):raise ValueError('Symmetry coefficient budget')
    return np.array([float(x) for x in result])

def full_cell(path, *, exact_atom_types=False):
    path=Path(path)
    if path.stat().st_size>2*1024**2:raise ValueError('CIF byte budget')
    import CifFile
    data=CifFile.ReadCif(str(path))
    if len(data.keys())!=1:raise ValueError('Exactly one explicit CIF phase required')
    block=data[data.keys()[0]]
    if any('modulation' in key.lower() or 'fourier' in key.lower() for key in block.keys()):
        raise ValueError('Modulated CIFs require a separately validated structure model')
    lengths=np.array([number(block['_cell_length_'+v]) for v in 'abc'])
    alpha,beta,gamma=np.deg2rad([number(block['_cell_angle_'+v]) for v in ('alpha','beta','gamma')])
    if np.any(lengths<=0) or not np.all((np.array([alpha,beta,gamma])>0)&(np.array([alpha,beta,gamma])<np.pi)):
        raise ValueError('Positive lengths and physical cell angles required')
    a,b,c=lengths
    basis=np.array([[a,b*np.cos(gamma),c*np.cos(beta)],
        [0.,b*np.sin(gamma),c*(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma)],
        [0.,0.,0.]])
    squared_height=c*c-basis[0,2]**2-basis[1,2]**2
    if squared_height<=1e-12:
        raise ValueError('Nondegenerate positive cell metric required')
    basis[2,2]=np.sqrt(squared_height)
    metric=basis.T@basis
    reciprocal=np.linalg.inv(metric)
    reciprocal_lengths=np.sqrt(np.diag(reciprocal))
    operations=block.get('_space_group_symop_operation_xyz',block.get('_symmetry_equiv_pos_as_xyz'))
    if operations is None:raise ValueError('Full explicit CIF symmetry operators required')
    if isinstance(operations,str):operations=[operations]
    if len(operations)>384:raise ValueError('Symmetry operation budget')
    matrices=[]
    for operation in operations:
        parts=operation.lower().replace(' ','').split(',')
        if len(parts)!=3:raise ValueError('Three affine coordinate expressions required')
        components=np.array([affine(s) for s in parts])
        rotation,shift=components[:,:3],components[:,3]
        if not np.isclose(abs(np.linalg.det(rotation)),1.) or not np.allclose(rotation,np.rint(rotation)):
            raise ValueError('Unimodular integral CIF lattice rotations required')
        if not np.allclose(rotation.T@metric@rotation,metric,rtol=1e-8,atol=1e-8):
            raise ValueError('CIF operation does not preserve the declared cell metric')
        matrices.append((rotation,shift))
    anisotropic={}
    aniso_labels=block.get('_atom_site_aniso_label',[])
    for i,label in enumerate(aniso_labels):
        tensor=np.zeros((3,3))
        for j,k in [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]:
            ukey=f'_atom_site_aniso_U_{j+1}{k+1}'
            bkey=f'_atom_site_aniso_B_{j+1}{k+1}'
            if ukey not in block and bkey not in block:
                raise ValueError('Six explicit anisotropic U or B components are required')
            value=number(block[ukey][i]) if ukey in block else number(block[bkey][i])/(8*np.pi**2)
            if ukey in block and bkey in block and not np.isclose(value,number(block[bkey][i])/(8*np.pi**2),rtol=1e-6,atol=1e-10):
                raise ValueError('Conflicting U and B displacement parameters require explicit resolution')
            tensor[j,k]=tensor[k,j]=value
        anisotropic[label]=tensor*np.outer(reciprocal_lengths,reciprocal_lengths)
    positions,tensors,species,occupancies=[],[],[],[]
    site_counts={}
    labels=block['_atom_site_label']
    if not 1<=len(labels)<=1000:
        raise ValueError('Asymmetric atom-site budget exceeded')
    for i,label in enumerate(labels):
        xyz=np.array([number(block['_atom_site_fract_'+v][i]) for v in 'xyz'])
        occupancy=number(block.get('_atom_site_occupancy',['1']*len(labels))[i])
        if not 0<=occupancy<=1:
            raise ValueError('Site occupancies must lie in [0, 1]')
        raw_type=str(block['_atom_site_type_symbol'][i])
        symbol=raw_type if exact_atom_types else re.match(r'^[A-Z][a-z]?',raw_type).group()
        if label in anisotropic:
            tensor=anisotropic[label]
        else:
            key='_atom_site_U_iso_or_equiv'
            bkey='_atom_site_B_iso_or_equiv'
            if key not in block and bkey not in block:raise ValueError('Explicit isotropic/anisotropic ADP required')
            uiso=number(block[key][i]) if key in block else number(block[bkey][i])/(8*np.pi**2)
            if key in block and bkey in block and not np.isclose(uiso,number(block[bkey][i])/(8*np.pi**2),rtol=1e-6,atol=1e-10):
                raise ValueError('Conflicting U and B displacement parameters require explicit resolution')
            tensor=uiso*reciprocal
        if not np.isfinite(tensor).all() or np.linalg.eigvalsh(tensor).min() < -1e-12:
            raise ValueError('Thermal tensor has a negative eigenvalue')
        seen={}
        for rotation,shift in matrices:
            transformed=(rotation@xyz+shift)%1.
            transformed[np.isclose(transformed,1.,atol=1e-9)]=0.
            transformed_tensor=rotation@tensor@rotation.T
            key=tuple(np.round(transformed,8))
            if key in seen:
                if not np.allclose(seen[key],transformed_tensor,rtol=1e-6,atol=1e-10):
                    raise ValueError('ADP violates special-position symmetry')
                continue
            seen[key]=transformed_tensor
            positions.append(transformed);tensors.append(transformed_tensor)
            species.append(symbol);occupancies.append(occupancy)
            if len(positions)>10000:
                raise ValueError('Expanded atom budget exceeded')
        declared=block.get('_atom_site_symmetry_multiplicity')
        if declared is not None:
            multiplicity=number(declared[i])
            if not multiplicity.is_integer() or len(seen)!=int(multiplicity):
                raise ValueError('Explicit CIF operators do not reproduce declared site multiplicity')
        site_counts[label]=len(seen)
    return np.asarray(positions),np.asarray(tensors),species,np.asarray(occupancies),site_counts

def independent_factors(path,hkl,scattering):
    """Return the Friedel-averaged F² in the caller's scattering-length units.

    This reference requires explicit full CIF operators and ADPs, and checks
    declared site multiplicities. It does not independently certify the CIF's
    space-group assignment, specimen correspondence or scattering-length data.
    """
    hkl=np.asarray(hkl,float)
    if (hkl.ndim!=2 or hkl.shape[1]!=3 or len(hkl)<1 or len(hkl)>20000 or
        not np.isfinite(hkl).all() or np.any(np.abs(hkl)>4096) or
        not np.allclose(hkl,np.rint(hkl),rtol=0,atol=1e-12)):
        raise ValueError('Bounded integer h,k,l rows are required')
    xyz,tensor,species,occupancy,counts=full_cell(path)
    if len(hkl)*len(xyz)>20000000:
        raise ValueError('Structure-factor work budget exceeded')
    phase=2*np.pi*(hkl@xyz.T)
    exponent=-2*np.pi**2*np.einsum('hi,aij,hj->ha',hkl,tensor,hkl)
    thermal=np.exp(exponent)
    coefficients=np.array([scattering[e] for e in species])*occupancy
    if not np.isfinite(coefficients).all():
        raise ValueError('Finite explicitly supplied scattering lengths required')
    forward=(np.exp(1j*phase)*thermal)@coefficients
    reverse=(np.exp(-1j*phase)*thermal)@coefficients
    return (np.abs(forward)**2+np.abs(reverse)**2)/2,counts
