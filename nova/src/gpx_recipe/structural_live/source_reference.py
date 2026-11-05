"""Explicit derived reference preparation for rounded published CIF values.

This is opt-in reference construction, never a silent relaxation of child
mapping and never a mutation of the source CIF or an authoritative fitted GPX.
Coordinates move only within printed precision. ADP symmetry projections must
fit within the quoted one-standard-uncertainty plus rounding interval, and are
reported as a derived reference assumption rather than measured corrections.
"""
from __future__ import annotations
import copy
import re

import numpy as np

from ..symmetry.displacements import transform_uij, matrix


def numeric_precision(token):
    match=re.fullmatch(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:\((\d+)\))?([eE][+-]?\d+)?',str(token).strip())
    if not match:raise ValueError('A finite CIF numeric value with known precision is required')
    mantissa,su,exponent=match.groups();power=int((exponent or 'e0')[1:])
    decimals=len(mantissa.split('.')[1]) if '.' in mantissa else 0
    quantum=10.**(power-decimals)
    # Integer coordinates conventionally encode exact special values. Do not
    # give a literal zero a fictitious half-unit freedom.
    rounding=.5*quantum if '.' in mantissa or exponent else 0.
    value=float(mantissa+(exponent or ''))
    sigma=int(su)*quantum if su else 0.
    if not np.isfinite([value,rounding,sigma]).all():raise ValueError('Nonfinite CIF number')
    return value,rounding,sigma


def symmetry_operations(sg):
    return [(sign*np.asarray(r,float),sign*np.asarray(t,float)+center)
            for sign in ([1,-1] if sg.get('SGInv') else [1])
            for r,t in sg['SGOps'] for center in sg['SGCen']]


def project_rounded_site(xyz, precision, operations, *, maximum_shift=5e-5):
    x=np.asarray(xyz,float);p=np.minimum(np.asarray(precision,float),maximum_shift)
    if x.shape!=(3,) or p.shape!=(3,) or not np.isfinite(x).all() or not np.isfinite(p).all() or np.any(p<0):
        raise ValueError('Invalid coordinate precision interval')
    matrices=[];residuals=[]
    for rotation,translation in operations:
        r=np.asarray(rotation);t=np.asarray(translation);a=r-np.eye(3)
        value=a@x+t;residual=value-np.rint(value)
        budget=(abs(r)+np.eye(3))@p+1e-12
        if np.all(abs(residual)<=budget):
            matrices.append(a);residuals.append(residual)
    if not matrices:raise ValueError('Identity operation missing from reference symmetry')
    a=np.concatenate(matrices);residual=np.concatenate(residuals)
    # Weighted minimum shift; exact printed integers are held to numerical
    # tolerance, independently of other coordinates' printed precision.
    scale=np.maximum(p,1e-12)
    scaled_shift=np.linalg.lstsq(a*scale[None,:],-residual,rcond=1e-12)[0]
    delta=scale*scaled_shift;projected=x+delta
    if np.max(abs(a@delta+residual),initial=0)>1e-10 or np.any(abs(delta)>p+1e-10):
        raise ValueError('Source rounding intervals cannot satisfy site symmetry together')
    stabilizers=[r for r,t in operations if np.max(abs((r@projected+t-projected+.5)%1-.5))<1e-9]
    if not stabilizers:raise ValueError('Canonical site lacks a verified stabilizer')
    return projected,stabilizers


def cif_precision(path):
    import CifFile
    cif=CifFile.ReadCif(str(path))
    blocks=[cif[k] for k in cif.keys() if '_atom_site_label' in cif[k]]
    if len(blocks)!=1:raise ValueError('Reference precision requires one unambiguous atom-site block')
    block=blocks[0];labels=list(block['_atom_site_label'])
    if len(labels)!=len(set(labels)):raise ValueError('Duplicate source site labels')
    coordinates={label:[numeric_precision(block['_atom_site_fract_'+axis][i]) for axis in 'xyz']
                 for i,label in enumerate(labels)}
    adps={}
    if '_atom_site_aniso_label' in block:
        for i,label in enumerate(block['_atom_site_aniso_label']):
            if label in adps:raise ValueError('Duplicate source anisotropic site label')
            adps[label]=[numeric_precision(block['_atom_site_aniso_U_'+key][i])
                         for key in ['11','22','33','12','13','23']]
    return coordinates,adps


def canonicalize_public_reference(phase, source_cif, native_spc):
    """Edit only an explicitly derived public-CIF reference and return deltas."""
    working=copy.deepcopy(phase)
    receipt=_canonicalize(working,source_cif,native_spc)
    phase.clear();phase.update(working)
    return receipt


def _canonicalize(phase, source_cif, native_spc):
    coordinates,adps=cif_precision(source_cif)
    cx,ct,cs,cia=phase['General']['AtomPtrs'];cell=phase['General']['Cell'][1:7]
    operations=symmetry_operations(phase['General']['SGData']);rows=[]
    if set(coordinates)!={a[ct-1] for a in phase['Atoms']}:
        raise ValueError('Reference native labels do not exactly match CIF labels')
    for atom in phase['Atoms']:
        label=atom[ct-1];quoted=coordinates[label]
        before=np.asarray(atom[cx:cx+3],float)
        if not np.allclose(before,[x[0] for x in quoted],rtol=0,atol=1e-10):
            raise ValueError('Reference coordinates differ from source; fitted GPX needs its own explicit provenance')
        projected,stabilizers=project_rounded_site(before,[x[1] for x in quoted],operations)
        old_multiplicity=native_spc.SytSym(before,phase['General']['SGData'])[1]
        new_sym,new_multiplicity,*_=native_spc.SytSym(projected,phase['General']['SGData'])
        if old_multiplicity!=new_multiplicity:
            raise ValueError('Reference projection changes native site multiplicity')
        atom[cx:cx+3]=projected.tolist();atom[cs]=new_sym;atom[cs+1]=new_multiplicity
        row=dict(label=label,coordinate_before=before.tolist(),coordinate_after=projected.tolist(),
                 maximum_coordinate_change=float(np.max(abs(projected-before))),multiplicity=new_multiplicity)
        if atom[cia]=='A':
            if label not in adps:raise ValueError('Anisotropic reference has no source precision metadata')
            before_u=np.asarray(atom[cia+2:cia+8],float);quoted_u=adps[label]
            if not np.allclose(before_u,[x[0] for x in quoted_u],rtol=0,atol=1e-10):
                raise ValueError('Native ADPs differ from quoted source before canonicalization')
            tensors=np.array([transform_uij(before_u,cell,cell,np.eye(3),r) for r in stabilizers])
            projected_u=tensors.mean(axis=0);budget=np.array([x[1]+x[2] for x in quoted_u])
            if np.any(abs(projected_u-before_u)>budget+1e-10):
                raise ValueError('ADP symmetry projection exceeds reported uncertainty and rounding')
            matrix(projected_u)
            if not all(np.allclose(transform_uij(projected_u,cell,cell,np.eye(3),r),projected_u,atol=1e-10,rtol=0)
                       for r in stabilizers):raise ValueError('Projected ADP is not invariant under its site symmetry')
            atom[cia+2:cia+8]=projected_u.tolist()
            row.update(adp_before=before_u.tolist(),adp_after=projected_u.tolist(),adp_change=(projected_u-before_u).tolist(),
                       adp_allowed_change=budget.tolist())
        rows.append(row)
    return dict(schema='structural-derived-reference/v1',policy='printed-coordinate-precision_and_quoted-ADP-uncertainty',
                source_cif=str(source_cif),source_preserved=True,changes_are_derived_reference_assumptions=True,sites=rows)
