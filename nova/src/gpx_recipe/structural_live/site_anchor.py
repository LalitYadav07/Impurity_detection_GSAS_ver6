"""Restore exact parent-derived zero coordinates after child CIF serialization.

Source atomic positions never
move. A unique parent orbit must lie inside the child's printed-precision box;
the resulting exact anchor then passes the ordinary strict orbit checks.
"""
import copy
import numpy as np

from .source_reference import cif_precision,symmetry_operations


def anchor_child(parent, child, transform, origin, child_cif, native_spc):
    working=copy.deepcopy(child)
    receipt=_anchor(parent,working,np.asarray(transform,float),np.asarray(origin,float),child_cif,native_spc)
    child.clear();child.update(working)
    return receipt


def _anchor(parent, child, transform, origin, child_cif, native_spc):
    coordinates,_=cif_precision(child_cif);pcx,pct,_,_=parent['General']['AtomPtrs']
    cx,ct,cs,_=child['General']['AtomPtrs'];iso=child.get('ISODISTORT',{})
    baseline=np.asarray(iso.get('G2parentCoords',[a[cx:cx+3] for a in child['Atoms']]),float)
    inv=np.linalg.inv(transform.T);operations=symmetry_operations(parent['General']['SGData'])
    rows=[];new=[]
    for atom,old in zip(child['Atoms'],baseline):
        if atom[ct-1] not in coordinates:raise ValueError('Child precision metadata has no matching site label')
        quoted=coordinates[atom[ct-1]];precision=np.minimum([v[1] for v in quoted],5e-5)
        # Exact special coordinates in mode tables may differ from their
        # rounded atom-site serialization; require those tables to agree.
        if np.any(abs((old-np.array([v[0] for v in quoted])+.5)%1-.5)>precision+1e-7):
            raise ValueError('Child mode baseline is outside its serialized site interval')
        target=transform.T@old+origin;matches={}
        for index,source in enumerate(parent['Atoms']):
            if source[pct].casefold()!=atom[ct].casefold():continue
            for rotation,translation in operations:
                position=rotation@source[pcx:pcx+3]+translation
                # Parent translations generate distinct supercell sites; use
                # the integer image nearest this specific child target.
                image=position+np.rint(target-position)
                anchor=inv@(image-origin);delta=anchor-old
                if np.all(abs(delta)<=precision+1e-7):
                    matches.setdefault(index,[]).append(anchor)
        if len(matches)!=1:raise ValueError('No unique parent orbit inside child serialization precision')
        index,options=next(iter(matches.items()))
        anchor=min(options,key=lambda q:float(np.linalg.norm(q-old)))
        if any(np.max(abs(q-anchor))>2e-7 for q in options):
            raise ValueError('Parent operations produce distinct possible child anchors')
        old_mult=native_spc.SytSym(old,child['General']['SGData'])[1]
        sym,mult,*_=native_spc.SytSym(anchor,child['General']['SGData'])
        snapped=[]
        if mult!=old_mult:
            # Native space-group translations include float32 thirds; basis
            # inversion also creates signed machine-zero coordinates. GenAtom
            # can classify these differently at the unit-cell seam. Preserve
            # only the old components within operation-roundoff precision,
            # then require unchanged native multiplicity AND strict parent
            # orbit matching in the caller. Genuine rounding corrections are
            # larger and are not undone here.
            snapped=np.flatnonzero(abs(anchor-old)<=1e-7).tolist()
            stable=anchor.copy();stable[snapped]=old[snapped]
            stable_sym,stable_mult,*_=native_spc.SytSym(stable,child['General']['SGData'])
            if stable_mult==old_mult:anchor=stable;sym=stable_sym;mult=stable_mult
        if mult!=old_mult:raise ValueError('Anchoring changes child site multiplicity')
        atom[cx:cx+3]=anchor.tolist();atom[cs]=sym;atom[cs+1]=mult;new.append(anchor)
        rows.append(dict(child=atom[ct-1],parent=parent['Atoms'][index][pct-1],
                         before=old.tolist(),after=anchor.tolist(),printed_half_unit=list(map(float,precision)),
                         operation_roundoff_preserved_axes=snapped,
                         maximum_shift=float(np.max(abs(anchor-old)))))
    if iso:
        iso['G2parentCoords']=np.asarray(new)
        by_id={atom[-1]:xyz for atom,xyz in zip(child['Atoms'],new)}
        offsets=[]
        for var in iso.get('G2VarList',[]):
            if var.name not in ['dAx','dAy','dAz'] or var.atom not in by_id:
                raise ValueError('Cannot anchor unknown native displacement variable')
            offsets.append(float(by_id[var.atom]['xyz'.index(var.name[-1])]))
        iso['G2coordOffset']=offsets
    return dict(policy='exact_parent_anchor_within_child_serialization_precision',source_coordinates_unchanged=True,
                sites=rows,maximum_shift=max((r['maximum_shift'] for r in rows),default=0.))
