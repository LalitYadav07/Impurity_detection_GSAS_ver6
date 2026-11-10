"""Shared exact-parent expansion for finite commensurate k discovery."""
import numpy as np


def parent_k_points(parent, *, maximum_denominator=8):
    from ..commensurate_k import recover_points
    from GSASII import GSASIIlattice as L
    if type(maximum_denominator) is not int or not 1 <= maximum_denominator <= 8:
        raise ValueError('Declare a finite supported k denominator')
    # Expand exact native operations. Preserve mixed-site chemistry as one
    # composite spglib type, rather than duplicating atoms at the same position.
    positions=[];motifs=[]
    for atom in parent['atoms']:
        if atom['occupancy']<=0:continue
        if not np.isfinite(atom['occupancy']) or atom['occupancy']>1+1e-7:raise ValueError('Review nonphysical parent occupancies before k screening')
        orbit=[]
        for op in parent['operations']:
            pos=(np.asarray(op['rotation'])@atom['xyz']+op['translation'])%1
            if any(np.max(np.abs((pos-old+.5)%1-.5))<1e-7 for old in orbit):continue
            orbit.append(pos)
            found=next((i for i,old in enumerate(positions) if np.max(np.abs((pos-old+.5)%1-.5))<1e-7),None)
            if found is None:found=len(positions);positions.append(pos);motifs.append([])
            motifs[found].append((atom['type'],round(atom['occupancy'],7)))
    if not positions or len(positions)>5000:raise ValueError('Expanded parent exceeds the k-survey size budget')
    types={};numbers=[]
    for motif in motifs:
        token=tuple(sorted(motif));types.setdefault(token,len(types)+1);numbers.append(types[token])
    structure=(L.cell2AB(parent['cell'])[0].T,np.asarray(positions),numbers)
    return recover_points(structure,parent['operations'],maximum_denominator=maximum_denominator)
