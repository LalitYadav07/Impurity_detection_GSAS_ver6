"""Physical near-neighbor change, separate from diffraction detectability.

The fixed parent graph is geometric, not a guessed chemical-bond model.
Matched unwrapped atom identities and periodic images are retained. Reports
never assign symmetry, drop weak examples, or qualify candidate exclusion.
"""
from dataclasses import dataclass

import numpy as np

from ..structural_live.optimization.chemical_geometry import periodic_contacts
from ..structural_live.optimization.physical_prior_geometry import metric_matrix

SCHEMA='radar-structural-neighbor-change/v1'
STRONG_FRACTION=.02


def _owned(value):
    value=np.asarray(value)
    return np.frombuffer(value.tobytes(),dtype=value.dtype).reshape(value.shape)


def _coordinates(value):
    value=np.asarray(value,float)
    if value.ndim!=2 or value.shape[1]!=3 or not 1<=len(value)<=4096 or not np.isfinite(value).all():
        raise ValueError('Finite matching unwrapped atom coordinates required')
    return value


def _distances(xyz,metric,pairs,images):
    first=xyz[pairs[:,0]];second=xyz[pairs[:,1]]
    vectors=second-first+images;g=metric_matrix(metric)
    squared=np.einsum('ni,ij,nj->n',vectors,g,vectors)
    eps=np.finfo(float).eps;coordinate_error=4*eps*(abs(first)+abs(second)+abs(images))
    bound=64*eps*np.einsum('ni,ij,nj->n',abs(vectors),abs(g),abs(vectors))
    bound+=2*np.einsum('ni,ij,nj->n',abs(vectors),abs(g),coordinate_error)
    bound+=np.einsum('ni,ij,nj->n',coordinate_error,abs(g),coordinate_error)
    if np.any(squared < -bound):raise ValueError('Unresolved near-neighbor distance arithmetic')
    distances=np.sqrt(np.maximum(squared,0.))
    errors=bound/np.maximum(distances,np.finfo(float).tiny)
    return distances,errors


@dataclass(frozen=True)
class NeighborGraph:
    coordinates:np.ndarray
    metric:np.ndarray
    pairs:np.ndarray
    images:np.ndarray
    excluded_coincident_contacts:int
    partially_occupied_sites:int

    def compare(self,coordinates,metric,*,reference_coordinates=None,reference_metric=None):
        """Compare to the corresponding parent at the same acquisition context.

        Supplying both reference arrays removes ordinary parent-cell changes
        while retaining the child's symmetry-breaking strain and positions.
        Fractional coordinates must not be independently wrapped or reordered.
        """
        xyz=_coordinates(coordinates)
        if xyz.shape!=self.coordinates.shape:raise ValueError('Retain the complete matching atom identities')
        if (reference_coordinates is None)!=(reference_metric is None):
            raise ValueError('A matched parent reference needs both coordinates and cell metric')
        ref=self.coordinates if reference_coordinates is None else _coordinates(reference_coordinates)
        if ref.shape!=xyz.shape:raise ValueError('The parent reference changed atom identities')
        ref_g=self.metric if reference_metric is None else reference_metric
        before,be=_distances(ref,ref_g,self.pairs,self.images)
        after,ae=_distances(xyz,metric,self.pairs,self.images)
        if np.any(before<=be):raise ValueError('Reference contact precision cannot resolve relative change')
        relative=abs(after-before)/before
        error=(ae+(1+relative)*be)/before+8*np.finfo(float).eps*(1+relative)
        above=relative-error>STRONG_FRACTION
        boundary=(relative-error<=STRONG_FRACTION)&(relative+error>=STRONG_FRACTION)
        index=int(np.argmax(relative));unresolved=bool(self.excluded_coincident_contacts)
        strong=None if unresolved or (boundary.any() and not above.any()) else bool(above.any())
        return dict(schema=SCHEMA,contacts=len(relative),strong_fraction=STRONG_FRACTION,
            maximum_relative_change=float(relative[index]),median_relative_change=float(np.median(relative)),
            percentile95_relative_change=float(np.quantile(relative,.95)),
            maximum_absolute_change_angstrom=float(np.max(abs(after-before))),
            contacts_certainly_above_threshold=int(above.sum()),threshold_boundary_contacts=int(boundary.sum()),
            strong_neighbor_distance_change=strong,excluded_coincident_contacts=self.excluded_coincident_contacts,
            partially_occupied_sites=self.partially_occupied_sites,
            largest_change=dict(atoms=self.pairs[index].tolist(),image=self.images[index].tolist(),
                reference_angstrom=float(before[index]),current_angstrom=float(after[index]),
                relative_arithmetic_allowance=float(error[index])),
            scope='Fixed parent nearest-distance shells, not chemical bonds or a diffraction-detectability threshold',
            screening_qualified=False)


def from_parent(coordinates,direct_metric,occupancy,*,max_images=2000000):
    xyz=_coordinates(coordinates);g=metric_matrix(direct_metric);occ=np.asarray(occupancy,float)
    if occ.shape!=(len(xyz),) or not np.isfinite(occ).all() or np.any(occ<0) or np.any(occ>1) or not occ.any():
        raise ValueError('One finite physical occupancy per parent atom required')
    active=np.flatnonzero(occ>0)
    # A primitive cell edge is an upper bound on a site's nearest periodic
    # image. Existing skew-cell enumeration checks every relevant image.
    cutoff=float(np.sqrt(np.min(np.diag(g))))
    contacts=periodic_contacts(xyz[active],g,cutoff,max_images=max_images)
    nonzero=[r for r in contacts if r[3]>0.]
    excluded=len(contacts)-len(nonzero)
    nearest=np.full(len(active),np.inf)
    for i,j,_,d in nonzero:nearest[i]=min(nearest[i],d);nearest[j]=min(nearest[j],d)
    if not np.isfinite(nearest).all():raise ValueError('The bounded parent contact search has no complete nearest-distance graph')
    # This arithmetic-scale shell tolerance groups symmetry-equivalent
    # parent distances; it is not a physical distortion or signal threshold.
    selected=[(int(active[i]),int(active[j]),shift) for i,j,shift,d in nonzero
              if d<=max(nearest[i],nearest[j])*(1+1e-8)]
    if not selected:raise ValueError('The complete parent nearest-distance graph is empty')
    pairs=np.asarray([(i,j) for i,j,_ in selected],np.int64)
    images=np.asarray([s for _,_,s in selected],np.int64)
    return NeighborGraph(_owned(xyz),_owned(g),_owned(pairs),_owned(images),excluded,
        int(np.count_nonzero((occ>0)&(occ<1))))
