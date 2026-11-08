"""Native irrep-family amplitudes, preserving unknown and origin-gauge modes.

These are geometric simulation targets. They are not probabilities, measured
detectability limits, evidence for the primary order parameter, or permission
to assert experimental absence. The parent convention and exact k are kept;
integer reciprocal shifts are not identified in a centered parent lattice.
"""
from fractions import Fraction
import re

import numpy as np

from ..symmetry.contracts import digest
from .structural_mode_directions import from_basis
from .structural_mode_projection import Projection,descriptor
from .tof_kernel_arrays import checksum

SCHEMA='radar-native-structural-mode-families/v1'
NORMALIZATION='baseline-complete-occupied-site-cartesian-rms/native-gauge-section/v1'
MAX_BATCH_ROWS=256
AUDIT_TOLERANCE=1e-10
LABEL=re.compile(r'^(?P<parent>[^\[\]]+)\[(?P<k>[^\[\]]+)\]'
    r'(?P<irrep>[A-Za-z][A-Za-z0-9_+\-]*)\((?P<opd>[^\[\]]*)\)'
    r'\[(?P<site>[^:\[\]]+):(?P<wyckoff>[^:\[\]]+):dsp\](?P<component>.+)$')


def family_label(label,*,parent_sha256):
    """Parse only the explicit native displacement grammar; otherwise unknown."""
    if not isinstance(label,str) or not checksum(parent_sha256):
        raise ValueError('Native mode-family labels require their exact parent identity')
    found=LABEL.fullmatch(label)
    if found is None:return None
    values=found.groupdict();parts=values['k'].split(',')
    if len(parts)!=3 or any(not p.strip() or len(p.strip())>80 for p in parts):return None
    try:k=[str(Fraction(p.strip())) for p in parts]
    except (ValueError,ZeroDivisionError):return None
    identity=dict(parent_sha256=parent_sha256,parent_convention=values['parent'],
        k=k,irrep=values['irrep'])
    return dict(id=digest(identity),**identity,order_parameter_direction=values['opd'],
        site=values['site'],wyckoff=values['wyckoff'],component=values['component'])


class FamilyProjection:
    """One complete native candidate; no family is inferred from an SG number."""
    def __init__(self,projection):
        if not isinstance(projection,Projection):
            raise TypeError('Family extraction requires a checked native mode projection')
        d=projection.metadata;a=projection.arrays
        mapping=a['physical_to_native'];gauges=a['gauge_basis'];gram=a['physical_gram']
        directions=from_basis(mapping,gauges,gram,d['native_mode_labels'])
        n,m=mapping.shape
        left=(directions.physical/directions.native_scale[:,None]).T
        cholesky=np.linalg.cholesky((gram+gram.T)/2)
        families={};opaque=[]
        for i,label in enumerate(d['native_mode_labels']):
            parsed=family_label(label,parent_sha256=d['source_sha256'])
            if parsed is None:
                opaque.append(dict(native_index=i,label=label,reason='unrecognized_native_displacement_label'))
                continue
            row=families.setdefault(parsed['id'],dict(id=parsed['id'],parent_sha256=parsed['parent_sha256'],
                parent_convention=parsed['parent_convention'],k=parsed['k'],irrep=parsed['irrep'],
                native_indices=[],native_labels=[],order_parameter_directions=[]))
            row['native_indices'].append(i);row['native_labels'].append(label)
            if parsed['order_parameter_direction'] not in row['order_parameter_directions']:
                row['order_parameter_directions'].append(parsed['order_parameter_direction'])
        rows=[];factors=[]
        for identifier in sorted(families):
            row=families[identifier];indices=row['native_indices'];sub=left[:,indices]
            leakage=sub@gauges[indices]
            bound=np.abs(sub)@np.abs(gauges[indices])
            # This is the native coordinate reconstruction tolerance, never a
            # scattering cutoff or a rule for deleting weak physical axes.
            residual=float(np.max(np.abs(leakage),initial=0.))
            # Compare with this family's response to complete physical
            # coordinates. A local |L_S| |G_S| denominator can itself be
            # roundoff-sized for a gauge outside this irrep, incorrectly
            # converting harmless SVD noise into order-one relative leakage.
            # Do not use an absolute floor: weak-unit but genuinely coupled
            # coordinates must still remain unresolved.
            response=sub@mapping[indices]
            response_bound=np.abs(sub)@np.abs(mapping[indices])
            scale=float(np.max(response_bound,initial=0.))
            invariant=residual<=AUDIT_TOLERANCE*max(scale,np.finfo(float).tiny)
            pure_gauge=bool(directions.gauge_only[indices].all())
            row.update(origin_gauge_invariant=bool(invariant),pure_origin_gauge=pure_gauge,
                gauge_reconstruction_residual=residual,gauge_reconstruction_scale=scale,
                gauge_contraction_roundoff_scale=float(np.max(bound,initial=0.)),
                amplitude_target_available=bool(invariant and not pure_gauge))
            factor=sub.T@cholesky
            if (not np.isfinite(factor).all() or not np.isfinite(leakage).all() or not np.isfinite(bound).all()
                    or not np.isfinite(response).all() or not np.isfinite(response_bound).all()):
                raise ValueError('Native family projection is not numerically resolved')
            rows.append(row);factors.append(factor)
        self.families=tuple(rows);self.opaque_native_axes=tuple(opaque)
        self._mapping=mapping;self._factors=tuple(factors);self.mode_dimensions=m
        self.metadata=dict(schema=SCHEMA,projection=descriptor(projection),normalization=NORMALIZATION,
            families=list(self.families),opaque_native_axes=list(self.opaque_native_axes),
            native_axes=n,physical_dimensions=m,recognized_native_axes=n-len(opaque),
            all_native_axes_accounted=True,experimental_mode_absence_qualified=False,
            candidate_pruning_authorized=False)

    def native_amplitudes(self,points):
        """The same indexed amplitude units consumed by native ApplyModeDisp."""
        p=np.asarray(points)
        if (p.dtype.kind!='f' or p.ndim!=2 or not 1<=len(p)<=MAX_BATCH_ROWS
                or p.shape[1]!=self.mode_dimensions or not np.isfinite(p).all()):
            raise ValueError('Project a bounded batch of complete finite physical mode coordinates')
        result=p.astype(np.float64)@self._mapping.T
        if not np.isfinite(result).all():raise ValueError('Native mode amplitudes exceed finite numeric range')
        return result

    def project(self,points):
        """Continuous baseline RMS per family; missing targets remain masked."""
        native=self.native_amplitudes(points);values=np.empty((len(native),len(self.families)),np.float64)
        known=np.zeros(len(self.families),bool)
        for i,(row,factor) in enumerate(zip(self.families,self._factors)):
            physical=native[:,row['native_indices']]@factor
            scale=np.max(abs(physical),axis=1,initial=0.)
            scaled=physical/np.where(scale>0,scale,1.)[:,None]
            values[:,i]=scale*np.sqrt(np.sum(scaled*scaled,axis=1))
            known[i]=row['amplitude_target_available']
        if not np.isfinite(values).all():raise ValueError('Family amplitude exceeded finite baseline coordinates')
        return dict(amplitudes=values,known=np.broadcast_to(known,values.shape).copy(),
            families=[r['id'] for r in self.families],normalization=NORMALIZATION,
            experimental_mode_absence_qualified=False,candidate_pruning_authorized=False)
