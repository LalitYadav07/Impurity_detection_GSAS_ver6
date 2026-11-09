"""Bound complete ISODISTORT strain maps and typed family supervision.

No scattering calculation or new family is guessed from a space-group number.
Native cell parameters provide the targets; the measured cold curve is never
used. Normalized ISO coordinates and physical dimensionless family magnitudes
are distinct outputs. Neither establishes experimental presence or absence.
"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import io
import re

import numpy as np

from ..symmetry.contracts import validate_candidate,fields
from .structural_strain import StrainProjection,cell_metric
from .structural_mode_families import family_label,STRAIN_NORMALIZATION,MAX_BATCH_ROWS
from .tof_kernel_arrays import read_bounded,checksum

SCHEMA='radar-bound-isodistort-strain-families/v1'
# The strain portion is tiny, but complete displacement matrices can be large.
# Do not impose the smaller JSON-sidecar budget on the original native CIF.
MAX_CIF_BYTES=64*1024**2
CELL_KEYS=('length_a','length_b','length_c','angle_alpha','angle_beta','angle_gamma')
_TERM=re.compile(r'([+-]?)(?:(\d+(?:/\d+)?|\d*\.\d+)\*?)?([abc])(?:/(\d+))?')


def _values(block,tag):
    if tag not in block:raise ValueError('Complete ISODISTORT strain table required: '+tag)
    v=block[tag]
    return [v] if isinstance(v,str) else list(v)


def _integer(value):
    if not isinstance(value,str) or re.fullmatch(r'\d{1,6}',value) is None:
        raise ValueError('Bounded integer strain indices required')
    return int(value)


def _indexed(block,id_tag,value_tag,count):
    ids=[_integer(v) for v in _values(block,id_tag)];values=_values(block,value_tag)
    if len(ids)!=count or len(values)!=count or sorted(ids)!=list(range(1,count+1)):
        raise ValueError('Strain IDs must retain each declared coordinate exactly once')
    return [v for _,v in sorted(zip(ids,values))]


def _transform(value):
    """Small rational linear grammar; no expression evaluator or code loading."""
    if not isinstance(value,str) or not 1<=len(value)<=2000 or value.count(';')!=1:
        raise ValueError('Explicit ISO parent-to-child basis and origin required')
    basis,origin=value.replace(' ','').split(';');rows=[]
    for expression in basis.split(','):
        parts=re.findall(r'[+-]?[^+-]+',expression)
        if not parts or ''.join(parts)!=expression:raise ValueError('Invalid rational child basis')
        row=[Fraction(0),Fraction(0),Fraction(0)]
        for term in parts:
            m=_TERM.fullmatch(term)
            if m is None:raise ValueError('Only rational linear child-basis terms are supported')
            sign,coefficient,axis,denominator=m.groups()
            try:row['abc'.index(axis)]+=(-1 if sign=='-' else 1)*Fraction(coefficient or '1')/int(denominator or '1')
            except (ValueError,ZeroDivisionError) as error:raise ValueError('Finite rational child basis required') from error
        rows.append(row)
    try:shift=[Fraction(v) for v in origin.split(',')]
    except (ValueError,ZeroDivisionError) as error:raise ValueError('Finite rational child origin required') from error
    if len(rows)!=3 or len(shift)!=3:raise ValueError('Three complete child basis rows and origin components required')
    matrix=np.array(rows,float);offset=np.array(shift,float)
    if not np.isfinite(matrix).all() or not np.isfinite(offset).all():raise ValueError('Finite ISO transformation required')
    return matrix,offset


def _tensor(e):
    return np.array([[e[0],e[5]/2,e[4]/2],[e[5]/2,e[1],e[3]/2],[e[4]/2,e[3]/2,e[2]]])


@dataclass(frozen=True)
class StrainFamilies:
    projection:StrainProjection
    metadata:dict
    families:tuple
    indices:tuple

    def project(self,child_metrics,*,parent_metrics=None):
        """Use the same explicit current parent context as the native curve.

        For one family, sym(B ΔS B^-1) is the Cartesian infinitesimal strain
        tensor; B.T B is the parent direct metric. Its Frobenius norm is
        independent of laboratory rotation, dimensionless, and not an atomic
        displacement or a calibrated experimental detection threshold.
        """
        child=np.asarray(child_metrics,float)
        parent=(np.broadcast_to(self.projection.parent,child.shape) if parent_metrics is None
                else np.asarray(parent_metrics,float))
        if (child.ndim!=3 or child.shape[1:]!=(3,3) or not 1<=len(child)<=MAX_BATCH_ROWS
                or parent.shape!=child.shape or not np.isfinite(child).all() or not np.isfinite(parent).all()):
            raise ValueError('A bounded complete batch of child/current-parent metrics required')
        values=np.empty((len(child),len(self.families)),np.float64)
        native=np.empty((len(child),self.projection.dimensions),np.float64)
        for i,(c,p) in enumerate(zip(child,parent)):
            native[i]=self.projection.project(c,parent_metric=p)['amplitudes']
            b=np.linalg.cholesky(p).T
            for j,columns in enumerate(self.indices):
                e=self.projection.matrix[:,columns]@native[i,list(columns)]
                deformation=np.linalg.solve(b.T,(b@_tensor(e)).T).T
                tensor=(deformation+deformation.T)/2
                scale=float(np.max(abs(tensor)))
                values[i,j]=scale*np.sqrt(np.sum((tensor/(scale or 1.))**2))
        if not np.isfinite(values).all():raise ValueError('Nonfinite physical strain-family target')
        return dict(amplitudes=values,native_amplitudes=native,known=np.ones(values.shape,bool),
            families=[v['id'] for v in self.families],units='dimensionless',normalization=STRAIN_NORMALIZATION,
            experimental_mode_presence_qualified=False,experimental_mode_absence_qualified=False,
            candidate_pruning_authorized=False)

    def project_native(self,parameters,parent_a,*,baseline_a,cell_mapping,mode_dimensions):
        """Project the exact reciprocal-metric coordinates used by GSAS-II."""
        from ..structural_live.optimization.physical_prior_geometry import native_reciprocal_matrix
        from ..structural_live.optimization.chemical_geometry import direct_metric_from_reciprocal
        points=np.asarray(parameters);parents=np.asarray(parent_a);a0=np.asarray(baseline_a)
        mapping=np.asarray(cell_mapping)
        if (type(mode_dimensions)is not int or mode_dimensions<0 or points.ndim!=2
                or not 1<=len(points)<=MAX_BATCH_ROWS or points.dtype.kind!='f'
                or parents.shape!=(len(points),6) or parents.dtype.kind!='f'
                or a0.shape!=(6,) or mapping.ndim!=2 or mapping.shape[0]!=6
                or not 1<=mapping.shape[1]<=6 or mapping.shape[1]+mode_dimensions!=points.shape[1]
                or not all(np.isfinite(v).all() for v in [points,parents,a0,mapping])
                or np.linalg.matrix_rank(mapping)!=mapping.shape[1]):
            raise ValueError('Complete finite native cell coordinates and their original map required')
        child_a=a0+points[:,mode_dimensions:]@mapping.T
        def direct(rows):
            return np.asarray([direct_metric_from_reciprocal(native_reciprocal_matrix(a)) for a in rows])
        return self.project(direct(child_a),parent_metrics=direct(parents))


def from_metadata(value,*,parent_sha256=None,candidate=None):
    """Validate captured data without reopening or executing an uploaded CIF."""
    d=fields(value,['schema','parent_sha256','child_sha256','candidate','parent_cell','basis','matrix',
        'normalizations','native_labels','opaque_native_axes','normalization','observations_included',
        'scientific_result','candidate_pruning_authorized'])
    labels=d['native_labels'];checked=validate_candidate(d['candidate'])
    if (d['schema']!=SCHEMA or d['normalization']!=STRAIN_NORMALIZATION
            or not checksum(d['parent_sha256']) or not checksum(d['child_sha256'])
            or checked!=d['candidate'] or checked['child'] is None or checked['magnetic_space_group'] is not None
            or checked['child']['sha256']!=d['child_sha256']
            or parent_sha256 is not None and parent_sha256!=d['parent_sha256']
            or candidate is not None and checked!=validate_candidate(candidate)
            or any(d[k]is not False for k in ['observations_included','scientific_result','candidate_pruning_authorized'])
            or not isinstance(labels,list) or not 1<=len(labels)<=6
            or any(not isinstance(v,str) or not 1<=len(v)<=1000 or any(ord(c)<32 for c in v) for v in labels)
            or len(set(labels))!=len(labels)):
        raise ValueError('Complete bound nuclear strain metadata required')
    projection=StrainProjection(cell_metric(d['parent_cell']),d['basis'],d['matrix'],d['normalizations'])
    if (projection.dimensions!=len(labels)
            or not np.allclose(projection.basis,checked['transformation_matrix'],rtol=1e-14,atol=1e-14)):
        raise ValueError('Captured strain basis or dimensions changed')
    families={};opaque=[]
    for i,label in enumerate(labels):
        parsed=family_label(label,parent_sha256=d['parent_sha256'],quantity='lattice_strain')
        if parsed is None:opaque.append(dict(native_index=i,label=label));continue
        entry={k:parsed[k] for k in ['id','parent_sha256','parent_convention','k','irrep','quantity','units','normalization']}
        row=families.setdefault(entry['id'],dict(entry=entry,indices=[]));row['indices'].append(i)
    if d['opaque_native_axes']!=opaque:raise ValueError('Captured strain inventory lost an unresolved coordinate')
    ordered=[families[k] for k in sorted(families)]
    return StrainFamilies(projection,d,tuple(v['entry'] for v in ordered),tuple(tuple(v['indices']) for v in ordered))


def read(path,*,parent_sha256,candidate):
    """Read the exact child; absence of strain metadata stays unavailable.

    Malformed or partial strain metadata fails rather than silently becoming
    zero targets. Sparse zeros are admitted only inside the declared complete
    six-row engineering-strain matrix. Candidate origins are checked modulo
    integer parent translations, which do not change a uniform strain tensor.
    """
    import CifFile
    if not checksum(parent_sha256):raise ValueError('Exact parent identity required')
    candidate=validate_candidate(candidate);raw=read_bounded(path,MAX_CIF_BYTES)
    child_sha256=hashlib.sha256(raw).hexdigest()
    if (candidate['child'] is None or candidate['magnetic_space_group'] is not None
            or candidate['child']['sha256']!=child_sha256):
        raise ValueError('Strain map differs from the exact nuclear child')
    doc=CifFile.ReadCif(io.StringIO(raw.decode('utf-8',errors='strict')))
    if len(doc.keys())!=1:raise ValueError('One complete ISODISTORT child block required')
    block=doc[list(doc.keys())[0]]
    count=_integer(block.get('_iso_strainmode_number','0'))
    if not count:
        if any(str(k).startswith('_iso_strain')
               and k!='_iso_strainmode_number' for k in block.keys()):
            raise ValueError('Partial strain metadata cannot become absent mode targets')
        return None
    if count>6:raise ValueError('At most six independent uniform strain coordinates are supported')
    labels=list(map(str,_indexed(block,'_iso_strainmode_ID','_iso_strainmode_label',count)))
    if len(set(labels))!=count:raise ValueError('Duplicate native strain labels')
    norms=np.array(list(map(float,_indexed(block,'_iso_strainmodenorm_ID','_iso_strainmodenorm_value',count))))
    axes=_indexed(block,'_iso_strain_ID','_iso_strain_label',6)
    if axes!=['E_'+str(i) for i in range(1,7)]:raise ValueError('The complete native engineering-strain axes are required')
    matrix=np.zeros((6,count));seen=set()
    rows=_values(block,'_iso_strainmodematrix_row');cols=_values(block,'_iso_strainmodematrix_col')
    values=_values(block,'_iso_strainmodematrix_value')
    if not len(rows)==len(cols)==len(values):raise ValueError('Incomplete sparse strain-mode matrix')
    for r,c,value in zip(rows,cols,values):
        r=_integer(r)-1;c=_integer(c)-1
        if not 0<=r<6 or not 0<=c<count or (r,c) in seen:raise ValueError('Invalid or duplicate sparse strain-matrix coordinate')
        seen.add((r,c));matrix[r,c]=float(value)
    basis,origin=_transform(block.get('_iso_parent-to-child.transform_Pp_abc'))
    delta=origin-np.asarray(candidate['origin_shift'])
    if (not np.allclose(basis,candidate['transformation_matrix'],rtol=1e-14,atol=1e-14)
            or not np.allclose(delta,np.rint(delta),rtol=0,atol=1e-12)):
        raise ValueError('ISO strain transform differs from the reviewed child embedding')
    parent_cell=[float(block['_iso_parentcell_'+k]) for k in CELL_KEYS]
    opaque=[]
    for i,label in enumerate(labels):
        parsed=family_label(label,parent_sha256=parent_sha256,quantity='lattice_strain')
        if parsed is None:opaque.append(dict(native_index=i,label=label));continue
    metadata=dict(schema=SCHEMA,parent_sha256=parent_sha256,child_sha256=child_sha256,candidate=candidate,
        parent_cell=parent_cell,basis=basis.tolist(),matrix=matrix.tolist(),normalizations=norms.tolist(),
        native_labels=labels,opaque_native_axes=opaque,normalization=STRAIN_NORMALIZATION,
        observations_included=False,scientific_result=False,candidate_pruning_authorized=False)
    return from_metadata(metadata)
