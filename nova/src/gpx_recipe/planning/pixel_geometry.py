"""Nominal POWGEN rectangular-pixel geometry and angular surface measure.

The IDF reader supports the explicit Cartesian hierarchy, intrinsic nested
axis rotations and rectangular detector form used by the archived POWGEN IDF.
Other placement constructs fail explicitly. It does not read logs, apply a
survey, determine scintillator efficiency/depth or certify real hardware.
Coordinates passed to the count model are cm, +x along beam and z vertical.
"""
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET
import hashlib,math
import numpy as np

VERSION='powgen-nominal-rectangular-pixel-surface/1'
IDF_NAMESPACE='http://www.mantidproject.org/IDF/1.0'
NATIVE_TO_PLANNING=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])

def _finite(value):
    x=float(value)
    if not math.isfinite(x):raise ValueError('Finite IDF geometry values are required')
    return x

def _rotation(element,angle_attribute,depth=0):
    if depth>32:raise ValueError('IDF rotation nesting exceeds the supported bound')
    allowed={angle_attribute,'axis-x','axis-y','axis-z'}
    if angle_attribute=='rot':allowed|={'x','y','z','name'}
    if set(element.attrib)-allowed:raise ValueError('Unsupported IDF location/rotation attributes')
    angle=np.deg2rad(_finite(element.get(angle_attribute,'0')))
    axis=np.array([_finite(element.get('axis-'+k,'1' if k=='z' else '0')) for k in 'xyz'])
    norm=np.linalg.norm(axis)
    if norm==0:raise ValueError('An IDF rotation axis cannot vanish')
    axis/=norm;x,y,z=axis;cross=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    result=np.eye(3)*np.cos(angle)+(1-np.cos(angle))*np.outer(axis,axis)+np.sin(angle)*cross
    children=list(element)
    if children:
        if len(children)!=1 or children[0].tag!='{'+IDF_NAMESPACE+'}rot':
            raise ValueError('Only nested intrinsic rotations are supported in IDF locations')
        result=result@_rotation(children[0],'val',depth+1)
    return result

def _triangle_angle(a,b,c):
    first=np.linalg.norm(a,axis=-1);second=np.linalg.norm(b,axis=-1);third=np.linalg.norm(c,axis=-1)
    numerator=np.einsum('...i,...i->...',a,np.cross(b,c))
    denominator=(first*second*third+np.einsum('...i,...i->...',a,b)*third
        +np.einsum('...i,...i->...',b,c)*first+np.einsum('...i,...i->...',c,a)*second)
    return 2*np.arctan2(numerator,denominator)

def rectangle_solid_angle(center_cm,u_cm,v_cm,observer_cm=(0.,0.,0.)):
    """Exact geometric solid angle of a rectangle, via two spherical triangles.

    No far-field approximation, mesh refinement, detector response or fitted
    normalization is used. The observer must be off the collecting plane.
    """
    center,u,v,observer=np.broadcast_arrays(*[np.asarray(x,float) for x in (center_cm,u_cm,v_cm,observer_cm)])
    if (center.ndim<1 or center.shape[-1]!=3 or center.size>3*2**20
            or any(not np.isfinite(a).all() for a in (center,u,v,observer))):
        raise ValueError('Bounded finite rectangle/observer coordinates are required')
    lu=np.linalg.norm(u,axis=-1);lv=np.linalg.norm(v,axis=-1);normal=np.cross(u,v)
    if np.any(lu<=0) or np.any(lv<=0) or np.any(abs(np.sum(u*v,axis=-1))>1e-12*lu*lv):
        raise ValueError('A rectangle requires two nonzero orthogonal edges')
    if np.any(abs(np.sum((center-observer)*normal,axis=-1))<=np.finfo(float).tiny):
        raise ValueError('Solid angle is undefined for a coplanar observer')
    c=center-observer;a=c-u/2-v/2;b=c+u/2-v/2;d=c+u/2+v/2;e=c-u/2+v/2
    return abs(_triangle_angle(a,b,d)+_triangle_angle(a,d,e))

@dataclass(frozen=True)
class PixelFace:
    center_cm: np.ndarray
    u_cm: np.ndarray
    v_cm: np.ndarray
    calibration_position_cm: np.ndarray | None = None

    def __post_init__(self):
        arrays=[np.array(x,float,copy=True) for x in (self.center_cm,self.u_cm,self.v_cm)]
        if any(x.shape!=(3,) or not np.isfinite(x).all() for x in arrays):
            raise ValueError('A pixel needs explicit finite center and edge vectors')
        center,u,v=arrays;lu=np.linalg.norm(u);lv=np.linalg.norm(v)
        if min(lu,lv)<=0 or abs(u@v)>1e-12*lu*lv:raise ValueError('Pixel edges must be nonzero and orthogonal')
        nominal=center.copy() if self.calibration_position_cm is None else np.array(self.calibration_position_cm,float,copy=True)
        if nominal.shape!=(3,) or not np.isfinite(nominal).all():raise ValueError('A finite fixed pixel calibration position is required')
        for key,value in zip(('center_cm','u_cm','v_cm','calibration_position_cm'),[*arrays,nominal]):
            value.setflags(write=False);object.__setattr__(self,key,value)

    def targets(self,unit_face_points):
        unit=np.asarray(unit_face_points,float)
        if (unit.ndim!=2 or unit.shape[1]!=2 or not 1<=len(unit)<=2**20
                or not np.isfinite(unit).all() or np.any(unit<0) or np.any(unit>1)):
            raise ValueError('Bounded two-dimensional unit-face coordinates are required')
        return self.center_cm+(unit[:,0,None]-.5)*self.u_cm+(unit[:,1,None]-.5)*self.v_cm

    def angular_density(self,scattering_points_cm,unit_face_points):
        """Return targets and dOmega/(du dv) for the same sampled rays.

        This is a geometric collecting face seen from either side. Real
        front/back efficiency, shielding and conversion-depth response are
        separate supplied physics. Pointwise density can exceed4pi close to
        a face; its full geometric integral is at most2pi, off the plane.
        """
        targets=self.targets(unit_face_points);points=np.asarray(scattering_points_cm,float)
        if points.shape!=targets.shape or not np.isfinite(points).all():
            raise ValueError('Every surface target needs its paired scattering position')
        rays=targets-points;distance=np.linalg.norm(rays,axis=1)
        if np.any(distance<=0):raise ValueError('Scattering positions cannot coincide with detector targets')
        area_normal=np.cross(self.u_cm,self.v_cm)
        omega=abs(rays@area_normal)/distance**3
        return targets,omega

    def solid_angle(self,observer_cm=(0.,0.,0.)):
        return float(rectangle_solid_angle(self.center_cm,self.u_cm,self.v_cm,observer_cm))

@dataclass(frozen=True)
class PixelGeometry:
    detector_ids: np.ndarray
    calibration_positions_cm: np.ndarray
    face_centers_cm: np.ndarray
    u_cm: np.ndarray
    v_cm: np.ndarray
    depth_cm: np.ndarray
    bank_names: np.ndarray
    metadata: dict

    def face(self,index):
        return PixelFace(self.face_centers_cm[index],self.u_cm[index],self.v_cm[index],self.calibration_positions_cm[index])

def powgen_idf_pixels(path):
    """Read the supported POWGEN IDF subset without Mantid or facility access.

    All detector identities, transforms, face dimensions and placement come
    from this file. Validate against the experiment's actual loaded detector
    geometry/calibration before use. Unsupported positioning, cyclic types,
    noncanonical source/sample axes and duplicate IDs fail explicitly.
    """
    source=Path(path);data=source.read_bytes()
    if len(data)>2*1024**2 or b'<!DOCTYPE' in data or b'<!ENTITY' in data:raise ValueError('A bounded entity-free instrument XML is required')
    try:root=ET.fromstring(data)
    except ET.ParseError as exc:raise ValueError('Invalid instrument XML') from exc
    q=lambda tag:'{'+IDF_NAMESPACE+'}'+tag
    if root.tag!=q('instrument') or root.get('name') not in ('PG3','POWGEN'):raise ValueError('A POWGEN instrument definition is required')
    defaults=root.find(q('defaults'))
    if defaults is None:raise ValueError('Explicit IDF defaults are required')
    length=defaults.find(q('length'));angle=defaults.find(q('angle'));frame=defaults.find(q('reference-frame'))
    if (length is None or length.get('unit') not in ('metre','meter') or angle is None or angle.get('unit')!='degree'
            or frame is None or any(defaults.find(q(x)) is not None for x in ['offsets','components-are-facing'])):
        raise ValueError('Only explicit metre/degree Cartesian IDF conventions are supported')
    expected={'along-beam':('axis','z'),'pointing-up':('axis','y'),'handedness':('val','right'),'theta-sign':('axis','x')}
    if any(frame.find(q(tag)) is None or frame.find(q(tag)).get(key)!=value for tag,(key,value) in expected.items()):
        raise ValueError('The canonical POWGEN native coordinate frame is required')
    types={}
    for item in root.findall(q('type')):
        name=item.get('name')
        if not name or name in types:raise ValueError('Unique explicit IDF type names are required')
        types[name]=item
    parts=[];sample=[];moderator=[]
    def walk(component,position,rotation,ancestry):
        name=component.get('type')
        if name not in types or name in ancestry or len(ancestry)>32:raise ValueError('Unknown, cyclic or excessively nested instrument type')
        kind=types[name];role=kind.get('is','').lower();locations=component.findall(q('location'))
        if not locations:raise ValueError('Explicit component locations are required')
        if any(child.tag!=q('location') for child in component):raise ValueError('Unsupported component placement/parameter children')
        if component.find(q('locations')) is not None:raise ValueError('Unsupported IDF placement shorthand')
        for location in locations:
            local=np.array([_finite(location.get(k,'0')) for k in 'xyz']);rot=_rotation(location,'rot')
            center=position+rotation@local;orient=rotation@rot
            if role=='samplepos':sample.append(center);continue
            if role=='source':moderator.append(center);continue
            if role=='monitor':continue
            if role in ('rectangular_detector','rectangulardetector'):
                if len(locations)!=1 or component.get('idfillbyfirst','y') not in ('x','y'):raise ValueError('Unsupported rectangular detector identity mapping')
                nx,ny=int(kind.get('xpixels','0')),int(kind.get('ypixels','0'))
                if min(nx,ny)<=0 or nx*ny+sum(len(p[0]) for p in parts)>2**17:raise ValueError('Bounded rectangular detector dimensions are required')
                if set(kind.attrib)-{'name','is','type','xpixels','xstart','xstep','ypixels','ystart','ystep'}:
                    raise ValueError('Unsupported rectangular detector placement attributes')
                if any(child.tag!=q('properties') or len(child) or child.attrib for child in kind):
                    raise ValueError('Unsupported rectangular detector parameters')
                pixel=types.get(kind.get('type'));box=None if pixel is None else pixel.find(q('cuboid'))
                if pixel is None or pixel.get('is','').lower()!='detector' or box is None:raise ValueError('Rectangular pixels require explicit detector cuboids')
                def point(tag):
                    p=box.find(q(tag))
                    if p is None:raise ValueError('Missing pixel cuboid vertex')
                    return np.array([_finite(p.get(k,'0')) for k in 'xyz'])
                origin=point('left-front-bottom-point');u=point('right-front-bottom-point')-origin;v=point('left-front-top-point')-origin;w=point('left-back-bottom-point')-origin
                lengths=np.linalg.norm([u,v,w],axis=1)
                if np.any(lengths<=0) or np.max(abs(np.array([u,v,w])@np.array([u,v,w]).T-np.diag(lengths**2)))>1e-12*np.max(lengths)**2:
                    raise ValueError('Pixel cuboids must have three nonzero orthogonal edges')
                ix,iy=np.meshgrid(np.arange(nx),np.arange(ny),indexing='ij');ix=ix.ravel();iy=iy.ravel()
                try:first=int(component.get('idstart'));stride=int(component.get('idstepbyrow'));step=int(component.get('idstep','1'))
                except (TypeError,ValueError) as exc:raise ValueError('Explicit integer detector ID mapping is required') from exc
                if min(first,stride,step)<0 or stride==0 or step==0:raise ValueError('Positive detector identity steps are required')
                ids=first+(ix*stride+iy*step if component.get('idfillbyfirst','y')=='y' else iy*stride+ix*step)
                local_points=np.column_stack((_finite(kind.get('xstart'))+ix*_finite(kind.get('xstep')),
                    _finite(kind.get('ystart'))+iy*_finite(kind.get('ystep')),np.zeros(nx*ny)))
                nominal=center+local_points@orient.T;face=nominal+orient@(origin+(u+v)/2)
                bank=location.get('name',name)
                if len(bank)>64:raise ValueError('Bounded bank names are required')
                parts.append((ids,nominal,face,np.tile(orient@u,(len(ids),1)),np.tile(orient@v,(len(ids),1)),np.tile(orient@w,(len(ids),1)),np.full(len(ids),bank)))
                continue
            if role=='detector':raise ValueError('Standalone detector placements are not supported')
            children=kind.findall(q('component'))
            for child in children:walk(child,center,orient,ancestry+(name,))
    for component in root.findall(q('component')):walk(component,np.zeros(3),np.eye(3),())
    if (not parts or len(sample)!=1 or len(moderator)!=1 or np.linalg.norm(sample[0])>1e-12
            or abs(moderator[0][0])+abs(moderator[0][1])>1e-12 or moderator[0][2]>=0):
        raise ValueError('A canonical single source/sample and explicit rectangular detectors are required')
    arrays=[np.concatenate([p[k] for p in parts]) for k in range(7)];order=np.argsort(arrays[0]);arrays=[a[order] for a in arrays]
    if len(arrays[0])>2**20 or len(np.unique(arrays[0]))!=len(arrays[0]):raise ValueError('Detector identities must be unique and bounded')
    for k in range(1,6):arrays[k]=100*arrays[k]@NATIVE_TO_PLANNING.T
    if any(not np.isfinite(a).all() or abs(a).max(initial=0)>1e8 for a in arrays[1:6]):raise ValueError('Instrument coordinates exceed the finite geometry bound')
    for a in arrays:a.setflags(write=False)
    return PixelGeometry(*arrays,dict(version=VERSION,idf_sha256=hashlib.sha256(data).hexdigest(),idf_name=root.get('name'),
        valid_from=root.get('valid-from'),last_modified=root.get('last-modified'),panels=len(parts),detectors=len(arrays[0]),
        nominal_source_to_sample_m=float(-moderator[0][2]),axis_native_to_planning=NATIVE_TO_PLANNING.tolist(),
        geometry_source='Nominal embedded IDF; no log/survey/efficiency/depth inference',experimental_accuracy_certified=False))
