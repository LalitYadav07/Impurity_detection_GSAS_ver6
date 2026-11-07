"""Absolute-cell selection of data-only native kernels, never saved curves.

The metric table locates optional physics. It is not native authorization:
imported kernels still require full original-kernel verification and every
actual compiled input must match at use. All fitted coordinates remain free.
"""
from __future__ import annotations

from collections import Counter
import copy
import os
import time

import numpy as np

from ..symmetry.contracts import digest
from .tof_kernel_arrays import checksum,MAX_BYTES
from .identity import ACQUISITION_PARAMETERS

SCHEMA='radar-tof-kernel-geometry/v1'
MATCH_RTOL=1e-10  # Locator only; actual reuse requires exact compiled bits.
# Recorded conditions do not identify a raw TOF peak function. The current
# native cell/profile arguments still decide every individual primitive hit;
# atomic intensities, corrections and observations are always computed afresh.


class GeometryUnavailable(ValueError):
    """Use fresh native calculations without narrowing the scientific search."""


def validate_geometry(value,count):
    if (type(count)is not int or not 1<=count<=512 or not isinstance(value,dict)
            or set(value)!={'schema','setting_sha256','profile_sha256','runtime','cells'}
            or value['schema']!=SCHEMA
            or any(not checksum(value[k]) for k in ('setting_sha256','profile_sha256'))
            or not isinstance(value['runtime'],dict)
            or set(value['runtime'])!={'kernel_sha256','binary_sha256'}
            or any(not checksum(v) for v in value['runtime'].values())
            or not isinstance(value['cells'],list) or len(value['cells'])!=count
            or any(not isinstance(row,list) or len(row)!=6
                or any(type(v) not in (int,float) or not np.isfinite(v) or abs(v)>1e6 for v in row)
                for row in value['cells'])):
        raise ValueError('Declare bounded absolute native cells and their kernel context')
    cells=np.asarray(value['cells'],float)
    if len({row.astype('<f8').tobytes() for row in cells})!=len(cells):
        raise ValueError('Kernel geometry cells must be unique native metrics')
    for a in cells:
        metric=np.array([[a[0],a[3]/2,a[4]/2],[a[3]/2,a[1],a[5]/2],[a[4]/2,a[5]/2,a[2]]])
        if np.linalg.eigvalsh(metric).min()<=0:
            raise ValueError('Kernel geometry needs positive reciprocal cell metrics')
    return value


def _setting(source):
    from .identity import _plain
    n=source.fast.native;n._check()
    phase=n.histo_phases[1][source.fast.slow.phase.name]
    return digest(_plain(dict(symmetry=phase['General']['SGData'],
        transformation=source.fast.meta['transformation_matrix'],origin=source.fast.meta['origin_shift'])))


def _profile(source):
    from .identity import _plain
    n=source.fast.native;n._check();phase=n.histo_phases[1][source.fast.slow.phase.name]
    histogram=n.histo_phases[0][source.fast.slow.hist.name]
    hfx=f":{histogram['hId']}:";phfx=f"{phase['pId']}:{histogram['hId']}:"
    parameters={}
    for prefix,role in [(hfx,'bank'),(phfx,'phase')]:
        for key,value in n.parms.items():
            if key.startswith(prefix):
                name=key[len(prefix):]
                if (name not in {'Scale','wtFactor'} and not name.startswith('Back;')
                        and not (role=='bank' and name in ACQUISITION_PARAMETERS)):
                    parameters[role+'/'+name]=value
    controls={name:n.controls.get(prefix+name) for prefix,name in
        [(hfx,'histType'),(hfx,'instType'),(phfx,'SizeType'),(phfx,'MustrainType')]}
    # The actual axis/range is checked per native peak. Almost-identical
    # theoretical and reduced grids need not have identical float64 bytes.
    return digest(_plain(dict(parameters=parameters,controls=controls,
        lookup=n.histdict.get(hfx+'pdabc'))))


def cell_for_point(source,point):
    """Native canonical cell in the current basis, with no atomic trial."""
    p=source.check_domain(np.asarray(point,float));n=source.fast.native;n._check()
    theta=np.r_[source.lift@p,0.,0.];theta[:source.fast.nm]=0.
    _,values=source.fast._peak_geometry_values(theta)
    parameters=n.parms.copy();parameters.update(zip(n.vary,values));n.maps.Dict2Map(parameters)
    prefix=f'{source.fast.slow.phase.id}::A'
    a=np.asarray([parameters[prefix+str(i)] for i in range(6)],float)
    if not np.isfinite(a).all():raise GeometryUnavailable('Nonfinite current native metric')
    return a


def point_for_cell(source,cell,*,mode_point=None):
    """Map a stored cell into current coordinates; never clip or expand bounds.

    This proposes a coordinate only. The actual forward/optimizer retains
    its complete chemical feasibility check, all displacement modes and bounds.
    """
    a=np.asarray(cell,float);m=source.mode_dimensions;d=len(source.radii)
    if a.shape!=(6,) or not np.isfinite(a).all():
        raise GeometryUnavailable('A complete finite native reciprocal metric is required')
    point=np.zeros(d) if mode_point is None else np.asarray(mode_point,float).copy()
    source.check_domain(point)
    if np.any(source.lift[m:,:m]):
        raise GeometryUnavailable('Unsupported mixed displacement/cell coordinate lift')
    matrix=source.fast.slow.a_scale*(source.fast.basis['cell_basis']@source.lift[m:,m:])
    delta=a-source.fast.basis['baseline_a']
    cells=np.linalg.lstsq(matrix,delta,rcond=1e-12)[0]
    if np.linalg.norm(matrix@cells-delta)>MATCH_RTOL*max(np.linalg.norm(a),1e-30):
        raise GeometryUnavailable('Stored cell is outside the current native metric subspace')
    point[m:]=cells
    try:source.check_domain(point)
    except ValueError as error:
        raise GeometryUnavailable('Stored cell is outside the current reviewed bounds; use its original native proposal') from error
    if np.linalg.norm(cell_for_point(source,point)-a)>MATCH_RTOL*max(np.linalg.norm(a),1e-30):
        raise GeometryUnavailable('Stored cell cannot be represented in the current native convention')
    return point


def specification(source,points,*,model_sha256,bank):
    """Deduplicate instrument contexts, not scientific models or starts."""
    from .native_tof_kernels import context
    _,_,_,_,runtime=context(source)
    if (not isinstance(points,(list,tuple,np.ndarray))
            or isinstance(points,np.ndarray) and points.ndim!=2
            or not 1<=len(points)<=5000):
        raise ValueError('Declare a bounded kernel proposal portfolio')
    cells=[];indices={};representatives=[];mapping=[]
    for point in points:
        a=cell_for_point(source,point);key=a.astype('<f8').tobytes()
        if key not in indices:
            if len(cells)>=512:raise ValueError('Split kernel preparation into bounded cell stores')
            indices[key]=len(cells);cells.append(a.tolist());representatives.append(np.asarray(point,float).copy())
        mapping.append(indices[key])
    geometry=dict(schema=SCHEMA,setting_sha256=_setting(source),profile_sha256=_profile(source),
        runtime=runtime,cells=cells)
    validate_geometry(geometry,len(cells))
    return dict(model_sha256=model_sha256,bank=bank,cell_count=len(cells),geometry=geometry),representatives,mapping


def prepare(source,points,root,*,model_sha256,bank,check_cancel=None):
    """Resume complete native kernel preparation into an ordinary KernelStore."""
    from .kernel_store import KernelStore
    from .native_tof_kernels import capture,verify
    if check_cancel is not None:check_cancel()
    spec,representatives,mapping=specification(source,points,model_sha256=model_sha256,bank=bank)
    store=KernelStore.open_or_create(root,spec);prepared=[];requalified=[]
    for index,point in enumerate(representatives):
        if check_cancel is not None:check_cancel()
        if str(index) in store._read()['chunks']:
            verify(store.get(index),source,check_cancel=check_cancel);requalified.append(index)
        else:
            store.put(index,capture(source,point,check_cancel=check_cancel).data);prepared.append(index)
        if _setting(source)!=spec['geometry']['setting_sha256'] or _profile(source)!=spec['geometry']['profile_sha256']:
            raise ValueError('Native geometry context changed during preparation')
    return store,dict(proposal_to_cell=mapping,prepared_cells=prepared,requalified_cells=requalified,
        observations_included=False,scientific_result=False,coverage=store.coverage())


class PreparedKernelBank:
    """One bounded owned reader, automatically requalified in a fresh worker."""
    def __init__(self,store,source,*,maximum_bytes=MAX_BYTES,check_cancel=None):
        from .kernel_store import KernelStore
        from .native_tof_kernels import context
        if type(maximum_bytes)is not int or not 0<maximum_bytes<=MAX_BYTES:
            raise ValueError('A bounded prepared-kernel resident budget is required')
        self.store=store if isinstance(store,KernelStore) else KernelStore(store)
        self.source=source;self.maximum_bytes=maximum_bytes;self.check_cancel=check_cancel;self.pid=os.getpid()
        spec=self.store.document['specification'];geometry=spec.get('geometry')
        if geometry is None:raise GeometryUnavailable('Legacy kernel chunks have no absolute-cell mapping')
        validate_geometry(geometry,spec['cell_count']);self.store_id=self.store.document['id']
        self.geometry=copy.deepcopy(geometry)
        cells=np.asarray(geometry['cells'],dtype='<f8')
        self.cells=np.frombuffer(cells.tobytes(),dtype=cells.dtype).reshape(cells.shape)
        _,_,_,_,runtime=context(source);self.runtime=runtime
        self.cached=None;self.requests=self.hits=self.loads=self.native_peak_checks=0
        self.archived_peak_hits=self.archived_peak_misses=0
        self.resident_bytes=0;self.seconds=0.;self.bypasses=Counter()

    def _drop(self):
        if self.cached is not None:
            self.archived_peak_hits+=self.cached[2].hits
            self.archived_peak_misses+=self.cached[2].misses
        self.cached=None;self.resident_bytes=0

    def resolve(self,point):
        """Find compatible optional data; all actual peak inputs remain checked."""
        from .native_tof_kernels import verify
        began=time.monotonic();self.requests+=1
        try:
            if self.check_cancel is not None:self.check_cancel()
            self.source.fast.native._check()
            if os.getpid()!=self.pid:
                self._drop();raise GeometryUnavailable('Prepared-kernel ownership cannot cross a fork')
            if self.geometry['runtime']!=self.runtime or _setting(self.source)!=self.geometry['setting_sha256']:
                raise GeometryUnavailable('Prepared kernel setting or native runtime differs')
            if _profile(self.source)!=self.geometry['profile_sha256']:
                raise GeometryUnavailable('Current instrument/profile needs original native kernels')
            a=cell_for_point(self.source,point)
            distances=np.linalg.norm(self.cells-a,axis=1)/max(np.linalg.norm(a),1e-30)
            index=int(np.argmin(distances))
            if distances[index]>MATCH_RTOL:raise GeometryUnavailable('No prepared absolute cell matches this native point')
            document=self.store._read()
            if document['id']!=self.store_id:
                self._drop();raise GeometryUnavailable('Prepared kernel specification changed')
            descriptor=document['chunks'].get(str(index))
            if descriptor is None:raise GeometryUnavailable('The matching kernel cell has not been prepared')
            if self.cached is not None and self.cached[:2]==(index,descriptor):
                self.hits+=1;return self.cached[2]
            self._drop()  # No two expanded chunks coexist in the resident cache.
            data=self.store.get(index)
            accounted=data.numeric_bytes+1024*data.metadata['reflection_count']
            if accounted>self.maximum_bytes:raise GeometryUnavailable('Prepared kernel exceeds this resident budget')
            kernel=verify(data,self.source,check_cancel=self.check_cancel)
            self.loads+=1;self.native_peak_checks+=kernel.diagnostics()['native_peak_calls']
            self.resident_bytes=accounted;self.cached=(index,descriptor,kernel)
            return kernel
        except InterruptedError:
            raise
        except GeometryUnavailable as error:
            # An unprepared cell or a temporarily changed profile must not
            # evict an unrelated, still-qualified resident cell. Actual-use
            # context and compiled-input checks remain required on return.
            self.bypasses[str(error)[:300]]+=1;return None
        except (OSError,ValueError) as error:
            self._drop();self.bypasses[str(error)[:300]]+=1;return None
        finally:self.seconds+=time.monotonic()-began

    def diagnostics(self):
        hits=self.archived_peak_hits+(self.cached[2].hits if self.cached is not None else 0)
        misses=self.archived_peak_misses+(self.cached[2].misses if self.cached is not None else 0)
        return dict(requests=self.requests,resident_hits=self.hits,loads=self.loads,
            primitive_hits=hits,primitive_misses=misses,
            native_peak_checks=self.native_peak_checks,resident_bytes=self.resident_bytes,
            maximum_resident_bytes=self.maximum_bytes,bypasses=dict(self.bypasses),seconds=self.seconds,
            absolute_cell_lookup_only=True,scientific_candidates_removed=False,
            native_qualification_persisted=False,observations_reused=False)
