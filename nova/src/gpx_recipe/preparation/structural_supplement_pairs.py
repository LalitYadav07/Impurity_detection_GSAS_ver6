"""Explicit sparse signed-coordinate additions to a native paired cohort.

A supplement is not complete named-mode coverage and cannot choose new folds.
Every ray belongs to a retained original block, parent cell and mode basis.
Checksums bind data; they do not certify native provenance or mode absence.
"""
import copy
from dataclasses import dataclass

import numpy as np

from ..symmetry.contracts import digest,fields,validate_candidate
from .tof_kernel_arrays import checksum

RAW_SCHEMA='radar-structural-ranking-additive-paired/v1'
PACKED_SCHEMA='radar-structural-ranking-additive-pairs/v1'
SCHEMAS={RAW_SCHEMA,PACKED_SCHEMA}
SCHEMA='radar-native-sparse-signed-mode-additions/v1'
NAMED_BASIS='named-native-axes/v1'
WHITENED_BASIS='physical-gram-eigenaxes/v1'
SOURCE_KEYS=('pairs_sha256','manifest_identity','projection_identity')
MAX_CONTEXT_BYTES=64*1024**2
OMITTED_SUMMARIES={'displacement_rms','relative_metric_norm'}


def present(document):
    return isinstance(document,dict) and document.get('schema') in SCHEMAS


def record(document):
    value=fields(document.get('supplement'),['schema','sources','rays','direction_basis','native_protocol_sha256',
        'native_curves_sha256','native_result_sha256','producer_implementation_sha256'])
    sources=value['sources'];rays=value['rays'];count=document.get('trajectories')
    if (value['schema']!=SCHEMA or not isinstance(value['direction_basis'],str)
            or value['direction_basis'] not in {NAMED_BASIS,WHITENED_BASIS}
            or not isinstance(sources,list) or not 1<=len(sources)<=256
            or not isinstance(rays,list) or not 1<=len(rays)<=4096
            or type(count)is not int or not 1<=count<=4096
            or any(not checksum(value[k]) for k in ['native_protocol_sha256','native_curves_sha256','native_result_sha256','producer_implementation_sha256'])
            or 'mode_coverage' in document or 'trajectory_slice' in document):
        raise ValueError('Sparse additions need their explicit original sources and complete declared rays')
    keys=[]
    for source in sources:
        source=fields(source,SOURCE_KEYS)
        if any(not checksum(source[k]) for k in SOURCE_KEYS):raise ValueError('Every added ray needs exact original paired and coordinate identities')
        keys.append(tuple(source[k] for k in SOURCE_KEYS))
    if len(set(keys))!=len(keys):raise ValueError('Sparse additions cannot repeat a source binding')
    seen=set();used=set();rows=1
    for ray in rays:
        r=fields(ray,['source','parent','axis','sign','amplitudes'])
        amplitudes=r['amplitudes']
        if (type(r['source'])is not int or not 0<=r['source']<len(sources)
                or type(r['parent'])is not int or not 0<=r['parent']<count
                or type(r['axis'])is not int or not 0<=r['axis']<2048
                or type(r['sign'])is not int or r['sign'] not in (-1,1)
                or not isinstance(amplitudes,list) or not 1<=len(amplitudes)<=32
                or any(type(v) not in (int,float) or not np.isfinite(v) or v<=0 for v in amplitudes)
                or any(b<=a for a,b in zip(amplitudes,amplitudes[1:]))):
            raise ValueError('Retain every sealed signed ray, parent and positive amplitude step')
        key=(r['source'],r['parent'],r['axis'],r['sign'])
        if key in seen:raise ValueError('A sparse ray identity cannot be duplicated')
        seen.add(key);used.add(r['source']);rows+=1+len(amplitudes)
    if used!=set(range(len(sources))) or rows>32769:
        raise ValueError('Sparse additions must retain every bound source within the existing point limit')
    return value


def layout(document):
    d=record(document);trajectory=[-1];roles=[0]
    for ray in d['rays']:
        trajectory.extend([ray['parent']]*(1+len(ray['amplitudes'])))
        roles.extend([1]+[2]*len(ray['amplitudes']))
    return np.array(trajectory,np.int64),np.array(roles,np.int8)


def _relative_rows(actual,expected):
    scale=np.max(abs(expected),axis=1,initial=0.)
    error=np.max(abs(actual-expected),axis=1,initial=0.)
    return bool(np.all(error<=1e-10*np.maximum(scale,np.finfo(float).tiny)))


def validate_arrays(document,arrays,projection):
    """Use every physical coordinate, retaining fresh geometric SG labels."""
    from .structural_mode_directions import from_basis
    from .structural_mode_projection import descriptor
    d=record(document);m=document['mode_dimensions'];p=arrays['parameters']
    if p.dtype!=np.float64 or arrays['parent_a'].dtype!=np.float64:
        raise ValueError('Sparse native mode and parent-cell coordinates must retain float64 values')
    if projection is None or any(s['projection_identity']!=descriptor(projection)['identity'] for s in d['sources']):
        raise ValueError('Sparse rays require the exact original complete native coordinate map')
    a=projection.arrays
    if d['direction_basis']==NAMED_BASIS:
        directions=from_basis(a['physical_to_native'],a['gauge_basis'],a['physical_gram'],projection.metadata['native_mode_labels'])
        physical=directions.physical;gauge_only=directions.gauge_only
    else:
        from ..structural_live.optimization.physical_prior_geometry import orthonormal_lift
        physical=orthonormal_lift(a['physical_gram']).T
        # Preserve the producer's explicit physical-norm normalization. These
        # are mixed physical directions, not individual named native modes.
        norms=np.einsum('ni,ij,nj->n',physical,a['physical_gram'],physical)
        physical=physical/np.sqrt(norms)[:,None];gauge_only=np.zeros(len(physical),bool)
    if np.any(p[arrays['role']!=2,:m]!=0):
        raise ValueError('Signed-mode additions retain exact null displacement controls')
    expected=np.zeros(p[:,:m].shape,np.float64);cursor=1
    for ray in d['rays']:
        axis=ray['axis'];count=len(ray['amplitudes'])
        if axis>=len(physical) or gauge_only[axis]:
            raise ValueError('A demonstrated origin gauge cannot become a fresh physical hypothesis')
        expected[cursor+1:cursor+1+count]=np.asarray(ray['amplitudes'])[:,None]*ray['sign']*physical[axis]
        cells=np.broadcast_to(p[cursor,m:],p[cursor:cursor+1+count,m:].shape)
        if not _relative_rows(p[cursor:cursor+1+count,m:],cells):
            raise ValueError('An added mode ray changed its original parent-control cell coordinates')
        cursor+=1+count
    if len(p)!=cursor or not _relative_rows(p[:,:m],expected):
        raise ValueError('Added native-mode amplitudes differ from their full original physical basis')


@dataclass
class _Source:
    document:dict
    q:np.ndarray|None
    parents:dict|None
    controls:dict|None
    projection_identity:str|None


class Catalog:
    """Bind sparse additions to retained originals before any fold assignment."""
    def __init__(self,*,arrays=True):
        self.arrays=arrays;self.originals={};self.additions={};self.context_bytes=0;self.grids={}

    def add(self,document,*,name,arrays=None,projection=None):
        document=dict(document,candidate=validate_candidate(document['candidate']))
        if present(document):
            record(document)
            if name in self.additions:raise ValueError('Sparse addition logical names must be unique')
            self.additions[name]=(document,arrays if self.arrays else None)
            return
        if 'supplement' in document:raise ValueError('Sparse additions require their explicit paired schema')
        from .structural_mode_projection import descriptor
        identity=None if projection is None else descriptor(projection)['identity']
        if not self.arrays and document.get('mode_projection') is not None:
            # Metadata preflight cannot derive a projection identity from its
            # two file hashes. Full array admission checks it in the worker.
            identity=None
        key=(document['pairs_sha256'],digest(document))
        if key in self.originals:return
        q=parents=controls=None
        if self.arrays:
            if arrays is None:raise ValueError('Complete native arrays are required for context admission')
            raw=arrays['q'].tobytes();grid_key=(arrays['q'].dtype.str,raw)
            if grid_key not in self.grids:
                self.grids[grid_key]=arrays['q'];self.context_bytes+=2*len(raw)+512
            q=self.grids[grid_key];parents={};controls={};m=document['mode_dimensions']
            for i,(group,cell) in enumerate(zip(document['groups'],arrays['parent_a'])):
                if group not in parents:parents[group]=np.frombuffer(cell.tobytes(),dtype=cell.dtype)
                if arrays['role'][i]!=2:
                    values=arrays['parameters'][i,m:]
                    controls.setdefault(group,np.frombuffer(values.tobytes(),dtype=values.dtype))
            self.context_bytes+=sum(v.nbytes+256 for v in parents.values())
            self.context_bytes+=sum(v.nbytes+256 for v in controls.values())
            if self.context_bytes>MAX_CONTEXT_BYTES:raise ValueError('Original parent context index exceeds its bounded allocation')
        self.originals[key]=_Source(document,q,parents,controls,identity)

    def finish(self):
        result={}
        for name,(document,arrays) in self.additions.items():
            d=record(document);sources=[]
            for supplied in d['sources']:
                source=self.originals.get((supplied['pairs_sha256'],supplied['manifest_identity']))
                if source is None:raise ValueError('Sparse additions must travel with every exact original paired block')
                original=source.document
                keys=['source_sha256','instrument_sha256','child_sha256','candidate','seed','parent_seed',
                    'parent_relative_metric_range','trajectories','steps','mode_dimensions','cell_dimensions','declared_child_group','grid']
                if (any(original.get(k)!=document.get(k) for k in keys)
                        or original.get('mode_projection')!=document.get('mode_projection')
                        or self.arrays and source.projection_identity!=supplied['projection_identity']):
                    raise ValueError('Sparse additions changed their original parent, profile, sampling or mode coordinates')
                sources.append(source)
            if self.arrays:
                if arrays is None:raise ValueError('Sparse addition arrays are missing from context admission')
                if any(not np.array_equal(arrays['q'],source.q) for source in sources):
                    raise ValueError('Added curves changed their exact original coordinate grid')
                if not np.array_equal(arrays['parent_a'][0],sources[0].parents['parent:'+document['source_sha256']]):
                    raise ValueError('Sparse additions changed their original anchor cell')
            cursor=1;prefix=f'parent:{document["source_sha256"]}:{document["seed"]}:{document["parent_seed"]}:trajectory-'
            for ray in d['rays']:
                source=sources[ray['source']];group=prefix+str(ray['parent']);count=1+len(ray['amplitudes'])
                if group not in source.document['groups']:
                    raise ValueError('A sparse ray cannot invent a new original trajectory')
                if self.arrays and not np.array_equal(arrays['parent_a'][cursor:cursor+count],
                        np.broadcast_to(source.parents[group],arrays['parent_a'][cursor:cursor+count].shape)):
                    raise ValueError('Added hypotheses must keep their original parent cell and physical group')
                if self.arrays and (group not in source.controls or not _relative_rows(
                        arrays['parameters'][cursor:cursor+count,document['mode_dimensions']:],
                        np.broadcast_to(source.controls[group],arrays['parameters'][cursor:cursor+count,document['mode_dimensions']:].shape))):
                    raise ValueError('Added cell coordinates differ from their original native parent control')
                cursor+=count
            result[name]=dict(sources=copy.deepcopy(d['sources']),supplement_identity=digest(d))
        return result


def verify_context(documents,block_factory,*,subset_factory=None,names=None):
    """Read only bound originals and one supplement's arrays at a time.

    Legacy cohorts without additions allocate no new numerical context index.
    Metadata admission alone cannot verify the native coordinates or cells.
    """
    added=[i for i,d in enumerate(documents) if present(d)]
    if not added:return {}
    names=[str(i) for i in range(len(documents))] if names is None else names
    if len(names)!=len(documents):raise ValueError('Context names differ from the complete paired inventory')
    wanted={(s['pairs_sha256'],s['manifest_identity']) for i in added for s in record(documents[i])['sources']}
    originals=[i for i,d in enumerate(documents) if not present(d) and (d['pairs_sha256'],digest(d)) in wanted]
    def selected(indices):
        wanted_indices=set(indices)
        iterator=(subset_factory(indices) if subset_factory is not None else
                  (b for i,b in enumerate(block_factory()) if i in wanted_indices))
        count=0
        for i,block in zip(indices,iterator):
            if block.manifest!=documents[i]:raise ValueError('A paired context changed between admission passes')
            count+=1;yield i,block
        if count!=len(indices):raise ValueError('A bound paired context is missing from array admission')
    catalog=Catalog()
    for i,block in selected(originals):
        catalog.add(block.manifest,name=names[i],arrays=block.arrays,projection=block.mode_projection)
    result={}
    for i,block in selected(added):
        # Retain the small original context index, not all supplement arrays.
        current=copy.copy(catalog);current.additions={}
        current.add(block.manifest,name=names[i],arrays=block.arrays,projection=block.mode_projection)
        result.update(current.finish())
    return result


def validate_inventory(value,blocks):
    if not isinstance(value,dict) or not value or not set(value)<set(blocks):
        raise ValueError('An additive plan needs its complete original and supplement inventory')
    originals={row['examples_sha256'] for name,row in blocks.items() if name not in value}
    for name,row in value.items():
        d=fields(row,['sources','supplement_identity'])
        if (not checksum(d['supplement_identity']) or not isinstance(d['sources'],list) or not 1<=len(d['sources'])<=256):
            raise ValueError('Sparse plan sources need their captured identities')
        seen=set()
        for source in d['sources']:
            s=fields(source,SOURCE_KEYS);key=tuple(s[k] for k in SOURCE_KEYS)
            if any(not checksum(v) for v in key) or key in seen or s['pairs_sha256'] not in originals:
                raise ValueError('Sparse plan sources differ from the retained original inventory')
            seen.add(key)
    return value


def write(sources,arrays,rays,destination,*,direction_basis,native_protocol_sha256,native_curves_sha256,
          native_result_sha256,producer_implementation_sha256,names='examples'):
    """Adapt native curve data without regenerating or relabelling any point.

    Newly derived fields are the explicit original trajectory IDs and ambiguity
    flags. Every supplied numerical array is
    checked byte-for-byte on readback. This adapter does not independently
    certify the producer's diffraction calculations.
    """
    from pathlib import Path
    from ..execution import write_json
    from ..symmetry.compiler import file_sha
    from . import structural_mode_projection as projection
    from .structural_paired_blocks import read,filenames,source_paths,ARRAYS,MAX_BYTES
    sources=[Path(p) for p in sources]
    if not 1<=len(sources)<=256:raise ValueError('Sparse additions need a bounded original source inventory')
    blocks=[read(p,names=names) for p in sources]
    if any(block.mode_projection is None or present(block.manifest) for block in blocks):
        raise ValueError('Sparse additions bind complete original mapped blocks, never another supplement')
    before=[{str(p):file_sha(p) for p in source_paths(source,names=names)} for source in sources]
    required={'q','theory_intensity','background','parent_intensity','parent_background','parent_a','parameters','role','geometry_groups'}
    if (not isinstance(arrays,dict) or not required<=set(arrays)<=ARRAYS-OMITTED_SUMMARIES
            or any(type(v)is not np.ndarray or v.dtype.kind not in 'fiub' or not np.isfinite(v).all() for v in arrays.values())
            or sum(v.nbytes for v in arrays.values())>MAX_BYTES):
        raise ValueError('Complete bounded numeric native curve data required')
    original=blocks[0].manifest;mode_map=blocks[0].mode_projection
    keys=['source_sha256','instrument_sha256','child_sha256','candidate','seed','parent_seed',
        'trajectories','steps','mode_dimensions','cell_dimensions','declared_child_group']
    document={k:copy.deepcopy(original[k]) for k in keys}
    for k in ('grid','parent_relative_metric_range'):
        if k in original:document[k]=copy.deepcopy(original[k])
    from .structural_mode_projection import descriptor
    bindings=[dict(pairs_sha256=b.manifest['pairs_sha256'],manifest_identity=digest(b.manifest),
        projection_identity=descriptor(b.mode_projection)['identity']) for b in blocks]
    document.update(schema=RAW_SCHEMA,mode_projection=mode_map.stamp,observations_included=False,scientific_result=False,
        supplement=dict(schema=SCHEMA,sources=bindings,rays=copy.deepcopy(rays),direction_basis=direction_basis,
            native_protocol_sha256=native_protocol_sha256,native_curves_sha256=native_curves_sha256,
            native_result_sha256=native_result_sha256,
            producer_implementation_sha256=producer_implementation_sha256))
    trajectory,roles=layout(document);values=dict(arrays);p=arrays['parameters'];m=document['mode_dimensions']
    if (p.shape!=(len(trajectory),m+document['cell_dimensions']) or arrays['geometry_groups'].shape!=(len(trajectory),2)
            or not np.array_equal(arrays['role'],roles)):
        raise ValueError('Sparse native coordinates or point roles have incomplete dimensions')
    for k,v in dict(trajectory=trajectory,ambiguous_geometry=arrays['geometry_groups'][:,0]!=arrays['geometry_groups'][:,1]).items():
        if k not in values:values[k]=v
    prefix=f'parent:{document["source_sha256"]}:{document["seed"]}:{document["parent_seed"]}:trajectory-'
    document.update(groups=['parent:'+document['source_sha256']]+[prefix+str(v) for v in trajectory[1:]],
        examples=len(trajectory),bins=len(values['q']))
    validate_arrays(document,values,mode_map)
    catalog=Catalog()
    for i,block in enumerate(blocks):catalog.add(block.manifest,name=str(i),arrays=block.arrays,projection=block.mode_projection)
    catalog.add(document,name='addition',arrays=values,projection=mode_map);catalog.finish()
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    primary,packed=filenames(names);np.savez_compressed(destination/packed,**values)
    document['pairs_sha256']=file_sha(destination/packed)
    projection.copy(sources[0],destination,expected=document,stamp=mode_map.stamp)
    write_json(destination/primary,document);result=read(destination,names=names)
    if any(result.arrays[k].dtype!=v.dtype or result.arrays[k].tobytes()!=v.tobytes() for k,v in arrays.items()):
        raise ValueError('Sparse native adaptation changed a supplied numerical array')
    if any({str(p):file_sha(p) for p in source_paths(source,names=names)}!=prior for source,prior in zip(sources,before)):
        raise ValueError('An original paired source changed during sparse adaptation')
    return result


def admit_diagnostic(source,native_directory,protocol_file,destination,*,names='examples'):
    """Import the existing sealed native diagnostic, without fitting or edits.

    This verifies all protocol rays and points, input/producer/curve identities,
    and native control receipts. It is a format adapter, not authority to turn
    an uploaded declaration into a native-production certificate.
    """
    import io
    import json
    from pathlib import Path
    from ..symmetry.compiler import file_sha
    from ..live.array_limits import validate_npz_bytes
    from .tof_kernel_arrays import read_bounded
    from .structural_paired_blocks import read,MAX_BYTES
    native_directory=Path(native_directory);protocol_file=Path(protocol_file)
    result_file=native_directory/'result.json';curve_file=native_directory/'curves.npz'
    if any(p.is_symlink() for p in (protocol_file,result_file,curve_file)):
        raise ValueError('Native supplement artifacts must be regular captured files')
    protocol=json.loads(read_bounded(protocol_file,8*1024**2));detail=json.loads(read_bounded(result_file,8*1024**2))
    if (protocol.get('schema') not in {'radar-native-rare-family-coverage-protocol/v1','radar-native-signed-mode-supplement-protocol/v1'}
            or detail.get('schema') not in {'radar-structural-training-supplement-native-diagnostic/v1','radar-structural-native-signed-mode-supplement/v1'}
            or not isinstance(protocol.get('units'),list) or not 1<=len(protocol['units'])<=4096
            or type(detail.get('unit_index'))is not int or not 0<=detail['unit_index']<len(protocol['units'])
            or not checksum(protocol.get('package_sha256')) or detail.get('package_sha256')!=protocol['package_sha256']
            or detail.get('protocol_sha256')!=file_sha(protocol_file)
            or any(detail.get(k)is not False for k in ['observations_included','scientific_result','candidate_pruning_authorized'])
            or detail.get('test_curves_scored')!=0 or detail.get('training_pairs_added')!=0):
        raise ValueError('A complete observation-free native supplement protocol and diagnostic are required')
    original=read(source,names=names);d=original.manifest
    hashes=[d[k] for k in ['source_sha256','instrument_sha256','child_sha256']]
    request=protocol.get('request')
    if (not isinstance(request,dict) or not isinstance(request.get('settings'),dict)
            or protocol.get('input_sha256')!=hashes or detail.get('input_sha256')!=hashes
            or validate_candidate(request['settings'].get('candidate'))!=d['candidate']
            or validate_candidate(detail.get('candidate'))!=d['candidate']
            or request.get('grid')!=d.get('grid')
            or detail.get('mode_dimensions')!=d['mode_dimensions'] or detail.get('cell_dimensions')!=d['cell_dimensions']):
        raise ValueError('Native supplement producer inputs differ from the retained original context')
    unit=protocol['units'][detail['unit_index']];amplitudes=protocol.get('amplitudes')
    if (not isinstance(unit,dict) or unit.get('index')!=detail['unit_index'] or set(unit)!={'index','rays'}
            or not isinstance(unit.get('rays'),list) or not unit['rays']):
        raise ValueError('The native supplement unit lost its complete sealed ray scope')
    origins=[dict(kind='original_anchor')];rays=[];groups=['parent:'+d['source_sha256']]
    for supplied in unit['rays']:
        ray=fields(supplied,['axis','sign','original_trajectory','group','parent_a'])
        if not isinstance(amplitudes,list):raise ValueError('Native amplitude steps are missing')
        rays.append(dict(source=0,parent=ray['original_trajectory'],axis=ray['axis'],
            sign=ray['sign'],amplitudes=list(amplitudes)))
        origins.append(dict(kind='parent_control',**ray))
        origins.extend(dict(kind='fresh_native_hypothesis',amplitude=value,**ray) for value in amplitudes)
        groups.extend([ray['group']]*(1+len(amplitudes)))
    controls=detail.get('parent_controls');control_rows=[i for i,o in enumerate(origins) if o['kind']!='fresh_native_hypothesis']
    if (detail.get('origins')!=origins or detail.get('groups')!=groups
            or detail.get('points')!=len(origins) or detail.get('new_native_hypotheses')!=len(rays)*len(amplitudes)
            or detail.get('independent_native_checks')!=2*len(rays)+1
            or not isinstance(controls,list) or [v.get('row') for v in controls]!=control_rows):
        raise ValueError('Native diagnostic omitted or changed a sealed ray, amplitude, original group or control')
    for control in controls:
        c=fields(control,['row','intensity_scale','relative_error'])
        if (type(c['row'])is not int or type(c['intensity_scale']) not in (int,float)
                or type(c['relative_error']) not in (int,float)
                or not np.isfinite([c['intensity_scale'],c['relative_error']]).all()
                or abs(c['intensity_scale']-1)>1e-8 or not 0<=c['relative_error']<=1e-8):
            raise ValueError('A native parent control failed its retained intensity/profile comparison')
    raw=read_bounded(curve_file,MAX_BYTES)
    if file_sha(curve_file)!=detail.get('curves_sha256'):raise ValueError('Native supplement curve bytes changed')
    validate_npz_bytes(raw,maximum_bytes=MAX_BYTES)
    with np.load(io.BytesIO(raw),allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
    if len(arrays.get('role',[]))!=len(origins):raise ValueError('Native supplement arrays omit a sealed point')
    for i,origin in enumerate(origins[1:],1):
        if not np.array_equal(arrays['parent_a'][i],origin['parent_a']):
            raise ValueError('Native point cells differ from their sealed original parent contexts')
    basis=(WHITENED_BASIS if protocol['schema']=='radar-native-rare-family-coverage-protocol/v1'
           else protocol.get('direction_basis'))
    result=write([source],arrays,rays,destination,direction_basis=basis,native_protocol_sha256=file_sha(protocol_file),
        native_curves_sha256=detail['curves_sha256'],native_result_sha256=file_sha(result_file),
        producer_implementation_sha256=detail['package_sha256'],names=names)
    if result.manifest['groups']!=groups:raise ValueError('Native adaptation changed original group identities')
    return result
