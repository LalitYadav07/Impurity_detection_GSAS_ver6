"""Sparse conditional mode intervals from every original native curve alias.

These are geometric supervision data, not experimental presence/absence
certificates. A missing map/family or gauge-dependent amplitude remains
unknown. Noise siblings share intervals without duplicating physical evidence.
"""
from array import array
from dataclasses import dataclass
import hashlib
import json
import sys

import numpy as np

from .structural_mode_families import FamilyProjection,NORMALIZATION,MAX_BATCH_ROWS
from .structural_mode_projection import descriptor
from .structural_paired_blocks import PairedBlock
from .structural_replay_limits import MAX_ROWS
from .tof_kernel_arrays import array_identity,checksum

SCHEMA='radar-native-conditional-mode-family-targets/v1'
MAX_BYTES=256*1024**2
MAX_BATCH_BYTES=16*1024**2
_PERMIT=object()
CONTEXT=('parent_sha256','instrument_sha256','input_sha256','mode_projection_references',
    'classes','families','normalization','references_sha256')


def _json_sha(value):
    hashed=hashlib.sha256()
    for chunk in json.JSONEncoder(sort_keys=True,separators=(',',':'),allow_nan=False).iterencode(value):hashed.update(chunk.encode())
    return hashed.hexdigest()


def _binding(context,arrays_sha256,cohort):
    return _json_sha(dict(context=context,arrays_sha256=arrays_sha256,groups=cohort['groups'],
        split=np.asarray(cohort['split']).tolist(),
        compatible_labels_sha256=array_identity(dict(labels=np.asarray(cohort['compatible_labels'])))))


def _immutable(value,dtype=None):
    value=np.asarray(value,dtype=dtype)
    return np.frombuffer(value.tobytes(),dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True)
class _Block:
    geometry:np.ndarray
    values:np.ndarray
    known:np.ndarray
    families:tuple


class TargetRows:
    """Owned, bounded sparse intervals; never a dense rows/classes/modes cube."""
    def __init__(self,arrays,metadata,*,_permit=None):
        if _permit is not _PERMIT:raise TypeError('Build targets from checked native paired blocks')
        self.arrays=arrays;self.metadata=metadata

    def __len__(self):return len(self.arrays['row_group'])

    def verify(self,cohort):
        """A target table cannot be borrowed after source/row/fold changes."""
        d=self.metadata;context={k:d[k] for k in CONTEXT}
        if (d['schema']!=SCHEMA or len(self)!=len(cohort['groups'])
                or d['normalization']!=NORMALIZATION or d['absence_from_missing_family'] is not False
                or d['experimental_mode_absence_qualified'] is not False or d['candidate_pruning_authorized'] is not False
                or any(d[k]!=cohort[k] for k in ['parent_sha256','instrument_sha256','input_sha256'])
                or d['mode_projection_references']!=cohort.get('mode_projection_references',[None]*len(d['input_sha256']))
                or d['classes']!=np.asarray(cohort['classes']).tolist()
                or d['references_sha256']!=_json_sha(cohort['references'])
                or d['arrays_sha256']!=array_identity(self.arrays)
                or d['identity']!=_binding(context,d['arrays_sha256'],cohort)):
            raise ValueError('Mode-family targets differ from the exact native cohort, aliases or original folds')

    def batch(self,indices):
        selected=np.asarray(indices)
        if (selected.ndim!=1 or selected.dtype.kind not in 'iu' or not 1<=len(selected)<=MAX_BATCH_ROWS
                or np.any(selected<0) or np.any(selected>=len(self))):
            raise ValueError('Read a bounded batch of complete mode-target rows')
        a=self.arrays;groups=a['row_group'][selected];offsets=a['offsets']
        lengths=offsets[groups+1]-offsets[groups]
        if int(lengths.sum())*40>MAX_BATCH_BYTES:
            raise ValueError('Split the mode-target minibatch; no target may be truncated')
        positions=np.concatenate([np.arange(int(offsets[g]),int(offsets[g+1]),dtype=np.int64) for g in groups])
        return dict(row=np.repeat(np.arange(len(selected),dtype=np.int64),lengths.astype(np.int64)),
            space_group=a['space_group'][positions].astype(np.int64),family=a['family'][positions].astype(np.int64),
            lower=a['lower'][positions].copy(),upper=a['upper'][positions].copy())

    def counts(self,indices):
        """Row-weighted target counts; these are not independent sample counts."""
        indices=np.asarray(indices)
        if indices.ndim!=1 or indices.dtype.kind not in 'iu' or np.any(indices<0) or np.any(indices>=len(self)):
            raise ValueError('Mode-target counts need valid fitting or holdout row indices')
        a=self.arrays;weights=np.bincount(a['row_group'][indices],minlength=len(a['offsets'])-1)
        groups=np.repeat(np.arange(len(weights)),np.diff(a['offsets']).astype(np.int64))
        families=len(self.metadata['families']);shape=(len(self.metadata['classes']),families)
        if not families:return np.zeros(shape,np.int64)
        encoded=a['space_group'].astype(np.int64)*families+a['family']
        return np.bincount(encoded,weights=weights[groups],minlength=shape[0]*shape[1]).reshape(shape).astype(np.int64)


def build(block_factory,cohort,*,_maximum_bytes=MAX_BYTES):
    """Bind every alias before fitting; input values and original folds stay fixed."""
    if not callable(block_factory) or type(_maximum_bytes)is not int or not 1<=_maximum_bytes<=MAX_BYTES:
        raise ValueError('Use a bounded checked paired-block source for family targets')
    classes=np.asarray(cohort['classes']);labels=np.asarray(cohort['compatible_labels'])
    split=np.asarray(cohort['split']);refs=cohort['references'];groups=cohort['groups'];n=len(groups)
    hashes=cohort['input_sha256'];maps=cohort.get('mode_projection_references',[None]*len(hashes))
    if (not 1<=n<=MAX_ROWS or classes.dtype!=np.int32 or classes.ndim!=1
            or not 1<=len(classes)<=230 or np.any(classes<1) or np.any(classes>230) or np.any(np.diff(classes)<=0)
            or labels.dtype!=bool or labels.shape!=(n,len(classes)) or not labels.any(1).all()
            or split.dtype!=np.int8 or split.shape!=(n,) or not set(split)<={0,1,2,3}
            or not isinstance(groups,list) or any(not isinstance(g,str) or not 1<=len(g)<=500 for g in groups)
            or len(refs)!=n or not isinstance(hashes,list) or not 1<=len(hashes)<=5000
            or not isinstance(maps,list) or len(maps)!=len(hashes) or any(not checksum(v) for v in hashes)
            or any(not checksum(cohort.get(k)) for k in ['parent_sha256','instrument_sha256'])):
        raise ValueError('Family targets need complete native identities, labels, references and folds')
    used=n*4;blocks=[];families={};projection_ids=[];opaque=0
    def reserve(size):
        nonlocal used
        used+=size
        if used>_maximum_bytes:raise ValueError('Complete mode-target storage exceeds its bounded allocation')
    reserve(0)
    for index,block in enumerate(block_factory()):
        if not isinstance(block,PairedBlock) or index>=len(hashes):
            raise ValueError('Family targets require the exact checked paired block order')
        d=block.manifest;a=block.arrays;projection=block.mode_projection
        if (d['pairs_sha256']!=hashes[index] or d['source_sha256']!=cohort['parent_sha256']
                or d['instrument_sha256']!=cohort['instrument_sha256']):
            raise ValueError('Mode targets differ from the captured native input cohort')
        identity=None if projection is None else descriptor(projection)['identity']
        if identity!=maps[index]:raise ValueError('Mode targets cannot borrow another native coordinate map')
        geometry=_immutable(a['geometry_groups']);count=len(geometry);reserve(geometry.nbytes)
        if projection is None:
            blocks.append(_Block(geometry,np.empty((count,0)),np.empty((count,0),bool),()))
            projection_ids.append(None);continue
        model=FamilyProjection(projection);nf=len(model.families)
        reserve(count*nf*9)
        values=np.empty((count,nf),np.float64);known=np.empty((count,nf),bool)
        parameters=a['parameters'][:,:model.mode_dimensions]
        if parameters.dtype.kind!='f':raise ValueError('Native family coordinates must retain their floating point values')
        for start in range(0,count,MAX_BATCH_ROWS):
            result=model.project(parameters[start:start+MAX_BATCH_ROWS])
            values[start:start+MAX_BATCH_ROWS]=result['amplitudes'];known[start:start+MAX_BATCH_ROWS]=result['known']
        ids=[]
        for family in model.families:
            entry={k:family[k] for k in ['id','parent_sha256','parent_convention','k','irrep']}
            if entry['id'] not in families:reserve(len(json.dumps(entry).encode())+256)
            previous=families.setdefault(entry['id'],entry)
            if previous!=entry:raise ValueError('One mode-family identity cannot describe different parent conventions')
            ids.append(entry['id'])
        opaque+=len(model.opaque_native_axes);projection_ids.append(identity)
        blocks.append(_Block(geometry,_immutable(values),_immutable(known),tuple(ids)))
    if len(blocks)!=len(hashes):raise ValueError('Mode-target block inventory is incomplete')
    ordered=sorted(families);family_index={identifier:i for i,identifier in enumerate(ordered)}
    # Arrays grow only with known class/family intervals, not the Cartesian
    # product of every row, symmetry class and unobserved mode family.
    offsets=array('Q',[0]);class_values=array('B');family_values=array('I');lower=array('d');upper=array('d')
    row_group=np.empty(n,np.int32);aliases={};unknown=ambiguous=zeros=0
    raw_references=canonical_references=0
    for row,original in enumerate(refs):
        if not isinstance(original,(list,tuple)) or not original:
            raise ValueError('Every training row needs all original native aliases')
        key=[]
        for raw in original:
            if (not isinstance(raw,(list,tuple)) or len(raw)!=2 or any(type(v)is not int for v in raw)
                    or not 0<=raw[0]<len(blocks) or not 0<=raw[1]<len(blocks[raw[0]].geometry)):
                raise ValueError('Invalid original native alias reference')
            key.append(tuple(raw))
        raw_references+=len(key);key=tuple(sorted(set(key)))
        wanted=tuple(np.flatnonzero(labels[row]).tolist())
        if key in aliases:
            group_id,expected,fold,physical_group=aliases[key]
            if expected!=wanted or fold!=int(split[row]) or physical_group!=groups[row]:
                raise ValueError('Equivalent native aliases cannot change conditional labels or cross a holdout')
            row_group[row]=group_id;continue
        reserve(sys.getsizeof(key)+sum(sys.getsizeof(v) for v in key)+128)
        group_id=len(aliases);aliases[key]=(group_id,wanted,int(split[row]),groups[row]);row_group[row]=group_id
        canonical_references+=len(key);native_classes=set()
        for b,i in key:native_classes.update(map(int,blocks[b].geometry[i]))
        if native_classes!={int(classes[c]) for c in wanted}:
            raise ValueError('Conditional class labels differ from the complete original native aliases')
        for column in wanted:
            eligible=[(blocks[b],i) for b,i in key if int(classes[column]) in blocks[b].geometry[i]]
            common=None;by_point=[]
            for block,i in eligible:
                available={identifier:float(block.values[i,j]) for j,identifier in enumerate(block.families) if block.known[i,j]}
                by_point.append(available);common=set(available) if common is None else common&available.keys()
            common=sorted(common or ());unknown+=len(ordered)-len(common)
            for identifier in common:
                values=[point[identifier] for point in by_point];lo=min(values);hi=max(values)
                reserve(21);class_values.append(column);family_values.append(family_index[identifier]);lower.append(lo);upper.append(hi)
                ambiguous+=lo!=hi;zeros+=hi==0.
        offsets.append(len(lower))
    if (offsets.itemsize!=8 or family_values.itemsize!=4 or lower.itemsize!=8):
        raise ValueError('Native family targets require the installed fixed numerical layout')
    arrays=dict(row_group=_immutable(row_group),offsets=_immutable(offsets,np.uint64),
        space_group=_immutable(class_values,np.uint8),family=_immutable(family_values,np.uint32),
        lower=_immutable(lower,np.float64),upper=_immutable(upper,np.float64))
    context=dict(parent_sha256=cohort['parent_sha256'],instrument_sha256=cohort['instrument_sha256'],
        input_sha256=list(hashes),mode_projection_references=projection_ids,
        classes=classes.tolist(),families=[families[k] for k in ordered],normalization=NORMALIZATION,
        references_sha256=_json_sha(refs))
    metadata=dict(schema=SCHEMA,**context,arrays_sha256=array_identity(arrays),rows=n,
        native_points=sum(len(b.geometry) for b in blocks),unique_alias_groups=len(aliases),
        original_references=raw_references,canonical_references=canonical_references,
        intervals=len(lower),ambiguous_intervals=int(ambiguous),exact_zero_intervals=int(zeros),
        unknown_class_family_targets=int(unknown),opaque_native_axes=opaque,
        mapped_blocks=sum(v is not None for v in projection_ids),total_blocks=len(blocks),
        storage_bytes=sum(a.nbytes for a in arrays.values()),available=bool(len(lower)),
        absence_from_missing_family=False,experimental_mode_absence_qualified=False,candidate_pruning_authorized=False)
    # Bind the original class/fold provenance too; no calibration labels or
    # fitting-only values are selected or normalized by this data constructor.
    metadata['identity']=_binding(context,metadata['arrays_sha256'],cohort)
    return TargetRows(arrays,metadata,_permit=_PERMIT)
