"""Shared conditional-mode training and bounded data-only readout metadata.

Learned amplitude estimates are not experimental presence/absence claims.
Unknown conditions remain explicit, including families missing from a map.
"""
import hashlib
import json
import math
from fractions import Fraction

import numpy as np

from ..symmetry.contracts import fields,digest
from . import structural_mode_readout as readout
from .structural_family_targets import TargetRows
from .structural_mode_families import NORMALIZATION
from .structural_model_files import MAX_METADATA_BYTES
from .tof_kernel_arrays import checksum

SCHEMA='radar-trained-conditional-mode-readout/v1'
METRICS='radar-conditional-mode-validation/v1'
OBJECTIVE='all-compatible-class-and-conditional-mode-pinball/v1'
SELECTION='all-rows-seen/validation-normalized-class-nll-plus-macro-mode-pinball/v1'
ARRAY='mode_spec'


def validate(spec):
    d=fields(spec,['schema','classes','families','normalization','scales','known_intervals',
        'targets_identity','fitting_split_sha256','fitting_alias_groups','quantiles',
        'mode_head_trained','intervals_calibrated','experimental_mode_absence_qualified',
        'candidate_pruning_authorized'])
    classes=d['classes'];families=d['families'];scales=d['scales'];counts=d['known_intervals']
    if (d['schema']!=SCHEMA or not isinstance(classes,list) or not 1<=len(classes)<=230
            or any(type(c)is not int or not 1<=c<=230 for c in classes) or classes!=sorted(set(classes))
            or not isinstance(families,list) or not 1<=len(families)<=readout.MAX_FAMILIES
            or d['normalization']!=NORMALIZATION or not isinstance(scales,list) or len(scales)!=len(families)
            or any(type(v)is not float or not math.isfinite(v) or v<=0 for v in scales)
            or not isinstance(counts,list) or len(counts)!=len(classes)
            or any(not isinstance(row,list) or len(row)!=len(families)
                   or any(type(v)is not int or not 0<=v<=1_000_000 for v in row) for row in counts)
            or not any(v for row in counts for v in row)
            or any(not checksum(d[k]) for k in ['targets_identity','fitting_split_sha256'])
            or type(d['fitting_alias_groups'])is not int or not 1<=d['fitting_alias_groups']<=1_000_000
            or any(v>d['fitting_alias_groups'] for row in counts for v in row)
            or d['quantiles']!=[.1,.9] or d['mode_head_trained'] is not True
            or any(d[k]is not False for k in ['intervals_calibrated','experimental_mode_absence_qualified','candidate_pruning_authorized'])):
        raise ValueError('Complete unqualified conditional-mode normalization and training identities required')
    _families(families)
    return d


def _families(values):
    if not isinstance(values,list) or not 1<=len(values)<=readout.MAX_FAMILIES:
        raise ValueError('A complete bounded mode-family inventory is required')
    identifiers=[];parents=set()
    for family in values:
        f=fields(family,['id','parent_sha256','parent_convention','k','irrep'])
        k=f['k']
        if (not checksum(f['id']) or not checksum(f['parent_sha256'])
                or any(not isinstance(f[key],str) or not 1<=len(f[key])<=1000 for key in ['parent_convention','irrep'])
                or not isinstance(k,list) or len(k)!=3
                or any(not isinstance(v,str) or not 1<=len(v)<=80 for v in k)):
            raise ValueError('Every mode family needs its exact parent convention, k and irrep')
        try:canonical=[str(Fraction(v)) for v in k]
        except (ValueError,ZeroDivisionError) as error:raise ValueError('Exact rational mode-family k required') from error
        if canonical!=k or digest({key:value for key,value in f.items() if key!='id'})!=f['id']:
            raise ValueError('Mode family identity changed its parent convention or propagation vector')
        identifiers.append(f['id']);parents.add(f['parent_sha256'])
    if identifiers!=sorted(set(identifiers)) or len(parents)!=1:
        raise ValueError('Retain the complete ordered mode-family inventory for one parent')
    return values


def encode(spec):
    raw=json.dumps(validate(spec),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if len(raw)>MAX_METADATA_BYTES:raise ValueError('Complete mode readout metadata exceeds the portable bound')
    return np.frombuffer(raw,np.uint8)


def decode(value,metadata):
    if type(value)is not np.ndarray or value.dtype!=np.uint8 or value.ndim!=1 or not 0<len(value)<=MAX_METADATA_BYTES:
        raise ValueError('A bounded data-only mode specification is required')
    spec=validate(json.loads(value.tobytes()))
    if spec!=metadata:raise ValueError('Mode readout metadata differs from its bound weight artifact')
    return spec


def prepare(cohort,split):
    table=cohort.get('mode_family_targets')
    if table is None:return None
    if not isinstance(table,TargetRows):raise ValueError('Checked native family targets required')
    table.verify(cohort)
    if not table.metadata['families'] or not table.counts(np.flatnonzero(split==0)).any():return None
    normal=readout.fit(table,cohort)
    if normal.metadata['fitting_split_sha256']!=hashlib.sha256(split.tobytes()).hexdigest():
        raise ValueError('Mode and class training must share the exact independent reservation')
    spec=validate(dict(schema=SCHEMA,classes=table.metadata['classes'],families=table.metadata['families'],
        normalization=normal.metadata['normalization'],scales=normal.metadata['scales'],
        known_intervals=normal.metadata['known_intervals'],targets_identity=table.metadata['identity'],
        fitting_split_sha256=normal.metadata['fitting_split_sha256'],
        fitting_alias_groups=normal.metadata['fitting_alias_groups'],quantiles=[.1,.9],
        mode_head_trained=True,intervals_calibrated=False,experimental_mode_absence_qualified=False,
        candidate_pruning_authorized=False))
    return Context(table,normal,spec)


def queries(model,embedding,target):
    import torch
    return model.mode_head(embedding,*[torch.from_numpy(np.asarray(target[k],np.int64)).to(embedding.device)
        for k in ['row','space_group','family']])


def shared(model,values):
    from .structural_training_backend import device
    embedding=model.embedding(values.to(device(model)))
    return model.head[-1](embedding).to(values.device),embedding


class Context:
    def __init__(self,table,normal,spec):self.table=table;self.normal=normal;self.spec=spec

    def objective(self,model,embedding,indices):
        target=self.normal.targets(self.table,indices)
        prediction=queries(model,embedding,target)
        return readout.loss(prediction,target,rows=len(indices),class_count=len(self.spec['classes']))

    def metrics(self,*,interval_evaluation=None,groups=None):
        return Metrics(self,interval_evaluation=interval_evaluation,groups=groups)

    def transport(self,model,cpu,inputs):
        """Audit all trained class/family conditions on each representative."""
        import torch
        actual,embedding=shared(model,inputs);expected,other=shared(cpu,inputs)
        conditions=np.argwhere(np.asarray(self.spec['known_intervals'])>0)
        largest=0.;count=0;width=max(1,2048//len(inputs))
        for first in range(0,len(conditions),width):
            selected=conditions[first:first+width];n=len(selected)
            target=dict(row=np.repeat(np.arange(len(inputs),dtype=np.int64),n),
                space_group=np.tile(selected[:,0],len(inputs)),family=np.tile(selected[:,1],len(inputs)))
            left=queries(model,embedding,target).cpu();right=queries(cpu,other,target)
            if (not torch.isfinite(left).all() or not torch.isfinite(right).all()
                    or not torch.allclose(left,right,rtol=1e-4,atol=1e-5)):
                raise ValueError('GPU mode readout does not reproduce portable CPU estimates')
            largest=max(largest,float((left-right).abs().max()));count+=len(left)
        return actual,expected,dict(mode_queries=count,maximum_mode_absolute_error=largest)


class Metrics:
    def __init__(self,context,*,interval_evaluation=None,groups=None):
        self.context=context;shape=(len(context.spec['classes']),len(context.spec['families']))
        if interval_evaluation is not None:
            if (digest(interval_evaluation.spec)!=digest(context.spec) or not isinstance(groups,list)
                    or len(groups)!=len(context.table)):
                raise ValueError('Mode interval evaluation needs its complete source-bound row groups')
        self.interval_evaluation=interval_evaluation;self.groups=groups
        self.count=np.zeros(shape,np.int64);self.pinball=np.zeros(shape,np.float64)
        self.contained=np.zeros(shape,np.int64);self.rows=0;self.known_rows=0

    def add(self,model,embedding,indices):
        target=self.context.normal.targets(self.context.table,indices)
        prediction=queries(model,embedding,target).detach().cpu().numpy().astype(np.float64)
        if prediction.shape!=(len(target['row']),2) or not np.isfinite(prediction).all():
            raise ValueError('Finite complete conditional-mode predictions required')
        truth=np.column_stack((target['lower'],target['upper']));residual=truth-prediction
        quantiles=np.array([.1,.9]);per_query=np.maximum(quantiles*residual,(quantiles-1)*residual).mean(1)
        c,f=target['space_group'],target['family']
        np.add.at(self.count,(c,f),1);np.add.at(self.pinball,(c,f),per_query)
        np.add.at(self.contained,(c,f),(prediction[:,0]<=truth[:,0])&(prediction[:,1]>=truth[:,1]))
        self.rows+=len(indices);self.known_rows+=len(np.unique(target['row']))
        if self.interval_evaluation is not None:
            self.interval_evaluation.add(prediction,target,[self.groups[i] for i in indices])

    def finish(self):
        known=self.count>0;means=self.pinball/np.maximum(self.count,1)
        conditions=[dict(space_group_index=int(c),family_index=int(f),queries=int(self.count[c,f]),
                         pinball_loss=float(means[c,f]),target_interval_containment=float(self.contained[c,f]/self.count[c,f]))
                    for c,f in np.argwhere(known)]
        result=dict(schema=METRICS,classes=self.context.spec['classes'],families=[f['id'] for f in self.context.spec['families']],
            rows=self.rows,known_rows=self.known_rows,known_queries=int(self.count.sum()),conditions=conditions,
            available=bool(known.any()),loss=float(means[known].mean()) if known.any() else None,
            experimental_mode_absence_qualified=False,candidate_pruning_authorized=False)
        return validate_metrics(result)


def validate_metrics(value):
    d=fields(value,['schema','classes','families','rows','known_rows','known_queries','conditions','available','loss',
                    'experimental_mode_absence_qualified','candidate_pruning_authorized'])
    if (d['schema']!=METRICS or not isinstance(d['classes'],list) or not 1<=len(d['classes'])<=230
            or any(type(c)is not int or not 1<=c<=230 for c in d['classes']) or d['classes']!=sorted(set(d['classes']))
            or not isinstance(d['families'],list) or not 1<=len(d['families'])<=readout.MAX_FAMILIES
            or any(not checksum(v) for v in d['families']) or d['families']!=sorted(set(d['families']))
            or type(d['rows'])is not int or not 1<=d['rows']<=1_000_000
            or type(d['known_rows'])is not int or not 0<=d['known_rows']<=d['rows']
            or type(d['known_queries'])is not int or d['known_queries']<d['known_rows']
            or not isinstance(d['conditions'],list) or type(d['available'])is not bool
            or d['available']!=bool(d['conditions']) or d['experimental_mode_absence_qualified'] is not False
            or d['candidate_pruning_authorized'] is not False):raise ValueError('Invalid conditional-mode validation scope')
    identities=[];losses=[];queries_count=0
    for condition in d['conditions']:
        r=fields(condition,['space_group_index','family_index','queries','pinball_loss','target_interval_containment'])
        if (type(r['space_group_index'])is not int or not 0<=r['space_group_index']<len(d['classes'])
                or type(r['family_index'])is not int or not 0<=r['family_index']<len(d['families'])
                or type(r['queries'])is not int or not 1<=r['queries']<=d['known_rows']
                or type(r['pinball_loss'])is not float or not math.isfinite(r['pinball_loss']) or r['pinball_loss']<0
                or type(r['target_interval_containment'])is not float or not 0<=r['target_interval_containment']<=1):
            raise ValueError('Mode validation must retain every known conditional denominator')
        identities.append((r['space_group_index'],r['family_index']));losses.append(r['pinball_loss']);queries_count+=r['queries']
    if (identities!=sorted(set(identities)) or queries_count!=d['known_queries']
            or (bool(losses) and (type(d['loss'])is not float or not math.isclose(d['loss'],sum(losses)/len(losses),rel_tol=1e-12,abs_tol=0)))
            or (not losses and (d['loss'] is not None or d['known_rows']!=0 or d['known_queries']!=0))):
        raise ValueError('Mode validation cannot change its conditions or macro loss')
    return d


def infer(model,embedding,spec,*,space_groups):
    """Return raw learned ranges; no uncalibrated prediction becomes absence."""
    validate(spec)
    if (model.training or embedding.shape!=(1,64)
            or not isinstance(space_groups,list) or not space_groups or len(set(space_groups))!=len(space_groups)
            or any(type(c)is not int or c not in spec['classes'] for c in space_groups)):
        raise ValueError('Mode estimates need the complete requested retained class set')
    counts=np.asarray(spec['known_intervals']);target={'row':[],'space_group':[],'family':[]};result=[];positions=[]
    for group in space_groups:
        c=spec['classes'].index(group)
        for f,family in enumerate(spec['families']):
            row=dict(space_group=group,family_index=f,status='unresolved',lower_estimate=None,upper_estimate=None,
                     reason='No known fitting targets for this class and family',intervals_calibrated=False)
            if counts[c,f]:
                target['row'].append(0);target['space_group'].append(c);target['family'].append(f);positions.append(len(result))
            result.append(row)
    if positions:
        target={k:np.asarray(v,np.int64) for k,v in target.items()}
        values=np.concatenate([queries(model,embedding,{k:v[first:first+2048] for k,v in target.items()}).detach().cpu().numpy()
            for first in range(0,len(positions),2048)])
        from .symmetry_mode_normalization import physical
        decoded=physical(values,np.asarray(spec['scales'])[np.asarray(target['family']),None])
        for index,bounds in zip(positions,decoded):
            result[index].update(lower_estimate=float(bounds[0]),upper_estimate=float(bounds[1]),
                reason='Learned 10/90 amplitude estimates; independent detectability calibration and native confirmation remain required')
    return validate_estimates(dict(schema=SCHEMA,available=True,space_groups=list(space_groups),families=spec['families'],
        readout_identity=digest(spec),rows=result,amplitude_normalization=spec['normalization'],
        reason='Conditional learned amplitude estimates; independent sensitivity calibration is unavailable',
        intervals_calibrated=False,experimental_mode_absence_qualified=False,candidate_pruning_authorized=False))


def unavailable(reason):
    return validate_estimates(dict(schema=SCHEMA,available=False,space_groups=[],families=[],readout_identity=None,
        rows=[],amplitude_normalization=NORMALIZATION,reason=reason,intervals_calibrated=False,
        experimental_mode_absence_qualified=False,candidate_pruning_authorized=False))


def validate_estimates(value,*,space_groups=None):
    d=fields(value,['schema','available','space_groups','families','readout_identity','rows','amplitude_normalization',
        'reason','intervals_calibrated','experimental_mode_absence_qualified','candidate_pruning_authorized'])
    groups=d['space_groups'];families=d['families'];rows=d['rows']
    if (d['schema']!=SCHEMA or type(d['available'])is not bool or d['amplitude_normalization']!=NORMALIZATION
            or any(d[k]is not False for k in ['intervals_calibrated','experimental_mode_absence_qualified','candidate_pruning_authorized'])
            or not isinstance(d['reason'],str) or not 1<=len(d['reason'])<=1000
            or not isinstance(groups,list) or any(type(c)is not int or not 1<=c<=230 for c in groups)
            or len(groups)!=len(set(groups)) or len(groups)>230 or not isinstance(rows,list)):
        raise ValueError('Mode estimates cannot acquire unqualified presence, absence or pruning authority')
    if not d['available']:
        if groups or families or rows or d['readout_identity'] is not None:
            raise ValueError('Unavailable mode estimates cannot claim learned amplitudes')
        return d
    _families(families)
    if (not groups or not checksum(d['readout_identity']) or len(rows)!=len(groups)*len(families)
            or space_groups is not None and groups!=space_groups):
        raise ValueError('Mode estimates must retain every requested class/family condition')
    for index,row in enumerate(rows):
        r=fields(row,['space_group','family_index','status','lower_estimate','upper_estimate','reason','intervals_calibrated'])
        lower,upper=r['lower_estimate'],r['upper_estimate']
        if (type(r['space_group'])is not int or r['space_group']!=groups[index//len(families)]
                or type(r['family_index'])is not int or r['family_index']!=index%len(families)
                or r['status']!='unresolved' or r['intervals_calibrated'] is not False
                or not isinstance(r['reason'],str) or not 1<=len(r['reason'])<=1000
                or (lower is None)!=(upper is None)
                or lower is not None and (type(lower)is not float or type(upper)is not float
                    or not math.isfinite(lower) or not math.isfinite(upper) or not 0<=lower<=upper)):
            raise ValueError('Uncalibrated mode estimates cannot become a physical presence/absence classification')
    return d


def scope_estimates(value,space_groups):
    validate_estimates(value)
    if not value['available']:return value
    if (not space_groups or space_groups!=[c for c in value['space_groups'] if c in space_groups]):
        raise ValueError('Requested mode classes differ from their prepared ranking order')
    return validate_estimates(dict(value,space_groups=list(space_groups),
        rows=[r for r in value['rows'] if r['space_group'] in space_groups]),space_groups=space_groups)
