"""Three physical curves and attention; all native candidates remain available."""
import hashlib
import io
from pathlib import Path

import numpy as np

from ..live.array_limits import validate_npz_bytes
from ..symmetry.contracts import validate_candidate
from ..symmetry.resources import allocation
from .structural_attention import network
from . import structural_attention_weights as weights
from .structural_features import three_curve_features
from .structural_model_files import document,MAX_BYTES
from .structural_ranker import _class_routes
from .structural_support import distances,MAX_BYTES as SUPPORT_BYTES,STREAM_METHOD
from .tof_kernel_arrays import read_bounded,checksum

ARRAYS={'parameters','q_edges','normalization','classes','reference_features','support_prototypes'}


class Ranker:
    def __init__(self,directory,*,threads=1,metadata_name='model.json',weights_name='weights.npz'):
        import torch
        d=document(directory,metadata_name,weights_name)
        raw=read_bounded(Path(directory)/weights_name,MAX_BYTES)
        classes=d.get('classes');bins=d.get('bins')
        if (d.get('schema')!=weights.SCHEMA or d.get('architecture')!=weights.ARCHITECTURE
                or d.get('representation')!=weights.REPRESENTATION or d.get('channels')!=3
                or d.get('reference_kind')!=weights.REFERENCE_KIND
                or d.get('normalization_method')!=weights.NORMALIZATION
                or d.get('weights_sha256')!=hashlib.sha256(raw).hexdigest()
                or not isinstance(classes,list) or not 2<=len(classes)<=230
                or any(type(c) is not int or not 1<=c<=230 for c in classes)
                or classes!=sorted(set(classes)) or type(bins) is not int or not 64<=bins<=8192
                or d.get('background_required') is not True):
            raise ValueError('Unsupported or changed three-curve attention model')
        if 'dataset' in d:
            from .structural_training_counts import validate as validate_counts
            counts=validate_counts(d['dataset'])
            if (counts['unique_rows']!=d.get('examples')
                    or counts['generated_pairs']!=d.get('original_examples')
                    or counts['trajectory_groups']!=d.get('independent_groups')
                    or [counts['partitions'][p]['rows'] for p in ('train','validation','test')]!=d.get('partition_rows')):
                raise ValueError('Attention training-count receipt differs from its model')
        candidates=d.get('candidates');groups=d.get('candidate_groups')
        if (not isinstance(candidates,list) or not 1<=len(candidates)<=5000
                or not isinstance(groups,dict)
                or any(not checksum(d.get(k)) for k in ('parent_sha256','instrument_sha256'))):
            raise ValueError('Three-curve ranking needs exact candidate and physical-context identities')
        candidates=[validate_candidate(c) for c in candidates];identifiers=[c['id'] for c in candidates]
        if (len(identifiers)!=len(set(identifiers)) or set(groups)!=set(identifiers)
                or any(type(v) is not int or not 1<=v<=230 for v in groups.values())
                or any(not c['child'] or c['magnetic_space_group'] is not None for c in candidates)
                or 'class_candidate_ids' not in d):
            raise ValueError('Three-curve ranking needs complete native child-to-family mappings')
        routes=_class_routes(classes,groups,d['class_candidate_ids'])
        if d.get('training_domain') is not None:
            from .structural_domain import validate
            validate(d['training_domain'])
        validate_npz_bytes(raw,maximum_bytes=MAX_BYTES)
        with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
            if set(archive.files)!=ARRAYS:raise ValueError('Unexpected three-curve model array inventory')
            a={k:archive[k] for k in ARRAYS}
        edges=a['q_edges'];normal=a['normalization'];reference=a['reference_features']
        if (edges.dtype!=np.float64 or edges.shape!=(bins+1,) or not np.isfinite(edges).all()
                or edges[0]<=0 or np.any(np.diff(edges)<=0)
                or normal.dtype!=np.float64 or normal.shape!=(7,) or not np.isfinite(normal).all()
                or np.any(normal[:3]!=0) or np.any(normal[3:6]<=0) or normal[6]<0
                or a['classes'].dtype!=np.int32 or not np.array_equal(a['classes'],classes)
                or reference.dtype!=np.float32 or reference.shape!=(3,bins)
                or not np.isfinite(reference).all()
                or hashlib.sha256(reference.tobytes()).hexdigest()!=d.get('reference_feature_sha256')):
            raise ValueError('Invalid three-curve normalization, reference or coordinate grid')
        with np.errstate(over='ignore',under='ignore'):
            mean=normal[:3].astype(np.float32).reshape(3,1)
            scale=normal[3:6].astype(np.float32).reshape(3,1)
        if (not np.isfinite(scale).all() or np.any(scale<=0)
                or not np.array_equal(normal[3:6],scale.ravel().astype(np.float64))):
            raise ValueError('Three-curve scales must retain their exact trained float32 values')
        prototypes=a['support_prototypes']
        if (d.get('support_method')!=STREAM_METHOD or prototypes.dtype!=np.float32
                or prototypes.ndim!=2 or not 1<=len(prototypes)<=256
                or prototypes.shape[1]!=3*bins or prototypes.nbytes>SUPPORT_BYTES
                or not np.isfinite(prototypes).all()):
            raise ValueError('Invalid three-curve training-support index')
        self.allocation=allocation(threads);torch.set_num_threads(self.allocation['workers'])
        self.model=network(len(classes));weights.restore(self.model,a['parameters'],d.get('tensor_layout_sha256'))
        self.metadata=d;self.edges=edges;self.mean=mean;self.scale=scale;self.reference=reference
        self.routes=routes;self.support_prototypes=prototypes;self.support_limit=float(normal[6])

    def rank(self,q,intensity,*,background,parent_sha256,instrument_sha256,valid=None,reference=None):
        import torch
        if (parent_sha256!=self.metadata['parent_sha256'] or instrument_sha256!=self.metadata['instrument_sha256']):
            raise ValueError('This ranker needs its exact prepared parent and instrument context')
        if (not isinstance(reference,dict) or set(reference)-{'q','intensity','background','valid'}
                or not {'q','intensity','background'}<=set(reference)
                or not np.array_equal(np.asarray(reference['q']),np.asarray(q))):
            raise ValueError('Three-curve ranking requires a freshly calculated parent reference on the same native Q grid')
        mask=valid
        if reference.get('valid') is not None:
            other=np.asarray(reference['valid'])
            if other.dtype!=bool or other.shape!=np.shape(q):raise ValueError('Invalid native parent validity metadata')
            if valid is not None and (np.asarray(valid).dtype!=bool or np.shape(valid)!=other.shape):
                raise ValueError('Invalid native observation validity metadata')
            mask=other if valid is None else other&valid
        prepared=three_curve_features(q,intensity,reference['intensity'],self.edges,
            background=background,reference_background=reference['background'],valid=mask)
        if np.any(prepared['coverage']<.999):
            raise ValueError('This structural ranker is not qualified for missing coordinate intervals')
        feature=(prepared['features']-self.mean)/self.scale
        distance=float(distances(self.support_prototypes,feature.reshape(1,-1))[0])
        inside=bool(distance<=self.support_limit+np.finfo(np.float32).eps*max(1.,self.support_limit))
        support=dict(status='within_empirical_synthetic_support' if inside else 'outside_prepared_support',
            nearest_training_rms=distance,validation_maximum_rms=self.support_limit,statistical_coverage_qualified=False)
        scores=None
        if inside:
            with torch.no_grad():scores=self.model(torch.from_numpy(feature[None])).softmax(1)[0].numpy()
            if not np.isfinite(scores).all():raise ValueError('Three-curve model inference produced nonfinite scores')
        order=range(len(self.metadata['classes'])) if scores is None else np.argsort(-scores,kind='stable')
        families=[dict(space_group=self.metadata['classes'][i],model_score=None if scores is None else float(scores[i]),
            candidate_ids=list(self.routes[str(self.metadata['classes'][i])])) for i in order]
        return dict(families=families,ranking_available=inside,
            candidate_order=(list(dict.fromkeys(key for row in families for key in row['candidate_ids']))
                if inside else sorted(self.metadata['candidate_groups'])),support=support,
            screening_qualified=False,candidate_pruning_authorized=False,native_refinement_required=True,
            complete_native_search_required=True,
            interpretation=('Relative attention scores order hypotheses; all native candidates remain available'
                if inside else 'This pattern is outside the prepared simulations; retain all families for native refinement'))
