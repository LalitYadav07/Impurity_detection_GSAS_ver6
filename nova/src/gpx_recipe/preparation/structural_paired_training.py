"""Bounded three-curve attention training with trajectory-isolated evaluation."""
import copy
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np

from ..execution import write_json
from ..live.array_limits import validate_npz_bytes
from ..symmetry.resources import allocation
from .structural_attention import network
from . import structural_attention_weights as weights
from .structural_attention_metrics import evaluate
from .structural_model_files import MAX_BYTES,MAX_METADATA_BYTES
from .structural_ranker import _class_routes
from .structural_support import fit_batches as fit_support
from .structural_training_data import feature_rows,batches,channel_statistics
from .structural_training_counts import describe as describe_counts
from . import structural_training_checkpoint as checkpoint
from .tof_kernel_arrays import checksum


def _inputs(cohort):
    rows=feature_rows(cohort['features']);targets=np.asarray(cohort['compatible_labels'])
    classes=np.asarray(cohort['classes']);split=np.asarray(cohort['split']);n=len(rows)
    if (rows.shape[1]!=3 or cohort.get('channels')!=3
            or cohort.get('representation')!=weights.REPRESENTATION
            or cohort.get('reference_kind')!=weights.REFERENCE_KIND
            or cohort.get('partition_method')!='class-supported-whole-trajectories/v1'
            or classes.dtype!=np.int32 or classes.ndim!=1 or not 2<=len(classes)<=230
            or np.any(classes<1) or np.any(classes>230) or np.any(np.diff(classes)<=0)
            or targets.dtype!=bool or targets.shape!=(n,len(classes)) or not targets.any(1).all()
            or split.dtype!=np.int8 or split.shape!=(n,) or set(split)!={0,1,2}
            or not all(targets[split==code].any(0).all() for code in range(3))
            or any(not checksum(cohort.get(k)) for k in ('parent_sha256','instrument_sha256'))):
        raise ValueError('A complete three-curve cohort with every family in independent holdouts is required')
    groups=cohort['groups'];assignments={}
    if len(groups)!=n or any(not isinstance(g,str) or not 1<=len(g)<=500 for g in groups):
        raise ValueError('Complete physical trajectory groups required')
    for group,code in zip(groups,split):assignments.setdefault(group,set()).add(int(code))
    if any(len(codes)!=1 for codes in assignments.values()):raise ValueError('Physical trajectories cannot cross training holdouts')
    reference=np.asarray(cohort['reference_features']);edges=np.asarray(cohort['q_edges'])
    if (reference.dtype!=np.float32 or reference.shape!=(3,rows.shape[2]) or not np.isfinite(reference).all()
            or hashlib.sha256(reference.tobytes()).hexdigest()!=cohort.get('reference_feature_sha256')
            or edges.dtype!=np.float64 or edges.shape!=(rows.shape[2]+1,) or not np.isfinite(edges).all()
            or edges[0]<=0 or np.any(np.diff(edges)<=0)):
        raise ValueError('The exact finite three-curve reference and Q grid are required')
    provenance=cohort.get('example_simulations')
    if (not isinstance(provenance,list) or len(provenance)!=n
            or any(not isinstance(p,list) or not p or any(not isinstance(v,dict) for v in p) for p in provenance)):
        raise ValueError('Clean/noisy simulation provenance is required for the training curriculum')
    clean=np.array([any(v.get('kind')=='clean_native' for v in p) for p in provenance],bool)
    routes=_class_routes(classes,cohort['candidate_groups'],cohort.get('class_candidate_ids'))
    if cohort.get('training_domain') is not None:
        from .structural_domain import validate
        validate(cohort['training_domain'])
    return rows,targets,classes,split,clean,routes


def _representatives(rows,clean,targets,classes):
    # Training examples alone select strong clean prototypes. No validation
    # curve, test curve or held-out score chooses a warmup example.
    best=np.full(len(classes),-np.inf);chosen=np.full(len(classes),-1,np.int64)
    for selected,values in batches(rows,clean):
        signal=np.linalg.norm(values[:,2],axis=1)
        for c in range(len(classes)):
            allowed=np.flatnonzero(targets[selected,c])
            if len(allowed):
                local=allowed[np.argmax(signal[allowed])]
                if signal[local]>best[c]:best[c]=signal[local];chosen[c]=selected[local]
    if np.any(chosen<0):raise ValueError('Every family needs clean independent training examples')
    return chosen


def train(cohort,destination,*,epochs=60,seed=89113,threads=4):
    import torch
    if (type(epochs) is not int or not 1<=epochs<=1000
            or type(seed) is not int or not 0<=seed<2**32):
        raise ValueError('Bounded attention training epochs and seed required')
    rows,targets,classes,split,clean,routes=_inputs(cohort)
    dataset=describe_counts(cohort['groups'],split.tolist(),clean.tolist(),
        generated_pairs=cohort['original_examples'],
        simulated_observations=cohort.get('augmented_examples',0))
    train_rows,validation_rows,test_rows=(np.flatnonzero(split==code) for code in range(3))
    clean_rows=train_rows[clean[train_rows]]
    representatives=_representatives(rows,clean_rows,targets,classes)
    budget=allocation(threads);torch.set_num_threads(budget['workers'])
    mean,scale=channel_statistics(rows,train_rows,minimum_scale=.01,center=False)
    out=Path(destination);began=time.monotonic()
    run_identity=checkpoint.identity(cohort,rows,targets,split,clean,mean,scale,
        epochs=epochs,seed=seed,workers=budget['workers'])

    with checkpoint.owned_run(out,run_identity):
        def progress(state,**values):
            write_json(out/'progress.json',dict(state=state,dataset=dataset,**values))

        rng=np.random.default_rng(seed);visited=np.zeros(len(rows),bool);history=[]
        clean_epochs=min(10,epochs//4);best=None;best_key=None;selected_epoch=None
        warmup_steps=200
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);model=network(len(classes))
            # Keep convolution/readout initialization independent of how many
            # attention parameters consume the random stream. This is the same
            # controlled initialization used in the attention/CNN pilot.
            torch.manual_seed(seed);control=network(len(classes),attention=False)
            model.local.load_state_dict(control.local.state_dict())
            model.head.load_state_dict(control.head.state_dict());del control
            optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=.00001)
            recovered=checkpoint.restore(out,run_identity,model,optimizer,scheduler,rng,visited,
                epochs=epochs,warmup_steps=warmup_steps)
            warmup_done=completed_epochs=0;previous_seconds=0.
            if recovered is not None:
                warmup_done=recovered['warmup'];completed_epochs=recovered['epochs']
                history=recovered['history'];best=recovered['best'];best_key=recovered['best_key']
                selected_epoch=recovered['selected_epoch'];previous_seconds=recovered['elapsed_seconds']
                progress('recovering_training',warmup_completed=warmup_done,epochs_completed=completed_epochs)

            def save_checkpoint(warmup,completed):
                checkpoint.save(out,run_identity,model,optimizer,scheduler,rng,visited,
                    warmup=warmup,epochs=completed,history=history,best=best,selected_epoch=selected_epoch,
                    best_key=best_key,elapsed_seconds=previous_seconds+time.monotonic()-began)

            if recovered is None:save_checkpoint(0,0)

            def update(selected,values):
                model.train();logits=model(torch.from_numpy((values-mean)/scale))
                allowed=torch.from_numpy(targets[selected])
                objective=(torch.logsumexp(logits,1)-torch.logsumexp(logits.masked_fill(~allowed,-torch.inf),1)).mean()
                if not torch.isfinite(objective):raise ValueError('Nonfinite attention training objective; no model was published')
                optimizer.zero_grad(set_to_none=True);objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                visited[selected]=True

            for step in range(warmup_done,warmup_steps):
                # Even a 230-class model uses bounded feature/activation batches.
                for selected,values in batches(rows,representatives,size=64):update(selected,values)
                if (step+1)%20==0:
                    save_checkpoint(step+1,0)
                    progress('clean_prototype_warmup',step=step+1,steps=warmup_steps)
            for epoch in range(completed_epochs,epochs):
                pool=clean_rows if epoch<clean_epochs else train_rows
                buckets=[pool[targets[pool,c]] for c in range(len(classes))]
                order=np.concatenate([np.concatenate([rng.choice(bucket,4,replace=True) for bucket in buckets])
                    for _ in range(int(np.ceil(len(pool)/(4*len(classes)))))])
                batch_size=min(64,4*len(classes))
                for chosen,values in batches(rows,order,size=batch_size):update(chosen,values)
                if epoch in (clean_epochs,epochs-1):
                    # Make checkpoints eligible after the first noisy epoch,
                    # rather than discovering unseen rows only at the very end
                    # and forcing validation to choose the final checkpoint.
                    missing=train_rows[~visited[train_rows]]
                    for chosen,values in batches(rows,rng.permutation(missing),size=64):update(chosen,values)
                model.eval();metrics=evaluate(model,rows,targets,validation_rows,mean,scale,classes)
                eligible=epoch>=clean_epochs and visited[train_rows].all()
                key=(-metrics['macro_top5'],-metrics['top5'],metrics['loss'])
                if eligible and (best_key is None or key<best_key):
                    best_key=key;selected_epoch=epoch+1;best=copy.deepcopy(model.state_dict())
                history.append(dict(epoch=epoch+1,validation=metrics,checkpoint_eligible=bool(eligible),
                    training_rows_seen=int(visited[train_rows].sum())))
                scheduler.step()
                save_checkpoint(warmup_steps,epoch+1)
                progress('training',epoch=epoch+1,epochs=epochs,
                    selected_epoch=selected_epoch,validation=metrics,training_rows_seen=int(visited[train_rows].sum()))
            if best is None:raise ValueError('No attention checkpoint saw the complete declared training pool')
            model.load_state_dict(best);model.eval()
            progress('calibrating_training_support',selected_epoch=selected_epoch)
            support=fit_support(rows,train_rows,validation_rows,targets,mean,scale,seed=seed)
            # Test labels have no influence on initialization, normalization,
            # sampling, prototypes, stopping or checkpoint selection.
            progress('scoring_holdout',selected_epoch=selected_epoch)
            test=evaluate(model,rows,targets,test_rows,mean,scale,classes)
            prior=evaluate(model,rows,targets,test_rows,mean,scale,classes,prior=targets[train_rows].sum(0))
        progress('packing_model',selected_epoch=selected_epoch)
        plane,layout=weights.pack(model)
        arrays=dict(parameters=plane,q_edges=cohort['q_edges'],
            normalization=np.r_[mean.ravel().astype(np.float64),scale.ravel().astype(np.float64),support['limit']],
            classes=classes,reference_features=cohort['reference_features'],support_prototypes=support['prototypes'])
        stream=io.BytesIO();np.savez_compressed(stream,**arrays);raw=stream.getvalue()
        validate_npz_bytes(raw,maximum_bytes=MAX_BYTES)
        metadata=dict(schema=weights.SCHEMA,architecture=weights.ARCHITECTURE,representation=weights.REPRESENTATION,
            channels=3,reference_kind=weights.REFERENCE_KIND,normalization_method=weights.NORMALIZATION,
            parent_sha256=cohort['parent_sha256'],instrument_sha256=cohort['instrument_sha256'],
            input_sha256=cohort['input_sha256'],candidates=cohort['candidates'],candidate_groups=cohort['candidate_groups'],
            class_candidate_ids=routes,weights_sha256=hashlib.sha256(raw).hexdigest(),tensor_layout_sha256=layout,
            classes=classes.tolist(),bins=rows.shape[2],epochs=epochs,seed=seed,allocation=budget,
            counts={name:targets[split==code].sum(0).tolist() for code,name in enumerate(['train','validation','test'])},
            partition_rows=[len(train_rows),len(validation_rows),len(test_rows)],dataset=dataset,test=test,prior_only_test=prior,
            selected_epoch=selected_epoch,checkpoint_selection='all-training-rows-seen/validation-macro-top5-then-top5-then-loss',
            curriculum=dict(warmup_steps=warmup_steps,clean_epochs=clean_epochs,gradient_norm_limit=1.,
                initialization='shared-convolution-and-readout/independent-attention-parameters/v1',
                sampling='four draws per compatible family; bounded balanced minibatches; first-noisy-epoch coverage pass',
                training_rows_seen=int(visited[train_rows].sum()),clean_training_representatives=representatives.tolist()),
            feature_storage=rows.storage,support_training_pool_rows=support['training_pool_rows'],
            support_method=support['method'],support_validation_rows=support['validation_rows'],support_coverage_qualified=False,
            augmentation=cohort.get('augmentation'),augmentation_sha256=cohort.get('augmentation_sha256'),
            augmented_examples=cohort.get('augmented_examples',0),reference_feature_sha256=cohort['reference_feature_sha256'],
            training_domain=cohort.get('training_domain'),ambiguous_examples=cohort['ambiguous_examples'],
            examples=len(rows),original_examples=cohort['original_examples'],independent_groups=len(set(cohort['groups'])),
            background_required=True,screening_qualified=False,candidate_pruning_authorized=False,scientific_result=False,
            seconds=previous_seconds+time.monotonic()-began,history=history,
            training_recovery=dict(identity=run_identity,warmup_completed_on_resume=warmup_done,
                epochs_completed_on_resume=completed_epochs,
                scope='Owned worker-local CPU state; checkpoints are not portable library artifacts'))
        encoded=(json.dumps(metadata,indent=2,allow_nan=False)+'\n').encode()
        if len(encoded)>MAX_METADATA_BYTES:raise ValueError('Attention model metadata exceeds the portable reader budget')
        (out/'weights.npz').write_bytes(raw);write_json(out/'model.json',metadata)
        progress('complete',examples=len(rows),selected_epoch=selected_epoch,
            screening_qualified=False,seconds=time.monotonic()-began)
        return metadata
