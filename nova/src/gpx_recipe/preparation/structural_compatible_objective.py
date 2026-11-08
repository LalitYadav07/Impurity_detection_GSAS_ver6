"""Supervise every distinct compatible symmetry, without alias multiplicity.

Equal mass within an ambiguous target set is a training convention, not a
physical prior or a probability that each space group is present. Calibration
and native refinement still determine the useful work set and interpretation.
"""
import math

OBJECTIVE='uniform-distinct-compatible-space-group-cross-entropy/v1'
SELECTION='all-training-rows-seen/validation-macro-compatible-nll-then-loss/v1'


def terms(logits,compatible):
    """Return row losses and individual negative log scores, with gradients."""
    import torch
    if (logits.ndim!=2 or compatible.shape!=logits.shape or compatible.dtype!=torch.bool
            or not len(logits) or not compatible.any(1).all()):
        raise ValueError('Every training row needs its complete nonempty compatible class set')
    negative=-torch.nn.functional.log_softmax(logits,dim=1)
    losses=negative.masked_fill(~compatible,0.).sum(1)/compatible.sum(1)
    if not torch.isfinite(negative).all() or not torch.isfinite(losses).all():
        raise ValueError('Nonfinite all-compatible training scores')
    return losses,negative


def checkpoint_key(metrics):
    """New runs retain rare aliases; old numerical checkpoints stay readable."""
    if not isinstance(metrics,dict):raise ValueError('A checked validation metric is required')
    if metrics.get('loss_method')==OBJECTIVE:
        values=(metrics.get('macro_compatible_nll'),metrics.get('loss'))
    elif 'loss_method' not in metrics:
        # Legacy recovery is still bound to its exact code/data identity.
        values=tuple(metrics.get(k) for k in ('macro_top5','top5','loss'))
        if all(type(v) in (int,float) for v in values):values=(-values[0],-values[1],values[2])
    else:raise ValueError('Unknown structural validation objective')
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in values):
        raise ValueError('Finite validation selection scores are required')
    return tuple(values)


def validate_policy(metadata):
    """Check a newly declared policy without reinterpreting legacy models."""
    if 'training_objective' not in metadata and metadata.get('checkpoint_selection')!=SELECTION:return
    history=metadata.get('history');epochs=metadata.get('epochs')
    if (metadata.get('training_objective')!=OBJECTIVE or metadata.get('checkpoint_selection')!=SELECTION
            or type(epochs)is not int or not 1<=epochs<=1000
            or not isinstance(history,list) or len(history)!=epochs):
        raise ValueError('All-compatible training needs its exact objective and validation history')
    eligible=[]
    for i,row in enumerate(history):
        if (not isinstance(row,dict) or row.get('epoch')!=i+1
                or type(row.get('checkpoint_eligible'))is not bool
                or not isinstance(row.get('validation'),dict)
                or row['validation'].get('loss_method')!=OBJECTIVE):
            raise ValueError('All-compatible training history changed its selection policy')
        checkpoint_key(row['validation'])
        if row['checkpoint_eligible']:eligible.append(row)
    if not eligible or metadata.get('selected_epoch')!=min(eligible,key=lambda row:checkpoint_key(row['validation']))['epoch']:
        raise ValueError('The structural model lost its all-compatible validation-selected checkpoint')
    if not isinstance(metadata.get('test'),dict) or metadata['test'].get('loss_method')!=OBJECTIVE:
        raise ValueError('The structural test uses a different objective from training')
