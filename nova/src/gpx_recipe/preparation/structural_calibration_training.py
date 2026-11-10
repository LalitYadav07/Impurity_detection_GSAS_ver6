"""Bounded frozen-model predictions for independent adaptive-set calibration."""
import numpy as np

from .structural_training_data import batches
from .structural_training_backend import forward
from . import structural_adaptive_sets as adaptive


class Logits:
    """Capture an existing inference pass, not a second expensive prediction."""
    def __init__(self,indices,classes):
        self.indices=np.asarray(indices);self.classes=classes;self.position=0
        if (self.indices.dtype.kind not in 'iu' or self.indices.ndim!=1 or not 1<=len(self.indices)<=adaptive.MAX_ROWS
                or type(classes)is not int or not 2<=classes<=230):
            raise ValueError('Bounded ordered inference rows and native classes required')
        self.values=np.empty((len(self.indices),classes),np.float32)

    def add(self,indices,values):
        end=self.position+len(indices);values=np.asarray(values)
        if (not np.array_equal(indices,self.indices[self.position:end]) or end>len(self.indices)
                or values.dtype!=np.float32 or values.shape!=(len(indices),self.classes)
                or not np.isfinite(values).all()):
            raise ValueError('Frozen inference rows changed order, dimensions or values')
        self.values[self.position:end]=values;self.position=end

    def finish(self):
        if self.position!=len(self.indices):raise ValueError('Frozen inference did not cover every reserved row')
        return self.values


def calibrate(model,rows,targets,split,groups,classes,mean,scale,*,identities,
              _mode_context=None,_mode_calibration=None):
    import torch
    if model.training:raise ValueError('Freeze the selected model before independent calibration')
    if (_mode_context is None)!=(_mode_calibration is None):
        raise ValueError('Joint calibration needs both its native mode context and interval accumulator')
    if _mode_context is not None:
        from ..symmetry.contracts import digest
        if (digest(_mode_context.spec)!=digest(_mode_calibration.spec)
                or _mode_calibration.identities!=identities
                or _mode_calibration.groups!=sorted({g for g,c in zip(groups,split) if c==3})
                or _mode_calibration.used_sha!=digest(sorted({g for g,c in zip(groups,split) if c!=3}))):
            raise ValueError('Interval calibration belongs to a different readout or independent partition')
    selected=np.flatnonzero(split==3);capture=Logits(selected,len(classes))
    for indices,values in batches(rows,selected):
        with torch.no_grad():
            inputs=torch.from_numpy((values-mean)/scale)
            if _mode_context is None:logits=forward(model,inputs)
            else:
                from .structural_joint_modes import shared,queries
                logits,embedding=shared(model,inputs)
                target=_mode_context.normal.targets(_mode_context.table,indices)
                prediction=queries(model,embedding,target).detach().cpu().numpy()
                _mode_calibration.add(prediction,target,[groups[i] for i in indices])
        capture.add(indices,logits.numpy())
    return adaptive.calibrate(capture.finish(),targets[selected],[groups[i] for i in selected],
        classes=classes.tolist(),identities=identities,used_groups=sorted({g for g,c in zip(groups,split) if c!=3}))
