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


def calibrate(model,rows,targets,split,groups,classes,mean,scale,*,identities):
    import torch
    if model.training:raise ValueError('Freeze the selected model before independent calibration')
    selected=np.flatnonzero(split==3);capture=Logits(selected,len(classes))
    for indices,values in batches(rows,selected):
        with torch.no_grad():logits=forward(model,torch.from_numpy((values-mean)/scale))
        capture.add(indices,logits.numpy())
    return adaptive.calibrate(capture.finish(),targets[selected],[groups[i] for i in selected],
        classes=classes.tolist(),identities=identities,used_groups=sorted({g for g,c in zip(groups,split) if c!=3}))
