"""Actual-device mode gradients, preserving float64 CPU target objectives."""
import copy
import os
import sys

import numpy as np
import pytest

torch=pytest.importorskip('torch')
pytestmark=pytest.mark.skipif(sys.platform!='darwin' or not torch.backends.mps.is_available(),reason='Actual Apple MPS required')


def test_mps_mode_head_retains_float64_targets_gradients_and_portable_predictions():
    from gpx_recipe.preparation.structural_attention import network
    from gpx_recipe.preparation import structural_mode_readout as modes
    from gpx_recipe.preparation import structural_attention_weights as weights
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')=='0'
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8413);encoder=network(1).eval();head=modes.network(1,2).eval()
        gpu=copy.deepcopy(encoder).to('mps');other=copy.deepcopy(head).to('mps')
        values=torch.randn(4,3,128)
        row=np.repeat(np.arange(4),2).astype(np.int64);family=np.tile(np.arange(2),4).astype(np.int64)
        query=[torch.from_numpy(v) for v in (row,np.zeros(8,np.int64),family)]
        targets=dict(row=row,space_group=np.zeros(8,np.int64),lower=np.full(8,1e-200),upper=np.full(8,.4))
        cpu=head(encoder.embedding(values),*query)
        prediction=other(gpu.embedding(values.to('mps')),*[v.to('mps') for v in query])
        actual=modes.loss(prediction,targets,rows=4,class_count=1)
        expected=modes.loss(cpu,targets,rows=4,class_count=1)
        assert actual.dtype==torch.float64 and actual.device.type=='cpu'
        torch.testing.assert_close(prediction.cpu(),cpu,rtol=1e-4,atol=1e-5)
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
        actual.backward();torch.mps.synchronize()
        assert float(gpu.local[0].weight.grad.abs().sum().cpu())>0
        assert float(other.family.weight.grad.abs().sum().cpu())>0
        plane,layout=weights.pack(other);restored=modes.network(1,2)
        weights.restore(restored,plane,layout)
        with torch.no_grad():torch.testing.assert_close(restored(encoder.embedding(values),*query),cpu,rtol=0,atol=0)
