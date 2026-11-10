"""Actual optimizer/device recovery with a large visitation bitmap.

This verifies numerical continuation, not a trained scientific model or
the accuracy of 200,000 synthesized diffraction examples.
"""
import copy

import numpy as np
import pytest

torch=pytest.importorskip('torch')
from gpx_recipe.preparation import structural_training_backend as backend
from gpx_recipe.preparation.structural_attention import network
from gpx_recipe.preparation.structural_attention_weights import pack
from test_structural_replay_large import layout


@pytest.mark.parametrize('kind',['cpu','mps'])
def test_real_optimizer_recovers_large_rows_without_changing_devices_or_parameters(tmp_path,kind):
    if kind=='mps' and not torch.backends.mps.is_available():pytest.skip('A real MPS device is required')
    engine=backend.Backend(kind);recovery=engine.checkpoint
    rng=np.random.default_rng(718);visited=np.zeros(200000,bool);row_layout=layout()
    with torch.random.fork_rng(devices=[]),engine.random_state(921):
        torch.manual_seed(714);model=engine.place(network(2))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=3,eta_min=.00001)
        values=torch.randn(2,3,64)
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            backend.forward(model,values).square().mean().backward();optimizer.step()
        before=pack(copy.deepcopy(model).cpu())[0];np_before=copy.deepcopy(rng.bit_generator.state)
        recovery.save(tmp_path,'a'*64,model,optimizer,scheduler,rng,visited,
            warmup=20,epochs=0,history=[],best=None,selected_epoch=None,best_key=None,
            elapsed_seconds=1.,row_layout=row_layout)
        restored=engine.place(network(2));other=torch.optim.AdamW(restored.parameters(),lr=.0003,weight_decay=.0001)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(other,T_max=3,eta_min=.00001)
        current=pack(copy.deepcopy(restored).cpu())[0]
        bad=dict(row_layout,rows=200001)
        with pytest.raises(ValueError,match='row layout'):
            recovery.restore(tmp_path,'a'*64,restored,other,schedule,rng,visited,
                epochs=3,warmup_steps=200,row_layout=bad)
        np.testing.assert_array_equal(pack(copy.deepcopy(restored).cpu())[0],current)
        assert rng.bit_generator.state==np_before
        result=recovery.restore(tmp_path,'a'*64,restored,other,schedule,rng,visited,
            epochs=3,warmup_steps=200,row_layout=row_layout)
        assert result['warmup']==20 and result['epochs']==0
        np.testing.assert_array_equal(pack(copy.deepcopy(restored).cpu())[0],before)
        assert rng.bit_generator.state==np_before and not visited.any()
        for first,second in zip(optimizer.state.values(),other.state.values()):
            for key in first:assert torch.equal(first[key].cpu(),second[key].cpu())
