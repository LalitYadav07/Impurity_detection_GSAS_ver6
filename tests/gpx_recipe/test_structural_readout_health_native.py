"""Regress the observed many-family failure without altering scientific labels."""
import json

import numpy as np
import pytest

torch=pytest.importorskip('torch')
from gpx_recipe.preparation.structural_attention import network
from gpx_recipe.preparation.structural_attention_weights import LEGACY_ARCHITECTURE
from gpx_recipe.preparation.structural_readout_health import forward,training_health
from gpx_recipe.preparation.structural_training_data import feature_rows


def test_normalized_default_resists_the_observed_common_negative_readout_shift():
    torch.set_num_threads(1);torch.manual_seed(73261)
    values=torch.randn(4,3,512,requires_grad=True)
    legacy=network(16,architecture=LEGACY_ARCHITECTURE);legacy.eval()
    current=network(16);current.eval()
    with torch.no_grad():
        legacy.head[1].bias.fill_(-100.);current.head[1].bias.fill_(-100.)
    _,old_active=forward(legacy,values)
    logits,active=forward(current,values)
    assert not old_active.any() and active.all()
    assert (logits-logits[0]).abs().max()>.001
    logits.square().sum().backward()
    assert values.grad.abs().max()>1e-6
    assert current.attention.qkv.weight.grad.abs().sum()>0


def test_training_readout_checks_are_bounded_and_preserve_dropout_rng(monkeypatch):
    torch.set_num_threads(1);torch.manual_seed(65173)
    model=network(16);model.train()
    rows=feature_rows(np.random.default_rng(726).normal(size=(130,3,128)).astype(np.float32))
    sizes=[];take=rows.take
    def bounded(indices):sizes.append(len(indices));return take(indices)
    monkeypatch.setattr(rows,'take',bounded)
    before=torch.get_rng_state().clone()
    result=training_health(model,rows,np.arange(130),np.zeros((3,1),np.float32),np.ones((3,1),np.float32))
    assert result['readout_active'] and result['representatives']==result['active_readouts']==130
    assert sizes==[64,64,2] and result['maximum_logit_difference']>0
    assert model.training and torch.equal(before,torch.get_rng_state())
    assert all(p.grad is None for p in model.parameters())


def test_dead_training_readout_fails_before_holdout_scoring_or_model_publication(tmp_path,monkeypatch):
    from gpx_recipe.preparation import structural_paired_training as training
    from test_structural_paired_training_native import data
    _,cohort=data(tmp_path/'pairs');original=training.network
    def inactive(count,**kwargs):
        model=original(count,architecture=LEGACY_ARCHITECTURE,**kwargs)
        with torch.no_grad():model.head[1].weight.zero_();model.head[1].bias.fill_(-100.)
        return model
    monkeypatch.setattr(training,'network',inactive)
    monkeypatch.setattr(training,'evaluate',lambda *a,**k:pytest.fail('Dead training reached held-out scoring'))
    out=tmp_path/'refused'
    with pytest.raises(ValueError,match='inactive on every training representative'):
        training.train(cohort,out,epochs=1,threads=1)
    progress=json.loads((out/'progress.json').read_text())
    assert progress['state']=='training_rejected' and not progress['readout_health']['readout_active']
    assert (out/'training-checkpoint.npz').is_file()
    assert not (out/'model.json').exists() and not (out/'weights.npz').exists()


def test_nonfinite_readout_releases_its_inspection_hook():
    model=network(2);model.eval();last=model.head[-1]
    before=dict(last._forward_pre_hooks)
    with pytest.raises(ValueError,match='invalid activations'):
        forward(model,torch.full((1,3,128),float('nan')))
    assert dict(last._forward_pre_hooks)==before
