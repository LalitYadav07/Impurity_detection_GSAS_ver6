"""Independent attention algebra, resolution and portable-layout controls."""
import itertools

import numpy as np
import pytest

torch=pytest.importorskip('torch')
from gpx_recipe.preparation.structural_attention import network
from gpx_recipe.preparation.structural_linear_attention import attend
from gpx_recipe.preparation import structural_attention_weights as weights


@pytest.mark.parametrize('masked',[False,True])
def test_associative_attention_matches_explicit_dense_kernel_values_and_gradients(masked):
    torch.manual_seed(723);torch.set_num_threads(1)
    original=[torch.randn(2,4,17,8,dtype=torch.float64) for _ in range(3)]
    a=[v.clone().requires_grad_() for v in original]
    b=[v.clone().requires_grad_() for v in original]
    valid=torch.ones(2,17,dtype=torch.bool) if masked else None
    if masked:valid[0,3:9]=False;valid[1]=False
    actual=attend(*a,valid)
    q,k,v=b;positive_q=torch.nn.functional.elu(q)+1.;positive_k=torch.nn.functional.elu(k)+1.
    kernel=positive_q@positive_k.transpose(-1,-2)
    if valid is not None:kernel=kernel.masked_fill(~valid[:,None,None,:],0.)
    expected=(kernel@v)/kernel.sum(-1,keepdim=True).clamp_min(1e-6)
    if valid is not None:expected=expected.masked_fill(~valid[:,None,:,None],0.)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    objective=torch.randn_like(actual)
    actual_grad=torch.autograd.grad((actual*objective).sum(),a)
    expected_grad=torch.autograd.grad((expected*objective).sum(),b)
    for left,right in zip(actual_grad,expected_grad):
        torch.testing.assert_close(left,right,rtol=1e-11,atol=1e-12)


@pytest.mark.parametrize('bins',[64,129,4096,8192])
def test_attention_visits_every_local_location_before_fixed_readout(bins):
    torch.manual_seed(31);torch.set_num_threads(1)
    model=network(16);seen=[]
    handle=model.attention.register_forward_pre_hook(lambda module,args:seen.append(args[0].shape))
    try:
        values=torch.randn(2,3,bins);values[:,2]=0
        logits=model(values);assert logits.shape==(2,16) and torch.isfinite(logits).all()
        logits.square().mean().backward()
    finally:handle.remove()
    assert seen==[torch.Size([2,(bins+3)//4,32])]
    assert model.attention.qkv.weight.grad.abs().sum()>0
    assert model.position_scale.grad is not None and torch.isfinite(model.position_scale.grad)
    assert model.amplitude.weight.grad.abs().sum()>0


def test_null_three_curve_input_has_finite_logits_and_weight_gradients():
    torch.manual_seed(31);torch.set_num_threads(1)
    model=network(3);value=model(torch.zeros(2,3,257))
    assert torch.isfinite(value).all()
    value.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('source,target',list(itertools.permutations(sorted(weights.ARCHITECTURES),2)))
def test_old_and_new_architecture_names_cannot_reinterpret_numeric_planes(source,target):
    torch.manual_seed(91)
    plane,layout=weights.pack(network(4,architecture=source))
    with pytest.raises(ValueError,match='installed architecture'):
        weights.restore(network(4,architecture=target),plane,layout)


@pytest.mark.parametrize('bins',[0,63,8193])
def test_full_local_network_does_not_allocate_unbounded_coordinate_work(bins):
    with pytest.raises(ValueError,match='bounded physical Q'):
        network(3)(torch.zeros(1,3,bins))


def test_portable_readout_shape_is_installed_not_chosen_by_uploaded_metadata():
    with pytest.raises(ValueError,match='exactly 128'):
        network(3,tokens=64)
    model=network(230);plane,layout=weights.pack(model)
    assert plane.dtype==np.float32 and plane.nbytes<8*1024**2 and len(layout)==64
