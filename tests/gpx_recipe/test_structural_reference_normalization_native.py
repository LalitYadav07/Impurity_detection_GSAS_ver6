"""Encoder conditioning and legacy transport, not scientific accuracy."""
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from gpx_recipe.preparation.structural_attention import network
from gpx_recipe.preparation import structural_attention_weights as weights


@pytest.mark.parametrize('joint',[False,True])
def test_difference_amplitude_is_not_promoted_to_unit_scale(joint):
    torch.set_num_threads(1);torch.manual_seed(9182)
    architecture=weights.MODE_ARCHITECTURE if joint else weights.ARCHITECTURE
    model=network(2,architecture=architecture,**({'family_count':2} if joint else {})).eval()
    profile=torch.exp(-((torch.linspace(-1,1,128))/.2)**2)
    captured=[];amplitudes=[]
    first=model.local[0].register_forward_pre_hook(lambda _,args:captured.append(args[0].detach().clone()))
    second=model.amplitude.register_forward_pre_hook(lambda _,args:amplitudes.append(args[0].detach().clone()))
    try:
        for size in [0.,1e-7,1e-5,.01]:
            values=torch.stack((torch.ones(128),torch.ones(128)+size*profile,size*profile))[None]
            with torch.no_grad():assert torch.isfinite(model(values)).all()
    finally:first.remove();second.remove()
    assert captured[0][:,2].count_nonzero()==0
    assert amplitudes[0][:,2].count_nonzero()==0
    for i,size in enumerate([1e-7,1e-5,.01],1):
        native_rms=float((size*profile).square().mean().sqrt())
        assert float(captured[i][:,2].square().mean().sqrt())==pytest.approx(native_rms,rel=.01)
        assert float(amplitudes[i][:,2])==pytest.approx(np.log1p(native_rms),rel=.01)
    # The old architecture is still explicit and retains its former behavior.
    old=network(2,architecture=weights.PER_CHANNEL_ARCHITECTURE).eval();legacy=[]
    handle=old.local[0].register_forward_pre_hook(lambda _,args:legacy.append(args[0].detach()))
    try:
        with torch.no_grad():old(torch.stack((torch.ones(128),torch.ones(128),.01*profile))[None])
    finally:handle.remove()
    assert float(legacy[0][:,2].square().mean().sqrt())==pytest.approx(1.,rel=1e-6)


@pytest.mark.parametrize('joint',[False,True])
def test_equal_tensor_shapes_cannot_switch_old_and_new_encoder_semantics(joint):
    torch.manual_seed(12)
    old_arch=weights.PER_CHANNEL_MODE_ARCHITECTURE if joint else weights.PER_CHANNEL_ARCHITECTURE
    new_arch=weights.MODE_ARCHITECTURE if joint else weights.ARCHITECTURE
    kw={'family_count':2} if joint else {}
    old=network(2,architecture=old_arch,**kw).eval()
    new=network(2,architecture=new_arch,**kw).eval()
    assert [(k,list(v.shape)) for k,v in old.state_dict().items()]==[(k,list(v.shape)) for k,v in new.state_dict().items()]
    plane,old_layout=weights.pack(old);_,new_layout=weights.pack(new)
    assert old_layout!=new_layout
    with pytest.raises(ValueError,match='installed architecture'):weights.restore(new,plane,old_layout)
    restored=network(2,architecture=old_arch,**kw).eval();weights.restore(restored,plane,old_layout)
    values=torch.randn(3,3,128)
    with torch.no_grad():assert torch.equal(old(values),restored(values))
