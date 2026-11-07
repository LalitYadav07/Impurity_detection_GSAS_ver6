import pytest

torch = pytest.importorskip('torch')
from gpx_recipe.preparation.structural_attention import network


@pytest.mark.parametrize('attention', [False, True])
def test_three_curve_encoder_trains_and_rejects_masked_payload_leakage(attention):
    torch.manual_seed(73261); torch.set_num_threads(1)
    model = network(9, attention=attention, tokens=64)
    values = torch.randn(2, 3, 512)
    coverage = torch.ones(2, 512); coverage[:, 64:192] = 0
    logits = model(values, coverage)
    assert logits.shape == (2, 9) and torch.isfinite(logits).all()
    logits.square().mean().backward()
    if attention:
        assert model.attention.layers[0].self_attn.in_proj_weight.grad.abs().sum() > 0
    model.eval()
    changed = values.clone(); changed[:, :, 64:192] = 1e8
    with torch.no_grad():
        torch.testing.assert_close(model(changed, coverage), model(values, coverage), rtol=0, atol=0)
    with pytest.raises(ValueError, match='three'): model(torch.randn(2, 8, 512))
    with pytest.raises(ValueError, match='masked'): model(values, torch.zeros_like(coverage))
    coverage[0, 1] = float('nan')
    with pytest.raises(ValueError, match='coverage'): model(values, coverage)
