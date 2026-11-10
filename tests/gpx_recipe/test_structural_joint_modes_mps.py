"""Actual MPS shared-mode training and audited portable CPU inference."""
import os

import numpy as np
import pytest

torch=pytest.importorskip('torch')


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='An actual Apple MPS device is required')
@pytest.mark.parametrize('compressed',[False,True])
def test_joint_training_uses_mps_and_checks_every_trained_condition_on_portable_cpu(tmp_path,monkeypatch,compressed):
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')=='0'
    from test_structural_joint_modes_native import cohort
    from gpx_recipe.preparation.structural_paired_training import train
    from gpx_recipe.preparation.structural_ranker import Ranker
    if compressed:
        from gpx_recipe.preparation import structural_history
        monkeypatch.setattr(structural_history,'INLINE_BYTES',1)
    _,value=cohort(tmp_path/'source')
    result=train(value,tmp_path/'model',epochs=2,threads=2)
    transport=result['training_device_transport']
    assert transport['device']=='mps' and transport['portable_inference_device']=='cpu'
    assert transport['mode_queries']>0 and np.isfinite(transport['maximum_mode_absolute_error'])
    assert result['mode_head_trained'] and result['test']['mode_validation']['available']
    model=Ranker(tmp_path/'model')
    assert {p.device.type for p in model.model.parameters()}=={'cpu'}
    assert model.metadata['history']==result['history']
    assert not model.metadata['mode_readout']['experimental_mode_absence_qualified']
